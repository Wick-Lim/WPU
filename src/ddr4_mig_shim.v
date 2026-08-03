// ============================================================================
// ddr4_mig_shim.v -- ddr5_xbar memory side  ->  one AXI4 read-only master
//
//   WHY THIS EXISTS (and why it is a shim rather than a change to the xbar)
//
//   glm_q4k_system drives the xbar's requester port as
//       assign xreq_valid = any_pending;
//   over a COMBINATIONAL priority mux that picks {addr,tag} from whichever weight
//   family currently outranks the others.  If the memory refuses, the next cycle can
//   present a DIFFERENT request while valid is still high.
//
//   That is internally consistent -- `lb_accept = xreq_fire & sel_lb` is evaluated in
//   the same cycle, so nothing is lost -- and it is MEASURED to be harmless to the
//   token stream: with a ~50%-ready LFSR on mem_req_ready, the loopback_rest gate
//   still committed a bit-exact stream (`make loopback-rest -DTB_REQ_STALL`,
//   stall_cyc=7688).
//
//   But AXI4 requires ARADDR/ARID to hold from ARVALID until ARREADY.  The same run
//   measured **unstable_cyc=60** -- 60 cycles where a held request changed payload.
//   Feeding a MIG directly would therefore violate the protocol.
//
//   So the fix lives HERE, in new default-off code, instead of in the frozen,
//   bit-exact, netlist-pinned system RTL: a per-channel REGISTERED SKID SLOT.
//     * mem_req_ready[c] is a registered "my slot is empty" flop.
//     * {addr,tag} are captured on the accepted cycle and never move again.
//     * AXI AR is driven from the SLOT, so ARADDR/ARID are stable by construction.
//   The 60 unstable cycles simply cannot reach AXI: they happen before acceptance.
//
//   TOPOLOGY.  A Xilinx MIG exposes ONE AXI slave port, so the N_CH channel requests
//   are arbitrated (round-robin, no starvation) onto a single AR channel, and the
//   channel index rides in the high bits of ARID.  Read data is demuxed back to the
//   originating channel by RID.  The xbar's own per-channel response FIFO
//   (RESP_QD deep, mem_resp_ready[c] = cnt[c] != RESP_QD) is the only flow control on
//   the return path, so RREADY is simply that channel's ready.
//
//   NOT VERIFIED HERE (stated, not pretended): real MIG timing, refresh behaviour,
//   ECC, address mapping to physical DDR4 rank/bank/row/col, and anything about a
//   board.  This module is verified against an AXI4 protocol checker + memory model
//   in simulation only.
// ============================================================================
`default_nettype none

module ddr4_mig_shim #(
    parameter integer N_CH    = 4,      // xbar channels
    parameter integer ADDR_W  = 32,     // xbar request address width
    parameter integer DATA_W  = 256,    // beat width (must equal AXI data width)
    parameter integer TAG_W   = 8,      // xbar tag width
    // AXI ID carries {channel, tag}; width must cover both.
    parameter integer CH_IDX_W = (N_CH <= 1) ? 1 : $clog2(N_CH),
    parameter integer AXI_ID_W = CH_IDX_W + TAG_W,
    parameter integer AXI_ADDR_W = 32
)(
    input  wire                      clk,
    input  wire                      rst,

    // ---- xbar memory side (this module is the "memory") ----
    input  wire [N_CH-1:0]           mem_req_valid,
    output wire [N_CH-1:0]           mem_req_ready,
    input  wire [N_CH*ADDR_W-1:0]    mem_req_addr,
    input  wire [N_CH*TAG_W-1:0]     mem_req_tag,
    output wire [N_CH-1:0]           mem_resp_valid,
    input  wire [N_CH-1:0]           mem_resp_ready,
    output wire [N_CH*DATA_W-1:0]    mem_resp_data,
    output wire [N_CH*TAG_W-1:0]     mem_resp_tag,

    // ---- AXI4 read-only master (to MIG) ----
    output wire [AXI_ID_W-1:0]       m_axi_arid,
    output wire [AXI_ADDR_W-1:0]     m_axi_araddr,
    output wire [7:0]                m_axi_arlen,
    output wire [2:0]                m_axi_arsize,
    output wire [1:0]                m_axi_arburst,
    output wire                      m_axi_arvalid,
    input  wire                      m_axi_arready,
    input  wire [AXI_ID_W-1:0]       m_axi_rid,
    input  wire [DATA_W-1:0]         m_axi_rdata,
    input  wire [1:0]                m_axi_rresp,
    input  wire                      m_axi_rlast,
    input  wire                      m_axi_rvalid,
    output wire                      m_axi_rready,

    // ---- observation (drives nothing) ----
    output reg  [31:0]               dbg_ar_issued,
    output reg  [31:0]               dbg_r_returned,
    output reg  [31:0]               dbg_rresp_err
);

    localparam integer AXSIZE = (DATA_W ==  64) ? 3 :
                                (DATA_W == 128) ? 4 :
                                (DATA_W == 256) ? 5 :
                                (DATA_W == 512) ? 6 : 5;

    // ---------------- per-channel registered skid slot ----------------------
    //   This is the whole point of the module.  ready is a FLOP, and the payload is
    //   captured once; nothing downstream can ever see it move.
    reg [N_CH-1:0]        slot_full;
    reg [ADDR_W-1:0]      slot_addr [0:N_CH-1];
    reg [TAG_W-1:0]       slot_tag  [0:N_CH-1];

    assign mem_req_ready = ~slot_full;      // registered: slot_full is a flop

    // ---------------- round-robin arbitration onto the single AR channel -----
    reg  [CH_IDX_W-1:0]   rr;
    reg  [CH_IDX_W-1:0]   gnt;
    reg                   gnt_v;
    integer               ai, idx;
    always @* begin
        gnt_v = 1'b0;
        gnt   = {CH_IDX_W{1'b0}};
        for (ai = 0; ai < N_CH; ai = ai + 1) begin
            idx = (rr + ai) % N_CH;
            if (!gnt_v && slot_full[idx]) begin
                gnt_v = 1'b1;
                gnt   = idx[CH_IDX_W-1:0];
            end
        end
    end

`ifdef INJ_MIG_NOSKID
    // INJECTION (never a normal build): drive AR straight from the requester instead
    //   of from the registered slot -- the shim minus the only thing it is for.  The
    //   requester changes a held payload every cycle, so ARADDR/ARID move while
    //   ARVALID is high: the A1 assertion MUST fail.
    assign m_axi_arvalid = |mem_req_valid;
    assign m_axi_araddr  = AXI_ADDR_W'(mem_req_addr[gnt*ADDR_W +: ADDR_W]);
    assign m_axi_arid    = {gnt, mem_req_tag[gnt*TAG_W +: TAG_W]};
`else
    assign m_axi_arvalid = gnt_v;
    assign m_axi_araddr  = AXI_ADDR_W'(slot_addr[gnt]);
    assign m_axi_arid    = {gnt, slot_tag[gnt]};
`endif
    assign m_axi_arlen   = 8'd0;            // single beat per xbar request
    assign m_axi_arsize  = AXSIZE[2:0];
    assign m_axi_arburst = 2'b01;           // INCR

    wire ar_fire = m_axi_arvalid && m_axi_arready;

    // ---------------- read data demux back to the originating channel -------
    wire [CH_IDX_W-1:0] r_ch  = m_axi_rid[AXI_ID_W-1 -: CH_IDX_W];
    wire [TAG_W-1:0]    r_tag = m_axi_rid[TAG_W-1:0];

    genvar gc;
    generate
    for (gc = 0; gc < N_CH; gc = gc + 1) begin : g_resp
        assign mem_resp_valid[gc] = m_axi_rvalid && (r_ch == gc[CH_IDX_W-1:0]);
        assign mem_resp_data [gc*DATA_W +: DATA_W] = m_axi_rdata;
        assign mem_resp_tag  [gc*TAG_W  +: TAG_W ] = r_tag;
    end
    endgenerate

    //   The xbar never backpressures its own drain (resp_ready tied 1 there); the only
    //   flow control on the return path is the per-channel response FIFO, exposed as
    //   mem_resp_ready[c].  So RREADY is exactly the addressed channel's ready.
    assign m_axi_rready = mem_resp_ready[r_ch];

    integer si;
    always @(posedge clk) begin
        if (rst) begin
            slot_full      <= {N_CH{1'b0}};
            rr             <= {CH_IDX_W{1'b0}};
            dbg_ar_issued  <= 32'd0;
            dbg_r_returned <= 32'd0;
            dbg_rresp_err  <= 32'd0;
            for (si = 0; si < N_CH; si = si + 1) begin
                slot_addr[si] <= {ADDR_W{1'b0}};
                slot_tag [si] <= {TAG_W{1'b0}};
            end
        end else begin
            // capture: one accepted request per channel per cycle
            for (si = 0; si < N_CH; si = si + 1)
                if (mem_req_valid[si] && !slot_full[si]) begin
                    slot_addr[si] <= mem_req_addr[si*ADDR_W +: ADDR_W];
                    slot_tag [si] <= mem_req_tag [si*TAG_W  +: TAG_W ];
                    slot_full[si] <= 1'b1;
                end
            // release on AR handshake, and rotate the arbiter
            if (ar_fire) begin
                slot_full[gnt] <= 1'b0;
                rr             <= (gnt == N_CH-1) ? {CH_IDX_W{1'b0}}
                                                  : (gnt + 1'b1);
                dbg_ar_issued  <= dbg_ar_issued + 32'd1;
            end
            if (m_axi_rvalid && m_axi_rready) begin
                dbg_r_returned <= dbg_r_returned + 32'd1;
                if (m_axi_rresp != 2'b00) dbg_rresp_err <= dbg_rresp_err + 32'd1;
            end
        end
    end

    // ---- AXI4 payload-stability assertion (simulation only) ----------------
    //   The property this module exists to guarantee.  `ifndef YOSYS, not
    //   `ifndef SYNTHESIS -- nothing in this repo defines SYNTHESIS.
`ifndef YOSYS
    reg                  ar_held;
    reg [AXI_ADDR_W-1:0] ar_addr_q;
    reg [AXI_ID_W-1:0]   ar_id_q;
    always @(posedge clk) begin
        if (rst) ar_held <= 1'b0;
        else begin
            if (m_axi_arvalid && !m_axi_arready) begin
                if (ar_held && ((m_axi_araddr !== ar_addr_q) || (m_axi_arid !== ar_id_q)))
                    $fatal(1, "ddr4_mig_shim: ARADDR/ARID changed while ARVALID held and ARREADY low -- AXI4 violation the skid slot exists to prevent");
                ar_addr_q <= m_axi_araddr;
                ar_id_q   <= m_axi_arid;
                ar_held   <= 1'b1;
            end else ar_held <= 1'b0;
        end
    end
`endif

endmodule
`default_nettype wire
