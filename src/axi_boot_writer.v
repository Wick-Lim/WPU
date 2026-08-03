// ============================================================================
// axi_boot_writer.v -- boot_loader's DDR write face -> AXI4 write channels (L3 S7)
//
//   THE MISSING QUARTER OF THE MEMORY STORY.  ddr4_mig_shim services the runtime
//   READ path (AR/R).  boot_loader's power-up Flash->DDR copy needs DDR WRITES --
//   ddr_we/ddr_addr/ddr_wdata against a ddr_ready.  AXI4's write channels
//   (AW/W/B) are INDEPENDENT of the read channels, so boot and runtime share the
//   single MIG slave port with NO arbiter: this module owns AW/W/B, the shim owns
//   AR/R, and they never contend.  (Boot finishes before inference starts anyway
//   -- boot_loader's `done` is the gate -- but the channel split makes the
//   no-contention property structural rather than sequenced.)
//
//   FLOW CONTROL.  One in-flight write at a time: capture on ddr_we&&ddr_ready,
//   hold AW/W stable until each accepts, count the B response, then re-assert
//   ddr_ready.  Single-beat bursts (AWLEN=0).  Simple on purpose: the boot copy
//   is bulk and one-time; W throughput is not the L3 bottleneck (the SPI read
//   side is orders of magnitude slower).
//
//   WHY ddr_ready IS REGISTERED: boot_loader presents ddr_we/addr/wdata
//   combinationally and may retarget them if not accepted -- the same waveform
//   shape the read side measured (unstable_cyc=60).  The registered slot makes
//   AWADDR/WDATA stable-by-construction on AXI, same argument as the shim.
//
//   NOT VERIFIED HERE (board-only, stated): real MIG write timing, ECC, and the
//   physical DDR4.  The gate proves AXI write-channel protocol correctness and
//   data integrity against a behavioural AXI write slave.
// ============================================================================
`default_nettype none

module axi_boot_writer #(
    parameter integer ADDR_W     = 32,
    parameter integer DATA_W     = 64,
    parameter integer AXI_ADDR_W = 32,
    parameter integer AXI_DATA_W = 64
)(
    input  wire                   clk,
    input  wire                   rst,

    // ---- boot_loader DDR face (this module is the "DDR") ----
    input  wire                   ddr_we,
    input  wire [ADDR_W-1:0]      ddr_addr,      // WORD address
    input  wire [DATA_W-1:0]      ddr_wdata,
    output wire                   ddr_ready,

    // ---- AXI4 write-only master (to MIG; shares the port with the read shim) --
    output wire [AXI_ADDR_W-1:0]  m_axi_awaddr,
    output wire [7:0]             m_axi_awlen,
    output wire [2:0]             m_axi_awsize,
    output wire [1:0]             m_axi_awburst,
    output reg                    m_axi_awvalid,
    input  wire                   m_axi_awready,
    output wire [AXI_DATA_W-1:0]  m_axi_wdata,
    output wire [AXI_DATA_W/8-1:0] m_axi_wstrb,
    output wire                   m_axi_wlast,
    output reg                    m_axi_wvalid,
    input  wire                   m_axi_wready,
    input  wire [1:0]             m_axi_bresp,
    input  wire                   m_axi_bvalid,
    output wire                   m_axi_bready,

    // ---- observation ----
    output reg  [31:0]            dbg_writes,
    output reg  [31:0]            dbg_bresp_err
);

    localparam integer AXSIZE = (AXI_DATA_W ==  32) ? 2 :
                                (AXI_DATA_W ==  64) ? 3 :
                                (AXI_DATA_W == 128) ? 4 : 3;
    localparam integer BYTES  = DATA_W/8;

    reg                  slot_full;
    reg [ADDR_W-1:0]     slot_addr;
    reg [DATA_W-1:0]     slot_data;
    reg                  aw_done, w_done;

    assign ddr_ready = ~slot_full;                  // registered flop

`ifdef INJ_BWR_NOSLOT
    // INJECTION (never a normal build): drive AW straight from the requester
    //   instead of the registered slot.  The boot requester retargets a refused
    //   write combinationally, so AWADDR moves while AWVALID is held: the
    //   stability assertion and the TB's payload audit MUST fail.
    assign m_axi_awaddr  = AXI_ADDR_W'(ddr_addr) << $clog2(BYTES);
`else
    assign m_axi_awaddr  = AXI_ADDR_W'(slot_addr) << $clog2(BYTES);  // word -> byte
`endif
    assign m_axi_awlen   = 8'd0;
    assign m_axi_awsize  = AXSIZE[2:0];
    assign m_axi_awburst = 2'b01;
    assign m_axi_wdata   = AXI_DATA_W'(slot_data);
    assign m_axi_wstrb   = {(AXI_DATA_W/8){1'b1}};
    assign m_axi_wlast   = 1'b1;
    assign m_axi_bready  = 1'b1;

    always @(posedge clk) begin
        if (rst) begin
            slot_full <= 1'b0; slot_addr <= 0; slot_data <= 0;
            m_axi_awvalid <= 1'b0; m_axi_wvalid <= 1'b0;
            aw_done <= 1'b0; w_done <= 1'b0;
            dbg_writes <= 32'd0; dbg_bresp_err <= 32'd0;
        end else begin
            // capture one write; payload never moves after this
            if (ddr_we && !slot_full) begin
                slot_addr <= ddr_addr;
                slot_data <= ddr_wdata;
                slot_full <= 1'b1;
                m_axi_awvalid <= 1'b1;
                m_axi_wvalid  <= 1'b1;
                aw_done <= 1'b0; w_done <= 1'b0;
            end
            if (m_axi_awvalid && m_axi_awready) begin m_axi_awvalid <= 1'b0; aw_done <= 1'b1; end
            if (m_axi_wvalid  && m_axi_wready ) begin m_axi_wvalid  <= 1'b0; w_done  <= 1'b1; end
            if (slot_full && m_axi_bvalid
                && (aw_done || (m_axi_awvalid && m_axi_awready))
                && (w_done  || (m_axi_wvalid  && m_axi_wready))) begin
                slot_full  <= 1'b0;
                dbg_writes <= dbg_writes + 32'd1;
                if (m_axi_bresp != 2'b00) dbg_bresp_err <= dbg_bresp_err + 32'd1;
            end
        end
    end

    // ---- AXI write payload-stability assertion (simulation only) -------------
`ifndef YOSYS
    reg [AXI_ADDR_W-1:0] aw_q;  reg aw_h;
    always @(posedge clk) begin
        if (rst) aw_h <= 1'b0;
        else begin
            if (m_axi_awvalid && !m_axi_awready) begin
                if (aw_h && (m_axi_awaddr !== aw_q))
                    $fatal(1, "axi_boot_writer: AWADDR changed while AWVALID held -- AXI4 violation");
                aw_q <= m_axi_awaddr;  aw_h <= 1'b1;
            end else aw_h <= 1'b0;
        end
    end
`endif

endmodule
`default_nettype wire
