// ============================================================================
// l3_top.v -- the L3 BOARD TOP: glm_q4k_system_cdc wired to REAL subsystems
//
//   bringup_harness.v exists to measure a fit: it buries every memory-side port
//   and feeds every response from an LFSR.  THIS file is the opposite: the same
//   fitted compact config, but every stub replaced by the real thing --
//
//     UART pins  -> uart_host_bridge -> start/prompt_tok/start_pos/s_len
//     SPI pins   -> spi_flash_reader -> boot_loader (power-up image copy)
//     boot DDR writes  -> address-decoded:
//                           0xE... -> em_store  (embedding, LUTRAM, async read)
//                           0xF... -> fn_store  (final-norm gamma, LUTRAM)
//                           else   -> axi_boot_writer (AW/W/B to the MIG)
//     runtime mem_req/resp -> ddr4_mig_shim (AR/R to the SAME MIG port; read
//                             and write channels are independent, so boot and
//                             runtime share the port with no arbiter)
//
//   The die-side latency story is the PROVEN one: LOOPBACK / LOOPBACK_FW /
//   LOOPBACK_REST route the aw/fw/rw/lw/gn weight pulls out through ddr5_xbar
//   (thus through the shim, thus from real DDR4) with the die clock-gated per
//   beat, and SELF_KV=1 closes the KV read internally.  Those parameters became
//   reachable from this top in commit 0959277; every stub input they replace is
//   tied off here.
//
//   em/fn stay SAME-CYCLE combinational pulls by contract (glm_model_q4k:514
//   "em_val answered the SAME cycle"), so their stores are LUT-RAM (async read),
//   boot-filled.  At the fitted config that is VOCAB*MODEL_DIM*16b = 512 Kb for
//   em -- [EST] ~8 K LUTs as LUTRAM; moving it to BRAM needs a die clock-gate
//   term (the LOOPBACK trick) and is left as a named optimisation, not assumed.
//
//   WHAT THIS FILE IS NOT (stated, not pretended):
//     * not fitted -- no Vivado in this environment; `make l3-elab` proves it
//       ELABORATES (iverilog) and passes yosys hierarchy -check, nothing more;
//     * the DDR image layout for the loopback address encoding needs its own
//       packer mode (the S5 crosscheck covered the wl staging layout); OPEN;
//     * MIG IP instantiation, pins, clocks and constraints are board-side.
// ============================================================================
`default_nettype none

module l3_top #(
    // fitted compact config -- MUST match fpga/bringup_harness.v (the measured fit)
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,
    parameter integer N_DENSE    = 3,
    parameter integer VOCAB      = 256,
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 2,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    parameter integer ACT_HW     = 1,
    parameter integer CACHE_SLOTS = 2,
    parameter integer FLASH_LAT   = 8,
    parameter integer KV_CTX      = 1024,
    parameter integer KV_RESIDENT = 8,
    parameter integer EFIFO_DEPTH = 8,
    parameter integer DDR_NCH     = 2,
    parameter integer DDR_ADDR_W  = 40,   // loopback keys + marker at 32 (see LB_MARKER_LSB)
    parameter integer DDR_DATA_W  = 256,
    parameter integer DDR_TAG_W   = 8,
    parameter integer DDR_ROW_LAT = 10,
    parameter integer DDR_RESP_QD = 4,
    parameter integer WL_KMAX     = 256,
    parameter integer WL_ADDR_W   = 24,
    parameter integer LOADER_KLEN = MODEL_DIM,
    parameter integer REQ_AW      = 2,
    parameter integer TOK_AW      = 3,
    parameter integer UART_CLK_DIV = 434,     // host_clk / baud
    parameter integer LB_MARKER_LSB = 32,     // loopback marker byte position (24 = the
                                              //   committed default, which the guards REJECT
                                              //   at this geometry -- the liveness leg proves it)
    parameter integer BOOT_SEGS    = 3,       // weights / em / fn
    // derived (mirror glm_q4k_system_cdc exactly; do NOT override)
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer A_KMAX     = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    parameter integer A_OMAX     = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer A_NGMAX    = (A_OMAX + PE_N - 1) / PE_N,
    parameter integer A_GRPW     = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX),
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1),
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer WL_DATA_W  = 256,
    parameter integer BOOT_DW    = 64,
    parameter integer CHW        = (DDR_NCH <= 1) ? 1 : $clog2(DDR_NCH),
    parameter integer AXI_ID_W   = CHW + DDR_TAG_W
)(
    input  wire host_clk,
    input  wire host_rst,
    input  wire core_clk,
    input  wire core_rst,

    // ---- board pins ----
    input  wire uart_rx,
    output wire uart_tx,
    output wire spi_cs_n,
    output wire spi_sclk,
    output wire spi_mosi,
    input  wire spi_miso,
    output wire boot_done_led,
    output wire boot_fail_led,

    // ---- ONE AXI4 master to the MIG (reads: runtime; writes: boot) ----
    output wire [AXI_ID_W-1:0]     m_axi_arid,
    output wire [31:0]             m_axi_araddr,
    output wire [7:0]              m_axi_arlen,
    output wire [2:0]              m_axi_arsize,
    output wire [1:0]              m_axi_arburst,
    output wire                    m_axi_arvalid,
    input  wire                    m_axi_arready,
    input  wire [AXI_ID_W-1:0]     m_axi_rid,
    input  wire [DDR_DATA_W-1:0]   m_axi_rdata,
    input  wire [1:0]              m_axi_rresp,
    input  wire                    m_axi_rlast,
    input  wire                    m_axi_rvalid,
    output wire                    m_axi_rready,
    output wire [31:0]             m_axi_awaddr,
    output wire [7:0]              m_axi_awlen,
    output wire [2:0]              m_axi_awsize,
    output wire [1:0]              m_axi_awburst,
    output wire                    m_axi_awvalid,
    input  wire                    m_axi_awready,
    output wire [BOOT_DW-1:0]      m_axi_wdata,
    output wire [BOOT_DW/8-1:0]    m_axi_wstrb,
    output wire                    m_axi_wlast,
    output wire                    m_axi_wvalid,
    input  wire                    m_axi_wready,
    input  wire [1:0]              m_axi_bresp,
    input  wire                    m_axi_bvalid,
    output wire                    m_axi_bready
);

    // ===================== host: UART bridge ================================
    wire        u_start;
    wire [15:0] u_tok, u_pos;
    wire [7:0]  u_slen;
    wire        tok_valid;
    wire [TOKW-1:0] next_tok;
    wire        busy, done;

    uart_host_bridge #(.CLK_DIV(UART_CLK_DIV)) u_uart (
        .clk(host_clk), .rst(host_rst),
        .uart_rx(uart_rx), .uart_tx(uart_tx),
        .start(u_start), .prompt_tok(u_tok), .start_pos(u_pos), .s_len(u_slen),
        .tok_valid(tok_valid), .next_tok({{(16-TOKW){1'b0}}, next_tok}));

    // ===================== boot: SPI -> boot_loader ==========================
    wire              bf_req;
    wire [31:0]       bf_addr;
    wire              bf_ready, bf_rvalid;
    wire [BOOT_DW-1:0] bf_rdata;
    wire              b_we;
    wire [31:0]       b_addr;
    wire [BOOT_DW-1:0] b_wdata;
    wire              b_ready;
    wire              boot_busy, boot_done, boot_fail;
    wire [2:0]        boot_err;

    spi_flash_reader #(.ADDR_W(32), .DATA_W(BOOT_DW)) u_spi (
        .clk(host_clk), .rst(host_rst),
        .flash_req(bf_req), .flash_addr(bf_addr), .flash_ready(bf_ready),
        .flash_rvalid(bf_rvalid), .flash_rdata(bf_rdata),
        .spi_cs_n(spi_cs_n), .spi_sclk(spi_sclk), .spi_mosi(spi_mosi),
        .spi_miso(spi_miso));

    //   segments: 0 = DDR weight image, 1 = embedding table, 2 = final-norm gamma.
    //   The em/fn ddr_base values put them in the 0xE/0xF decode windows below.
    //   Segment geometry is a BOARD IMAGE property; these are the sim defaults and
    //   the real image generator must agree (named open item in the L3 doc).
    reg boot_start_q;
    always @(posedge host_clk) boot_start_q <= !host_rst;   // one-shot after reset
    wire boot_start = !host_rst && !boot_start_q;

    boot_loader #(.ADDR_W(32), .DATA_W(BOOT_DW), .SEG_MAX(4), .BURST(8),
                  .LEN_W(16), .INTEGRITY(0)) u_boot (
        .clk(host_clk), .rst(host_rst), .start(boot_start),
        .seg_count(3'd3),
        .seg_flash_base({32'h0, 32'h0002_0000, 32'h0001_0000, 32'h0000_0000}),
        .seg_ddr_base  ({32'h0, 32'hF000_0000, 32'hE000_0000, 32'h0000_0000}),
        .seg_len       ({16'h0, 16'd32,        16'd8192,      16'd16384}),
        .mf_magic(32'h4D4F_444C), .mf_version(16'd1), .mf_len(32'd0), .mf_crc(32'd0),
        .flash_req(bf_req), .flash_addr(bf_addr), .flash_ready(bf_ready),
        .flash_rvalid(bf_rvalid), .flash_rdata(bf_rdata),
        .ddr_we(b_we), .ddr_addr(b_addr), .ddr_wdata(b_wdata), .ddr_ready(b_ready),
        .busy(boot_busy), .done(boot_done),
        /* verilator lint_off PINCONNECTEMPTY */ .words_done(), /* lint_on */
        .boot_fail(boot_fail), .err_code(boot_err));

    assign boot_done_led = boot_done;
    assign boot_fail_led = boot_fail;

    // ---- boot write decode: em / fn stores vs DDR (AXI) ---------------------
    wire b_is_em  = (b_addr[31:28] == 4'hE);
    wire b_is_fn  = (b_addr[31:28] == 4'hF);
    wire b_is_ddr = !b_is_em && !b_is_fn;
    wire aw_ready;
    assign b_ready = b_is_ddr ? aw_ready : 1'b1;   // stores accept every cycle

    axi_boot_writer #(.ADDR_W(32), .DATA_W(BOOT_DW), .AXI_DATA_W(BOOT_DW)) u_bwr (
        .clk(host_clk), .rst(host_rst),
        .ddr_we(b_we && b_is_ddr), .ddr_addr(b_addr), .ddr_wdata(b_wdata),
        .ddr_ready(aw_ready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        /* verilator lint_off PINCONNECTEMPTY */
        .dbg_writes(), .dbg_bresp_err());

    // ---- em / fn stores: LUTRAM, async read (the SAME-CYCLE contract) -------
    //   64b boot words carry 4 bf16 elements, MSB-first to match the SPI image.
    localparam integer EM_ELEMS = VOCAB * MODEL_DIM;      // 256*128 = 32768
    reg [15:0] em_store [0:EM_ELEMS-1];
    reg [15:0] fn_store [0:MODEL_DIM-1];
    integer bi;
    always @(posedge host_clk) begin
        if (b_we && b_is_em)
            for (bi = 0; bi < 4; bi = bi + 1)
                em_store[{b_addr[13:0], 2'b00} + bi[1:0]]
                    <= b_wdata[BOOT_DW-1-16*bi -: 16];
        if (b_we && b_is_fn)
            for (bi = 0; bi < 4; bi = bi + 1)
                fn_store[{b_addr[4:0], 2'b00} + bi[1:0]]
                    <= b_wdata[BOOT_DW-1-16*bi -: 16];
    end

    // ===================== runtime memory: MIG shim (AR/R) ===================
    wire [DDR_NCH-1:0]              mem_req_valid, mem_req_ready;
    wire [DDR_NCH*DDR_ADDR_W-1:0]   mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]    mem_req_tag;
    wire [DDR_NCH-1:0]              mem_resp_valid, mem_resp_ready;
    wire [DDR_NCH*DDR_DATA_W-1:0]   mem_resp_data;
    wire [DDR_NCH*DDR_TAG_W-1:0]    mem_resp_tag;

    ddr4_mig_shim #(.N_CH(DDR_NCH), .ADDR_W(DDR_ADDR_W), .DATA_W(DDR_DATA_W),
                    .TAG_W(DDR_TAG_W)) u_shim (
        .clk(core_clk), .rst(core_rst),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        /* verilator lint_off PINCONNECTEMPTY */
        .dbg_ar_issued(), .dbg_r_returned(), .dbg_rresp_err());

    // ===================== the product top ===================================
    wire        em_req;  wire [TOKW-1:0] em_tok;  wire [DIMW-1:0] em_idx;
    wire        fn_req;  wire [DIMW-1:0] fn_idx;
    wire [15:0] em_val = em_store[em_tok * MODEL_DIM + em_idx];  // async: same cycle
    wire [15:0] fn_val = fn_store[fn_idx];

    //   inference must not start before the weights are in DDR
    wire sys_start = u_start && boot_done && !boot_fail;

    glm_q4k_system_cdc #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .ACT_HW(ACT_HW), .CACHE_SLOTS(CACHE_SLOTS),
        .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX), .KV_RESIDENT(KV_RESIDENT),
        .EFIFO_DEPTH(EFIFO_DEPTH),
        .RESIDENT(1),                    // runtime weights from DDR, never Flash
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN),
        .REQ_AW(REQ_AW), .TOK_AW(TOK_AW),
        // the PHY-closure path, reachable since 0959277
        .LOOPBACK(1), .LOOPBACK_FW(1), .LOOPBACK_REST(1), .SELF_KV(1),
        .EXPERT_STALL(1), .SYS_REQ_LANES(1), .LB_MARKER_LSB(LB_MARKER_LSB),
        .HDR_LATE(1)
    ) u_sys (
        .host_clk(host_clk), .host_rst(host_rst),
        .core_clk(core_clk), .core_rst(core_rst),
        .start(sys_start),
        .prompt_tok(u_tok[TOKW-1:0]), .start_pos(u_pos[POSW-1:0]),
        .s_len(u_slen[IDXW:0]),
        .busy(busy), .done(done), .next_tok(next_tok), .tok_valid(tok_valid),
        /* verilator lint_off PINCONNECTEMPTY */
        .logits(),
        // em / fn: REAL boot-filled stores (same-cycle async reads)
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(), .idx_fresh(), .idx_win(),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        // families covered by LOOPBACK/LOOPBACK_FW/LOOPBACK_REST/SELF_KV:
        //   the die consumes the xbar round-trip internally; these stub inputs
        //   are don't-care and tied off.
        .gn_req(), .gn_which(), .gn_idx(), .gn_val(16'h0),
        .aw_req(), .aw_sel(), .aw_grp(), .aw_k(),
        .aw_q({(PE_N*4){1'b0}}),
        .aw_d({(16*PE_N*A_NSB){1'b0}}), .aw_dmin({(16*PE_N*A_NSB){1'b0}}),
        .aw_scales({(96*PE_N*A_NSB){1'b0}}),
        .rw_req(), .rw_k(),
        .rw_q({(4*N_EXPERT){1'b0}}),
        .rw_d({(16*N_EXPERT*R_NSB){1'b0}}), .rw_dmin({(16*N_EXPERT*R_NSB){1'b0}}),
        .rw_scales({(96*N_EXPERT*R_NSB){1'b0}}),
        .fw_req(), .fw_sel(), .fw_grp(), .fw_k(), .fw_shared(), .fw_eidx(),
        .fw_q({(4*TN){1'b0}}), .fw_q_up({(4*TN){1'b0}}),
        .fw_d_g({(16*TN*FF_NSB_D){1'b0}}), .fw_dmin_g({(16*TN*FF_NSB_D){1'b0}}),
        .fw_scales_g({(96*TN*FF_NSB_D){1'b0}}),
        .fw_d_u({(16*TN*FF_NSB_D){1'b0}}), .fw_dmin_u({(16*TN*FF_NSB_D){1'b0}}),
        .fw_scales_u({(96*TN*FF_NSB_D){1'b0}}),
        .lw_req(), .lw_vtile(), .lw_k(), .lw_col({(LM_TN*16){1'b0}}),
        .kc_ckv({(KV_LORA*16){1'b0}}), .kc_krope({(ROPE*16){1'b0}}),
        .kc_req(), .kc_idx(),
        .kv_row_sel(), .kv_row_in({ROW_BITS{1'b0}}),
        // KV spill / cold-expert flash: sized out at L3 (KV_RESIDENT covers the
        // demo context) -- never granted, tied off
        .flash_req(), .flash_is_expert(), .flash_expert_id(), .flash_row_idx(),
        .flash_done(1'b0), .flash_row({ROW_BITS{1'b0}}),
        .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}),
        // runtime memory: the REAL shim
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        // weight-loader staging port: observability-only at the product top
        .wl_mem_en(), .wl_mem_addr(), .wl_mem_data({WL_DATA_W{1'b0}}),
        // telemetry: unconnected (outputs)
        .argmax_o(), .h_state(), .mdl_busy(),
        .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(),
        .ec_hit_count(), .ec_miss_count(), .ec_demand_stall_cycles(),
        .ec_pf_issued(), .ec_pf_hit(),
        .kv_row_valid(), .kv_row_out(), .kv_busy(),
        .kv_append_count(), .kv_resident_lo(), .kv_overflowed(),
        .ec_dropped(), .xbar_req_count(), .xbar_resp_count(),
        .xbar_resp_valid(), .xbar_resp_data(),
        .loader_busy(), .loader_done_count(), .loader_beat_count(),
        .loader_w_q(), .loader_in_valid()
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, done, boot_busy, boot_err, em_req, fn_req,
                     |u_tok, |u_pos, |u_slen};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
`default_nettype wire
