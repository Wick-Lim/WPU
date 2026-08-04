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
    parameter integer WT_SEGLEN    = 16384,   // DDR weight seg length in boot words
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
    output wire [DDR_ADDR_W-1:0]   m_axi_araddr,
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

    // ---- header-store geometry (used by the boot segment table below) --------
    localparam integer AWW  = (16+16+96)*PE_N*A_NSB;            // 256b at the fit
    localparam integer FWW  = (16+16+96)*TN*FF_NSB_D * 2;       // gate+up paired
    localparam integer RWW  = (16+16+96)*N_EXPERT*R_NSB;
    localparam integer AW_AB = LAYW + 4 + A_GRPW;               // {ly, sel, grp}
    //   fw store grp-field widths are DERIVED per pass type (routed vs shared);
    //   the first version used literals 5/6 -- correct at the fitted config,
    //   out of range (fw_grp[5:0] of a 4-bit wire) at any smaller one.
    localparam integer GR_R  = ((INTER_MOE  >MODEL_DIM)?INTER_MOE  :MODEL_DIM)/TN;
    localparam integer GR_S  = ((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN;
    localparam integer GW_R  = (GR_R<=1)?1:$clog2(GR_R);
    localparam integer GW_S  = (GR_S<=1)?1:$clog2(GR_S);
    localparam integer FR_AB = LAYW + 1 + EIDXW + GW_R;         // {ly, sel1, eidx, grp}
    localparam integer FS_AB = LAYW + 1 + GW_S;                 // {ly, sel1, grp}

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

    //   The segment table is DERIVED from the store geometry -- the first version
    //   hand-maintained these numbers and immediately shipped a real bug: the
    //   fw-routed dense space is 4096 x 16 = 65,536 boot words, one past what
    //   LEN_W=16 can express, and a table entry of 16'd16384 quarter-filled the
    //   store while elaborating cleanly.  Derived lengths make that class of
    //   error impossible; LEN_W=20 covers every store at any supported config.
    localparam integer AW_SEGLEN  = (1 << AW_AB) * (AWW/BOOT_DW);
    localparam integer FWR_SEGLEN = (1 << FR_AB) * (FWW/BOOT_DW);
    localparam integer FWS_SEGLEN = (1 << FS_AB) * (FWW/BOOT_DW);
    localparam integer RW_SEGLEN  = (1 << ((LAYW<3)?3:LAYW)) * (RWW/BOOT_DW);
    localparam integer EM_SEGLEN  = (VOCAB*MODEL_DIM)/4;      // 4 bf16 / boot word
    localparam integer FN_SEGLEN  = MODEL_DIM/4;
    //   WT_SEGLEN is a top PARAMETER (below), not derived: the DDR weight image
    //   length is a board-image property, and the E2E sim overrides it small.
    //   flash layout: segments packed back-to-back in seg order (0..6)
    localparam integer FB0 = 0;
    localparam integer FB1 = FB0 + WT_SEGLEN;
    localparam integer FB2 = FB1 + EM_SEGLEN;
    localparam integer FB3 = FB2 + FN_SEGLEN;
    localparam integer FB4 = FB3 + AW_SEGLEN;
    localparam integer FB5 = FB4 + FWR_SEGLEN;
    localparam integer FB6 = FB5 + FWS_SEGLEN;
    boot_loader #(.ADDR_W(32), .DATA_W(BOOT_DW), .SEG_MAX(8), .BURST(8),
                  .LEN_W(20), .INTEGRITY(0)) u_boot (
        .clk(host_clk), .rst(host_rst), .start(boot_start),
        .seg_count(4'd7),
        //   seg 0 weights->DDR, 1 em, 2 fn, 3 aw-store, 4 fw-routed, 5 fw-shared, 6 rw
        .seg_flash_base({32'h0,
                         32'(FB6), 32'(FB5), 32'(FB4), 32'(FB3),
                         32'(FB2), 32'(FB1), 32'(FB0)}),
        .seg_ddr_base  ({32'h0,
                         32'hD000_0000, 32'hC000_0000, 32'hB000_0000, 32'hA000_0000,
                         32'hF000_0000, 32'hE000_0000, 32'h0000_0000}),
        .seg_len       ({20'h0,
                         20'(RW_SEGLEN), 20'(FWS_SEGLEN), 20'(FWR_SEGLEN), 20'(AW_SEGLEN),
                         20'(FN_SEGLEN), 20'(EM_SEGLEN), 20'(WT_SEGLEN)}),        .mf_magic(32'h4D4F_444C), .mf_version(16'd1), .mf_len(32'd0), .mf_crc(32'd0),
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
    wire b_is_hdr = (b_addr[31:28] >= 4'hA) && (b_addr[31:28] <= 4'hD);
    wire b_is_ddr = !b_is_em && !b_is_fn && !b_is_hdr;
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
    localparam integer EM_ELEMS = VOCAB * MODEL_DIM;
    localparam integer EMA_W    = $clog2(EM_ELEMS/4);
    localparam integer FNA_W    = $clog2((MODEL_DIM/4 < 2) ? 2 : MODEL_DIM/4);
    reg [15:0] em_store [0:EM_ELEMS-1];
    reg [15:0] fn_store [0:MODEL_DIM-1];
    integer bi;
    always @(posedge host_clk) begin
        if (b_we && b_is_em)
            for (bi = 0; bi < 4; bi = bi + 1)
                em_store[{b_addr[EMA_W-1:0], 2'b00} + bi[1:0]]
                    <= b_wdata[BOOT_DW-1-16*bi -: 16];
        if (b_we && b_is_fn)
            for (bi = 0; bi < 4; bi = bi + 1)
                fn_store[{b_addr[FNA_W-1:0], 2'b00} + bi[1:0]]
                    <= b_wdata[BOOT_DW-1-16*bi -: 16];
    end

    // ===================== runtime memory: MIG shim (AR/R) ===================
    wire [DDR_NCH-1:0]              mem_req_valid, mem_req_ready;
    wire [DDR_NCH*DDR_ADDR_W-1:0]   mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]    mem_req_tag;
    wire [DDR_NCH-1:0]              mem_resp_valid, mem_resp_ready;
    wire [DDR_NCH*DDR_DATA_W-1:0]   mem_resp_data;
    wire [DDR_NCH*DDR_TAG_W-1:0]    mem_resp_tag;

    //   AXI_ADDR_W must carry the FULL loopback address: the marker byte sits at
    //   [LB_MARKER_LSB +: 8] = [32 +: 8] here, and the shim's default 32-bit AXI
    //   address would silently TRUNCATE it -- every marked read would alias into
    //   the plain-DDR space.  Found while designing the E2E gate, not by it.
    ddr4_mig_shim #(.N_CH(DDR_NCH), .ADDR_W(DDR_ADDR_W), .DATA_W(DDR_DATA_W),
                    .TAG_W(DDR_TAG_W), .AXI_ADDR_W(DDR_ADDR_W)) u_shim (
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

    // ===================== dequant HEADER stores (item 0, second half) ========
    //   HDR_LATE=1 (c54224a) moved the matmul's header consumption to accept
    //   time, so a sync-read BRAM addressed by the pass key delivers EXACTLY in
    //   time: key settles on the edge entering the start cycle, the BRAM samples
    //   it one edge later, data valid during the first-accept cycle.
    //   Sizing (f686217): aw dense 1.5 Mb; fw MUST split (dense would need ~24 Mb
    //   against a 12.6 Mb budget) into routed + shared/dense; rw is per-layer
    //   only.  ~190 of 360 RAMB36, against a measured fit using 0.
    //   Boot decode windows: 0xA aw / 0xB fw-routed / 0xC fw-shared / 0xD rw.
    //   64b boot words assemble MSB-first into the wide store words.

    wire [LAYW-1:0]  db_layer;
    wire [3:0]       aw_sel;
    wire [A_GRPW-1:0] aw_grp;
    wire [1:0]       fw_sel;
    wire [FF_GWD-1:0] fw_grp;
    wire             fw_shared;
    wire [EIDXW-1:0] fw_eidx;

    reg [AWW-1:0] aw_store [0:(1<<AW_AB)-1];
    reg [FWW-1:0] fwr_store[0:(1<<FR_AB)-1];
    reg [FWW-1:0] fws_store[0:(1<<FS_AB)-1];
    reg [RWW-1:0] rw_store [0:(1<<((LAYW<3)?3:LAYW))-1];

    // sync reads in the CORE clock domain (the die's domain)
    reg [AWW-1:0] aw_q_st;
    reg [FWW-1:0] fwr_q_st, fws_q_st;
    reg [RWW-1:0] rw_q_st;
    reg           fw_sh_q;
    always @(posedge core_clk) begin
        aw_q_st  <= aw_store [{db_layer, aw_sel, aw_grp}];
        //   fw_sel is {0=gate(+up), 2=down}: bit [1] distinguishes them; bit [0]
        //   is ALWAYS ZERO and using it aliased gate and down headers into the
        //   same slot -- found by inspection while writing the E2E gate.
        fwr_q_st <= fwr_store[{db_layer, fw_sel[1], fw_eidx, fw_grp[GW_R-1:0]}];
        fws_q_st <= fws_store[{db_layer, fw_sel[1], fw_grp[GW_S-1:0]}];
        rw_q_st  <= rw_store [db_layer];
        fw_sh_q  <= fw_shared;      // align the split-select with the read latency
    end
    wire [FWW-1:0] fw_q_st = fw_sh_q ? fws_q_st : fwr_q_st;

    // unpack: word = {scales, dmin, d} per bus, gate half then up half for fw
    localparam integer AWD_B = 16*PE_N*A_NSB;
    localparam integer FWD_B = 16*TN*FF_NSB_D;
    localparam integer FWS_B = 96*TN*FF_NSB_D;
    localparam integer RWD_B = 16*N_EXPERT*R_NSB;
    wire [AWD_B-1:0]           aw_d_st      = aw_q_st[0 +: AWD_B];
    wire [AWD_B-1:0]           aw_dmin_st   = aw_q_st[AWD_B +: AWD_B];
    wire [96*PE_N*A_NSB-1:0]   aw_scales_st = aw_q_st[2*AWD_B +: 96*PE_N*A_NSB];
    wire [FWD_B-1:0]           fw_d_g_st      = fw_q_st[0 +: FWD_B];
    wire [FWD_B-1:0]           fw_dmin_g_st   = fw_q_st[FWD_B +: FWD_B];
    wire [FWS_B-1:0]           fw_scales_g_st = fw_q_st[2*FWD_B +: FWS_B];
    wire [FWD_B-1:0]           fw_d_u_st      = fw_q_st[2*FWD_B+FWS_B +: FWD_B];
    wire [FWD_B-1:0]           fw_dmin_u_st   = fw_q_st[3*FWD_B+FWS_B +: FWD_B];
    wire [FWS_B-1:0]           fw_scales_u_st = fw_q_st[3*FWD_B+2*FWS_B +: FWS_B];
    wire [RWD_B-1:0]           rw_d_st      = rw_q_st[0 +: RWD_B];
    wire [RWD_B-1:0]           rw_dmin_st   = rw_q_st[RWD_B +: RWD_B];
    wire [96*N_EXPERT*R_NSB-1:0] rw_scales_st = rw_q_st[2*RWD_B +: 96*N_EXPERT*R_NSB];

    // ---- boot fill (host clock; boot completes before inference starts) ------
    //   64b sub-words assemble MSB-first; a store word commits on its LAST
    //   sub-word.  Boot address low bits select the sub-word.
    localparam integer AW_SW  = AWW/BOOT_DW;
    localparam integer FW_SW  = FWW/BOOT_DW;
    localparam integer RW_SW  = RWW/BOOT_DW;
    //   sub-word select widths are DERIVED -- the first version hardcoded the
    //   fitted config's counts ([1:0]/[3:0]), which silently mis-filled every
    //   store at any other geometry (e.g. rw at the E2E config is 8 sub-words,
    //   not 16).  Same bug class as the hand-maintained seg table.
    localparam integer AW_SWW = $clog2(AW_SW);
    localparam integer FW_SWW = $clog2(FW_SW);
    localparam integer RW_SWW = $clog2(RW_SW);
    wire b_is_aws = (b_addr[31:28] == 4'hA);
    wire b_is_fwr = (b_addr[31:28] == 4'hB);
    wire b_is_fws = (b_addr[31:28] == 4'hC);
    wire b_is_rws = (b_addr[31:28] == 4'hD);
    reg [FWW-BOOT_DW-1:0] b_sh;             // widest assembler (top word arrives last)
    always @(posedge host_clk) begin
        if (b_we && (b_is_aws || b_is_fwr || b_is_fws || b_is_rws))
            b_sh <= {b_sh[FWW-2*BOOT_DW-1:0], b_wdata};
        if (b_we && b_is_aws && (b_addr[AW_SWW-1:0] == AW_SW-1))
            aw_store[b_addr[AW_SWW +: AW_AB]] <= {b_sh[AWW-BOOT_DW-1:0], b_wdata};
        if (b_we && b_is_fwr && (b_addr[FW_SWW-1:0] == FW_SW-1))
            fwr_store[b_addr[FW_SWW +: FR_AB]] <= {b_sh[FWW-BOOT_DW-1:0], b_wdata};
        if (b_we && b_is_fws && (b_addr[FW_SWW-1:0] == FW_SW-1))
            fws_store[b_addr[FW_SWW +: FS_AB]] <= {b_sh[FWW-BOOT_DW-1:0], b_wdata};
        if (b_we && b_is_rws && (b_addr[RW_SWW-1:0] == RW_SW-1))
            rw_store[b_addr[RW_SWW +: ((LAYW<3)?3:LAYW)]] <= {b_sh[RWW-BOOT_DW-1:0], b_wdata};
    end

    wire kvf_req;
    reg  [3:0] kvf_sh;
    always @(posedge core_clk) begin
        if (core_rst) kvf_sh <= 4'b0;
        else          kvf_sh <= {kvf_sh[2:0], kvf_req};
    end
    wire kvf_done = kvf_sh[3] && kvf_req;   // 4-cycle latency, held-req protocol

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
        .db_layer(db_layer), .idx_fresh(), .idx_win(),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        // families covered by LOOPBACK/LOOPBACK_FW/LOOPBACK_REST/SELF_KV:
        //   the die consumes the xbar round-trip internally; these stub inputs
        //   are don't-care and tied off.
        .gn_req(), .gn_which(), .gn_idx(), .gn_val(16'h0),
        .aw_req(), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(),
        .aw_q({(PE_N*4){1'b0}}),
        .aw_d(aw_d_st), .aw_dmin(aw_dmin_st),
        .aw_scales(aw_scales_st),
        .rw_req(), .rw_k(),
        .rw_q({(4*N_EXPERT){1'b0}}),
        .rw_d(rw_d_st), .rw_dmin(rw_dmin_st),
        .rw_scales(rw_scales_st),
        .fw_req(), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(), .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q({(4*TN){1'b0}}), .fw_q_up({(4*TN){1'b0}}),
        .fw_d_g(fw_d_g_st), .fw_dmin_g(fw_dmin_g_st),
        .fw_scales_g(fw_scales_g_st),
        .fw_d_u(fw_d_u_st), .fw_dmin_u(fw_dmin_u_st),
        .fw_scales_u(fw_scales_u_st),
        .lw_req(), .lw_vtile(), .lw_k(), .lw_col({(LM_TN*16){1'b0}}),
        .kc_ckv({(KV_LORA*16){1'b0}}), .kc_krope({(ROPE*16){1'b0}}),
        .kc_req(), .kc_idx(),
        .kv_row_sel(), .kv_row_in({ROW_BITS{1'b0}}),
        // KV spill / cold-expert flash.  KV_RESIDENT covers every REAL row at
        // the demo context -- but the FIRST token of a sequence still issues a
        // COLD gather: the commit pulse fires AFTER a token's gathers (causal:
        // attend 0..s_len-1, append at s_len), so the empty-KV first gather
        // falls outside [resident_lo, append_count) and the pager HOLDS
        // flash_req until flash_done.  The proven self-kv TBs answered it with
        // a ZERO row (the reference shadow reads zeros there too); tying
        // flash_done=0 -- the first version here -- deadlocks the die on token
        // 0.  Found by the E2E gate.  kvf_sh answers every flash_req with a
        // zero row after a short fixed latency.
        .flash_req(kvf_req), .flash_is_expert(), .flash_expert_id(), .flash_row_idx(),
        .flash_done(kvf_done), .flash_row({ROW_BITS{1'b0}}),
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
