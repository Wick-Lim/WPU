`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// weight_loader_q4k.v  --  WEIGHT-SIDE DMA / sequencer for glm_matmul_q4k
//                          (ACCEL_GLM52 local-device Q4_K target)
//----------------------------------------------------------------------------
// FUNCTION  (the Q4_K sibling of weight_loader.v)
//   glm_matmul_q4k is a GGML Q4_K super-block GEMM that PULLS its weights: at
//   `start` it latches, per output column pj and super-block sb, the fp16 d,
//   the fp16 dmin and the 96-bit packed 6-bit scales/mins; then it walks K
//   consuming one 4-bit Q4_K code per column (w_q[pj]) each beat in_valid is
//   asserted.  This loader is the master that DRIVES that pull from a simple
//   read-memory (the TB models real DDR5/Flash):
//
//     1. Given a tile DESCRIPTOR {base, k_len, n_sblk}, read the tile's per-
//        (column, super-block) HEADER (d/dmin/scales) into the packed
//        mm_w_d/mm_w_dmin/mm_w_scales buses, then present them with the `start`
//        pulse (clearing glm_matmul_q4k's banks + latching k_len/params).
//     2. Stream the Q4_K weight codes mm_w_q[k] one beat per K position,
//        asserting in_valid, so the GEMM consumes exactly k_len beats
//        k=0..k_len-1.
//
//   The activation side (a_col) is driven separately; this loader owns ONLY the
//   weight side.  It is the beat master: it drives in_valid, so the activation
//   provider keeps pace with the weight stream.
//
//   This is a MECHANICAL retarget of weight_loader.v: identical FSM
//   (S_SCALE -> S_START -> S_STREAM -> S_DONE), identical latency-1 capture
//   pipeline and start/stream timing.  Only the WEIGHT FORMAT changes:
//     * the prior FP8 SCALE region (one bf16 word per (K-block, col)) becomes the
//       Q4_K HEADER region (one {d,dmin,scales} word per (col, super-block)).
//     * the prior FP8 CODE region (8-bit weight byte per col per beat) becomes the Q4_K
//       CODE region (4-bit quant code per col per beat, mm_w_q[4*PE_N]).
//   DATA_W stays 256 (a Q4_K super-block header packs into one word; wide enough
//   for a nibble-code beat too), aligning with the ddr5_xbar beat width.
//
// WEIGHT-MEMORY (DESCRIPTOR) LAYOUT  -- word-addressed:
//   A tile occupies a contiguous region from `base`:
//     * HEADER region : base + (pj*n_sblk + sb),  pj=0..PE_N-1, sb=0..n_sblk-1
//                       (pj OUTER, super-block INNER) one packed header word each:
//                         word[15:0]    = d    (fp16 super-block scale)
//                         word[31:16]   = dmin (fp16 super-block min)
//                         word[127:32]  = scales (96b: 8x6b block-scales + 8x6b mins)
//                       ( n_sblk*PE_N words )
//     * CODE   region : base + n_sblk*PE_N + k,   for k=0..k_len-1
//                       one Q4_K code ROW per word: word[4*pj +: 4] = code W[k][pj],
//                       the EXACT packing glm_matmul_q4k expects on w_q.
//                       ( k_len words )
//   NOTE the on-disk GGUF native order is qs[128] = 4-bit nibbles for all 256
//   weights of a super-block; the packer (tools/ckpt_pack_q4k.py) unpacks nibble
//   k of column pj into this per-beat CODE layout so this loader (like its prior
//   FP8 ancestor on tag 'fp8-verified-baseline') streams one code-row word per K-beat.
//
//   The HEADER word for (col pj, super-block sb) is placed on the mm_w_* buses at
//   bus slot (pj*NSB + sb) -- the EXACT (col-outer, super-block-inner, compile-
//   time NSB stride) index glm_matmul_q4k packs its w_d/w_dmin/w_scales with.
//   Memory ADDRESS uses the compact per-tile stride (pj*n_sblk+sb); the bus SLOT
//   uses the compile-time NSB stride, so short tiles (n_sblk < NSB) land in the
//   right column banks with the unused super-blocks ZERO-filled (never X).
//
// READ-MEMORY INTERFACE (TB models DDR5/Flash):  mem_en+mem_addr presented
//   combinationally on cycle t -> mem_data valid on cycle t+1 (registered single-
//   port RAM read; latency = 1), exactly as weight_loader.v.
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch (the combinational
//   request block fully defaults its outputs); NO combinational loop.
//============================================================================
module weight_loader_q4k #(
    parameter integer PE_N   = 4,        // output columns (== glm_matmul_q4k PE_N)
    parameter integer KMAX   = 256,      // max K per tile (== glm_matmul_q4k KMAX)
    parameter integer ADDR_W = 24,       // weight-memory address width
    parameter integer DATA_W = 256,      // memory data width (Q4_K super-block header + code beat)
    // ---- WEIGHT-READ-PATH SECDED ECC (USAGE_GAPS §B / finding #32) ----
    //   DEFAULT-OFF (WEIGHT_ECC=0): the read path is a bare wire -- the shipped
    //   bit-exact 1155-test datapath is BYTE-IDENTICAL (proven by `make
    //   weight-loader-ecc-equiv`).  WHEN ON (WEIGHT_ECC=1): every DATA_W read
    //   word is modelled as ECC_LANE_W-wide SECDED lanes (the weight memory here
    //   stores no check bits, so the codec is a DECODE STAGE on the read data --
    //   each lane is (re-)encoded to its stored codeword, an injectable test
    //   port `ecc_err_inject` XORs bit-flips into that codeword, and the decoder
    //   CORRECTS single-bit errors + FLAGS double-bit errors).  Corrected-error
    //   count (scrub/telemetry) + a sticky ecc_uncorrectable are exposed.
    parameter integer WEIGHT_ECC = 0,    // 0 = OFF (byte-identical); 1 = SECDED read path
    parameter integer ECC_LANE_W = 64,   // SECDED payload lane width (DATA_W split into lanes)
    parameter integer ECC_CNT_W  = 32,   // corrected-error counter width
    // ---- derived geometry (mirror glm_matmul_q4k) ----
    localparam integer NSB = (KMAX + 255) / 256,       // #super-blocks (== w_* banks per col)
    localparam integer NB8 = (KMAX + 31)  / 32,        // #Q8_0 32-weight blocks (== 8*NSB)
    localparam integer KW  = $clog2(KMAX + 1),         // k_len width
    localparam integer SBW = $clog2(NSB + 1),          // super-block-count width
    localparam integer PJW = $clog2(PE_N + 1),         // COLUMN-index width (0..PE_N)
    localparam integer NSW = $clog2(NSB*PE_N + 1),     // header-word counter width
    // ---- ECC lane geometry (extended-Hamming SECDED, same construction as
    //      src/ecc_secded.v: P Hamming parity bits + 1 overall parity bit).
    //      ECC_P is the closed form for a POWER-OF-TWO lane width; an
    //      elaboration guard below re-derives the exact SECDED P and $fatals if
    //      it disagrees (so a pathological ECC_LANE_W cannot mis-size the port).
    localparam integer ECC_NLANE = DATA_W / ECC_LANE_W,
    localparam integer ECC_P     = $clog2(ECC_LANE_W) + 1,       // Hamming parity bits / lane
    localparam integer ECC_LCODE = ECC_LANE_W + ECC_P + 1,       // codeword bits / lane
    localparam integer ECC_CTOT  = ECC_NLANE * ECC_LCODE         // total stored-codeword width
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    // ---- tile descriptor command ----
    input  wire                       load,       // 1-cycle pulse: start a tile load
    input  wire [ADDR_W-1:0]          desc_base,  // tile base address in weight mem
    input  wire [KW-1:0]              desc_klen,  // K length (#weight beats)
    input  wire [SBW-1:0]             desc_nsblk, // #super-blocks for this tile (<= NSB)
    input  wire [2:0]                 desc_wtype, // 0=Q4_K 1=Q6_K 2=Q8_0 3=F16 4=Q5_K (undriven->Q4_K)

    // ---- read-memory interface (TB models DDR5/Flash; mem_data valid t+1) ----
    output reg                        mem_en,     // combinational request strobe
    output reg  [ADDR_W-1:0]          mem_addr,   // combinational read address
    input  wire [DATA_W-1:0]          mem_data,

    // ---- glm_matmul_q4k WEIGHT-side drive ----
    output wire                       mm_start,
    output wire [KW-1:0]              mm_k_len,
    output wire [ 4*PE_N-1:0]         mm_w_q,       // 4-bit Q4_K codes W[k][*], PE_N packed
    output wire [16*PE_N*NSB-1:0]     mm_w_d,       // fp16 d    per (col, super-block)
    output wire [16*PE_N*NSB-1:0]     mm_w_dmin,    // fp16 dmin per (col, super-block)
    output wire [96*PE_N*NSB-1:0]     mm_w_scales,  // 96b scales per (col, super-block)
    output wire                       mm_in_valid,
    // ---- ADDED: mixed-type (Q6_K/Q8_0/F16) type broadcast + high-precision buses ----
    output wire [ 3*PE_N-1:0]         mm_w_type,    // per-column type (tile is one type)
    output wire [16*PE_N-1:0]         mm_w_hp,      // per beat: Q6_K/Q8_0/F16 code lane / col
    output wire [128*PE_N*NSB-1:0]    mm_w_q6_sc,   // Q6_K 16xint8 scales per (col, super-block)
    output wire [16*PE_N*NB8-1:0]     mm_w_q8_d,    // Q8_0 fp16 d per (col, 32-weight block)

    // ---- status ----
    output reg                        busy,
    output reg                        done,       // 1-cycle pulse when tile streamed

    // ---- WEIGHT-READ-PATH SECDED ECC (WEIGHT_ECC=1 only; tied off when OFF) ----
    //   ecc_err_inject : TEST/fault-injection port -- XOR mask over the per-lane
    //                    STORED codeword (lane l occupies bits [l*ECC_LCODE +:
    //                    ECC_LCODE]).  Models bit-rot in the resident weight
    //                    array.  Ignored (optimized away) when WEIGHT_ECC=0.
    //   ecc_corr_count : running count of SECDED-CORRECTED single-bit errors on
    //                    the read path (scrub/telemetry); accumulates since rst.
    //   ecc_uncorrectable : STICKY registered flag -- set on any double-bit
    //                    (detected, uncorrectable) read; cleared on rst / load.
    input  wire [ECC_CTOT-1:0]        ecc_err_inject,
    output wire [ECC_CNT_W-1:0]       ecc_corr_count,
    output wire                       ecc_uncorrectable
);
    // -----------------------------------------------------------------------
    // FSM states  (identical to weight_loader.v)
    // -----------------------------------------------------------------------
    localparam [2:0] S_IDLE   = 3'd0,
                     S_SCALE  = 3'd1,   // read per-(col,super-block) headers into buses
                     S_START  = 3'd2,   // pulse start (+ latch params) ; present code k=0
                     S_STREAM = 3'd3,   // stream Q4_K code rows w_q[k] + in_valid
                     S_DONE   = 3'd4;   // signal completion

    // weight-type enum (matches glm_matmul_q4k / the desc_wtype field)
    localparam [2:0] WT_Q4K = 3'd0, WT_Q6K = 3'd1, WT_Q80 = 3'd2, WT_F16 = 3'd3,
                     WT_Q5K = 3'd4;

    // -----------------------------------------------------------------------
    // GENERATE-TIME PRECONDITION (Q8_0 header-pack): the Q8_0 path co-packs the 8
    // per-32-block fp16 d of a super-block into ONE header word and writes it as
    // q8d_q[128*rd_slot +: 128] on the bus of width 16*PE_N*NB8.  That is lossless
    // only when NB8 == 8*NSB, i.e. KMAX is a WHOLE multiple of 256 (no partial
    // trailing super-block: a leftover <=224 K would make NB8 < 8*NSB and the
    // 128-bit write would truncate).  All real configs use 256-multiple K (WL_KMAX
    // defaults to 256); this elaboration-time assertion guards any other config.
    if (KMAX % 256 != 0)
        $fatal(1, "weight_loader_q4k: Q8_0 header-pack requires KMAX a whole multiple of 256 (NB8==8*NSB)");

    // GENERATE-TIME PRECONDITION (lane ceiling): the CODE-region beat carries the
    // per-column codes in ONE DATA_W word -- 4b/col for Q4_K (mm_w_q = rd_data[4*PE_N-1:0])
    // and 16b/col for the mixed high-precision lane (mm_w_hp = rd_data[16*PE_N-1:0]).
    // The 16b/col slice is the binding one: it needs DATA_W >= 16*PE_N.  Below that,
    // a SELRANGE is flagged under --lint-only, but IVERILOG SILENTLY ZERO-FILLS the
    // out-of-range bits and produces a wrong-but-running netlist -- exactly how PE_N>16
    // at the shipped DATA_W=256 slipped through.  Scaling lanes past 16 is therefore a
    // DELIBERATE memory-bus-width decision (DATA_W=256->16 lanes, 8192->512,
    // 32768->2048), and this makes DATA_W < 16*PE_N a loud build failure in BOTH tools
    // instead of a silent one in iverilog.  ($error, not $fatal: elaboration-time, so
    // it also fails the --lint-only pass where the SELRANGE lives.)
    if (16*PE_N > DATA_W)
        $error("weight_loader_q4k: DATA_W must be >= 16*PE_N (the mixed-type code lane is 16b/col in one beat) -- widen DATA_W to scale lanes past DATA_W/16; see the lane-ceiling note in this header");

    reg [2:0]                state;

    // latched descriptor
    reg [ADDR_W-1:0]         base_q;
    reg [KW-1:0]             klen_q;
    reg [SBW-1:0]            nsblk_q;
    reg [2:0]                wtype_q;   // latched tile type (Q4_K default on reset/undriven)

    // assembled weight header buses (zero-filled for unused super-block banks)
    reg [16*PE_N*NSB-1:0]    d_q;
    reg [16*PE_N*NSB-1:0]    dmin_q;
    reg [96*PE_N*NSB-1:0]    scales_q;
    reg [128*PE_N*NSB-1:0]   q6sc_q;    // Q6_K 16xint8 scales / (col, super-block)
    reg [16*PE_N*NB8-1:0]    q8d_q;     // Q8_0 fp16 d / (col, 32-weight block)

    // header-phase counters / latency-1 capture pipeline
    reg [NSW-1:0]            hd_iss;     // #header reads issued (linear memory index)
    reg [NSW-1:0]            hd_cap;     // #header words captured
    reg [PJW-1:0]            pj_iss;     // column of the read currently being issued (0..PE_N-1)
    reg [SBW-1:0]            sb_iss;     // super-block of the read currently being issued
    reg                      rd_v;       // a header read was requested LAST cycle
    reg [NSW-1:0]            rd_slot;    // its (pj*NSB+sb) bus slot

    // code-phase counters / in-flight read
    reg [KW-1:0]             cd_iss;     // #code reads requested (next k to fetch)
    reg [KW-1:0]             beat_cnt;   // #code-row beats driven to the GEMM
    reg                      code_pending; // mem_data this cycle is a valid w_q row

    // total header words for this tile (type-parametric), and the code-region base.
    //   F16 has NO header region (ns=0).  Q4_K/Q6_K/Q8_0 share the super-block-
    //   granular header count nsblk*PE_N (Q8_0 packs its 8 per-32-block fp16 d into
    //   one super-block header word).  Q4_K is the case DEFAULT, so an undriven
    //   desc_wtype (x/z) selects the exact byte-identical Q4_K geometry.
    reg  [NSW-1:0]           ns;
    always @* begin
        case (wtype_q)
            WT_F16:  ns = {NSW{1'b0}};
            default: ns = NSW'(nsblk_q) * NSW'(PE_N);
        endcase
    end
    // ---- Q5_K guard -------------------------------------------------------
    //   glm_matmul_q4k CAN consume Q5_K (it rides w_hp), but THIS loader cannot
    //   yet lay out a Q5_K tile: that needs the packer to emit pre-assembled
    //   5-bit codes and a Q5_K header geometry (176 B/super-block: d, dmin,
    //   scales[12], qh[32], qs[128]) -- docs/GLM53_FLASH_PORT.md 4.2 item 6.
    //   Without this guard a Q5_K descriptor takes the `default` above and
    //   streams Q4_K geometry: same widths, wrong bytes, no error -- the exact
    //   silent-wrongness this repo builds must-fail pairs to prevent.
    //   Simulation-only: a per-cycle $fatal is not synthesizable.  `ifndef YOSYS,
    //   NOT `synthesis translate_off (a legacy hot comment yosys warns about) and
    //   NOT `ifndef SYNTHESIS -- nothing in this repo defines SYNTHESIS, so that
    //   guard would exclude nothing.  Same convention as src/mla_attn_q4k.v and
    //   src/swiglu_expert_q4k.v.  yosys predefines YOSYS.
`ifndef YOSYS
    always @(posedge clk)
        if (!rst && (state == S_STREAM) && (wtype_q == WT_Q5K))
            $fatal(1, "weight_loader_q4k: desc_wtype==WT_Q5K but this loader has no Q5_K tile geometry (packer item, docs/GLM53_FLASH_PORT.md 4.2) -- it would stream Q4_K geometry: same widths, wrong bytes, no error. Drive glm_matmul_q4k directly for Q5_K.");
`endif
    wire [ADDR_W-1:0]        ns_ext    = {{(ADDR_W-NSW){1'b0}}, ns};
    wire [ADDR_W-1:0]        code_base = base_q + ns_ext;
    // bus slot (col-outer, super-block-inner, compile-time NSB stride) of the
    // read currently being issued.
    wire [NSW-1:0]           cur_slot  = NSW'(pj_iss) * NSW'(NSB) + NSW'(sb_iss);

    // ECC-corrected read word the datapath consumes (driven below; == mem_data
    // when WEIGHT_ECC==0).  Declared here so the combinational outputs can use it.
    wire [DATA_W-1:0] rd_data;

    // -----------------------------------------------------------------------
    // Combinational weight-side outputs.  Identical shape/timing to
    // weight_loader.v; only the code field width (8->4) changes.
    // -----------------------------------------------------------------------
    assign mm_start     = (state == S_START);
    assign mm_k_len     = klen_q;
    assign mm_w_d       = d_q;
    assign mm_w_dmin    = dmin_q;
    assign mm_w_scales  = scales_q;
    assign mm_in_valid  = (state == S_STREAM) & code_pending & (beat_cnt < klen_q);
    assign mm_w_q       = mm_in_valid ? rd_data[4*PE_N-1:0] : {(4*PE_N){1'b0}};
    // ---- ADDED mixed-type drives (Q4_K reads none of these -> byte-identical) ----
    //   The tile type is broadcast to all PE_N columns (real tensors are uniform).
    //   mm_w_hp carries the per-column 16-bit code lane (Q6_K low 6 / Q8_0 low 8 /
    //   F16 all 16); mm_w_q above still carries the Q4_K 4-bit codes unchanged.
    assign mm_w_type    = {PE_N{wtype_q}};
    assign mm_w_hp      = mm_in_valid ? rd_data[16*PE_N-1:0] : {(16*PE_N){1'b0}};
    assign mm_w_q6_sc   = q6sc_q;
    assign mm_w_q8_d    = q8d_q;

    // -----------------------------------------------------------------------
    // WEIGHT-READ-PATH SECDED ECC  (USAGE_GAPS §B / finding #32)
    //   `rd_data` is the read word the datapath actually consumes.  When
    //   WEIGHT_ECC==0 it is a BARE WIRE (== mem_data) -> the shipped netlist is
    //   byte-identical.  When WEIGHT_ECC==1 it is the SECDED-corrected read word:
    //   each ECC_LANE_W lane is (re-)encoded to its stored codeword (the weight
    //   memory here holds no check bits, so ECC is modelled as a decode stage on
    //   the read data), an injectable fault mask XORs bit-flips into that
    //   codeword, and the decoder corrects singles / flags doubles.
    // -----------------------------------------------------------------------
    // Re-derive the EXACT SECDED parity count and guard the closed-form ECC_P /
    // the lane divisibility -- an elaboration $fatal beats a silent mis-size.
    function integer wl_calc_p;
        input integer dw;
        integer p;
        begin
            p = 0;
            while ((1 << p) < (dw + p + 1)) p = p + 1;
            wl_calc_p = p;
        end
    endfunction

    generate
    if (WEIGHT_ECC == 0) begin : g_wecc_off
        // ---- OFF: bare read path (byte-identical to the proven default) ----
        assign rd_data          = mem_data;
        assign ecc_corr_count   = {ECC_CNT_W{1'b0}};
        assign ecc_uncorrectable = 1'b0;
        // (ecc_err_inject is unused here and is optimized away.)
    end else begin : g_wecc_on
        // ---- ON: per-lane SECDED decode of the read word ----
        if (DATA_W % ECC_LANE_W != 0)
            $fatal(1, "weight_loader_q4k: WEIGHT_ECC requires DATA_W a whole multiple of ECC_LANE_W");
        if (ECC_P != wl_calc_p(ECC_LANE_W))
            $fatal(1, "weight_loader_q4k: ECC_P closed form disagrees with SECDED parity count -- use a power-of-two ECC_LANE_W");

        wire [DATA_W-1:0]     dec_data;
        wire [ECC_NLANE-1:0]  lane_serr;
        wire [ECC_NLANE-1:0]  lane_derr;

        genvar l;
        for (l = 0; l < ECC_NLANE; l = l + 1) begin : g_lane
            wire [ECC_LANE_W-1:0] clean_lane = mem_data[l*ECC_LANE_W +: ECC_LANE_W];
            wire [ECC_LCODE-1:0]  enc_code;    // stored codeword for this lane
            wire [ECC_LCODE-1:0]  rx_code;     // + injected bit-flips
            wire [ECC_LANE_W-1:0] lane_dout;

            // ENCODE the clean lane -> its stored codeword (models on-write ECC).
            ecc_secded #(.DATA_W(ECC_LANE_W)) u_enc (
                .data_in   (clean_lane),
                .code_out  (enc_code),
                .code_in   ({ECC_LCODE{1'b0}}),
                .data_out  (/* unused */),
                .single_err(/* unused */),
                .double_err(/* unused */)
            );

            // inject the test fault mask, then DECODE (correct single / flag double).
            assign rx_code = enc_code ^ ecc_err_inject[l*ECC_LCODE +: ECC_LCODE];
            ecc_secded #(.DATA_W(ECC_LANE_W)) u_dec (
                .data_in   ({ECC_LANE_W{1'b0}}),
                .code_out  (/* unused */),
                .code_in   (rx_code),
                .data_out  (lane_dout),
                .single_err(lane_serr[l]),
                .double_err(lane_derr[l])
            );

            assign dec_data[l*ECC_LANE_W +: ECC_LANE_W] = lane_dout;
        end

        assign rd_data = dec_data;

        // popcount of lanes that CORRECTED a single-bit error this read.
        integer si;
        reg [ECC_CNT_W-1:0] serr_pop;
        always @* begin
            serr_pop = {ECC_CNT_W{1'b0}};
            for (si = 0; si < ECC_NLANE; si = si + 1)
                if (lane_serr[si]) serr_pop = serr_pop + 1'b1;
        end
        wire any_derr = |lane_derr;

        // A read result is present the cycle after a request (mem_data valid at
        // mem_en+1), so gate telemetry on the registered request strobe.
        reg                  mem_en_q;
        reg [ECC_CNT_W-1:0]  corr_cnt_r;
        reg                  unc_r;
        always @(posedge clk) begin
            if (rst) begin
                mem_en_q   <= 1'b0;
                corr_cnt_r <= {ECC_CNT_W{1'b0}};
                unc_r      <= 1'b0;
            end else begin
                mem_en_q <= mem_en;
                if (load) unc_r <= 1'b0;          // clear sticky at each tile load
                if (mem_en_q) begin               // a genuine read returned this cycle
                    corr_cnt_r <= corr_cnt_r + serr_pop;
                    if (any_derr) unc_r <= 1'b1;  // sticky double-bit (uncorrectable)
                end
            end
        end
        assign ecc_corr_count    = corr_cnt_r;
        assign ecc_uncorrectable = unc_r;
    end
    endgenerate

    // -----------------------------------------------------------------------
    // Combinational read-request: present the next address on the bus so the
    // registered RAM returns its data one cycle later.  Fully defaulted -> no
    // latch.  Address advances off the registered counters (no comb loop).
    // -----------------------------------------------------------------------
    always @* begin
        mem_en   = 1'b0;
        mem_addr = {ADDR_W{1'b0}};
        case (state)
            S_SCALE:  if (hd_iss < ns) begin
                          mem_en   = 1'b1;
                          mem_addr = base_q + {{(ADDR_W-NSW){1'b0}}, hd_iss};
                      end
            S_START:  begin
                          mem_en   = 1'b1;             // prefetch code word k=0
                          mem_addr = code_base;
                      end
            S_STREAM: if (cd_iss < klen_q) begin
                          mem_en   = 1'b1;             // prefetch next code word
                          mem_addr = code_base + {{(ADDR_W-KW){1'b0}}, cd_iss};
                      end
            default:  begin
                          mem_en   = 1'b0;
                          mem_addr = {ADDR_W{1'b0}};
                      end
        endcase
    end

    // -----------------------------------------------------------------------
    // Control FSM (single synchronous, active-high reset, no latch/comb loop).
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            base_q       <= {ADDR_W{1'b0}};
            klen_q       <= {KW{1'b0}};
            nsblk_q      <= {SBW{1'b0}};
            wtype_q      <= WT_Q4K;
            d_q          <= {(16*PE_N*NSB){1'b0}};
            dmin_q       <= {(16*PE_N*NSB){1'b0}};
            scales_q     <= {(96*PE_N*NSB){1'b0}};
            q6sc_q       <= {(128*PE_N*NSB){1'b0}};
            q8d_q        <= {(16*PE_N*NB8){1'b0}};
            hd_iss       <= {NSW{1'b0}};
            hd_cap       <= {NSW{1'b0}};
            pj_iss       <= {SBW{1'b0}};
            sb_iss       <= {SBW{1'b0}};
            rd_v         <= 1'b0;
            rd_slot      <= {NSW{1'b0}};
            cd_iss       <= {KW{1'b0}};
            beat_cnt     <= {KW{1'b0}};
            code_pending <= 1'b0;
        end else begin
            // ---- per-cycle defaults (1-cycle pulses / capture marker) ----
            done <= 1'b0;
            rd_v <= 1'b0;

            // ---- latency-1 header CAPTURE: a read requested last cycle returns now ----
            //   distribute the packed header word into the three weight buses at its
            //   (pj*NSB+sb) bus slot.  d[15:0], dmin[31:16], scales[127:32].
            if (rd_v) begin
                // type-parametric header-word decode into the per-type bus at its
                // (pj*NSB+sb) slot.  Q4_K is the DEFAULT branch (2'b00 + undriven
                // x/z) -> byte-identical to the proven path.
                case (wtype_q)
                    // Q6_K word: {d[15:0], sc[16xint8 = 128b]} -> shared d bus + q6 bus
                    WT_Q6K: begin
                        d_q   [16*rd_slot  +: 16 ] <= rd_data[15:0];
                        q6sc_q[128*rd_slot +: 128] <= rd_data[143:16];
                    end
                    // Q8_0 word: 8 fp16 d (one per 32-block co-packed in the super-block)
                    WT_Q80: begin
                        q8d_q [128*rd_slot +: 128] <= rd_data[127:0];
                    end
                    WT_F16: ;   // F16 has no header region (ns==0 -> rd_v never set here)
                    // Q4_K: {d, dmin, scales} -- UNCHANGED
                    default: begin
                        d_q     [16*rd_slot +: 16] <= rd_data[15:0];
                        dmin_q  [16*rd_slot +: 16] <= rd_data[31:16];
                        scales_q[96*rd_slot +: 96] <= rd_data[127:32];
                    end
                endcase
                hd_cap <= hd_cap + 1'b1;
            end

            case (state)
                // ---- idle: wait for a descriptor ----
                S_IDLE: begin
                    if (load) begin
                        base_q       <= desc_base;
                        klen_q       <= desc_klen;
                        nsblk_q      <= desc_nsblk;
                        wtype_q      <= desc_wtype;              // latch tile type
                        d_q          <= {(16*PE_N*NSB){1'b0}};  // zero-fill unused banks
                        dmin_q       <= {(16*PE_N*NSB){1'b0}};
                        scales_q     <= {(96*PE_N*NSB){1'b0}};
                        q6sc_q       <= {(128*PE_N*NSB){1'b0}};
                        q8d_q        <= {(16*PE_N*NB8){1'b0}};
                        hd_iss       <= {NSW{1'b0}};
                        hd_cap       <= {NSW{1'b0}};
                        pj_iss       <= {SBW{1'b0}};
                        sb_iss       <= {SBW{1'b0}};
                        cd_iss       <= {KW{1'b0}};
                        beat_cnt     <= {KW{1'b0}};
                        code_pending <= 1'b0;
                        busy         <= 1'b1;
                        state        <= S_SCALE;
                    end
                end

                // ---- read all per-(col,super-block) headers into the buses (latency-1) ----
                S_SCALE: begin
                    if (hd_iss < ns) begin           // a request is on the bus now
                        rd_v    <= 1'b1;             // its data returns next cycle
                        rd_slot <= cur_slot;         // -> bus slot (pj*NSB+sb)
                        hd_iss  <= hd_iss + 1'b1;
                        // advance (pj,sb) in col-outer / super-block-inner order
                        if (sb_iss == nsblk_q - 1'b1) begin
                            sb_iss <= {SBW{1'b0}};
                            pj_iss <= pj_iss + 1'b1;
                        end else begin
                            sb_iss <= sb_iss + 1'b1;
                        end
                    end
                    // all headers captured (ns >= PE_N always for Q4_K)
                    if (hd_cap == ns)
                        state <= S_START;
                end

                // ---- start pulse (latch params, clear banks); prefetch k=0 ----
                S_START: begin
                    code_pending <= 1'b1;            // mem_data next cycle = w_q[0]
                    cd_iss       <= {{(KW-1){1'b0}}, 1'b1};
                    state        <= S_STREAM;
                end

                // ---- stream Q4_K code rows: drive in_valid+w_q, prefetch next k ----
                S_STREAM: begin
                    if (mm_in_valid)
                        beat_cnt <= beat_cnt + 1'b1;

                    if (cd_iss < klen_q) begin
                        code_pending <= 1'b1;        // another row in flight
                        cd_iss       <= cd_iss + 1'b1;
                    end else begin
                        code_pending <= 1'b0;        // no more rows in flight
                    end

                    // last beat just driven -> finish next cycle.
                    if (mm_in_valid && (beat_cnt == klen_q - 1'b1))
                        state <= S_DONE;
                end

                // ---- completion ----
                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
