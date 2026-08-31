// weight_loader_q4k_noecc_ref.v -- REFERENCE for `make weight-ecc-equiv`, NOT built into any design.
//   = the pre-ECC loader (git 9907504) + the pj_iss column-width fix, with the WEIGHT_ECC
//     ports/params/generate PHYSICALLY ABSENT.  weight-ecc-equiv proves the shipped loader at
//     WEIGHT_ECC=0 is byte-identical to THIS (i.e. turning ECC on/off changes no cells).
//   KEEP IN SYNC: any change to the loader's non-ECC core must be mirrored here, or the gate
//     fails by design.  Generated once; do not add ECC logic here.
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
    // ---- derived geometry (mirror glm_matmul_q4k) ----
    localparam integer NSB = (KMAX + 255) / 256,       // #super-blocks (== w_* banks per col)
    localparam integer NB8 = (KMAX + 31)  / 32,        // #Q8_0 32-weight blocks (== 8*NSB)
    localparam integer KW  = $clog2(KMAX + 1),         // k_len width
    localparam integer SBW = $clog2(NSB + 1),          // super-block-count width
    localparam integer PJW = $clog2(PE_N + 1),         // COLUMN-index width (0..PE_N) -- see weight-ecc-equiv
    localparam integer NSW = $clog2(NSB*PE_N + 1)      // header-word counter width
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    // ---- tile descriptor command ----
    input  wire                       load,       // 1-cycle pulse: start a tile load
    input  wire [ADDR_W-1:0]          desc_base,  // tile base address in weight mem
    input  wire [KW-1:0]              desc_klen,  // K length (#weight beats)
    input  wire [SBW-1:0]             desc_nsblk, // #super-blocks for this tile (<= NSB)
    input  wire [2:0]                 desc_wtype, // 0=Q4_K 1=Q6_K 2=Q8_0 3=F16 (undriven->Q4_K)

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
    output reg                        done        // 1-cycle pulse when tile streamed
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
    reg [PJW-1:0]            pj_iss;     // column of the read currently being issued
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
    wire [ADDR_W-1:0]        ns_ext    = {{(ADDR_W-NSW){1'b0}}, ns};
    wire [ADDR_W-1:0]        code_base = base_q + ns_ext;
    // bus slot (col-outer, super-block-inner, compile-time NSB stride) of the
    // read currently being issued.
    wire [NSW-1:0]           cur_slot  = NSW'(pj_iss) * NSW'(NSB) + NSW'(sb_iss);

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
    assign mm_w_q       = mm_in_valid ? mem_data[4*PE_N-1:0] : {(4*PE_N){1'b0}};
    // ---- ADDED mixed-type drives (Q4_K reads none of these -> byte-identical) ----
    //   The tile type is broadcast to all PE_N columns (real tensors are uniform).
    //   mm_w_hp carries the per-column 16-bit code lane (Q6_K low 6 / Q8_0 low 8 /
    //   F16 all 16); mm_w_q above still carries the Q4_K 4-bit codes unchanged.
    assign mm_w_type    = {PE_N{wtype_q}};
    assign mm_w_hp      = mm_in_valid ? mem_data[16*PE_N-1:0] : {(16*PE_N){1'b0}};
    assign mm_w_q6_sc   = q6sc_q;
    assign mm_w_q8_d    = q8d_q;

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
                        d_q   [16*rd_slot  +: 16 ] <= mem_data[15:0];
                        q6sc_q[128*rd_slot +: 128] <= mem_data[143:16];
                    end
                    // Q8_0 word: 8 fp16 d (one per 32-block co-packed in the super-block)
                    WT_Q80: begin
                        q8d_q [128*rd_slot +: 128] <= mem_data[127:0];
                    end
                    WT_F16: ;   // F16 has no header region (ns==0 -> rd_v never set here)
                    // Q4_K: {d, dmin, scales} -- UNCHANGED
                    default: begin
                        d_q     [16*rd_slot +: 16] <= mem_data[15:0];
                        dmin_q  [16*rd_slot +: 16] <= mem_data[31:16];
                        scales_q[96*rd_slot +: 96] <= mem_data[127:32];
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
