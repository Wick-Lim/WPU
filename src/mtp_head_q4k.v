`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// mtp_head_q4k.v  --  GLM-5.2 (Q4_K) Multi-Token Prediction head (the Q4_K-native
//                     sibling of mtp_head.v; prior FP8 sibling mtp_head_fp8 preserved
//                     on tag 'fp8-verified-baseline').  num_nextn_predict_layers=1.
//----------------------------------------------------------------------------
// FUNCTION  (identical math + FSM + flow to mtp_head.v; the ONLY change is that
//   the big WEIGHT matmuls run in GGML Q4_K numerics -- the ggml-Q4_K reference
//   tools/q4k_ref.py -- so the Q4_K-typed weights run with NO re-quantization)
//
//     a    = RMSNorm(h_t)                          // bf16 (modules_to_not_convert)
//     b    = RMSNorm(embed(tok_{t+1}))             // bf16
//     cat  = [ a ; b ]                              // concat -> 2*MODEL_DIM (bf16)
//     h'   = W_proj @ cat                           // *** Q4_K *** glm_matmul_q4k
//     y    = decoder_block_q4k( h' , pos , kv )     // *** Q4_K *** one GLM-5.2 layer
//     xN   = RMSNorm_final( y )                     // bf16 shared final norm
//     logits[V] = W_lm[V,MODEL_DIM] . xN            // *** bf16 *** glm_matmul_pipe
//     argmax    = arg max_v logits[v]               // speculative next-next token
//
//   QUANT SPLIT (modules_to_not_convert preserved -- a real GLM-5.2 boundary, the
//   same one glm_decoder_block_q4k.v draws, not FP8-specific):
//     * Q4_K (GGML Q4_K: 4-bit codes + per-256-elem-super-block fp16 d/dmin scales;
//       activations stay bf16 -- NO activation quant, unlike the prior FP8 track):
//         - the W_proj combine projection  (glm_matmul_q4k)
//         - everything inside the decoder layer's big linears (glm_decoder_block_q4k)
//     * bf16 (UNCHANGED):
//         - the THREE RMSNorms (a, b, final)        -- rmsnorm_unit (bf16)
//         - the residual stream + softmax/rope tails inside the decoder block
//         - the LM head GEMV                          -- glm_matmul_pipe (bf16).
//           The GLM-5.2 LM head ("lm_head") is in modules_to_not_convert and
//           stays bf16, so it routes through the bf16 matmul exactly as mtp_head.v.
//
//   PURE ORCHESTRATOR -- REIMPLEMENTS NO ARITHMETIC.  Same serial-reuse discipline
//   as mtp_head.v:
//     * ONE rmsnorm_unit reused for the 3 norms (cn_which selects the gamma).
//     * ONE glm_matmul_q4k(PE_M=1,PE_N=PROJ_TN,KMAX=2*MODEL_DIM) as the Q4_K
//       combine projection, walked over MODEL_DIM/PROJ_TN output tiles.  The
//       activation is the bf16 concat `cat`, fed DIRECT (Q4_K quantizes ONLY the
//       weights -- NO activation quant, the SAME bf16-direct discipline
//       swiglu_expert_q4k / mla_attn_q4k use).
//     * ONE glm_decoder_block_q4k (one GLM-5.2 decoder layer) run ONCE on h'.
//     * ONE glm_matmul_pipe(PE_M=1,PE_N=LM_TN,KMAX=MODEL_DIM) bf16 LM-head GEMV.
//     * a 1-elt/cycle argmax scan -- identical to mtp_head.v's tail.
//
//----------------------------------------------------------------------------
// ACTIVATIONS for the W_proj projection (bf16, fed DIRECT)
//   Q4_K quantizes ONLY the weights, so unlike the prior FP8 track there is NO
//   activation quant: the bf16 concat `cat` streams straight into glm_matmul_q4k
//   (a_col), which multiplies it against the fp32-dequantized Q4_K weights.  The
//   prior FP8 sibling's per-vector pow2 activation shift (csh / a_shift, the
//   134-emax E4M3 headroom trick) is GONE.
//
//----------------------------------------------------------------------------
// WEIGHT PULL INTERFACE  (vs mtp_head.v: the Q4_K matmuls now pull Q4_K 4-bit codes
//   + per-super-block scales; the bf16 LM head + the three gammas are UNCHANGED)
//   * combine/final RMSNorm gamma (cn_*)   : bf16, UNCHANGED.
//   * decoder RMSNorm gamma (gn_*)         : bf16, UNCHANGED.
//   * W_proj weight pull (pw_*)            : pw_q = PROJ_TN Q4_K 4-bit lanes (was
//                                            bf16); pw_d/pw_dmin/pw_scales = per-
//                                            super-block scales for the pw_ptile tile.
//   * decoder attention weight pull (aw_*) : aw_q Q4_K codes + aw_d/aw_dmin/aw_scales.
//   * decoder cache read (kc_*)            : bf16 latent KV, UNCHANGED.
//   * decoder router weight pull (rw_*)    : rw_q Q4_K codes + rw_d/rw_dmin/rw_scales.
//   * decoder FFN expert weight pull (fw_*): fw_q/fw_q_up Q4_K + fw_{d,dmin,scales}_{g,u}.
//   * shared LM-head weight pull (lw_*)    : lw_col bf16, UNCHANGED (bf16 LM head).
//   Q4_K codes are pulled per K-beat (qualified by the *_req + index); block scales
//   are answered from the current tile/expert selector (latched at the matmul's
//   start cycle, exactly as glm_decoder_block_q4k / swiglu_expert_q4k expect).
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic; handshake-driven so each Q4_K leaf unit's own latency
//   is absorbed via its busy/out_valid -- structurally identical to mtp_head.v,
//   only the per-tile matmul latency term changes from the bf16 to the Q4_K pipe):
//   L_mtp = 2*L_rmsnorm(MODEL_DIM)
//         + (MODEL_DIM/PROJ_TN) * L_matmul_q4k(K=2*MODEL_DIM, PE_N=PROJ_TN)
//         + L_decoder_block_q4k(params, S)
//         + L_rmsnorm(MODEL_DIM)
//         + (VOCAB/LM_TN) * L_matmul_pipe(K=MODEL_DIM, PE_N=LM_TN)   // bf16 LM head
//         + VOCAB                                                    // argmax scan
//   No data-dependent stall; sync active-high reset; no latch; no comb loop (all
//   feedback rides the rmsnorm / decoder_block / matmul pipeline registers).
//============================================================================
module mtp_head_q4k #(
    // ---- model / slice config (small-but-faithful, ACCEL_GLM52 §8.1) ----
    parameter integer MODEL_DIM  = 128,
    parameter integer VOCAB      = 256,
    // ---- decoder_block slice params (passed straight through) ----
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    // ---- PE_M : token ROWS (batch B) sharing ONE weight fetch (ULTRA_PERF#2) ----
    //   PE_M (default 1 == byte-identical to the committed single-token head) is the
    //   number of MTP token-rows carried through the head at once.  The B rows share
    //   the SAME weight fetch (W_proj, the decoder-block weights, and the LM head),
    //   pos, s_len and KV -- one Flash fetch feeds all B rows.  Each row carries its
    //   own bf16 tail (the 3 RMSNorms, the dynamic-quant a_shift, the argmax) so row r
    //   is BIT-IDENTICAL to a PE_M=1 run on row r's own (h_t[r], emb_t1[r]).
    parameter integer PE_M       = 1,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,// 2.5 fp32
    parameter integer TN         = 4,
    // ---- vestigial FP8 block size (weight_block_size=[128,128] from DeepSeek-V3 /
    //      the prior GLM-5.2-FP8 track); Q4_K uses 256-elem super-blocks -- kept as
    //      a live param threaded to glm_decoder_block_q4k ----
    parameter integer BLK        = 128,
    // ---- GEMV tile widths.  VOCAB % LM_TN == 0 ; MODEL_DIM % PROJ_TN == 0. ----
    parameter integer LM_TN      = 4,           // LM-head VOCAB cols/pass
    parameter integer PROJ_TN    = 4,           // combine-proj output cols/pass
    // ====================================================================
    // derived (do NOT override) -- mirror decoder_block_q4k's port-width derivations
    // ====================================================================
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
    // ---- vestigial FP8 [128,128]-block scale counts (from the prior FP8 track;
    //      superseded by the Q4_K 256-elem super-block counts below) ----
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,  // attention scales
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,  // dense FFN scales
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK,  // router scales
    // ---- Q4_K super-block counts (#256-wide K super-blocks per weight family) ----
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,  // attention super-blocks
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,  // dense FFN super-blocks
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,  // router super-blocks
    // head-level derived
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer LMKW       = $clog2(MODEL_DIM + 1),    // LM matmul k_len width
    // combine-projection derived
    parameter integer CK         = 2 * MODEL_DIM,            // concat length (K)
    parameter integer CKIW       = $clog2(CK),               // concat index width
    parameter integer PKW        = $clog2(CK + 1),           // proj matmul k_len width
    parameter integer NPTILE     = MODEL_DIM / PROJ_TN,      // proj output tiles
    parameter integer PTW        = (NPTILE <= 1) ? 1 : $clog2(NPTILE),
    parameter integer PROJ_NB    = (CK + BLK - 1) / BLK,     // proj vestigial FP8 K-block count
    parameter integer PROJ_NSB   = (CK + 255) / 256          // proj Q4_K super-blocks
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- control ----
    input  wire                          start,      // 1-cycle pulse: begin
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: logits valid
    input  wire                          mode,       // 0=DENSE FFN, 1=MoE FFN (block)
    input  wire [POSW-1:0]               pos,        // query position t (RoPE)
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)

    // ---- data in (bf16, PE_M rows row-major: row r elt i @ [16*(MODEL_DIM*r+i)+:16]) ----
    input  wire [MODEL_DIM*16*PE_M-1:0]  h_t,        // main-model hidden state @ t
    input  wire [MODEL_DIM*16*PE_M-1:0]  emb_t1,     // embedding of predicted tok t+1

    // ---- outputs (PE_M rows: logits row r @ [VOCAB*16*r+:VOCAB*16], argmax @ [TOKW*r+:TOKW]) ----
    output reg  [VOCAB*16*PE_M-1:0]      logits,     // VOCAB bf16 t+2 logits (per row)
    output reg  [TOKW*PE_M-1:0]          argmax,     // arg max logit (spec. t+2 token, per row)

    // ---- DeepSeek-MTP chain hidden state (ADDITIVE; P1.3b) ----
    //   h_mtp packs the decoder-block output xcur (h_mtp[16*i +: 16] = xcur[i]),
    //   i.e. the pre-final-norm layer output that seeds the NEXT MTP chain step
    //   (a = RMSNorm(h_mtp) for predict-layer k+1).  Latched in S_DBW alongside
    //   the xcur write and held stable through the LM-head + argmax phases, so it
    //   is valid at `done`.  Purely additive: existing callers leave it
    //   unconnected and are byte-identical (named-port instantiation).
    output reg  [MODEL_DIM*16*PE_M-1:0]  h_mtp,

    // ---- combine/final RMSNorm gamma pull (cn_which: 0=h_t,1=emb,2=final) ----
    output wire                          cn_req,
    output wire [1:0]                    cn_which,
    output wire [DIMW-1:0]               cn_idx,
    input  wire [15:0]                   cn_val,

    // ---- Q4_K combine-projection weight pull ----
    //   pw_q[t]     = Q4_K 4-bit code W_proj[ptile*PROJ_TN+t][pw_k]
    //   pw_d/pw_dmin/pw_scales = per-(col,super-block) Q4_K dequant params for the
    //                 pw_ptile output tile.
    output wire                          pw_req,
    output wire [PTW-1:0]                pw_ptile,   // which MODEL_DIM output tile
    output wire [CKIW-1:0]               pw_k,       // concat reduction index 0..2*MD-1
    input  wire [PROJ_TN*4-1:0]          pw_q,       // PROJ_TN Q4_K 4-bit weight lanes
    input  wire [16*PROJ_TN*PROJ_NSB-1:0] pw_d,      // fp16 d per (col,super-block)
    input  wire [16*PROJ_TN*PROJ_NSB-1:0] pw_dmin,   // fp16 dmin
    input  wire [96*PROJ_TN*PROJ_NSB-1:0] pw_scales, // 6-bit scales

    // ---- decoder_block RMSNorm gamma pull (pre-attn/pre-FFN) ----
    output wire                          gn_req,
    output wire                          gn_which,   // 0=pre-attn, 1=pre-FFN
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,

    // ---- decoder_block attention weight pull (Q4_K codes + block scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*4-1:0]             aw_q,       // PE_N Q4_K 4-bit weight lanes
    input  wire [16*PE_N*A_NSB-1:0]      aw_d,       // fp16 d per (col,super-block)
    input  wire [16*PE_N*A_NSB-1:0]      aw_dmin,    // fp16 dmin
    input  wire [96*PE_N*A_NSB-1:0]      aw_scales,  // 6-bit scales

    // ---- decoder_block attention KV-cache read ----
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    input  wire                          kc_valid,

    // ---- decoder_block MoE router weight pull (W_g column; Q4_K codes + scales) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [4*N_EXPERT-1:0]         rw_q,       // N_EXPERT Q4_K 4-bit = W_g[k,*]
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_d,       // fp16 d
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_dmin,    // fp16 dmin
    input  wire [96*N_EXPERT*R_NSB-1:0]  rw_scales,  // 6-bit scales

    // ---- decoder_block FFN expert weight pull (qualified; Q4_K codes + scales) ----
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [4*TN-1:0]               fw_q,       // GATE/DOWN Q4_K 4-bit lanes
    input  wire [4*TN-1:0]               fw_q_up,    // UP companion Q4_K 4-bit lanes
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_g,     // GATE/DOWN fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_g,  // GATE/DOWN fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_g,// GATE/DOWN 6-bit scales
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_u,     // UP fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_u,  // UP fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_u,// UP 6-bit scales

    // ---- shared LM-head weight pull (bf16, UNCHANGED): lw_col[t]=W_lm[vtile*LM_TN+t][lw_k]
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col
);
    `include "glm_fp.vh"

    integer ii;
    integer rr;        // PE_M row loop variable

    //========================================================================
    // latched inputs + working buffers (bf16) -- PER ROW ([0:PE_M-1] leading dim)
    //========================================================================
    reg [15:0] hbuf [0:PE_M-1][0:MODEL_DIM-1]; // latched h_t  (RMSNorm source, phase 0)
    reg [15:0] ebuf [0:PE_M-1][0:MODEL_DIM-1]; // latched emb  (RMSNorm source, phase 1)
    reg [15:0] cbuf [0:PE_M-1][0:CK-1];        // concat [a;b] (= RMSNorm outputs) (bf16 proj act)
    reg [15:0] hprime [0:PE_M-1][0:MODEL_DIM-1];// h' = W_proj @ cat  (decoder block input)
    reg [15:0] xcur   [0:PE_M-1][0:MODEL_DIM-1];// decoder-block output y (final-norm source)
    reg [15:0] xn     [0:PE_M-1][0:MODEL_DIM-1];// final-normed (LM-head input)
    reg [15:0] lbuf   [0:PE_M-1][0:VOCAB-1];   // LM-head logits scratch (bf16)

    reg            mode_q;
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   slen_q;

    // packed view of h' for the decoder block's wide x_vec port (PE_M rows row-major)
    reg [MODEL_DIM*16*PE_M-1:0] hp_vec;
    always @* begin
        for (rr = 0; rr < PE_M; rr = rr + 1)
            for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
                hp_vec[16*(MODEL_DIM*rr + ii) +: 16] = hprime[rr][ii];
    end

    //========================================================================
    // rmsnorm_unit (LEN=MODEL_DIM, LANES=1) reused for the 3 norms -- PE_M
    //   REPLICATED, lockstep off ONE shared gamma pull (gamma is the SAME for every
    //   row -> one cn_* pull answers all PE_M units).  bf16, UNCHANGED numerics
    //   (modules_to_not_convert).  Each unit PULLS x (reduce pass) from the
    //   phase-selected per-row source, then gamma (normalize pass) via cn_*
    //   (cn_which = phase tells the system which gamma).  At PE_M=1 this collapses
    //   to the committed single u_norm exactly.
    //========================================================================
    reg              cn_start;
    reg  [1:0]       cn_phase;          // 0=norm(h_t),1=norm(emb),2=final norm
    wire [PE_M-1:0]  cn_in_req, cn_g_req, cn_y_valid, cn_busy, cn_done;
    wire [16*PE_M-1:0] cn_y_out;
    reg  [16*PE_M-1:0] cn_x_in;
    reg              cn_x_valid;
    reg  [16*PE_M-1:0] cn_gamma_in;
    reg              cn_g_valid;
    genvar gcn;
    generate
    for (gcn = 0; gcn < PE_M; gcn = gcn + 1) begin : CN
        rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_norm (
            .clk(clk), .rst(rst), .start(cn_start),
            .in_req(cn_in_req[gcn]), .x_in(cn_x_in[16*gcn +: 16]), .x_valid(cn_x_valid),
            .g_req(cn_g_req[gcn]), .gamma_in(cn_gamma_in[16*gcn +: 16]), .g_valid(cn_g_valid),
            .y_valid(cn_y_valid[gcn]), .y_out(cn_y_out[16*gcn +: 16]),
            .busy(cn_busy[gcn]), .done(cn_done[gcn])
        );
    end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */
    wire _cn_busy_unused = &{1'b0, cn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // norm beat counters (LANES=1 -> beat == element index).  Reset at cn_start.
    // All PE_M units run lockstep -> instance-0's handshake drives the counters.
    reg [DIMW:0] cn_ridx;   // reduce read index (x pull)
    reg [DIMW:0] cn_widx;   // normalize write index (y store)
    reg [DIMW:0] cn_gidx;   // gamma pull index

    // gamma pull is COMBINATIONAL (answered same cycle), registered 1 cycle.
    assign cn_req   = cn_g_req[0];
    assign cn_which = cn_phase;
    assign cn_idx   = cn_gidx[DIMW-1:0];

    //========================================================================
    // ONE glm_decoder_block_q4k (one GLM-5.2 decoder layer) run ONCE on h'.
    //   All Q4_K weight / cache pulls forwarded straight out.
    //========================================================================
    reg                       db_start;
    wire                      db_busy, db_done;
    wire [MODEL_DIM*16*PE_M-1:0] db_y;
    glm_decoder_block_q4k #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK),
        .PE_M(PE_M)
    ) u_block (
        .clk(clk), .rst(rst), .start(db_start), .busy(db_busy), .done(db_done),
        .mode(mode_q), .pos(pos_q), .s_len(slen_q),
        .x_vec(hp_vec), .y_out(db_y),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _db_busy_unused = &{1'b0, db_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // Q4_K COMBINE-PROJECTION GEMV : glm_matmul_q4k as a 1xPROJ_TN tile, K=2*MD,
    //   per-super-block scales.  A row (M=1) = cat[1,2*MODEL_DIM] (bf16, fed DIRECT
    //   -- no activation quant) ; W tile (N=PROJ_TN) = W_proj[ptile..][k] (Q4_K
    //   4-bit codes) + pw_d/pw_dmin/pw_scales.  On beat k present pp_a = cat[k] and
    //   pw_q = W col.
    //========================================================================
    reg                  pp_start;
    reg                  pp_in_valid;
    reg  [PKW-1:0]       pp_klen;
    reg  [16*PE_M-1:0]   pp_a;            // cat[k] per row (PE_M bf16 lanes, drop-in act)
    wire                 pp_busy, pp_ov;
    wire [16*PE_M*PROJ_TN-1:0] pp_c;      // PE_M x PROJ_TN result tile (bf16)
    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(PROJ_TN), .KMAX(CK)) u_proj (
        .clk(clk), .rst(rst), .start(pp_start), .k_len(pp_klen),
        .w_d(pw_d), .w_dmin(pw_dmin), .w_scales(pw_scales),
        .in_valid(pp_in_valid), .a_col(pp_a), .w_q(pw_q),
        .busy(pp_busy), .out_valid(pp_ov), .c_out(pp_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _pp_busy_unused = &{1'b0, pp_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // projection sequencing
    reg [PTW-1:0]   ptile;          // current output tile
    reg [CKIW:0]    pk;             // current K beat (0..CK)
    reg             pp_streaming;
    reg [CKIW-1:0]  pk_present;     // K index currently registered in pp_a
    reg             pp_pres_valid;  // mirrors a presented beat (weight pull qualifier)

    assign pw_req   = pp_pres_valid;
    assign pw_ptile = ptile;        // also selects pw_scale (block scales for tile)
    assign pw_k     = pk_present;

    //========================================================================
    // SHARED LM-HEAD GEMV : glm_matmul_pipe as a 1xLM_TN tile, K=MODEL_DIM.
    //   bf16 (the GLM-5.2 lm_head is in modules_to_not_convert -> stays bf16),
    //   identical to mtp_head.v's tail.
    //========================================================================
    reg                  mm_start;
    reg                  mm_in_valid;
    reg  [LMKW-1:0]      mm_klen;
    reg  [16*PE_M-1:0]   mm_a;            // xN[k] per row (PE_M lanes)
    wire                 mm_busy, mm_ov;
    wire [16*PE_M*LM_TN-1:0] mm_c;        // PE_M x LM_TN result tile (bf16)
    glm_matmul_pipe #(.PE_M(PE_M), .PE_N(LM_TN), .KMAX(MODEL_DIM)) u_lm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_klen),
        .in_valid(mm_in_valid), .a_col(mm_a), .w_row(lw_col),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _mm_busy_unused = &{1'b0, mm_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    reg [VTW-1:0]  vtile;          // current VOCAB tile
    reg [DIMW:0]   lm_k;           // current K beat (0..MODEL_DIM)
    reg            lm_streaming;
    reg [DIMW-1:0] lk_present;     // K index currently registered in mm_a
    reg            mm_pres_valid;  // mirrors a presented beat

    assign lw_req   = mm_pres_valid;
    assign lw_vtile = vtile;
    assign lw_k     = lk_present;

    //========================================================================
    // MASTER FSM  (byte-for-byte the same control as mtp_head.v)
    //========================================================================
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_NORM   = 4'd1,    // run rmsnorm pass (phase 0/1/2)
        S_PROJ   = 4'd2,    // stream K beats of combine projection (current ptile)
        S_PROJW  = 4'd3,    // wait pp_ov; store PROJ_TN h' elts; next ptile / block
        S_DBW    = 4'd5,    // db_start pulsed in S_PROJW; wait db_done; xcur<=y; final norm
        S_LMTILE = 4'd6,    // stream K beats for current vtile
        S_LMWAIT = 4'd7,    // wait mm_ov; store LM_TN logits; next vtile
        S_ARGMAX = 4'd8,    // scan lbuf for argmax (fp32 compare)
        S_DONE   = 4'd9;
    reg [3:0] state;

    reg [TOKW:0]   am_i;                  // argmax scan index (SHARED: same index all rows)
    reg [15:0]     am_best [0:PE_M-1];    // best logit value per row (bf16; magnitude compare)
    reg [TOKW-1:0] am_arg  [0:PE_M-1];    // arg max per row

    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            logits       <= {VOCAB*16*PE_M{1'b0}};
            argmax       <= {TOKW*PE_M{1'b0}};
            h_mtp        <= {MODEL_DIM*16*PE_M{1'b0}};
            mode_q       <= 1'b0;
            pos_q        <= {POSW{1'b0}};
            slen_q       <= {(IDXW+1){1'b0}};
            cn_start     <= 1'b0; cn_phase <= 2'd0;
            cn_x_in      <= {16*PE_M{1'b0}}; cn_x_valid <= 1'b0;
            cn_gamma_in  <= {16*PE_M{1'b0}}; cn_g_valid <= 1'b0;
            db_start     <= 1'b0;
            pp_start     <= 1'b0; pp_in_valid <= 1'b0;
            pp_klen      <= {PKW{1'b0}}; pp_a <= {16*PE_M{1'b0}};
            ptile        <= {PTW{1'b0}};
            pk           <= {(CKIW+1){1'b0}};
            pp_streaming <= 1'b0;
            pk_present   <= {CKIW{1'b0}};
            pp_pres_valid<= 1'b0;
            mm_start     <= 1'b0; mm_in_valid <= 1'b0;
            mm_klen      <= {LMKW{1'b0}}; mm_a <= {16*PE_M{1'b0}};
            vtile        <= {VTW{1'b0}};
            lm_k         <= {(DIMW+1){1'b0}};
            lm_streaming <= 1'b0;
            lk_present   <= {DIMW{1'b0}};
            mm_pres_valid<= 1'b0;
            am_i         <= {(TOKW+1){1'b0}};
            for (rr=0; rr<PE_M; rr=rr+1) begin
                am_best[rr] <= 16'h0;
                am_arg[rr]  <= {TOKW{1'b0}};
                for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                    hbuf[rr][ii] <= 16'h0; ebuf[rr][ii] <= 16'h0;
                    hprime[rr][ii] <= 16'h0; xcur[rr][ii] <= 16'h0; xn[rr][ii] <= 16'h0;
                end
                for (ii=0; ii<CK; ii=ii+1)    cbuf[rr][ii] <= 16'h0;
                for (ii=0; ii<VOCAB; ii=ii+1) lbuf[rr][ii] <= 16'h0;
            end
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            cn_start <= 1'b0;
            db_start <= 1'b0;
            pp_start <= 1'b0;
            mm_start <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy   <= 1'b1;
                    mode_q <= mode;
                    pos_q  <= pos;
                    slen_q <= s_len;
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                            hbuf[rr][ii] <= h_t   [16*(MODEL_DIM*rr + ii) +: 16];
                            ebuf[rr][ii] <= emb_t1[16*(MODEL_DIM*rr + ii) +: 16];
                        end
                    end
                    // launch RMSNorm(h_t) : phase 0 (reduce source = hbuf)
                    cn_phase <= 2'd0;
                    cn_start <= 1'b1;
                    state    <= S_NORM;
                end
            end
            //---------------------------------------------------------------- rmsnorm pass
            // Reduce pass: answer cn_in_req from the phase source.  Normalize pass:
            // answer cn_g_req with the combinational gamma (registered 1 cycle).
            // Store y to the phase destination: phase0 -> cbuf[0..MD-1] (=a),
            // phase1 -> cbuf[MD..2MD-1] (=b), phase2 -> xn (final-normed).
            S_NORM: begin
                cn_x_valid <= 1'b0; cn_g_valid <= 1'b0;
                if (cn_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        case (cn_phase)
                            2'd0:    cn_x_in[16*rr +: 16] <= hbuf[rr][cn_ridx[DIMW-1:0]];
                            2'd1:    cn_x_in[16*rr +: 16] <= ebuf[rr][cn_ridx[DIMW-1:0]];
                            default: cn_x_in[16*rr +: 16] <= xcur[rr][cn_ridx[DIMW-1:0]];
                        endcase
                    cn_x_valid <= 1'b1;
                end
                if (cn_g_req[0]) begin
                    // gamma pull is COMBINATIONAL (cn_*); the SAME gamma feeds every row.
                    for (rr=0; rr<PE_M; rr=rr+1)
                        cn_gamma_in[16*rr +: 16] <= cn_val;
                    cn_g_valid  <= 1'b1;
                end
                if (cn_y_valid[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        case (cn_phase)
                            2'd0:    cbuf[rr][cn_widx[CKIW-1:0]]                       <= cn_y_out[16*rr +: 16];
                            2'd1:    cbuf[rr][MODEL_DIM[CKIW-1:0] + cn_widx[CKIW-1:0]] <= cn_y_out[16*rr +: 16];
                            default: xn[rr][cn_widx[DIMW-1:0]]                         <= cn_y_out[16*rr +: 16];
                        endcase
                    end
                end
                if (cn_done[0]) begin
                    if (cn_phase == 2'd0) begin
                        // a done -> run RMSNorm(emb) : phase 1
                        cn_phase <= 2'd1;
                        cn_start <= 1'b1;
                        state    <= S_NORM;
                    end else if (cn_phase == 2'd1) begin
                        // concat ready -> launch Q4_K combine projection (tile 0)
                        ptile        <= {PTW{1'b0}};
                        pp_klen      <= CK[PKW-1:0];
                        pp_start     <= 1'b1;
                        pp_streaming <= 1'b1;
                        pk           <= {(CKIW+1){1'b0}};
                        pp_in_valid  <= 1'b0;       // first beat presented in S_PROJ
                        state        <= S_PROJ;
                    end else begin
                        // final norm done -> launch LM head (vtile 0)
                        vtile        <= {VTW{1'b0}};
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        mm_in_valid  <= 1'b0;       // first beat presented in S_LMTILE
                        state        <= S_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- proj tile stream
            // Present cat[k] as a_col and Q4_K W_proj[ptile][k] (pw_q) as w_row
            // each K beat.  pk_present latches the beat index so the weight pull
            // aligns.  pw_scale + csh (a_shift) were latched into u_proj at start.
            S_PROJ: begin
                if (pp_streaming) begin
                    if (pk < CK[CKIW:0]) begin
                        for (rr=0; rr<PE_M; rr=rr+1)
                            pp_a[16*rr +: 16] <= cbuf[rr][pk[CKIW-1:0]];
                        pk_present    <= pk[CKIW-1:0];
                        pp_in_valid   <= 1'b1;
                        pp_pres_valid <= 1'b1;
                        pk            <= pk + 1'b1;
                    end else begin
                        pp_in_valid   <= 1'b0;
                        pp_pres_valid <= 1'b0;
                        pp_streaming  <= 1'b0;
                        state         <= S_PROJW;
                    end
                end
            end
            //---------------------------------------------------------------- proj tile wait
            S_PROJW: begin
                pp_in_valid   <= 1'b0;
                pp_pres_valid <= 1'b0;
                if (pp_ov) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<PROJ_TN; ii=ii+1)
                            hprime[rr][ptile*PROJ_TN + ii] <= pp_c[16*(rr*PROJ_TN + ii) +: 16];
                    if (ptile == (NPTILE[PTW-1:0]-1'b1)) begin
                        // all output tiles done -> run the Q4_K decoder block on h'
                        // (db_start pulses here; go straight to the wait state)
                        db_start <= 1'b1;
                        state    <= S_DBW;
                    end else begin
                        ptile        <= ptile + 1'b1;
                        pp_klen      <= CK[PKW-1:0];
                        pp_start     <= 1'b1;
                        pp_streaming <= 1'b1;
                        pk           <= {(CKIW+1){1'b0}};
                        state        <= S_PROJ;
                    end
                end
            end
            //---------------------------------------------------------------- decoder block
            S_DBW: begin
                if (db_done) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                            xcur[rr][ii] <= db_y[16*(MODEL_DIM*rr + ii) +: 16];
                            // latch the packed per-row chain hidden state alongside xcur
                            // (h_mtp row r elt i = xcur[r][i]); held stable to `done`.
                            h_mtp[16*(MODEL_DIM*rr + ii) +: 16] <= db_y[16*(MODEL_DIM*rr + ii) +: 16];
                        end
                    // launch final rmsnorm over xcur : phase 2.  (xcur is updated
                    // this edge; the reduce pass starts next cycle so it is in place.)
                    cn_phase <= 2'd2;
                    cn_start <= 1'b1;
                    state    <= S_NORM;
                end
            end
            //---------------------------------------------------------------- LM head tile stream
            S_LMTILE: begin
                if (lm_streaming) begin
                    if (lm_k < MODEL_DIM[DIMW:0]) begin
                        for (rr=0; rr<PE_M; rr=rr+1)
                            mm_a[16*rr +: 16] <= xn[rr][lm_k[DIMW-1:0]];
                        lk_present    <= lm_k[DIMW-1:0];
                        mm_in_valid   <= 1'b1;
                        mm_pres_valid <= 1'b1;
                        lm_k          <= lm_k + 1'b1;
                    end else begin
                        mm_in_valid   <= 1'b0;
                        mm_pres_valid <= 1'b0;
                        lm_streaming  <= 1'b0;
                        state         <= S_LMWAIT;
                    end
                end
            end
            //---------------------------------------------------------------- LM head tile wait
            S_LMWAIT: begin
                mm_in_valid   <= 1'b0;
                mm_pres_valid <= 1'b0;
                if (mm_ov) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<LM_TN; ii=ii+1)
                            lbuf[rr][vtile*LM_TN + ii] <= mm_c[16*(rr*LM_TN + ii) +: 16];
                    if (vtile == (NVTILE[VTW-1:0]-1'b1)) begin
                        am_i <= {(TOKW+1){1'b0}};
                        for (rr=0; rr<PE_M; rr=rr+1) begin
                            am_best[rr] <= 16'hFF80;    // -inf (bf16)
                            am_arg[rr]  <= {TOKW{1'b0}};
                        end
                        state   <= S_ARGMAX;
                    end else begin
                        vtile        <= vtile + 1'b1;
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        state        <= S_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- argmax
            S_ARGMAX: begin
                if (am_i < VOCAB[TOKW:0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        if (bf16_gt(lbuf[rr][am_i[TOKW-1:0]], am_best[rr])) begin
                            am_best[rr] <= lbuf[rr][am_i[TOKW-1:0]];
                            am_arg[rr]  <= am_i[TOKW-1:0];
                        end
                    am_i <= am_i + 1'b1;
                end else begin
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        for (ii=0; ii<VOCAB; ii=ii+1)
                            logits[16*(VOCAB*rr + ii) +: 16] <= lbuf[rr][ii];
                        argmax[TOKW*rr +: TOKW] <= am_arg[rr];
                    end
                    state  <= S_DONE;
                end
            end
            //----------------------------------------------------------------
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

    //========================================================================
    // rmsnorm pull beat counters (mirror the unit's beat order; LANES=1 so
    // beat == element index).  Reset at each cn_start.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            cn_ridx <= {(DIMW+1){1'b0}};
            cn_widx <= {(DIMW+1){1'b0}};
            cn_gidx <= {(DIMW+1){1'b0}};
        end else begin
            if (cn_start) begin
                cn_ridx <= {(DIMW+1){1'b0}};
                cn_widx <= {(DIMW+1){1'b0}};
                cn_gidx <= {(DIMW+1){1'b0}};
            end else begin
                if (cn_in_req[0])  cn_ridx <= cn_ridx + 1'b1;
                if (cn_y_valid[0]) cn_widx <= cn_widx + 1'b1;
                if (cn_g_req[0])   cn_gidx <= cn_gidx + 1'b1;
            end
        end
    end

    //========================================================================
    // bf16 greater-than (strict) on a direct sign + 15-bit |.| magnitude compare.
    // Treats -0 == +0; ignores nan (finite logits).  Used for the argmax ONLY.
    // bf16->fp32 is a pure low-zero-extend, so this ordering is byte-for-byte
    // identical to the previous fp32 compare -- the comparator is just 31b->15b
    // and am_best 32b->16b.
    //========================================================================
    function automatic bf16_gt(input [15:0] a, input [15:0] b);
        reg sa, sb;
        reg [14:0] ma, mb;
        begin
            sa = a[15]; sb = b[15];
            ma = a[14:0]; mb = b[14:0];
            if (sa != sb) begin
                if ((ma == 15'b0) && (mb == 15'b0)) bf16_gt = 1'b0; // +0 vs -0
                else bf16_gt = (sb == 1'b1);
            end else if (sa == 1'b0) begin
                bf16_gt = (ma > mb);
            end else begin
                bf16_gt = (ma < mb);
            end
        end
    endfunction

endmodule
/* verilator lint_on DECLFILENAME */
