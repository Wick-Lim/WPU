`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_decoder_block_q4k.v  --  ONE GLM-5.2 (Q4_K) decoder layer (ACCEL_GLM52 §2,§6)
//----------------------------------------------------------------------------
// FUNCTION  (the Q4_K-native sibling of glm_decoder_block.v; prior FP8 sibling
//   glm_decoder_block_fp8 preserved on tag 'fp8-verified-baseline' -- IDENTICAL FSM, dataflow,
//   dense/MoE modes, streamed residual adds and FFN combine; the ONLY change is
//   that the big LINEAR WEIGHT matmuls run Q4_K instead of bf16)
//
//     h = x + mla_attn_q4k( rmsnorm(x), pos, kv_cache )      // attention sub-block
//     y = h + FFN(          rmsnorm(h) )                      // FFN sub-block
//
//   FFN is selected by the MODE input:
//     MODE==0 (DENSE, first_k_dense_replace layers): ONE swiglu_expert_q4k at
//             INTER_DENSE, no router.    FFN(z) = swiglu(z).
//     MODE==1 (MoE):  moe_router_q4k(z) picks TOPK of N_EXPERT experts; each routed
//             expert runs swiglu_expert_q4k(z, expert_weights) scaled by its routed
//             gate; the ALWAYS-ON shared expert runs swiglu_expert_q4k(z, shared)
//             with weight 1; combine = Σ_e gate_e * y_e + y_shared (fp32 accum).
//
//   QUANT SPLIT (modules_to_not_convert preserved -- a real GLM-5.2 boundary, not
//   FP8-specific):  the bf16 residual STREAM, the two pre-RMSNorms,
//   softmax/rope/sigmoid/topk tails stay bf16 EXACTLY as glm_decoder_block.v.
//   ONLY the leaf units' big weight GEMMs are Q4_K (GGML Q4_K: 4-bit codes +
//   per-256-elem-super-block fp16 d/dmin scales; activations stay bf16 -- NO
//   activation quant, unlike the prior FP8 track).  The residual adds are bf16
//   (glm_fp.vh, §6).
//
//============================================================================
// PE_M BATCHING (B residual-hidden ROWS share ONE weight fetch)    (ULTRA_PERF#2)
//----------------------------------------------------------------------------
//   PE_M (default 1 == byte-identical to the committed single-token layer) is the
//   number of token ROWS (the residual hidden) carried through the layer at once.
//   The B rows share the SAME decode step: pos, s_len, the KV cache, the per-layer
//   gammas and ALL weight matrices.  The three Q4_K leaf wrappers
//   (mla_attn_q4k / moe_router_q4k / swiglu_expert_q4k) are ALREADY PE_M-capable:
//   each streams PE_M activation rows against ONE shared weight fetch and emits
//   PE_M result rows, so the weight pull streams (aw_*/rw_*/fw_*) are IDENTICAL to
//   PE_M=1 -- ONE Flash fetch feeds all B rows.  This module just carries a B-WIDE
//   residual stream and threads the per-row bf16 tail (RMSNorm, residual adds, the
//   MoE combine) over it.
//
//   PER-ROW (replicated B-wide, lockstep):
//     * the residual buffers xbuf/nrm/hbuf/fbuf and the fp32 combine accumulator
//       facc all carry a leading [0:PE_M-1] row dim;
//     * the two pre-RMSNorms are PE_M replicated rmsnorm_units in lockstep off ONE
//       shared gamma pull (gamma is the SAME for every row);
//     * the two residual adds and the MoE finalize stream PE_M elements/cycle (one
//       fp32 adder + one fp32->bf16 narrow PER ROW), same element index for all
//       rows -> the cycle structure is UNCHANGED.
//
//   PER-ROW MoE ROUTING (the one place rows genuinely diverge):
//     moe_router_q4k emits per-row {sel_idx, sel_weight}; rows may pick DIFFERENT
//     experts.  Because ONE expert's weights are fetched per evaluation (shared by
//     all rows), the combine iterates the expert evaluations and ACCUMULATES into
//     a row ONLY when that row selected the current expert (row_active), with that
//     row's routed gate.  At PE_M=1 the loop iterates exactly the row's TOPK
//     selected experts (NEVAL=TOPK) -> the committed TOPK+1 datapath, byte for
//     byte.  At PE_M>1 it iterates all N_EXPERT routed experts (NEVAL=N_EXPERT),
//     each a single shared fetch, every row accumulating only its own selected
//     experts in expert-index order -> row r's combine is the SAME set of routed
//     terms as a PE_M=1 run on row r (bit-identical for TOPK<=2 by fp32-add
//     commutativity, which is the GLM-5.2 slice config), then the shared expert.
//
//   At PE_M=1 every PE_M-indexed construct constant-folds to the committed
//   single-row datapath (identical ports), so the committed TBs instantiate this
//   unchanged.
//----------------------------------------------------------------------------
// EXPERT REUSE STRATEGY (MoE mode -- SERIAL, one swiglu_expert_q4k instance)
//   ONE swiglu_expert_q4k(INTER_MOE) instance is time-multiplexed over the routed
//   experts followed by the 1 shared expert (NEVAL+1 evaluations).  Before each
//   evaluation the orchestrator drives the FFN weight-pull select (fw_eidx /
//   fw_shared) ALONGSIDE the expert's own (w_sel/w_grp/w_k + block-scale) request,
//   so the system answers the correct expert's Q4_K codes + super-block scales that
//   cycle.  After each routed expert's y the orchestrator scales it by each row's
//   routed gate (fp32 mul) and adds into that row's per-element fp32 accumulator
//   (only for rows that selected the expert); the shared expert is added weight 1.
//
//----------------------------------------------------------------------------
// WEIGHT PULL INTERFACE (Q4_K 4-bit codes + per-256-elem-super-block fp16 d/dmin +
//   6-bit block scales -- the request/response streams are INDEPENDENT of PE_M: ONE
//   fetch feeds all B rows)
//     * gamma pull (gn_*)            : bf16 RMSNorm learned scale (UNCHANGED).
//     * attention weight pull (aw_*) : aw_q = PE_N Q4_K 4-bit lanes;
//                                      aw_d/aw_dmin/aw_scales = per-super-block scales.
//     * attention cache read (kc_*)  : latent KV cache (UNCHANGED, bf16).
//     * router weight pull (rw_*)    : rw_q = N_EXPERT Q4_K 4-bit lanes of W_g[k,*].
//     * FFN expert weight pull (fw_*): fw_q / fw_q_up = TN Q4_K 4-bit lanes;
//                                      fw_d_*/fw_dmin_*/fw_scales_* = super-block scales;
//                                      qualified by fw_shared / fw_eidx.
//
//----------------------------------------------------------------------------
// STYLE: sync active-high reset; no latch; no comb loop; deterministic latency.
//============================================================================
module glm_decoder_block_q4k #(
    // ROUTE_EXPOSE (docs/ULTRA_PERF_MODULES.md rank 4): publish the captured
    //   top-k union at T_ROUTE so the system can prefetch it.  0 (DEFAULT) ties
    //   the ports to constants so NO registers are inferred and the netlist is
    //   byte-for-byte the committed block; 1 registers and drives them.
    parameter integer ROUTE_EXPOSE = 0,
    // ---- model / slice config (small-but-faithful) ----
    parameter integer MODEL_DIM  = 128,
    // ---- mla_attn_q4k slice params (passed straight through) ----
    parameter integer PER_ROW_POS = 0,   // 1 = per-row query positions via pos_vec (P1.3a)
    parameter integer PER_ROW_SLEN= 0,   // 1 = per-row causal extents via s_len_vec (P1.3d)
    parameter integer PER_ROW_SEQ = 0,   // 1 = per-row sequence ids via seq_vec (A2; kc_seq out)
    // DSA_REAL_IDX: threaded straight to mla_attn_q4k (see its header at :153-162).
    //   0 (default) = the indexer is fed ZERO key-index vectors, so every key scores 0
    //     and top-K keeps keys 0..min(S,TOPK)-1 by tie-break -- QUERY-INDEPENDENT.
    //   1 = real, query-dependent key-index vectors, so top-K actually depends on the query.
    // This is a NO-OP in the DENSE regime (S <= TOPK: no keys are pulled at all, per
    // mla_attn_q4k.v:165-169), which is why the committed S_MAX=8/TOPK_ATTN=8 slice shows
    // no difference for either value. It becomes load-bearing the moment S_MAX > TOPK_ATTN.
    parameter integer DSA_REAL_IDX = 0,
    // INTRA_CAUSAL (default 0): threaded straight to mla_attn_q4k (see its header at
    //   :171-190).  0 = today's shared-context batch (byte-identical -- no intra-batch
    //   key is injected, every INTRA construct constant-folds away).  1 (5b-leaf) =
    //   within ONE PE_M=B pass row j attends, in addition to the s_reg shared cached
    //   keys, the CURRENT-token keys of the earlier rows 0..j-1 computed in that pass
    //   (intra-batch causal attention).  Forwarded here so the composed system can drive
    //   a position-accurate batched verify; requires PER_ROW_POS=1, PER_ROW_SEQ=0.
    parameter integer INTRA_CAUSAL = 0,
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,           // DSA budget (dense for slice)
    // attention union-slot scratch depth (mla SWIN). Default = mla's own default
    //   (byte-identical). For PER_ROW_SEQ=1 the caller sets SWIN >= PE_M*TOPK_ATTN.
    parameter integer SWIN       = (S_MAX < TOPK_ATTN) ? S_MAX : TOPK_ATTN,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    parameter integer POSW       = 20,
    // ---- MoE / FFN config ----
    parameter integer N_EXPERT   = 8,           // routed experts
    parameter integer TOPK       = 2,           // experts/token
    parameter integer INTER_MOE  = 64,          // MoE expert inter size
    parameter integer INTER_DENSE= 256,         // dense-front inter size
    parameter [31:0]  RSCALE     = 32'h40200000,// routed_scaling_factor 2.5 fp32
    parameter integer TN         = 4,           // swiglu output-tile width
    // ---- vestigial FP8 block size (weight_block_size=[128,128] from DeepSeek-V3 /
    //      the prior GLM-5.2-FP8 track); Q4_K uses 256-elem super-blocks -- kept as
    //      a live param threaded to the leaf units ----
    parameter integer BLK        = 128,
    // ---- PE_M : residual-hidden ROWS (batch B) sharing one weight fetch ----
    parameter integer PE_M       = 1,
    // ---- ACT_HW : activation HW lanes (0 = full width) -- result-invariant
    //      resource knob threaded to glm_act (router sigmoid + both silus) ----
    parameter integer ACT_HW     = 0,
    // ---- GU_CONC : rank 9 -- run the SwiGLU gate and up GEMVs concurrently.
    //   Pure forwarding; default 0 keeps every swiglu_expert_q4k instance on the
    //   committed serial path, so this module's default netlist is unchanged.
    parameter integer GU_CONC    = 0,
    // ---- derived (do NOT override) ----
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer SEQW       = (PE_M  <= 1) ? 1 : $clog2(PE_M),   // per-row seq-id width (PER_ROW_SEQ)
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    // mla_attn_q4k re-derived sizings (mirror its own derivations for port widths)
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
    parameter integer TKIW       = (TOPK <= 1) ? 1 : $clog2(TOPK), // sel_* index w
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_GWM     = $clog2(((INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KWM     = $clog2(FF_KMAX_M + 1),
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1), // router KMAX = same family
    // ---- vestigial FP8 [128,128]-block scale counts (from the prior FP8 track;
    //      superseded by the Q4_K 256-elem super-block counts) ----
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,  // attention scales
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,  // dense FFN scales
    parameter integer FF_NB_M    = (FF_KMAX_M + BLK - 1) / BLK,  // MoE  FFN scales
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK,  // router scales
    // ---- Q4_K super-block counts (#256-wide K super-blocks per weight family) ----
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,  // attention super-blocks
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,  // dense FFN super-blocks
    parameter integer FF_NSB_M   = (FF_KMAX_M + 255) / 256,  // MoE  FFN super-blocks
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,  // router super-blocks
    // ---- MoE expert-evaluation count (loop bound) ----
    //   PE_M==1 : iterate the row's TOPK selected experts (committed datapath).
    //   PE_M>1  : iterate all N_EXPERT routed experts once (one shared fetch each);
    //             each row accumulates only its own selected experts.
    parameter integer NEVAL      = (PE_M == 1) ? TOPK : N_EXPERT,
    parameter integer EVW        = $clog2(NEVAL + 2)
)(
    input  wire                         clk,
    input  wire                         rst,        // sync, active-high

    // ---- control ----
    input  wire                         start,      // 1-cycle pulse: begin layer
    output reg                          busy,
    output reg                          done,       // 1-cycle pulse: y valid
    input  wire                         mode,       // 0=DENSE FFN, 1=MoE FFN
    input  wire [POSW-1:0]              pos,        // token position (RoPE) -- SHARED (row 0)
    input  wire [IDXW:0]               s_len,      // S causal keys (<= S_MAX) -- SHARED (row 0)
    input  wire [POSW*PE_M-1:0]        pos_vec,    // per-row positions  (PER_ROW_POS=1; row0=pos)
    input  wire [(IDXW+1)*PE_M-1:0]    s_len_vec,  // per-row extents    (PER_ROW_SLEN=1; row0=s_len)
    input  wire [SEQW*PE_M-1:0]        seq_vec,    // per-row sequence ids (PER_ROW_SEQ=1; row0=seq0)

    // ---- residual stream in / out (bf16, PE_M rows row-major) ----
    //   row r element k = x_vec[16*(MODEL_DIM*r + k) +: 16]
    input  wire [MODEL_DIM*16*PE_M-1:0] x_vec,      // PE_M * MODEL_DIM bf16
    output reg  [MODEL_DIM*16*PE_M-1:0] y_out,      // PE_M * MODEL_DIM bf16

    // ---- KV latent WRITE-BACK exposure (forwarded from mla_attn_q4k) ----
    //   The committed row's packed [c_kv | k_rope] latent + a valid pulse; ADDITIVE.
    output wire [(KV_LORA+ROPE)*16-1:0] kv_lat_row,
    output wire                         kv_lat_valid,
    // ---- PE_M-WIDE KV latent WRITE-BACK exposure (forwarded from mla_attn_q4k;
    //      5b-sys).  ALL B rows' committed current-token latents (row r packed
    //      [c_kv | k_rope] at kv_lat_row_all[r*(KV_LORA+ROPE)*16 +: (KV_LORA+ROPE)*16])
    //      plus a per-row valid.  At PE_M=1, kv_lat_row_all == kv_lat_row (row 0).
    //      ADDITIVE -- the narrow kv_lat_row above is UNCHANGED (kept for the model's
    //      narrow pager port). ----
    output wire [PE_M*(KV_LORA+ROPE)*16-1:0] kv_lat_row_all,
    output wire [PE_M-1:0]                   kv_lat_valid_all,

    // ---- RMSNorm gamma pull (combinational, which norm: 0=pre-attn,1=pre-FFN) ----
    output wire                         gn_req,     // need a gamma element this cyc
    output wire                         gn_which,   // 0=pre-attn, 1=pre-FFN
    output wire [$clog2(MODEL_DIM)-1:0] gn_idx,     // gamma element index
    input  wire [15:0]                  gn_val,     // gamma[gn_idx] (bf16) -- SHARED

    // ---- attention weight pull (forwarded mla_attn_q4k w_*; Q4_K codes + scales) ----
    output wire                         aw_req,
    output wire [3:0]                   aw_sel,     // 0=W_dq..6=W_o
    output wire [A_GRPW-1:0]            aw_grp,
    output wire [A_KCW-1:0]             aw_k,
    input  wire [PE_N*4-1:0]            aw_q,       // PE_N Q4_K 4-bit weight lanes
    input  wire [16*PE_N*A_NSB-1:0]     aw_d,       // fp16 d per (col,super-block)
    input  wire [16*PE_N*A_NSB-1:0]     aw_dmin,    // fp16 dmin
    input  wire [96*PE_N*A_NSB-1:0]     aw_scales,  // 6-bit scales

    // ---- attention cache read (forwarded mla_attn_q4k kc_*) ----
    output wire                         kc_req,
    output wire [IDXW-1:0]              kc_idx,
    output wire [SEQW-1:0]              kc_seq,     // PER_ROW_SEQ=1: fetched key's sequence window
    input  wire [KV_LORA*16-1:0]        kc_ckv,
    input  wire [ROPE*16-1:0]           kc_krope,
    input  wire                         kc_valid,

    // ---- MoE router weight pull (W_g column; Q4_K codes + scales) ----
    output wire                         rw_req,
    output wire [R_KW-1:0]              rw_k,
    input  wire [4*N_EXPERT-1:0]        rw_q,       // N_EXPERT Q4_K 4-bit = W_g[k,*]
    input  wire [16*N_EXPERT*R_NSB-1:0] rw_d,       // fp16 d
    input  wire [16*N_EXPERT*R_NSB-1:0] rw_dmin,    // fp16 dmin
    input  wire [96*N_EXPERT*R_NSB-1:0] rw_scales,  // 6-bit scales

    // ---- FFN expert weight pull (swiglu_expert_q4k w_*), qualified by which expert ----
    output wire                         fw_req,
    output wire [1:0]                   fw_sel,     // 0=GATE,1=UP,2=DOWN (swiglu)
    output wire [FF_GWD-1:0]            fw_grp,     // dense-sized group (>= moe)
    output wire [FF_KWD-1:0]            fw_k,       // dense-sized k (>= moe)
    output reg                          fw_shared,  // 1 = shared expert weights
    output reg  [EIDXW-1:0]             fw_eidx,    // routed expert id (MoE)
    // ---- EARLY ROUTING EXPOSURE (docs/ULTRA_PERF_MODULES.md rank 4) ----------
    //   The die is strictly DEMAND-pull: the system only learns which experts a
    //   token needs when the datapath asks for their bytes, so every miss is
    //   fully exposed.  But the set is DETERMINISTIC and known the cycle routing
    //   completes (T_ROUTE), long before the first expert is evaluated.  These
    //   two ports publish it at that instant so the system can start fetching the
    //   REST of the union while expert 0 is still being computed.  They are pure
    //   OBSERVATION -- nothing inside this block reads them, and the datapath,
    //   its ordering and its outputs are untouched whether or not anyone listens.
    output wire                         route_valid, // 1-cycle pulse at T_ROUTE done
    output wire [PE_M*TOPK*EIDXW-1:0]   route_set,   // the captured top-k union
    input  wire [4*TN-1:0]              fw_q,       // GATE/DOWN Q4_K 4-bit lanes
    input  wire [4*TN-1:0]              fw_q_up,    // UP companion Q4_K 4-bit lanes
    input  wire [16*TN*FF_NSB_D-1:0]    fw_d_g,     // GATE/DOWN fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]    fw_dmin_g,  // GATE/DOWN fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]    fw_scales_g,// GATE/DOWN 6-bit scales
    input  wire [16*TN*FF_NSB_D-1:0]    fw_d_u,     // UP fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]    fw_dmin_u,  // UP fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]    fw_scales_u // UP 6-bit scales
);
    `include "glm_fp.vh"

    integer ii;
    integer rr;        // PE_M row loop variable

    //========================================================================
    // residual-stream buffers (bf16) -- PER ROW
    //========================================================================
    reg [15:0] xbuf [0:PE_M-1][0:MODEL_DIM-1];   // latched x (residual base for attn add)
    reg [15:0] nrm  [0:PE_M-1][0:MODEL_DIM-1];   // rmsnorm output (fed to attn / ffn)
    reg [15:0] hbuf [0:PE_M-1][0:MODEL_DIM-1];   // h = x + attn  (residual base for ffn add)
    reg [15:0] fbuf [0:PE_M-1][0:MODEL_DIM-1];   // FFN output    (before final residual add)
    reg        mode_q;
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   slen_q;

    // packed views for sub-unit wide ports (PE_M rows row-major)
    reg [MODEL_DIM*16*PE_M-1:0] nrm_vec;
    always @* begin
        for (rr = 0; rr < PE_M; rr = rr + 1)
            for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
                nrm_vec[16*(MODEL_DIM*rr + ii) +: 16] = nrm[rr][ii];
    end

    //========================================================================
    // RMSNorm sub-unit -- PE_M replicated, lockstep off ONE shared gamma pull.
    //   bf16, UNCHANGED numerics from glm_decoder_block.v (modules_to_not_convert).
    //   Gamma is the SAME for every row -> one gn_* pull answers all PE_M units.
    //========================================================================
    reg              rn_start;
    reg              rn_src;            // 0=xbuf, 1=hbuf  (reduce-pass source)
    wire [PE_M-1:0]  rn_in_req, rn_g_req, rn_y_valid, rn_busy, rn_done;
    wire [16*PE_M-1:0] rn_y_out;
    reg  [16*PE_M-1:0] rn_x_in;
    reg  [16*PE_M-1:0] rn_gamma_in;
    reg              rn_x_valid;
    reg              rn_g_valid;
    genvar grn;
    generate
    for (grn = 0; grn < PE_M; grn = grn + 1) begin : RN
        rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_rn (
            .clk(clk), .rst(rst), .start(rn_start),
            .in_req(rn_in_req[grn]), .x_in(rn_x_in[16*grn +: 16]), .x_valid(rn_x_valid),
            .g_req(rn_g_req[grn]), .gamma_in(rn_gamma_in[16*grn +: 16]), .g_valid(rn_g_valid),
            .y_valid(rn_y_valid[grn]), .y_out(rn_y_out[16*grn +: 16]),
            .busy(rn_busy[grn]), .done(rn_done[grn])
        );
    end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rn_busy_unused = &{1'b0, rn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // rmsnorm beat counters (LANES=1 -> beat index == element index).  SHARED:
    // all PE_M units run lockstep, so instance-0's handshake drives the counters.
    reg [$clog2(MODEL_DIM+1)-1:0] rn_ridx;   // reduce read index (x pull)
    reg [$clog2(MODEL_DIM+1)-1:0] rn_widx;    // normalize write index (y store)
    reg [$clog2(MODEL_DIM+1)-1:0] rn_gidx;    // gamma pull index (per g_req beat)
    reg                            rn_gwhich; // which norm is active (0/1)

    //========================================================================
    // gamma pull is COMBINATIONAL: assert gn_req whenever the shared rmsnorm_units
    // request a gamma element (rn_g_req[0]); present the current gamma beat index
    // (rn_gidx) and which norm (rn_gwhich) so the system answers gn_val the SAME
    // cycle.  rn_gidx mirrors the unit's gamma-pass beat (LANES=1 -> beat==idx).
    //========================================================================
    assign gn_req   = rn_g_req[0];
    assign gn_which = rn_gwhich;
    assign gn_idx   = rn_gidx[$clog2(MODEL_DIM)-1:0];

    //========================================================================
    // mla_attn_q4k sub-block (full attention; Q4_K weight projections).  PE_M rows
    // share ONE w_*/kc_* stream (forwarded to aw_*/kc_*).
    //========================================================================
    reg                       at_start;
    wire                      at_busy, at_done;
    wire [MODEL_DIM*16*PE_M-1:0] at_out;
    mla_attn_q4k #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK(TOPK_ATTN), .SWIN(SWIN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .PE_M(PE_M), .PER_ROW_POS(PER_ROW_POS), .PER_ROW_SLEN(PER_ROW_SLEN),
        .PER_ROW_SEQ(PER_ROW_SEQ), .DSA_REAL_IDX(DSA_REAL_IDX),
        .INTRA_CAUSAL(INTRA_CAUSAL)
    ) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .busy(at_busy), .done(at_done),
        .pos(pos_q), .s_len(slen_q), .x_vec(nrm_vec),
        .pos_vec(pos_vec), .s_len_vec(s_len_vec), .seq_vec(seq_vec),
        .w_req(aw_req), .w_sel(aw_sel), .w_grp(aw_grp), .w_k(aw_k),
        .w_q(aw_q), .w_d(aw_d), .w_dmin(aw_dmin), .w_scales(aw_scales),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_seq(kc_seq), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid), .out(at_out),
        .kv_lat_row(kv_lat_row), .kv_lat_valid(kv_lat_valid),
        .kv_lat_row_all(kv_lat_row_all), .kv_lat_valid_all(kv_lat_valid_all)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _at_busy_unused = &{1'b0, at_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // moe_router_q4k (MoE mode).  Its Q4_K W_g pull is forwarded to rw_*.  PE_M
    // rows produce per-row {sel_idx, sel_weight} off ONE shared W_g stream.
    //========================================================================
    reg                       rt_start;
    wire                      rt_busy, rt_done;
    wire [TOPK*EIDXW*PE_M-1:0] rt_sel_idx;
    wire [TOPK*16*PE_M-1:0]    rt_sel_weight;
    moe_router_q4k #(
        .HIDDEN(MODEL_DIM), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .SCALE(RSCALE), .KMAX(FF_KMAX_M), .PE_M(PE_M), .ACT_HW(ACT_HW)
    ) u_router (
        .clk(clk), .rst(rst), .start(rt_start), .busy(rt_busy), .done(rt_done),
        .x_vec(nrm_vec),
        .w_req(rw_req), .w_k(rw_k),
        .w_q(rw_q), .w_d(rw_d), .w_dmin(rw_dmin), .w_scales(rw_scales),
        .sel_idx(rt_sel_idx), .sel_weight(rt_sel_weight)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rt_busy_unused = &{1'b0, rt_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // captured routing result (PER ROW)
    reg [EIDXW-1:0] sel_e   [0:PE_M-1][0:TOPK-1];
    reg [15:0]      sel_w   [0:PE_M-1][0:TOPK-1];

    //========================================================================
    // FFN expert (one DENSE swiglu_expert_q4k + one MoE instance), both PE_M wide.
    //   Only the instance matching `mode_q` is started; the other stays idle.  Both
    //   share the normalized input nrm_vec and pull their Q4_K weights+scales through
    //   the SAME fw_* ports (qualified by fw_shared / fw_eidx, shared by all rows).
    //========================================================================
    // ---- dense expert ----
    reg                  ed_start;
    wire                 ed_busy, ed_done;
    wire [MODEL_DIM*16*PE_M-1:0] ed_y;
    wire                 ed_wreq;
    wire [1:0]           ed_wsel;
    wire [FF_GWD-1:0]    ed_wgrp;
    wire [FF_KWD-1:0]    ed_wk;
    swiglu_expert_q4k #(
        .HIDDEN(MODEL_DIM), .INTER(INTER_DENSE), .TN(TN), .KMAX(FF_KMAX_D),
        .PE_M(PE_M), .ACT_HW(ACT_HW), .GU_CONC(GU_CONC)
    ) u_dense (
        .clk(clk), .rst(rst), .start(ed_start), .busy(ed_busy), .done(ed_done),
        .x_vec(nrm_vec),
        .w_req(ed_wreq), .w_sel(ed_wsel), .w_grp(ed_wgrp), .w_k(ed_wk),
        .w_q(fw_q), .w_q_up(fw_q_up),
        .w_d(fw_d_g), .w_dmin(fw_dmin_g), .w_scales(fw_scales_g),
        .w_d_up(fw_d_u), .w_dmin_up(fw_dmin_u), .w_scales_up(fw_scales_u),
        .y_out(ed_y)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _ed_busy_unused = &{1'b0, ed_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- moe expert (serial reuse) ----
    reg                  em_start;
    wire                 em_busy, em_done;
    wire [MODEL_DIM*16*PE_M-1:0] em_y;
    wire                 em_wreq;
    wire [1:0]           em_wsel;
    wire [FF_GWM-1:0]    em_wgrp;
    wire [FF_KWM-1:0]    em_wk;
    swiglu_expert_q4k #(
        .HIDDEN(MODEL_DIM), .INTER(INTER_MOE), .TN(TN), .KMAX(FF_KMAX_M),
        .PE_M(PE_M), .ACT_HW(ACT_HW), .GU_CONC(GU_CONC)
    ) u_moe (
        .clk(clk), .rst(rst), .start(em_start), .busy(em_busy), .done(em_done),
        .x_vec(nrm_vec),
        .w_req(em_wreq), .w_sel(em_wsel), .w_grp(em_wgrp), .w_k(em_wk),
        .w_q(fw_q), .w_q_up(fw_q_up),
        .w_d(fw_d_g[16*TN*FF_NSB_M-1:0]),
        .w_dmin(fw_dmin_g[16*TN*FF_NSB_M-1:0]),
        .w_scales(fw_scales_g[96*TN*FF_NSB_M-1:0]),
        .w_d_up(fw_d_u[16*TN*FF_NSB_M-1:0]),
        .w_dmin_up(fw_dmin_u[16*TN*FF_NSB_M-1:0]),
        .w_scales_up(fw_scales_u[96*TN*FF_NSB_M-1:0]), .y_out(em_y)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _em_busy_unused = &{1'b0, em_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // FFN weight-pull MUX: forward whichever expert instance is active (shared by
    // all PE_M rows).  Widths: dense grp/k are the WIDE family (FF_GWD/FF_KWD >=
    // moe's), so the moe fields zero-extend.  Q4_K code/scale responses are width-
    // shared (same TN) -- dense uses all FF_NB_D scale K-blocks, moe slices low FF_NB_M.
    // CONFIG GUARD (task B4 followup): the zero-extensions below assume the dense
    // FFN is at least as wide as the MoE FFN (FF_GWD >= FF_GWM, FF_KWD >= FF_KWM),
    // i.e. INTER_DENSE >= INTER_MOE.  At the real GLM-5.2 config this holds
    // (12288 >= 2048).  An INTER_MOE > INTER_DENSE misconfig would make the
    // {(FF_GWD-FF_GWM){1'b0}} replication count NEGATIVE and silently produce a
    // malformed netlist -- fail LOUDLY instead of silently.
    initial begin
        if (FF_GWM > FF_GWD)
            $fatal(1, "glm_decoder_block_q4k: FF_GWM(%0d) > FF_GWD(%0d) -- MoE FFN wider than dense; fw_grp zero-extension requires INTER_DENSE >= INTER_MOE", FF_GWM, FF_GWD);
        if (FF_KWM > FF_KWD)
            $fatal(1, "glm_decoder_block_q4k: FF_KWM(%0d) > FF_KWD(%0d) -- MoE FFN wider than dense; fw_k zero-extension requires INTER_DENSE >= INTER_MOE", FF_KWM, FF_KWD);
    end

    assign fw_req = mode_q ? em_wreq : ed_wreq;
    assign fw_sel = mode_q ? em_wsel : ed_wsel;
    // Clamp the zero-extension count to >= 0 so a misconfig cannot produce a
    // NEGATIVE replication (an iverilog elab error / a yosys huge-unsigned wrap);
    // for every valid config (FF_GWD >= FF_GWM) this equals FF_GWD-FF_GWM exactly
    // -> byte-identical.  The initial-block guard above $fatals with a clear msg.
    localparam integer FF_GXT = (FF_GWD > FF_GWM) ? (FF_GWD - FF_GWM) : 0;
    localparam integer FF_KXT = (FF_KWD > FF_KWM) ? (FF_KWD - FF_KWM) : 0;
    assign fw_grp = mode_q ? {{FF_GXT{1'b0}}, em_wgrp} : ed_wgrp;
    assign fw_k   = mode_q ? {{FF_KXT{1'b0}}, em_wk}   : ed_wk;

    //========================================================================
    // FFN combine accumulator (MoE): facc[r][d] += gate * y[r][d] (fp32) per row;
    // shared added with weight 1.  Per-row, per-element fp32 accumulator array.
    //========================================================================
    reg [31:0] facc [0:PE_M-1][0:MODEL_DIM-1];

    //========================================================================
    // MASTER FSM  (same control as glm_decoder_block.v; PE_M threads the tail)
    //========================================================================
    localparam [4:0]
        T_IDLE   = 5'd0,
        T_RN1    = 5'd1,    // pre-attn rmsnorm(x) -> nrm
        T_ATTN   = 5'd2,    // mla_attn_q4k(nrm) -> at_out
        T_RADD1  = 5'd3,    // h = x + at_out (bf16, per-elt, per-row)
        T_RN2    = 5'd4,    // pre-ffn rmsnorm(h) -> nrm
        T_FFN_D  = 5'd5,    // dense swiglu(nrm) -> fbuf
        T_ROUTE  = 5'd6,    // moe_router_q4k(nrm) -> sel
        T_EXPW   = 5'd8,    // wait expert done
        T_ACC    = 5'd9,    // scale+accumulate expert y into facc (1 elt/cycle/row)
        T_FCOMB  = 5'd10,   // finalize fbuf from facc (MoE, 1 elt/cycle/row)
        T_RADD2  = 5'd11,   // y = h + fbuf (bf16, per-elt, per-row)
        T_DONE   = 5'd12,
        T_ESCAN  = 5'd13;   // PE_M>1: scan expert axis, skip non-union experts
    reg [4:0] state;

    // residual-add element cursor (shared by T_RADD1 / T_RADD2)
    reg [$clog2(MODEL_DIM+1)-1:0] radd_i;
    // FFN combine element cursor (shared by T_ACC / T_FCOMB), streamed ONE element
    // per cycle PER ROW (one fp32_mul+fp32_add per row).
    reg [$clog2(MODEL_DIM+1)-1:0] comb_i;

    // MoE expert loop bookkeeping
    reg [EVW-1:0]            exp_i;        // 0..NEVAL-1 routed, then shared
    reg                      exp_is_shared;
    // next routed-expert slot (PE_M==1 path), addresses sel_e[0][.].
    wire [TKIW-1:0] exp_nxt = exp_i[TKIW-1:0] + 1'b1;

    //========================================================================
    // UNION-SKIP scan cursor (PE_M>1 path only -- mirrors batched_moe.v S_SCAN)
    //   The naive PE_M>1 loop visits ALL N_EXPERT routed experts, one shared
    //   Flash fetch each, even ones no batched row selected.  Because the
    //   workload is Flash-bound, fetching an unused expert wastes the bottleneck.
    //   Instead we SCAN the expert axis and evaluate ONLY the UNION of experts
    //   some row selected, SKIPPING (never fetching) the rest.  `esc` is the scan
    //   cursor 0..N_EXPERT; the terminal value N_EXPERT ends the routed loop and
    //   the always-on shared expert runs exactly as before.  A skipped expert
    //   contributed gate=0 to every row, so the per-row combine is UNCHANGED
    //   (byte-identical output); we just issue FEWER fetches.  DEAD at PE_M==1:
    //   that path keeps the committed per-row TOPK loop and never enters T_ESCAN.
    //========================================================================
    localparam integer ECW = $clog2(N_EXPERT + 1);
    reg [ECW-1:0]           esc;          // union scan cursor (0..N_EXPERT)
    // any_has : is the current scan candidate `esc` in the UNION -- did ANY of the
    // PE_M rows select it?  Pure combinational over the captured routing (sel_e).
    reg          any_has;
    integer      ahr, aht;
    always @* begin
        any_has = 1'b0;
        for (ahr = 0; ahr < PE_M; ahr = ahr + 1)
            for (aht = 0; aht < TOPK; aht = aht + 1)
                // esc < N_EXPERT whenever membership is tested (T_ESCAN guards the
                // terminal esc==N_EXPERT first); the low-bit compare is the exact
                // expert-id match in range.
                if ((esc < ECW'(N_EXPERT)) &&
                    (sel_e[ahr][aht] == esc[EIDXW-1:0]))
                    any_has = 1'b1;
    end

    //========================================================================
    // PER-ROW MoE gate + active mask (combinational from fw_eidx + the captured
    // per-row routing).  row_active[r] : does the CURRENT expert evaluation
    // contribute to row r?  cur_gate_f[r] : that row's routed gate (fp32) for the
    // current expert (the matching sel_w; don't-care when inactive).
    //   PE_M==1 : we iterate the row's own selected experts -> row_active[0] is
    //   ALWAYS 1 and cur_gate_f[0] is sel_w[0][slot], folding to the committed
    //   datapath exactly.  The shared expert (exp_is_shared) is active for all rows
    //   with gate 1.0 (bypassed in the accumulate, weight 1).
    //========================================================================
    reg              row_active [0:PE_M-1];
    reg [31:0]       cur_gate_f [0:PE_M-1];
    integer          cgr, cgt;
    always @* begin
        for (cgr = 0; cgr < PE_M; cgr = cgr + 1) begin
            if (exp_is_shared) begin
                row_active[cgr] = 1'b1;
                cur_gate_f[cgr] = 32'h3F80_0000;     // 1.0 (weight 1, bypassed)
            end else begin
                row_active[cgr] = 1'b0;
                cur_gate_f[cgr] = 32'h0;
                for (cgt = 0; cgt < TOPK; cgt = cgt + 1)
                    if (sel_e[cgr][cgt] == fw_eidx) begin
                        row_active[cgr] = 1'b1;
                        cur_gate_f[cgr] = bf16_to_fp32(sel_w[cgr][cgt]);
                    end
            end
        end
    end

    //========================================================================
    // SHARED fp32 add + bf16 narrow datapath -- REPLICATED PER ROW.  The four
    // mutually-exclusive streaming states T_RADD1 / T_RADD2 / T_ACC / T_FCOMB each
    // need at most ONE fp32 add and ONE fp32->bf16 narrow per cycle PER ROW on a
    // single (shared) element index.
    //   T_RADD1 : hbuf[r] <- bf16( x[r] + attn[r] )
    //   T_RADD2 : y[r]    <- bf16( h[r] + ffn[r]  )
    //   T_ACC   : facc[r] <- facc[r] + gate[r]*em_y[r]   (fp32; held when inactive)
    //   T_FCOMB : fbuf[r] <- bf16( facc[r] )             (narrow only)
    //========================================================================
    reg  [$clog2(MODEL_DIM)-1:0] sh_idx;     // state-muxed element index (SHARED)
    reg  [31:0]                  sh_add_a [0:PE_M-1];
    reg  [31:0]                  sh_add_b [0:PE_M-1];
    reg                          sh_nsel;    // narrow src: 0=add result, 1=facc
    wire [31:0]                  sh_add_s     [0:PE_M-1];
    wire [31:0]                  sh_narrow_in [0:PE_M-1];
    wire [15:0]                  sh_narrow_bf [0:PE_M-1];
    // em_y element as fp32 for the MoE accumulate (per row).
    wire [31:0]                  sh_emy_f     [0:PE_M-1];
    genvar gsh;
    generate
    for (gsh = 0; gsh < PE_M; gsh = gsh + 1) begin : SHN
        assign sh_add_s[gsh]     = fp32_add(sh_add_a[gsh], sh_add_b[gsh]);
        assign sh_narrow_in[gsh] = sh_nsel ? facc[gsh][sh_idx] : sh_add_s[gsh];
        assign sh_narrow_bf[gsh] = fp32_to_bf16(sh_narrow_in[gsh]);
        assign sh_emy_f[gsh]     = bf16_to_fp32(em_y[16*MODEL_DIM*gsh + 16*comb_i +: 16]);
    end
    endgenerate

    integer shr;
    always @* begin
        sh_idx  = radd_i[$clog2(MODEL_DIM)-1:0];
        sh_nsel = 1'b0;
        for (shr = 0; shr < PE_M; shr = shr + 1) begin
            sh_add_a[shr] = 32'h0;
            sh_add_b[shr] = 32'h0;
        end
        case (state)
        T_RADD1: begin
            sh_idx = radd_i[$clog2(MODEL_DIM)-1:0];
            for (shr = 0; shr < PE_M; shr = shr + 1) begin
                sh_add_a[shr] = bf16_to_fp32(xbuf[shr][sh_idx]);
                sh_add_b[shr] = bf16_to_fp32(at_out[16*MODEL_DIM*shr + 16*radd_i +: 16]);
            end
        end
        T_RADD2: begin
            sh_idx = radd_i[$clog2(MODEL_DIM)-1:0];
            for (shr = 0; shr < PE_M; shr = shr + 1) begin
                sh_add_a[shr] = bf16_to_fp32(hbuf[shr][sh_idx]);
                sh_add_b[shr] = bf16_to_fp32(fbuf[shr][sh_idx]);
            end
        end
        T_ACC: begin
            sh_idx = comb_i[$clog2(MODEL_DIM)-1:0];
            for (shr = 0; shr < PE_M; shr = shr + 1) begin
                sh_add_a[shr] = facc[shr][sh_idx];
                sh_add_b[shr] = exp_is_shared ? sh_emy_f[shr]
                                              : fp32_mul(cur_gate_f[shr], sh_emy_f[shr]);
            end
        end
        T_FCOMB: begin
            sh_idx  = comb_i[$clog2(MODEL_DIM)-1:0];
            sh_nsel = 1'b1;
        end
        default: ;
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            state      <= T_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            y_out      <= {MODEL_DIM*16*PE_M{1'b0}};
            mode_q     <= 1'b0;
            pos_q      <= {POSW{1'b0}};
            slen_q     <= {(IDXW+1){1'b0}};
            rn_start   <= 1'b0; rn_src <= 1'b0; rn_gwhich <= 1'b0;
            rn_x_in    <= {16*PE_M{1'b0}}; rn_x_valid <= 1'b0;
            rn_gamma_in<= {16*PE_M{1'b0}}; rn_g_valid <= 1'b0;
            // NOTE: rn_ridx / rn_widx / rn_gidx are owned EXCLUSIVELY by the
            // dedicated rmsnorm beat-counter always-block below (single-driver).
            at_start   <= 1'b0;
            rt_start   <= 1'b0;
            ed_start   <= 1'b0; em_start <= 1'b0;
            fw_shared  <= 1'b0; fw_eidx <= {EIDXW{1'b0}};
            radd_i     <= {$clog2(MODEL_DIM+1){1'b0}};
            comb_i     <= {$clog2(MODEL_DIM+1){1'b0}};
            exp_i      <= {EVW{1'b0}};
            exp_is_shared <= 1'b0;
            esc        <= {ECW{1'b0}};
            for (rr=0; rr<PE_M; rr=rr+1) begin
                for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                    xbuf[rr][ii] <= 16'h0; nrm[rr][ii] <= 16'h0; hbuf[rr][ii] <= 16'h0;
                    fbuf[rr][ii] <= 16'h0; facc[rr][ii] <= 32'h0;
                end
                for (ii=0; ii<TOPK; ii=ii+1) begin
                    sel_e[rr][ii] <= {EIDXW{1'b0}}; sel_w[rr][ii] <= 16'h0;
                end
            end
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            rn_start <= 1'b0;
            at_start <= 1'b0;
            rt_start <= 1'b0;
            ed_start <= 1'b0;
            em_start <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            T_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy   <= 1'b1;
                    mode_q <= mode;
                    pos_q  <= pos;
                    slen_q <= s_len;
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<MODEL_DIM; ii=ii+1)
                            xbuf[rr][ii] <= x_vec[16*(MODEL_DIM*rr + ii) +: 16];
                    // launch pre-attn rmsnorm over x (reduce source = xbuf)
                    rn_src    <= 1'b0;
                    rn_gwhich <= 1'b0;          // pre-attn gamma
                    rn_start  <= 1'b1;
                    state     <= T_RN1;
                end
            end
            //---------------------------------------------------------------- pre-attn norm
            T_RN1: begin
                rn_x_valid <= 1'b0; rn_g_valid <= 1'b0;
                if (rn_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        rn_x_in[16*rr +: 16] <= xbuf[rr][rn_ridx[$clog2(MODEL_DIM)-1:0]];
                    rn_x_valid <= 1'b1;
                end
                if (rn_g_req[0]) begin
                    // gamma pull is COMBINATIONAL (gn_*); SAME gamma to every row.
                    for (rr=0; rr<PE_M; rr=rr+1)
                        rn_gamma_in[16*rr +: 16] <= gn_val;
                    rn_g_valid   <= 1'b1;
                end
                if (rn_y_valid[0])
                    for (rr=0; rr<PE_M; rr=rr+1)
                        nrm[rr][rn_widx[$clog2(MODEL_DIM)-1:0]] <= rn_y_out[16*rr +: 16];
                if (rn_done[0]) begin
                    at_start <= 1'b1;                // launch MLA attention (Q4_K)
                    state    <= T_ATTN;
                end
            end
            //---------------------------------------------------------------- attention
            T_ATTN: begin
                if (at_done) begin
                    // h = x + attn  : start residual add (combinational fp32 add)
                    radd_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_RADD1;
                end
            end
            //---------------------------------------------------------------- residual add 1
            T_RADD1: begin
                // h = x + attn  (shared fp32 add + bf16 narrow, PER ROW)
                for (rr=0; rr<PE_M; rr=rr+1)
                    hbuf[rr][radd_i[$clog2(MODEL_DIM)-1:0]] <= sh_narrow_bf[rr];
                if (radd_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1) begin
                    // launch pre-ffn rmsnorm over h
                    rn_src    <= 1'b1;
                    rn_gwhich <= 1'b1;          // pre-ffn gamma
                    rn_start  <= 1'b1;
                    state     <= T_RN2;
                end else begin
                    radd_i <= radd_i + 1'b1;
                end
            end
            //---------------------------------------------------------------- pre-ffn norm
            T_RN2: begin
                rn_x_valid <= 1'b0; rn_g_valid <= 1'b0;
                if (rn_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        rn_x_in[16*rr +: 16] <= hbuf[rr][rn_ridx[$clog2(MODEL_DIM)-1:0]];
                    rn_x_valid <= 1'b1;
                end
                if (rn_g_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        rn_gamma_in[16*rr +: 16] <= gn_val;     // combinational gamma answer
                    rn_g_valid  <= 1'b1;
                end
                if (rn_y_valid[0])
                    for (rr=0; rr<PE_M; rr=rr+1)
                        nrm[rr][rn_widx[$clog2(MODEL_DIM)-1:0]] <= rn_y_out[16*rr +: 16];
                if (rn_done[0]) begin
                    if (mode_q) begin
                        // MoE: route first
                        rt_start <= 1'b1;
                        state    <= T_ROUTE;
                    end else begin
                        // DENSE: single swiglu expert
                        ed_start  <= 1'b1;
                        fw_shared <= 1'b0;
                        fw_eidx   <= {EIDXW{1'b0}};
                        state     <= T_FFN_D;
                    end
                end
            end
            //---------------------------------------------------------------- dense ffn
            T_FFN_D: begin
                if (ed_done) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<MODEL_DIM; ii=ii+1)
                            fbuf[rr][ii] <= ed_y[16*(MODEL_DIM*rr + ii) +: 16];
                    radd_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_RADD2;
                end
            end
            //---------------------------------------------------------------- moe route
            T_ROUTE: begin
                if (rt_done) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<TOPK; ii=ii+1) begin
                            sel_e[rr][ii] <= rt_sel_idx[EIDXW*(TOPK*rr + ii) +: EIDXW];
                            sel_w[rr][ii] <= rt_sel_weight[16*(TOPK*rr + ii) +: 16];
                        end
                    // clear combine accumulator
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<MODEL_DIM; ii=ii+1) facc[rr][ii] <= 32'h0;
                    exp_i         <= {EVW{1'b0}};
                    exp_is_shared <= 1'b0;
                    // select first expert's weights and start swiglu.  The per-row
                    // gate is COMBINATIONAL (cur_gate_f from fw_eidx + captured
                    // routing), so only fw_eidx needs latching here.
                    //   PE_M==1 : first routed = row 0's slot-0 expert (== committed).
                    //   PE_M>1  : iterate all experts -> start at expert 0.
                    fw_shared  <= 1'b0;
                    if (PE_M == 1) begin
                        // committed path: launch row 0's slot-0 expert directly.
                        fw_eidx  <= rt_sel_idx[EIDXW*0 +: EIDXW];
                        em_start <= 1'b1;
                        state    <= T_EXPW;
                    end else begin
                        // UNION-SKIP: scan the expert axis from 0, fetch/evaluate
                        // only union members (T_ESCAN drives fw_eidx + em_start).
                        esc   <= {ECW{1'b0}};
                        state <= T_ESCAN;
                    end
                end
            end
            //---------------------------------------------------------------- expert union scan (PE_M>1)
            T_ESCAN: begin
                // Mirror batched_moe.v S_SCAN: advance the expert cursor until a
                // UNION member (any_has) or past the last expert.  Non-members are
                // SKIPPED with NO weight fetch (em_start stays deasserted this cyc)
                // -- the Flash-bandwidth win.  Never entered at PE_M==1.
                if (esc == ECW'(N_EXPERT)) begin
                    // whole routed union processed -> run the shared expert (wt 1).
                    exp_is_shared <= 1'b1;
                    fw_shared     <= 1'b1;
                    fw_eidx       <= {EIDXW{1'b0}};
                    em_start      <= 1'b1;
                    state         <= T_EXPW;
                end else if (any_has) begin
                    // esc is in the union -> ONE shared fetch feeds all B rows.
                    fw_shared <= 1'b0;
                    fw_eidx   <= esc[EIDXW-1:0];
                    em_start  <= 1'b1;
                    state     <= T_EXPW;
                end else begin
                    esc <= esc + 1'b1;      // not selected by any row -> skip (no fetch)
                end
            end
            //---------------------------------------------------------------- expert wait
            T_EXPW: begin
                // em_y (u_moe.y_out) holds stable after em_done until the next
                // em_start, so T_ACC can stream it across MODEL_DIM cycles.
                if (em_done) begin
                    comb_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_ACC;
                end
            end
            //---------------------------------------------------------------- scale+accumulate (1 elt/cycle/row)
            T_ACC: begin
                // facc[r][comb_i] += gate[r] * em_y[r][comb_i]  (one fp32 MAC/row).
                // Accumulate ONLY for rows that selected this expert (row_active);
                // the routed-gate multiply is bypassed for the shared expert.
                // At PE_M==1 row_active[0] is always 1 -> unconditional, identical
                // to the committed datapath.
                for (rr=0; rr<PE_M; rr=rr+1)
                    if (row_active[rr])
                        facc[rr][comb_i[$clog2(MODEL_DIM)-1:0]] <= sh_add_s[rr];
                if (comb_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1) begin
                    // last element accumulated -> advance to next expert / finalize
                    if (exp_is_shared) begin
                        comb_i <= {$clog2(MODEL_DIM+1){1'b0}};
                        state  <= T_FCOMB;
                    end else if (PE_M == 1) begin
                        // ---- committed PE_M==1 per-row TOPK loop (UNCHANGED) ----
                        if (exp_i == EVW'(NEVAL-1)) begin
                            // routed experts done -> run shared expert (weight 1).
                            exp_is_shared <= 1'b1;
                            fw_shared     <= 1'b1;
                            fw_eidx       <= {EIDXW{1'b0}};
                            em_start      <= 1'b1;
                            state         <= T_EXPW;
                        end else begin
                            // next routed expert = row 0's slot exp_nxt expert.
                            exp_i      <= exp_i + 1'b1;
                            fw_shared  <= 1'b0;
                            fw_eidx    <= sel_e[0][exp_nxt];
                            em_start   <= 1'b1;
                            state      <= T_EXPW;
                        end
                    end else begin
                        // ---- PE_M>1 UNION-SKIP ----
                        // advance the scan cursor past the expert just evaluated and
                        // re-scan for the next union member (T_ESCAN skips / never
                        // fetches non-selected experts; when the cursor reaches
                        // N_EXPERT it launches the shared expert).
                        esc   <= esc + 1'b1;
                        state <= T_ESCAN;
                    end
                end else begin
                    comb_i <= comb_i + 1'b1;
                end
            end
            //---------------------------------------------------------------- moe combine finalize (1 elt/cycle/row)
            T_FCOMB: begin
                // narrow each fp32 accumulator to bf16 -> fbuf, one elt/cycle/row
                for (rr=0; rr<PE_M; rr=rr+1)
                    fbuf[rr][comb_i[$clog2(MODEL_DIM)-1:0]] <= sh_narrow_bf[rr];
                if (comb_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1) begin
                    radd_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_RADD2;
                end else begin
                    comb_i <= comb_i + 1'b1;
                end
            end
            //---------------------------------------------------------------- residual add 2
            T_RADD2: begin
                // y = h + ffn  (shared fp32 add + bf16 narrow, PER ROW)
                for (rr=0; rr<PE_M; rr=rr+1)
                    y_out[16*MODEL_DIM*rr + 16*radd_i +: 16] <= sh_narrow_bf[rr];
                if (radd_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1)
                    state <= T_DONE;
                else
                    radd_i <= radd_i + 1'b1;
            end
            //----------------------------------------------------------------
            T_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= T_IDLE;
            end
            default: state <= T_IDLE;
            endcase
        end
    end

    //========================================================================
    // rmsnorm pull beat counters (mirror the unit's internal beat order;
    // LANES=1 so beat == element index).  Reset at each rn_start.  All PE_M units
    // run lockstep -> instance-0's handshake drives the shared counters.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            rn_ridx <= {$clog2(MODEL_DIM+1){1'b0}};
            rn_widx <= {$clog2(MODEL_DIM+1){1'b0}};
            rn_gidx <= {$clog2(MODEL_DIM+1){1'b0}};
        end else begin
            if (rn_start) begin
                rn_ridx <= {$clog2(MODEL_DIM+1){1'b0}};
                rn_widx <= {$clog2(MODEL_DIM+1){1'b0}};
                rn_gidx <= {$clog2(MODEL_DIM+1){1'b0}};
            end else begin
                if (rn_in_req[0])  rn_ridx <= rn_ridx + 1'b1;
                if (rn_y_valid[0]) rn_widx <= rn_widx + 1'b1;
                if (rn_g_req[0])   rn_gidx <= rn_gidx + 1'b1;
            end
        end
    end

    // rn_src is the reduce-pass source selector; it is read implicitly by the
    // T_RN1/T_RN2 states answering rn_in_req from xbuf/hbuf respectively, so the
    // signal itself is informational.  Waive unused.
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rn_src_unused = &{1'b0, rn_src};
    /* verilator lint_on UNUSEDSIGNAL */


    // ---- rank 4 routing exposure (parameter-gated; default = no hardware) ----
    generate
    if (ROUTE_EXPOSE == 0) begin : g_route_off
        assign route_valid = 1'b0;
        assign route_set   = {(PE_M*TOPK*EIDXW){1'b0}};
    end else begin : g_route_on
        reg                        rv_r;
        reg [PE_M*TOPK*EIDXW-1:0]  rs_r;
        always @(posedge clk) begin
            if (rst) begin
                rv_r <= 1'b0;
                rs_r <= {(PE_M*TOPK*EIDXW){1'b0}};
            end else begin
                rv_r <= (state == T_ROUTE) && rt_done;
                if ((state == T_ROUTE) && rt_done)
                    rs_r <= rt_sel_idx[PE_M*TOPK*EIDXW-1:0];
            end
        end
        assign route_valid = rv_r;
        assign route_set   = rs_r;
    end
    endgenerate
endmodule
/* verilator lint_on DECLFILENAME */
