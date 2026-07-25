//============================================================================
// DEAD / NOT IN PRODUCT HIERARCHY -- superseded by glm_q4k_system (single-
// context product top), kept for reference only.
//   Multi-context batching is a DECLARED NON-GOAL as of the 2026-07 single-
//   context spec decision (docs/USAGE_GAPS.md): the shipped top glm_q4k_system
//   never passes NSEQ (pager defaults to 1). This batched multi-sequence top is
//   instantiated NOWHERE (repo-wide it survives only in docs/comments), is in
//   NO Makefile gate, and has NO testbench -- its Q4_K bit-exactness is the
//   prior-FP8 result with the re-run PENDING (docs/OPERATION_FLOW.md:331,
//   PRODUCT_ROADMAP.md). The model-level per-row batched path that gates DO
//   exercise lives inline in glm_model_q4k / glm_decoder_block_q4k. Retained
//   because this is the ONLY home of the top-owned per-(layer,seq) kv_mem store
//   + the N_STEPS continuous-batching decode-loop RTL. Do not add to a build.
//============================================================================
`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_q4k_soc_ms.v -- MULTI-SEQUENCE batched SoC top (Q4_K local-device track)
//----------------------------------------------------------------------------
// Q4_K-native sibling of the prior glm_fp8_soc_ms (tag 'fp8-verified-baseline'; see
// docs/Q4K_SYSTEM_PLAN.md §1.2).
// IDENTICAL retarget to §1.1: u_model die swap glm_model_fp8 -> glm_model_q4k,
// the three weight-bus families FP8 -> Q4_K (aw_col/aw_scale ->
// aw_q/aw_d/aw_dmin/aw_scales; rw_col/rw_scale -> rw_q/rw_d/rw_dmin/rw_scales;
// fw_col/fw_col_up/fw_scale_g/fw_scale_u -> fw_q/fw_q_up + fw_d_g/fw_dmin_g/
// fw_scales_g + fw_d_u/fw_dmin_u/fw_scales_u), and the _NB->_NSB derived-param
// swap (FP8 128-wide K-blocks -> Q4_K 256-wide super-blocks, ceil(K/256)).  No
// datapath math is touched: the PE_M batching, per-row pos_vec/s_len_vec/seq_vec,
// and NSEQ pager wiring are weight-format-independent and carried through
// UNCHANGED (they already exist on glm_model_q4k).
//
// The deployable batched decode top: runs glm_model_q4k at PE_M=B with
// PER_ROW_SEQ=1 so B DIFFERENT sequences (users) are decoded in ONE forward,
// each row r attending its OWN KV window via a REAL kv_cache_pager (NSEQ=B).
// This is the continuous-batching integration: the B sequences SHARE the
// query-side weight/expert fetch while each keeps its own KV history.
//
// The weight-pull responders (em_*/gn_*/aw_*/rw_*/fw_*/fn_*/lw_*) are
// PE_M-AGNOSTIC and pass straight through to the model, same as the PE_M=1 SoC.
// Includes the routed-expert cache (expert_cache_pf): a resident GDDR6 slot now
// holds one expert's Q4_K weights (~44% fewer bytes than the prior FP8), so more
// experts fit per GB -- byte-agnostic controller, doc-only vs the prior FP8 top.
//
// REAL PER-LAYER KV STORE: this module OWNS the KV data in kv_mem -- L*NSEQ
// windows (one per (layer, sequence)) x KV_RESIDENT positions.  The host WRITES
// it during prefill/decap (per (seq, layer, pos)); the model READS it
// COMBINATIONALLY (window = db_layer*NSEQ+kc_seq, position = kc_idx), kc_valid =
// 1-cycle registered kc_req.  KV latent rows are BF16, NOT quantized
// (ROW_BITS=(KV_LORA+ROPE)*16 is invariant), so this path is FORMAT-IDENTICAL to
// the prior FP8 top.  The pager runs alongside as the per-sequence resident-window /
// Flash-overflow timing model.
//
// HOST FSM (one batched decode step):
//   IDLE -> PREFILL (write each sequence's s_len_r[s] prompt latents for EACH of
//   the L layers into kv_mem, seq by seq) -> RUN (one PE_M=B forward) -> DECAP
//   (write each sequence's new decode-token latent for each layer) -> DONE (commit
//   B argmax tokens).  Row r decodes at position s_len_r[r] with sequence id r
//   (PER_ROW_POS/SLEN/SEQ=1).
//
// STYLE: synchronous, ACTIVE-HIGH reset; no latch; deterministic.
//============================================================================
module glm_q4k_soc_ms #(
    parameter integer MODEL_DIM  = 16,
    parameter integer L          = 2,
    parameter integer N_DENSE    = 1,
    parameter integer VOCAB      = 16,
    parameter integer H_HEADS    = 2,
    parameter integer NOPE       = 4,
    parameter integer ROPE       = 4,
    parameter integer V_DIM      = 4,
    parameter integer Q_LORA     = 8,
    parameter integer KV_LORA    = 8,
    parameter integer S_MAX      = 4,      // >= B*TOPK_ATTN (multi-seq union depth)
    parameter integer TOPK_ATTN  = 2,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 2,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 4,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 8,
    parameter integer INTER_DENSE= 160,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 2,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 2,
    parameter integer PE_M       = 2,      // batch rows == sequences decoded together
    parameter integer N_STEPS    = 1,      // decode steps per start (>1 = multi-token loop)
    // KV pager
    parameter integer KV_CTX     = 1024,   // logical KV context capacity (positions)
    parameter integer KV_RESIDENT= 16,     // KV ring capacity per sequence (POWER OF TWO)
    parameter integer FLASH_LAT  = 8,
    // routed-expert cache (GDDR6 cache + Flash) -- shared by the B sequences: the
    //   batched forward's MoE union-skip means expert episodes are DEDUPLICATED
    //   across sequences, so the cache sees fewer distinct fetches than B separate
    //   decodes.  Demand-only here (prefetch tied off) for a self-contained top.
    //   Each resident slot holds one expert's Q4_K weights (~44% fewer bytes/weight
    //   than the prior FP8) -- the controller is byte-agnostic, so this is a doc-only note.
    parameter integer CACHE_SLOTS = 4,
    parameter integer EFIFO_DEPTH = 16,
    // ---- derived (do NOT override) ----
    parameter integer NSEQ       = PE_M,
    parameter integer STPW       = (N_STEPS <= 1) ? 1 : $clog2(N_STEPS),  // decode-step counter width
    parameter integer WINW       = (L*PE_M <= 1) ? 1 : $clog2(L*PE_M),  // KV window id = layer*NSEQ+seq
    parameter integer SWIN       = PE_M*TOPK_ATTN,               // multi-seq attention scratch
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer SEQW       = (PE_M  <= 1) ? 1 : $clog2(PE_M),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM),
    parameter integer LAYW       = (L<=1)?1:$clog2(L),
    parameter integer EIDXW      = (N_EXPERT<=1)?1:$clog2(N_EXPERT),
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer HQK        = H_HEADS*QK_DIM,
    parameter integer HNOPE      = H_HEADS*NOPE,
    parameter integer HV         = H_HEADS*V_DIM,
    parameter integer A_KMAX     = (MODEL_DIM>Q_LORA)?
                       ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV):((KV_LORA>HV)?KV_LORA:HV))
                     :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV):((KV_LORA>HV)?KV_LORA:HV)),
    parameter integer A_OMAX     = (HQK>MODEL_DIM)?
                       ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                     :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV):((HNOPE>HV)?HNOPE:HV)),
    parameter integer A_NGMAX    = (A_OMAX+PE_N-1)/PE_N,
    parameter integer A_GRPW     = (A_NGMAX<=1)?1:$clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX <=1)?1:$clog2(A_KMAX),
    parameter integer FF_KMAX_D  = (INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM,
    parameter integer FF_KMAX_M  = (INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM,
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN+1),
    parameter integer FF_KWD     = $clog2(FF_KMAX_D+1),
    parameter integer R_KW       = $clog2(FF_KMAX_M+1),
    // ---- Q4_K super-block counts (256-elem super-blocks; ceil(K/256)) ----
    //   the prior FP8 counted 128-wide K-blocks (_NB); Q4_K counts 256-wide super-blocks
    //   (_NSB) -- one (d fp16, dmin fp16, scales 96-bit) header per col per SB.
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,  // attention super-blocks
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,  // dense FFN super-blocks
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,  // router super-blocks
    parameter integer NVTILE     = VOCAB/LM_TN,
    parameter integer VTW        = (NVTILE<=1)?1:$clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer EFW        = (EFIFO_DEPTH <= 1) ? 1 : $clog2(EFIFO_DEPTH)
)(
    input  wire                          clk,
    input  wire                          rst,          // sync, active-high

    //========================== HOST interface (B sequences) ================
    input  wire                          start,        // 1-cycle: begin a batched decode step
    input  wire [PE_M*TOKW-1:0]          prompt_tok,   // B tokens to decode (row r = seq r)
    input  wire [(IDXW+1)*PE_M-1:0]      s_len_vec,    // per-seq prompt length / decode position
    output reg                           busy,
    output reg                           done,         // 1-cycle: batched step committed
    output reg  [PE_M*TOKW-1:0]          next_tok,     // B committed next tokens (argmax)
    output reg                           tok_valid,    // 1-cycle pulse with next_tok
    output wire [PE_M*VOCAB*16-1:0]      logits,       // B * VOCAB bf16 (observability)

    //========================== weight-pull responder ports (PE_M-agnostic) =
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    // ---- attention weight pull (Q4_K: 4-bit codes + super-block d/dmin/scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*4-1:0]             aw_q,         // PE_N Q4_K 4-bit weight lanes
    input  wire [16*PE_N*A_NSB-1:0]      aw_d,         // fp16 d per (col, super-block)
    input  wire [16*PE_N*A_NSB-1:0]      aw_dmin,      // fp16 dmin
    input  wire [96*PE_N*A_NSB-1:0]      aw_scales,    // 6-bit scales
    // ---- MoE router weight pull (Q4_K) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [4*N_EXPERT-1:0]         rw_q,         // N_EXPERT Q4_K 4-bit = W_g[k,*]
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_d,         // fp16 d
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_dmin,      // fp16 dmin
    input  wire [96*N_EXPERT*R_NSB-1:0]  rw_scales,    // 6-bit scales
    // ---- FFN expert weight pull (Q4_K, paired gate/up) ----
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [4*TN-1:0]               fw_q,         // GATE/DOWN Q4_K 4-bit lanes
    input  wire [4*TN-1:0]               fw_q_up,      // UP companion Q4_K 4-bit lanes
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_g,       // GATE/DOWN fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_g,    // GATE/DOWN fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_g,  // GATE/DOWN 6-bit scales
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_u,       // UP fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_u,    // UP fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_u,  // UP 6-bit scales
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,

    //========================== latent-KV read (REAL, module-owned) =========
    //  The KV DATA is stored IN this module (kv_mem, a real per-(layer,seq) KV
    //  store the host populates during prefill), not stubbed from the TB.  The
    //  model's PER-LAYER read (db_layer, kc_seq, kc_idx) is served COMBINATIONALLY
    //  from kv_mem[db_layer*NSEQ+kc_seq][kc_idx]; kc_valid is the 1-cycle registered
    //  ack of kc_req.  KV latent rows are BF16 (not quantized) -- format-identical
    //  to the prior FP8 top.  kc_req/kc_idx/kc_seq are exposed for observability.
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    output wire [SEQW-1:0]               kc_seq,     // multi-seq: which sequence's window

    //========================== KV append (latent ROW source) ===============
    //  During PREFILL/DECAP the host writes one latent row per (seq, LAYER, pos)
    //  into kv_mem AND appends to the pager (window model); the ROW BYTES come from
    //  kv_row_in -- the computed latent a real prefill datapath writes back.
    //  kv_seq_sel/kv_layer_sel/kv_row_sel tell the source which (seq, layer, pos).
    output wire [SEQW-1:0]               kv_seq_sel,
    output wire [LAYW-1:0]               kv_layer_sel,
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,
    output wire [ROW_BITS-1:0]           kv_row_out, // pager gathered row (observability)

    //========================== Flash overflow (cold KV) ====================
    output wire                          flash_req,
    output wire [KVPOSW-1:0]             flash_idx,
    output wire [SEQW-1:0]               flash_seq,
    input  wire                          flash_done,
    input  wire [ROW_BITS-1:0]           flash_row,

    //========================== routed-expert cache Flash channel ============
    output wire                          ec_flash_req,       // an expert-weight Flash fetch in flight
    output wire [EIDXW-1:0]              ec_flash_expert_id, // which routed expert
    input  wire                          ec_flash_done,      // the PHY served it
    output wire [31:0]                   ec_hit_count,       // cache stats (observability)
    output wire [31:0]                   ec_miss_count
);

    //------------------------------------------------------------------------
    // 1) MODEL -- PE_M=B, per-row position / extent / sequence.
    //------------------------------------------------------------------------
    reg                       mdl_start;
    wire                      mdl_busy, mdl_done;
    wire [PE_M*TOKW-1:0]      mdl_argmax;
    // per-row DYNAMIC causal extent / query position = prompt length + decode step.
    //   The loop appends one decode-token KV per step, so at step k row r attends
    //   keys 0..(s_len_r + k) and queries at position s_len_r + k.  N_STEPS=1 ->
    //   dec_step is always 0 -> slen_cur_vec == s_len_vec (byte-identical single step).
    reg [STPW-1:0]            dec_step;               // current decode step (0..N_STEPS-1)
    reg [PE_M*TOKW-1:0]       cur_tok;                // tokens fed to THIS step's forward
    wire [(IDXW+1)*PE_M-1:0]  slen_cur_vec;
    genvar r;
    generate
        for (r = 0; r < PE_M; r = r + 1) begin : g_slencur
            assign slen_cur_vec[(IDXW+1)*r +: (IDXW+1)] =
                s_len_vec[(IDXW+1)*r +: (IDXW+1)] + dec_step;
        end
    endgenerate
    wire [POSW-1:0]           pos0   = {{(POSW-(IDXW+1)){1'b0}}, slen_cur_vec[0 +: (IDXW+1)]};
    wire [IDXW:0]             slen0  = slen_cur_vec[0 +: (IDXW+1)];
    wire [POSW*PE_M-1:0]      pos_vec;
    generate
        for (r = 0; r < PE_M; r = r + 1) begin : g_posvec
            assign pos_vec[POSW*r +: POSW] =
                {{(POSW-(IDXW+1)){1'b0}}, slen_cur_vec[(IDXW+1)*r +: (IDXW+1)]};
        end
    endgenerate
    // per-row sequence id: row r -> sequence r
    wire [SEQW*PE_M-1:0]      seq_vec;
    generate
        for (r = 0; r < PE_M; r = r + 1) begin : g_seqvec
            assign seq_vec[SEQW*r +: SEQW] = r[SEQW-1:0];
        end
    endgenerate

    // model KV read: kc_req/kc_idx/kc_seq are TOP outputs (drive the module's kv_mem
    // read + the pager window model); kc_ckv/kc_krope are served COMBINATIONALLY from
    // kv_mem (below); kc_valid is the 1-cycle registered ack of kc_req.
    reg                       kc_valid_r;
    always @(posedge clk) if (rst) kc_valid_r <= 1'b0; else kc_valid_r <= kc_req;
    wire [KV_LORA*16-1:0]     kc_ckv_w;
    wire [ROPE*16-1:0]        kc_krope_w;

    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .SWIN(SWIN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT),
        .TOPK(TOPK), .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE),
        .TN(TN), .BLK(BLK), .LM_TN(LM_TN), .PE_M(PE_M),
        .PER_ROW_POS(1), .PER_ROW_SLEN(1), .PER_ROW_SEQ(1)
    ) u_model (
        .clk(clk), .rst(rst), .start(mdl_start), .busy(mdl_busy), .done(mdl_done),
        .token_id(cur_tok), .pos(pos0), .s_len(slen0),
        .pos_vec(pos_vec), .s_len_vec(slen_cur_vec), .seq_vec(seq_vec),
        .logits(logits), .argmax(mdl_argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_seq(kc_seq),
        .kc_ckv(kc_ckv_w), .kc_krope(kc_krope_w), .kc_valid(kc_valid_r),
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .h_state()
    );

    //------------------------------------------------------------------------
    // 2) HOST FSM : PREFILL B sequences -> RUN batched forward -> DECAP -> DONE.
    //------------------------------------------------------------------------
    localparam [2:0] H_IDLE=3'd0, H_PREFILL=3'd1, H_RUN=3'd2, H_RUNW=3'd3,
                     H_DECAP=3'd4, H_DONE=3'd5;
    reg [2:0]        hstate;
    reg [SEQW-1:0]   cur_seq;                  // sequence being (pre)filled
    reg [IDXW:0]     cur_pos;                  // position within cur_seq's prompt
    reg [LAYW-1:0]   cur_layer;                // layer being written (each (seq,pos) has L latents)
    reg              ap_valid;                 // drive a KV write / pager append this cycle
    reg [SEQW-1:0]   ap_seq;
    reg [LAYW-1:0]   ap_layer;
    reg [KVPOSW-1:0] ap_pos;

    // this sequence's prompt length (per-seq s_len)
    wire [IDXW:0]    cur_slen = s_len_vec[(IDXW+1)*cur_seq +: (IDXW+1)];

    assign kv_seq_sel   = ap_seq;
    assign kv_layer_sel = ap_layer;
    assign kv_row_sel   = ap_pos;

    always @(posedge clk) begin
        if (rst) begin
            hstate    <= H_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            next_tok  <= {PE_M*TOKW{1'b0}};
            tok_valid <= 1'b0;
            mdl_start <= 1'b0;
            cur_seq   <= {SEQW{1'b0}};
            cur_pos   <= {(IDXW+1){1'b0}};
            cur_layer <= {LAYW{1'b0}};
            dec_step  <= {STPW{1'b0}};
            cur_tok   <= {PE_M*TOKW{1'b0}};
            ap_valid  <= 1'b0;
            ap_seq    <= {SEQW{1'b0}};
            ap_layer  <= {LAYW{1'b0}};
            ap_pos    <= {KVPOSW{1'b0}};
        end else begin
            done      <= 1'b0;
            tok_valid <= 1'b0;
            mdl_start <= 1'b0;
            ap_valid  <= 1'b0;
            case (hstate)
                H_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        cur_seq   <= {SEQW{1'b0}};
                        cur_pos   <= {(IDXW+1){1'b0}};
                        cur_layer <= {LAYW{1'b0}};
                        dec_step  <= {STPW{1'b0}};        // step 0
                        cur_tok   <= prompt_tok;          // first step decodes the prompt tokens
                        // empty prompts -> straight to RUN
                        if (s_len_vec[0 +: (IDXW+1)] == {(IDXW+1){1'b0}} && PE_M == 1)
                            hstate <= H_RUN;
                        else
                            hstate <= H_PREFILL;
                    end
                end
                H_PREFILL: begin
                    // write EACH LAYER's latent of (cur_seq, cur_pos) (the model's KV
                    //   is per-layer).  Inner loop = layer; then position; then sequence.
                    if (cur_pos < cur_slen) begin
                        ap_valid <= 1'b1;
                        ap_seq   <= cur_seq;
                        ap_layer <= cur_layer;
                        ap_pos   <= {{(KVPOSW-(IDXW+1)){1'b0}}, cur_pos};
                        if (cur_layer == (L-1)) begin       // all L layers of this pos done
                            cur_layer <= {LAYW{1'b0}};
                            cur_pos   <= cur_pos + 1'b1;
                        end else
                            cur_layer <= cur_layer + 1'b1;
                    end else begin
                        // this sequence's prefill done -> next sequence (or RUN)
                        cur_pos   <= {(IDXW+1){1'b0}};
                        cur_layer <= {LAYW{1'b0}};
                        if (cur_seq == (NSEQ-1)) begin
                            hstate <= H_RUN;
                        end else begin
                            cur_seq <= cur_seq + 1'b1;
                        end
                    end
                end
                H_RUN: begin
                    mdl_start <= 1'b1;                 // launch the batched forward
                    hstate    <= H_RUNW;
                end
                H_RUNW: begin
                    if (mdl_done) begin
                        next_tok  <= mdl_argmax;       // capture B argmax tokens
                        tok_valid <= 1'b1;             // STREAM this step's B tokens
                        cur_seq   <= {SEQW{1'b0}};
                        cur_layer <= {LAYW{1'b0}};
                        hstate    <= H_DECAP;
                    end
                end
                H_DECAP: begin
                    // write each sequence's decode-token latent at its CURRENT decode
                    //   position s_len_r + dec_step, into EACH LAYER's window (layer
                    //   inner, then sequence).  On the last (seq,layer): if more decode
                    //   steps remain, feed this step's tokens back as the next input and
                    //   loop to RUN (position advances via dec_step); else finish.
                    ap_valid <= 1'b1;
                    ap_seq   <= cur_seq;
                    ap_layer <= cur_layer;
                    ap_pos   <= {{(KVPOSW-(IDXW+1)){1'b0}},
                                 slen_cur_vec[(IDXW+1)*cur_seq +: (IDXW+1)]};
                    if (cur_layer == (L-1)) begin
                        cur_layer <= {LAYW{1'b0}};
                        if (cur_seq == (NSEQ-1)) begin
                            if (dec_step == (N_STEPS-1)) begin
                                hstate <= H_DONE;
                            end else begin
                                dec_step <= dec_step + 1'b1;   // advance the decode step
                                cur_tok  <= next_tok;          // this step's tokens -> next input
                                hstate   <= H_RUN;
                            end
                        end else begin
                            cur_seq <= cur_seq + 1'b1;
                        end
                    end else
                        cur_layer <= cur_layer + 1'b1;
                end
                H_DONE: begin
                    done      <= 1'b1;                 // batched decode of N_STEPS tokens/seq done
                    busy      <= 1'b0;
                    hstate    <= H_IDLE;
                end
                default: hstate <= H_IDLE;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // 3) KV PAGER -- NSEQ INDEPENDENT resident windows (per sequence).  Models
    //    the per-sequence KV window + Flash overflow timing: gather is driven by
    //    the model's kc_* so gather_seq = kc_seq tracks WHICH sequence's window
    //    each read hits (resident vs cold), and append is driven per-seq by the
    //    host FSM.  Row layout: [ c_kv(KV_LORA*16) | k_rope(ROPE*16) ].  The KV
    //    DATA the model consumes is the (per-layer) store below; the pager's
    //    row_out is exposed for observability / a future real per-layer path.
    //------------------------------------------------------------------------
    wire [KVPOSW-1:0] pg_gather_idx = {{(KVPOSW-IDXW){1'b0}}, kc_idx};
    wire              pg_row_valid, pg_busy;

    kv_cache_pager #(
        .ROW_BITS(ROW_BITS), .RESIDENT(KV_RESIDENT), .S_MAX(KV_CTX),
        .FLASH_LAT(FLASH_LAT), .NSEQ(NSEQ)
    ) u_kvpager (
        .clk(clk), .rst(rst),
        .append_valid(ap_valid), .append_row(kv_row_in), .append_seq(ap_seq),
        .gather_valid(kc_req), .gather_idx(pg_gather_idx), .gather_seq(kc_seq),
        .row_valid(pg_row_valid), .row_out(kv_row_out), .busy(pg_busy),
        .flash_req(flash_req), .flash_idx(flash_idx), .flash_seq(flash_seq),
        .flash_done(flash_done), .flash_row(flash_row),
        .append_count(), .resident_lo(), .overflowed(),
        .ecc_serr(), .ecc_derr()
    );

    //------------------------------------------------------------------------
    // 3b) REAL PER-LAYER KV STORE (kv_mem): the module OWNS the KV data.  L*NSEQ
    //    windows x KV_RESIDENT positions, one row per (layer, sequence, position);
    //    the host writes it during prefill/decap and the model reads it
    //    COMBINATIONALLY (kc_valid = registered kc_req).  window = layer*NSEQ + seq.
    //    KV rows are BF16 (unquantized) so this is format-identical to the prior FP8 top.
    //    (Cold / Flash overflow of kv_mem beyond KV_RESIDENT is future work; the
    //    pager above already models that window/timing.)
    //------------------------------------------------------------------------
    localparam integer RPTRW  = (KV_RESIDENT <= 1) ? 1 : $clog2(KV_RESIDENT);
    reg  [ROW_BITS-1:0] kv_mem [0:L*NSEQ*KV_RESIDENT-1];
    wire [WINW-1:0]     ap_win = ap_layer*NSEQ + ap_seq;    // write (layer,seq) window
    wire [WINW-1:0]     g_win  = db_layer*NSEQ + kc_seq;    // read  (layer,seq) window
    // write on prefill/decap (kv_row_in is the (seq,layer,pos) latent the host feeds)
    always @(posedge clk)
        if (ap_valid)
            kv_mem[ap_win*KV_RESIDENT + ap_pos[RPTRW-1:0]] <= kv_row_in;
    // combinational read: serve the model's per-layer KV.  kc_idx (the key position,
    //   < KV_RESIDENT) is used DIRECTLY -- part-selecting kc_idx[RPTRW-1:0] would read
    //   past its IDXW width when IDXW < RPTRW and poison the index with X.
    reg [ROW_BITS-1:0] kv_rd;
    always @* kv_rd = kv_mem[g_win*KV_RESIDENT + kc_idx];
    assign kc_ckv_w   = kv_rd[0          +: KV_LORA*16];
    assign kc_krope_w = kv_rd[KV_LORA*16 +: ROPE*16];

    //------------------------------------------------------------------------
    // 4) ROUTED-EXPERT CACHE (demand-only): detect each distinct (layer, expert)
    //    MoE episode from the model's fw_* stream, FIFO it, serve via
    //    expert_cache_pf (GDDR6 cache + Flash).  With B sequences batched, the
    //    model's MoE union-skip already DEDUPLICATES experts ACROSS sequences, so
    //    the cache sees the batched (shared) episode stream -- fewer distinct Flash
    //    fetches than B independent decodes (the continuous-batching bandwidth win
    //    on the dominant expert-weight traffic).  A resident slot holds one
    //    expert's Q4_K weights (byte-agnostic controller; doc-only vs the prior FP8).
    //    Prefetch is tied off here.
    //------------------------------------------------------------------------
    wire moe_layer  = (db_layer >= N_DENSE[LAYW-1:0]);
    wire cur_routed = fw_req && !fw_shared && moe_layer;

    reg             ep_active;
    reg [EIDXW-1:0] ep_eidx;
    reg [LAYW-1:0]  ep_layer;
    wire new_episode = cur_routed &&
                       (!ep_active || (fw_eidx != ep_eidx) || (db_layer != ep_layer));

    reg [EIDXW-1:0] efifo [0:EFIFO_DEPTH-1];
    reg [EFW:0]     ef_wr, ef_rd;
    wire            ef_empty = (ef_wr == ef_rd);
    wire [EFW:0]    ef_cnt   = ef_wr - ef_rd;
    wire            ef_full  = (ef_cnt == EFIFO_DEPTH[EFW:0]);
    reg             awaiting;
    wire            ec_req_valid = (!ef_empty) && (!awaiting);
    wire [EIDXW-1:0] ec_req_id   = efifo[ef_rd[EFW-1:0]];
    wire            ec_resp_valid, ec_hit, ec_busy;
    wire [CSLOTW-1:0] ec_resp_slot;

    integer efi;
    always @(posedge clk) begin
        if (rst) begin
            ep_active <= 1'b0; ep_eidx <= {EIDXW{1'b0}}; ep_layer <= {LAYW{1'b0}};
            ef_wr <= {(EFW+1){1'b0}}; ef_rd <= {(EFW+1){1'b0}}; awaiting <= 1'b0;
            for (efi = 0; efi < EFIFO_DEPTH; efi = efi + 1) efifo[efi] <= {EIDXW{1'b0}};
        end else begin
            ep_active <= cur_routed;
            if (cur_routed) begin ep_eidx <= fw_eidx; ep_layer <= db_layer; end
            if (new_episode && !ef_full) begin
                efifo[ef_wr[EFW-1:0]] <= fw_eidx;
                ef_wr <= ef_wr + 1'b1;
            end
            if (ec_req_valid) awaiting <= 1'b1;
            if (awaiting && ec_resp_valid) begin awaiting <= 1'b0; ef_rd <= ef_rd + 1'b1; end
        end
    end

    expert_cache_pf #(
        .SLOTS(CACHE_SLOTS), .N_EXPERT(N_EXPERT), .FLASH_LAT(FLASH_LAT), .CACHE_HIT_LAT(0)
    ) u_ecache (
        .clk(clk), .rst(rst),
        .req_valid(ec_req_valid), .req_expert_id(ec_req_id),
        .resp_valid(ec_resp_valid), .hit(ec_hit), .resp_slot(ec_resp_slot), .busy(ec_busy),
        .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}), .pf_ready(),
        .flash_req(ec_flash_req), .flash_expert_id(ec_flash_expert_id), .flash_done(ec_flash_done),
        .hit_count(ec_hit_count), .miss_count(ec_miss_count),
        .demand_stall_cycles(), .pf_issued(), .pf_hit()
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, mdl_busy, pg_busy, pg_row_valid, ec_hit, ec_busy, ec_resp_slot};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
