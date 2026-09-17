//============================================================================
// glm53f_decoder_block.v -- the first GLM-5.3-Flash module where a real sublayer
// lives INSIDE the hyper-connection: KDA attention wired into the attention site
// of a two-site mHC block.
//
//   streams [4,D]
//     -> mHC attention site: collapse -> attn_norm -> glm53f_kda_attn -> mix
//     -> mHC FFN site:       collapse -> ffn_norm  -> <FFN handshake>  -> mix
//     -> streams'
//
// BOTH SIDES ARE NOW WIRED, so this is a COMPLETE GLM-5.3-Flash decoder layer for
// blocks 0-2: glm53f_kda_attn in the attention site (fetching its own nine Q8_0
// projections, threading its own recurrence and conv history) and
// glm53f_swiglu_q8 in the FFN site (gate/up/down off Q8_0, asymmetric clamp).
// Those three blocks are the dense front (GLM53F_N_DENSE = 3) and they are KDA,
// since the first MLA block is 3.
//
// WHY THE FFN IS A SIBLING AND NOT swiglu_expert_q4k: that unit's w_q port is
// FOUR BITS PER LANE and the census says this FFN is Q8_0
// (`blk.N.ffn_{gate,up,down}` [12288,4096] x3 [scan]). It cannot carry these
// weights at all -- not "less accurately", at all.
//
// THE TWO ATTENTION ARMS.  ATTN_KIND = 0 is KDA (34 of 45 blocks) --
// glm53f_kda_attn; ATTN_KIND = 1 is MLA+DSA (11) -- glm53f_mla_attn, which is
// NoPE and therefore a sibling of mla_attn_q4k rather than a re-dimensioning of
// it (no W_kr, no rotation, no k_rope cache; a zero-width rotary tail is not
// expressible in Verilog). Both present the same start/busy/done handshake and
// the same [D] bf16 in / out, so the site takes either.
//   THE MLA ARM'S KV CACHE IS NOT HELD HERE. Its ports come straight out: at the
// real shapes a 1 M-token context is ~1 GB of latent per MLA block, and that
// residency decision belongs with the model. The same open item already exists
// for KDA's 4.19 MB/layer recurrent state.
//
// THE TWO FFN ARMS.  FFN_KIND = 0 is the dense Q8_0 SwiGLU of blocks 0-2;
// FFN_KIND = 1 is the MoE of blocks 3-44 -- glm53f_moe_ffn, i.e. the router, the
// Q4_K/Q5_K/Q6_K routed experts and the always-on Q8_0 shared expert.  Both
// present the same start/busy/done handshake and the same [D] bf16 in / out, so
// the site takes either without changing.
//
// NOTHING IS PARAMETERISED-BUT-UNBUILT ANY MORE. Both attention arms and both FFN
// arms exist and are gated, so a complete decoder layer exists for all 45 blocks.
// What remains above this line is MODEL-level: nothing stacks 45 of these, no
// per-tensor weight descriptor exists for any type, and the KV / recurrent state
// residency is still an open decision.
//
// The mHC sublayer contract is what makes this a wiring job rather than a
// redesign: attn_norm/ffn_norm sit between `collapsed` and the sublayer on all 46
// blocks [scan], so the sublayer still sees [D] bf16 in and [D] bf16 out -- the
// same shape mla_attn_q4k and the FFN already present.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_DECODER_BLOCK_V
`define GLM53F_DECODER_BLOCK_V
`include "glm_fp.vh"

module glm53f_decoder_block #(
    parameter integer MODEL_DIM   = 16,
    parameter integer H           = 4,      // hc_mult: residual streams
    parameter integer KH          = 2,      // KDA heads
    parameter integer DK          = 4,
    parameter integer DV          = 4,
    parameter integer RANK        = 4,
    parameter integer CONV_K      = 4,
    parameter integer TN          = 2,
    parameter integer KMAX        = 32,
    parameter integer INTER       = 32,     // dense FFN inter size
    // ATTN_KIND / FFN_KIND are PRESENCE, not selection:
    //   0 = only the first arm is in silicon, 1 = only the second,
    //   2 = BOTH, and `attn_sel` / `ffn_sel` pick per token.
    // 0 and 1 are what a single-kind build wants and are unchanged. 2 is what a
    // TIME-MULTIPLEXED model top needs: the repo's GLM-5.2 top runs ONE decoder
    // block L times and annotates each weight pull with the layer index, and
    // GLM-5.3-Flash cannot do that with an elaboration-time arm because its 45
    // layers are a MIX -- 34 KDA and 11 MLA, 3 dense and 42 MoE. Both machines
    // have to be in silicon anyway; what 2 adds is being able to choose per layer.
    parameter integer ATTN_KIND   = 0,      // 0 = KDA (34/45 blocks).
                                            // 1 = MLA+DSA (11): glm53f_mla_attn.
                                            // 2 = both, chosen by attn_sel.
    parameter integer QK          = 8,      // qk_nope_head_dim (real 256)
    parameter integer QLORA       = 8,      // q_lora_rank      (real 1536)
    parameter integer KVL         = 8,      // kv_lora_rank     (real 512)
    parameter integer VD          = 8,      // v_head_dim       (real 256)
    parameter integer SMAX        = 4,      // attention window (real 2048)
    parameter [31:0]  MLA_SCALE   = 32'h3EB504F3,   // 1/sqrt(qk_nope_head_dim)
    parameter integer MLA_PMAX    = (KH*QK > QLORA)
                                  ? ((KH*QK > KVL) ? KH*QK : KVL)
                                  : ((QLORA > KVL) ? QLORA : KVL),
    parameter integer FFN_KIND    = 0,      // 0 = dense Q8_0 SwiGLU (blocks 0-2).
                                            // 1 = MoE (blocks 3-44): glm53f_moe_ffn.
                                            // 2 = both, chosen by ffn_sel.
    parameter integer N_EXPERT    = 8,      // expert_count (real 288)
    parameter integer TOPK        = 3,      // expert_used_count (real 8)
    parameter [31:0]  EXPERT_SCALE = 32'h40200000,   // expert_weights_scale = 2.5
    parameter integer EIDXW       = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter [15:0]  SWIGLU_LIM  = 16'h4120,  // 10.0 bf16 = swiglu_limit
    parameter [31:0]  EPS         = 32'h358637BD,
    parameter [31:0]  RMS_EPS     = 32'h3727C5AC,
    parameter [31:0]  INV_SQRT_DK = 32'h3F000000,
    parameter integer ITERS       = 20,
    parameter integer RECIP_ITERS = 4,
    parameter integer DLANES      = 1,
    parameter integer PMAX_OUT    = (KH*DK > MODEL_DIM) ? KH*DK : MODEL_DIM
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    output wire                          busy,
    output wire                          done,

    // the four residual streams
    input  wire                          streams_load,
    input  wire [32*H*MODEL_DIM-1:0]     streams_init,
    output wire [32*H*MODEL_DIM-1:0]     streams_cur,

    // mHC weights, per site
    input  wire [8*((2+H)*H)*H*MODEL_DIM-1:0]      a_w_q,
    input  wire [16*((2+H)*H)*(H*MODEL_DIM/32)-1:0] a_w_d,
    input  wire [32*((2+H)*H)-1:0]       a_base,
    input  wire [31:0]                   a_s0, a_s1, a_s2,
    input  wire [8*((2+H)*H)*H*MODEL_DIM-1:0]      f_w_q,
    input  wire [16*((2+H)*H)*(H*MODEL_DIM/32)-1:0] f_w_d,
    input  wire [32*((2+H)*H)-1:0]       f_base,
    input  wire [31:0]                   f_s0, f_s1, f_s2,

    // the block's own learned norms
    input  wire [16*MODEL_DIM-1:0]       attn_norm_w,
    input  wire [16*MODEL_DIM-1:0]       ffn_norm_w,

    // ---- KDA sublayer: its own Q8_0 weight pull, constants and state ----
    output wire                          kda_w_req,
    output wire [3:0]                    kda_w_sel,
    output wire [$clog2(PMAX_OUT/TN+1)-1:0] kda_w_grp,
    output wire [$clog2(KMAX+1)-1:0]     kda_w_k,
    input  wire [16*TN-1:0]              kda_w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] kda_w_q8_d,
    input  wire [32*KH-1:0]              decay_in,
    input  wire [32*KH*DK-1:0]           dt_bias_in,
    input  wire [32*3*KH*DK*CONV_K-1:0]  conv_w_in,
    input  wire [16*DV-1:0]              onorm_w_in,
    input  wire [32*KH*DK*DV-1:0]        kda_s_in,
    output wire [32*KH*DK*DV-1:0]        kda_s_out,
    input  wire [32*3*KH*DK*(CONV_K-1)-1:0] kda_hist_in,
    output wire [32*3*KH*DK*(CONV_K-1)-1:0] kda_hist_out,

    // ---- FFN sublayer: its own Q8_0 weight pull ----
    output wire                          ffn_w_req,
    output wire [1:0]                    ffn_w_sel,
    output wire [$clog2((INTER>MODEL_DIM?INTER:MODEL_DIM)/TN+1)-1:0] ffn_w_grp,
    output wire [$clog2(KMAX+1)-1:0]     ffn_w_k,
    input  wire [16*TN-1:0]              ffn_w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] ffn_w_q8_d,

    // ---- MoE arm (FFN_KIND = 1 only; tied off and inert at FFN_KIND = 0) ----
    // These exist unconditionally because Verilog cannot declare a port
    // conditionally.  At FFN_KIND = 0 the generate arm drives every output to 0
    // and reads none of the inputs, so a dense block is byte-identical to what it
    // was before the MoE arm existed -- `make dec-block` is the check on that.
    output wire                          moe_rw_req,
    output wire [$clog2(MODEL_DIM+1)-1:0] moe_rw_k,
    input  wire [32*N_EXPERT-1:0]        moe_rw_row,   // ffn_gate_inp, F32
    input  wire [32*N_EXPERT-1:0]        moe_bias,     // exp_probs_b, F32
    output wire                          moe_fw_shared,
    output wire [EIDXW-1:0]              moe_fw_eidx,
    input  wire [2:0]                    moe_wt_gate, moe_wt_up, moe_wt_down,
    input  wire [2:0]                    moe_wt_sh_gate, moe_wt_sh_up, moe_wt_sh_down,
    input  wire [4*TN-1:0]               ffn_w_q,
    input  wire [16*TN*((KMAX+255)/256)-1:0]  ffn_w_d, ffn_w_dmin,
    input  wire [96*TN*((KMAX+255)/256)-1:0]  ffn_w_scales,
    input  wire [128*TN*((KMAX+255)/256)-1:0] ffn_w_q6_sc,

    // ---- MLA arm (ATTN_KIND = 1 only; tied off and inert at ATTN_KIND = 0) ----
    // The KV CACHE IS NOT HELD HERE. The sublayer publishes its new latent and
    // pulls old ones by index, and those ports come straight out to whoever owns
    // memory: at the real shapes a 1 M-token context is ~1 GB of latent per MLA
    // block, and that residency decision belongs with the model, not with a
    // decoder block. It is the same shape this block already uses for weights.
    // Runtime arm selection. Read ONLY where the matching KIND is 2; a build with
    // one arm ignores them, so a single-kind instantiation need not drive them.
    input  wire                      attn_sel,   // 0 = KDA, 1 = MLA
    input  wire                      ffn_sel,    // 0 = dense, 1 = MoE

    input  wire [$clog2(SMAX+1)-1:0] mla_s_len,
    output wire                      mla_w_req,
    output wire [2:0]                mla_w_sel,
    output wire [$clog2(KH)-1:0]     mla_w_head,
    output wire [$clog2(MLA_PMAX/TN+1)-1:0] mla_w_grp,
    output wire [$clog2(KMAX+1)-1:0]        mla_w_k,
    input  wire [16*TN-1:0]          mla_w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] mla_w_q8_d,
    output wire                      mla_ckv_wr,
    output wire [16*KVL-1:0]         mla_ckv_out,
    output wire                      mla_c_req,
    output wire [$clog2(SMAX)-1:0]   mla_c_idx,
    input  wire [16*KVL-1:0]         mla_c_vec
);
`ifndef YOSYS
    initial begin
        if (ATTN_KIND > 2)
            $fatal(1, "glm53f_decoder_block: ATTN_KIND must be 0 (KDA only), 1 (MLA only) or 2 (both, chosen by attn_sel)");
        if (FFN_KIND > 2)
            $fatal(1, "glm53f_decoder_block: FFN_KIND must be 0 (dense only), 1 (MoE only) or 2 (both, chosen by ffn_sel)");
    end
`endif

    // ---- the two-site hyper-connection ----
    wire              hc_sub_start, hc_sub_is_ffn;
    wire [16*MODEL_DIM-1:0] hc_sub_vec;
    reg               hc_sub_done;
    reg  [16*MODEL_DIM-1:0] hc_sub_out;

    glm53f_hc_block #(.H(H), .D(MODEL_DIM), .QK(32), .RMS_EPS(RMS_EPS), .EPS(EPS),
                      .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS), .DLANES(DLANES)) u_hc (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .streams_load(streams_load), .streams_init(streams_init), .streams_cur(streams_cur),
        .a_w_q(a_w_q), .a_w_d(a_w_d), .a_base(a_base), .a_s0(a_s0), .a_s1(a_s1), .a_s2(a_s2),
        .f_w_q(f_w_q), .f_w_d(f_w_d), .f_base(f_base), .f_s0(f_s0), .f_s1(f_s1), .f_s2(f_s2),
        .attn_norm_w(attn_norm_w), .ffn_norm_w(ffn_norm_w),
        .sub_start(hc_sub_start), .sub_is_ffn(hc_sub_is_ffn), .sub_vec(hc_sub_vec),
        .sub_done(hc_sub_done), .sub_out(hc_sub_out));

    // ---- the KDA sublayer, sitting in the attention site ----
    reg                     kda_start;
    reg  [16*MODEL_DIM-1:0] kda_x;
    wire                    kda_busy;
    wire                    kda_done;
    wire [32*MODEL_DIM-1:0] kda_y;

    // Both arms present the SAME handshake to the mHC attention site -- start /
    // busy / done and one [D] bf16 vector in, one out -- which is why swapping
    // them is a generate and not a redesign of the site. The KDA arm's outputs are
    // fp32-TYPED and bf16-VALUED; the MLA arm's are bf16, so it is widened here
    // rather than the site being changed.
    // Each arm is instantiated when its KIND ALLOWS it (0/2 for KDA, 1/2 for MLA),
    // and `use_mla` picks which one runs. With one arm present `use_mla` is a
    // constant, so a single-kind build is exactly what it was.
    wire use_mla = (ATTN_KIND == 1) ? 1'b1 : (ATTN_KIND == 2) ? attn_sel : 1'b0;
    wire k_done, m_done;
    wire [32*MODEL_DIM-1:0] k_y, m_y;
    assign kda_done = use_mla ? m_done : k_done;
    assign kda_y    = use_mla ? m_y    : k_y;

    generate
    if (ATTN_KIND != 1) begin : g_kda
        glm53f_kda_attn #(.MODEL_DIM(MODEL_DIM), .H(KH), .DK(DK), .DV(DV), .RANK(RANK),
                          .CONV_K(CONV_K), .TN(TN), .KMAX(KMAX), .EPS(RMS_EPS),
                          .INV_SQRT_DK(INV_SQRT_DK)) u_kda (
            .clk(clk), .rst(rst), .start(kda_start & ~use_mla), .busy(kda_busy), .done(k_done),
            .x_in(kda_x),
            .w_req(kda_w_req), .w_sel(kda_w_sel), .w_grp(kda_w_grp), .w_k(kda_w_k),
            .w_hp(kda_w_hp), .w_q8_d(kda_w_q8_d),
            .decay_in(decay_in), .dt_bias_in(dt_bias_in), .conv_w_in(conv_w_in),
            .onorm_w_in(onorm_w_in),
            .s_in(kda_s_in), .s_out(kda_s_out),
            .hist_in(kda_hist_in), .hist_out(kda_hist_out),
            .y_out(k_y));
    end else begin : g_no_kda
        assign k_done = 1'b0;
        assign k_y    = {32*MODEL_DIM{1'b0}};
        assign kda_s_out    = {32*KH*DK*DV{1'b0}};
        assign kda_hist_out = {32*3*KH*DK*(CONV_K-1){1'b0}};
        assign kda_w_req = 1'b0; assign kda_w_sel = 4'd0;
        assign kda_w_grp = {$clog2(PMAX_OUT/TN+1){1'b0}};
        assign kda_w_k   = {$clog2(KMAX+1){1'b0}};
    end
    if (ATTN_KIND != 0) begin : g_mla
        wire [16*MODEL_DIM-1:0] mla_y;
        glm53f_mla_attn #(.MODEL_DIM(MODEL_DIM), .H(KH), .QK(QK), .QLORA(QLORA),
                          .KVL(KVL), .VD(VD), .SMAX(SMAX), .TN(TN), .KMAX(KMAX),
                          .SCALE(MLA_SCALE), .RMS_EPS(RMS_EPS)) u_mla (
            .clk(clk), .rst(rst), .start(kda_start & use_mla), .busy(),
            .done(m_done), .x_in(kda_x), .s_len(mla_s_len),
            .w_req(mla_w_req), .w_sel(mla_w_sel), .w_head(mla_w_head),
            .w_grp(mla_w_grp), .w_k(mla_w_k),
            .w_hp(mla_w_hp), .w_q8_d(mla_w_q8_d),
            .ckv_wr(mla_ckv_wr), .ckv_out(mla_ckv_out),
            .c_req(mla_c_req), .c_idx(mla_c_idx), .c_vec(mla_c_vec),
            .y_out(mla_y));
        genvar mi;
        for (mi = 0; mi < MODEL_DIM; mi = mi + 1) begin : g_widen
            assign m_y[32*mi +: 32] = {mla_y[16*mi +: 16], 16'h0000};
        end
    end else begin : g_no_mla
        assign m_done = 1'b0;
        assign m_y    = {32*MODEL_DIM{1'b0}};
        assign mla_w_req   = 1'b0;
        assign mla_w_sel   = 3'd0;
        assign mla_w_head  = {$clog2(KH){1'b0}};
        assign mla_w_grp   = {$clog2(MLA_PMAX/TN+1){1'b0}};
        assign mla_w_k     = {$clog2(KMAX+1){1'b0}};
        assign mla_ckv_wr  = 1'b0;
        assign mla_ckv_out = {16*KVL{1'b0}};
        assign mla_c_req   = 1'b0;
        assign mla_c_idx   = {$clog2(SMAX){1'b0}};
    end
    endgenerate
    // ---- the dense FFN, sitting in the FFN site ----
    reg                     ffn_start;
    reg  [16*MODEL_DIM-1:0] ffn_x;
    wire                    ffn_busy;
    wire                    ffn_done;
    wire [16*MODEL_DIM-1:0] ffn_y;

    // Both arms present the SAME handshake to the mHC FFN site -- start/busy/done
    // and one bf16 vector in, one out -- which is why swapping them is a generate
    // and not a redesign of the site.
    localparam integer FNSB = (KMAX + 255) / 256;
    // Same shape as the attention site: each arm exists when its KIND allows it,
    // and `use_moe` picks. With one arm present this is a constant and the build
    // is exactly what it was.
    wire use_moe = (FFN_KIND == 1) ? 1'b1 : (FFN_KIND == 2) ? ffn_sel : 1'b0;
    // BOTH arms drive a weight pull, and at FFN_KIND=2 both are in silicon -- so
    // the pull has to be MUXED, not shared. Wiring both onto the block's ffn_w_*
    // ports gives two drivers and X; measured, the KIND=2 build then failed 236 of
    // dec-block's 676 checks with sel pointing at the arm that was supposed to be
    // identical. The attention site has no such conflict because its two arms own
    // separate port groups (kda_w_* and mla_w_*).
    wire d_done, e_done;
    wire [16*MODEL_DIM-1:0] d_y, e_y;
    wire d_wreq, e_wreq;
    wire [1:0] d_wsel, e_wsel;
    wire [$clog2((INTER>MODEL_DIM?INTER:MODEL_DIM)/TN+1)-1:0] d_wgrp, e_wgrp;
    wire [$clog2(KMAX+1)-1:0] d_wk, e_wk;
    assign ffn_done  = use_moe ? e_done : d_done;
    assign ffn_y     = use_moe ? e_y    : d_y;
    assign ffn_w_req = use_moe ? e_wreq : d_wreq;
    assign ffn_w_sel = use_moe ? e_wsel : d_wsel;
    assign ffn_w_grp = use_moe ? e_wgrp : d_wgrp;
    assign ffn_w_k   = use_moe ? e_wk   : d_wk;

    generate
    if (FFN_KIND != 1) begin : g_dense
        // The dense front is Q8_0 on all three tensors [scan], so the Q4_K/Q6_K
        // header buses are tied off HERE rather than inside the SwiGLU -- the unit
        // itself is type-generic now, because the MoE experts are a Q4_K/Q5_K/Q6_K
        // mix. The MoE arm below drives them.
        glm53f_swiglu_mt #(.HIDDEN(MODEL_DIM), .INTER(INTER), .TN(TN), .KMAX(KMAX),
                           .LIM(SWIGLU_LIM)) u_ffn (
            .clk(clk), .rst(rst), .start(ffn_start & ~use_moe), .busy(ffn_busy),
            .done(d_done), .x_in(ffn_x),
            .wt_gate(3'd2), .wt_up(3'd2), .wt_down(3'd2),      // Q8_0
            .w_req(d_wreq), .w_sel(d_wsel), .w_grp(d_wgrp), .w_k(d_wk),
            .w_q({4*TN{1'b0}}), .w_hp(ffn_w_hp),
            .w_d({16*TN*FNSB{1'b0}}), .w_dmin({16*TN*FNSB{1'b0}}),
            .w_scales({96*TN*FNSB{1'b0}}), .w_q6_sc({128*TN*FNSB{1'b0}}),
            .w_q8_d(ffn_w_q8_d),
            .y_out(d_y));
    end else begin : g_no_dense
        assign d_done = 1'b0;
        assign d_y    = {16*MODEL_DIM{1'b0}};
        assign d_wreq = 1'b0; assign d_wsel = 2'd0;
        assign d_wgrp = {$clog2((INTER>MODEL_DIM?INTER:MODEL_DIM)/TN+1){1'b0}};
        assign d_wk   = {$clog2(KMAX+1){1'b0}};
    end
    if (FFN_KIND != 0) begin : g_moe
        // blocks 3-44.  The routed experts and the shared expert carry DIFFERENT
        // weight types ([scan]: routed Q4_K/Q5_K/Q6_K, shared Q8_0), so the two
        // type triples come in separately and are passed straight through.
        glm53f_moe_ffn #(.HIDDEN(MODEL_DIM), .INTER(INTER), .E(N_EXPERT),
                         .TOPK(TOPK), .TN(TN), .KMAX(KMAX), .SCALE(EXPERT_SCALE),
                         .LIMIT(SWIGLU_LIM), .RECIP_ITERS(RECIP_ITERS),
                         .IDXW(EIDXW)) u_ffn (
            .clk(clk), .rst(rst), .start(ffn_start & use_moe), .busy(),
            .done(e_done),
            .x_in(ffn_x),
            .rw_req(moe_rw_req), .rw_k(moe_rw_k), .rw_row(moe_rw_row),
            .bias_in(moe_bias),
            .fw_req(e_wreq), .fw_shared(moe_fw_shared), .fw_eidx(moe_fw_eidx),
            .fw_sel(e_wsel), .fw_grp(e_wgrp), .fw_k(e_wk),
            .wt_gate(moe_wt_gate), .wt_up(moe_wt_up), .wt_down(moe_wt_down),
            .wt_sh_gate(moe_wt_sh_gate), .wt_sh_up(moe_wt_sh_up),
            .wt_sh_down(moe_wt_sh_down),
            .w_q(ffn_w_q), .w_hp(ffn_w_hp),
            .w_d(ffn_w_d), .w_dmin(ffn_w_dmin), .w_scales(ffn_w_scales),
            .w_q6_sc(ffn_w_q6_sc), .w_q8_d(ffn_w_q8_d),
            .y_out(e_y), .dbg_sel_idx(), .dbg_sel_weight());
    end else begin : g_no_moe
        assign e_done = 1'b0;
        assign e_y    = {16*MODEL_DIM{1'b0}};
        assign e_wreq = 1'b0; assign e_wsel = 2'd0;
        assign e_wgrp = {$clog2((INTER>MODEL_DIM?INTER:MODEL_DIM)/TN+1){1'b0}};
        assign e_wk   = {$clog2(KMAX+1){1'b0}};
        assign moe_rw_req    = 1'b0;
        assign moe_rw_k      = {$clog2(MODEL_DIM+1){1'b0}};
        assign moe_fw_shared = 1'b0;
        assign moe_fw_eidx   = {EIDXW{1'b0}};
    end
    endgenerate

    integer i;
    localparam [1:0] R_IDLE = 2'd0, R_ATTN = 2'd1, R_FFN = 2'd2;
    reg [1:0] rst_st;

    always @(posedge clk) begin
        if (rst) begin
            rst_st <= R_IDLE; hc_sub_done <= 1'b0;
            kda_start <= 1'b0; ffn_start <= 1'b0;
        end else begin
            hc_sub_done <= 1'b0; kda_start <= 1'b0; ffn_start <= 1'b0;

            case (rst_st)
                // route the mHC site's request to whichever sublayer it names
                R_IDLE: if (hc_sub_start) begin
`ifdef INJ_DBLK_SITE_SWAP
                    // must FAIL: the two sites' sublayers exchanged. The site the
                    // mHC block is asking for is carried on sub_is_ffn; ignoring
                    // it sends the FFN's input through KDA and vice versa.
                    if (hc_sub_is_ffn) begin
                        kda_x <= hc_sub_vec; kda_start <= 1'b1; rst_st <= R_ATTN;
                    end else begin
                        ffn_x <= hc_sub_vec; ffn_start <= 1'b1; rst_st <= R_FFN;
                    end
`else
`ifdef INJ_DBLK_NO_KDA
                    // must FAIL: BOTH sites routed to the FFN, i.e. the
                    // attention sublayer never runs. The residual streams still
                    // move and the block still completes -- which is exactly why
                    // this needs to be checked rather than assumed.
                    ffn_x <= hc_sub_vec; ffn_start <= 1'b1; rst_st <= R_FFN;
`else
                    if (hc_sub_is_ffn) begin
                        ffn_x <= hc_sub_vec; ffn_start <= 1'b1; rst_st <= R_FFN;
                    end else begin
                        kda_x <= hc_sub_vec; kda_start <= 1'b1; rst_st <= R_ATTN;
                    end
`endif
`endif
                end

                // KDA's y_out is fp32-typed but bf16-VALUED (it comes off
                // glm_matmul_q4k's bf16 c_out), so narrowing here is lossless.
                R_ATTN: if (kda_done) begin
                    for (i = 0; i < MODEL_DIM; i = i + 1)
                        hc_sub_out[16*i +: 16] <= fp32_to_bf16(kda_y[32*i +: 32]);
                    hc_sub_done <= 1'b1; rst_st <= R_IDLE;
                end

                R_FFN: if (ffn_done) begin
                    hc_sub_out <= ffn_y;
                    hc_sub_done <= 1'b1; rst_st <= R_IDLE;
                end

                default: rst_st <= R_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_DECODER_BLOCK_V
