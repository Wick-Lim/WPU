//============================================================================
// glm53f_mla_attn.v -- the whole GLM-5.3-Flash MLA sublayer, for the 11 blocks
// that are not KDA. This is the module that goes where mla_attn_q4k goes.
//
//     glm53f_mla_proj   x -> qa (q folded through W_uk) and the RAW latent c_kv
//     glm53f_mla_score  qa + the cached latents -> ctx in the LATENT basis
//     glm53f_mla_out    ctx_lat -> W_uv per head -> W_o -> y
//
// [D] bf16 in, [D] bf16 out: the same contract mla_attn_q4k and glm53f_kda_attn
// present, so the mHC attention site takes it without changing.
//
// ---- THE KV CACHE IS PULLED, NOT HELD, AND THAT IS A DECISION ----
// The cache port (c_req / c_idx / c_vec) and the new latent (ckv_out / ckv_wr)
// come STRAIGHT OUT of this module to whoever owns memory. Nothing here stores a
// latent. That is deliberate:
//   * at the real shapes a latent is kv_lora_rank = 512 values, so a 1 M-token
//     context is ~1 GB per MLA block -- a residency decision (BRAM? DDR? paged?)
//     that belongs with the model, not inside an attention sublayer, and one this
//     port has not made yet. The same open item already exists for KDA's
//     4.19 MB/layer recurrent state.
//   * it is the shape this repo already uses for weights: publish an address,
//     have the system answer combinationally.
// The consequence worth stating plainly: THIS MODULE IS NOT A COMPLETE ATTENTION
// LAYER ON ITS OWN. It is complete given a cache. `make glm53f-mla-attn` supplies
// one from the testbench, which is exactly what the system will have to do.
//
// ---- ONE WEIGHT CHANNEL, BECAUSE THE TWO STAGES NEVER OVERLAP ----
// The projection front and the output stage each raise their own weight pull, and
// they are strictly sequential (the output stage cannot start until the score has
// consumed every cached key). So they are MUXED onto one channel here rather than
// exported as two, with w_sel extended: 0..3 are the front's (W_dq, W_uq, W_uk,
// W_dkv), 4 is W_uv and 5 is W_o. Fewer ports for the decoder block to carry, and
// the mux is a fact about the schedule rather than an assumption -- `busy` on the
// inactive stage is what makes it true.
//
// NoPE: there is no W_kr, no rotation, and no k_rope cache anywhere in here. See
// glm53f_mla_proj's header for why that makes this a sibling of mla_attn_q4k
// rather than a re-parameterisation of it.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_MLA_ATTN_V
`define GLM53F_MLA_ATTN_V
`include "glm_fp.vh"

module glm53f_mla_attn #(
    parameter integer MODEL_DIM = 16,
    parameter integer H         = 2,
    parameter integer QK        = 8,
    parameter integer QLORA     = 8,
    parameter integer KVL       = 8,
    parameter integer VD        = 8,
    parameter integer SMAX      = 4,
    parameter integer TN        = 2,
    parameter integer KMAX      = 32,
    parameter [31:0]  SCALE     = 32'h3EB504F3,   // 1/sqrt(QK)
    parameter [31:0]  RMS_EPS   = 32'h3727C5AC,
    parameter integer PMAX_OUT  = (H*QK > QLORA)
                                ? ((H*QK > KVL) ? H*QK : KVL)
                                : ((QLORA > KVL) ? QLORA : KVL)
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output wire                      busy,
    output reg                       done,

    input  wire [16*MODEL_DIM-1:0]   x_in,
    input  wire [$clog2(SMAX+1)-1:0] s_len,      // keys INCLUDING this token

    // ---- the one weight channel (see the header) ----
    output wire                      w_req,
    output reg  [2:0]                w_sel,      // 0..3 front, 4 W_uv, 5 W_o
    output reg  [$clog2(H)-1:0]      w_head,
    output reg  [$clog2(PMAX_OUT/TN+1)-1:0] w_grp,
    output reg  [$clog2(KMAX+1)-1:0]        w_k,
    input  wire [16*TN-1:0]          w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] w_q8_d,

    // ---- the KV cache, owned OUTSIDE this module ----
    output wire                      ckv_wr,     // pulse: append ckv_out at s_len-1
    output wire [16*KVL-1:0]         ckv_out,
    output wire                      c_req,
    output wire [$clog2(SMAX)-1:0]   c_idx,
    input  wire [16*KVL-1:0]         c_vec,

    output wire [16*MODEL_DIM-1:0]   y_out
);
    localparam integer NB8 = (KMAX + 31) / 32;
    localparam integer OPMAX = (VD > MODEL_DIM) ? VD : MODEL_DIM;

    // ---- the projection front ----
    reg          p_start;
    wire         p_busy, p_done, p_wreq;
    wire [1:0]   p_wsel;
    wire [$clog2(H)-1:0] p_whead;
    wire [$clog2(PMAX_OUT/TN+1)-1:0] p_wgrp;
    wire [$clog2(KMAX+1)-1:0] p_wk;
    wire [16*H*KVL-1:0] qa;
    glm53f_mla_proj #(.MODEL_DIM(MODEL_DIM), .H(H), .QK(QK), .QLORA(QLORA),
                      .KVL(KVL), .TN(TN), .KMAX(KMAX), .RMS_EPS(RMS_EPS)) u_p (
        .clk(clk), .rst(rst), .start(p_start), .busy(p_busy), .done(p_done),
        .x_in(x_in), .w_req(p_wreq), .w_sel(p_wsel), .w_head(p_whead),
        .w_grp(p_wgrp), .w_k(p_wk), .w_hp(w_hp), .w_q8_d(w_q8_d),
        .qa_out(qa), .ckv_out(ckv_out));

    // ---- the absorbed-latent inner loop ----
    reg          s_start;
    wire         s_busy, s_done;
    wire [16*H*KVL-1:0] ctx_lat;
    glm53f_mla_score #(.H(H), .KVL(KVL), .SMAX(SMAX), .SCALE(SCALE),
                       .RMS_EPS(RMS_EPS)) u_s (
        .clk(clk), .rst(rst), .start(s_start), .busy(s_busy), .done(s_done),
        .qa_in(qa), .s_len(s_len),
        .c_req(c_req), .c_idx(c_idx), .c_vec(c_vec), .ctx_out(ctx_lat));

    // ---- W_uv per head, then W_o ----
    reg          o_start;
    wire         o_busy, o_done, o_wreq, o_wsel;
    wire [$clog2(H)-1:0] o_whead;
    wire [$clog2(OPMAX/TN+1)-1:0] o_wgrp;
    wire [$clog2(KMAX+1)-1:0] o_wk;
    wire [16*MODEL_DIM-1:0] o_y;
    glm53f_mla_out #(.MODEL_DIM(MODEL_DIM), .H(H), .KVL(KVL), .VD(VD),
                     .TN(TN), .KMAX(KMAX)) u_o (
        .clk(clk), .rst(rst), .start(o_start), .busy(o_busy), .done(o_done),
        .ctx_lat_in(ctx_lat), .w_req(o_wreq), .w_sel(o_wsel), .w_head(o_whead),
        .w_grp(o_wgrp), .w_k(o_wk), .w_hp(w_hp), .w_q8_d(w_q8_d), .y_out(o_y));
`ifdef INJ_MLAA_SKIP_OUT
    // must FAIL: hand the LATENT context straight out, skipping W_uv and W_o.
    // Right width, plausible magnitudes, the entire output stage absent.
    assign y_out = ctx_lat[16*MODEL_DIM-1:0];
`else
    assign y_out = o_y;
`endif

    localparam [2:0] A_IDLE=3'd0, A_PROJ=3'd1, A_SCORE=3'd2, A_OUT=3'd3, A_FIN=3'd4;
    reg [2:0] ast;

    // ---- the weight-channel mux ----
    // SELECTED BY THE WRAPPER'S STATE, NOT BY w_req, and that distinction cost a
    // debugging pass: glm_matmul_q4k LATCHES its header buses on `start`, and a
    // stage pulses mm_start while still in S_PREP -- one cycle BEFORE it raises
    // w_req. Gating the mux on w_req therefore published the OTHER stage's
    // address at exactly the moment the engine sampled w_q8_d, so every group
    // after the first dequantised against the wrong block scale. The end-to-end
    // gate caught it (ckv wrong while `make mla-proj` stayed bitwise, i.e. the
    // composition rather than the arithmetic). `ast` is stable for the whole
    // stage, so it is the correct selector.
`ifdef INJ_MLAA_MUX_ON_WREQ
    // must FAIL, and this leg exists because the bug was REAL: selecting the
    // weight channel by w_req instead of by the wrapper's state publishes the
    // other stage's address in the cycle glm_matmul_q4k latches its headers
    // (mm_start is pulsed in S_PREP, one cycle before w_req rises), so every
    // group after the first dequantises against the wrong block scale. The unit
    // gates stayed bitwise while this was broken; only the composed gate saw it.
    wire use_out = ~p_wreq;
`else
    wire use_out = (ast == A_OUT);
`endif
    assign w_req = p_wreq | o_wreq;
    always @* begin
        if (!use_out) begin
            w_sel  = {1'b0, p_wsel};
            w_head = p_whead;
            w_grp  = p_wgrp;
            w_k    = p_wk;
        end else begin
            w_sel  = o_wsel ? 3'd5 : 3'd4;
            w_head = o_whead;
            w_grp  = {{($clog2(PMAX_OUT/TN+1) - $clog2(OPMAX/TN+1)){1'b0}}, o_wgrp};
            w_k    = o_wk;
        end
    end

    // the new latent is published for one cycle when the front finishes
`ifdef INJ_MLAA_NO_CKVWR
    // must FAIL: never publish this token's latent, so the cache holds only the
    // OLDER keys. Invisible on a corpus where s_len == 1 never appears -- which
    // is why the generator requires one.
    assign ckv_wr = 1'b0;
`else
    assign ckv_wr = p_done;
`endif
    assign busy = (ast != A_IDLE);

    always @(posedge clk) begin
        if (rst) begin
            ast <= A_IDLE; done <= 1'b0;
            p_start <= 1'b0; s_start <= 1'b0; o_start <= 1'b0;
        end else begin
            done <= 1'b0; p_start <= 1'b0; s_start <= 1'b0; o_start <= 1'b0;
            case (ast)
                A_IDLE:  if (start) begin p_start <= 1'b1; ast <= A_PROJ; end
                // the score may only start once the cache HOLDS this token's
                // latent -- ckv_wr fired on p_done, so one cycle of separation is
                // the contract with the cache owner, not an optimisation.
                A_PROJ:  if (p_done) begin s_start <= 1'b1; ast <= A_SCORE; end
                A_SCORE: if (s_done) begin o_start <= 1'b1; ast <= A_OUT; end
                A_OUT:   if (o_done) ast <= A_FIN;
                A_FIN:   begin done <= 1'b1; ast <= A_IDLE; end
                default: ast <= A_IDLE;
            endcase
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, p_busy, s_busy, o_busy};
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
`endif // GLM53F_MLA_ATTN_V
