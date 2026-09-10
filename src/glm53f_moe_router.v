//============================================================================
// glm53f_moe_router.v -- GLM-5.3-Flash's MoE router.
//
//   logits = W_g @ x                     W_g is F32 [HIDDEN, E]  [scan]
//   scores = sigmoid(logits)             [gguf] expert_gating_func = 2
//   idx    = TOP-K(scores + exp_probs_b) [scan] blk.N.exp_probs_b.bias [288] F32
//   w      = scores[idx] / sum * SCALE   [gguf] expert_weights_norm = True,
//                                                expert_weights_scale = 2.5
//
// WHY A SIBLING OF moe_router_q4k RATHER THAN A REUSE.  That module's MATH is
// already right for this model -- sigmoid, top-k, renormalise, scale -- and it
// was written for GLM-5.2. Two things differ, and both are in the checkpoint:
//   * its gate weights are Q4_K (`w_q` is 4 bits per lane); GLM-5.3-Flash's
//     `ffn_gate_inp` is F32 [4096, 288];
//   * it has no `exp_probs_b` input at all, and GLM-5.3-Flash has one on every
//     one of its 43 MoE blocks.
//
// THE BIAS AFFECTS SELECTION ONLY -- AND THAT IS AN ASSUMPTION, NOT A READING.
// The DeepSeek-v3 convention llama.cpp follows adds exp_probs_b when CHOOSING
// experts and weights by the UNBIASED sigmoid scores. No GLM-5.3-Flash modeling
// source is checked out on this branch, so it could not be confirmed. Both
// readings pick the SAME experts and differ only in the weights, so the
// difference is real and quiet. tools/glm53_flash_ref.py implements BOTH and
// `make glm53f-ref` asserts they disagree; INJ_MOER_BIAS_WEIGHTS here is the
// other reading, and it must fail. See docs/GLM53_FLASH_PORT.md 4.3s.
//
// fp32 SIGMOID, NOT glm_act, AND THE REASON IS DISCRETENESS.  Everywhere else a
// 1.2 % bf16 sigmoid error is a tolerance question. Here it is not: the scores
// feed a TOP-K, so near a tie that error changes WHICH EXPERT RUNS -- a discrete,
// unbounded output change, not a small one. This is the case fp32_sigmoid_pipe
// was built for.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_MOE_ROUTER_V
`define GLM53F_MOE_ROUTER_V
`include "glm_fp.vh"
`include "glm_fp_recip.vh"
`include "glm_fp_pipe_lat.vh"

module glm53f_moe_router #(
    parameter integer HIDDEN = 16,
    parameter integer E      = 8,               // expert_count (real 288)
    parameter integer TOPK   = 2,               // expert_used_count (real 8)
    parameter [31:0]  SCALE  = 32'h40200000,    // expert_weights_scale = 2.5
    parameter integer RECIP_ITERS = 4,
    parameter integer IDXW   = (E <= 1) ? 1 : $clog2(E)
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    input  wire [16*HIDDEN-1:0]    x_in,        // bf16 token
    output wire                    w_req,       // need W_g row k this cycle
    output wire [$clog2(HIDDEN+1)-1:0] w_k,
    input  wire [32*E-1:0]         w_row,       // F32 W_g[k][*], all E experts
    input  wire [32*E-1:0]         bias_in,     // exp_probs_b, F32

    output reg  [TOPK*IDXW-1:0]    sel_idx,
    output reg  [TOPK*16-1:0]      sel_weight   // bf16
);
    localparam integer KW = $clog2(HIDDEN + 1);

    reg [31:0] acc  [0:E-1];       // logits
    reg [31:0] scr  [0:E-1];       // sigmoid(logits) -- UNBIASED, these weight
    reg [31:0] chs  [0:E-1];       // scores + bias   -- these select
    reg [KW-1:0] kcnt;
    reg [15:0]   ei, eo;
    integer      e;

    localparam [2:0] S_IDLE=3'd0, S_MAC=3'd1, S_SIG=3'd2, S_TOPK=3'd3,
                     S_SUM=3'd4,  S_RCP=3'd5, S_OUT=3'd6, S_FIN=3'd7;
    reg [2:0] st;

    wire stream = (st == S_MAC);
    assign w_req = stream;
    assign w_k   = kcnt;
    wire [31:0] xk = bf16_to_fp32(x_in[16*kcnt +: 16]);

    // ---- fp32 sigmoid, streamed over the E logits ----
    reg         sg_iv;
    reg  [31:0] sg_x;
    wire        sg_ov;
    wire [31:0] sg_y;
    fp32_sigmoid_pipe #(.RECIP_ITERS(RECIP_ITERS)) u_sig (
        .clk(clk), .rst(rst), .valid_in(sg_iv), .x(sg_x),
        .valid_out(sg_ov), .result(sg_y));

    // ---- the proven top-K, fed the BIASED scores ----
    reg               tk_start;
    wire              tk_loadreq, tk_busy, tk_done;
    reg  [31:0]       tk_score;
    reg               tk_sv;
    wire [TOPK*IDXW-1:0] tk_idx;
    wire [TOPK*32-1:0]   tk_score_o;
    wire [TOPK-1:0]      tk_valid;
    wire [E-1:0]         tk_mask;
    reg  [15:0]          ti;
    topk_select #(.N(E), .K(TOPK), .SCORE_W(32), .LANES_IN(1)) u_topk (
        .clk(clk), .rst(rst), .start(tk_start),
        .load_req(tk_loadreq), .score_in(tk_score), .score_valid(tk_sv),
        .sel_idx_o(tk_idx), .sel_score_o(tk_score_o), .sel_valid_o(tk_valid),
        .mask_o(tk_mask), .busy(tk_busy), .done(tk_done));

    reg [TOPK*IDXW-1:0] idx_r;
    reg [31:0] wsel [0:TOPK-1];
    reg [31:0] ssum, rinv;
    reg [7:0]  ri;
    integer    t;

    function automatic [IDXW-1:0] idx_of(input [TOPK*IDXW-1:0] v, input integer j);
        begin idx_of = v[IDXW*j +: IDXW]; end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            sg_iv <= 1'b0; tk_start <= 1'b0; tk_sv <= 1'b0;
            kcnt <= 0; ei <= 0; eo <= 0; ti <= 0; ri <= 0;
        end else begin
            done <= 1'b0; sg_iv <= 1'b0; tk_start <= 1'b0; tk_sv <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    for (e = 0; e < E; e = e + 1) acc[e] <= 32'd0;
                    kcnt <= 0; busy <= 1'b1; st <= S_MAC;
                end

                // logits = W_g @ x, fp32, one k per cycle across all E experts
                S_MAC: begin
                    for (e = 0; e < E; e = e + 1)
                        acc[e] <= fp32_add(acc[e], fp32_mul(w_row[32*e +: 32], xk));
                    if (kcnt == HIDDEN - 1) begin kcnt <= 0; ei <= 0; eo <= 0; st <= S_SIG; end
                    else                          kcnt <= kcnt + 1'b1;
                end

                // sigmoid, streamed; keep BOTH the unbiased score and score+bias
                S_SIG: begin
                    if (ei < E[15:0]) begin
                        sg_iv <= 1'b1; sg_x <= acc[ei]; ei <= ei + 16'd1;
                    end
                    if (sg_ov) begin
                        scr[eo] <= sg_y;
                        chs[eo] <= fp32_add(sg_y, bias_in[32*eo +: 32]);
                        eo <= eo + 16'd1;
                        if (eo == E[15:0] - 16'd1) begin
                            ti <= 0; tk_start <= 1'b1; st <= S_TOPK;
                        end
                    end
                end

                // TOP-K over the BIASED scores -- selection only
                S_TOPK: begin
                    if (tk_loadreq && ti < E[15:0]) begin
`ifdef INJ_MOER_NO_BIAS
                        tk_score <= scr[ti];      // must FAIL: bias never selects
`else
                        tk_score <= chs[ti];
`endif
                        tk_sv <= 1'b1; ti <= ti + 16'd1;
                    end
                    if (tk_done) begin
                        idx_r <= tk_idx;
                        ssum  <= 32'd0;
                        ri    <= 0;
                        st    <= S_SUM;
                    end
                end

                // sum the SELECTED weights, sequentially
                S_SUM: begin
`ifdef INJ_MOER_BIAS_WEIGHTS
                    // must FAIL: the other reading of exp_probs_b -- weight by the
                    // BIASED score instead of the unbiased sigmoid. Same experts,
                    // different weights; see the header.
                    ssum <= fp32_add(ssum, chs[idx_of(idx_r, ri)]);
                    wsel[ri] <= chs[idx_of(idx_r, ri)];
`else
                    ssum <= fp32_add(ssum, scr[idx_of(idx_r, ri)]);
                    wsel[ri] <= scr[idx_of(idx_r, ri)];
`endif
                    if (ri == TOPK[7:0] - 8'd1) begin ri <= 0; st <= S_RCP; end
                    else                              ri <= ri + 8'd1;
                end

                S_RCP: begin
                    if (ri == 0) rinv <= fp32_recip_seed(ssum);
                    else         rinv <= fp32_recip_step(ssum, rinv);
                    if (ri == RECIP_ITERS[7:0]) begin ri <= 0; st <= S_OUT; end
                    else                              ri <= ri + 8'd1;
                end

                // w = score/sum * SCALE, emitted bf16 with the indices
                S_OUT: begin
                    for (t = 0; t < TOPK; t = t + 1) begin
                        sel_idx[IDXW*t +: IDXW] <= idx_of(idx_r, t);
                        sel_weight[16*t +: 16]  <= fp32_to_bf16(
                            fp32_mul(fp32_mul(wsel[t], rinv), SCALE));
                    end
                    st <= S_FIN;
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_MOE_ROUTER_V
