//============================================================================
// glm53f_kda_attn.v -- the complete KDA sublayer: the layer datapath plus the
// projection engine that fetches its own weights.
//
//   glm53f_kda_layer  the datapath and the two pieces of state
//   glm53f_kda_gemv   the nine Q8_0 projections, streamed off glm_matmul_q4k
//
// This is the module that goes where mla_attn_q4k goes, for 34 of the 45 blocks.
// It presents the same shape to the system that swiglu_expert_q4k does -- a
// weight-pull stream (w_req / w_sel / w_grp / w_k answered with w_hp and w_q8_d)
// -- and it threads the recurrent state and conv history in and out, which is the
// part no existing attention slot carries.
//
// Splitting it in two rather than folding the engine into the layer keeps the
// layer's own gate (`make kda-layer`) valid and separately meaningful: that one
// checks the datapath against the reference with the projections handed to it,
// this one checks that the same datapath still lands when the projections come
// off real Q8_0 weights. Two claims, two gates.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_KDA_ATTN_V
`define GLM53F_KDA_ATTN_V

module glm53f_kda_attn #(
    parameter integer MODEL_DIM = 16,
    parameter integer H         = 2,
    parameter integer DK        = 4,
    parameter integer DV        = 4,
    parameter integer RANK      = 4,
    parameter integer CONV_K    = 4,
    parameter integer TN        = 2,
    parameter integer KMAX      = 32,
    parameter [31:0]  EPS         = 32'h3727C5AC,
    parameter [31:0]  INV_SQRT_DK = 32'h3F000000,
    parameter integer PMAX_IN   = (MODEL_DIM > H*DV) ? MODEL_DIM : H*DV,
    parameter integer PMAX_OUT  = (H*DK > MODEL_DIM) ? H*DK : MODEL_DIM
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    output wire                          busy,
    output wire                          done,

    input  wire [16*MODEL_DIM-1:0]       x_in,

    // weight pull, the swiglu_expert_q4k shape
    output wire                          w_req,
    output wire [3:0]                    w_sel,
    output wire [$clog2(PMAX_OUT/TN+1)-1:0] w_grp,
    output wire [$clog2(KMAX+1)-1:0]        w_k,
    input  wire [16*TN-1:0]              w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] w_q8_d,

    input  wire [32*H-1:0]               decay_in,
    input  wire [32*H*DK-1:0]            dt_bias_in,
    input  wire [32*3*H*DK*CONV_K-1:0]   conv_w_in,
    input  wire [16*DV-1:0]              onorm_w_in,

    input  wire [32*H*DK*DV-1:0]         s_in,
    output wire [32*H*DK*DV-1:0]         s_out,
    input  wire [32*3*H*DK*(CONV_K-1)-1:0] hist_in,
    output wire [32*3*H*DK*(CONV_K-1)-1:0] hist_out,

    output wire [32*MODEL_DIM-1:0]       y_out
);
    wire                  pr_req, pr_done;
    wire [3:0]            pr_sel;
    wire [32*PMAX_IN-1:0] pr_in;
    wire [32*PMAX_OUT-1:0] pr_out;

    glm53f_kda_layer #(.MODEL_DIM(MODEL_DIM), .H(H), .DK(DK), .DV(DV), .RANK(RANK),
                       .CONV_K(CONV_K), .EPS(EPS), .INV_SQRT_DK(INV_SQRT_DK)) u_layer (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_in(x_in),
        .proj_req(pr_req), .proj_sel(pr_sel), .proj_in(pr_in),
        .proj_done(pr_done), .proj_out(pr_out),
        .decay_in(decay_in), .dt_bias_in(dt_bias_in), .conv_w_in(conv_w_in),
        .onorm_w_in(onorm_w_in),
        .s_in(s_in), .s_out(s_out), .hist_in(hist_in), .hist_out(hist_out),
        .y_out(y_out));

    glm53f_kda_gemv #(.MODEL_DIM(MODEL_DIM), .H(H), .DK(DK), .DV(DV), .RANK(RANK),
                      .TN(TN), .KMAX(KMAX)) u_gemv (
        .clk(clk), .rst(rst),
        .proj_req(pr_req), .proj_sel(pr_sel), .proj_in(pr_in),
        .proj_done(pr_done), .proj_out(pr_out),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_hp(w_hp), .w_q8_d(w_q8_d));
endmodule
`endif // GLM53F_KDA_ATTN_V
