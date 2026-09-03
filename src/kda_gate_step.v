//============================================================================
// kda_gate_step.v -- KDA forget gate + input gate (beta), ONE decode token.
//
// The elementwise stage between the gate projections and the recurrence
// (Glm5NextTextForgetGate.forward + the beta line of LinearAttention.forward):
//     t[h,d]  = decay[h] * (f[h,d] + dt_bias[h,d])       decay = exp(A_log[h])
//     g[h,d]  = -5.0 * sigmoid(t[h,d])                    lower_bound branch
//     ge[h,d] = exp(g[h,d])                               -> kda_recur g_in
//     beta[h] = sigmoid(b[h])                             -> kda_recur beta_in
//
// DESIGN DECISIONS
//   * decay = exp(A_log) is a function of a STATIC weight (ssm_a [64] F32), so it
//     is host-precomputed once per layer and arrives as fp32 decay_in.  No exp
//     unit is spent on it.
//   * lower_bound = -5.0 for this checkpoint selects the `lb * sigmoid` branch;
//     the reference's softplus branch is dead code here and is not built.
//   * The pipe is chained on valid HANDSHAKES (glm_act out_valid -> exp
//     valid_in -> exp valid_out), not on latency constants, so it stays correct
//     if either sub-pipe is re-timed.
//
// ACCURACY CONTRACT -- read before calling any of this exact.
//   The only sigmoid in this repo (glm_act, MODE 0) is bf16-in / bf16-out with a
//   polynomial exp; fp32_exp_pipe is a Horner polynomial.  The reference does
//   all of this in fp32.  So t is ROUNDED TO bf16 before the sigmoid, and both
//   transcendental stages are approximate: ge_out and beta_out are TOLERANCE
//   legs, and the bound is traced to a measured cause (tools/kda_gate_gen.py
//   reports the error the bf16 argument rounding alone induces).  If that is
//   too coarse for the recurrence, the finding is "the KDA gate path needs an
//   fp32 sigmoid" -- to be stated, not hidden.
//
//   What IS exact and checked BITWISE: the sign of zero.  fp32 sigmoid saturates
//   to exactly 0.0 for large negative t, and IEEE requires -5.0 * +0.0 = -0.0.
//   The reference pins that; g_out is exposed so the TB checks the sign bit on
//   every saturated element.  fp32_mul(-5.0, +0.0) was probed to return
//   0x80000000 before this module relied on it.
//
// INJECTION.  -DINJ_GATE_DECAY_AFTER applies decay AFTER the sigmoid
// (sigmoid(f + dt) * decay) -- the plausible misreading of `decay_rate * g`.
// `make kda-gate` must fail with it.
//============================================================================
`timescale 1ns/1ps
`ifndef KDA_GATE_STEP_V
`define KDA_GATE_STEP_V
`include "glm_fp.vh"

module kda_gate_step #(
    parameter integer H  = 3,
    parameter integer DK = 8,
    parameter [31:0]  LOWER_BOUND = 32'hC0A00000,   // -5.0 (GGUF kda.gate_lower_bound)
    parameter integer ACT_HW = 0
)(
    input  wire                    clk,
    input  wire                    rst,          // sync, active-high
    input  wire                    start,
    output reg                     busy,
    output reg                     done,         // ge_out / beta_out / g_out valid

    input  wire [32*H-1:0]         decay_in,     // exp(A_log[h]), fp32, host-precomputed
    input  wire [32*H-1:0]         b_in,         // b_proj output, fp32
    input  wire [32*H*DK-1:0]      f_in,         // f_b(f_a(h)) output, fp32
    input  wire [32*H*DK-1:0]      dt_bias_in,   // fp32

    output reg  [32*H*DK-1:0]      g_out,        // -5*sigmoid(t), fp32 (sign-of-zero bitwise)
    output reg  [32*H*DK-1:0]      ge_out,       // exp(g), fp32 (tolerance)
    output reg  [32*H-1:0]         beta_out      // sigmoid(b), fp32 (tolerance)
);
    localparam integer N = H*DK;
    integer i, h;

    // ---- stage 1: t = decay*(f+dt) in fp32, rounded to bf16 for glm_act ----
    reg  [16*N-1:0] t_bf;
    reg  [16*H-1:0] b_bf;
    reg             s1_v;
    wire            sg_ov, sb_ov;
    wire [16*N-1:0] sg_bf;                      // sigmoid(t), bf16
    wire [16*H-1:0] sb_bf;                      // sigmoid(b), bf16

    glm_act #(.MODE(0), .LANES(N), .HW_LANES(ACT_HW)) u_sig_g (
        .clk(clk), .rst(rst), .in_valid(s1_v), .x_in(t_bf), .out_valid(sg_ov), .y_out(sg_bf));
    glm_act #(.MODE(0), .LANES(H), .HW_LANES(ACT_HW)) u_sig_b (
        .clk(clk), .rst(rst), .in_valid(s1_v), .x_in(b_bf), .out_valid(sb_ov), .y_out(sb_bf));

    // ---- stage 2: g = LB * sigmoid(t) (fp32), then exp(g) through the pipe ----
    // One exp pipe per element: the slice is small; a real layer time-multiplexes.
    reg             s2_v;
    reg  [32*N-1:0] g_q;
    wire [N-1:0]    ex_ov;
    wire [32*N-1:0] ex_r;
    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_exp
            fp32_exp_pipe u_exp (.clk(clk), .rst(rst), .valid_in(s2_v),
                                 .x(g_q[32*gi +: 32]), .valid_out(ex_ov[gi]),
                                 .result(ex_r[32*gi +: 32]));
        end
    endgenerate

    localparam [1:0] S_IDLE = 2'd0, S_SIG = 2'd1, S_EXP = 2'd2;
    reg [1:0] state;
    reg [31:0] tf;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; s1_v <= 1'b0; s2_v <= 1'b0;
        end else begin
            done <= 1'b0; s1_v <= 1'b0; s2_v <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                busy <= 1'b1;
                for (i = 0; i < N; i = i + 1) begin
                    h = i / DK;
`ifdef INJ_GATE_DECAY_AFTER
                    // INJECTION (never a normal build): feed sigmoid (f+dt) and apply
                    // decay afterwards -- the misreading of `decay_rate * g`.
                    tf = fp32_add(f_in[32*i +: 32], dt_bias_in[32*i +: 32]);
`else
                    tf = fp32_mul(decay_in[32*h +: 32],
                                  fp32_add(f_in[32*i +: 32], dt_bias_in[32*i +: 32]));
`endif
                    t_bf[16*i +: 16] <= fp32_to_bf16(tf);
                end
                for (h = 0; h < H; h = h + 1) b_bf[16*h +: 16] <= fp32_to_bf16(b_in[32*h +: 32]);
                s1_v <= 1'b1; state <= S_SIG;
            end
            S_SIG: if (sg_ov) begin
                // both glm_act instances share LAT, so sb_ov coincides with sg_ov
                for (i = 0; i < N; i = i + 1) begin
`ifdef INJ_GATE_DECAY_AFTER
                    h = i / DK;
                    g_q[32*i +: 32] <= fp32_mul(LOWER_BOUND,
                                       fp32_mul(bf16_to_fp32(sg_bf[16*i +: 16]), decay_in[32*h +: 32]));
`else
                    g_q[32*i +: 32] <= fp32_mul(LOWER_BOUND, bf16_to_fp32(sg_bf[16*i +: 16]));
`endif
                end
                for (h = 0; h < H; h = h + 1) beta_out[32*h +: 32] <= bf16_to_fp32(sb_bf[16*h +: 16]);
                s2_v <= 1'b1; state <= S_EXP;
            end
            S_EXP: if (ex_ov[0]) begin
                g_out  <= g_q;
                ge_out <= ex_r;
                done <= 1'b1; busy <= 1'b0; state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // KDA_GATE_STEP_V
