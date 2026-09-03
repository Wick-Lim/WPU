//============================================================================
// kda_conv_step.v -- KDA short causal conv, ONE decode token, C channels.
//
// Every KDA layer runs its q, k and v through a depthwise K-tap causal conv
// (`causal_conv1d_update`, seq_len == 1) before the recurrence:
//     window[c]  = [ state[c][0..K-2] , x[c] ]          oldest -> newest
//     state'[c]  =   window[c][1..K-1]                    shift in x
//     conv[c]    = SUM_k w[c][k] * window[c][k]           NO bias
//     y[c]       = silu(conv[c])
// GLM-5.3-Flash: K = 4 (GGUF ssm.conv_kernel), C = 3 * 64 * 128 per layer,
// weights ssm_conv1d_{q,k,v}.weight [4, 1, 8192] F32.  This repo had no conv
// unit of any kind; this is it, at a parameterized slice.
//
// ACCURACY CONTRACT.  torch runs this conv in bf16 with an implementation-
// defined accumulation order, which no fixed datapath can match bitwise.  So
// the ORDER IS PINNED HERE and is this repo's contract (the same stance the KDA
// reductions and the llama.cpp comparison take):
//     conv = bf16_RNE( ((w0*s0 + w1*s1) + w2*s2) + w3*x )   fp32 mul/add,
//                                                           taps ASCENDING
// and the pre-activation bf16 is EXPOSED on conv_out so that leg is checked
// BITWISE (fp32_mul / fp32_add / fp32_to_bf16 are exact primitives, modulo
// fp32_add's pinned 1-ULP gap -- `make fp-ieee`).  y_out then goes through
// glm_act's polynomial SiLU and is a TOLERANCE leg.
//
// ORIENTATION TRAP.  F.conv1d CORRELATES -- it does not flip the kernel -- so
// w[K-1] multiplies the NEWEST sample.  A reversed tap order is the plausible
// wrong reading; -DINJ_CONV_FLIP makes exactly that edit and `make kda-conv`
// must fail with it.
//
// TIMING.  The 4-tap dot is combinational (4 fp32 mul + 3 fp32 add) into one
// register, then glm_act (LAT 6).  Fine at the slice; a real layer would
// pipeline the dot.  done pulses when y_out is valid.
//============================================================================
`timescale 1ns/1ps
`ifndef KDA_CONV_STEP_V
`define KDA_CONV_STEP_V
`include "glm_fp.vh"

module kda_conv_step #(
    parameter integer C = 8,        // channels (lanes)
    parameter integer K = 4,        // taps (GLM-5.3-Flash: 4)
    parameter integer ACT_HW = 0    // glm_act HW_LANES resource knob (0 = full)
)(
    input  wire                    clk,
    input  wire                    rst,          // sync, active-high
    input  wire                    start,        // 1-cycle: x_in / w_in / s_in valid
    output reg                     busy,
    output reg                     done,         // 1-cycle: y_out (and conv_out) valid

    input  wire [32*C-1:0]         x_in,         // newest sample, fp32, per channel
    input  wire [32*C*K-1:0]       w_in,         // taps, channel-major, oldest..newest
    input  wire [32*C*(K-1)-1:0]   s_in,         // history, channel-major, oldest..newest
    output reg  [32*C*(K-1)-1:0]   s_out,        // shifted history (x_in became newest)

    output reg  [16*C-1:0]         conv_out,     // pre-activation bf16 (bitwise leg)
    output wire [16*C-1:0]         y_out         // silu(conv), bf16 (tolerance leg)
);
    integer c, k;
    reg [31:0] acc, win;
    reg        act_v;
    wire       act_ov;
    localparam [1:0] S_IDLE = 2'd0, S_ACT = 2'd1;
    reg [1:0]  state;

    glm_act #(.MODE(1), .LANES(C), .HW_LANES(ACT_HW)) u_silu (
        .clk(clk), .rst(rst),
        .in_valid(act_v), .x_in(conv_out),
        .out_valid(act_ov), .y_out(y_out)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; act_v <= 1'b0;
        end else begin
            done  <= 1'b0;
            act_v <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                busy <= 1'b1;
                for (c = 0; c < C; c = c + 1) begin
                    // dot over the window, taps ASCENDING: oldest history first,
                    // x (newest) last, multiplied by w[K-1].
                    acc = 32'h00000000;
                    for (k = 0; k < K; k = k + 1) begin
                        win = (k < K-1) ? s_in[32*(c*(K-1) + k) +: 32] : x_in[32*c +: 32];
`ifdef INJ_CONV_FLIP
                        // INJECTION (never a normal build): reversed taps -- the
                        // "convolution flips the kernel" misreading.
                        acc = fp32_add(acc, fp32_mul(w_in[32*(c*K + (K-1-k)) +: 32], win));
`else
                        acc = fp32_add(acc, fp32_mul(w_in[32*(c*K + k) +: 32], win));
`endif
                    end
                    conv_out[16*c +: 16] <= fp32_to_bf16(acc);
                    // shift the history: drop the oldest, append x
                    for (k = 0; k < K-2; k = k + 1)
                        s_out[32*(c*(K-1) + k) +: 32] <= s_in[32*(c*(K-1) + k + 1) +: 32];
                    s_out[32*(c*(K-1) + (K-2)) +: 32] <= x_in[32*c +: 32];
                end
                act_v <= 1'b1;        // conv_out is registered this edge; glm_act samples it next
                state <= S_ACT;
            end
            S_ACT: if (act_ov) begin
                done <= 1'b1; busy <= 1'b0; state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // KDA_CONV_STEP_V
