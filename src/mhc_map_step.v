//============================================================================
// mhc_map_step.v -- the mHC map: the 24 mixed logits of a block's
// hyper-connection turned into (pre[H], post[H], comb[HxH]).
// Glm5NextTextHyperConnection.forward, everything except the two GEMMs.
//
//   pre[h]     = sigmoid(mixed[h]      * scale0 + base[h])       + eps
//   post[h]    = 2 * sigmoid(mixed[H+h] * scale1 + base[H+h])
//   comb[i][j] = softmax_j(mixed[2H+i*H+j] * scale2 + base[2H+i*H+j]) + eps,
//                then Sinkhorn (src/mhc_sinkhorn.v)
//
// SCOPE.  A full hyper-connection is: unweighted RMSNorm over H*D = 16384, a
// [(2+H)*H, H*D] GEMV producing `mixed`, THIS unit, then the collapse
// (sum_h pre[h]*streams[h], D wide) and the mix (comb @ streams).  The norm and
// the GEMV are existing units and the collapse/mix are D-wide streaming
// datapaths; what did not exist is the map, which is where every numerical trap
// lives.  Same scoping as kda_gate_step.v.
//
// THE TRAPS, each a must-fail injection in test/mhc_map_step_tb.v:
//   * `post` is 2*sigmoid, not sigmoid.  A plain sigmoid halves every sublayer's
//     contribution to its residual stream -- silent, and wrong everywhere.
//                                                            [INJ_MAP_POST_NO2]
//   * `pre` carries a +eps that `post` does not.  eps = 1e-6 keeps the collapse
//     weights strictly positive; it is also the reason an fp32 sigmoid is needed
//     at all, since bf16 cannot represent 1 + 1e-6.         [INJ_MAP_PRE_NOEPS]
//   * `comb` gets its own +eps AFTER the softmax divide, not before.
//                                                           [INJ_MAP_COMB_NOEPS]
//   * the softmax subtracts the row max before exp.  fp32_exp_pipe overflows to
//     +inf above ~88 and flushes below ~-88, so on a wide row the unshifted
//     version does not merely round differently, it saturates.
//                                                         [INJ_MAP_SOFTMAX_NOMAX]
//
// PRECISION.  The reference evaluates sigmoid and softmax in float64 and casts
// the result to fp32; this unit is fp32 throughout (fp32_sigmoid_pipe over
// fp32_exp_pipe, and x*recip(y) for the softmax divide since there is no
// divider).  So the gate is a MEASURED bound, not an equality claim, and the
// binding error term is fp32_exp_pipe's polynomial (2.3e-4, `make fp-sigmoid`) --
// not the sigmoid wrapper and not the reciprocal.  See
// docs/GLM53_FLASH_PORT.md 4.3j.
//
// COST.  (H+2) + (2H + LAT_SIG) + H + (H*H + LAT_EXP) + H + (RECIP_ITERS+1) + H
// + 1 + NPASS*(2H+RECIP_ITERS+2) + 3 cycles, of which 548 are Sinkhorn; the TB
// measures the total and pins it.
// Twice per block over 45 blocks => ~63k cycles/token.
//============================================================================
`timescale 1ns/1ps
`ifndef MHC_MAP_STEP_V
`define MHC_MAP_STEP_V
`include "glm_fp.vh"
`include "glm_fp_recip.vh"
`include "glm_fp_pipe_lat.vh"

module mhc_map_step #(
    parameter integer H           = 4,              // hc_mult
    parameter [31:0]  EPS         = 32'h358637BD,   // 1e-6 fp32
    parameter integer ITERS       = 20,             // hc_sinkhorn_iters
    parameter integer RECIP_ITERS = 4
)(
    input  wire        clk,
    input  wire        rst,          // sync, active-high
    input  wire        start,
    output reg         busy,
    output reg         done,

    input  wire [32*(2+H)*H-1:0] mixed_in,   // fn @ flat, fp32
    input  wire [32*(2+H)*H-1:0] base_in,    // the block's mHC bias
    input  wire [31:0]           scale0,     // pre / post / comb gains
    input  wire [31:0]           scale1,
    input  wire [31:0]           scale2,

    output reg  [32*H-1:0]       pre_out,
    output reg  [32*H-1:0]       post_out,
    output reg  [32*H*H-1:0]     comb_out
);
    localparam integer NN      = H*H;
    localparam integer LAT_SIG = `FP_EXP_LAT + 2 + RECIP_ITERS;
    localparam [31:0]  TWO     = 32'h40000000;

    localparam [3:0] S_IDLE = 4'd0,  S_LOGIT = 4'd1,  S_SIG  = 4'd2,
                     S_MAXR = 4'd3,  S_EXP   = 4'd4,  S_RSUM = 4'd5,
                     S_RRCP = 4'd6,  S_NORM  = 4'd7,  S_LOAD = 4'd8,
                     S_SINK = 4'd9,  S_FIN   = 4'd10;

    reg [3:0]  st;
    reg [31:0] pa  [0:H-1];      // pre  sigmoid arguments
    reg [31:0] pb  [0:H-1];      // post sigmoid arguments
    reg [31:0] cl  [0:NN-1];     // comb logits
    reg [31:0] mx  [0:H-1];      // per-row max
    reg [31:0] ev  [0:NN-1];     // exp values
    reg [31:0] rs  [0:H-1];      // per-row sums
    reg [31:0] rr  [0:H-1];      // per-row reciprocals
    reg [31:0] cb0 [0:NN-1];     // softmax output + eps, pre-Sinkhorn
    reg [7:0]  t, ii, oo;

    integer k, i;

    // Total order on fp32 (no NaNs on this path): map to a monotone unsigned key.
    // Local rather than added to glm_fp.vh -- every pinned netlist baseline in the
    // repo depends on that header being untouched.
    function automatic fp32_gt(input [31:0] a, input [31:0] b);
        reg [31:0] ka, kb;
        begin
            ka = a[31] ? ~a : (a | 32'h80000000);
            kb = b[31] ? ~b : (b | 32'h80000000);
            fp32_gt = (ka > kb);
        end
    endfunction

    // ---- shared fp32 sigmoid: 2H args streamed through one pipe ----
    reg         sig_iv;
    reg  [31:0] sig_x;
    wire        sig_ov;
    wire [31:0] sig_y;
    fp32_sigmoid_pipe #(.RECIP_ITERS(RECIP_ITERS)) u_sig (
        .clk(clk), .rst(rst), .valid_in(sig_iv), .x(sig_x),
        .valid_out(sig_ov), .result(sig_y));

    // ---- shared exp for the softmax: NN args streamed through one pipe ----
    reg         exp_iv;
    reg  [31:0] exp_x;
    wire        exp_ov;
    wire [31:0] exp_y;
    fp32_exp_pipe u_exp (
        .clk(clk), .rst(rst), .valid_in(exp_iv), .x(exp_x),
        .valid_out(exp_ov), .result(exp_y));

    // ---- Sinkhorn ----
    reg                sink_start;
    wire               sink_busy, sink_done;
    reg  [32*NN-1:0]   sink_in;
    wire [32*NN-1:0]   sink_out;
    mhc_sinkhorn #(.H(H), .EPS(EPS), .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS)) u_sink (
        .clk(clk), .rst(rst), .start(sink_start), .busy(sink_busy), .done(sink_done),
        .c_in(sink_in), .c_out(sink_out));

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            sig_iv <= 1'b0; exp_iv <= 1'b0; sink_start <= 1'b0;
            t <= 8'd0; ii <= 8'd0; oo <= 8'd0;
        end else begin
            done <= 1'b0; sig_iv <= 1'b0; exp_iv <= 1'b0; sink_start <= 1'b0;

            case (st)
                S_IDLE: if (start) begin
                    t <= 8'd0; ii <= 8'd0; oo <= 8'd0;
                    busy <= 1'b1; st <= S_LOGIT;
                end

                // ---- w*scale + base for all 24, H lanes wide ----
                S_LOGIT: begin
                    if (t < H) begin
                        for (k = 0; k < H; k = k + 1)
                            cl[t*H + k] <= fp32_add(
                                fp32_mul(mixed_in[32*(2*H + t*H + k) +: 32], scale2),
                                base_in[32*(2*H + t*H + k) +: 32]);
                    end else if (t == H) begin
                        for (k = 0; k < H; k = k + 1)
                            pa[k] <= fp32_add(fp32_mul(mixed_in[32*k +: 32], scale0),
                                              base_in[32*k +: 32]);
                    end else begin
                        for (k = 0; k < H; k = k + 1)
                            pb[k] <= fp32_add(fp32_mul(mixed_in[32*(H+k) +: 32], scale1),
                                              base_in[32*(H+k) +: 32]);
                    end
                    if (t == H + 1) begin t <= 8'd0; st <= S_SIG; end
                    else                  t <= t + 8'd1;
                end

                // ---- 2H sigmoids: issue back-to-back, collect as they emerge ----
                S_SIG: begin
                    if (ii < 2*H) begin
                        sig_iv <= 1'b1;
                        sig_x  <= (ii < H) ? pa[ii] : pb[ii - H];
                        ii     <= ii + 8'd1;
                    end
                    if (sig_ov) begin
                        if (oo < H) begin
`ifdef INJ_MAP_PRE_NOEPS
                            pre_out[32*oo +: 32] <= sig_y;                  // must FAIL
`else
                            pre_out[32*oo +: 32] <= fp32_add(sig_y, EPS);
`endif
                        end else begin
`ifdef INJ_MAP_POST_NO2
                            post_out[32*(oo-H) +: 32] <= sig_y;             // must FAIL
`else
                            post_out[32*(oo-H) +: 32] <= fp32_mul(TWO, sig_y);
`endif
                        end
                        oo <= oo + 8'd1;
                        if (oo == 2*H - 1) begin t <= 8'd0; ii <= 8'd0; oo <= 8'd0; st <= S_MAXR; end
                    end
                end

                // ---- per-row max, H lanes ----
                S_MAXR: begin
                    for (k = 0; k < H; k = k + 1) begin
                        if (t == 0) mx[k] <= cl[k*H];
                        else        mx[k] <= fp32_gt(mx[k], cl[k*H + t]) ? mx[k] : cl[k*H + t];
                    end
                    if (t == H - 1) begin t <= 8'd0; st <= S_EXP; end
                    else                  t <= t + 8'd1;
                end

                // ---- exp(logit - rowmax), NN streamed ----
                S_EXP: begin
                    if (ii < NN) begin
                        exp_iv <= 1'b1;
`ifdef INJ_MAP_SOFTMAX_NOMAX
                        exp_x  <= cl[ii];                                   // must FAIL
`else
                        exp_x  <= fp32_add(cl[ii], {~mx[ii/H][31], mx[ii/H][30:0]});
`endif
                        ii     <= ii + 8'd1;
                    end
                    if (exp_ov) begin
                        ev[oo] <= exp_y;
                        oo <= oo + 8'd1;
                        if (oo == NN - 1) begin t <= 8'd0; ii <= 8'd0; oo <= 8'd0; st <= S_RSUM; end
                    end
                end

                // ---- per-row sequential sum. The reference divides by the BARE
                //      sum here; comb's eps arrives after the divide.
                S_RSUM: begin
                    for (k = 0; k < H; k = k + 1) begin
                        if (t == 0) rs[k] <= ev[k*H];
                        else        rs[k] <= fp32_add(rs[k], ev[k*H + t]);
                    end
                    if (t == H - 1) begin t <= 8'd0; st <= S_RRCP; end
                    else                  t <= t + 8'd1;
                end

                S_RRCP: begin
                    for (k = 0; k < H; k = k + 1) begin
                        if (t == 0) rr[k] <= fp32_recip_seed(rs[k]);
                        else        rr[k] <= fp32_recip_step(rs[k], rr[k]);
                    end
                    if (t == RECIP_ITERS) begin t <= 8'd0; st <= S_NORM; end
                    else                        t <= t + 8'd1;
                end

                S_NORM: begin
                    for (k = 0; k < H; k = k + 1)
`ifdef INJ_MAP_COMB_NOEPS
                        cb0[k*H + t] <= fp32_mul(ev[k*H + t], rr[k]);        // must FAIL
`else
                        cb0[k*H + t] <= fp32_add(fp32_mul(ev[k*H + t], rr[k]), EPS);
`endif
                    if (t == H - 1) begin t <= 8'd0; st <= S_LOAD; end
                    else                  t <= t + 8'd1;
                end

                // One cycle to hand cb0 to Sinkhorn. Worth a state: sampling cb0
                // in the last S_NORM cycle would miss that cycle's own writes,
                // and recomputing the last column inline would duplicate the
                // INJ_MAP_COMB_NOEPS ifdef in two places.
                S_LOAD: begin
                    for (i = 0; i < NN; i = i + 1)
                        sink_in[32*i +: 32] <= cb0[i];
                    sink_start <= 1'b1;
                    st <= S_SINK;
                end

                S_SINK: if (sink_done) begin comb_out <= sink_out; st <= S_FIN; end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end

                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // MHC_MAP_STEP_V
