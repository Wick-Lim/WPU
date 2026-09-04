//============================================================================
// mhc_sinkhorn.v -- the Sinkhorn projection inside GLM-5.3-Flash's
// Manifold-Constrained Hyper-Connections (Glm5NextTextHyperConnection).
//
// Projects a HxH row-stochastic matrix toward the doubly-stochastic manifold by
// alternating column and row normalisations.  H = hc_mult = 4: the block carries
// FOUR parallel residual streams and this matrix re-mixes them.
//
// WHY 20 ITERATIONS, AND WHY NO EARLY EXIT.  `hc_sinkhorn_iters = 20` is a
// PUBLISHED CONFIG CONSTANT, not a convergence tolerance.  The precision study
// (docs/GLM53_FLASH_PORT.md 4.3i) measured the double-stochasticity residual over
// 200 draws: the MEDIAN reaches the eps floor (1.0e-6) at 20 iterations, but the
// worst is 4.8e-4, and it grows with the spread of the softmax logits -- whose
// trained value is not published.  So the matrix the model actually uses is
// whatever 20 iterations produce, doubly stochastic or not, and reproducing the
// model means running exactly 20.  A convergence-based early exit would be both
// faster and LESS FAITHFUL.  (An earlier version of that study measured one draw
// and concluded 20 was "past the plateau"; it is not.)  Bonus: the latency is
// therefore data-independent, NPASS is a constant, and no comparator is needed.
//
// FOUR TRAPS.  The first three are must-fail injections in
// test/mhc_sinkhorn_tb.v; the fourth is measured and deliberately NOT gated --
// see the note under it, because a gate that cannot fail is worse than no gate:
//   1. The loop is NOT `ITERS` symmetric passes.  The reference does ONE COLUMN
//      normalise, then `ITERS-1` iterations of (row, then column):
//          NPASS = 1 + (ITERS-1)*2 = 39 passes, pass 0 = COLUMN.
//      Hence `axis_row = pass[0]` -- odd passes are rows.      [INJ_SINK_SYMM]
//   2. It starts on the COLUMN axis.  Starting on rows transposes the whole
//      schedule and changes the fixed point.                [INJ_SINK_ROWFIRST]
//   3. Every normalise divides by (sum + EPS), never a bare sum.  EPS is what
//      keeps the smallest entries at ~1e-8 instead of underflowing, and it is
//      why the residual has a 1e-6 floor at all.               [INJ_SINK_NOEPS]
//   4. The reduction is SEQUENTIAL in index order.  numpy reduces pairwise for
//      length >= 8; at H = 4 it is sequential (measured 0/4000 differing on both
//      axes), so this datapath matches it and does.  But it is NOT a gated claim:
//      measured, a pairwise reduction moves the result by at most 40 ULP while
//      the reciprocal substitution below already moves it up to 49 ULP, so the
//      order is BELOW THE NOISE FLOOR of a divider-free datapath and no tolerance
//      the DUT can meet would separate them.  `INJ_SINK_PAIRWISE` exists to
//      reproduce that measurement and is listed as must-fail NOWHERE.  (Contrast
//      KDA, where the reduction is over 128+ terms and the order is decisive.)
//
// NO DIVIDER.  This repo has no fp32 divide, so each normalise is x * recip(y)
// with glm_fp_recip.vh's Newton reciprocal.  That substitution runs 39 times here
// (156 divisions replaced).  Measured (study Q5, +1 ULP on every reciprocal --
// the worst case the primitive is gated to): `comb` moves by 2.7e-7 and the
// residual is UNCHANGED to four digits.  Four orders below the 2.8e-3 a bf16 map
// would cost, so this is not the binding error term; the softmax's exp is.
//
// NOT BITWISE, BY CONSTRUCTION.  Two reasons, both stated rather than hidden:
// the reciprocal above, and fp32_add being 1 ULP low on ~0.04% of pairs
// (`make fp-ieee`).  The gate is therefore a MEASURED ULP bound, like the rest of
// the fp32 units in this port -- not an equality claim.
//
// COST.  Per pass: (H+1) sum cycles + (RECIP_ITERS+1) Newton cycles + H scale
// cycles = 14 at H=4 (each phase transitions on its own last cycle, so there is no
// extra one), so 39*14 + 2 = 548 cycles per invocation -- pinned to the cycle by
// the TB.  mHC runs twice per block over 45 blocks => ~49k cycles/token of
// Sinkhorn.  The H lanes run
// in parallel because H is a small fixed constant; serialising them would be 1/4
// the adders and 4x the cycles, and that trade is available if area binds.
//============================================================================
`timescale 1ns/1ps
`ifndef MHC_SINKHORN_V
`define MHC_SINKHORN_V
`include "glm_fp.vh"
`include "glm_fp_recip.vh"

module mhc_sinkhorn #(
    parameter integer H           = 4,              // hc_mult
    parameter [31:0]  EPS         = 32'h358637BD,   // 1e-6 fp32
    parameter integer ITERS       = 20,             // hc_sinkhorn_iters (a CONSTANT)
    parameter integer RECIP_ITERS = 4               // glm_fp_recip.vh plateau
)(
    input  wire              clk,
    input  wire              rst,          // sync, active-high
    input  wire              start,
    output reg               busy,
    output reg               done,         // c_out valid

    input  wire [32*H*H-1:0] c_in,         // row-major: c[i][j] at [(i*H+j)*32 +: 32]
    output reg  [32*H*H-1:0] c_out
);
    // ---- must-fail injections (test/mhc_sinkhorn_tb.v); none defined in a build ----
    // Each is one of the four traps in the header, and each must make the gate FAIL.
`ifdef INJ_SINK_SYMM
    localparam integer NPASS = ITERS * 2;                // trap 1: symmetric passes
`else
    localparam integer NPASS = 1 + (ITERS - 1) * 2;      // 39 at ITERS=20
`endif

    localparam [2:0] S_IDLE  = 3'd0,
                     S_SUM   = 3'd1,
                     S_RECIP = 3'd2,
                     S_SCALE = 3'd3,
                     S_FIN   = 3'd4;

    reg [2:0]  st;
    reg [31:0] c   [0:H*H-1];
    reg [31:0] acc [0:H-1];
    reg [31:0] rcp [0:H-1];
    reg [15:0] pass;                                     // 0 .. NPASS-1
    reg [7:0]  t;                                        // step within a phase

`ifdef INJ_SINK_ROWFIRST
    wire axis_row = ~pass[0];                            // trap 2: rows first
`elsif INJ_SINK_SYMM
    wire axis_row = ~pass[0];                            // symmetric schedule starts on rows
`else
    wire axis_row = pass[0];                             // reference: pass 0 is a COLUMN pass
`endif
`ifdef INJ_SINK_PAIRWISE
    reg [31:0] acc2 [0:H-1];                             // trap 4: pairwise partner
`endif

    integer k, i;

    // element index for lane k, step t:
    //   row axis -> lane k is row k,    sum over columns j = t  -> k*H + t
    //   col axis -> lane k is column k, sum over rows    i = t  -> t*H + k
    function integer eidx(input integer lane, input integer step, input integer is_row);
        begin
            eidx = is_row ? (lane * H + step) : (step * H + lane);
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            pass <= 16'd0; t <= 8'd0;
        end else begin
            done <= 1'b0;
            case (st)
                S_IDLE: begin
                    if (start) begin
                        for (i = 0; i < H*H; i = i + 1)
                            c[i] <= c_in[i*32 +: 32];
                        pass <= 16'd0; t <= 8'd0;
                        busy <= 1'b1; st <= S_SUM;
                    end
                end

                // ---- sequential reduction along the current axis, then + EPS ----
                // The phase is H+1 cycles wide in every variant below, so the
                // latency check in the TB stays a clean test of NPASS alone.
                S_SUM: begin
`ifdef INJ_SINK_PAIRWISE
                    for (k = 0; k < H; k = k + 1) begin
                        if (t == 0) begin
                            acc[k]  <= fp32_add(c[eidx(k, 0, axis_row)], c[eidx(k, 1, axis_row)]);
                            acc2[k] <= fp32_add(c[eidx(k, 2, axis_row)], c[eidx(k, 3, axis_row)]);
                        end else if (t == 1) begin
                            acc[k]  <= fp32_add(acc[k], acc2[k]);
                        end else if (t == H) begin
                            acc[k]  <= fp32_add(acc[k], EPS);
                        end
                    end
`else
                    for (k = 0; k < H; k = k + 1) begin
                        if (t == 0)
                            acc[k] <= c[eidx(k, 0, axis_row)];
                        else if (t < H)
                            acc[k] <= fp32_add(acc[k], c[eidx(k, t, axis_row)]);
`ifdef INJ_SINK_NOEPS
                        else
                            acc[k] <= acc[k];                  // trap 3: bare sum
`else
                        else
                            acc[k] <= fp32_add(acc[k], EPS);   // never a bare sum
`endif
                    end
`endif
                    if (t == H) begin t <= 8'd0; st <= S_RECIP; end
                    else               t <= t + 8'd1;
                end

                // ---- Newton reciprocal of (sum + EPS), H lanes in parallel ----
                S_RECIP: begin
                    for (k = 0; k < H; k = k + 1) begin
                        if (t == 0) rcp[k] <= fp32_recip_seed(acc[k]);
                        else        rcp[k] <= fp32_recip_step(acc[k], rcp[k]);
                    end
                    if (t == RECIP_ITERS) begin t <= 8'd0; st <= S_SCALE; end
                    else                        t <= t + 8'd1;
                end

                // ---- x * recip(y) in place ----
                S_SCALE: begin
                    for (k = 0; k < H; k = k + 1)
                        c[eidx(k, t, axis_row)] <= fp32_mul(c[eidx(k, t, axis_row)], rcp[k]);
                    if (t == H - 1) begin
                        t <= 8'd0;
                        if (pass == NPASS - 1) st <= S_FIN;
                        else begin pass <= pass + 16'd1; st <= S_SUM; end
                    end else t <= t + 8'd1;
                end

                S_FIN: begin
                    for (i = 0; i < H*H; i = i + 1)
                        c_out[i*32 +: 32] <= c[i];
                    done <= 1'b1; busy <= 1'b0; st <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // MHC_SINKHORN_V
