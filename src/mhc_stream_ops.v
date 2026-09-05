//============================================================================
// mhc_stream_ops.v -- the two D-wide datapaths of the mHC residual path.
//
// This is what makes a block carry FOUR residual streams instead of one:
//
//   COLLAPSE (mode 0)   collapsed[d] = sum_h pre[h] * streams[h][d]
//                       -> the sublayer's input, one [D] vector.
//   MIX      (mode 1)   streams'[h][d] = sum_g comb[h][g]*streams[g][d]
//                                        + post[h]*sub_out[d]
//                       -> the residual update, back to [H,D].
//
// WHERE THIS SITS.  mHC WRAPS a sublayer; it does not change the sublayer's
// contract.  The GGUF census settles that: attn_norm[4096] and ffn_norm[4096]
// exist on all 46 blocks ALONGSIDE the hc_* tensors, so `collapsed` is NOT
// normalised here -- the block's own learned norm still applies to it, and
// attention / FFN still see [D] in and [D] out.  That is why the KDA layer
// wrapper and this path do not block each other.
//
// REDUCTION ORDER IS THE TRAP.  numpy's `comb @ streams` does NOT reduce in this
// order -- measured 300/300 differing on [4,4]@[4,D].  The gap is one fp32
// rounding (5.0e-7 of the output RMS; it is an FMA-vs-mul-then-add difference),
// but it is not zero, so a golden built on `@` would depend on whichever BLAS the
// host links.  tools/glm53_flash_ref.py therefore pins the order in `hc_mix` --
// multiply, accumulate sequentially over g, add the post term LAST -- and its
// self-test asserts numpy does NOT match, so the pin stays live.  This datapath
// reproduces that order exactly, one accumulation per cycle.
//   Beware the metric when checking any of this: comb is doubly stochastic, so the
// four terms nearly cancel and some outputs land near zero.  A plain relative
// error divides by those and reports ~1e-2 for a 1-ULP difference.  Compare
// against the output RMS.
//
// fp32 THROUGHOUT, and that is a measured choice, not an inherited one.  This
// repo's residual stream is bf16.  Carrying the four mHC streams in bf16 instead
// costs 5.9e-3 RMS relative after 90 mHC sites -- about one bf16 rounding, so the
// contractive mix does not compound it, but it is 3-4x what a bf16 MAP costs and
// ~5x the fp32 map's own comb error.  The storage argument that would justify it
// does not exist: the streams are ONE running [H,D] buffer for the whole model,
// 64 KB at H=4, D=4096.  So fp32.
//
// DLANES is a throughput knob and MUST NOT change the answer -- every d is
// independent, so the TB runs the same vectors at DLANES 1, 2 and 4 and requires
// bit-identical results.  Cost per site at H=4, D=4096:
//   COLLAPSE ceil(D/DLANES)*H + MIX ceil(D/DLANES)*(H+1) = 36,864/DLANES cycles.
// At DLANES=8 that is 4,608/site, 415k cycles/token over 90 sites -- 24% of an
// HBM4 token, so DLANES is sized against the tier, not left at 1.
//============================================================================
`timescale 1ns/1ps
`ifndef MHC_STREAM_OPS_V
`define MHC_STREAM_OPS_V
`include "glm_fp.vh"

module mhc_stream_ops #(
    parameter integer H      = 4,      // hc_mult: parallel residual streams
    parameter integer D      = 64,     // model dim (slice)
    parameter integer DLANES = 1       // d-elements per cycle; must not change the answer
)(
    input  wire                clk,
    input  wire                rst,          // sync, active-high
    input  wire                start,
    input  wire                mode,         // 0 = COLLAPSE, 1 = MIX
    output reg                 busy,
    output reg                 done,

    input  wire [32*H*D-1:0]   streams_in,   // [h][d] at [(h*D+d)*32 +: 32], fp32
    input  wire [32*H-1:0]     pre_in,       // COLLAPSE weights
    input  wire [32*H*H-1:0]   comb_in,      // MIX matrix, comb[h][g] at [(h*H+g)*32 +: 32]
    input  wire [32*H-1:0]     post_in,      // MIX sublayer placement
    input  wire [32*D-1:0]     sub_in,       // MIX sublayer output

    output reg  [32*D-1:0]     collapsed_out,
    output reg  [32*H*D-1:0]   streams_out
);
    localparam integer NG = (D + DLANES - 1) / DLANES;   // d-groups

    localparam [1:0] S_IDLE = 2'd0, S_ACC = 2'd1, S_WB = 2'd2, S_FIN = 2'd3;

    reg [1:0]  st;
    reg        md;
    reg [31:0] acc [0:H*DLANES-1];        // MIX: one accumulator per (out h, lane)
    reg [15:0] g;                          // d-group index
    reg [7:0]  t;                          // accumulation step
    integer    h, j, dd, tt;   // tt: integer copy of t, so index arithmetic
                               // like t*D is done in integer width, not 8-bit

    // d index for lane j of the current group; groups past D are masked at write-back
    function integer didx(input integer grp, input integer lane);
        begin didx = grp * DLANES + lane; end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; g <= 16'd0; t <= 8'd0; md <= 1'b0;
        end else begin
            done <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    md <= mode; g <= 16'd0; t <= 8'd0; busy <= 1'b1; st <= S_ACC;
                end

                // one term per cycle, in the pinned sequential order
                S_ACC: begin
                    tt = t;
                    for (j = 0; j < DLANES; j = j + 1) begin
                        dd = didx(g, j);
                        if (dd < D) begin
                            if (!md) begin
                                // COLLAPSE: acc[j] over h = 0..H-1
`ifdef INJ_OPS_COLLAPSE_NOPRE
                                // must FAIL: an unweighted sum of the streams
                                if (t == 0) acc[j] <= streams_in[32*(0*D+dd) +: 32];
                                else        acc[j] <= fp32_add(acc[j],
                                                        streams_in[32*(tt*D+dd) +: 32]);
`else
                                if (t == 0)
                                    acc[j] <= fp32_mul(pre_in[32*0 +: 32],
                                                       streams_in[32*(0*D+dd) +: 32]);
                                else
                                    acc[j] <= fp32_add(acc[j],
                                                fp32_mul(pre_in[32*tt +: 32],
                                                         streams_in[32*(tt*D+dd) +: 32]));
`endif
                            end else begin
                                // MIX: H accumulators per lane, over g = 0..H-1, post LAST
                                for (h = 0; h < H; h = h + 1) begin
`ifdef INJ_OPS_POST_FIRST
                                    // must FAIL: the post term added FIRST, not last.
                                    // Pure rounding-order change -- the values are the
                                    // same, the fp32 result is not (120/120 in the
                                    // generator's self-test).
                                    if (t == 0)
                                        acc[h*DLANES+j] <= fp32_mul(post_in[32*h +: 32],
                                                                    sub_in[32*dd +: 32]);
                                    else
                                        acc[h*DLANES+j] <= fp32_add(acc[h*DLANES+j],
                                            fp32_mul(comb_in[32*(h*H+(tt-1)) +: 32],
                                                     streams_in[32*((tt-1)*D+dd) +: 32]));
`else
                                    if (t == 0)
`ifdef INJ_OPS_MIX_TRANSPOSE
                                        acc[h*DLANES+j] <= fp32_mul(comb_in[32*(0*H+h) +: 32],
                                                            streams_in[32*(0*D+dd) +: 32]);
`else
                                        acc[h*DLANES+j] <= fp32_mul(comb_in[32*(h*H+0) +: 32],
                                                            streams_in[32*(0*D+dd) +: 32]);
`endif
                                    else if (t < H)
`ifdef INJ_OPS_MIX_TRANSPOSE
                                        acc[h*DLANES+j] <= fp32_add(acc[h*DLANES+j],
                                            fp32_mul(comb_in[32*(tt*H+h) +: 32],
                                                     streams_in[32*(tt*D+dd) +: 32]));
`else
                                        acc[h*DLANES+j] <= fp32_add(acc[h*DLANES+j],
                                            fp32_mul(comb_in[32*(h*H+tt) +: 32],
                                                     streams_in[32*(tt*D+dd) +: 32]));
`endif
                                    else
`ifdef INJ_OPS_MIX_NOPOST
                                        acc[h*DLANES+j] <= acc[h*DLANES+j];
`else
                                        acc[h*DLANES+j] <= fp32_add(acc[h*DLANES+j],
                                            fp32_mul(post_in[32*h +: 32], sub_in[32*dd +: 32]));
`endif
`endif
                                end
                            end
                        end
                    end
                    // COLLAPSE needs H terms; MIX needs H + the post term
                    if (t == (md ? H : H - 1)) begin t <= 8'd0; st <= S_WB; end
                    else                             t <= t + 8'd1;
                end

                S_WB: begin
                    for (j = 0; j < DLANES; j = j + 1) begin
                        dd = didx(g, j);
                        if (dd < D) begin
                            if (!md) collapsed_out[32*dd +: 32] <= acc[j];
                            else for (h = 0; h < H; h = h + 1)
                                streams_out[32*(h*D+dd) +: 32] <= acc[h*DLANES+j];
                        end
                    end
                    if (g == NG - 1) st <= S_FIN;
                    else begin g <= g + 16'd1; st <= S_ACC; end
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // MHC_STREAM_OPS_V
