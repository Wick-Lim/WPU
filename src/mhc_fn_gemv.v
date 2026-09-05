//============================================================================
// mhc_fn_gemv.v -- the mHC `fn` GEMV: raw residual streams -> the 24 mixed
// logits that src/mhc_map_step.v turns into pre/post/comb.
//
//   flat  = streams.flatten() / sqrt(mean(streams^2) + rms_eps)     [H*D]
//   mixed = hc_{attn,ffn}_fn @ flat                                 [(2+H)*H]
//
// WHY THIS UNIT EXISTS INSTEAD OF glm_matmul_q4k.  The existing GEMM is
// bf16-in / bf16-out (its `a_col` port is 16*PE_M).  Measured, rounding `flat` to
// bf16 before this GEMV costs 2.9e-3 to 6.0e-3 relative on pre/post/comb -- about
// 40x the mHC map's own gated bound, and worse than the bf16 MAP that the fp32
// map was built to replace.  Feeding an fp32 map from bf16 activations would give
// back everything it bought.  So this is an fp32-ACTIVATION x Q8_0-WEIGHT GEMV,
// and it is deliberately a small dedicated unit rather than a widening of
// glm_matmul_q4k: that engine is netlist-pinned and used by every top, and
// widening it for a 24-output GEMV would re-pin baselines across the repo for no
// benefit. (hc_*_fn really is Q8_0 [16384,24] -- read from the GGUF census, not
// inferred; the F32 bucket's size made F32 a tempting and wrong guess.)
//
// THE RMS IS FOLDED PAST THE GEMV, and that is measured too.  `rms` is a SCALAR,
// so fn @ (raw/rms) == (fn @ raw)/rms in real arithmetic; in fp32 the orders
// differ, and the difference is 7.4e-5 on `mixed` but only 3-7e-6 once the
// sigmoid and softmax compress it -- far inside the map's bounds.  The payoff is
// structural: ONE streaming pass over the streams computes both sum(x^2) and all
// 24 dot products, so there is no separate normalise pass and no 64 KB fp32
// `flat` buffer.  mean = sum * (1/K) is exact because K = H*D is a power of two.
//
// REDUCTION ORDER.  Strictly sequential in k, accumulating from +0.0.  The
// reference (tools/glm53_flash_ref.py) reaches `mixed` via numpy's `fn @ flat`,
// i.e. BLAS, whose order is neither sequential nor reproducible; measured, that
// difference is 1.05e-4 on `mixed` and 6e-6 propagated to the map outputs, so the
// order is NOT decisive here and the reference is left alone.
//   KLANES IS DELIBERATELY ABSENT.  Unlike mhc_stream_ops.v, where every d is
// independent and DLANES cannot change the answer, splitting this K-reduction
// across lanes regroups the accumulation and DOES change the result. Widening is
// numerically safe (the measurement above bounds a far larger reordering) but it
// needs its own re-derived golden, so it is a stated follow-up rather than an
// untested parameter. At K = 16384 this unit is 16386 cycles per site.
//============================================================================
`timescale 1ns/1ps
`ifndef MHC_FN_GEMV_V
`define MHC_FN_GEMV_V
`include "glm_fp.vh"
`include "q4k.vh"

module mhc_fn_gemv #(
    parameter integer H       = 4,                 // hc_mult
    parameter integer D       = 64,                // model dim (slice)
    parameter integer QK      = 32,                // Q8_0 block length
    parameter [31:0]  RMS_EPS = 32'h3727C5AC       // 1e-5 fp32
)(
    input  wire                        clk,
    input  wire                        rst,        // sync, active-high
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    input  wire [32*H*D-1:0]           x_in,       // RAW streams, flattened, fp32
    input  wire [8*((2+H)*H)*H*D-1:0]  w_q,        // Q8_0 int8 codes, row-major
    input  wire [16*((2+H)*H)*(H*D/32)-1:0] w_d,   // Q8_0 fp16 scales, per row per block
    output reg  [32*((2+H)*H)-1:0]     mixed_out
);
    localparam integer K     = H*D;
    localparam integer ROWS  = (2+H)*H;
    localparam integer NB    = K/QK;
    localparam integer LOG2K = $clog2(K);
    // 1/K as an exact fp32 power of two -- valid only because K is a power of two,
    // which the elaboration check below enforces rather than assumes.
    localparam [31:0]  INV_K = (127 - LOG2K) << 23;

    localparam [1:0] S_IDLE = 2'd0, S_MAC = 2'd1, S_NORM = 2'd2, S_FIN = 2'd3;

    reg [1:0]  st;
    reg [31:0] acc [0:ROWS-1];
    reg [31:0] sq;
    reg [31:0] rinv;
    reg [31:0] kk;
    integer    r;

    wire [31:0] xk = x_in[32*kk +: 32];

    // dequantised weight for row r at the current k
    function automatic [31:0] wval(input integer row, input integer kidx);
        begin
            wval = fp32_mul(fp16_to_fp32(w_d[16*(row*NB + (kidx/QK)) +: 16]),
                            s8_to_fp32 (w_q [ 8*(row*K  +  kidx)      +:  8]));
        end
    endfunction

`ifndef YOSYS
    initial begin
        if ((1 << LOG2K) != K)
            $fatal(1, "mhc_fn_gemv: H*D must be a power of two for the exact 1/K fold");
    end
`endif

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; kk <= 32'd0; sq <= 32'd0;
        end else begin
            done <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    for (r = 0; r < ROWS; r = r + 1) acc[r] <= 32'd0;
                    sq <= 32'd0; kk <= 32'd0; busy <= 1'b1; st <= S_MAC;
                end

                // one k per cycle: sum(x^2) and all ROWS dot products together
                S_MAC: begin
`ifdef INJ_GEMV_Q8_NOSCALE
                    // must FAIL: int8 codes used without their fp16 block scale
                    for (r = 0; r < ROWS; r = r + 1)
                        acc[r] <= fp32_add(acc[r],
                                    fp32_mul(s8_to_fp32(w_q[8*(r*K + kk) +: 8]), xk));
`else
                    for (r = 0; r < ROWS; r = r + 1)
                        acc[r] <= fp32_add(acc[r], fp32_mul(wval(r, kk), xk));
`endif
                    sq <= fp32_add(sq, fp32_mul(xk, xk));
                    if (kk == K - 1) begin kk <= 32'd0; st <= S_NORM; end
                    else                    kk <= kk + 32'd1;
                end

                S_NORM: begin
`ifdef INJ_GEMV_MEAN_SUM
                    // must FAIL: the bare sum of squares, not the mean
                    rinv <= fp32_rsqrt(fp32_add(sq, RMS_EPS));
`elsif INJ_GEMV_NO_EPS
                    // must FAIL: rms_eps dropped
                    rinv <= fp32_rsqrt(fp32_mul(sq, INV_K));
`else
                    rinv <= fp32_rsqrt(fp32_add(fp32_mul(sq, INV_K), RMS_EPS));
`endif
                    st <= S_FIN;
                end

                S_FIN: begin
                    for (r = 0; r < ROWS; r = r + 1)
                        mixed_out[32*r +: 32] <= fp32_mul(acc[r], rinv);
                    done <= 1'b1; busy <= 1'b0; st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // MHC_FN_GEMV_V
