//============================================================================
// fp32_sigmoid_pipe.v -- fp32 sigmoid, sigma(x) = 1 / (1 + exp(-x)).
//
// WHY THIS EXISTS.  Until now the repo's only sigmoid was glm_act (MODE 0):
// bf16 in / bf16 out, polynomial exp, input railed at +/-16.  Three separate
// GLM-5.3-Flash paths turned out to be limited by it, each measured:
//   * KDA forget gate / beta  -- 1.24 % / 1.34 % relative error on the two values
//     the recurrence consumes, and it CANNOT reach the reference's exact-0
//     saturation: railed at -16, sigma(-16)=1.13e-7 is representable in bf16, so
//     g = -5*sigma lands at -5.63e-7 instead of -0.0   (ledger 4.3e)
//   * KDA o_norm gamma_eff = weight*sigma(gate)         (ledger 4.3f)
//   * mHC pre = sigma(.) + 1e-6 -- bf16 cannot represent 1 + 1e-6 at all, and
//     the mHC study concluded the whole map must be fp32   (ledger 4.3i)
//
// STRUCTURE.
//     e = exp(-x)                            fp32_exp_pipe (LAT_EXP)
//     y = 1 + e                              one fp32_add        (1 stage)
//     r = 1/y                                Newton, RECIP_ITERS stages
//     out = mux(saturation, r)               (1 stage)
//   LAT = LAT_EXP + 3 + RECIP_ITERS, exposed as the LAT parameter: the exp pipe,
//   then stage A, then RECIP_ITERS+1 Newton stages (the seed occupies one), then
//   the output mux -- 53 at RECIP_ITERS=4, VERIFIED by measuring valid_in ->
//   valid_out rather than counted by eye.  (It read `+ 2` until 2026-09-04, off by
//   one; mhc_map_step.v schedules on this number, which is how that surfaced.)
//   One Newton iteration per pipeline stage: each is 3 dependent fp ops, so
//   folding all four into one combinational block would be a 12-op path.
//
// SATURATION -- the property that makes this worth building.  fp32_exp_pipe
// FLUSHES SUBNORMALS TO ZERO and overflows to +inf (measured):
//     x >~ +88  ->  exp(-x) = 0     ->  y = 1      ->  sigma = EXACTLY 1.0
//     x <~ -88  ->  exp(-x) huge/inf ->              sigma = EXACTLY 0.0
// Both are what the reference does in fp32, and neither is reachable in bf16.
// sigma = 1.0 exactly is what lets mHC form `pre = sigma + 1e-6` = 1.000001,
// representable in fp32; sigma = 0.0 exactly is what lets the KDA forget gate
// produce -5.0 * +0.0 = -0.0, the signed-zero contract the golden pins.
// +inf from the exp is muxed out BEFORE the Newton chain (the seed subtraction
// would wrap on it); a subnormal reciprocal simply flushes to 0, which is the
// same answer to within 6e-39.
//
// ACCURACY: fp32_exp_pipe is a degree-4 Horner (measured ~10 ULP in the normal
// range) and the Newton reciprocal is within 1 ULP, so this is NOT bit-exact --
// it is a fp32-precision sigmoid, gated to a MEASURED ULP bound by
// `make fp-sigmoid`.  Against the ~1.2e-2 relative error of the bf16 path it
// replaces, that is four orders of magnitude.
//============================================================================
`timescale 1ns/1ps
`ifndef FP32_SIGMOID_PIPE_V
`define FP32_SIGMOID_PIPE_V
`include "glm_fp.vh"
`include "glm_fp_recip.vh"
`include "glm_fp_pipe_lat.vh"

module fp32_sigmoid_pipe #(
    parameter integer RECIP_ITERS = 4     // measured plateau; see glm_fp_recip.vh
)(
    input  wire        clk,
    input  wire        rst,               // sync, active-high
    input  wire        valid_in,
    input  wire [31:0] x,
    output wire        valid_out,
    output wire [31:0] result
);
    localparam integer LAT_EXP = `FP_EXP_LAT;
    localparam integer LAT     = LAT_EXP + 3 + RECIP_ITERS;   // exposed for callers; measured

    // ---- exp(-x) ----
    wire [31:0] negx = {~x[31], x[30:0]};
    wire        e_ov;
    wire [31:0] e_r;
    fp32_exp_pipe u_exp (.clk(clk), .rst(rst), .valid_in(valid_in), .x(negx),
                         .valid_out(e_ov), .result(e_r));

    // ---- stage A: y = 1 + e, and classify the saturation cases ----
    // e == +inf  -> sigma = 0   (x very negative)
    // e == 0     -> sigma = 1   (x very positive; y would be exactly 1 anyway,
    //                            but muxing it is free and documents the intent)
    reg        a_v;
    reg [31:0] a_y;
    reg [1:0]  a_sat;                       // 0 = normal, 1 = force 0.0, 2 = force 1.0
    always @(posedge clk) begin
        if (rst) begin a_v <= 1'b0; a_sat <= 2'd0; a_y <= 32'h3F800000; end
        else begin
            a_v <= e_ov;
            if ((e_r[30:23] == 8'hFF) && (e_r[22:0] == 23'd0)) begin
                a_sat <= 2'd1; a_y <= 32'h3F800000;             // +inf -> sigma 0
            end else if (e_r[30:0] == 31'd0) begin
                a_sat <= 2'd2; a_y <= 32'h3F800000;             // 0    -> sigma 1
            end else if (fp32_add(32'h3F800000, e_r) == 32'h3F800000) begin
                // 1 + e rounded to EXACTLY 1.0 (any e below ~eps/2, i.e. x >~ 17):
                // the correctly-rounded sigma is then exactly 1.0, and forcing it
                // is exact.  Letting Newton run on y = 1.0 instead lands 1 ULP low
                // (measured 0x3F7FFFFF at x = 21.6) -- which would break the mHC
                // caller, whose whole requirement is that sigma + 1e-6 differs
                // from 1.0.  This mux is the fix, and it costs one comparator.
                a_sat <= 2'd2; a_y <= 32'h3F800000;
            end else begin
                a_sat <= 2'd0; a_y <= fp32_add(32'h3F800000, e_r);
            end
        end
    end

    // ---- Newton stages: one iteration per stage, y carried alongside ----
    reg [31:0] n_r   [0:RECIP_ITERS];
    reg [31:0] n_y   [0:RECIP_ITERS];
    reg [1:0]  n_sat [0:RECIP_ITERS];
    reg        n_v   [0:RECIP_ITERS];
    integer s;
    always @(posedge clk) begin
        if (rst) begin
            for (s = 0; s <= RECIP_ITERS; s = s + 1) begin
                n_v[s] <= 1'b0; n_sat[s] <= 2'd0;
                n_r[s] <= 32'h00000000; n_y[s] <= 32'h3F800000;
            end
        end else begin
            n_v[0]   <= a_v;
            n_y[0]   <= a_y;
            n_sat[0] <= a_sat;
            n_r[0]   <= fp32_recip_seed(a_y);
            for (s = 1; s <= RECIP_ITERS; s = s + 1) begin
                n_v[s]   <= n_v[s-1];
                n_y[s]   <= n_y[s-1];
                n_sat[s] <= n_sat[s-1];
                n_r[s]   <= fp32_recip_step(n_y[s-1], n_r[s-1]);
            end
        end
    end

    // ---- output stage: apply the saturation mux ----
    reg        o_v;
    reg [31:0] o_r;
    always @(posedge clk) begin
        if (rst) begin o_v <= 1'b0; o_r <= 32'h00000000; end
        else begin
            o_v <= n_v[RECIP_ITERS];
            case (n_sat[RECIP_ITERS])
                2'd1:    o_r <= 32'h00000000;      // exactly 0.0
                2'd2:    o_r <= 32'h3F800000;      // exactly 1.0
                default: o_r <= n_r[RECIP_ITERS];
            endcase
        end
    end

    assign valid_out = o_v;
    assign result    = o_r;
endmodule
`endif // FP32_SIGMOID_PIPE_V
