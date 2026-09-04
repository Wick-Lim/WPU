//============================================================================
// glm_fp_recip.vh -- fp32 reciprocal by Newton-Raphson.
//
// WHY A NEW FILE.  This repo has fp32 mul/add/rsqrt/exp and NO divide.  The
// GLM-5.3-Flash port needs one: sigmoid(x) = 1/(1+exp(-x)), and the mHC
// precision study (docs/GLM53_FLASH_PORT.md 4.3i) showed that path must be fp32.
// It lives in its own header rather than in glm_fp.vh so that adding it cannot
// perturb any module that includes glm_fp.vh -- every pinned netlist baseline in
// the repo depends on that file being untouched.
//
// METHOD.  r <- r * (2 - y*r), seeded by the classic exponent-negation trick
//     r0 = 0x7EF311C3 - y_bits                 (~8 correct bits)
// Newton roughly doubles the correct bits per iteration: 8 -> 16 -> 32.
//
// MEASURED (4003 vectors: y in [1,2], [1,1e3] and exp(0..80) -- the range
// 1+exp(-x) actually spans -- against numpy fp32 1.0/y):
//     iters=1   3995/4003 off, worst 42804 ULP
//     iters=2   3523/4003 off, worst   109 ULP
//     iters=3   1687/4003 off, worst     2 ULP
//     iters=4   1251/4003 off, worst     1 ULP     <- plateau
//     iters=5   1251/4003 off, worst     1 ULP
// So RECIP_ITERS = 4 is the point where more iterations stop buying anything.
// It is NOT bit-exact: Newton converges to within 1 ULP but is not correctly
// rounded, and fp32_add itself is 1 ULP low on ~0.04% of pairs (`make fp-ieee`).
// A 1-ULP reciprocal is ~1e-7 relative -- against the 1.2e-2 the bf16 glm_act
// sigmoid costs on the same paths, which is the whole point of building this.
//
// DOMAIN.  Callers must pass y > 0 and finite.  y = +inf and y = 0 are NOT
// handled here (the seed subtraction wraps); fp32_sigmoid_pipe muxes those cases
// out before the Newton chain.
//============================================================================
`ifndef GLM_FP_RECIP_VH
`define GLM_FP_RECIP_VH
`include "glm_fp.vh"

// One Newton step, exposed so a pipelined caller can put each in its own stage.
function automatic [31:0] fp32_recip_step(input [31:0] y, input [31:0] r);
    reg [31:0] t;
    begin
        t = fp32_mul(y, r);                                    // y*r
        t = fp32_add(32'h40000000, {~t[31], t[30:0]});         // 2 - y*r
        fp32_recip_step = fp32_mul(r, t);
    end
endfunction

// ~8-bit seed. Valid for finite y > 0; see the DOMAIN note above.
function automatic [31:0] fp32_recip_seed(input [31:0] y);
    begin
        fp32_recip_seed = 32'h7EF311C3 - y;
    end
endfunction

// Combinational reciprocal (deep: ITERS * 3 fp ops). A pipelined caller should
// use the seed + step functions directly, one step per stage.
function automatic [31:0] fp32_recip(input [31:0] y, input integer iters);
    reg [31:0] r;
    integer i;
    begin
        r = fp32_recip_seed(y);
        for (i = 0; i < iters; i = i + 1) r = fp32_recip_step(y, r);
        fp32_recip = r;
    end
endfunction

`endif // GLM_FP_RECIP_VH
