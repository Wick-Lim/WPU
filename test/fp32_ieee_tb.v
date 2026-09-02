//============================================================================
// fp32_ieee_tb.v -- measure and PIN how far src/glm_fp.vh's fp32_add / fp32_mul
// are from IEEE round-to-nearest-even.
//
// fp32_add is not exactly IEEE: ~0.04% of random pairs come out 1 ULP low,
// concentrated at exponent gaps 4-5. That has been harmless because every proven
// path in this repo ends in bf16, where a 1-ULP fp32 difference survives rounding
// in ~0.001% of cases. src/kda_recur.v is the first fp32-OUTPUT consumer, so it
// is the first place the gap shows.
//
// This gate turns that from a latent surprise into a tracked number. TB_MAX_ADD
// is a CEILING in parts-per-10000, not a target: an exactly-rounded adder would
// score 0 and still pass.
//============================================================================
`timescale 1ns/1ps
`include "glm_fp.vh"
`ifndef TB_VEC
    `define TB_VEC "build/fp32_ieee_vec.txt"
`endif
`ifndef TB_MAX_ADD
    `define TB_MAX_ADD 10      // parts per 10000; measured 4
`endif
`ifndef TB_MAX_MUL
    `define TB_MAX_MUL 0       // fp32_mul measured exactly conformant
`endif

module fp32_ieee_tb;
    integer fd, code, n, bad_a, bad_m, ppm_a, ppm_m;
    reg [31:0] a, c, ea, em;
    initial begin
        n = 0; bad_a = 0; bad_m = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[fp32_ieee] FAIL: cannot open %s", `TB_VEC); $finish; end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%h %h %h %h", a, c, ea, em);
            if (code == 4) begin
                n = n + 1;
                if (fp32_add(a, c) !== ea) bad_a = bad_a + 1;
                if (fp32_mul(a, c) !== em) bad_m = bad_m + 1;
            end
        end
        $fclose(fd);
        ppm_a = (bad_a * 10000) / n;
        ppm_m = (bad_m * 10000) / n;
        $display("[fp32_ieee] fp32_add non-conformance %0d/%0d (%0d ppt10k, ceiling %0d); fp32_mul %0d/%0d (%0d ppt10k, ceiling %0d)",
                 bad_a, n, ppm_a, `TB_MAX_ADD, bad_m, n, ppm_m, `TB_MAX_MUL);
        if (ppm_a > `TB_MAX_ADD || ppm_m > `TB_MAX_MUL)
            $display("[fp32_ieee] FAILED: a primitive regressed past its pinned IEEE-conformance ceiling");
        else
            $display("[fp32_ieee] ALL %0d TESTS PASSED (fp32_add within its pinned 1-ULP non-conformance ceiling; fp32_mul exactly IEEE)", n);
        $finish;
    end
endmodule
