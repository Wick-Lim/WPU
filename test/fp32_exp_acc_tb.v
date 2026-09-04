//============================================================================
// fp32_exp_acc_tb.v -- measure and PIN fp32_exp_pipe's distance from a
// correctly-rounded exp over x in [-40, 40].
//
// Measured 1899 ULP = 2.3e-4 relative.  This is the precision ceiling for the
// fp32 sigmoid and for the mHC map (whose softmax is an exp), so it is pinned
// here rather than rediscovered downstream.  TB_MAX_ULP is a CEILING, not a
// target: a better polynomial would score lower and still pass.
// Subnormal / inf goldens are skipped -- the pipe is FTZ by construction and
// ULP is not a meaningful metric there (see fp32_sigmoid_tb's FTZ leg).
//============================================================================
`timescale 1ns/1ps
`include "glm_fp.vh"
`ifndef TB_VEC
    `define TB_VEC "build/fp32_exp_acc_vec.txt"
`endif
`ifndef TB_MAX_ULP
    `define TB_MAX_ULP 4096
`endif

module fp32_exp_acc_tb;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, vin = 0;
    reg  [31:0] x;
    wire        vo;
    wire [31:0] r;
    fp32_exp_pipe dut (.clk(clk), .rst(rst), .valid_in(vin), .x(x), .valid_out(vo), .result(r));

    integer fd, code, i, k, n, cnt, d, mx, errors;
    reg [31:0] xv, ev;
    initial begin
        cnt = 0; mx = 0; errors = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[fp32_exp_acc] FAIL: cannot open %s", `TB_VEC); $finish; end
        code = $fscanf(fd, "%d", n);
        @(negedge clk); rst = 0;
        for (i = 0; i < n; i = i + 1) begin
            code = $fscanf(fd, "%h %h", xv, ev);
            @(negedge clk); x = xv; vin = 1;
            @(negedge clk); vin = 0;
            k = 0;
            while (vo !== 1'b1 && k < 80) begin @(negedge clk); k = k + 1; end
            if (ev[30:23] != 8'd0 && ev[30:23] != 8'hFF && r[30:23] != 8'd0) begin
                cnt = cnt + 1;
                d = $signed({1'b0, r[30:0]}) - $signed({1'b0, ev[30:0]});
                if (d < 0) d = -d;
                if (d > mx) mx = d;
            end
        end
        $fclose(fd);
        if (cnt == 0) begin
            $display("[fp32_exp_acc] FAIL: no normal-range points compared -- the corpus is vacuous");
            errors = errors + 1;
        end
        if (mx > `TB_MAX_ULP) begin
            $display("[fp32_exp_acc] FAIL: worst %0d ULP exceeds the pinned ceiling %0d", mx, `TB_MAX_ULP);
            errors = errors + 1;
        end
        if (errors == 0)
            $display("[fp32_exp_acc] ALL %0d TESTS PASSED (fp32_exp_pipe vs correctly-rounded exp over x in [-40,40]: worst %0d ULP, ceiling %0d -- this is the precision ceiling for the fp32 sigmoid and the mHC softmax)",
                     cnt, mx, `TB_MAX_ULP);
        else
            $display("[fp32_exp_acc] FAILED (worst %0d ULP)", mx);
        $finish;
    end
    initial begin #1500000000; $display("[fp32_exp_acc] FAIL: timeout"); $finish; end
endmodule
