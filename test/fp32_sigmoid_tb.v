//============================================================================
// fp32_sigmoid_tb.v -- gate for src/fp32_sigmoid_pipe.v.
//
// Three legs:
//   saturation-to-1  BITWISE  where the golden is exactly 1.0  (mHC needs this:
//                             pre = sigma + 1e-6 must be 1.000001, not 1.0)
//   saturation-to-0  BITWISE  where the golden is exactly 0.0  (the KDA forget
//                             gate needs -5.0 * sigma == -0.0)
//   SUBNORMAL golden FLUSH   where the correctly-rounded sigma is subnormal (a
//                    band around x in [-88, -103]), this repo's fp32 pipes are
//                    FTZ by construction -- fp32_exp_pipe's header says so -- and
//                    the DUT returns exactly 0.0.  That is a stated property, not
//                    an error, so it is checked AS a flush and the largest golden
//                    that gets flushed is REPORTED, which is where the flush
//                    boundary actually is.  Scoring these by ULP would be
//                    meaningless: one ULP is ~1e-45 there, so a 9e-41 absolute
//                    difference reads as 63476 ULP.
//   everywhere else  ULP bound, reported.  exp is ~10 ULP + Newton <=1 ULP, so
//                    this cannot be bitwise; TB_ULP is the measured envelope.
// A unit accurate in the middle but unable to saturate would pass a pure
// tolerance gate and still be useless for both callers -- hence two bitwise legs
// and a vacuity check on each.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/fp32_sigmoid_vec.txt"
`endif
// Measured envelope on the committed corpus: worst 790 ULP (~9.4e-5 relative).
// Pinned at 1024 for headroom. The dominant term is NOT this module: fp32_exp_pipe
// itself is 1899 ULP (2.3e-4) over the same range (`make fp-sigmoid` measures that
// separately), and sigma = 1/(1+e) COMPRESSES it, since d(sigma)/sigma =
// -(e/(1+e)) * de/e and e/(1+e) < 1. So the sigmoid is more accurate than the exp
// it is built on, and improving it means improving the exp.
`ifndef TB_ULP
    `define TB_ULP 1024
`endif

module fp32_sigmoid_tb;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, vin = 0;
    reg  [31:0] x;
    wire        vo;
    wire [31:0] r;
    fp32_sigmoid_pipe dut (.clk(clk), .rst(rst), .valid_in(vin), .x(x), .valid_out(vo), .result(r));

    integer fd, code, i, k, n, errors, checks, n_one, n_zero, n_sub, d, maxulp;
    reg [31:0] xv, ev, max_flushed;

    initial begin
        errors = 0; checks = 0; n_one = 0; n_zero = 0; n_sub = 0; maxulp = 0; max_flushed = 32'h00000000;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[fp32_sigmoid] FAIL: cannot open %s", `TB_VEC); $finish; end
        code = $fscanf(fd, "%d", n);
        @(negedge clk); rst = 0;
        for (i = 0; i < n; i = i + 1) begin
            code = $fscanf(fd, "%h %h", xv, ev);
            @(negedge clk); x = xv; vin = 1;
            @(negedge clk); vin = 0;
            k = 0;
            while (vo !== 1'b1 && k < 80) begin @(negedge clk); k = k + 1; end
            if (vo !== 1'b1) begin
                $display("[fp32_sigmoid] FAIL i=%0d: no valid_out", i); errors = errors + 1;
            end
            checks = checks + 1;
            if (ev == 32'h3F800000) begin
                n_one = n_one + 1;
                if (r !== 32'h3F800000) begin
                    $display("FAIL i=%0d x=%h: golden is exactly 1.0, got %h (mHC needs sigma+1e-6 != 1.0)", i, xv, r);
                    errors = errors + 1;
                end
            end else if (ev == 32'h00000000) begin
                n_zero = n_zero + 1;
                if (r !== 32'h00000000) begin
                    $display("FAIL i=%0d x=%h: golden is exactly 0.0, got %h (KDA gate needs -5.0*sigma == -0.0)", i, xv, r);
                    errors = errors + 1;
                end
            end else if (ev[30:23] == 8'd0) begin
                // correctly-rounded sigma is SUBNORMAL: this repo's fp32 pipes are
                // FTZ, so the DUT must return exactly 0.0. Record the largest such
                // golden -- that is the measured flush boundary.
                n_sub = n_sub + 1;
                if (ev[22:0] > max_flushed[22:0]) max_flushed = ev;
                if (r !== 32'h00000000) begin
                    $display("FAIL i=%0d x=%h: golden %h is subnormal, FTZ requires exactly 0.0, got %h", i, xv, ev, r);
                    errors = errors + 1;
                end
            end else begin
                if (r[31] !== ev[31]) d = 1000000;
                else begin
                    d = $signed({1'b0, r[30:0]}) - $signed({1'b0, ev[30:0]});
                    if (d < 0) d = -d;
                end
                if (d > maxulp) maxulp = d;
                if (d > `TB_ULP) begin
                    $display("FAIL i=%0d x=%h: got %h exp %h (%0d ULP > %0d)", i, xv, r, ev, d, `TB_ULP);
                    errors = errors + 1;
                end
            end
        end
        $fclose(fd);
        if (n_one == 0 || n_zero == 0) begin
            $display("[fp32_sigmoid] FAIL: corpus never saturated (to 1.0: %0d, to 0.0: %0d) -- the bitwise legs are vacuous", n_one, n_zero);
            errors = errors + 1;
        end
        if (errors == 0)
            $display("[fp32_sigmoid] ALL %0d TESTS PASSED (%0d bitwise saturation-to-1.0, %0d bitwise saturation-to-0.0 -- neither reachable in bf16; %0d subnormal goldens correctly FTZ-flushed, largest flushed %h; %0d normal points within %0d ULP, worst observed %0d)",
                     checks, n_one, n_zero, n_sub, max_flushed, checks-n_one-n_zero-n_sub, `TB_ULP, maxulp);
        else
            $display("[fp32_sigmoid] %0d/%0d FAILED (worst %0d ULP)", errors, checks, maxulp);
        $finish;
    end
    initial begin #20000000; $display("[fp32_sigmoid] FAIL: timeout"); $finish; end
endmodule
