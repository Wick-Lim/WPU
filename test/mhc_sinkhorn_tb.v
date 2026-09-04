//============================================================================
// mhc_sinkhorn_tb.v -- gate for the mHC Sinkhorn projection
// (src/mhc_sinkhorn.v, vectors from tools/mhc_sinkhorn_gen.py).
//
// TOLERANCE leg by construction, not by convenience: the DUT has no divider
// (x*recip(y), glm_fp_recip.vh) and fp32_add is 1 ULP low on ~0.04% of pairs
// (`make fp-ieee`), so bitwise equality with the fp32 true-division reference is
// not available.  The generator EMULATES the DUT exactly and measured the gap at
// 12 ULP / 1.8e-7 over this corpus; TB_ULP is 64, ~5x headroom, and the TB prints
// the worst it saw so a regression moves a number.
//
// The latency check is a real test, not bookkeeping: `done` must arrive at
//     NPASS*(H+1 + RECIP_ITERS+1 + H) + 2
// cycles, which pins NPASS = 1 + (ITERS-1)*2 = 39.  A symmetric 40-pass schedule
// fails here as well as numerically.
//
// Must FAIL: -DINJ_SINK_SYMM, -DINJ_SINK_ROWFIRST, -DINJ_SINK_NOEPS.
// -DINJ_SINK_PAIRWISE is NOT in that list and must not be added: measured, the
// reduction order moves the result by <=40 ULP against the reciprocal
// substitution's <=49 ULP, so at H=4 it is unobservable here by construction.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/mhc_sinkhorn_vec.txt"
`endif
`ifndef TB_ULP
    `define TB_ULP 64
`endif

module mhc_sinkhorn_tb;
    localparam integer H = 4, ITERS = 20, RECIP_ITERS = 4;
    localparam integer NN = H*H;
    localparam integer NPASS   = 1 + (ITERS-1)*2;
    localparam integer EXP_LAT = NPASS*((H+1) + (RECIP_ITERS+1) + H) + 2;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    reg  [32*NN-1:0] c_in;
    wire [32*NN-1:0] c_out;

    mhc_sinkhorn #(.H(H), .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .c_in(c_in), .c_out(c_out));

    integer fd, code, t, i, ntest, hf, itf, errors, checks, lat, worst_ulp;
    integer d;
    reg [31:0] t32;
    reg [31:0] exp_c [0:NN-1];

    // fp32 ULP distance. Every comb entry is strictly positive (softmax + eps
    // scaled by positive reciprocals), so the bit patterns are monotone in value
    // and a plain integer difference IS the ULP distance. A sign flip would mean
    // something structural broke, so it is reported as unbounded rather than
    // silently folded into a small number.
    function integer ulp(input [31:0] a, input [31:0] b);
        begin
            if (a[31] !== b[31]) ulp = 1000000000;
            else if (a[30:0] >= b[30:0]) ulp = a[30:0] - b[30:0];
            else                          ulp = b[30:0] - a[30:0];
        end
    endfunction

    initial begin
        errors = 0; checks = 0; worst_ulp = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mhc_sinkhorn] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, itf);
        if (hf != H || itf != ITERS) begin
            $display("[mhc_sinkhorn] FAIL: vector H/ITERS %0d/%0d != TB %0d/%0d", hf, itf, H, ITERS);
            $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < NN; i = i + 1) begin code = $fscanf(fd, "%h", t32); c_in[32*i +: 32] = t32; end
            for (i = 0; i < NN; i = i + 1) begin code = $fscanf(fd, "%h", t32); exp_c[i] = t32; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            lat = 1;
            while (done !== 1'b1 && lat < 4*EXP_LAT) begin @(negedge clk); lat = lat + 1; end

            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[mhc_sinkhorn] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end else if (lat != EXP_LAT) begin
                $display("[mhc_sinkhorn] FAIL t%0d: done at %0d cycles, expected %0d (NPASS=%0d)",
                         t, lat, EXP_LAT, NPASS);
                errors = errors + 1;
            end

            for (i = 0; i < NN; i = i + 1) begin
                checks = checks + 1;
                d = ulp(c_out[32*i +: 32], exp_c[i]);
                if (d > worst_ulp) worst_ulp = d;
                if (d > `TB_ULP) begin
                    $display("FAIL t%0d c[%0d][%0d]: got %h exp %h (%0d ULP > %0d)",
                             t, i/H, i%H, c_out[32*i +: 32], exp_c[i], d, `TB_ULP);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mhc_sinkhorn] ALL %0d TESTS PASSED (%0d matrices H=%0d, %0d Sinkhorn iters = %0d passes, done at %0d cycles: entries within %0d ULP of the fp32 true-division reference -- worst %0d; x*recip(y) leg, no divider)",
                     checks, ntest, H, ITERS, NPASS, EXP_LAT, `TB_ULP, worst_ulp);
        else
            $display("[mhc_sinkhorn] %0d/%0d FAILED (worst %0d ULP)", errors, checks, worst_ulp);
        $finish;
    end
    initial begin #20000000; $display("[mhc_sinkhorn] FAIL: timeout"); $finish; end
endmodule
