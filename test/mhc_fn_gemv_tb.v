//============================================================================
// mhc_fn_gemv_tb.v -- gate for the mHC `fn` GEMV (src/mhc_fn_gemv.v,
// vectors from tools/mhc_fn_gemv_gen.py).
//
// BITWISE.  The generator emulates this datapath exactly -- Q8_0 dequant as
// fp16_scale * int8_code, strictly sequential fp32 accumulation from +0.0, mean
// = sum * 2^-log2(K), and glm_fp.vh's Quake rsqrt reproduced bit for bit -- so
// there is nothing here to excuse a tolerance. fp32_add's measured 1-ULP
// non-conformance (`make fp-ieee`) is the only slack, and a K-long accumulation is
// exactly where it shows: measured worst 11 ULP over K=256, so TB_ULP is 64 (~6x)
// and the worst is printed. That slack is a function of K -- at the real
// K = H*D = 16384 there are 64x more adds in the chain, so a re-derived bound
// belongs with any re-slice, not a blanket widening of this one.
//
// A bitwise gate against an emulation only proves the RTL matches my model of it.
// What says the model is the RIGHT one is the generator's second, printed number:
// the deviation from the SPEC path (normalise first in float64, then GEMV), which
// is what the folded RMS and the approximate rsqrt actually cost against
// tools/glm53_flash_ref.py. Read both lines.
//
// Must FAIL: -DINJ_GEMV_Q8_NOSCALE, -DINJ_GEMV_MEAN_SUM, -DINJ_GEMV_NO_EPS
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/mhc_fn_gemv_vec.txt"
`endif
`ifndef TB_ULP
    `define TB_ULP 64
`endif

module mhc_fn_gemv_tb;
    localparam integer H = 4, D = 64, QK = 32;
    localparam integer K = H*D, ROWS = (2+H)*H, NB = K/QK;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    reg  [32*K-1:0]      x_in;
    reg  [8*ROWS*K-1:0]  w_q;
    reg  [16*ROWS*NB-1:0] w_d;
    wire [32*ROWS-1:0]   mixed_out;

    mhc_fn_gemv #(.H(H), .D(D), .QK(QK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_in(x_in), .w_q(w_q), .w_d(w_d), .mixed_out(mixed_out));

    integer fd, code, t, i, ntest, hf, df, errors, checks, w, d, worst;
    reg [31:0] t32; reg [15:0] t16; reg [7:0] t8;
    reg [31:0] e_mixed [0:ROWS-1];

    function integer ulp(input [31:0] a, input [31:0] b);
        begin
            if (a[31] !== b[31])
                ulp = (a[30:0] == 0 && b[30:0] == 0) ? 0 : 1000000000;
            else if (a[30:0] >= b[30:0]) ulp = a[30:0] - b[30:0];
            else                          ulp = b[30:0] - a[30:0];
        end
    endfunction

    initial begin
        errors = 0; checks = 0; worst = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mhc_gemv] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, df);
        if (hf != H || df != D) begin
            $display("[mhc_gemv] FAIL: vector H/D %0d/%0d != TB %0d/%0d", hf, df, H, D);
            $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < K;       i = i + 1) begin code=$fscanf(fd,"%h",t32); x_in[32*i +: 32]=t32; end
            for (i = 0; i < ROWS*K;  i = i + 1) begin code=$fscanf(fd,"%h",t8);  w_q [ 8*i +:  8]=t8;  end
            for (i = 0; i < ROWS*NB; i = i + 1) begin code=$fscanf(fd,"%h",t16); w_d [16*i +: 16]=t16; end
            for (i = 0; i < ROWS;    i = i + 1) begin code=$fscanf(fd,"%h",t32); e_mixed[i]=t32; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 20*K) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[mhc_gemv] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end

            for (i = 0; i < ROWS; i = i + 1) begin
                checks = checks + 1;
                d = ulp(mixed_out[32*i +: 32], e_mixed[i]);
                if (d > worst) worst = d;
                if (d > `TB_ULP) begin
                    $display("FAIL t%0d mixed[%0d]: got %h exp %h (%0d ULP)",
                             t, i, mixed_out[32*i +: 32], e_mixed[i], d);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mhc_gemv] ALL %0d TESTS PASSED (%0d vectors, K=%0d, %0d rows: fp32 activations x Q8_0 weights with the RMS folded past the GEMV, within %0d ULP of the bit-exact emulation -- worst %0d)",
                     checks, ntest, K, ROWS, `TB_ULP, worst);
        else
            $display("[mhc_gemv] %0d/%0d FAILED (worst %0d ULP)", errors, checks, worst);
        $finish;
    end
    initial begin #200000000; $display("[mhc_gemv] FAIL: timeout"); $finish; end
endmodule
