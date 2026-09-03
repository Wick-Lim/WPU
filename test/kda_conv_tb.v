//============================================================================
// kda_conv_tb.v -- gate for the KDA causal conv step against tools/kda_conv_gen.py.
//
// Two legs, two different claims, on ONE run:
//   conv_out : the pre-activation bf16, compared BITWISE.  fp32 mul/add over the
//              taps in the pinned ascending order, one RNE round -- all exact
//              primitives (fp32_add's pinned 1-ULP gap can in principle move a
//              bf16 rounding boundary; the corpus is checked to show it did not).
//   y_out    : silu(conv_out) through glm_act's polynomial, compared to a
//              TOLERANCE, as swiglu_expert_q4k already is.
//   s_out    : the shifted history, compared BITWISE (pure wiring).
// -DINJ_CONV_FLIP reverses the tap order in the DUT and must FAIL on conv_out.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/kda_conv_vec.txt"
`endif

module kda_conv_tb;
    localparam integer C = 8, K = 4;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    reg  [32*C-1:0]       x_in;
    reg  [32*C*K-1:0]     w_in;
    reg  [32*C*(K-1)-1:0] s_in;
    wire [32*C*(K-1)-1:0] s_out;
    wire [16*C-1:0]       conv_out, y_out;

    kda_conv_step #(.C(C), .K(K)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_in(x_in), .w_in(w_in), .s_in(s_in), .s_out(s_out),
        .conv_out(conv_out), .y_out(y_out)
    );

    integer fd, code, t, i, ntest, cf, kf, errors, checks;
    reg [31:0] tmp32; reg [15:0] tmp16;
    reg [15:0] exp_conv [0:C-1], exp_y [0:C-1];
    reg [31:0] exp_st   [0:C*(K-1)-1];
    real gr, er, tol;

    function real bf2r(input [15:0] b);   // bf16 bits -> real (finite)
        integer ex, mi; real m;
        begin
            ex = b[14:7]; mi = b[6:0];
            if (ex == 0) bf2r = 0.0;
            else begin m = 1.0 + mi / 128.0; bf2r = m * (2.0 ** (ex - 127)); if (b[15]) bf2r = -bf2r; end
        end
    endfunction

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[kda_conv] FAIL: cannot open %s", `TB_VEC); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, cf, kf);
        if (cf != C || kf != K) begin
            $display("[kda_conv] FAIL: vector shape (C=%0d,K=%0d) != TB (C=%0d,K=%0d)", cf, kf, C, K); $finish;
        end
        @(negedge clk); rst = 0;
        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < C*K;     i = i + 1) begin code=$fscanf(fd,"%h",tmp32); w_in[32*i +: 32]=tmp32; end
            for (i = 0; i < C*(K-1); i = i + 1) begin code=$fscanf(fd,"%h",tmp32); s_in[32*i +: 32]=tmp32; end
            for (i = 0; i < C;       i = i + 1) begin code=$fscanf(fd,"%h",tmp32); x_in[32*i +: 32]=tmp32; end
            for (i = 0; i < C;       i = i + 1) begin code=$fscanf(fd,"%h",tmp16); exp_conv[i]=tmp16; end
            for (i = 0; i < C;       i = i + 1) begin code=$fscanf(fd,"%h",tmp16); exp_y[i]=tmp16; end
            for (i = 0; i < C*(K-1); i = i + 1) begin code=$fscanf(fd,"%h",tmp32); exp_st[i]=tmp32; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            i = 0;
            while (done !== 1'b1 && i < 40) begin @(negedge clk); i = i + 1; end
            if (done !== 1'b1) begin $display("[kda_conv] FAIL t%0d: done never asserted", t); errors = errors + 1; end

            for (i = 0; i < C; i = i + 1) begin
                checks = checks + 1;                       // bitwise pre-activation
                if (conv_out[16*i +: 16] !== exp_conv[i]) begin
                    $display("FAIL t%0d conv[%0d]: got %h exp %h", t, i, conv_out[16*i +: 16], exp_conv[i]);
                    errors = errors + 1;
                end
                checks = checks + 1;                       // tolerance post-silu
                gr = bf2r(y_out[16*i +: 16]); er = bf2r(exp_y[i]);
                tol = 0.02 * (er < 0 ? -er : er) + 0.01;   // glm_act poly-silu + bf16 grid
                if ((gr - er) > tol || (er - gr) > tol) begin
                    $display("FAIL t%0d y[%0d]: got %f exp %f tol %f", t, i, gr, er, tol);
                    errors = errors + 1;
                end
            end
            for (i = 0; i < C*(K-1); i = i + 1) begin
                checks = checks + 1;                       // bitwise shift
                if (s_out[32*i +: 32] !== exp_st[i]) begin
                    $display("FAIL t%0d state[%0d]: got %h exp %h", t, i, s_out[32*i +: 32], exp_st[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[kda_conv] ALL %0d TESTS PASSED (%0d tokens, C=%0d K=%0d: conv_out bitwise, s_out bitwise, silu(conv) within glm_act tolerance)",
                     checks, ntest, C, K);
        else
            $display("[kda_conv] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #2000000; $display("[kda_conv] FAIL: timeout"); $finish; end
endmodule
