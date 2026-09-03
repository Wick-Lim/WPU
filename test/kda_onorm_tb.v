//============================================================================
// kda_onorm_tb.v -- gate for the KDA output norm step (tools/kda_onorm_gen.py).
// TOLERANCE leg only (Quake rsqrt + bf16 polynomial sigmoid + bf16 gamma_eff);
// the TB prints worst relative (|golden|>=1e-3) and absolute error so a
// regression moves a number.  -DINJ_ONORM_GATE_FIRST must FAIL.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/kda_onorm_vec.txt"
`endif
`ifndef TB_REL
    `define TB_REL 0.03
`endif
`ifndef TB_ABS
    `define TB_ABS 0.004
`endif

module kda_onorm_tb;
    localparam integer H = 3, DV = 16, N = H*DV;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    reg  [16*N-1:0]  x_in;
    reg  [32*N-1:0]  gate_in;
    reg  [16*DV-1:0] w_in;
    wire [16*N-1:0]  y_out;

    kda_onorm_step #(.H(H), .DV(DV)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_in(x_in), .gate_in(gate_in), .weight_in(w_in), .y_out(y_out));

    integer fd, code, t, i, ntest, hf, df, errors, checks;
    reg [31:0] t32; reg [15:0] t16;
    reg [15:0] exp_y [0:N-1];
    real gr, er, tol, ae, worst_rel, worst_abs;

    function real bf2r(input [15:0] b);
        integer ex, mi; real m;
        begin
            ex = b[14:7]; mi = b[6:0];
            if (ex == 0) bf2r = 0.0;
            else begin m = 1.0 + mi / 128.0; bf2r = m * (2.0 ** (ex - 127)); if (b[15]) bf2r = -bf2r; end
        end
    endfunction

    initial begin
        errors = 0; checks = 0; worst_rel = 0.0; worst_abs = 0.0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[kda_onorm] FAIL: cannot open %s", `TB_VEC); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, df);
        if (hf != H || df != DV) begin $display("[kda_onorm] FAIL: shape (%0d,%0d) != TB (%0d,%0d)", hf, df, H, DV); $finish; end
        @(negedge clk); rst = 0;
        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < DV; i = i + 1) begin code=$fscanf(fd,"%h",t16); w_in[16*i +: 16]=t16; end
            for (i = 0; i < N;  i = i + 1) begin code=$fscanf(fd,"%h",t32); gate_in[32*i +: 32]=t32; end
            for (i = 0; i < N;  i = i + 1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i = 0; i < N;  i = i + 1) begin code=$fscanf(fd,"%h",t16); exp_y[i]=t16; end
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            i = 0;
            while (done !== 1'b1 && i < 2000) begin @(negedge clk); i = i + 1; end
            if (done !== 1'b1) begin $display("[kda_onorm] FAIL t%0d: done never asserted", t); errors = errors + 1; end
            for (i = 0; i < N; i = i + 1) begin
                checks = checks + 1;
                gr = bf2r(y_out[16*i +: 16]); er = bf2r(exp_y[i]);
                tol = `TB_REL * (er < 0 ? -er : er) + `TB_ABS;
                ae = (gr - er > 0 ? gr - er : er - gr);
                if (ae > worst_abs) worst_abs = ae;
                if ((er < 0 ? -er : er) >= 1.0e-3 && ae / (er < 0 ? -er : er) > worst_rel) worst_rel = ae / (er < 0 ? -er : er);
                if (ae > tol) begin
                    $display("FAIL t%0d y[%0d]: got %h (%f) exp %h (%f) tol %f", t, i, y_out[16*i +: 16], gr, exp_y[i], er, tol);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[kda_onorm] ALL %0d TESTS PASSED (%0d tokens, H=%0d DV=%0d: gated RMSNorm within rel %0.3f + abs %0.4f -- worst rel %0.5f (|golden|>=1e-3), worst abs %e; Quake rsqrt + bf16 sigmoid + bf16 gamma_eff, TOLERANCE leg)",
                     checks, ntest, H, DV, `TB_REL, `TB_ABS, worst_rel, worst_abs);
        else
            $display("[kda_onorm] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #20000000; $display("[kda_onorm] FAIL: timeout"); $finish; end
endmodule
