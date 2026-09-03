//============================================================================
// kda_gate_tb.v -- gate for the KDA forget-gate + beta step (tools/kda_gate_gen.py).
//
// Legs, on one run:
//   g_out  saturation   : wherever the golden g is -0.0 (the fp32 reference's
//                          sigmoid saturated to exactly 0), the DUT CANNOT reach
//                          -0.0: glm_act rails its input at +/-16, and
//                          sigmoid(-16) = 1.13e-7 is representable in bf16, so the
//                          DUT lands at -5*sigma(-16) = -5.64e-7 (measured: bits
//                          b5174000).  The leg therefore requires the DUT's g to be
//                          NEGATIVE (a positive value would be a real bug) and
//                          within that rail floor, and REPORTS the floor.  Effect on
//                          ge: exp(-5.64e-7) = 0.99999944 instead of 1.0, ~5 fp32
//                          ULP at 1.0 -- a pinned, explained divergence of the bf16
//                          activation unit, not a datapath error.
//   g_out  value        : tolerance (bf16-rounded sigmoid argument + polynomial)
//   ge_out              : tolerance (as above, then the Horner exp)
//   beta_out            : tolerance (bf16 sigmoid)
// -DINJ_GATE_DECAY_AFTER applies decay after the sigmoid and must FAIL.
//
// TB_REL is the relative tolerance on ge/beta; TB_ABS the absolute floor.  Both
// are set from the MEASURED error of the bf16 argument rounding reported by the
// generator, with headroom for glm_act's polynomial -- not tuned until green.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/kda_gate_vec.txt"
`endif
`ifndef TB_REL
    `define TB_REL 0.03
`endif
`ifndef TB_ABS
    `define TB_ABS 0.002
`endif
// glm_act saturation floor on |g| where the golden is -0.0: 5 * sigmoid(-16) with
// headroom.  A DUT g below -6e-7 there means the sigmoid did not saturate at all.
`ifndef TB_SAT_FLOOR
    `define TB_SAT_FLOOR 6.0e-7
`endif

module kda_gate_tb;
    localparam integer H = 3, DK = 8, N = H*DK;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    reg  [32*H-1:0]  decay_in, b_in;
    reg  [32*N-1:0]  f_in, dt_in;
    wire [32*N-1:0]  g_out, ge_out;
    wire [32*H-1:0]  beta_out;

    kda_gate_step #(.H(H), .DK(DK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .decay_in(decay_in), .b_in(b_in), .f_in(f_in), .dt_bias_in(dt_in),
        .g_out(g_out), .ge_out(ge_out), .beta_out(beta_out)
    );

    integer fd, code, t, i, ntest, hf, df, errors, checks, n_signz;
    real    worst_floor;
    reg [31:0] tmp;
    reg [31:0] exp_g [0:N-1], exp_ge [0:N-1], exp_b [0:H-1];
    real gr, er, tol, worst_rel, worst_abs;
    // per-output worst errors: 0 = g, 1 = ge, 2 = beta.  The three outputs have
    // different ranges (g in [-5,0], ge and beta in (0,1]) so one pooled number
    // hides WHICH stage is imprecise -- and that is the finding this unit exists to
    // make precise.
    real w_abs [0:2]; real w_rel [0:2];

    function real f2r(input [31:0] b);
        integer ex, mi; real m;
        begin
            ex = b[30:23]; mi = b[22:0];
            if (ex == 0) f2r = 0.0;
            else begin m = 1.0 + mi / 8388608.0; f2r = m * (2.0 ** (ex - 127)); if (b[31]) f2r = -f2r; end
        end
    endfunction
    task chk_tol(input [31:0] got, input [31:0] exp, input integer idx, input [8*8-1:0] what, input integer tag);
        real ae;
        begin
            checks = checks + 1;
            gr = f2r(got); er = f2r(exp);
            tol = `TB_REL * (er < 0 ? -er : er) + `TB_ABS;
            // Statistics for the banner. The RELATIVE tracker is only meaningful away
            // from zero -- dividing by a golden of 1e-9 turns a 1e-9 absolute miss into
            // a "100%" relative one and once printed 8e27 here. So: relative only where
            // |golden| >= 1e-3, absolute everywhere. The pass/fail test above is
            // unaffected; it was always rel*|er| + abs.
            ae = (gr - er > 0 ? gr - er : er - gr);
            if (ae > worst_abs) worst_abs = ae;
            if (ae > w_abs[tag]) w_abs[tag] = ae;
            if ((er < 0 ? -er : er) >= 1.0e-3) begin
                if (ae / (er < 0 ? -er : er) > worst_rel)  worst_rel  = ae / (er < 0 ? -er : er);
                if (ae / (er < 0 ? -er : er) > w_rel[tag]) w_rel[tag] = ae / (er < 0 ? -er : er);
            end
            if ((gr - er) > tol || (er - gr) > tol) begin
                $display("FAIL t%0d %0s[%0d]: got %h (%f) exp %h (%f) tol %f", t, what, idx, got, gr, exp, er, tol);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0; n_signz = 0; worst_rel = 0.0; worst_abs = 0.0; worst_floor = 0.0;
        for (i = 0; i < 3; i = i + 1) begin w_abs[i] = 0.0; w_rel[i] = 0.0; end
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[kda_gate] FAIL: cannot open %s", `TB_VEC); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, df);
        if (hf != H || df != DK) begin $display("[kda_gate] FAIL: shape (%0d,%0d) != TB (%0d,%0d)", hf, df, H, DK); $finish; end
        @(negedge clk); rst = 0;
        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < H; i = i + 1) begin code=$fscanf(fd,"%h",tmp); decay_in[32*i +: 32]=tmp; end
            for (i = 0; i < H; i = i + 1) begin code=$fscanf(fd,"%h",tmp); b_in[32*i +: 32]=tmp; end
            for (i = 0; i < N; i = i + 1) begin code=$fscanf(fd,"%h",tmp); f_in[32*i +: 32]=tmp; end
            for (i = 0; i < N; i = i + 1) begin code=$fscanf(fd,"%h",tmp); dt_in[32*i +: 32]=tmp; end
            for (i = 0; i < N; i = i + 1) begin code=$fscanf(fd,"%h",tmp); exp_g[i]=tmp; end
            for (i = 0; i < N; i = i + 1) begin code=$fscanf(fd,"%h",tmp); exp_ge[i]=tmp; end
            for (i = 0; i < H; i = i + 1) begin code=$fscanf(fd,"%h",tmp); exp_b[i]=tmp; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            i = 0;
            while (done !== 1'b1 && i < 200) begin @(negedge clk); i = i + 1; end
            if (done !== 1'b1) begin $display("[kda_gate] FAIL t%0d: done never asserted", t); errors = errors + 1; end

            for (i = 0; i < N; i = i + 1) begin
                // saturation leg: the golden's sigmoid saturated to exactly 0 (g = -0.0).
                // glm_act rails at +/-16 so the DUT cannot reach -0.0; require NEGATIVE
                // (or -0.0) and within the rail floor, and record the worst |g| seen.
                if (exp_g[i] == 32'h80000000) begin
                    checks = checks + 1; n_signz = n_signz + 1;
                    gr = f2r(g_out[32*i +: 32]);
                    if (g_out[32*i +: 32] !== 32'h80000000 && (g_out[32*i +: 32] == 32'h00000000 || gr > 0.0 || gr < -`TB_SAT_FLOOR)) begin
                        $display("FAIL t%0d g[%0d] saturation: got %h (%e); must be -0.0 or negative within the glm_act rail floor %e", t, i, g_out[32*i +: 32], gr, `TB_SAT_FLOOR);
                        errors = errors + 1;
                    end
                    if (-gr > worst_floor) worst_floor = -gr;
                end else chk_tol(g_out[32*i +: 32], exp_g[i], i, "g", 0);
                chk_tol(ge_out[32*i +: 32], exp_ge[i], i, "ge", 1);
            end
            for (i = 0; i < H; i = i + 1) chk_tol(beta_out[32*i +: 32], exp_b[i], i, "beta", 2);
            @(negedge clk);
        end
        $fclose(fd);
        if (n_signz == 0) begin
            $display("[kda_gate] FAIL: corpus never produced a saturated -0.0, so the sign-of-zero leg is vacuous");
            errors = errors + 1;
        end
        $display("[kda_gate] per-output worst error: g abs %e rel %0.5f | ge abs %e rel %0.5f | beta abs %e rel %0.5f  (rel only where |golden|>=1e-3)",
                 w_abs[0], w_rel[0], w_abs[1], w_rel[1], w_abs[2], w_rel[2]);
        if (errors == 0)
            $display("[kda_gate] ALL %0d TESTS PASSED (%0d tokens, H=%0d DK=%0d: %0d saturation checks -- golden -0.0, DUT negative within glm_act's +/-16 rail floor, worst |g| %e (= -5*sigmoid(-16), bf16 cannot reach -0.0); g/ge/beta within rel %0.3f + abs %0.4f -- worst rel %0.5f (|golden|>=1e-3), worst abs %e -- bf16 sigmoid + Horner exp, TOLERANCE legs)",
                     checks, ntest, H, DK, n_signz, worst_floor, `TB_REL, `TB_ABS, worst_rel, worst_abs);
        else
            $display("[kda_gate] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #5000000; $display("[kda_gate] FAIL: timeout"); $finish; end
endmodule
