`timescale 1ns/1ps
//============================================================================
// test/glm_softmax_pipe_tb.v
//   Gate for docs/ULTRA_PERF_MODULES.md rank 14 -- "pipeline the softmax
//   shift/normalize passes to 1 issue/cycle".  `make rank14`.
//
// WHAT IS UNDER TEST
//   glm_softmax serializes two of its passes behind a single-op interlock:
//     src/glm_softmax.v S_SHIFT  `if (!sh_busy && sh_in_cnt < LEN_IW)`
//     src/glm_softmax.v S_NORM   `if (!nm_busy && nm_in_cnt < LEN_IW)`
//   so each pass pays the full fp-pipe latency PER ELEMENT instead of once.
//   SM_PIPE=1 deletes exactly those two `!*_busy && ` tokens.
//
// WHY A DUT-vs-DUT TEST, NOT A GOLDEN
//   The committed test/glm_softmax_tb.v checks against a numeric golden with a
//   REL_TOL of several bf16 ULP.  A tolerance check CANNOT prove bit-exactness,
//   and bit-exactness is the whole contract here: SM_PIPE must be a scheduling
//   change with NO numeric consequence.  So both variants are elaborated from the
//   SAME source, fed the SAME stimulus, and every output word is compared as a raw
//   16-bit pattern with ===.  Identity, not tolerance.
//
// MEASUREMENT
//   Cycles from `start` to `done` are counted per row for each variant, over a
//   campaign that deliberately includes the fp special-case paths (all-equal,
//   one-hot, large-magnitude, descending, random) so the win is not measured on
//   one lucky shape.
//
// INJECTION (`-DINJ_SM_PIPE_OOO`, run by `make rank14`)
//   Inside the SM_PIPE=1 branch only, S_SHIFT captures at the ISSUE index
//   (sh_in_cnt) instead of the in-order RESULT index (sh_out_cnt).  With the
//   interlock gone the issue counter runs FP_ADD_LAT+1 = 6 ahead, so every shifted
//   logit lands in the wrong slot.  The bit-exact assert MUST FAIL.  This is the
//   injection that matters: it proves the test is constraining the ORDERING
//   property that removing the interlock puts at risk, not merely "some output".
//============================================================================
module glm_softmax_pipe_tb;

    parameter integer LEN   = 8;
    parameter integer LANES = 1;
    localparam integer NROW = 12;

    reg clk = 1'b0;  always #5 clk = ~clk;
    reg rst;

    reg                   start;
    reg                   in_valid;
    reg  [LANES*16-1:0]   x_in;

    wire                  busy_r,  ov_r,  done_r;
    wire [LANES*16-1:0]   p_r;
    wire                  busy_p,  ov_p,  done_p;
    wire [LANES*16-1:0]   p_p;

    // Same source, same LEN/LANES, same stimulus -- ONLY SM_PIPE differs.
    glm_softmax #(.LEN(LEN), .LANES(LANES), .SM_PIPE(0)) dut_ref (
        .clk(clk), .rst(rst), .start(start), .in_valid(in_valid), .x_in(x_in),
        .busy(busy_r), .out_valid(ov_r), .p_out(p_r), .done(done_r));

    glm_softmax #(.LEN(LEN), .LANES(LANES), .SM_PIPE(1)) dut_pip (
        .clk(clk), .rst(rst), .start(start), .in_valid(in_valid), .x_in(x_in),
        .busy(busy_p), .out_valid(ov_p), .p_out(p_p), .done(done_p));

    integer errors, tests;
    integer cyc_ref_tot, cyc_pip_tot;
    integer r, i;

    reg [15:0] row  [0:NROW-1][0:LEN-1];
    reg [15:0] got_r[0:LEN-1];
    reg [15:0] got_p[0:LEN-1];
    integer    nr, np;

    // ---- capture: both DUTs emit LANES words per out_valid beat, row order ----
    always @(posedge clk) if (!rst) begin
        if (ov_r) begin
            for (i = 0; i < LANES; i = i + 1)
                if (nr + i < LEN) got_r[nr+i] = p_r[i*16 +: 16];
            nr = nr + LANES;
        end
        if (ov_p) begin
            for (i = 0; i < LANES; i = i + 1)
                if (np + i < LEN) got_p[np+i] = p_p[i*16 +: 16];
            np = np + LANES;
        end
    end

    // bf16 helper: assemble from sign/exp/mant so the campaign is explicit
    function [15:0] bf; input integer s; input integer e; input integer m;
        bf = {s[0], e[7:0], m[6:0]};
    endfunction

    integer seed;
    task build_rows; integer rr, ii; begin
        seed = 32'd12345;
        // r0: all equal (uniform softmax; exercises the max==x path everywhere)
        for (ii=0; ii<LEN; ii=ii+1) row[0][ii] = bf(0, 127, 0);          // 1.0
        // r1: one-hot large (dominant term; exp of large negative for the rest)
        for (ii=0; ii<LEN; ii=ii+1) row[1][ii] = bf(0, 120, 0);
        row[1][LEN/2] = bf(0, 132, 0);
        // r2: large POSITIVE magnitudes (the shift-by-max stability path)
        for (ii=0; ii<LEN; ii=ii+1) row[2][ii] = bf(0, 134, ii*8);
        // r3: large NEGATIVE magnitudes
        for (ii=0; ii<LEN; ii=ii+1) row[3][ii] = bf(1, 134, ii*8);
        // r4: descending
        for (ii=0; ii<LEN; ii=ii+1) row[4][ii] = bf(0, 130-ii, 0);
        // r5: ascending
        for (ii=0; ii<LEN; ii=ii+1) row[5][ii] = bf(0, 120+ii, 0);
        // r6: zeros (exp(0)=1 for all after shift)
        for (ii=0; ii<LEN; ii=ii+1) row[6][ii] = 16'h0000;
        // r7: alternating sign
        for (ii=0; ii<LEN; ii=ii+1) row[7][ii] = bf(ii[0], 127, ii*4);
        // r8..: pseudo-random but deterministic (no $random -- reproducible)
        for (rr=8; rr<NROW; rr=rr+1)
            for (ii=0; ii<LEN; ii=ii+1) begin
                seed = (seed * 32'd1103515245 + 32'd12345);
                row[rr][ii] = bf(seed[31], 8'd120 + seed[19:17], seed[16:10]);
            end
    end endtask

    // ---- drive ONE row into BOTH DUTs from one stream ----
    task run_row; input integer rr; integer ii; integer c0r, c0p, cc; begin
        nr = 0; np = 0;
        for (ii=0; ii<LEN; ii=ii+1) begin got_r[ii] = 16'hxxxx; got_p[ii] = 16'hxxxx; end

        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
        // feed LEN/LANES beats
        for (ii=0; ii<LEN; ii=ii+LANES) begin
            in_valid = 1'b1;
            x_in     = row[rr][ii];
            @(negedge clk);
        end
        in_valid = 1'b0;

        // count cycles to each DUT's done
        cc = 0; c0r = -1; c0p = -1;
        while (((c0r < 0) || (c0p < 0)) && (cc < 20000)) begin
            @(posedge clk);
            cc = cc + 1;
            if (done_r && (c0r < 0)) c0r = cc;
            if (done_p && (c0p < 0)) c0p = cc;
        end
        if (c0r < 0 || c0p < 0) begin
            $display("FAIL[row %0d]: a DUT never raised done (ref=%0d pip=%0d)", rr, c0r, c0p);
            errors = errors + 1;
        end else begin
            cyc_ref_tot = cyc_ref_tot + c0r;
            cyc_pip_tot = cyc_pip_tot + c0p;
        end
        repeat (4) @(negedge clk);

        // ---- A1: BIT-EXACT identity, every word, raw pattern ----
        tests = tests + 1;
        for (ii=0; ii<LEN; ii=ii+1) begin
            if (got_p[ii] !== got_r[ii]) begin
                if (errors < 8)
                    $display("FAIL[row %0d]: word %0d SM_PIPE=1 gave %h, SM_PIPE=0 gave %h (NOT bit-exact)",
                             rr, ii, got_p[ii], got_r[ii]);
                errors = errors + 1;
            end
        end
        // A2: neither DUT may emit X (a row of X would compare 'equal' by accident
        //     only if both were X -- !== catches that, but say it plainly anyway)
        tests = tests + 1;
        for (ii=0; ii<LEN; ii=ii+1)
            if (^got_r[ii] === 1'bx) begin
                $display("FAIL[row %0d]: reference word %0d is X", rr, ii);
                errors = errors + 1;
            end
    end endtask

    real spd;
    initial begin
        errors = 0; tests = 0; cyc_ref_tot = 0; cyc_pip_tot = 0;
        start = 1'b0; in_valid = 1'b0; x_in = {(LANES*16){1'b0}};
        nr = 0; np = 0;
        build_rows;

        rst = 1'b1; repeat (6) @(negedge clk); rst = 1'b0;
        repeat (2) @(negedge clk);

        for (r = 0; r < NROW; r = r + 1) run_row(r);

        $display("  [MEASURED] LEN=%0d LANES=%0d over %0d rows: serial %0d cyc -> 1-issue/cycle %0d cyc (%.3fx)",
                 LEN, LANES, NROW, cyc_ref_tot, cyc_pip_tot,
                 (cyc_pip_tot > 0) ? (1.0*cyc_ref_tot)/(1.0*cyc_pip_tot) : 0.0);

        // ---- A3: the change must actually REMOVE cycles, or it is not a lever ----
        tests = tests + 1;
        if (cyc_pip_tot >= cyc_ref_tot) begin
            $display("FAIL: SM_PIPE=1 did not reduce cycles (%0d -> %0d)", cyc_ref_tot, cyc_pip_tot);
            errors = errors + 1;
        end

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "glm_softmax_pipe_tb had mismatches");
        end
        spd = (1.0*cyc_ref_tot)/(1.0*cyc_pip_tot);
        $display("ALL %0d TESTS PASSED  (glm_softmax SM_PIPE=1 is BIT-EXACT vs SM_PIPE=0 over %0d rows and %.3fx faster)",
                 tests, NROW, spd);
        $finish;
    end

    initial begin
        #4000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
