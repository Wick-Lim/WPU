`timescale 1ns/1ps
//============================================================================
// test/l3_hash_mirror_tb.v -- Python packer hashes == Verilog stub hashes
//   (`make l3-hash-mirror`)
//
// WHY
//   The L3 end-to-end gate feeds the DUT from PACKER-built images and the
//   reference from the loopback TBs' VERILOG stub functions.  A single
//   mirrored bit of difference between tools/l3_image_pack.py and those
//   functions makes the tokens diverge after HOURS of sim.  This gate proves
//   the mirrors bit-exact on a 704-point probe set in seconds instead: the
//   packer writes build/l3_hash_probe.txt, and this TB recomputes every probe
//   with the VERILOG functions (copied verbatim from the loopback TBs) and
//   compares.
//
// INJECTION (`make l3-hash-mirror` requires FAILURE)
//   -DINJ_HASH_SKEW perturbs one multiplier constant in the Verilog f_h below.
//   Every probe family MUST then mismatch -- proving the comparison actually
//   constrains the hash, not merely file plumbing.
//============================================================================
module l3_hash_mirror_tb;

    // geometry constants matching the packer's defaults (tiny E2E config)
    localparam integer L = 2, N_EXPERT = 4, MODEL_DIM = 16, VOCAB = 16, LM_TN = 4;
    localparam integer NVT = VOCAB / LM_TN;

    // ---- Verilog hash primitives, VERBATIM from the loopback TBs ------------
    function automatic integer f_h; input integer seed; begin
`ifdef INJ_HASH_SKEW
        f_h = (seed*2654435761)^(seed<<13)^(seed*40509);
`else
        f_h = (seed*2654435761)^(seed<<13)^(seed*40503);
`endif
    end endfunction
    function automatic [15:0] gen_bf16; input integer seed;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = f_h(seed);
        s = h[3];
        e = 8'd124 + {6'b0,h[5:4]};
        m = h[12:6];
        gen_bf16 = {s,e,m};
    end endfunction
    function automatic [15:0] gen_fp16; input integer seed;
        reg [4:0] e; reg [9:0] m; integer h; begin
        h = f_h(seed);
        e = 5'd12 + {4'b0,h[4]};
        m = h[14:5];
        gen_fp16 = {1'b0,e,m};
    end endfunction
    function automatic [3:0] gen_q4; input integer seed; integer h; begin
        h = f_h(seed);
        gen_q4 = h[11:8];
    end endfunction
    function automatic [31:0] gen_s32; input integer seed; begin
        gen_s32 = f_h(seed*97 + 5);
    end endfunction

    function automatic [3:0] f_awq; input integer ly; input integer sel;
        input integer fo; input integer kk; begin
        f_awq = gen_q4(ly*7919 + sel*104729 + fo*611953 + kk*13 + 101);
    end endfunction
    function automatic [15:0] f_awd; input integer ly; input integer sel;
        input integer fo; begin
        f_awd = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 211);
    end endfunction
    function automatic [3:0] f_fwq; input integer ly; input integer sel;
        input integer shr; input integer eidx; input integer fo; input integer kk;
        begin
        f_fwq = gen_q4(ly*7919 + sel*104729 + shr*15485863 + eidx*350377
                       + fo*611953 + kk*13 + 503);
    end endfunction
    function automatic [15:0] f_fwd; input integer ly; input integer sel;
        input integer shr; input integer eidx; input integer fo; begin
        f_fwd = gen_fp16(ly*7919 + sel*104729 + shr*15485863 + eidx*350377
                         + fo*611953 + 521);
    end endfunction
    function automatic [3:0] f_rwq; input integer ly; input integer e;
        input integer kk; begin
        f_rwq = gen_q4(ly*7919 + e*350377 + kk*13 + 401);
    end endfunction
    function automatic [15:0] f_rwd; input integer ly; input integer e; begin
        f_rwd = gen_fp16(ly*7919 + e*350377 + 421);
    end endfunction

    integer fd, code, i, errors, tests;
    reg [8*8-1:0] name;
    reg [31:0] pyv, myv;

    initial begin
        errors = 0; tests = 0;
        fd = $fopen("build/l3_hash_probe.txt", "r");
        if (fd == 0) $fatal(1, "missing build/l3_hash_probe.txt (run tools/l3_image_pack.py)");

        for (i = 0; i < 64; i = i + 1) begin
            code = $fscanf(fd, "%s %h", name, pyv); myv = {28'd0, f_awq(i%L, i%7, i%16, i%8)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL awq[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv); myv = {16'd0, f_awd(i%L, i%7, i%16)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL awd[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv); myv = {28'd0, f_fwq(i%L, i%3, i&1, i%N_EXPERT, i%8, i%8)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL fwq[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv); myv = {16'd0, f_fwd(i%L, i%3, i&1, i%N_EXPERT, i%8)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL fwd[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv); myv = {28'd0, f_rwq(i%L, i%N_EXPERT, i%8)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL rwq[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv); myv = {16'd0, f_rwd(i%L, i%N_EXPERT)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL rwd[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv);
            myv = {16'd0, gen_bf16(((i%NVT)*LM_TN + (i%LM_TN))*MODEL_DIM + (i%MODEL_DIM) + 7603)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL lw[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv);
            myv = {16'd0, gen_bf16((i%VOCAB)*MODEL_DIM + (i%MODEL_DIM) + 7001)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL em[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv);
            myv = {16'd0, gen_bf16((i%MODEL_DIM) + 7207)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL fn[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv);
            myv = {16'd0, gen_bf16((i%L)*1024 + (i&1)*512 + (i%MODEL_DIM) + 7411)};
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL gn[%0d]: verilog %h != python %h", i, myv, pyv); end
            code = $fscanf(fd, "%s %h", name, pyv);
            myv = gen_s32((i%L)*7919 + (i%7)*104729 + (i%16)*611953 + 601 + (i%3));
            tests=tests+1; if (myv !== pyv) begin errors=errors+1;
                if (errors<6) $display("FAIL sc[%0d]: verilog %h != python %h", i, myv, pyv); end
        end
        $fclose(fd);

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "l3_hash_mirror_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (Python packer hash mirrors == Verilog stub hashes across 11 families x 64 probes)",
                 tests);
        $finish;
    end
endmodule
