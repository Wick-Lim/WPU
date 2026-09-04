//============================================================================
// mhc_map_step_tb.v -- gate for the mHC map (src/mhc_map_step.v,
// vectors from tools/mhc_map_gen.py).
//
// The golden is ref.hyper_connection's OWN post and comb (bitwise, checked in the
// generator's self-test), with `pre` validated against the reference's returned
// `collapsed`.  The bounds are the generator's PREDICTED envelope -- every exp
// perturbed by fp32_exp_pipe's measured 2.3e-4 and every sigmoid by
// fp32_sigmoid_pipe's measured 790 ULP, in the worst-case direction, pushed
// through the softmax renormalise and all 39 Sinkhorn passes -- with ~1.6x
// headroom.  So the numbers below come from the primitives, not from whatever the
// DUT happens to print.
//
// The LATENCY check is a design claim, not bookkeeping.  mHC deliberately has no
// convergence-based early exit (see src/mhc_sinkhorn.v), so the unit's latency
// must be DATA-INDEPENDENT.  The TB requires every vector to complete in exactly
// the same number of cycles, and that number to equal EXP_LAT_TOTAL.
//
// Must FAIL: -DINJ_MAP_POST_NO2, -DINJ_MAP_PRE_NOEPS, -DINJ_MAP_COMB_NOEPS,
//            -DINJ_MAP_SOFTMAX_NOMAX
//============================================================================
`timescale 1ns/1ps
`include "glm_fp_pipe_lat.vh"
`ifndef TB_VEC
    `define TB_VEC "build/mhc_map_vec.txt"
`endif
`ifndef TB_ULP_SIG
    `define TB_ULP_SIG 1024
`endif
`ifndef TB_ULP_COMB
    `define TB_ULP_COMB 16384
`endif

module mhc_map_step_tb;
    localparam integer H = 4, ITERS = 20, RECIP_ITERS = 4;
    localparam integer NN = H*H, NM = (2+H)*H;
    localparam integer LAT_SIG = `FP_EXP_LAT + 3 + RECIP_ITERS;   // measured, see the DUT
    localparam integer NPASS   = 1 + (ITERS-1)*2;
    // Every term derived from the FSM, then confirmed against the measured 700 --
    // not fitted to it.  `lat` counts posedges from the start-sampling edge, which
    // is itself the IDLE->LOGIT transition, hence the leading 1.
    //   1              start edge (IDLE -> S_LOGIT)
    //   H+2            S_LOGIT: H comb rows, then pre args, then post args
    //   2+LAT_SIG+2H-1 S_SIG:  one cycle to raise valid_in, the pipe, 2H collects
    //   H              S_MAXR
    //   2+FP_EXP_LAT+NN-1  S_EXP: same shape, NN collects
    //   H              S_RSUM        RECIP_ITERS+1  S_RRCP        H  S_NORM
    //   1              S_LOAD
    //   2+NPASS*(...)  mhc_sinkhorn's own latency (548, pinned by its own TB)
    //   2              S_SINK observing sink_done, then S_FIN raising done
    localparam integer EXP_LAT_TOTAL =
        1 + (H+2) + (2 + LAT_SIG + 2*H - 1) + H
          + (2 + `FP_EXP_LAT + NN - 1) + H + (RECIP_ITERS+1) + H
          + 1 + (2 + NPASS*((H+1) + (RECIP_ITERS+1) + H)) + 2;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    reg  [32*NM-1:0] mixed_in, base_in;
    reg  [31:0]      s0, s1, s2;
    wire [32*H-1:0]  pre_out, post_out;
    wire [32*NN-1:0] comb_out;

    mhc_map_step #(.H(H), .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .mixed_in(mixed_in), .base_in(base_in),
        .scale0(s0), .scale1(s1), .scale2(s2),
        .pre_out(pre_out), .post_out(post_out), .comb_out(comb_out));

    integer fd, code, t, i, ntest, hf, itf, errors, checks, lat, lat0;
    integer d, w_pre, w_post, w_comb;
    reg [31:0] t32;
    reg [31:0] e_pre [0:H-1];
    reg [31:0] e_post[0:H-1];
    reg [31:0] e_comb[0:NN-1];

    // fp32 ULP distance. pre/post/comb are all strictly positive here, so the bit
    // patterns are monotone in value; a sign flip is reported as unbounded rather
    // than folded into a small number.
    function integer ulp(input [31:0] a, input [31:0] b);
        begin
            if (a[31] !== b[31]) ulp = 1000000000;
            else if (a[30:0] >= b[30:0]) ulp = a[30:0] - b[30:0];
            else                          ulp = b[30:0] - a[30:0];
        end
    endfunction

    task chk(input [31:0] got, input [31:0] exp, input integer bound,
             input integer tno, input [127:0] nm, input integer idx);
        begin
            checks = checks + 1;
            d = ulp(got, exp);
            if (d > bound) begin
                $display("FAIL t%0d %0s[%0d]: got %h exp %h (%0d ULP > %0d)",
                         tno, nm, idx, got, exp, d, bound);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0; lat0 = -1;
        w_pre = 0; w_post = 0; w_comb = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mhc_map] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, itf);
        if (hf != H || itf != ITERS) begin
            $display("[mhc_map] FAIL: vector H/ITERS %0d/%0d != TB %0d/%0d", hf, itf, H, ITERS);
            $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < NM; i = i + 1) begin code=$fscanf(fd,"%h",t32); mixed_in[32*i +: 32]=t32; end
            for (i = 0; i < NM; i = i + 1) begin code=$fscanf(fd,"%h",t32); base_in[32*i +: 32]=t32; end
            code=$fscanf(fd,"%h",t32); s0=t32;
            code=$fscanf(fd,"%h",t32); s1=t32;
            code=$fscanf(fd,"%h",t32); s2=t32;
            for (i = 0; i < H;  i = i + 1) begin code=$fscanf(fd,"%h",t32); e_pre[i]=t32;  end
            for (i = 0; i < H;  i = i + 1) begin code=$fscanf(fd,"%h",t32); e_post[i]=t32; end
            for (i = 0; i < NN; i = i + 1) begin code=$fscanf(fd,"%h",t32); e_comb[i]=t32; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            lat = 1;
            while (done !== 1'b1 && lat < 4*EXP_LAT_TOTAL) begin @(negedge clk); lat = lat + 1; end

            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[mhc_map] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end else if (lat0 < 0) begin
                lat0 = lat;
                if (lat != EXP_LAT_TOTAL) begin
                    $display("[mhc_map] FAIL t%0d: done at %0d cycles, expected %0d",
                             t, lat, EXP_LAT_TOTAL);
                    errors = errors + 1;
                end
            end else if (lat != lat0) begin
                $display("[mhc_map] FAIL t%0d: latency %0d != %0d -- mHC latency must be DATA-INDEPENDENT",
                         t, lat, lat0);
                errors = errors + 1;
            end

            for (i = 0; i < H; i = i + 1) begin
                d = ulp(pre_out[32*i +: 32], e_pre[i]);   if (d > w_pre)  w_pre  = d;
                chk(pre_out[32*i +: 32],  e_pre[i],  `TB_ULP_SIG, t, "pre", i);
                d = ulp(post_out[32*i +: 32], e_post[i]); if (d > w_post) w_post = d;
                chk(post_out[32*i +: 32], e_post[i], `TB_ULP_SIG, t, "post", i);
            end
            for (i = 0; i < NN; i = i + 1) begin
                d = ulp(comb_out[32*i +: 32], e_comb[i]); if (d > w_comb) w_comb = d;
                chk(comb_out[32*i +: 32], e_comb[i], `TB_ULP_COMB, t, "comb", i);
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mhc_map] ALL %0d TESTS PASSED (%0d blocks H=%0d: pre/post within %0d ULP of the float64 reference -- worst %0d/%0d; comb (softmax + %0d Sinkhorn passes) within %0d ULP -- worst %0d; latency %0d cycles, data-independent)",
                     checks, ntest, H, `TB_ULP_SIG, w_pre, w_post, NPASS, `TB_ULP_COMB, w_comb, lat0);
        else
            $display("[mhc_map] %0d/%0d FAILED (worst pre %0d post %0d comb %0d ULP)",
                     errors, checks, w_pre, w_post, w_comb);
        $finish;
    end
    initial begin #40000000; $display("[mhc_map] FAIL: timeout"); $finish; end
endmodule
