//============================================================================
// glm53f_hc_block_tb.v -- gate for the GLM-5.3-Flash block residual skeleton
// (src/glm53f_hc_block.v, vectors from tools/glm53f_hc_block_gen.py).
//
// mhc-site already pins ONE site's numerics. What this gate is for is the WIRING
// a block adds: two sites driven from one instance with the weights muxed, each
// with its OWN learned norm between `collapsed` and its sublayer, and the streams
// the attention site writes threaded into the FFN site. All four injections
// target that, not the arithmetic.
//
// The golden composes both sites in full, with rmsnorm_unit modelled bit for bit
// (LANES=1: bf16 in, sequential fp32 sumsq, mean*1/LEN, +eps, the same Quake
// rsqrt, bf16(x*inv*gamma)), so the ONLY non-bitwise term in the whole chain is
// the one mhc_map_step already publishes -- its polynomial exp. Bounds are
// ABSOLUTE for the same reason as mhc-site: comb is doubly stochastic and its
// terms cancel, so a relative bound on the mixed streams gates nothing.
//
// STUB SUBLAYER: sub_out = 0.5 * sub_vec, exact in bf16 and COUPLED to the norm's
// output, so a skipped or mis-routed norm cannot hide behind it.
//
// Must FAIL: -DINJ_HCB_SAME_WEIGHTS, -DINJ_HCB_NORM_SWAP, -DINJ_HCB_SKIP_NORM,
//            -DINJ_HCB_STALE_STREAMS
//============================================================================
`timescale 1ns/1ps
`include "glm_fp.vh"
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_hc_block_vec.txt"
`endif
`ifndef TB_ABS_N
    `define TB_ABS_N 0.02
`endif
`ifndef TB_ABS_S
    `define TB_ABS_S 0.05
`endif

module glm53f_hc_block_tb;
    localparam integer H = 4, D = 64, QK = 32;
    localparam integer K = H*D, ROWS = (2+H)*H, NB = K/QK, NS = H*D;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0, sload = 0;
    reg  [32*NS-1:0]      s_init;
    reg  [8*ROWS*K-1:0]   aq, fq;
    reg  [16*ROWS*NB-1:0] ad, fd;
    reg  [32*ROWS-1:0]    ab, fb;
    reg  [31:0]           a0,a1,a2, f0,f1,f2;
    reg  [16*D-1:0]       agam, fgam;

    wire busy, done, sub_start, sub_is_ffn;
    wire [16*D-1:0] sub_vec;
    wire [32*NS-1:0] s_cur;
    reg              sub_done;
    reg  [16*D-1:0]  sub_out;

    glm53f_hc_block #(.H(H), .D(D), .QK(QK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .streams_load(sload), .streams_init(s_init), .streams_cur(s_cur),
        .a_w_q(aq), .a_w_d(ad), .a_base(ab), .a_s0(a0), .a_s1(a1), .a_s2(a2),
        .f_w_q(fq), .f_w_d(fd), .f_base(fb), .f_s0(f0), .f_s1(f1), .f_s2(f2),
        .attn_norm_w(agam), .ffn_norm_w(fgam),
        .sub_start(sub_start), .sub_is_ffn(sub_is_ffn), .sub_vec(sub_vec),
        .sub_done(sub_done), .sub_out(sub_out));

    // ---- stub sublayer + capture of what each site actually handed it ----
    reg [16*D-1:0] seen_a, seen_f;
    reg            got_a, got_f;
    reg [1:0]      spipe;
    integer        si;
    always @(posedge clk) begin
        if (rst) begin spipe <= 2'd0; sub_done <= 1'b0; got_a <= 1'b0; got_f <= 1'b0; end
        else begin
            sub_done <= 1'b0;
            if (sub_start) begin
                if (sub_is_ffn) begin seen_f <= sub_vec; got_f <= 1'b1; end
                else            begin seen_a <= sub_vec; got_a <= 1'b1; end
                spipe <= 2'd1;
            end else if (spipe != 2'd0) begin
                if (spipe == 2'd2) begin
                    for (si = 0; si < D; si = si + 1)
                        sub_out[16*si +: 16] <= fp32_to_bf16(
                            fp32_mul(bf16_to_fp32(sub_vec[16*si +: 16]), 32'h3F000000));
                    sub_done <= 1'b1; spipe <= 2'd0;
                end else spipe <= spipe + 2'd1;
            end
        end
    end

    integer fd_, code, t, i, ntest, hf, df, errors, checks, w;
    real    e, wn, ws;
    reg [31:0] t32; reg [15:0] t16; reg [7:0] t8;
    reg [15:0] e_na [0:D-1];
    reg [15:0] e_nf [0:D-1];
    reg [31:0] e_s  [0:NS-1];

    function real f2r(input [31:0] f);
        integer ex, i2; real m;
        begin
            ex = f[30:23];
            if (ex == 0) f2r = 0.0;
            else begin
                m = 1.0;
                for (i2 = 0; i2 < 23; i2 = i2 + 1)
                    if (f[22-i2]) m = m + (2.0 ** (-(i2+1)));
                f2r = m * (2.0 ** (ex - 127));
                if (f[31]) f2r = -f2r;
            end
        end
    endfunction
    function real b2r(input [15:0] b); begin b2r = f2r({b, 16'd0}); end endfunction
    function real absd(input real x); begin absd = (x < 0.0) ? -x : x; end endfunction

    initial begin
        errors = 0; checks = 0; wn = 0.0; ws = 0.0;
        fd_ = $fopen(`TB_VEC, "r");
        if (fd_ == 0) begin $display("[hc_block] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd_, "%d %d %d", ntest, hf, df);
        if (hf != H || df != D) begin
            $display("[hc_block] FAIL: vector H/D %0d/%0d != TB %0d/%0d", hf, df, H, D); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < NS; i = i + 1) begin code=$fscanf(fd_,"%h",t32); s_init[32*i +: 32]=t32; end
            for (i = 0; i < ROWS*K;  i=i+1) begin code=$fscanf(fd_,"%h",t8);  aq[8*i +: 8]=t8;  end
            for (i = 0; i < ROWS*NB; i=i+1) begin code=$fscanf(fd_,"%h",t16); ad[16*i +: 16]=t16; end
            for (i = 0; i < ROWS;    i=i+1) begin code=$fscanf(fd_,"%h",t32); ab[32*i +: 32]=t32; end
            code=$fscanf(fd_,"%h",t32); a0=t32; code=$fscanf(fd_,"%h",t32); a1=t32; code=$fscanf(fd_,"%h",t32); a2=t32;
            for (i = 0; i < D; i=i+1) begin code=$fscanf(fd_,"%h",t16); agam[16*i +: 16]=t16; end
            for (i = 0; i < ROWS*K;  i=i+1) begin code=$fscanf(fd_,"%h",t8);  fq[8*i +: 8]=t8;  end
            for (i = 0; i < ROWS*NB; i=i+1) begin code=$fscanf(fd_,"%h",t16); fd[16*i +: 16]=t16; end
            for (i = 0; i < ROWS;    i=i+1) begin code=$fscanf(fd_,"%h",t32); fb[32*i +: 32]=t32; end
            code=$fscanf(fd_,"%h",t32); f0=t32; code=$fscanf(fd_,"%h",t32); f1=t32; code=$fscanf(fd_,"%h",t32); f2=t32;
            for (i = 0; i < D; i=i+1) begin code=$fscanf(fd_,"%h",t16); fgam[16*i +: 16]=t16; end
            for (i = 0; i < D; i=i+1) begin code=$fscanf(fd_,"%h",t16); e_na[i]=t16; end
            for (i = 0; i < D; i=i+1) begin code=$fscanf(fd_,"%h",t16); e_nf[i]=t16; end
            for (i = 0; i < NS; i=i+1) begin code=$fscanf(fd_,"%h",t32); e_s[i]=t32; end

            got_a = 0; got_f = 0;
            @(negedge clk); sload = 1;
            @(negedge clk); sload = 0;
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 1000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1 || !got_a || !got_f) begin
                $display("[hc_block] FAIL t%0d: done=%b got_a=%b got_f=%b after %0d cycles",
                         t, done, got_a, got_f, w);
                errors = errors + 1;
            end

            for (i = 0; i < D; i = i + 1) begin
                checks = checks + 1;
                e = absd(b2r(seen_a[16*i +: 16]) - b2r(e_na[i]));
                if (e > wn) wn = e;
                if (e > `TB_ABS_N) begin
                    $display("FAIL t%0d attn normed[%0d]: got %h exp %h (abs %e)",
                             t, i, seen_a[16*i +: 16], e_na[i], e); errors = errors + 1;
                end
                checks = checks + 1;
                e = absd(b2r(seen_f[16*i +: 16]) - b2r(e_nf[i]));
                if (e > wn) wn = e;
                if (e > `TB_ABS_N) begin
                    $display("FAIL t%0d ffn normed[%0d]: got %h exp %h (abs %e)",
                             t, i, seen_f[16*i +: 16], e_nf[i], e); errors = errors + 1;
                end
            end
            for (i = 0; i < NS; i = i + 1) begin
                checks = checks + 1;
                e = absd(f2r(s_cur[32*i +: 32]) - f2r(e_s[i]));
                if (e > ws) ws = e;
                if (e > `TB_ABS_S) begin
                    $display("FAIL t%0d streams[%0d][%0d]: got %h exp %h (abs %e)",
                             t, i/D, i%D, s_cur[32*i +: 32], e_s[i], e); errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd_);
        if (errors == 0)
            $display("[hc_block] ALL %0d TESTS PASSED (%0d blocks H=%0d D=%0d: TWO mHC sites from one instance, weights and norms routed per site, streams threaded attn->FFN, within abs %0.3f / %0.3f -- worst normed %e, worst streams %e)",
                     checks, ntest, H, D, `TB_ABS_N, `TB_ABS_S, wn, ws);
        else
            $display("[hc_block] %0d/%0d FAILED (worst normed %e, streams %e)", errors, checks, wn, ws);
        $finish;
    end
    initial begin #1000000000; $display("[hc_block] FAIL: timeout"); $finish; end
endmodule
