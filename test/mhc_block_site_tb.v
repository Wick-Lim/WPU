//============================================================================
// mhc_block_site_tb.v -- gate for ONE whole mHC site
// (src/mhc_block_site.v, vectors from tools/mhc_block_site_gen.py).
//
// The golden composes the SPECIFICATION -- the bit-exact GEMV emulation, then
// tools/mhc_map_gen.ref_map (float64), then glm53_flash_ref's pinned hc_collapse
// / hc_mix -- so passing means the composition reproduces the spec, not merely
// that three units were wired together.
//
// The bound is implied by mhc_map_step's OWN gate (1024 ULP on pre/post, 16384 on
// comb): the generator perturbs those by exactly those amounts and pushes them
// through collapse and mix. This TB then adds ~2x headroom on the result. Nothing
// here is read off this DUT.
//
// STUB SUBLAYER: sub_out = 0.5 * collapsed. Exact in fp32 and COUPLED to
// collapsed, so a broken collapse cannot hide behind an independent value. The
// block's own attn_norm / ffn_norm would sit between collapsed_out and the real
// sublayer -- outside this module, per the GGUF census.
//
// Must FAIL: -DINJ_SITE_IGNORE_SUB, -DINJ_SITE_NO_UPDATE, -DINJ_SITE_PRE_FOR_POST
//============================================================================
`timescale 1ns/1ps
`include "glm_fp.vh"
`ifndef TB_VEC
    `define TB_VEC "build/mhc_block_site_vec.txt"
`endif
// ABSOLUTE bounds, not ULP: comb is doubly stochastic, so the mix's four terms
// nearly cancel and some outputs land near zero. A ULP or relative bound on those
// is dominated by the cancellation -- the generator's own envelope comes out at
// 1.4e6 ULP, which at these magnitudes is ~17% and gates nothing. Measured
// envelope implied by mhc_map's gate: collapsed 9.8e-4, streams 8.6e-3, against
// a corpus peak |value| of 14.8. Bounds below are ~4x that envelope.
`ifndef TB_ABS_COLL
    `define TB_ABS_COLL 0.004
`endif
`ifndef TB_ABS_STRM
    `define TB_ABS_STRM 0.035
`endif

module mhc_block_site_tb;
    localparam integer H = 4, D = 64, QK = 32;
    localparam integer K = H*D, ROWS = (2+H)*H, NB = K/QK, NS = H*D;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0, sload = 0;
    reg  [32*NS-1:0]      s_init;
    reg  [8*ROWS*K-1:0]   w_q;
    reg  [16*ROWS*NB-1:0] w_d;
    reg  [32*ROWS-1:0]    base_in;
    reg  [31:0]           sc0, sc1, sc2;

    wire busy, done, sub_start;
    wire [32*NS-1:0] s_cur;
    wire [32*D-1:0]  coll;
    reg              sub_done;
    reg  [32*D-1:0]  sub_in;

    mhc_block_site #(.H(H), .D(D), .QK(QK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .streams_load(sload), .streams_init(s_init), .streams_cur(s_cur),
        .w_q(w_q), .w_d(w_d), .base_in(base_in),
        .scale0(sc0), .scale1(sc1), .scale2(sc2),
        .sub_start(sub_start), .collapsed_out(coll),
        .sub_done(sub_done), .sub_in(sub_in));

    // ---- stub sublayer: sub_out = 0.5 * collapsed, two cycles after sub_start ----
    integer si;
    reg [1:0] sub_pipe;
    always @(posedge clk) begin
        if (rst) begin sub_pipe <= 2'd0; sub_done <= 1'b0; end
        else begin
            sub_done <= 1'b0;
            if (sub_start) sub_pipe <= 2'd1;
            else if (sub_pipe != 2'd0) begin
                if (sub_pipe == 2'd2) begin
                    for (si = 0; si < D; si = si + 1)
                        sub_in[32*si +: 32] <= fp32_mul(coll[32*si +: 32], 32'h3F000000);
                    sub_done <= 1'b1;
                    sub_pipe <= 2'd0;
                end else sub_pipe <= sub_pipe + 2'd1;
            end
        end
    end

    integer fd, code, t, i, ntest, hf, df, errors, checks, w;
    real    e, wc, ws;
    reg [31:0] t32; reg [15:0] t16; reg [7:0] t8;
    reg [31:0] e_coll [0:D-1];
    reg [31:0] e_strm [0:NS-1];

    function real f2r(input [31:0] f);
        integer ex, i; real m;
        begin
            ex = f[30:23];
            if (ex == 0) f2r = 0.0;
            else begin
                m = 1.0;
                for (i = 0; i < 23; i = i + 1)
                    if (f[22-i]) m = m + (2.0 ** (-(i+1)));
                f2r = m * (2.0 ** (ex - 127));
                if (f[31]) f2r = -f2r;
            end
        end
    endfunction

    function real aerr(input [31:0] a, input [31:0] b);
        real x;
        begin x = f2r(a) - f2r(b); aerr = (x < 0.0) ? -x : x; end
    endfunction

    initial begin
        errors = 0; checks = 0; wc = 0.0; ws = 0.0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mhc_site] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, df);
        if (hf != H || df != D) begin
            $display("[mhc_site] FAIL: vector H/D %0d/%0d != TB %0d/%0d", hf, df, H, D);
            $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < NS;      i = i + 1) begin code=$fscanf(fd,"%h",t32); s_init[32*i +: 32]=t32; end
            for (i = 0; i < ROWS*K;  i = i + 1) begin code=$fscanf(fd,"%h",t8);  w_q[8*i +: 8]=t8;  end
            for (i = 0; i < ROWS*NB; i = i + 1) begin code=$fscanf(fd,"%h",t16); w_d[16*i +: 16]=t16; end
            for (i = 0; i < ROWS;    i = i + 1) begin code=$fscanf(fd,"%h",t32); base_in[32*i +: 32]=t32; end
            code=$fscanf(fd,"%h",t32); sc0=t32;
            code=$fscanf(fd,"%h",t32); sc1=t32;
            code=$fscanf(fd,"%h",t32); sc2=t32;
            for (i = 0; i < D;  i = i + 1) begin code=$fscanf(fd,"%h",t32); e_coll[i]=t32; end
            for (i = 0; i < NS; i = i + 1) begin code=$fscanf(fd,"%h",t32); e_strm[i]=t32; end

            @(negedge clk); sload = 1;
            @(negedge clk); sload = 0;
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 400000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[mhc_site] FAIL t%0d: done never asserted (%0d cycles)", t, w);
                errors = errors + 1;
            end

            for (i = 0; i < D; i = i + 1) begin
                checks = checks + 1;
                e = aerr(coll[32*i +: 32], e_coll[i]);
                if (e > wc) wc = e;
                if (e > `TB_ABS_COLL) begin
                    $display("FAIL t%0d collapsed[%0d]: got %h exp %h (abs %e)",
                             t, i, coll[32*i +: 32], e_coll[i], e);
                    errors = errors + 1;
                end
            end
            for (i = 0; i < NS; i = i + 1) begin
                checks = checks + 1;
                e = aerr(s_cur[32*i +: 32], e_strm[i]);
                if (e > ws) ws = e;
                if (e > `TB_ABS_STRM) begin
                    $display("FAIL t%0d streams[%0d][%0d]: got %h exp %h (abs %e)",
                             t, i/D, i%D, s_cur[32*i +: 32], e_strm[i], e);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mhc_site] ALL %0d TESTS PASSED (%0d sites H=%0d D=%0d: fn GEMV -> map -> collapse -> stub sublayer -> mix, streams carried, within abs %0.4f / %0.4f of the composed SPEC -- worst collapsed %e, worst streams %e)",
                     checks, ntest, H, D, `TB_ABS_COLL, `TB_ABS_STRM, wc, ws);
        else
            $display("[mhc_site] %0d/%0d FAILED (worst collapsed %e, streams %e)",
                     errors, checks, wc, ws);
        $finish;
    end
    initial begin #400000000; $display("[mhc_site] FAIL: timeout"); $finish; end
endmodule
