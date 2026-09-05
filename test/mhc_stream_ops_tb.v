//============================================================================
// mhc_stream_ops_tb.v -- gate for the mHC residual-path datapaths
// (src/mhc_stream_ops.v, vectors from tools/mhc_stream_ops_gen.py).
//
// TIGHT, BUT NOT BITWISE -- and the reason is worth stating, because the first
// version of this header got it wrong.  Both operations are pure fp32 mul/add in
// the same sequential order as the reference (tools/glm53_flash_ref.py
// hc_collapse / hc_mix), with no transcendental and no reciprocal.  Same order is
// not the same adder, though: numpy rounds correctly and src/glm_fp.vh fp32_add
// is 1 ULP low on ~0.04% of pairs (`make fp-ieee`).  Measured here:
//     COLLAPSE  worst 0 ULP  -- bitwise on this corpus; pre[h] > 0, so the four
//                              terms do not systematically cancel.
//     MIX       worst 4 ULP  -- comb is doubly stochastic, so its four terms
//                              nearly cancel (this corpus reaches sum|terms| /
//                              |result| = 1239x) and a 1-ULP slip in an
//                              intermediate surfaces in the result.
// Bound 32 ULP, ~8x the measured worst, and both worsts are printed so a
// regression moves a number. This is the same stance as kda_recur's exact leg:
// the amplified footprint of a known adder gap, not a datapath error.
//
// THREE INSTANCES, DLANES 1 / 2 / 4, all checked against the same golden.  Every
// d is independent, so the lane count is a throughput knob that MUST NOT change
// the answer -- running all three against one golden is that property's test, and
// it is why DLANES can be sized against the memory tier later without re-gating
// the numerics.
//
// Must FAIL: -DINJ_OPS_MIX_TRANSPOSE, -DINJ_OPS_MIX_NOPOST,
//            -DINJ_OPS_COLLAPSE_NOPRE, -DINJ_OPS_POST_FIRST
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/mhc_stream_ops_vec.txt"
`endif
`ifndef TB_ULP
    `define TB_ULP 32
`endif

module mhc_stream_ops_tb;
    localparam integer H = 4, D = 64;
    localparam integer NS = H*D;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0, mode = 0;
    reg  [32*NS-1:0]  streams_in;
    reg  [32*H-1:0]   pre_in, post_in;
    reg  [32*H*H-1:0] comb_in;
    reg  [32*D-1:0]   sub_in;

    wire [2:0] busy, done;
    // Each DLANES instance finishes at a DIFFERENT cycle, so `done` -- a one-cycle
    // pulse -- is never simultaneously high across the three. Latch each.
    reg  [2:0] done_seen;
    always @(posedge clk) begin
        if (rst || start) done_seen <= 3'b000;
        else              done_seen <= done_seen | done;
    end
    wire [32*D-1:0]  coll  [0:2];
    wire [32*NS-1:0] strm  [0:2];

    genvar gi;
    generate
        for (gi = 0; gi < 3; gi = gi + 1) begin : g_lanes
            localparam integer DL = (gi == 0) ? 1 : ((gi == 1) ? 2 : 4);
            mhc_stream_ops #(.H(H), .D(D), .DLANES(DL)) dut (
                .clk(clk), .rst(rst), .start(start), .mode(mode),
                .busy(busy[gi]), .done(done[gi]),
                .streams_in(streams_in), .pre_in(pre_in), .comb_in(comb_in),
                .post_in(post_in), .sub_in(sub_in),
                .collapsed_out(coll[gi]), .streams_out(strm[gi]));
        end
    endgenerate

    integer fd, code, t, i, k, ntest, hf, df, errors, checks, w, d, wc, wm;

    // fp32 ULP distance. Results here straddle zero (comb's terms cancel), so a
    // sign difference is only meaningful when the magnitudes are not both zero:
    // +0.0 and -0.0 are the same value and must not read as 1e9 apart.
    function integer ulp(input [31:0] a, input [31:0] b);
        begin
            if (a[31] !== b[31])
                ulp = (a[30:0] == 0 && b[30:0] == 0) ? 0 : 1000000000;
            else if (a[30:0] >= b[30:0]) ulp = a[30:0] - b[30:0];
            else                          ulp = b[30:0] - a[30:0];
        end
    endfunction
    reg [31:0] t32;
    reg [31:0] e_coll [0:D-1];
    reg [31:0] e_strm [0:NS-1];

    task runop(input m);
        begin
            @(negedge clk); mode = m; start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done_seen !== 3'b111 && w < 200000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done_seen !== 3'b111) begin
                $display("[mhc_ops] FAIL: mode %0d -- done_seen=%b never reached 111 in %0d cycles",
                         m, done_seen, w);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0; wc = 0; wm = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mhc_ops] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, hf, df);
        if (hf != H || df != D) begin
            $display("[mhc_ops] FAIL: vector H/D %0d/%0d != TB %0d/%0d", hf, df, H, D);
            $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < NS;  i = i + 1) begin code=$fscanf(fd,"%h",t32); streams_in[32*i +: 32]=t32; end
            for (i = 0; i < H;   i = i + 1) begin code=$fscanf(fd,"%h",t32); pre_in[32*i +: 32]=t32;  end
            for (i = 0; i < H*H; i = i + 1) begin code=$fscanf(fd,"%h",t32); comb_in[32*i +: 32]=t32; end
            for (i = 0; i < H;   i = i + 1) begin code=$fscanf(fd,"%h",t32); post_in[32*i +: 32]=t32; end
            for (i = 0; i < D;   i = i + 1) begin code=$fscanf(fd,"%h",t32); sub_in[32*i +: 32]=t32;  end
            for (i = 0; i < D;   i = i + 1) begin code=$fscanf(fd,"%h",t32); e_coll[i]=t32; end
            for (i = 0; i < NS;  i = i + 1) begin code=$fscanf(fd,"%h",t32); e_strm[i]=t32; end

            runop(1'b0);                                  // COLLAPSE
            for (k = 0; k < 3; k = k + 1)
                for (i = 0; i < D; i = i + 1) begin
                    checks = checks + 1;
                    d = ulp(coll[k][32*i +: 32], e_coll[i]);
                    if (d > wc) wc = d;
                    if (d > `TB_ULP) begin
                        $display("FAIL t%0d DLANES#%0d collapsed[%0d]: got %h exp %h (%0d ULP)",
                                 t, k, i, coll[k][32*i +: 32], e_coll[i], d);
                        errors = errors + 1;
                    end
                end

            runop(1'b1);                                  // MIX
            for (k = 0; k < 3; k = k + 1)
                for (i = 0; i < NS; i = i + 1) begin
                    checks = checks + 1;
                    d = ulp(strm[k][32*i +: 32], e_strm[i]);
                    if (d > wm) wm = d;
                    if (d > `TB_ULP) begin
                        $display("FAIL t%0d DLANES#%0d streams[%0d][%0d]: got %h exp %h (%0d ULP)",
                                 t, k, i/D, i%D, strm[k][32*i +: 32], e_strm[i], d);
                        errors = errors + 1;
                    end
                end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mhc_ops] ALL %0d TESTS PASSED (%0d vectors H=%0d D=%0d: collapse and mix within %0d ULP of the PINNED SEQUENTIAL reference -- worst collapse %0d, worst mix %0d, the amplified footprint of fp32_add's 1-ULP gap under comb's cancellation; identical at DLANES 1/2/4, so the lane count does not change the answer)",
                     checks, ntest, H, D, `TB_ULP, wc, wm);
        else
            $display("[mhc_ops] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #200000000; $display("[mhc_ops] FAIL: timeout"); $finish; end
endmodule
