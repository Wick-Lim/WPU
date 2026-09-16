//============================================================================
// glm53f_mla_score_tb.v -- src/glm53f_mla_score.v against
// tools/glm53f_mla_score_gen.py.
//
// BITWISE, not a tolerance, and that is a claim about the unit rather than luck:
// every piece it uses already has a bit-exact Python twin in this repo
// (rmsnorm_unit at LANES=1, glm_softmax, and glm_fp.vh's bf16/fp32 semantics), so
// this unit adds a new DATAFLOW and no new numerics. If it ever stops being
// bitwise, something changed in the arithmetic and the right response is to find
// out what, not to widen a bound.
//
// The corpus deliberately contains windows SHORTER than SMAX, because a unit that
// never wrote the -inf pad would pass every full-window test. The generator
// asserts both cases are present.
//
// Must FAIL:
//   INJ_MLAS_NOPAD   leave unused slots at +0 instead of -inf, so padding gets
//                    softmax weight -- the classic masking bug, and invisible
//                    whenever s_len == SMAX.
//   INJ_MLAS_NORESCALE  drop the 1/sqrt(qk) score scale.
//   INJ_MLAS_HEAD0_PROBS  weight every head's latent sum with head 0's
//                    probabilities -- right shape, right magnitudes, wrong
//                    attention: the failure a shape-only check misses.
// Each was measured against the golden BEFORE being written (6 windows, 96 ctx
// elements): NOPAD 48, NORESCALE 80, HEAD0_PROBS 40. NOPAD moves only the padded
// windows, which is the corpus requirement above, stated as a number.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_mla_score_vec.txt"
`endif

module glm53f_mla_score_tb;
    localparam integer H=2, KVL=8, SMAX=4;
    localparam integer JW=$clog2(SMAX), SW=$clog2(SMAX+1);

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*H*KVL-1:0] qa;
    reg  [SW-1:0]       slen;
    wire                busy, done, c_req;
    wire [JW-1:0]       c_idx;
    reg  [16*KVL-1:0]   c_vec;
    wire [16*H*KVL-1:0] ctx;

    // 1/sqrt(KVL) at the slice, matching the generator
    localparam [31:0] SCALE_TB = 32'h3EB504F3;   // 1/sqrt(8) = 0.35355338

    glm53f_mla_score #(.H(H), .KVL(KVL), .SMAX(SMAX), .SCALE(SCALE_TB)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .qa_in(qa), .s_len(slen),
        .c_req(c_req), .c_idx(c_idx), .c_vec(c_vec), .ctx_out(ctx));

    // the latent cache the DUT pulls from -- flat, explicit sensitivity
    reg [15:0] cache [0:SMAX*KVL-1];
    reg        ld_tick = 1'b0;
    integer    kk;
    always @(c_idx, ld_tick) begin
        for (kk = 0; kk < KVL; kk = kk + 1)
            c_vec[16*kk +: 16] = cache[c_idx*KVL + kk];
    end

    integer fd, code, t, i, ntest, e_h, e_k, e_s, errors, checks, w, sl;
    reg [15:0] t16;
    reg [15:0] e_ctx [0:H*KVL-1];

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mla_score] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d", ntest, e_h, e_k, e_s);
        if (e_h != H || e_k != KVL || e_s != SMAX) begin
            $display("[mla_score] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d", sl); slen = sl[SW-1:0];
            for (i = 0; i < H*KVL;    i = i + 1) begin code=$fscanf(fd,"%h",t16); qa[16*i +: 16]=t16; end
            for (i = 0; i < SMAX*KVL; i = i + 1) begin code=$fscanf(fd,"%h",t16); cache[i]=t16; end
            for (i = 0; i < H*KVL;    i = i + 1) begin code=$fscanf(fd,"%h",t16); e_ctx[i]=t16; end
            ld_tick = ~ld_tick;

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 2000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[mla_score] FAIL t%0d (s_len=%0d): done never asserted", t, sl);
                errors = errors + 1;
            end
            for (i = 0; i < H*KVL; i = i + 1) begin
                checks = checks + 1;
                if (ctx[16*i +: 16] !== e_ctx[i]) begin
                    if (errors < 8)
                        $display("FAIL t%0d (s_len=%0d) ctx[h=%0d][k=%0d]: got %h exp %h",
                                 t, sl, i / KVL, i % KVL, ctx[16*i +: 16], e_ctx[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mla_score] ALL %0d TESTS PASSED (%0d windows, H=%0d kv_lora=%0d SMAX=%0d: the absorbed-latent MLA inner loop -- score against the normalised latent, softmax with the unused slots pinned to -inf, and the weighted latent sum -- BITWISE against the reference)",
                     checks, ntest, H, KVL, SMAX);
        else
            $display("[mla_score] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #200000000; $display("[mla_score] FAIL: timeout"); $finish; end
endmodule
