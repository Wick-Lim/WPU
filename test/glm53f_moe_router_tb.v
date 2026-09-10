//============================================================================
// glm53f_moe_router_tb.v -- GLM-5.3-Flash's MoE router
// (src/glm53f_moe_router.v, vectors from tools/glm53f_moe_router_gen.py).
//
// Golden is glm53_flash_ref.moe_route, built from the checkpoint's own metadata:
// sigmoid gating (expert_gating_func = 2), top-k over scores + exp_probs_b,
// renormalise (expert_weights_norm = True), scale 2.5.
//
// THE SELECTED SET is checked exactly, the weights by tolerance -- because
// selection is DISCRETE. Getting an expert wrong runs a different expert; getting
// a weight slightly wrong is a small numeric error. Those are not the same
// failure and are not gated the same way.
//   The check is ORDER-INDEPENDENT on purpose. The reference returns the indices
// ascending; topk_select returns them score-descending. Neither is more correct --
// the model sums over the selected experts, so the emission order is a convention,
// not semantics. Comparing sets and then matching each weight TO ITS OWN INDEX
// checks what actually matters and cannot be fooled by a convention change.
//   The weight tolerance is relative-plus-absolute because a flat 0.01 is BELOW
// one bf16 ULP at these magnitudes (ULP is 0.0156 near 2.0) -- a bound no correct
// DUT could meet.
//   The corpus excludes draws whose top-k margin is under 1e-3, where the DUT's
// fp32 sigmoid and the reference's float64 one could legitimately disagree.
//
// Must FAIL: -DINJ_MOER_NO_BIAS (exp_probs_b never reaches the selection) and
//            -DINJ_MOER_BIAS_WEIGHTS (the OTHER reading of exp_probs_b: weight by
//            the biased score instead of the unbiased sigmoid -- same experts,
//            different weights, which is exactly why it needs a gate).
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_moe_router_vec.txt"
`endif
`ifndef TB_ABS
    `define TB_ABS 0.01
`endif
`ifndef TB_REL
    `define TB_REL 0.02
`endif

module glm53f_moe_router_tb;
    localparam integer HIDDEN=16, E=8, TOPK=2, IDXW=$clog2(E);

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*HIDDEN-1:0] x_in;
    reg  [32*E-1:0]      bias_in;
    wire busy, done, w_req;
    wire [$clog2(HIDDEN+1)-1:0] w_k;
    reg  [32*E-1:0] w_row;
    wire [TOPK*IDXW-1:0] sel_idx;
    wire [TOPK*16-1:0]   sel_weight;

    glm53f_moe_router #(.HIDDEN(HIDDEN), .E(E), .TOPK(TOPK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_in(x_in), .w_req(w_req), .w_k(w_k), .w_row(w_row), .bias_in(bias_in),
        .sel_idx(sel_idx), .sel_weight(sel_weight));

    // W_g served combinationally by row k, the way the system would
    reg [31:0] wg [0:HIDDEN*E-1];
    integer wj;
    always @* begin
        for (wj = 0; wj < E; wj = wj + 1)
            w_row[32*wj +: 32] = wg[w_k*E + wj];
    end

    integer fd, code, t, i, j, mt, ntest, p_h, p_e, p_k, errors, checks, w;
    integer e_idx [0:TOPK-1];
    reg [15:0] e_w [0:TOPK-1];
    real ee, wa, tolw;
    reg [31:0] t32; reg [15:0] t16;

    function real b2r(input [15:0] b);
        integer ex, i2; real m;
        begin
            ex = b[14:7];
            if (ex == 0) b2r = 0.0;
            else begin
                m = 1.0;
                for (i2 = 0; i2 < 7; i2 = i2 + 1) if (b[6-i2]) m = m + (2.0 ** (-(i2+1)));
                b2r = m * (2.0 ** (ex - 127));
                if (b[15]) b2r = -b2r;
            end
        end
    endfunction
    function real ab_(input real x); begin ab_ = (x<0.0)?-x:x; end endfunction

    initial begin
        errors = 0; checks = 0; wa = 0.0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[moe_router] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d", ntest, p_h, p_e, p_k);
        if (p_h != HIDDEN || p_e != E || p_k != TOPK) begin
            $display("[moe_router] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i=0;i<HIDDEN;i=i+1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i=0;i<HIDDEN;i=i+1)
                for (j=0;j<E;j=j+1) begin code=$fscanf(fd,"%h",t32); wg[i*E+j]=t32; end
            for (i=0;i<E;i=i+1)     begin code=$fscanf(fd,"%h",t32); bias_in[32*i +: 32]=t32; end
            for (i=0;i<TOPK;i=i+1)  begin code=$fscanf(fd,"%d",e_idx[i]); end
            for (i=0;i<TOPK;i=i+1)  begin code=$fscanf(fd,"%h",t16); e_w[i]=t16; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 100000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[moe_router] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end
            for (i = 0; i < TOPK; i = i + 1) begin
                // find this DUT index in the golden's set (order is a convention)
                mt = -1;
                for (j = 0; j < TOPK; j = j + 1)
                    if (sel_idx[IDXW*i +: IDXW] === e_idx[j][IDXW-1:0]) mt = j;
                checks = checks + 1;
                if (mt < 0) begin
                    $display("FAIL t%0d: expert %0d selected by the DUT is not in the golden set",
                             t, sel_idx[IDXW*i +: IDXW]);
                    errors = errors + 1;
                end else begin
                    checks = checks + 1;
                    ee = ab_(b2r(sel_weight[16*i +: 16]) - b2r(e_w[mt]));
                    if (ee > wa) wa = ee;
                    tolw = `TB_REL * ab_(b2r(e_w[mt])) + `TB_ABS;
                    if (ee > tolw) begin
                        $display("FAIL t%0d expert %0d weight: got %h (%f) exp %h (%f) tol %f",
                                 t, sel_idx[IDXW*i +: IDXW], sel_weight[16*i +: 16],
                                 b2r(sel_weight[16*i +: 16]), e_w[mt], b2r(e_w[mt]), tolw);
                        errors = errors + 1;
                    end
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[moe_router] ALL %0d TESTS PASSED (%0d tokens, E=%0d TOPK=%0d: F32 gate GEMV, fp32 sigmoid, top-k over scores+exp_probs_b, renormalised and scaled -- the selected SET exact, weights within rel %0.3f + abs %0.3f of their own index, worst %e)",
                     checks, ntest, E, TOPK, `TB_REL, `TB_ABS, wa);
        else
            $display("[moe_router] %0d/%0d FAILED (worst weight abs %e)", errors, checks, wa);
        $finish;
    end
    initial begin #20000000; $display("[moe_router] FAIL: timeout"); $finish; end
endmodule
