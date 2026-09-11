//============================================================================
// glm53f_swiglu_mt_tb.v -- glm53f_swiglu_mt on the type combinations the MoE
// experts actually use (vectors from tools/glm53f_swiglu_mt_gen.py).
//
// `make swiglu-q8` runs this same module in its Q8_0 configuration at the decoder
// block's slice. THIS gate is a different claim: that the RUNTIME weight type
// selects the right BUS. The engine reads Q4_K's code off w_q and every other
// type's off w_hp, with headers on w_d/w_dmin/w_scales, w_q6_sc and w_q8_d -- so
// the claim is plumbing, and the per-type arithmetic is already gated by
// `make mixedtype`.
//
// K = 256 here because Q4_K/Q5_K/Q6_K are 256-weight super-blocks. The MoE loop
// wants a much smaller slice, so the two claims are gated separately rather than
// at one compromise slice that would test neither well.
//
// The combinations are the checkpoint's [scan], not invented:
//     (Q4_K, Q4_K, Q5_K)   ffn_{gate,up}_exps Q4_K x42, ffn_down_exps Q5_K x40
//     (Q5_K, Q5_K, Q6_K)   the one Q5_K gate/up block, and the three Q6_K downs
//
// Must FAIL: -DINJ_SWQ8_Q4K_TYPE (w_type forced to Q4_K whatever the caller asked
// for). That is the whole point of a runtime type: with it forced, a Q5_K or Q6_K
// pass reads the wrong bus and decodes another type's bytes.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_swiglu_mt_vec.txt"
`endif

module glm53f_swiglu_mt_tb;
    localparam integer HIDDEN=256, INTER=256, TN=4, KMAX=256;
    localparam integer NSB=(KMAX+255)/256, NB8=KMAX/32;
    localparam integer GW=$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1);
    localparam integer MAXROW=(INTER>HIDDEN)?INTER:HIDDEN;
    localparam integer MAXK  =(INTER>HIDDEN)?INTER:HIDDEN;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*HIDDEN-1:0] x_in;
    reg  [2:0] wt_g, wt_u, wt_d;
    wire busy, done, w_req;
    wire [1:0] w_sel;
    wire [GW-1:0] w_grp;
    wire [$clog2(KMAX+1)-1:0] w_k;
    reg  [4*TN-1:0]          w_q;
    reg  [16*TN-1:0]         w_hp;
    reg  [16*TN*NSB-1:0]     w_d, w_dmin;
    reg  [96*TN*NSB-1:0]     w_scales;
    reg  [128*TN*NSB-1:0]    w_q6sc;
    reg  [16*TN*NB8-1:0]     w_q8d;
    wire [16*HIDDEN-1:0]     y_out;

    glm53f_swiglu_mt #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_in(x_in),
        .wt_gate(wt_g), .wt_up(wt_u), .wt_down(wt_d),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_q(w_q), .w_hp(w_hp), .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .w_q6_sc(w_q6sc), .w_q8_d(w_q8d), .y_out(y_out));

    // FLAT 1-D stores, indexed by a computed base.  Deliberately not the obvious
    // [pass][col][k] arrays: iverilog enumerates every element of a multi-dim
    // array into the sensitivity list of an `always @*` that reads it with a
    // variable index, and at 3*256*256 codes that compile never finishes (killed
    // at 10 min).  `make swiglu-q8`'s TB already uses this flat shape.
    localparam integer CSTR = MAXROW*MAXK;     // per-pass code stride
    localparam integer HSTR = MAXROW*NSB;      // per-pass header stride
    localparam integer QSTR = MAXROW*NB8;      // per-pass Q8_0 d stride
    reg [15:0]  cd   [0:3*HSTR-1];
    reg [15:0]  cdm  [0:3*HSTR-1];
    reg [95:0]  csc  [0:3*HSTR-1];
    reg [127:0] cq6  [0:3*HSTR-1];
    reg [15:0]  cq8  [0:3*QSTR-1];
    reg [15:0]  ccode[0:3*CSTR-1];

    // ld_tick toggles after each load so the fan-out re-evaluates even in the
    // (impossible here, but free to guard) case that a test opens on the same
    // (sel,grp,k) the previous one closed on.
    reg ld_tick = 1'b0;

    integer pj, sb, bb, hb, cb, qb, col;
    // EXPLICIT sensitivity, NOT `always @*`.  Reading a memory at a variable index
    // inside `always @*` makes iverilog enumerate EVERY element into the
    // sensitivity list; at 3*256*256 codes the compile does not terminate
    // (measured: >120 s and still climbing on a reduced probe, vs 1.0 s with this
    // list -- and the DUT itself elaborates in 1.0 s at KMAX=256, so the cost was
    // entirely here).  The stores are written only before `start`, so the request
    // signals plus ld_tick are the complete set of things that change the bus.
    always @(w_sel, w_grp, w_k, ld_tick) begin
        hb = w_sel*HSTR; cb = w_sel*CSTR; qb = w_sel*QSTR;
        w_q = 0; w_hp = 0; w_d = 0; w_dmin = 0; w_scales = 0; w_q6sc = 0; w_q8d = 0;
        for (pj = 0; pj < TN; pj = pj + 1) begin
            col = w_grp*TN + pj;
            // Q4_K's code rides w_q (4 bits); every other type rides w_hp.
            w_q [4*pj  +: 4]  = ccode[cb + col*MAXK + w_k][3:0];
            w_hp[16*pj +: 16] = ccode[cb + col*MAXK + w_k];
            for (sb = 0; sb < NSB; sb = sb + 1) begin
                w_d     [16*(pj*NSB+sb)  +: 16]  = cd [hb + col*NSB + sb];
                w_dmin  [16*(pj*NSB+sb)  +: 16]  = cdm[hb + col*NSB + sb];
                w_scales[96*(pj*NSB+sb)  +: 96]  = csc[hb + col*NSB + sb];
                w_q6sc  [128*(pj*NSB+sb) +: 128] = cq6[hb + col*NSB + sb];
            end
            for (bb = 0; bb < NB8; bb = bb + 1)
                w_q8d[16*(pj*NB8+bb) +: 16] = cq8[qb + col*NB8 + bb];
        end
    end

    integer fd, code, t, i, p, c, ntest, e_h, e_i, e_t, errors, checks, w;
    integer rows [0:2];
    integer kk   [0:2];
    real ee, wa, tl;
    reg [15:0] t16; reg [7:0] t8; reg [95:0] t96; reg [127:0] t128;
    reg [15:0] e_y [0:HIDDEN-1];
    real       e_to[0:HIDDEN-1];

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

    task read_pass(input integer pp, input integer nrow, input integer nk);
        begin
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin code=$fscanf(fd,"%h",t16); cd[pp*HSTR + c*NSB + sb]=t16; end
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin code=$fscanf(fd,"%h",t16); cdm[pp*HSTR + c*NSB + sb]=t16; end
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin code=$fscanf(fd,"%h",t96); csc[pp*HSTR + c*NSB + sb]=t96; end
            // 16 separate bytes, scale i at bit offset 8*i -- glm_matmul_mixed_tb.v's
            // convention.  One 128-bit word would reverse them.
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin
                    cq6[pp*HSTR + c*NSB + sb] = 0;
                    for (bb = 0; bb < 16; bb = bb + 1) begin
                        code=$fscanf(fd,"%h",t8);
                        cq6[pp*HSTR + c*NSB + sb][8*bb +: 8] = t8;
                    end
                end
            for (c = 0; c < nrow; c = c + 1)
                for (bb = 0; bb < NB8; bb = bb + 1) begin code=$fscanf(fd,"%h",t16); cq8[pp*QSTR + c*NB8 + bb]=t16; end
            for (c = 0; c < nrow; c = c + 1)
                for (i = 0; i < nk; i = i + 1) begin code=$fscanf(fd,"%h",t16); ccode[pp*CSTR + c*MAXK + i]=t16; end
        end
    endtask

    initial begin
        errors = 0; checks = 0; wa = 0.0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[swiglu_mt] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d", ntest, e_h, e_i, e_t);
        if (e_h != HIDDEN || e_i != INTER || e_t != TN) begin
            $display("[swiglu_mt] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d %d %d", p, i, c);
            wt_g = p[2:0]; wt_u = i[2:0]; wt_d = c[2:0];
            for (i = 0; i < HIDDEN; i = i + 1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            read_pass(0, INTER,  HIDDEN);
            read_pass(1, INTER,  HIDDEN);
            read_pass(2, HIDDEN, INTER);
            for (i = 0; i < HIDDEN; i = i + 1) begin code=$fscanf(fd,"%h",t16); e_y[i]=t16; end
            for (i = 0; i < HIDDEN; i = i + 1) begin code=$fscanf(fd,"%f",tl);  e_to[i]=tl; end
            ld_tick = ~ld_tick;

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 5000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[swiglu_mt] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end
            for (i = 0; i < HIDDEN; i = i + 1) begin
                checks = checks + 1;
                ee = ab_(b2r(y_out[16*i +: 16]) - b2r(e_y[i]));
                if (ee > wa) wa = ee;
                if (ee > e_to[i]) begin
                    if (errors < 6)
                        $display("FAIL t%0d (types %0d/%0d/%0d) y[%0d]: got %h (%f) exp %h (%f) tol %f",
                                 t, wt_g, wt_u, wt_d, i, y_out[16*i +: 16],
                                 b2r(y_out[16*i +: 16]), e_y[i], b2r(e_y[i]), e_to[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[swiglu_mt] ALL %0d TESTS PASSED (%0d tokens HIDDEN=%0d INTER=%0d over the checkpoint's own MoE type combos -- the runtime w_type selects the right bus: Q4_K off w_q, Q5_K/Q6_K off w_hp, headers off w_d/w_dmin/w_scales and w_q6_sc; worst abs %e)",
                     checks, ntest, HIDDEN, INTER, wa);
        else
            $display("[swiglu_mt] %0d/%0d FAILED (worst abs %e)", errors, checks, wa);
        $finish;
    end
    initial begin #4000000000; $display("[swiglu_mt] FAIL: timeout"); $finish; end
endmodule
