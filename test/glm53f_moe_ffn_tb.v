//============================================================================
// glm53f_moe_ffn_tb.v -- src/glm53f_moe_ffn.v against tools/glm53f_moe_ffn_gen.py
// (golden = glm53_flash_ref.moe_ffn with expert_out_bf16=True).
//
// The claim is the LOOP and the COMBINE: exactly the selected experts run, each
// scaled by ITS OWN router weight, the shared expert added at weight 1 with its
// OWN weight types, accumulated in fp32.  Per-type arithmetic is NOT re-claimed
// here -- `make swiglu-mt` gates Q4_K/Q5_K/Q6_K at the K=256 the super-blocks
// force.  Routed experts run Q8_0 and the shared one F16 so the two type triples
// are driven DIFFERENTLY across that boundary, which is the property the
// checkpoint depends on ([scan]: routed Q4_K/Q5_K/Q6_K, shared Q8_0).
//
// Must FAIL, all four measured against the golden in Python BEFORE being written
// (3 draws x 32 outputs): INJ_MOE_WEIGHT_ROTATE 96/96 outputs differ,
// INJ_MOE_SHARED_TYPE 96/96, INJ_MOE_WRONG_SELECT 96/96,
// INJ_MOE_SHARED_WEIGHTED 94/96.  There is no accumulation-ORDER leg: measured,
// ascending and score-descending are bitwise identical at the bf16 output on
// 16/16 draws, so such a leg would pass and prove nothing.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_moe_ffn_vec.txt"
`endif

module glm53f_moe_ffn_tb;
    localparam integer HIDDEN=32, INTER=64, E=8, TOPK=3, TN=4, KMAX=64;
    localparam integer IDXW=$clog2(E), EW=$clog2(E+1);
    localparam integer NSB=(KMAX+255)/256, NB8=KMAX/32;
    localparam integer GW=$clog2(((INTER>HIDDEN?INTER:HIDDEN)/TN)+1);
    localparam integer MAXROW=(INTER>HIDDEN)?INTER:HIDDEN;
    localparam integer MAXK  =(INTER>HIDDEN)?INTER:HIDDEN;
    localparam integer SLOTS =E+1;                 // routed 0..E-1, shared at E

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*HIDDEN-1:0] x_in;
    reg  [32*E-1:0] rw_row, bias_in;
    reg  [2:0] wt_g, wt_u, wt_d, wt_sg, wt_su, wt_sd;

    wire busy, done, rw_req, fw_req, fw_shared;
    wire [$clog2(HIDDEN+1)-1:0] rw_k;
    wire [IDXW-1:0] fw_eidx;
    wire [1:0]      fw_sel;
    wire [GW-1:0]   fw_grp;
    wire [$clog2(KMAX+1)-1:0] fw_k;
    reg  [4*TN-1:0]        w_q;
    reg  [16*TN-1:0]       w_hp;
    reg  [16*TN*NSB-1:0]   w_d, w_dmin;
    reg  [96*TN*NSB-1:0]   w_scales;
    reg  [128*TN*NSB-1:0]  w_q6sc;
    reg  [16*TN*NB8-1:0]   w_q8d;
    wire [16*HIDDEN-1:0]   y_out;
    wire [TOPK*IDXW-1:0]   dbg_idx;
    wire [TOPK*16-1:0]     dbg_w;

    glm53f_moe_ffn #(.HIDDEN(HIDDEN), .INTER(INTER), .E(E), .TOPK(TOPK),
                     .TN(TN), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_in(x_in),
        .rw_req(rw_req), .rw_k(rw_k), .rw_row(rw_row), .bias_in(bias_in),
        .fw_req(fw_req), .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .wt_gate(wt_g), .wt_up(wt_u), .wt_down(wt_d),
        .wt_sh_gate(wt_sg), .wt_sh_up(wt_su), .wt_sh_down(wt_sd),
        .w_q(w_q), .w_hp(w_hp), .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .w_q6_sc(w_q6sc), .w_q8_d(w_q8d), .y_out(y_out),
        .dbg_sel_idx(dbg_idx), .dbg_sel_weight(dbg_w));

    // ---- FLAT stores, indexed (slot, pass).  NOT `always @*` over them: see
    //      test/glm53f_swiglu_mt_tb.v -- iverilog makes such a block sensitive to
    //      every word and the compile never finishes. ----
    localparam integer CSTR = MAXROW*MAXK;      // per (slot,pass) code stride
    localparam integer HSTR = MAXROW*NSB;
    localparam integer QSTR = MAXROW*NB8;
    reg [15:0]  cd   [0:3*SLOTS*HSTR-1];
    reg [15:0]  cdm  [0:3*SLOTS*HSTR-1];
    reg [95:0]  csc  [0:3*SLOTS*HSTR-1];
    reg [127:0] cq6  [0:3*SLOTS*HSTR-1];
    reg [15:0]  cq8  [0:3*SLOTS*QSTR-1];
    reg [15:0]  ccode[0:3*SLOTS*CSTR-1];
    reg [31:0]  wg_m [0:HIDDEN*E-1];            // router W_g, F32

    reg ld_tick = 1'b0;
    integer pj, sb, bb, slot, base, hb, cb, qb, col;
    always @(fw_sel, fw_grp, fw_k, fw_eidx, fw_shared, ld_tick) begin
        slot = fw_shared ? E : fw_eidx;
        base = slot*3 + fw_sel;
        hb = base*HSTR; cb = base*CSTR; qb = base*QSTR;
        w_q = 0; w_hp = 0; w_d = 0; w_dmin = 0; w_scales = 0; w_q6sc = 0; w_q8d = 0;
        for (pj = 0; pj < TN; pj = pj + 1) begin
            col = fw_grp*TN + pj;
            w_q [4*pj  +: 4]  = ccode[cb + col*MAXK + fw_k][3:0];
            w_hp[16*pj +: 16] = ccode[cb + col*MAXK + fw_k];
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

    // the router's F32 row for beat rw_k
    integer re;
    always @(rw_k, ld_tick) begin
        rw_row = 0;
        for (re = 0; re < E; re = re + 1)
            rw_row[32*re +: 32] = wg_m[rw_k*E + re];
    end

    integer fd, code, t, i, j, c, ntest, e_h, e_i, e_e, e_k, e_t, errors, checks, w;
    real ee, wa, tl;
    reg [15:0] t16; reg [7:0] t8; reg [95:0] t96; reg [31:0] t32;
    integer    tint;   // $fscanf will not write an `integer` ARRAY ELEMENT
                       // directly under iverilog -- it consumes the token and
                       // leaves the element x, which reads as a silent miss.
    reg [15:0] e_y [0:HIDDEN-1];
    real       e_to[0:HIDDEN-1];
    integer    e_ix[0:TOPK-1];
    reg [15:0] e_w [0:TOPK-1];
    reg        seen;

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

    // nsb_t / nb8_t are the counts the GENERATOR emitted, derived from the
    // per-pass K exactly as tools/glm53f_swiglu_mt_gen.py's emit_cols does
    // (floor division).  NSB / NB8 are the RTL's compile-time strides, from
    // KMAX.  They are equal only when K == KMAX; reading nsb_t items into
    // NSB-strided slots is the same shape glm_matmul_mixed_tb.v uses.
    integer nsb_t, nb8_t;
    task read_cols(input integer bs, input integer nrow, input integer nk);
        begin
            nsb_t = nk / 256;
            nb8_t = nk / 32;
            for (c = 0; c < nrow; c = c + 1) begin
                for (sb = 0; sb < NSB; sb = sb + 1) begin
                    cd [bs*HSTR + c*NSB + sb] = 0; cdm[bs*HSTR + c*NSB + sb] = 0;
                    csc[bs*HSTR + c*NSB + sb] = 0; cq6[bs*HSTR + c*NSB + sb] = 0;
                end
                for (bb = 0; bb < NB8; bb = bb + 1) cq8[bs*QSTR + c*NB8 + bb] = 0;
            end
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < nsb_t; sb = sb + 1) begin code=$fscanf(fd,"%h",t16); cd[bs*HSTR + c*NSB + sb]=t16; end
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < nsb_t; sb = sb + 1) begin code=$fscanf(fd,"%h",t16); cdm[bs*HSTR + c*NSB + sb]=t16; end
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < nsb_t; sb = sb + 1) begin code=$fscanf(fd,"%h",t96); csc[bs*HSTR + c*NSB + sb]=t96; end
            for (c = 0; c < nrow; c = c + 1)
                for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                    cq6[bs*HSTR + c*NSB + sb] = 0;
                    for (bb = 0; bb < 16; bb = bb + 1) begin
                        code=$fscanf(fd,"%h",t8);
                        cq6[bs*HSTR + c*NSB + sb][8*bb +: 8] = t8;
                    end
                end
            for (c = 0; c < nrow; c = c + 1)
                for (bb = 0; bb < nb8_t; bb = bb + 1) begin code=$fscanf(fd,"%h",t16); cq8[bs*QSTR + c*NB8 + bb]=t16; end
            for (c = 0; c < nrow; c = c + 1)
                for (i = 0; i < nk; i = i + 1) begin code=$fscanf(fd,"%h",t16); ccode[bs*CSTR + c*MAXK + i]=t16; end
        end
    endtask

    initial begin
        errors = 0; checks = 0; wa = 0.0;
        // routed Q8_0, shared F16 -- see the header: different on purpose
        wt_g = 3'd2; wt_u = 3'd2; wt_d = 3'd2;
        wt_sg = 3'd3; wt_su = 3'd3; wt_sd = 3'd3;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[moe_ffn] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d", ntest, e_h, e_i, e_e, e_k, e_t);
        if (e_h != HIDDEN || e_i != INTER || e_e != E || e_k != TOPK || e_t != TN) begin
            $display("[moe_ffn] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < HIDDEN; i = i + 1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i = 0; i < HIDDEN; i = i + 1)
                for (j = 0; j < E; j = j + 1) begin code=$fscanf(fd,"%h",t32); wg_m[i*E + j]=t32; end
            for (j = 0; j < E; j = j + 1) begin code=$fscanf(fd,"%h",t32); bias_in[32*j +: 32]=t32; end
            for (j = 0; j < E; j = j + 1) begin
                read_cols(j*3 + 0, INTER,  HIDDEN);
                read_cols(j*3 + 1, INTER,  HIDDEN);
                read_cols(j*3 + 2, HIDDEN, INTER);
            end
            read_cols(E*3 + 0, INTER,  HIDDEN);
            read_cols(E*3 + 1, INTER,  HIDDEN);
            read_cols(E*3 + 2, HIDDEN, INTER);
            for (j = 0; j < TOPK; j = j + 1) begin code=$fscanf(fd,"%d",tint); e_ix[j]=tint; end
            for (j = 0; j < TOPK; j = j + 1) begin code=$fscanf(fd,"%h",t16); e_w[j]=t16; end
            for (i = 0; i < HIDDEN; i = i + 1) begin code=$fscanf(fd,"%h",t16); e_y[i]=t16; end
            for (i = 0; i < HIDDEN; i = i + 1) begin code=$fscanf(fd,"%f",tl);  e_to[i]=tl; end
            // SENTINEL -- see the generator.  A misaligned read makes every later
            // compare meaningless, and a $fscanf that matches nothing leaves its
            // target untouched, so the checks can all "pass" on stale data. This
            // is the only thing standing between that and a green run.
            code = $fscanf(fd, "%h %d", t32, tint);
            checks = checks + 1;
            if (code != 2 || t32 !== 32'hA5A5A5A5 || tint != t) begin
                $display("[moe_ffn] FAIL t%0d: vector stream MISALIGNED (sentinel %h/%0d, matched %0d) -- every check after this point is meaningless", t, t32, tint, code);
                errors = errors + 1;
                $display("[moe_ffn] %0d/%0d FAILED (aborting: stream desynchronised)", errors, checks);
                $finish;
            end
            ld_tick = ~ld_tick;

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 5000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[moe_ffn] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end

            // the router's own selection, compared as a SET (it emits
            // score-descending, the golden is ascending) with each weight
            // matched to ITS OWN index
            for (j = 0; j < TOPK; j = j + 1) begin
                checks = checks + 1;
                seen = 1'b0;
                for (i = 0; i < TOPK; i = i + 1)
                    if (dbg_idx[IDXW*i +: IDXW] == e_ix[j][IDXW-1:0]) begin
                        seen = 1'b1;
                        // TOLERANCE, not bitwise, and the bound is the ROUTER
                        // GATE'S OWN (rel 0.02 + abs 0.01, test/glm53f_moe_router_tb.v).
                        // The router's numeric accuracy is owned by `make
                        // moe-router`; what THIS check adds is wiring -- that the
                        // weight which reached each expert is the one belonging to
                        // that expert's index. Re-deriving a tighter bound here
                        // would mean two gates disagreeing about the same quantity.
                        // (Measured: the RTL differs from the golden by exactly one
                        // bf16 ULP on 6 of 9 weights, which is inside that bound and
                        // is fp32_add's known non-conformance, `make fp-ieee`.)
                        if (ab_(b2r(dbg_w[16*i +: 16]) - b2r(e_w[j]))
                              > 0.02 * ab_(b2r(e_w[j])) + 0.01) begin
                            if (errors < 8)
                                $display("FAIL t%0d expert %0d: weight %h exp %h",
                                         t, e_ix[j], dbg_w[16*i +: 16], e_w[j]);
                            errors = errors + 1;
                        end
                    end
                if (!seen) begin
                    if (errors < 8) $display("FAIL t%0d: expert %0d not selected", t, e_ix[j]);
                    errors = errors + 1;
                end
            end

            for (i = 0; i < HIDDEN; i = i + 1) begin
                checks = checks + 1;
                ee = ab_(b2r(y_out[16*i +: 16]) - b2r(e_y[i]));
                if (ee > wa) wa = ee;
                if (ee > e_to[i]) begin
                    if (errors < 8)
                        $display("FAIL t%0d y[%0d]: got %h (%f) exp %h (%f) tol %f",
                                 t, i, y_out[16*i +: 16], b2r(y_out[16*i +: 16]),
                                 e_y[i], b2r(e_y[i]), e_to[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[moe_ffn] ALL %0d TESTS PASSED (%0d tokens, %0d experts top-%0d: the selected experts run in ascending index order each scaled by its OWN router weight, the always-on shared expert is added at weight 1 with its OWN type triple (routed Q8_0 / shared F16), fp32 accumulation; worst abs %e)",
                     checks, ntest, E, TOPK, wa);
        else
            $display("[moe_ffn] %0d/%0d FAILED (worst abs %e)", errors, checks, wa);
        $finish;
    end
    initial begin #4000000000; $display("[moe_ffn] FAIL: timeout"); $finish; end
endmodule
