//============================================================================
// glm53f_swiglu_q8_tb.v -- GLM-5.3-Flash's dense FFN: clamped SwiGLU over Q8_0
// (src/glm53f_swiglu_q8.v, vectors from tools/glm53f_swiglu_q8_gen.py).
//
// swiglu_expert_q4k cannot be used here at all: its w_q port is FOUR BITS PER
// LANE and the census says this FFN is Q8_0 (blk.N.ffn_{gate,up,down}). This is
// the sibling that can carry it.
//
// Per-element tolerance comes from the vector file, shaped as `make swiglu` does:
// glm_act's polynomial silu is an approximation, so the DOWN reduction is a
// functional check and the bit-exact claim belongs to glm_matmul_q4k's own gate.
//
// Must FAIL: -DINJ_SWQ8_NOCLAMP (the clamp removed -- proves the golden really
// encodes it) and -DINJ_SWQ8_Q4K_TYPE (w_type left at Q4_K, which is also what an
// UNDRIVEN w_type reads as).
// NOT a must-fail: -DINJ_SWQ8_SYMCLAMP. Measured, the symmetric-gate reading moves
// the result 0.25 against a 2.9 tolerance -- inside it, because that tolerance is
// forced by glm_act's own silu error. The asymmetry is gated on a tighter slice by
// `make swiglu`. A gate that cannot fail is worse than none.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_swiglu_q8_vec.txt"
`endif

module glm53f_swiglu_q8_tb;
    localparam integer HIDDEN=16, INTER=32, TN=2, KMAX=32;
    localparam integer NB8=(KMAX+31)/32;
    localparam integer GW=$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1);
    localparam integer NG=INTER*HIDDEN, NU=INTER*HIDDEN, ND=HIDDEN*INTER;
    localparam integer OG=0, OU=OG+NG, OD=OU+NU, NCODE=OD+ND;
    localparam integer SG=0, SU=SG+INTER, SD=SU+INTER, NSCALE=SD+HIDDEN;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*HIDDEN-1:0] x_in;
    wire busy, done, w_req;
    wire [1:0] w_sel;
    wire [GW-1:0] w_grp;
    wire [$clog2(KMAX+1)-1:0] w_k;
    reg  [16*TN-1:0] w_hp;
    reg  [16*TN*NB8-1:0] w_q8d;
    wire [16*HIDDEN-1:0] y_out;

    glm53f_swiglu_q8 #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_in(x_in), .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_hp(w_hp), .w_q8_d(w_q8d), .y_out(y_out));

    reg [7:0]  cmem [0:NCODE-1];
    reg [15:0] smem [0:NSCALE-1];
    integer rco, rso, rk, jj;
    always @* begin
        case (w_sel)
            2'd0: begin rco=OG; rso=SG; rk=HIDDEN; end
            2'd1: begin rco=OU; rso=SU; rk=HIDDEN; end
            default: begin rco=OD; rso=SD; rk=INTER; end
        endcase
        w_hp  = {(16*TN){1'b0}};
        w_q8d = {(16*TN*NB8){1'b0}};
        for (jj = 0; jj < TN; jj = jj + 1) begin
            w_hp[16*jj +: 16]       = {8'd0, cmem[rco + (w_grp*TN + jj)*rk + w_k]};
            w_q8d[16*(jj*NB8) +: 16] = smem[rso + w_grp*TN + jj];
        end
    end

    integer fd, code, t, i, ntest, e_h, e_i, errors, checks, w;
    real e, wa, gr, tl;
    reg [15:0] t16; reg [7:0] t8;
    reg [15:0] e_y [0:HIDDEN-1];
    real       e_t [0:HIDDEN-1];

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
        if (fd == 0) begin $display("[swiglu_q8] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, e_h, e_i);
        if (e_h != HIDDEN || e_i != INTER) begin
            $display("[swiglu_q8] FAIL: vector HIDDEN/INTER %0d/%0d != TB %0d/%0d",
                     e_h, e_i, HIDDEN, INTER); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i=0;i<HIDDEN;i=i+1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i=0;i<NG;i=i+1)     begin code=$fscanf(fd,"%h",t8);  cmem[OG+i]=t8; end
            for (i=0;i<INTER;i=i+1)  begin code=$fscanf(fd,"%h",t16); smem[SG+i]=t16; end
            for (i=0;i<NU;i=i+1)     begin code=$fscanf(fd,"%h",t8);  cmem[OU+i]=t8; end
            for (i=0;i<INTER;i=i+1)  begin code=$fscanf(fd,"%h",t16); smem[SU+i]=t16; end
            for (i=0;i<ND;i=i+1)     begin code=$fscanf(fd,"%h",t8);  cmem[OD+i]=t8; end
            for (i=0;i<HIDDEN;i=i+1) begin code=$fscanf(fd,"%h",t16); smem[SD+i]=t16; end
            for (i=0;i<HIDDEN;i=i+1) begin code=$fscanf(fd,"%h",t16); e_y[i]=t16; end
            for (i=0;i<HIDDEN;i=i+1) begin code=$fscanf(fd,"%f",tl);  e_t[i]=tl; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 500000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[swiglu_q8] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end
            for (i = 0; i < HIDDEN; i = i + 1) begin
                checks = checks + 1;
                gr = b2r(e_y[i]);
                e  = ab_(b2r(y_out[16*i +: 16]) - gr);
                if (e > wa) wa = e;
                if (e > e_t[i]) begin
                    $display("FAIL t%0d y[%0d]: got %h (%f) exp %h (%f) tol %f",
                             t, i, y_out[16*i +: 16], b2r(y_out[16*i +: 16]), e_y[i], gr, e_t[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[swiglu_q8] ALL %0d TESTS PASSED (%0d tokens HIDDEN=%0d INTER=%0d: gate and up off Q8_0 weights, asymmetric clamp at 10.0, glm_act silu, down off Q8_0 -- within the per-element tolerance the generator emits; worst abs %e)",
                     checks, ntest, HIDDEN, INTER, wa);
        else
            $display("[swiglu_q8] %0d/%0d FAILED (worst abs %e)", errors, checks, wa);
        $finish;
    end
    initial begin #200000000; $display("[swiglu_q8] FAIL: timeout"); $finish; end
endmodule
