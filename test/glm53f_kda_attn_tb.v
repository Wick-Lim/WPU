//============================================================================
// glm53f_kda_attn_tb.v -- gate for the COMPLETE KDA sublayer
// (src/glm53f_kda_attn.v = glm53f_kda_layer + glm53f_kda_gemv).
//
// `make kda-layer` checks the datapath with its nine projections HANDED to it.
// This checks the same datapath when they are STREAMED off real Q8_0 weights
// through glm_matmul_q4k -- the difference between a unit and a sublayer.
//
// The golden runs on the DEQUANTISED weights, so the Q8_0 round trip is part of
// the INPUT, not of the error: any disagreement here is the streaming path.
// Bounds are the same composed ones `make kda-layer` uses (rel 6 % from 4.3e's
// 1-3 % per layer, abs 0.03 from kda_onorm_step's 0.004 times o_proj's fan-in),
// which is the point -- streaming the weights must not need a looser bound.
//
// THE WEIGHT RESPONDER answers combinationally on (w_sel, w_grp, w_k), exactly
// as the system would: TN Q8_0 codes for the active output columns at this beat,
// plus the fp16 block scale per column. Every projection here has K <= 16, so
// there is ONE Q8_0 block per row and `blk = pj*NB8 + (k>>5)` indexes block 0.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_kda_attn_vec.txt"
`endif
`ifndef TB_REL
    `define TB_REL 0.06
`endif
`ifndef TB_ABS
    `define TB_ABS 0.03
`endif

module glm53f_kda_attn_tb;
    localparam integer MD=16, H=2, DK=4, DV=4, RANK=4, CK=4, TN=2, KMAX=32;
    localparam integer HDK=H*DK, HDV=H*DV, C=3*HDK;
    localparam integer PO=(HDK>MD)?HDK:MD, NB8=(KMAX+31)/32;
    localparam integer NS=H*DK*DV, NH=C*(CK-1);

    // code offsets (rows*cols) and scale offsets (rows), in ORDER q k v b fa fb ga gb o
    localparam integer CO0=0,                    R0=HDK, K0=MD;
    localparam integer CO1=CO0+R0*K0,            R1=HDK, K1=MD;
    localparam integer CO2=CO1+R1*K1,            R2=HDV, K2=MD;
    localparam integer CO3=CO2+R2*K2,            R3=H,   K3=MD;
    localparam integer CO4=CO3+R3*K3,            R4=RANK,K4=MD;
    localparam integer CO5=CO4+R4*K4,            R5=HDK, K5=RANK;
    localparam integer CO6=CO5+R5*K5,            R6=RANK,K6=MD;
    localparam integer CO7=CO6+R6*K6,            R7=HDV, K7=RANK;
    localparam integer CO8=CO7+R7*K7,            R8=MD,  K8=HDV;
    localparam integer NCODE=CO8+R8*K8;
    localparam integer SO0=0, SO1=SO0+R0, SO2=SO1+R1, SO3=SO2+R2, SO4=SO3+R3,
                       SO5=SO4+R4, SO6=SO5+R5, SO7=SO6+R6, SO8=SO7+R7,
                       NSCALE=SO8+R8;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*MD-1:0]   x_in;
    reg  [32*H-1:0]    decay_in;
    reg  [32*HDK-1:0]  dtb_in;
    reg  [32*C*CK-1:0] cw_in;
    reg  [16*DV-1:0]   onw_in;
    reg  [32*NS-1:0]   s_in;
    reg  [32*NH-1:0]   h_in;

    wire busy, done, w_req;
    wire [3:0] w_sel;
    wire [$clog2(PO/TN+1)-1:0] w_grp;
    wire [$clog2(KMAX+1)-1:0]  w_k;
    reg  [16*TN-1:0] w_hp;
    reg  [16*TN*NB8-1:0] w_q8d;
    wire [32*NS-1:0] s_out;
    wire [32*NH-1:0] h_out;
    wire [32*MD-1:0] y_out;

    glm53f_kda_attn #(.MODEL_DIM(MD),.H(H),.DK(DK),.DV(DV),.RANK(RANK),
                      .CONV_K(CK),.TN(TN),.KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_in(x_in),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_hp(w_hp), .w_q8_d(w_q8d),
        .decay_in(decay_in), .dt_bias_in(dtb_in), .conv_w_in(cw_in), .onorm_w_in(onw_in),
        .s_in(s_in), .s_out(s_out), .hist_in(h_in), .hist_out(h_out), .y_out(y_out));

    reg [7:0]  cmem [0:NCODE-1];
    reg [15:0] smem [0:NSCALE-1];

    integer rco, rso, rk, jj;
    always @* begin
        case (w_sel)
            4'd0: begin rco=CO0; rso=SO0; rk=K0; end
            4'd1: begin rco=CO1; rso=SO1; rk=K1; end
            4'd2: begin rco=CO2; rso=SO2; rk=K2; end
            4'd3: begin rco=CO3; rso=SO3; rk=K3; end
            4'd4: begin rco=CO4; rso=SO4; rk=K4; end
            4'd5: begin rco=CO5; rso=SO5; rk=K5; end
            4'd6: begin rco=CO6; rso=SO6; rk=K6; end
            4'd7: begin rco=CO7; rso=SO7; rk=K7; end
            default: begin rco=CO8; rso=SO8; rk=K8; end
        endcase
        w_hp  = {(16*TN){1'b0}};
        w_q8d = {(16*TN*NB8){1'b0}};
        for (jj = 0; jj < TN; jj = jj + 1) begin
            w_hp[16*jj +: 16]      = {8'd0, cmem[rco + (w_grp*TN + jj)*rk + w_k]};
            w_q8d[16*(jj*NB8) +: 16] = smem[rso + w_grp*TN + jj];
        end
    end

    integer fd, code, t, i, ntest, errors, checks, w;
    integer p_md,p_h,p_dk,p_dv,p_rank,p_ck;
    real e, tol, wr, wa, gr;
    reg [31:0] t32; reg [15:0] t16; reg [7:0] t8;
    reg [31:0] e_y [0:MD-1];
    reg [31:0] e_s [0:NS-1];
    reg [31:0] e_h [0:NH-1];

    function real f2r(input [31:0] f);
        integer ex, i2; real m;
        begin
            ex = f[30:23];
            if (ex == 0) f2r = 0.0;
            else begin
                m = 1.0;
                for (i2 = 0; i2 < 23; i2 = i2 + 1) if (f[22-i2]) m = m + (2.0 ** (-(i2+1)));
                f2r = m * (2.0 ** (ex - 127));
                if (f[31]) f2r = -f2r;
            end
        end
    endfunction
    function real ab(input real x); begin ab = (x<0.0)?-x:x; end endfunction

    task chk1(input [31:0] got, input [31:0] exp, input integer tno,
              input [63:0] nm, input integer idx);
        begin
            checks = checks + 1;
            gr = f2r(exp); e = ab(f2r(got) - gr);
            tol = `TB_REL * ab(gr) + `TB_ABS;
            if (e > wa) wa = e;
            if (ab(gr) >= 1.0e-3 && e/ab(gr) > wr) wr = e/ab(gr);
            if (e > tol) begin
                $display("FAIL t%0d %0s[%0d]: got %h (%f) exp %h (%f) tol %f",
                         tno, nm, idx, got, f2r(got), exp, gr, tol);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0; wr = 0.0; wa = 0.0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[kda_attn] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d %d", ntest,p_md,p_h,p_dk,p_dv,p_rank,p_ck);
        if (p_md!=MD||p_h!=H||p_dk!=DK||p_dv!=DV||p_rank!=RANK||p_ck!=CK) begin
            $display("[kda_attn] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i=0;i<MD;i=i+1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            // codes and scales, per projection, in ORDER
            for (i=0;i<R0*K0;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO0+i]=t8; end
            for (i=0;i<R0;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO0+i]=t16; end
            for (i=0;i<R1*K1;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO1+i]=t8; end
            for (i=0;i<R1;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO1+i]=t16; end
            for (i=0;i<R2*K2;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO2+i]=t8; end
            for (i=0;i<R2;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO2+i]=t16; end
            for (i=0;i<R3*K3;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO3+i]=t8; end
            for (i=0;i<R3;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO3+i]=t16; end
            for (i=0;i<R4*K4;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO4+i]=t8; end
            for (i=0;i<R4;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO4+i]=t16; end
            for (i=0;i<R5*K5;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO5+i]=t8; end
            for (i=0;i<R5;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO5+i]=t16; end
            for (i=0;i<R6*K6;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO6+i]=t8; end
            for (i=0;i<R6;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO6+i]=t16; end
            for (i=0;i<R7*K7;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO7+i]=t8; end
            for (i=0;i<R7;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO7+i]=t16; end
            for (i=0;i<R8*K8;i=i+1) begin code=$fscanf(fd,"%h",t8); cmem[CO8+i]=t8; end
            for (i=0;i<R8;i=i+1)    begin code=$fscanf(fd,"%h",t16); smem[SO8+i]=t16; end
            for (i=0;i<H;i=i+1)     begin code=$fscanf(fd,"%h",t32); decay_in[32*i +: 32]=t32; end
            for (i=0;i<HDK;i=i+1)   begin code=$fscanf(fd,"%h",t32); dtb_in[32*i +: 32]=t32; end
            for (i=0;i<C*CK;i=i+1)  begin code=$fscanf(fd,"%h",t32); cw_in[32*i +: 32]=t32; end
            for (i=0;i<DV;i=i+1)    begin code=$fscanf(fd,"%h",t16); onw_in[16*i +: 16]=t16; end
            for (i=0;i<NS;i=i+1)    begin code=$fscanf(fd,"%h",t32); s_in[32*i +: 32]=t32; end
            for (i=0;i<NH;i=i+1)    begin code=$fscanf(fd,"%h",t32); h_in[32*i +: 32]=t32; end
            for (i=0;i<MD;i=i+1)    begin code=$fscanf(fd,"%h",t32); e_y[i]=t32; end
            for (i=0;i<NS;i=i+1)    begin code=$fscanf(fd,"%h",t32); e_s[i]=t32; end
            for (i=0;i<NH;i=i+1)    begin code=$fscanf(fd,"%h",t32); e_h[i]=t32; end

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 500000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[kda_attn] FAIL t%0d: done never asserted (%0d cycles)", t, w);
                errors = errors + 1;
            end
            for (i=0;i<MD;i=i+1) chk1(y_out[32*i +: 32], e_y[i], t, "y",     i);
            for (i=0;i<NS;i=i+1) chk1(s_out[32*i +: 32], e_s[i], t, "state", i);
            for (i=0;i<NH;i=i+1) chk1(h_out[32*i +: 32], e_h[i], t, "hist",  i);
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[kda_attn] ALL %0d TESTS PASSED (%0d decode steps: the WHOLE KDA sublayer -- nine Q8_0 projections streamed off glm_matmul_q4k, conv, forget gate, delta-rule recurrence, gated o_norm -- y, state and conv history within rel %0.3f + abs %0.3f; worst rel %0.5f (|golden|>=1e-3), worst abs %e)",
                     checks, ntest, `TB_REL, `TB_ABS, wr, wa);
        else
            $display("[kda_attn] %0d/%0d FAILED (worst rel %0.5f, abs %e)", errors, checks, wr, wa);
        $finish;
    end
    initial begin #500000000; $display("[kda_attn] FAIL: timeout"); $finish; end
endmodule
