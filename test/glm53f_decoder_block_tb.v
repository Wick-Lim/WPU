//============================================================================
// glm53f_decoder_block_tb.v -- gate for KDA attention wired INSIDE the mHC block
// (src/glm53f_decoder_block.v, vectors from tools/glm53f_decoder_block_gen.py).
//
// `make hc-block` checks the two mHC sites with stub sublayers. `make kda-attn`
// checks the KDA sublayer alone. This checks the composition: the attention
// site's collapse -> attn_norm -> KDA -> mix round trip, with the recurrence and
// the conv history advancing across it. The golden composes both goldens, so a
// routing mistake between the two machines cannot hide.
//
// TWO STUBS, one real: the KDA weight responder answers the nine Q8_0 projections
// the way the kda-attn gate does, and the FFN site keeps the 0.5*normed stub --
// GLM-5.3-Flash's dense FFN is Q8_0 and its MoE experts are a Q4_K/Q5_K/Q6_K mix,
// so swiglu_expert_q4k (4 bits per lane) cannot carry either.
//
// Must FAIL: -DINJ_DBLK_SITE_SWAP (the two sites' sublayers exchanged).
//============================================================================
`timescale 1ns/1ps
`include "glm_fp.vh"
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_decoder_block_vec.txt"
`endif
`ifndef TB_REL
    `define TB_REL 0.06
`endif
`ifndef TB_ABS
    `define TB_ABS 0.05
`endif

module glm53f_decoder_block_tb;
    localparam integer MD=16, H=4, KH=2, DK=4, DV=4, RANK=4, CK=4, TN=2, KMAX=32;
    localparam integer MIXN=(2+H)*H, HK=H*MD, NBH=HK/32;
    localparam integer HDK=KH*DK, HDV=KH*DV, C=3*HDK;
    localparam integer PO=(HDK>MD)?HDK:MD, NB8=(KMAX+31)/32;
    localparam integer NS=KH*DK*DV, NH=C*(CK-1), NST=H*MD;

    localparam integer R0=HDK,K0=MD, R1=HDK,K1=MD, R2=HDV,K2=MD, R3=KH,K3=MD,
                       R4=RANK,K4=MD, R5=HDK,K5=RANK, R6=RANK,K6=MD,
                       R7=HDV,K7=RANK, R8=MD,K8=HDV;
    localparam integer CO0=0, CO1=CO0+R0*K0, CO2=CO1+R1*K1, CO3=CO2+R2*K2,
                       CO4=CO3+R3*K3, CO5=CO4+R4*K4, CO6=CO5+R5*K5,
                       CO7=CO6+R6*K6, CO8=CO7+R7*K7, NCODE=CO8+R8*K8;
    localparam integer SO0=0, SO1=SO0+R0, SO2=SO1+R1, SO3=SO2+R2, SO4=SO3+R3,
                       SO5=SO4+R4, SO6=SO5+R5, SO7=SO6+R6, SO8=SO7+R7,
                       NSCALE=SO8+R8;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0, sload = 0;
    reg  [32*NST-1:0] s_init;
    reg  [8*MIXN*HK-1:0]   awq, fwq;
    reg  [16*MIXN*NBH-1:0] awd, fwd;
    reg  [32*MIXN-1:0]     ab, fb;
    reg  [31:0] a0,a1,a2, f0,f1,f2;
    reg  [16*MD-1:0] anw, fnw;
    reg  [32*KH-1:0] decay_in;
    reg  [32*HDK-1:0] dtb_in;
    reg  [32*C*CK-1:0] cw_in;
    reg  [16*DV-1:0] onw_in;
    reg  [32*NS-1:0] ks_in;
    reg  [32*NH-1:0] kh_in;

    wire busy, done, kw_req, ffn_start;
    wire [3:0] kw_sel;
    wire [$clog2(PO/TN+1)-1:0] kw_grp;
    wire [$clog2(KMAX+1)-1:0]  kw_k;
    reg  [16*TN-1:0]     kw_hp;
    reg  [16*TN*NB8-1:0] kw_q8d;
    wire [32*NST-1:0] s_cur;
    wire [32*NS-1:0]  ks_out;
    wire [32*NH-1:0]  kh_out;
    wire [16*MD-1:0]  ffn_vec;
    reg               ffn_done;
    reg  [16*MD-1:0]  ffn_out;

    glm53f_decoder_block #(.MODEL_DIM(MD),.H(H),.KH(KH),.DK(DK),.DV(DV),.RANK(RANK),
                           .CONV_K(CK),.TN(TN),.KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .streams_load(sload), .streams_init(s_init), .streams_cur(s_cur),
        .a_w_q(awq), .a_w_d(awd), .a_base(ab), .a_s0(a0), .a_s1(a1), .a_s2(a2),
        .f_w_q(fwq), .f_w_d(fwd), .f_base(fb), .f_s0(f0), .f_s1(f1), .f_s2(f2),
        .attn_norm_w(anw), .ffn_norm_w(fnw),
        .kda_w_req(kw_req), .kda_w_sel(kw_sel), .kda_w_grp(kw_grp), .kda_w_k(kw_k),
        .kda_w_hp(kw_hp), .kda_w_q8_d(kw_q8d),
        .decay_in(decay_in), .dt_bias_in(dtb_in), .conv_w_in(cw_in), .onorm_w_in(onw_in),
        .kda_s_in(ks_in), .kda_s_out(ks_out), .kda_hist_in(kh_in), .kda_hist_out(kh_out),
        .ffn_start(ffn_start), .ffn_vec(ffn_vec), .ffn_done(ffn_done), .ffn_out(ffn_out));

    // ---- KDA Q8_0 weight responder (same shape as the kda-attn gate) ----
    reg [7:0]  cmem [0:NCODE-1];
    reg [15:0] smem [0:NSCALE-1];
    integer rco, rso, rk, jj;
    always @* begin
        case (kw_sel)
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
        kw_hp  = {(16*TN){1'b0}};
        kw_q8d = {(16*TN*NB8){1'b0}};
        for (jj = 0; jj < TN; jj = jj + 1) begin
            kw_hp[16*jj +: 16]       = {8'd0, cmem[rco + (kw_grp*TN + jj)*rk + kw_k]};
            kw_q8d[16*(jj*NB8) +: 16] = smem[rso + kw_grp*TN + jj];
        end
    end

    // ---- FFN stub: 0.5 * the normed vector, exact in bf16 ----
    reg [1:0] fpipe;
    integer fi;
    always @(posedge clk) begin
        if (rst) begin fpipe <= 2'd0; ffn_done <= 1'b0; end
        else begin
            ffn_done <= 1'b0;
            if (ffn_start) fpipe <= 2'd1;
            else if (fpipe != 2'd0) begin
                if (fpipe == 2'd2) begin
                    for (fi = 0; fi < MD; fi = fi + 1)
                        ffn_out[16*fi +: 16] <= fp32_to_bf16(
                            fp32_mul(bf16_to_fp32(ffn_vec[16*fi +: 16]), 32'h3F000000));
                    ffn_done <= 1'b1; fpipe <= 2'd0;
                end else fpipe <= fpipe + 2'd1;
            end
        end
    end

    integer fd, code, t, i, ntest, errors, checks, w;
    integer p_md,p_h,p_kh,p_dk,p_dv,p_rank,p_ck;
    real e, tol, wr, wa, gr;
    reg [31:0] t32; reg [15:0] t16; reg [7:0] t8;
    reg [31:0] e_s [0:NST-1];
    reg [31:0] e_k [0:NS-1];
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
    function real ab_(input real x); begin ab_ = (x<0.0)?-x:x; end endfunction

    task chk1(input [31:0] got, input [31:0] exp, input integer tno,
              input [63:0] nm, input integer idx);
        begin
            checks = checks + 1;
            gr = f2r(exp); e = ab_(f2r(got) - gr);
            tol = `TB_REL * ab_(gr) + `TB_ABS;
            if (e > wa) wa = e;
            if (ab_(gr) >= 1.0e-3 && e/ab_(gr) > wr) wr = e/ab_(gr);
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
        if (fd == 0) begin $display("[dec_block] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d %d %d",
                       ntest,p_md,p_h,p_kh,p_dk,p_dv,p_rank,p_ck);
        if (p_md!=MD||p_h!=H||p_kh!=KH||p_dk!=DK||p_dv!=DV||p_rank!=RANK||p_ck!=CK) begin
            $display("[dec_block] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i=0;i<NST;i=i+1)      begin code=$fscanf(fd,"%h",t32); s_init[32*i +: 32]=t32; end
            for (i=0;i<MIXN*HK;i=i+1)  begin code=$fscanf(fd,"%h",t8);  awq[8*i +: 8]=t8; end
            for (i=0;i<MIXN*NBH;i=i+1) begin code=$fscanf(fd,"%h",t16); awd[16*i +: 16]=t16; end
            for (i=0;i<MIXN;i=i+1)     begin code=$fscanf(fd,"%h",t32); ab[32*i +: 32]=t32; end
            code=$fscanf(fd,"%h",t32); a0=t32; code=$fscanf(fd,"%h",t32); a1=t32; code=$fscanf(fd,"%h",t32); a2=t32;
            for (i=0;i<MD;i=i+1)       begin code=$fscanf(fd,"%h",t16); anw[16*i +: 16]=t16; end
            for (i=0;i<MIXN*HK;i=i+1)  begin code=$fscanf(fd,"%h",t8);  fwq[8*i +: 8]=t8; end
            for (i=0;i<MIXN*NBH;i=i+1) begin code=$fscanf(fd,"%h",t16); fwd[16*i +: 16]=t16; end
            for (i=0;i<MIXN;i=i+1)     begin code=$fscanf(fd,"%h",t32); fb[32*i +: 32]=t32; end
            code=$fscanf(fd,"%h",t32); f0=t32; code=$fscanf(fd,"%h",t32); f1=t32; code=$fscanf(fd,"%h",t32); f2=t32;
            for (i=0;i<MD;i=i+1)       begin code=$fscanf(fd,"%h",t16); fnw[16*i +: 16]=t16; end
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
            for (i=0;i<KH;i=i+1)    begin code=$fscanf(fd,"%h",t32); decay_in[32*i +: 32]=t32; end
            for (i=0;i<HDK;i=i+1)   begin code=$fscanf(fd,"%h",t32); dtb_in[32*i +: 32]=t32; end
            for (i=0;i<C*CK;i=i+1)  begin code=$fscanf(fd,"%h",t32); cw_in[32*i +: 32]=t32; end
            for (i=0;i<DV;i=i+1)    begin code=$fscanf(fd,"%h",t16); onw_in[16*i +: 16]=t16; end
            for (i=0;i<NS;i=i+1)    begin code=$fscanf(fd,"%h",t32); ks_in[32*i +: 32]=t32; end
            for (i=0;i<NH;i=i+1)    begin code=$fscanf(fd,"%h",t32); kh_in[32*i +: 32]=t32; end
            for (i=0;i<NST;i=i+1)   begin code=$fscanf(fd,"%h",t32); e_s[i]=t32; end
            for (i=0;i<NS;i=i+1)    begin code=$fscanf(fd,"%h",t32); e_k[i]=t32; end
            for (i=0;i<NH;i=i+1)    begin code=$fscanf(fd,"%h",t32); e_h[i]=t32; end

            @(negedge clk); sload = 1;
            @(negedge clk); sload = 0;
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 2000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[dec_block] FAIL t%0d: done never asserted (%0d cycles)", t, w);
                errors = errors + 1;
            end
            for (i=0;i<NST;i=i+1) chk1(s_cur[32*i +: 32],  e_s[i], t, "streams", i);
            for (i=0;i<NS;i=i+1)  chk1(ks_out[32*i +: 32], e_k[i], t, "kdastate", i);
            for (i=0;i<NH;i=i+1)  chk1(kh_out[32*i +: 32], e_h[i], t, "kdahist", i);
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[dec_block] ALL %0d TESTS PASSED (%0d decode steps: KDA attention wired INSIDE the two-site mHC block -- collapse, attn_norm, nine Q8_0 projections, the delta-rule recurrence, mix; residual streams, KDA state and conv history all within rel %0.3f + abs %0.3f; worst rel %0.5f (|golden|>=1e-3), worst abs %e)",
                     checks, ntest, `TB_REL, `TB_ABS, wr, wa);
        else
            $display("[dec_block] %0d/%0d FAILED (worst rel %0.5f, abs %e)", errors, checks, wr, wa);
        $finish;
    end
    initial begin #2000000000; $display("[dec_block] FAIL: timeout"); $finish; end
endmodule
