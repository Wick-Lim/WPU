//============================================================================
// glm53f_kda_layer_tb.v -- gate for ONE Kimi Delta Attention layer
// (src/glm53f_kda_layer.v, vectors from tools/glm53f_kda_layer_gen.py).
//
// The golden composes glm53_flash_ref end to end, so passing means the layer
// reproduces the SPECIFICATION rather than that four already-gated units were
// wired together. What is new here is the layer's own job: the projection
// sequence, the ONE conv over the CONCATENATION of q,k,v, the two pieces of state
// (the [H,DK,DV] recurrence AND the [3*H*DK,K-1] conv history), and the fp32/bf16
// boundaries between the units. All four injections target that.
//
// THE STUB RESPONDER answers the nine projections behaviourally, in fp32 and in
// the same sequential order the generator uses, from the weights in the vector
// file. That isolates the layer from the GEMV engine, which q4k/mixedtype already
// gate; driving glm_matmul_q4k with Q8_0 lanes (w_type=2, code on w_hp, fp16 d on
// w_q8_d -- all already engine inputs) is the follow-on.
//
// TOLERANCE, composed from the units' OWN published bounds rather than read off
// this DUT.  Relative: the gate path runs on glm_act's bf16 sigmoid and 4.3e puts
// a KDA layer's output error at 1-3 %, so 6 % is ~2x that.  Absolute: the binding
// term is kda_onorm_step, gated at abs 0.004, and o_proj sums H*DV = 8 of those,
// so 0.03 is the composed figure (~3x the measured worst of 9.9e-3).  An earlier
// 0.01 passed with 0.6 % margin, which is luck, not a bound.
//
// Must FAIL: -DINJ_KDAL_NO_STATE, -DINJ_KDAL_CONV_NOHIST, -DINJ_KDAL_QK_SWAP,
//            -DINJ_KDAL_GATE_ORDER
//============================================================================
`timescale 1ns/1ps
`include "glm_fp.vh"
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_kda_layer_vec.txt"
`endif
`ifndef TB_REL
    `define TB_REL 0.06
`endif
`ifndef TB_ABS
    `define TB_ABS 0.03
`endif

module glm53f_kda_layer_tb;
    localparam integer MD=16, H=2, DK=4, DV=4, RANK=4, CK=4;
    localparam integer HDK=H*DK, HDV=H*DV, C=3*HDK;
    localparam integer PI=(MD>HDV)?MD:HDV, PO=(HDK>MD)?HDK:MD;
    localparam integer NS=H*DK*DV, NH=C*(CK-1);

    localparam integer OFF_Q=0,               L_Q=HDK*MD;
    localparam integer OFF_K=OFF_Q+L_Q,       L_K=HDK*MD;
    localparam integer OFF_V=OFF_K+L_K,       L_V=HDV*MD;
    localparam integer OFF_B=OFF_V+L_V,       L_B=H*MD;
    localparam integer OFF_FA=OFF_B+L_B,      L_FA=RANK*MD;
    localparam integer OFF_FB=OFF_FA+L_FA,    L_FB=HDK*RANK;
    localparam integer OFF_GA=OFF_FB+L_FB,    L_GA=RANK*MD;
    localparam integer OFF_GB=OFF_GA+L_GA,    L_GB=HDV*RANK;
    localparam integer OFF_O=OFF_GB+L_GB,     L_O=MD*HDV;
    localparam integer NW=OFF_O+L_O;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*MD-1:0]   x_in;
    reg  [32*H-1:0]    decay_in;
    reg  [32*HDK-1:0]  dtb_in;
    reg  [32*C*CK-1:0] cw_in;
    reg  [16*DV-1:0]   onw_in;
    reg  [32*NS-1:0]   s_in;
    reg  [32*NH-1:0]   h_in;

    wire busy, done, proj_req;
    wire [3:0] proj_sel;
    wire [32*PI-1:0] proj_in;
    reg  proj_done;
    reg  [32*PO-1:0] proj_out;
    wire [32*NS-1:0] s_out;
    wire [32*NH-1:0] h_out;
    wire [32*MD-1:0] y_out;

    glm53f_kda_layer #(.MODEL_DIM(MD),.H(H),.DK(DK),.DV(DV),.RANK(RANK),.CONV_K(CK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_in(x_in),
        .proj_req(proj_req), .proj_sel(proj_sel), .proj_in(proj_in),
        .proj_done(proj_done), .proj_out(proj_out),
        .decay_in(decay_in), .dt_bias_in(dtb_in), .conv_w_in(cw_in), .onorm_w_in(onw_in),
        .s_in(s_in), .s_out(s_out), .hist_in(h_in), .hist_out(h_out), .y_out(y_out));

    // ---- stub projection responder: sequential fp32 GEMV from the vector file ----
    reg [31:0] wmem [0:NW-1];
    reg [2:0]  rp;
    reg [3:0]  rsel;
    reg [32*PI-1:0] rin;
    integer rr, cc, roff, rrows, rcols;
    reg [31:0] acc;
    always @(posedge clk) begin
        if (rst) begin rp <= 3'd0; proj_done <= 1'b0; end
        else begin
            proj_done <= 1'b0;
            if (proj_req) begin rsel <= proj_sel; rin <= proj_in; rp <= 3'd1; end
            else if (rp != 3'd0) begin
                if (rp == 3'd2) begin
                    case (rsel)
                        4'd0: begin roff=OFF_Q;  rrows=HDK;  rcols=MD;   end
                        4'd1: begin roff=OFF_K;  rrows=HDK;  rcols=MD;   end
                        4'd2: begin roff=OFF_V;  rrows=HDV;  rcols=MD;   end
                        4'd3: begin roff=OFF_B;  rrows=H;    rcols=MD;   end
                        4'd4: begin roff=OFF_FA; rrows=RANK; rcols=MD;   end
                        4'd5: begin roff=OFF_FB; rrows=HDK;  rcols=RANK; end
                        4'd6: begin roff=OFF_GA; rrows=RANK; rcols=MD;   end
                        4'd7: begin roff=OFF_GB; rrows=HDV;  rcols=RANK; end
                        default: begin roff=OFF_O; rrows=MD; rcols=HDV;  end
                    endcase
                    for (rr = 0; rr < PO; rr = rr + 1) begin
                        acc = 32'd0;
                        if (rr < rrows)
                            for (cc = 0; cc < rcols; cc = cc + 1)
                                acc = fp32_add(acc, fp32_mul(wmem[roff + rr*rcols + cc],
                                                             rin[32*cc +: 32]));
                        proj_out[32*rr +: 32] = acc;
                    end
                    proj_done <= 1'b1; rp <= 3'd0;
                end else rp <= rp + 3'd1;
            end
        end
    end

    integer fd, code, t, i, ntest, errors, checks, w;
    integer p_md,p_h,p_dk,p_dv,p_rank,p_ck;
    real e, tol, wr, wa, gr;
    reg [31:0] t32; reg [15:0] t16;
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
            gr = f2r(exp);
            e  = ab(f2r(got) - gr);
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
        if (fd == 0) begin $display("[kda_layer] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d %d", ntest,p_md,p_h,p_dk,p_dv,p_rank,p_ck);
        if (p_md!=MD||p_h!=H||p_dk!=DK||p_dv!=DV||p_rank!=RANK||p_ck!=CK) begin
            $display("[kda_layer] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i=0;i<MD;i=i+1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i=0;i<NW;i=i+1) begin code=$fscanf(fd,"%h",t32); wmem[i]=t32; end
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
            while (done !== 1'b1 && w < 200000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[kda_layer] FAIL t%0d: done never asserted (%0d cycles)", t, w);
                errors = errors + 1;
            end
            for (i=0;i<MD;i=i+1) chk1(y_out[32*i +: 32], e_y[i], t, "y",     i);
            for (i=0;i<NS;i=i+1) chk1(s_out[32*i +: 32], e_s[i], t, "state", i);
            for (i=0;i<NH;i=i+1) chk1(h_out[32*i +: 32], e_h[i], t, "hist",  i);
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[kda_layer] ALL %0d TESTS PASSED (%0d decode steps, H=%0d DK=%0d DV=%0d: 9 projections sequenced, ONE conv over concat(q,k,v), forget gate, delta-rule recurrence and gated o_norm -- y, the [H,DK,DV] state AND the conv history all within rel %0.3f + abs %0.3f; worst rel %0.5f (|golden|>=1e-3), worst abs %e)",
                     checks, ntest, H, DK, DV, `TB_REL, `TB_ABS, wr, wa);
        else
            $display("[kda_layer] %0d/%0d FAILED (worst rel %0.5f, abs %e)", errors, checks, wr, wa);
        $finish;
    end
    initial begin #200000000; $display("[kda_layer] FAIL: timeout"); $finish; end
endmodule
