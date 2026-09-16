//============================================================================
// glm53f_mla_proj_tb.v -- src/glm53f_mla_proj.v against
// tools/glm53f_mla_proj_gen.py.
//
// BITWISE, for the same reason `make mla-score` is: every piece already has an
// exact Python twin (matmul_q4k_col, rmsnorm_unit at LANES=1, q80_rows), and the
// golden runs on the DEQUANTISED weights so the Q8_0 round trip is part of the
// input rather than of the error -- the contract `make kda-attn` already uses.
//
// The weight service answers (w_sel, w_head, w_grp, w_k) combinationally with TN
// columns' Q8_0 codes and their fp16 block scales, the way the system will. The
// FOLD is the pass worth watching: for head h and output column k the reduction
// runs DOWN W_uk's rows, so the stored layout is [H][KV_LORA][QK]. Getting that
// axis wrong is a transpose with the right shape.
//
// Must FAIL, each measured against the golden before being written:
//   INJ_MLAP_NO_QNORM   drop the rmsnorm between W_dq and W_uq
//   INJ_MLAP_FOLD_HEAD0 fold every head with head 0's q -- right shape, heads mixed
//   INJ_MLAP_NORM_CKV   emit a normalised latent instead of the raw one. rmsnorm
//                       belongs on the READ side, once, inside glm53f_mla_score;
//                       doing it here too applies it twice. DOWNSTREAM that is
//                       only ~2.8e-06 relative and no shape check notices -- which
//                       is exactly how it got into the reference before being
//                       caught -- so the property gets a leg of its own here,
//                       where it is loud.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_mla_proj_vec.txt"
`endif

module glm53f_mla_proj_tb;
    localparam integer MD=16, H=2, QK=8, QLORA=8, KVL=8, TN=2, KMAX=32;
    localparam integer PMO = (H*QK > QLORA) ? ((H*QK > KVL) ? H*QK : KVL)
                                            : ((QLORA > KVL) ? QLORA : KVL);
    localparam integer NB8 = (KMAX+31)/32;
    localparam integer GW  = $clog2(PMO/TN + 1);
    localparam integer KW  = $clog2(KMAX + 1);
    localparam integer HW  = $clog2(H);

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*MD-1:0] x_in;
    wire busy, done, w_req;
    wire [1:0]    w_sel;
    wire [HW-1:0] w_head;
    wire [GW-1:0] w_grp;
    wire [KW-1:0] w_k;
    reg  [16*TN-1:0]     w_hp;
    reg  [16*TN*NB8-1:0] w_q8d;
    wire [16*H*KVL-1:0]  qa;
    wire [16*KVL-1:0]    ckv;

    glm53f_mla_proj #(.MODEL_DIM(MD), .H(H), .QK(QK), .QLORA(QLORA), .KVL(KVL),
                      .TN(TN), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_in(x_in),
        .w_req(w_req), .w_sel(w_sel), .w_head(w_head), .w_grp(w_grp), .w_k(w_k),
        .w_hp(w_hp), .w_q8_d(w_q8d), .qa_out(qa), .ckv_out(ckv));

    // ---- the Q8_0 weight store, flat and served on an EXPLICIT sensitivity list
    //      (see test/glm53f_swiglu_mt_tb.v: `always @*` over a big memory makes
    //      iverilog sensitive to every word and the compile never finishes) ----
    reg [7:0]  cdq  [0:QLORA*MD-1];      reg [15:0] sdq  [0:QLORA-1];
    reg [7:0]  cuq  [0:H*QK*QLORA-1];    reg [15:0] suq  [0:H*QK-1];
    reg [7:0]  cuk  [0:H*KVL*QK-1];      reg [15:0] suk  [0:H*KVL-1];
    reg [7:0]  cdkv [0:KVL*MD-1];        reg [15:0] sdkv [0:KVL-1];

    reg ld_tick = 1'b0;
    integer pj, row;
    always @(w_sel, w_head, w_grp, w_k, ld_tick) begin
        w_hp = 0; w_q8d = 0;
        for (pj = 0; pj < TN; pj = pj + 1) begin
            row = w_grp*TN + pj;
            case (w_sel)
                2'd0: begin
                    w_hp [16*pj +: 16] = {8'd0, cdq[row*MD + w_k]};
                    w_q8d[16*(pj*NB8) +: 16] = sdq[row];
                end
                2'd1: begin
                    w_hp [16*pj +: 16] = {8'd0, cuq[row*QLORA + w_k]};
                    w_q8d[16*(pj*NB8) +: 16] = suq[row];
                end
                2'd2: begin
                    // the FOLD: head w_head, output column `row`, reduction over w_k
                    w_hp [16*pj +: 16] = {8'd0, cuk[(w_head*KVL + row)*QK + w_k]};
                    w_q8d[16*(pj*NB8) +: 16] = suk[w_head*KVL + row];
                end
                default: begin
                    w_hp [16*pj +: 16] = {8'd0, cdkv[row*MD + w_k]};
                    w_q8d[16*(pj*NB8) +: 16] = sdkv[row];
                end
            endcase
        end
    end

    integer fd, code, t, i, ntest, e_md, e_h, e_qk, e_ql, e_kv, errors, checks, w;
    reg [15:0] t16; reg [7:0] t8;
    reg [15:0] e_qa [0:H*KVL-1];
    reg [15:0] e_ck [0:KVL-1];

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[mla_proj] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d", ntest, e_md, e_h, e_qk, e_ql, e_kv);
        if (e_md != MD || e_h != H || e_qk != QK || e_ql != QLORA || e_kv != KVL) begin
            $display("[mla_proj] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            for (i = 0; i < MD; i = i + 1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i = 0; i < QLORA*MD;    i = i + 1) begin code=$fscanf(fd,"%h",t8);  cdq[i]=t8;  end
            for (i = 0; i < QLORA;       i = i + 1) begin code=$fscanf(fd,"%h",t16); sdq[i]=t16; end
            for (i = 0; i < H*QK*QLORA;  i = i + 1) begin code=$fscanf(fd,"%h",t8);  cuq[i]=t8;  end
            for (i = 0; i < H*QK;        i = i + 1) begin code=$fscanf(fd,"%h",t16); suq[i]=t16; end
            for (i = 0; i < H*KVL*QK;    i = i + 1) begin code=$fscanf(fd,"%h",t8);  cuk[i]=t8;  end
            for (i = 0; i < H*KVL;       i = i + 1) begin code=$fscanf(fd,"%h",t16); suk[i]=t16; end
            for (i = 0; i < KVL*MD;      i = i + 1) begin code=$fscanf(fd,"%h",t8);  cdkv[i]=t8; end
            for (i = 0; i < KVL;         i = i + 1) begin code=$fscanf(fd,"%h",t16); sdkv[i]=t16; end
            for (i = 0; i < H*KVL;       i = i + 1) begin code=$fscanf(fd,"%h",t16); e_qa[i]=t16; end
            for (i = 0; i < KVL;         i = i + 1) begin code=$fscanf(fd,"%h",t16); e_ck[i]=t16; end
            ld_tick = ~ld_tick;

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 1000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[mla_proj] FAIL t%0d: done never asserted", t);
                errors = errors + 1;
            end
            for (i = 0; i < H*KVL; i = i + 1) begin
                checks = checks + 1;
                if (qa[16*i +: 16] !== e_qa[i]) begin
                    if (errors < 8)
                        $display("FAIL t%0d qa[h=%0d][k=%0d]: got %h exp %h",
                                 t, i/KVL, i%KVL, qa[16*i +: 16], e_qa[i]);
                    errors = errors + 1;
                end
            end
            for (i = 0; i < KVL; i = i + 1) begin
                checks = checks + 1;
                if (ckv[16*i +: 16] !== e_ck[i]) begin
                    if (errors < 8)
                        $display("FAIL t%0d ckv[%0d]: got %h exp %h",
                                 t, i, ckv[16*i +: 16], e_ck[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[mla_proj] ALL %0d TESTS PASSED (%0d tokens: W_dq -> rmsnorm -> W_uq, then the per-head fold of W_uk into q, and the RAW latent off W_dkv -- all four streamed as Q8_0 off glm_matmul_q4k, BITWISE against the reference)",
                     checks, ntest);
        else
            $display("[mla_proj] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #100000000; $display("[mla_proj] FAIL: timeout"); $finish; end
endmodule
