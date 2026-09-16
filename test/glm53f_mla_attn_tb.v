//============================================================================
// glm53f_mla_attn_tb.v -- src/glm53f_mla_attn.v, the whole GLM-5.3-Flash MLA
// sublayer, against tools/glm53f_mla_attn_gen.py.
//
// BITWISE END TO END. The golden COMPOSES the generators that already gate the
// parts (mla-proj, mla-score) plus matmul_q4k_col for W_uv and W_o -- nothing is
// re-derived, so a disagreement here means the COMPOSITION is wrong rather than
// the arithmetic.
//
// THE TESTBENCH OWNS THE KV CACHE, because the DUT deliberately does not: it
// publishes the new latent on ckv_wr and pulls old ones by index. At the real
// shapes a 1 M-token context is ~1 GB of latent per MLA block, and that residency
// decision belongs with the model. What the TB does here -- append on ckv_wr, then
// answer c_idx -- is exactly what the system will have to do.
//
// The corpus contains s_len == 1 (first token: the only key is the one written
// this step), a padded window and a full one. A unit that ignored ckv_wr would
// still pass an all-first-token corpus; one that never wrote the -inf pad would
// still pass an all-full-window one.
//
// Must FAIL:
//   INJ_MLAA_SKIP_OUT   feed ctx_lat straight out, skipping W_uv and W_o -- the
//                       output stage silently absent.
//   INJ_MLAA_NO_CKVWR   never publish the new latent, so the cache holds only the
//                       older keys. Invisible on any corpus where s_len == 1
//                       happens to be absent... which is why one is required.
//   INJ_MLAA_MUX_ON_WREQ  select the weight channel by w_req instead of by the
//                       wrapper's state. This is not a hypothetical: it is the
//                       bug this gate actually found. glm_matmul_q4k latches its
//                       header buses on `start`, which a stage pulses one cycle
//                       BEFORE it raises w_req, so the mux published the other
//                       stage's address exactly when the engine sampled w_q8_d.
//                       `make mla-proj` and `make mla-score` both stayed bitwise
//                       throughout -- only the composed gate could see it.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_mla_attn_vec.txt"
`endif

module glm53f_mla_attn_tb;
    localparam integer MD=16, H=2, QK=8, QLORA=8, KVL=8, VD=8, SMAX=4, TN=2, KMAX=32;
    localparam integer PMO = (H*QK > QLORA) ? ((H*QK > KVL) ? H*QK : KVL)
                                            : ((QLORA > KVL) ? QLORA : KVL);
    localparam integer NB8=(KMAX+31)/32, GW=$clog2(PMO/TN+1), KW=$clog2(KMAX+1);
    localparam integer HW=$clog2(H), JW=$clog2(SMAX), SW=$clog2(SMAX+1);

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*MD-1:0] x_in;
    reg  [SW-1:0]    slen;
    wire busy, done, w_req, ckv_wr, c_req;
    wire [2:0]    w_sel;
    wire [HW-1:0] w_head;
    wire [GW-1:0] w_grp;
    wire [KW-1:0] w_k;
    reg  [16*TN-1:0]     w_hp;
    reg  [16*TN*NB8-1:0] w_q8d;
    wire [16*KVL-1:0]    ckv;
    wire [JW-1:0]        c_idx;
    reg  [16*KVL-1:0]    c_vec;
    wire [16*MD-1:0]     y;

    glm53f_mla_attn #(.MODEL_DIM(MD), .H(H), .QK(QK), .QLORA(QLORA), .KVL(KVL),
                      .VD(VD), .SMAX(SMAX), .TN(TN), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_in(x_in), .s_len(slen),
        .w_req(w_req), .w_sel(w_sel), .w_head(w_head), .w_grp(w_grp), .w_k(w_k),
        .w_hp(w_hp), .w_q8_d(w_q8d),
        .ckv_wr(ckv_wr), .ckv_out(ckv), .c_req(c_req), .c_idx(c_idx),
        .c_vec(c_vec), .y_out(y));

    // ---- the Q8_0 weight store, flat, EXPLICIT sensitivity (see
    //      test/glm53f_swiglu_mt_tb.v for why `always @*` over a big memory is
    //      not an option here) ----
    reg [7:0]  cdq [0:QLORA*MD-1];      reg [15:0] sdq [0:QLORA-1];
    reg [7:0]  cuq [0:H*QK*QLORA-1];    reg [15:0] suq [0:H*QK-1];
    reg [7:0]  cuk [0:H*KVL*QK-1];      reg [15:0] suk [0:H*KVL-1];
    reg [7:0]  cdk [0:KVL*MD-1];        reg [15:0] sdk [0:KVL-1];
    reg [7:0]  cuv [0:H*VD*KVL-1];      reg [15:0] suv [0:H*VD-1];
    reg [7:0]  cwo [0:MD*H*VD-1];       reg [15:0] swo [0:MD-1];

    reg ld_tick = 1'b0;
    integer pj, row;
    always @(w_sel, w_head, w_grp, w_k, ld_tick) begin
        w_hp = 0; w_q8d = 0;
        for (pj = 0; pj < TN; pj = pj + 1) begin
            row = w_grp*TN + pj;
            case (w_sel)
                3'd0: begin w_hp[16*pj +: 16] = {8'd0, cdq[row*MD + w_k]};
                            w_q8d[16*(pj*NB8) +: 16] = sdq[row]; end
                3'd1: begin w_hp[16*pj +: 16] = {8'd0, cuq[row*QLORA + w_k]};
                            w_q8d[16*(pj*NB8) +: 16] = suq[row]; end
                3'd2: begin w_hp[16*pj +: 16] = {8'd0, cuk[(w_head*KVL + row)*QK + w_k]};
                            w_q8d[16*(pj*NB8) +: 16] = suk[w_head*KVL + row]; end
                3'd3: begin w_hp[16*pj +: 16] = {8'd0, cdk[row*MD + w_k]};
                            w_q8d[16*(pj*NB8) +: 16] = sdk[row]; end
                3'd4: begin w_hp[16*pj +: 16] = {8'd0, cuv[(w_head*VD + row)*KVL + w_k]};
                            w_q8d[16*(pj*NB8) +: 16] = suv[w_head*VD + row]; end
                default: begin w_hp[16*pj +: 16] = {8'd0, cwo[row*H*VD + w_k]};
                               w_q8d[16*(pj*NB8) +: 16] = swo[row]; end
            endcase
        end
    end

    // ---- the KV cache the TB owns: append on ckv_wr, answer c_idx ----
    reg [15:0] cache [0:SMAX*KVL-1];
    integer kk;
    always @(c_idx, ld_tick) begin
        for (kk = 0; kk < KVL; kk = kk + 1)
            c_vec[16*kk +: 16] = cache[c_idx*KVL + kk];
    end
    integer wi;
    always @(posedge clk) begin
        if (!rst && ckv_wr)
            for (wi = 0; wi < KVL; wi = wi + 1)
                cache[(slen - 1'b1)*KVL + wi] <= ckv[16*wi +: 16];
    end

    integer fd, code, t, i, ntest, e_md, e_h, e_qk, e_ql, e_kv, e_vd, e_sm;
    integer errors, checks, w, sl;
    reg [15:0] t16; reg [7:0] t8;
    reg [15:0] e_ckv [0:KVL-1];
    reg [15:0] e_y   [0:MD-1];

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[glm53f_mla_attn] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d %d %d", ntest, e_md, e_h, e_qk, e_ql, e_kv, e_vd, e_sm);
        if (e_md!=MD || e_h!=H || e_qk!=QK || e_ql!=QLORA || e_kv!=KVL || e_vd!=VD || e_sm!=SMAX) begin
            $display("[glm53f_mla_attn] FAIL: vector dims do not match the TB"); $finish;
        end
        repeat (3) @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d", sl); slen = sl[SW-1:0];
            for (i = 0; i < MD; i = i + 1) begin code=$fscanf(fd,"%h",t16); x_in[16*i +: 16]=t16; end
            for (i = 0; i < QLORA*MD;   i=i+1) begin code=$fscanf(fd,"%h",t8);  cdq[i]=t8;  end
            for (i = 0; i < QLORA;      i=i+1) begin code=$fscanf(fd,"%h",t16); sdq[i]=t16; end
            for (i = 0; i < H*QK*QLORA; i=i+1) begin code=$fscanf(fd,"%h",t8);  cuq[i]=t8;  end
            for (i = 0; i < H*QK;       i=i+1) begin code=$fscanf(fd,"%h",t16); suq[i]=t16; end
            for (i = 0; i < H*KVL*QK;   i=i+1) begin code=$fscanf(fd,"%h",t8);  cuk[i]=t8;  end
            for (i = 0; i < H*KVL;      i=i+1) begin code=$fscanf(fd,"%h",t16); suk[i]=t16; end
            for (i = 0; i < KVL*MD;     i=i+1) begin code=$fscanf(fd,"%h",t8);  cdk[i]=t8;  end
            for (i = 0; i < KVL;        i=i+1) begin code=$fscanf(fd,"%h",t16); sdk[i]=t16; end
            for (i = 0; i < H*VD*KVL;   i=i+1) begin code=$fscanf(fd,"%h",t8);  cuv[i]=t8;  end
            for (i = 0; i < H*VD;       i=i+1) begin code=$fscanf(fd,"%h",t16); suv[i]=t16; end
            for (i = 0; i < MD*H*VD;    i=i+1) begin code=$fscanf(fd,"%h",t8);  cwo[i]=t8;  end
            for (i = 0; i < MD;         i=i+1) begin code=$fscanf(fd,"%h",t16); swo[i]=t16; end
            // the pre-existing cache: SMAX-1 rows, only the first s_len-1 are real
            for (i = 0; i < (SMAX-1)*KVL; i=i+1) begin code=$fscanf(fd,"%h",t16); cache[i]=t16; end
            for (i = (SMAX-1)*KVL; i < SMAX*KVL; i=i+1) cache[i] = 16'h0000;
            for (i = 0; i < KVL; i=i+1) begin code=$fscanf(fd,"%h",t16); e_ckv[i]=t16; end
            for (i = 0; i < MD;  i=i+1) begin code=$fscanf(fd,"%h",t16); e_y[i]=t16;   end
            ld_tick = ~ld_tick;

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            w = 0;
            while (done !== 1'b1 && w < 2000000) begin @(negedge clk); w = w + 1; end
            checks = checks + 1;
            if (done !== 1'b1) begin
                $display("[glm53f_mla_attn] FAIL t%0d (s_len=%0d): done never asserted", t, sl);
                errors = errors + 1;
            end
            for (i = 0; i < KVL; i = i + 1) begin
                checks = checks + 1;
                if (ckv[16*i +: 16] !== e_ckv[i]) begin
                    if (errors < 8)
                        $display("FAIL t%0d ckv[%0d]: got %h exp %h", t, i, ckv[16*i +: 16], e_ckv[i]);
                    errors = errors + 1;
                end
            end
            for (i = 0; i < MD; i = i + 1) begin
                checks = checks + 1;
                if (y[16*i +: 16] !== e_y[i]) begin
                    if (errors < 8)
                        $display("FAIL t%0d (s_len=%0d) y[%0d]: got %h exp %h",
                                 t, sl, i, y[16*i +: 16], e_y[i]);
                    errors = errors + 1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[glm53f_mla_attn] ALL %0d TESTS PASSED (%0d tokens: the WHOLE NoPE MLA sublayer -- Q8_0 projections, the W_uk fold, the absorbed-latent score over a testbench-owned KV cache, then W_uv per head and W_o -- BITWISE against a golden COMPOSED from the already-gated unit generators)",
                     checks, ntest);
        else
            $display("[glm53f_mla_attn] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #400000000; $display("[glm53f_mla_attn] FAIL: timeout"); $finish; end
endmodule
