//============================================================================
// glm53f_moe_block_elab_tb.v -- the decoder block on its NON-DEFAULT arms.
//
// `make dec-block` exercises ATTN_KIND = 0 and FFN_KIND = 0 -- the KDA attention
// and dense Q8_0 SwiGLU of blocks 0-2 -- and nothing else. Without this, neither
// non-default arm (glm53f_moe_ffn in the FFN site, blocks 3-44; glm53f_mla_attn
// in the attention site, the 11 MLA blocks) is ever even ELABORATED by the gate.
// A port-width slip or a bad parameter pass there would surface only in a
// whole-model build, or not at all.
//
// WHAT THIS CLAIMS, PRECISELY.  Three things, and not a fourth:
//   1. the block elaborates at FFN_KIND = 1 AND at ATTN_KIND = 1 (both `$fatal`s
//      are gone and both instances' parameters and port widths line up);
//   2. on the MoE arm the router weight-request port is REACHABLE -- it is a live
//      wire out of glm53f_moe_ffn, not a constant;
//   3. on the DENSE arm that same port is held at 0, which is the tie-off the
//      generate promises so a dense block stays byte-identical.
//   4. on the MLA arm the attention weight-request port is reachable, and on the
//      KDA arm it is held at 0 -- the same contrast, on the other site.
// It does NOT claim either arm COMPUTES anything: that is `make moe-ffn` and
// `make glm53f-mla-attn`, which check them against references with must-fail legs.  Two gates, two
// claims, neither pretending to be the other.
//============================================================================
`timescale 1ns/1ps

module glm53f_moe_block_elab_tb;
    localparam integer MD=16, H=4, KH=2, DK=4, DV=4, RANK=4, CK=4, TN=2, KMAX=32;
    localparam integer INTER=32, NE=8, TK=3;
    localparam integer QK=8, QLORA=8, KVL=8, VD=8, SMAX=4;
    localparam integer EIDXW=$clog2(NE);
    localparam integer HDK=KH*DK, HDV=KH*DV, C=3*HDK;
    localparam integer MIX=(2+4)*4;
    localparam integer NSB=(KMAX+255)/256, NB8=(KMAX+31)/32;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, go = 0;

    wire moe_rw_req_m, moe_rw_req_d;
    wire mla_req_moe, mla_req_den, mla_req_mla, kda_req_mla;
    integer errors = 0, checks = 0, i, first_m = -1;
    reg seen_hi, seen_m, seen_mla, seen_kda_on_mla;

    // ---- the MoE arm ----
    glm53f_decoder_block #(.MODEL_DIM(MD), .H(H), .KH(KH), .DK(DK), .DV(DV),
                           .RANK(RANK), .CONV_K(CK), .TN(TN), .KMAX(KMAX),
                           .INTER(INTER), .FFN_KIND(1), .N_EXPERT(NE), .TOPK(TK)) u_moe (
        .clk(clk), .rst(rst), .start(go), .busy(), .done(),
        .streams_load(1'b0), .streams_init({4*16*MD{1'b0}}), .streams_cur(),
        .a_w_q({16*MIX*4*MD{1'b0}}), .a_w_d({16*MIX{1'b0}}), .a_base({16*MIX{1'b0}}),
        .a_s0(32'h3F800000), .a_s1(32'h3F800000), .a_s2(32'h3F800000),
        .f_w_q({16*MIX*4*MD{1'b0}}), .f_w_d({16*MIX{1'b0}}), .f_base({16*MIX{1'b0}}),
        .f_s0(32'h3F800000), .f_s1(32'h3F800000), .f_s2(32'h3F800000),
        .attn_norm_w({16*MD{1'b0}}), .ffn_norm_w({16*MD{1'b0}}),
        .kda_w_req(), .kda_w_sel(), .kda_w_grp(), .kda_w_k(),
        .kda_w_hp({16*TN{1'b0}}), .kda_w_q8_d({16*TN*NB8{1'b0}}),
        .decay_in({32*C{1'b0}}), .dt_bias_in({32*HDK{1'b0}}),
        .conv_w_in({32*C*CK{1'b0}}), .onorm_w_in({16*HDV{1'b0}}),
        .kda_s_in({32*KH*DK*DV{1'b0}}), .kda_s_out(),
        .kda_hist_in({32*C*(CK-1){1'b0}}), .kda_hist_out(),
        .ffn_w_req(), .ffn_w_sel(), .ffn_w_grp(), .ffn_w_k(),
        .ffn_w_hp({16*TN{1'b0}}), .ffn_w_q8_d({16*TN*NB8{1'b0}}),
        .moe_rw_req(moe_rw_req_m), .moe_rw_k(),
        .moe_rw_row({32*NE{1'b0}}), .moe_bias({32*NE{1'b0}}),
        .moe_fw_shared(), .moe_fw_eidx(),
        .moe_wt_gate(3'd0), .moe_wt_up(3'd0), .moe_wt_down(3'd4),
        .moe_wt_sh_gate(3'd2), .moe_wt_sh_up(3'd2), .moe_wt_sh_down(3'd2),
        .ffn_w_q({4*TN{1'b0}}), .ffn_w_d({16*TN*NSB{1'b0}}),
        .ffn_w_dmin({16*TN*NSB{1'b0}}), .ffn_w_scales({96*TN*NSB{1'b0}}),
        .ffn_w_q6_sc({128*TN*NSB{1'b0}}),
        .mla_s_len(3'd1), .mla_w_req(mla_req_moe), .mla_w_sel(), .mla_w_head(),
        .mla_w_grp(), .mla_w_k(), .mla_w_hp({16*TN{1'b0}}),
        .mla_w_q8_d({16*TN*NB8{1'b0}}), .mla_ckv_wr(), .mla_ckv_out(),
        .mla_c_req(), .mla_c_idx(), .mla_c_vec({16*KVL{1'b0}}));

    // ---- the dense arm, same dimensions, for the tie-off comparison ----
    glm53f_decoder_block #(.MODEL_DIM(MD), .H(H), .KH(KH), .DK(DK), .DV(DV),
                           .RANK(RANK), .CONV_K(CK), .TN(TN), .KMAX(KMAX),
                           .INTER(INTER), .FFN_KIND(0), .N_EXPERT(NE), .TOPK(TK)) u_den (
        .clk(clk), .rst(rst), .start(go), .busy(), .done(),
        .streams_load(1'b0), .streams_init({4*16*MD{1'b0}}), .streams_cur(),
        .a_w_q({16*MIX*4*MD{1'b0}}), .a_w_d({16*MIX{1'b0}}), .a_base({16*MIX{1'b0}}),
        .a_s0(32'h3F800000), .a_s1(32'h3F800000), .a_s2(32'h3F800000),
        .f_w_q({16*MIX*4*MD{1'b0}}), .f_w_d({16*MIX{1'b0}}), .f_base({16*MIX{1'b0}}),
        .f_s0(32'h3F800000), .f_s1(32'h3F800000), .f_s2(32'h3F800000),
        .attn_norm_w({16*MD{1'b0}}), .ffn_norm_w({16*MD{1'b0}}),
        .kda_w_req(), .kda_w_sel(), .kda_w_grp(), .kda_w_k(),
        .kda_w_hp({16*TN{1'b0}}), .kda_w_q8_d({16*TN*NB8{1'b0}}),
        .decay_in({32*C{1'b0}}), .dt_bias_in({32*HDK{1'b0}}),
        .conv_w_in({32*C*CK{1'b0}}), .onorm_w_in({16*HDV{1'b0}}),
        .kda_s_in({32*KH*DK*DV{1'b0}}), .kda_s_out(),
        .kda_hist_in({32*C*(CK-1){1'b0}}), .kda_hist_out(),
        .ffn_w_req(), .ffn_w_sel(), .ffn_w_grp(), .ffn_w_k(),
        .ffn_w_hp({16*TN{1'b0}}), .ffn_w_q8_d({16*TN*NB8{1'b0}}),
        .moe_rw_req(moe_rw_req_d), .moe_rw_k(),
        .moe_rw_row({32*NE{1'b0}}), .moe_bias({32*NE{1'b0}}),
        .moe_fw_shared(), .moe_fw_eidx(),
        .moe_wt_gate(3'd0), .moe_wt_up(3'd0), .moe_wt_down(3'd4),
        .moe_wt_sh_gate(3'd2), .moe_wt_sh_up(3'd2), .moe_wt_sh_down(3'd2),
        .ffn_w_q({4*TN{1'b0}}), .ffn_w_d({16*TN*NSB{1'b0}}),
        .ffn_w_dmin({16*TN*NSB{1'b0}}), .ffn_w_scales({96*TN*NSB{1'b0}}),
        .ffn_w_q6_sc({128*TN*NSB{1'b0}}),
        .mla_s_len(3'd1), .mla_w_req(mla_req_den), .mla_w_sel(), .mla_w_head(),
        .mla_w_grp(), .mla_w_k(), .mla_w_hp({16*TN{1'b0}}),
        .mla_w_q8_d({16*TN*NB8{1'b0}}), .mla_ckv_wr(), .mla_ckv_out(),
        .mla_c_req(), .mla_c_idx(), .mla_c_vec({16*KVL{1'b0}}));

    // ---- a third block on the MLA attention arm (ATTN_KIND = 1) ----
    glm53f_decoder_block #(.MODEL_DIM(MD), .H(H), .KH(KH), .DK(DK), .DV(DV),
                           .RANK(RANK), .CONV_K(CK), .TN(TN), .KMAX(KMAX),
                           .INTER(INTER), .FFN_KIND(0), .N_EXPERT(NE), .TOPK(TK),
                           .ATTN_KIND(1), .QK(QK), .QLORA(QLORA), .KVL(KVL),
                           .VD(VD), .SMAX(SMAX)) u_mla (
        .clk(clk), .rst(rst), .start(go), .busy(), .done(),
        .streams_load(1'b0), .streams_init({4*16*MD{1'b0}}), .streams_cur(),
        .a_w_q({16*MIX*4*MD{1'b0}}), .a_w_d({16*MIX{1'b0}}), .a_base({16*MIX{1'b0}}),
        .a_s0(32'h3F800000), .a_s1(32'h3F800000), .a_s2(32'h3F800000),
        .f_w_q({16*MIX*4*MD{1'b0}}), .f_w_d({16*MIX{1'b0}}), .f_base({16*MIX{1'b0}}),
        .f_s0(32'h3F800000), .f_s1(32'h3F800000), .f_s2(32'h3F800000),
        .attn_norm_w({16*MD{1'b0}}), .ffn_norm_w({16*MD{1'b0}}),
        .kda_w_req(kda_req_mla), .kda_w_sel(), .kda_w_grp(), .kda_w_k(),
        .kda_w_hp({16*TN{1'b0}}), .kda_w_q8_d({16*TN*NB8{1'b0}}),
        .decay_in({32*C{1'b0}}), .dt_bias_in({32*HDK{1'b0}}),
        .conv_w_in({32*C*CK{1'b0}}), .onorm_w_in({16*HDV{1'b0}}),
        .kda_s_in({32*KH*DK*DV{1'b0}}), .kda_s_out(),
        .kda_hist_in({32*C*(CK-1){1'b0}}), .kda_hist_out(),
        .ffn_w_req(), .ffn_w_sel(), .ffn_w_grp(), .ffn_w_k(),
        .ffn_w_hp({16*TN{1'b0}}), .ffn_w_q8_d({16*TN*NB8{1'b0}}),
        .moe_rw_req(), .moe_rw_k(), .moe_rw_row({32*NE{1'b0}}),
        .moe_bias({32*NE{1'b0}}), .moe_fw_shared(), .moe_fw_eidx(),
        .moe_wt_gate(3'd0), .moe_wt_up(3'd0), .moe_wt_down(3'd4),
        .moe_wt_sh_gate(3'd2), .moe_wt_sh_up(3'd2), .moe_wt_sh_down(3'd2),
        .ffn_w_q({4*TN{1'b0}}), .ffn_w_d({16*TN*NSB{1'b0}}),
        .ffn_w_dmin({16*TN*NSB{1'b0}}), .ffn_w_scales({96*TN*NSB{1'b0}}),
        .ffn_w_q6_sc({128*TN*NSB{1'b0}}),
        .mla_s_len(3'd1), .mla_w_req(mla_req_mla), .mla_w_sel(), .mla_w_head(),
        .mla_w_grp(), .mla_w_k(), .mla_w_hp({16*TN{1'b0}}),
        .mla_w_q8_d({16*TN*NB8{1'b0}}), .mla_ckv_wr(), .mla_ckv_out(),
        .mla_c_req(), .mla_c_idx(), .mla_c_vec({16*KVL{1'b0}}));

    initial begin
        repeat (4) @(negedge clk);
        rst = 0;

        // 1. it got here at all: FFN_KIND=1 elaborated without $fatal
        checks = checks + 1;

        // Both blocks are RUN, not merely reset.  Checking the ports while idle
        // would be vacuous -- an idle MoE block holds moe_rw_req at 0 too, so
        // "it stayed 0" would pass for either arm and prove nothing.  The weight
        // buses are tied to zero, which is valid data: the block still walks its
        // attention phase and reaches the FFN site, which is where the arms
        // diverge.  The MoE arm's router then asks for ffn_gate_inp rows.
        @(negedge clk); go = 1;
        @(negedge clk); go = 0;
        seen_hi = 1'b0; seen_m = 1'b0;
        // Bound MEASURED, not guessed: the MoE router first asks for a weight
        // row at cycle 2744 (printed below, so a change is visible rather than
        // absorbed).  20000 is comfortable headroom; 200000 cost 526 s of gate
        // time for the same claim.  Both arms are observed over the SAME window,
        // which is what makes the contrast between them meaningful.
        seen_mla = 1'b0; seen_kda_on_mla = 1'b0;
        for (i = 0; i < 20000 && !(seen_m && seen_mla); i = i + 1) begin
            @(negedge clk);
            if (moe_rw_req_d === 1'b1) seen_hi = 1'b1;
            if (moe_rw_req_m === 1'b1) begin seen_m = 1'b1; first_m = i; end
            if (mla_req_mla  === 1'b1) seen_mla = 1'b1;
            if (kda_req_mla  === 1'b1) seen_kda_on_mla = 1'b1;
        end
        $display("[moe_block_elab] MoE router first asked for a weight row at cycle %0d", first_m);

        // 2. the MoE arm really is in the site: its router asked for weights
        checks = checks + 1;
        if (!seen_m) begin
            $display("[moe_block_elab] FAIL: FFN_KIND=1 never asserted moe_rw_req -- the MoE FFN is not reached in the site");
            errors = errors + 1;
        end

        // 3. the dense arm holds the same port tied off across the whole run
        checks = checks + 1;
        if (seen_hi) begin
            $display("[moe_block_elab] FAIL: FFN_KIND=0 drove moe_rw_req -- the dense arm is not tied off");
            errors = errors + 1;
        end

        // 4. the MLA arm is really in the ATTENTION site: it asked for weights,
        //    and the KDA arm it replaced is held quiet in the same block.
        checks = checks + 1;
        if (!seen_mla) begin
            $display("[moe_block_elab] FAIL: ATTN_KIND=1 never asserted mla_w_req -- the MLA sublayer is not reached in the site");
            errors = errors + 1;
        end
        checks = checks + 1;
        if (seen_kda_on_mla) begin
            $display("[moe_block_elab] FAIL: ATTN_KIND=1 still drove kda_w_req -- the KDA arm is not tied off");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("[moe_block_elab] ALL %0d TESTS PASSED (the decoder block elaborates on BOTH non-default arms -- FFN_KIND=1, the MoE of blocks 3-44, and ATTN_KIND=1, the MLA of the 11 non-KDA blocks -- each with its ports live while the arm it replaced is held tied off in the same block; FUNCTION is `make moe-ffn` and `make glm53f-mla-attn`, not this)", checks);
        else
            $display("[moe_block_elab] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #40000000; $display("[moe_block_elab] FAIL: timeout"); $finish; end
endmodule
