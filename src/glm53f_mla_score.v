//============================================================================
// glm53f_mla_score.v -- the inner loop of GLM-5.3-Flash MLA attention, in the
// ABSORBED LATENT form. This is the part that runs once per CACHED TOKEN, so it
// is the whole decode cost of the 11 MLA blocks.
//
//     score[h][j] = ( qa[h] . rmsnorm(c_kv[j]) ) * SCALE
//     p[h][.]     = softmax over SMAX slots, slots >= s_len forced to -inf
//     ctx_lat[h]  = SUM_j p[h][j] * rmsnorm(c_kv[j])            [KVL wide]
//
// `qa` is q ALREADY FOLDED THROUGH W_uk: qa[h] = q[h] @ W_uk[h]. Folding it once
// is the point of the latent form -- expanding a key instead costs H*QK MACs over
// KVL for EVERY cached token (64*256*512 = 8.4 M at the real shapes) against
// H*KVL = 32 K for a dot against the latent, a factor of qk_nope_head_dim = 256.
// The output stays in the LATENT basis for the same reason: W_uv is linear, so
// SUM_j p_j (W_uv . ckv_j) == W_uv . (SUM_j p_j ckv_j) and the caller expands
// ONCE at the end instead of building a [H,256] value per token.
//
// NEITHER REWRITE IS FREE IN fp, and that is why the golden has to match the
// form rather than the formula. Measured in tools/glm53f_mla_ref.py, absorbed vs
// expanded at the checkpoint's widths: 517/768 outputs differ (rel 3.4e-5) for
// the q side, 693/768 (rel 8.2e-5) with the value side folded too. At a toy
// qk = 8 they are bitwise identical, which is why the reference's self-test runs
// a slice wide enough to see the difference.
//
// WHY THIS IS A SIBLING AND NOT mla_attn_q4k RE-DIMENSIONED. GLM-5.3-Flash is
// NoPE -- [gguf] rope.dimension_count = 0, [cfg] qk_rope_head_dim = 0,
// mla_use_nope = true -- so there is no W_kr, no k_rope cache and no rotation of
// q. mla_attn_q4k carries all three, and a zero-width rotary tail is not
// expressible in Verilog. The structural consequence, which the config guard
// already asserts, is `attention.key_length == kv_lora_rank` (both 512): on
// GLM-5.2 the key was the latent PLUS a 64-wide rotary tail. Its seven GEMMs are
// also Q4_K-native (w_q is four bits per lane) while these projections are Q8_0
// [scan] -- the same wall the FFN hit.
//
// ARITHMETIC, stated because the generator models it exactly rather than
// approximately: every product is bf16 (glm_fp.vh bf16_mul), every accumulation
// is fp32 (fp32_add) in index order, and each result is rounded to bf16 once at
// the end. The latent is normalised by the repo's own gated rmsnorm_unit with
// gamma = 1, and the softmax is the repo's own gated glm_softmax -- so this unit
// adds no new numerics, only a new dataflow.
//
// THE CACHE IS PULLED, NOT HELD. c_req/c_idx ask the caller for one key's latent;
// it is re-read on the second pass rather than stored, because storing SMAX*KVL
// here would be 2048*512*16 = 16.8 Mbit at the real shapes. Two passes over a
// pulled cache is the trade this makes, and it is why rmsnorm runs twice per key.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_MLA_SCORE_V
`define GLM53F_MLA_SCORE_V
`include "glm_fp.vh"

module glm53f_mla_score #(
    parameter integer H       = 2,              // heads (real 64)
    parameter integer KVL     = 8,              // kv_lora_rank (real 512)
    parameter integer SMAX    = 4,              // attention window (real 2048)
    parameter [31:0]  SCALE   = 32'h3D800000,   // 1/sqrt(qk_nope_head_dim)
    parameter [31:0]  RMS_EPS = 32'h3727C5AC,
    parameter [15:0]  NEG_BIG = 16'hFF80        // -inf bf16, the softmax pad
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output reg                       busy,
    output reg                       done,

    input  wire [16*H*KVL-1:0]       qa_in,     // q folded through W_uk
    input  wire [$clog2(SMAX+1)-1:0] s_len,     // real keys this step

    // latent cache pull: caller drives c_vec for key c_idx while c_req is high
    output reg                       c_req,
    output reg [$clog2(SMAX)-1:0]    c_idx,
    input  wire [16*KVL-1:0]         c_vec,

    output reg [16*H*KVL-1:0]        ctx_out    // ctx in the LATENT basis
);
    localparam integer JW = $clog2(SMAX);
    localparam integer KW = (KVL <= 1) ? 1 : $clog2(KVL);
    localparam integer HW = (H   <= 1) ? 1 : $clog2(H);
    localparam integer SW = $clog2(SMAX + 1);

    // ---- the repo's own gated rmsnorm, gamma = 1 ----
    reg          rn_start, rn_xv, rn_gv;
    reg  [15:0]  rn_x, rn_g;
    wire         rn_inreq, rn_greq, rn_yv, rn_busy, rn_done;
    wire [15:0]  rn_y;
    rmsnorm_unit #(.LEN(KVL), .LANES(1), .EPS(RMS_EPS)) u_rn (
        .clk(clk), .rst(rst), .start(rn_start),
        .in_req(rn_inreq), .x_in(rn_x), .x_valid(rn_xv),
        .g_req(rn_greq),  .gamma_in(rn_g), .g_valid(rn_gv),
        .y_valid(rn_yv), .y_out(rn_y), .busy(rn_busy), .done(rn_done));

    // ---- the repo's own gated softmax ----
    reg          sm_start, sm_iv;
    reg  [15:0]  sm_x;
    wire         sm_busy, sm_ov, sm_done;
    wire [15:0]  sm_p;
    glm_softmax #(.LEN(SMAX), .LANES(1)) u_sm (
        .clk(clk), .rst(rst), .start(sm_start),
        .in_valid(sm_iv), .x_in(sm_x),
        .busy(sm_busy), .out_valid(sm_ov), .p_out(sm_p), .done(sm_done));

    reg [15:0] ckvn [0:KVL-1];              // one normalised latent, re-read per pass
    reg [15:0] scr  [0:H*SMAX-1];
    reg [15:0] prb  [0:H*SMAX-1];
    reg [31:0] acc  [0:H*KVL-1];

    reg [JW:0]  jcnt;                        // key cursor
    reg [KW:0]  kcnt, xidx, gidx, yidx;
    reg [HW:0]  hcnt;
    reg [SW-1:0] fi, ci;
    reg         pass;                        // 0 = score, 1 = weighted latent sum
    reg [31:0]  dacc;

    localparam [3:0] S_IDLE=4'd0, S_RN=4'd1,  S_DOT=4'd2,  S_NEXTJ=4'd3,
                     S_SMF =4'd4, S_SMC=4'd5, S_SMN=4'd6,  S_OUT=4'd7,
                     S_FIN =4'd8;
    reg [3:0] st;
    integer i;

    wire [15:0] qa_h_k = qa_in[16*(hcnt*KVL + kcnt) +: 16];

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; c_req <= 1'b0;
            rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0;
            sm_start <= 1'b0; sm_iv <= 1'b0;
            jcnt <= 0; kcnt <= 0; hcnt <= 0; pass <= 1'b0;
            xidx <= 0; gidx <= 0; yidx <= 0; fi <= 0; ci <= 0; dacc <= 32'd0;
        end else begin
            done <= 1'b0; rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0;
            sm_start <= 1'b0; sm_iv <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; pass <= 1'b0; jcnt <= 0;
                    for (i = 0; i < H*KVL; i = i + 1) acc[i] <= 32'd0;
`ifdef INJ_MLAS_NOPAD
                    // must FAIL: leave the unused slots at +0 instead of -inf, so
                    // padding collects softmax weight. The classic masking bug --
                    // and INVISIBLE on any window where s_len == SMAX, which is
                    // why the corpus is required to contain shorter ones.
                    // Measured against the golden: 48 of 96 ctx elements move,
                    // all of them in the padded windows.
                    for (i = 0; i < H*SMAX; i = i + 1) scr[i] <= 16'h0000;
`else
                    // pad EVERY slot; the real ones are overwritten below
                    for (i = 0; i < H*SMAX; i = i + 1) scr[i] <= NEG_BIG;
`endif
                    c_req <= 1'b1; c_idx <= 0;
                    xidx <= 0; gidx <= 0; yidx <= 0;
                    rn_start <= 1'b1; st <= S_RN;
                end

                // normalise this key's latent (gamma = 1)
                S_RN: begin
                    if (rn_inreq) begin rn_x <= c_vec[16*xidx +: 16]; rn_xv <= 1'b1; xidx <= xidx + 1'b1; end
                    if (rn_greq)  begin rn_g <= 16'h3F80;             rn_gv <= 1'b1; gidx <= gidx + 1'b1; end
                    if (rn_yv)    begin ckvn[yidx] <= rn_y;                          yidx <= yidx + 1'b1; end
                    if (rn_done) begin
                        c_req <= 1'b0; hcnt <= 0; kcnt <= 0; dacc <= 32'd0; st <= S_DOT;
                    end
                end

                // pass 0: score[h][j] = bf16(fp32 sum_k bf16(qa[h][k]*ckvn[k]) * SCALE)
                // pass 1: acc[h][k] += bf16(p[h][j]*ckvn[k])      (fp32 accumulate)
                S_DOT: begin
                    if (!pass) begin
                        dacc <= fp32_add(dacc,
                                  bf16_to_fp32(bf16_mul(qa_h_k, ckvn[kcnt])));
                        if (kcnt == KVL[KW:0] - 1'b1) begin
`ifdef INJ_MLAS_NORESCALE
                            // must FAIL: drop the 1/sqrt(qk_nope) score scale.
                            // softmax is shift-invariant but NOT scale-invariant,
                            // so this sharpens or flattens every distribution.
                            // Measured: 80 of 96 ctx elements move.
                            scr[hcnt*SMAX + jcnt] <= fp32_to_bf16(
                                fp32_add(dacc, bf16_to_fp32(bf16_mul(qa_h_k, ckvn[kcnt]))));
`else
                            scr[hcnt*SMAX + jcnt] <= fp32_to_bf16(fp32_mul(
                                fp32_add(dacc, bf16_to_fp32(bf16_mul(qa_h_k, ckvn[kcnt]))),
                                SCALE));
`endif
                            kcnt <= 0; dacc <= 32'd0;
                            if (hcnt == H[HW:0] - 1'b1) st <= S_NEXTJ;
                            else hcnt <= hcnt + 1'b1;
                        end else kcnt <= kcnt + 1'b1;
                    end else begin
`ifdef INJ_MLAS_HEAD0_PROBS
                        // must FAIL: weight every head's latent sum with HEAD 0's
                        // probabilities. Right shape, right magnitudes, wrong
                        // attention -- the failure a shape-only check misses.
                        // Measured: 40 of 96 ctx elements move.
                        acc[hcnt*KVL + kcnt] <= fp32_add(acc[hcnt*KVL + kcnt],
                            bf16_to_fp32(bf16_mul(prb[jcnt], ckvn[kcnt])));
`else
                        acc[hcnt*KVL + kcnt] <= fp32_add(acc[hcnt*KVL + kcnt],
                            bf16_to_fp32(bf16_mul(prb[hcnt*SMAX + jcnt], ckvn[kcnt])));
`endif
                        if (kcnt == KVL[KW:0] - 1'b1) begin
                            kcnt <= 0;
                            if (hcnt == H[HW:0] - 1'b1) st <= S_NEXTJ;
                            else hcnt <= hcnt + 1'b1;
                        end else kcnt <= kcnt + 1'b1;
                    end
                end

                S_NEXTJ: begin
                    if (jcnt == s_len - 1'b1) begin
                        if (!pass) begin
                            hcnt <= 0; fi <= 0; ci <= 0; sm_start <= 1'b1; st <= S_SMF;
                        end else st <= S_OUT;
                    end else begin
                        jcnt <= jcnt + 1'b1;
                        c_idx <= c_idx + 1'b1; c_req <= 1'b1;
                        xidx <= 0; gidx <= 0; yidx <= 0;
                        rn_start <= 1'b1; st <= S_RN;
                    end
                end

                // softmax per head over ALL SMAX slots (pads are already -inf)
                S_SMF: begin
                    sm_iv <= 1'b1;
                    sm_x  <= scr[hcnt*SMAX + fi];
                    fi    <= fi + 1'b1;
                    if (fi == SMAX[SW-1:0] - 1'b1) st <= S_SMC;
                end
                S_SMC: begin
                    if (sm_ov) begin
                        prb[hcnt*SMAX + ci] <= (ci < s_len) ? sm_p : 16'h0000;
                        ci <= ci + 1'b1;
                    end
                    if (sm_done) st <= S_SMN;
                end
                S_SMN: begin
                    if (hcnt == H[HW:0] - 1'b1) begin
                        pass <= 1'b1; jcnt <= 0; hcnt <= 0; kcnt <= 0;
                        c_req <= 1'b1; c_idx <= 0;
                        xidx <= 0; gidx <= 0; yidx <= 0;
                        rn_start <= 1'b1; st <= S_RN;
                    end else begin
                        hcnt <= hcnt + 1'b1; fi <= 0; ci <= 0;
                        sm_start <= 1'b1; st <= S_SMF;
                    end
                end

                S_OUT: begin
                    for (i = 0; i < H*KVL; i = i + 1)
                        ctx_out[16*i +: 16] <= fp32_to_bf16(acc[i]);
                    st <= S_FIN;
                end
                S_FIN: begin done <= 1'b1; busy <= 1'b0; c_req <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_MLA_SCORE_V
