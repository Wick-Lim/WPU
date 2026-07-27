`timescale 1ns/1ps
//============================================================================
// glm_model_q4k_full_tb.v -- ASSEMBLED-FORWARD EXACTNESS TB for glm_model_q4k.
//----------------------------------------------------------------------------
// WHAT THIS PROVES  (the #1 correctness gap: the assembled Q4_K forward had NO
//   functional golden -- the model-level TBs ran the generic bf16 twin, not the
//   _q4k product, so ZERO lines of the assembled Q4_K numeric path were checked
//   vs any golden).
//
//   This drives the REAL product top `glm_model_q4k` at the committed slice
//   (MODEL_DIM=128, L=6, N_DENSE=3, VOCAB=256, ...) with the SAME weights +
//   inputs as the assembled numpy golden tools/glm_model_q4k_ref.py (emitted by
//   tools/glm_model_q4k_tb_gen.py as $readmemh hex), for several (token,pos,
//   s_len) cases, and asserts -- BIT-EXACT, X-aware:
//       (1) logits[VOCAB]   == golden logits_bits   (every bf16 pattern)
//       (2) argmax          == golden argmax
//       (3) h_state[MODEL_DIM] == golden xn (final-RMSNorm = LM-head input)
//
//   The golden APPLIES the Phase-1 MLA softmax scale 1/sqrt(QK_DIM); the DUT now
//   does too (src/mla_attn_q4k.v).  Before that fix this TB DIVERGES (unscaled
//   scores -> different softmax -> different logits/argmax), which is exactly the
//   divergence it is built to expose.
//
//   Weight delivery mirrors test/spec_decode_top_tb.v's pull responders, but the
//   Q4_K super-block d/dmin/scales are PER-COLUMN (read from ROM), not global
//   constants -- the committed slice reduces over up to 256 elements (8 Q4_K
//   sub-blocks per column), so the full per-column super-block is exercised.
//============================================================================
module glm_model_q4k_full_tb;
`ifdef SPEC_SLICE
    // ---- fast SPEC_SLICE smoke config (glm_model_q4k_ref.SPEC_SLICE); seconds/forward.
    //   Validates the whole harness (weight indexing, MLA scale, compare) quickly;
    //   the committed slice below is the real gate.  Vectors: build/mq4k_s (--spec).
    localparam integer MODEL_DIM = 16, L = 2, N_DENSE = 1, VOCAB = 16;
    localparam integer H_HEADS = 2, NOPE = 4, ROPE = 4, V_DIM = 4;
    localparam integer Q_LORA = 8, KV_LORA = 8, S_MAX = 2, TOPK_ATTN = 2;
    localparam integer THETA = 8000000, PE_N = 2, POSW = 20, N_EXPERT = 4;
    localparam integer TOPK = 2, INTER_MOE = 8, INTER_DENSE = 32;
    localparam [31:0]  RSCALE = 32'h40200000;
    localparam integer TN = 4, BLK = 128, LM_TN = 4;
    localparam DIR = "build/mq4k_s";
`else
    // ---- committed slice (glm_model_q4k.v module defaults == Config() golden) ----
    localparam integer MODEL_DIM = 128, L = 6, N_DENSE = 3, VOCAB = 256;
    localparam integer H_HEADS = 4, NOPE = 16, ROPE = 16, V_DIM = 32;
    localparam integer Q_LORA = 64, KV_LORA = 32, S_MAX = 8, TOPK_ATTN = 8;
    localparam integer THETA = 8000000, PE_N = 4, POSW = 20, N_EXPERT = 8;
    localparam integer TOPK = 2, INTER_MOE = 64, INTER_DENSE = 256;
    localparam [31:0]  RSCALE = 32'h40200000;
    localparam integer TN = 4, BLK = 128, LM_TN = 4;
    localparam DIR = "build/mq4k";
`endif

    // ---- ACT_HW under test: 0 = full-width glm_act (the shipping default).
    //   Overriding with -DTB_ACT_HW=n runs the SAME golden vectors through the
    //   lane-serialized activation datapath -- ALL tests passing IS the
    //   result-invariance proof for the ACT_HW resource knob. ----
`ifdef TB_ACT_HW
    localparam integer ACT_HW = `TB_ACT_HW;
`else
    localparam integer ACT_HW = 0;
`endif

    // ---- GU_CONC under test: 0 = committed serial gate-then-up passes.
    //   -DTB_GU_CONC=1 runs the SAME golden vectors with the up GEMV executed
    //   CONCURRENTLY with the gate GEMV.  This gate has to be the one that proves
    //   rank 9 bit-exact: the perf TB's binding check compares glm_q4k_system
    //   against a standalone glm_model_q4k and BOTH instantiate
    //   swiglu_expert_q4k, so a leaf parameter moves both sides identically and
    //   that check goes BLIND.  The numpy golden contains no RTL, so it cannot. ----
`ifdef TB_GU_CONC
    localparam integer GU_CONC = `TB_GU_CONC;
`else
    localparam integer GU_CONC = 0;
`endif

    // ---- derived widths (mirror the DUT) ----
    localparam integer IDXW   = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer HQK    = H_HEADS * QK_DIM;
    localparam integer HNOPE  = H_HEADS * NOPE;
    localparam integer HV     = H_HEADS * V_DIM;
    localparam integer EIDXW  = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT);
    localparam integer A_KMAX = (MODEL_DIM > Q_LORA) ?
                              ((MODEL_DIM > KV_LORA) ? ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                                                     : ((KV_LORA > HV) ? KV_LORA : HV))
                            : ((Q_LORA > KV_LORA) ? ((Q_LORA > HV) ? Q_LORA : HV)
                                                  : ((KV_LORA > HV) ? KV_LORA : HV));
    localparam integer A_OMAX = (HQK > MODEL_DIM) ?
                              ((HQK > HNOPE) ? ((HQK > HV) ? HQK : HV) : ((HNOPE > HV) ? HNOPE : HV))
                            : ((MODEL_DIM > HNOPE) ? ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                                                   : ((HNOPE > HV) ? HNOPE : HV));
    localparam integer A_NGMAX = (A_OMAX + PE_N - 1) / PE_N;
    localparam integer A_GRPW  = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX);
    localparam integer A_KCW   = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX);
    localparam integer FF_GWD  = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1);
    localparam integer FF_KMAX_D = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM;
    localparam integer FF_KWD  = $clog2(FF_KMAX_D + 1);
    localparam integer FF_KMAX_M = (INTER_MOE > MODEL_DIM) ? INTER_MOE : MODEL_DIM;
    localparam integer R_KW    = $clog2(FF_KMAX_M + 1);
    localparam integer A_NSB    = (A_KMAX    + 255) / 256;
    localparam integer FF_NSB_D = (FF_KMAX_D + 255) / 256;
    localparam integer R_NSB    = (FF_KMAX_M + 255) / 256;
    localparam integer LAYW    = (L     <= 1) ? 1 : $clog2(L);
    localparam integer TOKW    = (VOCAB <= 1) ? 1 : $clog2(VOCAB);
    localparam integer DIMW    = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM);
    localparam integer NVTILE  = VOCAB / LM_TN;
    localparam integer VTW     = (NVTILE <= 1) ? 1 : $clog2(NVTILE);

    // weight-select codes (mla_attn_q4k SEL_*)
    localparam [3:0] SEL_DQ=4'd0, SEL_UQ=4'd1, SEL_DKV=4'd2, SEL_KR=4'd3,
                     SEL_UK=4'd4, SEL_UV=4'd5, SEL_O=4'd6;

    localparam integer MAXCASE = 16;

    // ---- clock / reset ----
    reg clk = 1'b0;  always #5 clk = ~clk;
    reg rst;

    //========================================================================
    // WEIGHT ROMs (flat; $readmemh from the golden generator's hex).
    //========================================================================
    reg [15:0] EMB [0:VOCAB*MODEL_DIM-1];
    reg [15:0] GF  [0:MODEL_DIM-1];
    reg [15:0] WLM [0:VOCAB*MODEL_DIM-1];
    reg [15:0] G1  [0:L*MODEL_DIM-1];
    reg [15:0] G2  [0:L*MODEL_DIM-1];
    reg [15:0] CKV [0:L*S_MAX*KV_LORA-1];
    reg [15:0] KRP [0:L*S_MAX*ROPE-1];

    // Q4_K: <name>_C codes(4b, PACKED 16/64b-word), _D d(fp16), _DM dmin(fp16),
    //   _SC scales(96b).  Codes packed 16-per-word keeps the $readmemh arrays (and
    //   the combinational responders' sensitivity fan-in) ~16x smaller so the FULL
    //   committed-slice weight set compiles+runs in iverilog.  `RDC reads code fi.
    `define QW(NM, NCODE, NCOL) \
        reg [63:0] NM``_C  [0:((NCODE)+15)/16-1]; \
        reg [15:0] NM``_D  [0:(NCOL)-1];  \
        reg [15:0] NM``_DM [0:(NCOL)-1];  \
        reg [95:0] NM``_SC [0:(NCOL)-1]
    // read 4-bit code at flat index FI from a packed 64-bit code array ARR.
    `define RDC(ARR, FI) (ARR[(FI) >> 4][(((FI) & 15) << 2) +: 4])
    `QW(WDQ, L*Q_LORA*MODEL_DIM,          L*Q_LORA);
    `QW(WUQ, L*HQK*Q_LORA,                L*HQK);
    `QW(WUK, L*HNOPE*KV_LORA,             L*HNOPE);
    `QW(WUV, L*HV*KV_LORA,                L*HV);
    `QW(WO,  L*MODEL_DIM*HV,              L*MODEL_DIM);
    `QW(WG,  L*N_EXPERT*MODEL_DIM,        L*N_EXPERT);
    `QW(DG,  L*INTER_DENSE*MODEL_DIM,     L*INTER_DENSE);
    `QW(DU,  L*INTER_DENSE*MODEL_DIM,     L*INTER_DENSE);
    `QW(DD,  L*MODEL_DIM*INTER_DENSE,     L*MODEL_DIM);
    `QW(MG,  L*N_EXPERT*INTER_MOE*MODEL_DIM, L*N_EXPERT*INTER_MOE);
    `QW(MU,  L*N_EXPERT*INTER_MOE*MODEL_DIM, L*N_EXPERT*INTER_MOE);
    `QW(MD,  L*N_EXPERT*MODEL_DIM*INTER_MOE, L*N_EXPERT*MODEL_DIM);
    `QW(SHG, L*INTER_MOE*MODEL_DIM,       L*INTER_MOE);
    `QW(SHU, L*INTER_MOE*MODEL_DIM,       L*INTER_MOE);
    `QW(SHD, L*MODEL_DIM*INTER_MOE,       L*MODEL_DIM);

    // golden outputs (per case, loaded on demand)
    reg [15:0] GLOGIT [0:VOCAB-1];
    reg [15:0] GXN    [0:MODEL_DIM-1];
    reg [15:0] GARG_M [0:0];
    reg [TOKW-1:0] GARG;

    // stimulus (per case: token pos s_len)
    reg [31:0] STIM [0:3*MAXCASE-1];
    reg [31:0] NCASE_M [0:0];
    integer NCASE;

    task load_weights; begin
        $readmemh({DIR,"/emb.hex"}, EMB);
        $readmemh({DIR,"/gf.hex"},  GF);
        $readmemh({DIR,"/wlm.hex"}, WLM);
        $readmemh({DIR,"/g1.hex"},  G1);
        $readmemh({DIR,"/g2.hex"},  G2);
        $readmemh({DIR,"/ckv.hex"}, CKV);
        $readmemh({DIR,"/krp.hex"}, KRP);
        $readmemh({DIR,"/wdq_c.hex"},WDQ_C); $readmemh({DIR,"/wdq_d.hex"},WDQ_D); $readmemh({DIR,"/wdq_dm.hex"},WDQ_DM); $readmemh({DIR,"/wdq_sc.hex"},WDQ_SC);
        $readmemh({DIR,"/wuq_c.hex"},WUQ_C); $readmemh({DIR,"/wuq_d.hex"},WUQ_D); $readmemh({DIR,"/wuq_dm.hex"},WUQ_DM); $readmemh({DIR,"/wuq_sc.hex"},WUQ_SC);
        $readmemh({DIR,"/wuk_c.hex"},WUK_C); $readmemh({DIR,"/wuk_d.hex"},WUK_D); $readmemh({DIR,"/wuk_dm.hex"},WUK_DM); $readmemh({DIR,"/wuk_sc.hex"},WUK_SC);
        $readmemh({DIR,"/wuv_c.hex"},WUV_C); $readmemh({DIR,"/wuv_d.hex"},WUV_D); $readmemh({DIR,"/wuv_dm.hex"},WUV_DM); $readmemh({DIR,"/wuv_sc.hex"},WUV_SC);
        $readmemh({DIR,"/wo_c.hex"}, WO_C);  $readmemh({DIR,"/wo_d.hex"}, WO_D);  $readmemh({DIR,"/wo_dm.hex"}, WO_DM);  $readmemh({DIR,"/wo_sc.hex"}, WO_SC);
        $readmemh({DIR,"/wg_c.hex"}, WG_C);  $readmemh({DIR,"/wg_d.hex"}, WG_D);  $readmemh({DIR,"/wg_dm.hex"}, WG_DM);  $readmemh({DIR,"/wg_sc.hex"}, WG_SC);
        $readmemh({DIR,"/dg_c.hex"}, DG_C);  $readmemh({DIR,"/dg_d.hex"}, DG_D);  $readmemh({DIR,"/dg_dm.hex"}, DG_DM);  $readmemh({DIR,"/dg_sc.hex"}, DG_SC);
        $readmemh({DIR,"/du_c.hex"}, DU_C);  $readmemh({DIR,"/du_d.hex"}, DU_D);  $readmemh({DIR,"/du_dm.hex"}, DU_DM);  $readmemh({DIR,"/du_sc.hex"}, DU_SC);
        $readmemh({DIR,"/dd_c.hex"}, DD_C);  $readmemh({DIR,"/dd_d.hex"}, DD_D);  $readmemh({DIR,"/dd_dm.hex"}, DD_DM);  $readmemh({DIR,"/dd_sc.hex"}, DD_SC);
        $readmemh({DIR,"/mg_c.hex"}, MG_C);  $readmemh({DIR,"/mg_d.hex"}, MG_D);  $readmemh({DIR,"/mg_dm.hex"}, MG_DM);  $readmemh({DIR,"/mg_sc.hex"}, MG_SC);
        $readmemh({DIR,"/mu_c.hex"}, MU_C);  $readmemh({DIR,"/mu_d.hex"}, MU_D);  $readmemh({DIR,"/mu_dm.hex"}, MU_DM);  $readmemh({DIR,"/mu_sc.hex"}, MU_SC);
        $readmemh({DIR,"/md_c.hex"}, MD_C);  $readmemh({DIR,"/md_d.hex"}, MD_D);  $readmemh({DIR,"/md_dm.hex"}, MD_DM);  $readmemh({DIR,"/md_sc.hex"}, MD_SC);
        $readmemh({DIR,"/shg_c.hex"},SHG_C); $readmemh({DIR,"/shg_d.hex"},SHG_D); $readmemh({DIR,"/shg_dm.hex"},SHG_DM); $readmemh({DIR,"/shg_sc.hex"},SHG_SC);
        $readmemh({DIR,"/shu_c.hex"},SHU_C); $readmemh({DIR,"/shu_d.hex"},SHU_D); $readmemh({DIR,"/shu_dm.hex"},SHU_DM); $readmemh({DIR,"/shu_sc.hex"},SHU_SC);
        $readmemh({DIR,"/shd_c.hex"},SHD_C); $readmemh({DIR,"/shd_d.hex"},SHD_D); $readmemh({DIR,"/shd_dm.hex"},SHD_DM); $readmemh({DIR,"/shd_sc.hex"},SHD_SC);
    end endtask

    //========================================================================
    // DUT port wires
    //========================================================================
    reg               start;
    reg  [TOKW-1:0]   token_id;
    reg  [POSW-1:0]   pos;
    reg  [IDXW:0]     s_len;
    wire              busy, done;
    wire [VOCAB*16-1:0] logits;
    wire [TOKW-1:0]   argmax;
    wire [MODEL_DIM*16-1:0] h_state;

    wire              em_req;  wire [TOKW-1:0] em_tok; wire [DIMW-1:0] em_idx;
    wire [LAYW-1:0]   db_layer; wire idx_fresh; wire [LAYW-1:0] idx_win;
    wire              gn_req, gn_which; wire [DIMW-1:0] gn_idx;
    wire              aw_req; wire [3:0] aw_sel; wire [A_GRPW-1:0] aw_grp; wire [A_KCW-1:0] aw_k;
    wire              kc_req; wire [IDXW-1:0] kc_idx; wire kc_seq;
    wire              rw_req; wire [R_KW-1:0] rw_k;
    wire              fw_req; wire [1:0] fw_sel; wire [FF_GWD-1:0] fw_grp; wire [FF_KWD-1:0] fw_k;
    wire              fw_shared; wire [EIDXW-1:0] fw_eidx;
    wire              fn_req; wire [DIMW-1:0] fn_idx;
    wire              lw_req; wire [VTW-1:0] lw_vtile; wire [DIMW-1:0] lw_k;

    // combinational pull responses
    reg [15:0] em_val, gn_val, fn_val;
    reg [PE_N*4-1:0]         aw_q;
    reg [16*PE_N*A_NSB-1:0]  aw_d, aw_dmin;  reg [96*PE_N*A_NSB-1:0]  aw_scales;
    reg [KV_LORA*16-1:0] kc_ckv; reg [ROPE*16-1:0] kc_krope; reg kc_valid;
    reg [4*N_EXPERT-1:0] rw_q;
    reg [16*N_EXPERT*R_NSB-1:0] rw_d, rw_dmin; reg [96*N_EXPERT*R_NSB-1:0] rw_scales;
    reg [4*TN-1:0] fw_q, fw_q_up;
    reg [16*TN*FF_NSB_D-1:0] fw_d_g, fw_dmin_g, fw_d_u, fw_dmin_u;
    reg [96*TN*FF_NSB_D-1:0] fw_scales_g, fw_scales_u;
    reg [LM_TN*16-1:0] lw_col;

    integer t, re, ft, fo, cd, lt;
    integer lay, col, kk, cidx, up_col, up_cidx;

    // ---- embedding / gammas / final-norm / LM head ----
    // NOTE: the pull responders use EXPLICIT sensitivity lists (the DUT's request
    //   signals), NOT @*.  The weight ROMs are static (loaded once by $readmemh
    //   before the run), so sensitivity to the request address alone is correct --
    //   and it avoids iverilog building a combinational fan-in over every memory
    //   word (which makes @* over these large arrays compile in minutes).
    always @(em_tok or em_idx) em_val = EMB[em_tok*MODEL_DIM + em_idx];
    always @(fn_idx) fn_val = GF[fn_idx];
    always @(gn_which or db_layer or gn_idx)
        gn_val = gn_which ? G2[db_layer*MODEL_DIM + gn_idx]
                          : G1[db_layer*MODEL_DIM + gn_idx];
    always @(lw_vtile or lw_k) begin
        lw_col = {LM_TN*16{1'b0}};
        for (lt=0; lt<LM_TN; lt=lt+1)
            lw_col[16*lt +: 16] = WLM[(lw_vtile*LM_TN + lt)*MODEL_DIM + lw_k];
    end

    // ---- attention weight (aw_*): per-lane Q4_K column ----
    always @(db_layer or aw_sel or aw_grp or aw_k) begin
        lay = db_layer;
        aw_q      = {PE_N*4{1'b0}};
        aw_d      = {16*PE_N*A_NSB{1'b0}};
        aw_dmin   = {16*PE_N*A_NSB{1'b0}};
        aw_scales = {96*PE_N*A_NSB{1'b0}};
        for (t=0; t<PE_N; t=t+1) begin
            col = aw_grp*PE_N + t;
            // default (also covers SEL_DKV/SEL_KR: the DUT's x*W_dkv / x*W_kr
            // current-token latent is datapath-coverage-only -- never feeds the
            // attended keys (those come from the kc_* cache), so zero is safe and
            // faithful to the golden, which omits W_dkv/W_kr entirely).
            aw_q     [4*t +: 4]  = 4'h0;
            aw_d     [16*t +: 16] = 16'h0;
            aw_dmin  [16*t +: 16] = 16'h0;
            aw_scales[96*t +: 96] = 96'h0;
            case (aw_sel)
            SEL_DQ: if (col < Q_LORA) begin
                        cidx = lay*Q_LORA + col;
                        aw_q[4*t+:4]=`RDC(WDQ_C, cidx*MODEL_DIM + aw_k);
                        aw_d[16*t+:16]=WDQ_D[cidx]; aw_dmin[16*t+:16]=WDQ_DM[cidx]; aw_scales[96*t+:96]=WDQ_SC[cidx];
                    end
            SEL_UQ: if (col < HQK) begin
                        cidx = lay*HQK + col;
                        aw_q[4*t+:4]=`RDC(WUQ_C, cidx*Q_LORA + aw_k);
                        aw_d[16*t+:16]=WUQ_D[cidx]; aw_dmin[16*t+:16]=WUQ_DM[cidx]; aw_scales[96*t+:96]=WUQ_SC[cidx];
                    end
            SEL_UK: if (col < HNOPE) begin
                        cidx = lay*HNOPE + col;
                        aw_q[4*t+:4]=`RDC(WUK_C, cidx*KV_LORA + aw_k);
                        aw_d[16*t+:16]=WUK_D[cidx]; aw_dmin[16*t+:16]=WUK_DM[cidx]; aw_scales[96*t+:96]=WUK_SC[cidx];
                    end
            SEL_UV: if (col < HV) begin
                        cidx = lay*HV + col;
                        aw_q[4*t+:4]=`RDC(WUV_C, cidx*KV_LORA + aw_k);
                        aw_d[16*t+:16]=WUV_D[cidx]; aw_dmin[16*t+:16]=WUV_DM[cidx]; aw_scales[96*t+:96]=WUV_SC[cidx];
                    end
            SEL_O:  if (col < MODEL_DIM) begin
                        cidx = lay*MODEL_DIM + col;
                        aw_q[4*t+:4]=`RDC(WO_C, cidx*HV + aw_k);
                        aw_d[16*t+:16]=WO_D[cidx]; aw_dmin[16*t+:16]=WO_DM[cidx]; aw_scales[96*t+:96]=WO_SC[cidx];
                    end
            default: ;   // SEL_DKV / SEL_KR -> zero (above)
            endcase
        end
    end

    // ---- KV cache read (kc_*) ----
    always @(db_layer or kc_idx) begin
        kc_ckv   = {KV_LORA*16{1'b0}};
        kc_krope = {ROPE*16{1'b0}};
        for (cd=0; cd<KV_LORA; cd=cd+1) kc_ckv[16*cd+:16]   = CKV[(db_layer*S_MAX + kc_idx)*KV_LORA + cd];
        for (cd=0; cd<ROPE;    cd=cd+1) kc_krope[16*cd+:16] = KRP[(db_layer*S_MAX + kc_idx)*ROPE + cd];
    end
    always @(posedge clk) begin
        if (rst) kc_valid <= 1'b0;
        else     kc_valid <= kc_req;
    end

    // ---- router weight (rw_*): column e == expert e ----
    always @(db_layer or rw_k) begin
        rw_q      = {4*N_EXPERT{1'b0}};
        rw_d      = {16*N_EXPERT*R_NSB{1'b0}};
        rw_dmin   = {16*N_EXPERT*R_NSB{1'b0}};
        rw_scales = {96*N_EXPERT*R_NSB{1'b0}};
        for (re=0; re<N_EXPERT; re=re+1) begin
            cidx = db_layer*N_EXPERT + re;
            rw_q[4*re+:4]      = `RDC(WG_C, cidx*MODEL_DIM + rw_k);
            rw_d[16*re+:16]    = WG_D[cidx];
            rw_dmin[16*re+:16] = WG_DM[cidx];
            rw_scales[96*re+:96]= WG_SC[cidx];
        end
    end

    // ---- FFN expert weight (fw_*): dense | shared | routed-expert, gate/up/down ----
    always @(db_layer or fw_sel or fw_grp or fw_k or fw_shared or fw_eidx) begin
        lay = db_layer;
        fw_q     = {4*TN{1'b0}};      fw_q_up  = {4*TN{1'b0}};
        fw_d_g   = {16*TN*FF_NSB_D{1'b0}}; fw_dmin_g = {16*TN*FF_NSB_D{1'b0}};
        fw_d_u   = {16*TN*FF_NSB_D{1'b0}}; fw_dmin_u = {16*TN*FF_NSB_D{1'b0}};
        fw_scales_g = {96*TN*FF_NSB_D{1'b0}}; fw_scales_u = {96*TN*FF_NSB_D{1'b0}};
        for (ft=0; ft<TN; ft=ft+1) begin
            fo = fw_grp*TN + ft;
            if (lay < N_DENSE) begin
                if (fw_sel==2'd2) begin                          // DOWN (Dd): nout=MODEL_DIM
                    if (fo<MODEL_DIM) begin
                        cidx = lay*MODEL_DIM + fo;
                        fw_q[4*ft+:4]=`RDC(DD_C, cidx*INTER_DENSE + fw_k);
                        fw_d_g[16*ft+:16]=DD_D[cidx]; fw_dmin_g[16*ft+:16]=DD_DM[cidx]; fw_scales_g[96*ft+:96]=DD_SC[cidx];
                    end
                end else if (fo<INTER_DENSE) begin               // GATE (Dg)+UP (Du)
                    cidx = lay*INTER_DENSE + fo;
                    fw_q   [4*ft+:4]=`RDC(DG_C, cidx*MODEL_DIM + fw_k);
                    fw_q_up[4*ft+:4]=`RDC(DU_C, cidx*MODEL_DIM + fw_k);
                    fw_d_g[16*ft+:16]=DG_D[cidx]; fw_dmin_g[16*ft+:16]=DG_DM[cidx]; fw_scales_g[96*ft+:96]=DG_SC[cidx];
                    fw_d_u[16*ft+:16]=DU_D[cidx]; fw_dmin_u[16*ft+:16]=DU_DM[cidx]; fw_scales_u[96*ft+:96]=DU_SC[cidx];
                end
            end else if (fw_shared) begin
                if (fw_sel==2'd2) begin                          // shared DOWN (SHd)
                    if (fo<MODEL_DIM) begin
                        cidx = lay*MODEL_DIM + fo;
                        fw_q[4*ft+:4]=`RDC(SHD_C, cidx*INTER_MOE + fw_k);
                        fw_d_g[16*ft+:16]=SHD_D[cidx]; fw_dmin_g[16*ft+:16]=SHD_DM[cidx]; fw_scales_g[96*ft+:96]=SHD_SC[cidx];
                    end
                end else if (fo<INTER_MOE) begin                 // shared GATE (SHg)+UP (SHu)
                    cidx = lay*INTER_MOE + fo;
                    fw_q   [4*ft+:4]=`RDC(SHG_C, cidx*MODEL_DIM + fw_k);
                    fw_q_up[4*ft+:4]=`RDC(SHU_C, cidx*MODEL_DIM + fw_k);
                    fw_d_g[16*ft+:16]=SHG_D[cidx]; fw_dmin_g[16*ft+:16]=SHG_DM[cidx]; fw_scales_g[96*ft+:96]=SHG_SC[cidx];
                    fw_d_u[16*ft+:16]=SHU_D[cidx]; fw_dmin_u[16*ft+:16]=SHU_DM[cidx]; fw_scales_u[96*ft+:96]=SHU_SC[cidx];
                end
            end else begin                                       // routed expert fw_eidx
                if (fw_sel==2'd2) begin                          // expert DOWN (Md)
                    if (fo<MODEL_DIM) begin
                        cidx = (lay*N_EXPERT + fw_eidx)*MODEL_DIM + fo;
                        fw_q[4*ft+:4]=`RDC(MD_C, cidx*INTER_MOE + fw_k);
                        fw_d_g[16*ft+:16]=MD_D[cidx]; fw_dmin_g[16*ft+:16]=MD_DM[cidx]; fw_scales_g[96*ft+:96]=MD_SC[cidx];
                    end
                end else if (fo<INTER_MOE) begin                 // expert GATE (Mg)+UP (Mu)
                    cidx = (lay*N_EXPERT + fw_eidx)*INTER_MOE + fo;
                    fw_q   [4*ft+:4]=`RDC(MG_C, cidx*MODEL_DIM + fw_k);
                    fw_q_up[4*ft+:4]=`RDC(MU_C, cidx*MODEL_DIM + fw_k);
                    fw_d_g[16*ft+:16]=MG_D[cidx]; fw_dmin_g[16*ft+:16]=MG_DM[cidx]; fw_scales_g[96*ft+:96]=MG_SC[cidx];
                    fw_d_u[16*ft+:16]=MU_D[cidx]; fw_dmin_u[16*ft+:16]=MU_DM[cidx]; fw_scales_u[96*ft+:96]=MU_SC[cidx];
                end
            end
        end
    end

    //========================================================================
    // DUT
    //========================================================================
    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .ACT_HW(ACT_HW), .GU_CONC(GU_CONC)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .token_id(token_id), .pos(pos), .pos_vec({POSW{1'b0}}),
        .s_len_vec({(IDXW+1){1'b0}}), .seq_vec(1'b0), .s_len(s_len),
        .logits(logits), .argmax(argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_seq(kc_seq),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .h_state(h_state)
    );

    // soak unused request/status lines
    wire _unused = &{1'b0, busy, em_req, idx_fresh, idx_win, gn_req, aw_req,
        kc_req, kc_seq, rw_req, fw_req, fn_req, lw_req};

    //========================================================================
    // drive + compare
    //========================================================================
    integer tests, ci, v, d, mm;
    reg [TOKW-1:0] c_tok; reg [POSW-1:0] c_pos; reg [IDXW:0] c_slen;
    reg [15:0] got, exp_;

    task run_one_forward; begin
        @(negedge clk);
        start = 1'b1; @(negedge clk); start = 1'b0;
        wait (done === 1'b1);
        @(negedge clk);
    end endtask

    // safety timeout
    initial begin
        #2000000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end

    initial begin
        tests = 0; mm = 0;
        start = 1'b0; token_id = {TOKW{1'b0}}; pos = {POSW{1'b0}};
        s_len = {(IDXW+1){1'b0}};

        load_weights;
        $readmemh({DIR,"/ncase.hex"}, NCASE_M);
        $readmemh({DIR,"/stim.hex"},  STIM);
        NCASE = NCASE_M[0];

        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        for (ci=0; ci<NCASE; ci=ci+1) begin
            c_tok  = STIM[3*ci+0][TOKW-1:0];
            c_pos  = STIM[3*ci+1][POSW-1:0];
            c_slen = STIM[3*ci+2][IDXW:0];
            // load this case's golden
            $readmemh($sformatf("%s/logits_%0d.hex", DIR, ci), GLOGIT);
            $readmemh($sformatf("%s/xn_%0d.hex",     DIR, ci), GXN);
            $readmemh($sformatf("%s/argmax_%0d.hex", DIR, ci), GARG_M);
            GARG = GARG_M[0][TOKW-1:0];

            token_id = c_tok; pos = c_pos; s_len = c_slen;
            run_one_forward();

            // (1) logits bit-exact
            for (v=0; v<VOCAB; v=v+1) begin
                got = logits[16*v +: 16];
                exp_ = GLOGIT[v];
                if (^got === 1'bx) begin
                    $display("FAIL case %0d: logit[%0d] is X", ci, v); mm=mm+1;
                end else if (got !== exp_) begin
                    if (mm < 12)
                        $display("FAIL case %0d: logit[%0d] dut=%04x golden=%04x", ci, v, got, exp_);
                    mm = mm + 1;
                end else tests = tests + 1;
            end
            // (2) argmax
            if (^argmax === 1'bx) begin
                $display("FAIL case %0d: argmax X", ci); mm=mm+1;
            end else if (argmax !== GARG) begin
                $display("FAIL case %0d: argmax dut=%0d golden=%0d", ci, argmax, GARG); mm=mm+1;
            end else tests = tests + 1;
            // (3) h_state (final RMSNorm == golden xn) bit-exact
            for (d=0; d<MODEL_DIM; d=d+1) begin
                got = h_state[16*d +: 16];
                exp_ = GXN[d];
                if (^got === 1'bx) begin
                    $display("FAIL case %0d: h_state[%0d] X", ci, d); mm=mm+1;
                end else if (got !== exp_) begin
                    if (mm < 12)
                        $display("FAIL case %0d: h_state[%0d] dut=%04x golden=%04x", ci, d, got, exp_);
                    mm = mm + 1;
                end else tests = tests + 1;
            end
            $display("case %0d: token=%0d pos=%0d s_len=%0d -> argmax=%0d (golden %0d) %s",
                     ci, c_tok, c_pos, c_slen, argmax, GARG,
                     (argmax===GARG)?"MATCH":"MISMATCH");
        end

        if (mm != 0) begin
            $display("ASSEMBLED-FORWARD MISMATCH: %0d bit differences vs golden", mm);
            $fatal(1, "glm_model_q4k diverges from assembled golden");
        end
        $display("----------------------------------------------------------------");
        $display("ASSEMBLED glm_model_q4k == numpy golden (BIT-EXACT): logits+argmax+h_state");
        $display("  cases                 = %0d", NCASE);
        $display("  slice                 = MODEL_DIM=%0d L=%0d N_DENSE=%0d VOCAB=%0d", MODEL_DIM, L, N_DENSE, VOCAB);
        $display("  MLA softmax scale     = 1/sqrt(QK_DIM=%0d) applied (Phase-1 fix)", QK_DIM);
        $display("----------------------------------------------------------------");
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end
endmodule
