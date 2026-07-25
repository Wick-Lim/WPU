`timescale 1ns/1ps
//============================================================================
// glm_model_q4k_pem_tb.v -- BATCHED (PE_M>1) EXACTNESS TB for glm_model_q4k.
//                     (docs/SCALE_FUNCTIONAL.md OPEN item 3, `make batched-q4k`)
//----------------------------------------------------------------------------
// WHAT THIS PROVES  (the Q4_K batch-axis gap: until now the only PE_M>1 proof
//   on the Q4_K track was spec==greedy self-consistency -- no direct golden).
//
//   Two instances of the REAL product top `glm_model_q4k` run the SAME weight
//   set (the assembled-numpy-golden vectors of tools/glm_model_q4k_tb_gen.py,
//   shared with `make model-q4k`):
//     * ref : PE_M=1  -- the configuration proven BIT-EXACT vs the assembled
//             numpy golden by test/glm_model_q4k_full_tb.v (`make model-q4k`);
//     * bat : PE_M=B  -- B query tokens decoded in lockstep against the shared
//             sequence/KV window, ONE weight fetch shared by all B rows.
//
//   For each scenario (pos, s_len from a golden case; B distinct tokens):
//     (1) ref forward on (tok_r, pos, s_len) for every row r      (B runs)
//     (2) bat forward on {tok_0..tok_(B-1)} at the same (pos, s_len)
//     (3) row r of bat === run r of ref: logits[VOCAB], argmax, h_state
//         [MODEL_DIM] -- ALL BIT-EXACT (X-aware, uint16 pattern compare)
//     (4) ANCHOR: row 0's (tok, pos, s_len) IS golden case ci, so ref run 0
//         AND bat row 0 are ALSO compared BIT-EXACT vs the numpy golden hex
//         (logits_ci/xn_ci/argmax_ci) -- chaining the batch proof to the
//         assembled numpy reference, not just DUT-vs-DUT.
//
//   Weight ROMs + pull responders are the ones of glm_model_q4k_full_tb.v; the
//   two DUTs' request buses are MUXED into the single responder set (only one
//   DUT runs at a time), and the response buses (PE_M-independent widths) feed
//   both.  Scope: PE_M widening for a SHARED sequence (PER_ROW_*=0) -- the
//   per-row-KV multi-seq / decode-loop TBs remain FP8-only (tag fp8-verified-baseline).
//============================================================================
module glm_model_q4k_pem_tb;
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

    // ---- batch rows for the widened DUT ----
`ifndef TB_B
    `define TB_B 2
`endif
    localparam integer B    = `TB_B;
    localparam integer SEQW = (B <= 1) ? 1 : $clog2(B);

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
    // DUT port wires -- TWO DUTs (ref PE_M=1, bat PE_M=B).  The responders see
    // ONE muxed request bus (run_bat selects); responses feed both DUTs.
    //========================================================================
    reg               r_start, b_start;
    reg               run_bat;                 // 1 = the batched DUT is running
    reg  [TOKW-1:0]   r_token_id;
    reg  [B*TOKW-1:0] b_token_id;
    reg  [POSW-1:0]   pos;
    reg  [IDXW:0]     s_len;
    wire              r_busy, r_done, b_busy, b_done;
    wire [VOCAB*16-1:0]      r_logits;
    wire [TOKW-1:0]          r_argmax;
    wire [MODEL_DIM*16-1:0]  r_h_state;
    wire [B*VOCAB*16-1:0]     b_logits;
    wire [B*TOKW-1:0]         b_argmax;
    wire [B*MODEL_DIM*16-1:0] b_h_state;

    // per-DUT request buses
    wire r_em_req;  wire [TOKW-1:0] r_em_tok; wire [DIMW-1:0] r_em_idx;
    wire [LAYW-1:0] r_db_layer; wire r_idx_fresh; wire [LAYW-1:0] r_idx_win;
    wire r_gn_req, r_gn_which; wire [DIMW-1:0] r_gn_idx;
    wire r_aw_req; wire [3:0] r_aw_sel; wire [A_GRPW-1:0] r_aw_grp; wire [A_KCW-1:0] r_aw_k;
    wire r_kc_req; wire [IDXW-1:0] r_kc_idx; wire r_kc_seq;
    wire r_rw_req; wire [R_KW-1:0] r_rw_k;
    wire r_fw_req; wire [1:0] r_fw_sel; wire [FF_GWD-1:0] r_fw_grp; wire [FF_KWD-1:0] r_fw_k;
    wire r_fw_shared; wire [EIDXW-1:0] r_fw_eidx;
    wire r_fn_req; wire [DIMW-1:0] r_fn_idx;
    wire r_lw_req; wire [VTW-1:0] r_lw_vtile; wire [DIMW-1:0] r_lw_k;

    wire b_em_req;  wire [TOKW-1:0] b_em_tok; wire [DIMW-1:0] b_em_idx;
    wire [LAYW-1:0] b_db_layer; wire b_idx_fresh; wire [LAYW-1:0] b_idx_win;
    wire b_gn_req, b_gn_which; wire [DIMW-1:0] b_gn_idx;
    wire b_aw_req; wire [3:0] b_aw_sel; wire [A_GRPW-1:0] b_aw_grp; wire [A_KCW-1:0] b_aw_k;
    wire b_kc_req; wire [IDXW-1:0] b_kc_idx; wire [SEQW-1:0] b_kc_seq;
    wire b_rw_req; wire [R_KW-1:0] b_rw_k;
    wire b_fw_req; wire [1:0] b_fw_sel; wire [FF_GWD-1:0] b_fw_grp; wire [FF_KWD-1:0] b_fw_k;
    wire b_fw_shared; wire [EIDXW-1:0] b_fw_eidx;
    wire b_fn_req; wire [DIMW-1:0] b_fn_idx;
    wire b_lw_req; wire [VTW-1:0] b_lw_vtile; wire [DIMW-1:0] b_lw_k;

    // muxed request bus (the names the copied responders are sensitive to)
    wire [TOKW-1:0]   em_tok   = run_bat ? b_em_tok   : r_em_tok;
    wire [DIMW-1:0]   em_idx   = run_bat ? b_em_idx   : r_em_idx;
    wire [LAYW-1:0]   db_layer = run_bat ? b_db_layer : r_db_layer;
    wire              gn_which = run_bat ? b_gn_which : r_gn_which;
    wire [DIMW-1:0]   gn_idx   = run_bat ? b_gn_idx   : r_gn_idx;
    wire [3:0]        aw_sel   = run_bat ? b_aw_sel   : r_aw_sel;
    wire [A_GRPW-1:0] aw_grp   = run_bat ? b_aw_grp   : r_aw_grp;
    wire [A_KCW-1:0]  aw_k     = run_bat ? b_aw_k     : r_aw_k;
    wire              kc_req   = run_bat ? b_kc_req   : r_kc_req;
    wire [IDXW-1:0]   kc_idx   = run_bat ? b_kc_idx   : r_kc_idx;
    wire [R_KW-1:0]   rw_k     = run_bat ? b_rw_k     : r_rw_k;
    wire [1:0]        fw_sel   = run_bat ? b_fw_sel   : r_fw_sel;
    wire [FF_GWD-1:0] fw_grp   = run_bat ? b_fw_grp   : r_fw_grp;
    wire [FF_KWD-1:0] fw_k     = run_bat ? b_fw_k     : r_fw_k;
    wire              fw_shared= run_bat ? b_fw_shared: r_fw_shared;
    wire [EIDXW-1:0]  fw_eidx  = run_bat ? b_fw_eidx  : r_fw_eidx;
    wire [DIMW-1:0]   fn_idx   = run_bat ? b_fn_idx   : r_fn_idx;
    wire [VTW-1:0]    lw_vtile = run_bat ? b_lw_vtile : r_lw_vtile;
    wire [DIMW-1:0]   lw_k     = run_bat ? b_lw_k     : r_lw_k;

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
    // DUTs: ref (PE_M=1, the model-q4k-proven config) and bat (PE_M=B)
    //========================================================================
    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) dut_ref (
        .clk(clk), .rst(rst), .start(r_start), .busy(r_busy), .done(r_done),
        .token_id(r_token_id), .pos(pos), .pos_vec({POSW{1'b0}}),
        .s_len_vec({(IDXW+1){1'b0}}), .seq_vec(1'b0), .s_len(s_len),
        .logits(r_logits), .argmax(r_argmax),
        .em_req(r_em_req), .em_tok(r_em_tok), .em_idx(r_em_idx), .em_val(em_val),
        .db_layer(r_db_layer), .idx_fresh(r_idx_fresh), .idx_win(r_idx_win),
        .gn_req(r_gn_req), .gn_which(r_gn_which), .gn_idx(r_gn_idx), .gn_val(gn_val),
        .aw_req(r_aw_req), .aw_sel(r_aw_sel), .aw_grp(r_aw_grp), .aw_k(r_aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .kc_req(r_kc_req), .kc_idx(r_kc_idx), .kc_seq(r_kc_seq),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_valid(kc_valid),
        .rw_req(r_rw_req), .rw_k(r_rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(r_fw_req), .fw_sel(r_fw_sel), .fw_grp(r_fw_grp), .fw_k(r_fw_k),
        .fw_shared(r_fw_shared), .fw_eidx(r_fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(r_fn_req), .fn_idx(r_fn_idx), .fn_val(fn_val),
        .lw_req(r_lw_req), .lw_vtile(r_lw_vtile), .lw_k(r_lw_k), .lw_col(lw_col),
        .h_state(r_h_state)
    );

    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(B)
    ) dut_bat (
        .clk(clk), .rst(rst), .start(b_start), .busy(b_busy), .done(b_done),
        .token_id(b_token_id), .pos(pos), .pos_vec({B*POSW{1'b0}}),
        .s_len_vec({B*(IDXW+1){1'b0}}), .seq_vec({B*SEQW{1'b0}}), .s_len(s_len),
        .logits(b_logits), .argmax(b_argmax),
        .em_req(b_em_req), .em_tok(b_em_tok), .em_idx(b_em_idx), .em_val(em_val),
        .db_layer(b_db_layer), .idx_fresh(b_idx_fresh), .idx_win(b_idx_win),
        .gn_req(b_gn_req), .gn_which(b_gn_which), .gn_idx(b_gn_idx), .gn_val(gn_val),
        .aw_req(b_aw_req), .aw_sel(b_aw_sel), .aw_grp(b_aw_grp), .aw_k(b_aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .kc_req(b_kc_req), .kc_idx(b_kc_idx), .kc_seq(b_kc_seq),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_valid(kc_valid),
        .rw_req(b_rw_req), .rw_k(b_rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(b_fw_req), .fw_sel(b_fw_sel), .fw_grp(b_fw_grp), .fw_k(b_fw_k),
        .fw_shared(b_fw_shared), .fw_eidx(b_fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(b_fn_req), .fn_idx(b_fn_idx), .fn_val(fn_val),
        .lw_req(b_lw_req), .lw_vtile(b_lw_vtile), .lw_k(b_lw_k), .lw_col(lw_col),
        .h_state(b_h_state)
    );

    // soak unused request/status lines
    wire _unused = &{1'b0, r_busy, b_busy, r_em_req, b_em_req, r_idx_fresh,
        b_idx_fresh, r_idx_win, b_idx_win, r_gn_req, b_gn_req, r_aw_req, b_aw_req,
        r_kc_seq, b_kc_seq, r_rw_req, b_rw_req, r_fw_req, b_fw_req, r_fn_req,
        b_fn_req, r_lw_req, b_lw_req};

    //========================================================================
    // drive + compare
    //========================================================================
    integer tests, ci, v, d, mm, rrow;
    reg [TOKW-1:0] c_tok; reg [POSW-1:0] c_pos; reg [IDXW:0] c_slen;
    reg [15:0] got, exp_;

    // captured PE_M=1 reference outputs, one set per batch row
    reg [VOCAB*16-1:0]     REFL [0:B-1];
    reg [TOKW-1:0]         REFA [0:B-1];
    reg [MODEL_DIM*16-1:0] REFH [0:B-1];
    reg [B*TOKW-1:0]       btoks;

    task run_ref_forward; begin
        @(negedge clk);
        run_bat = 1'b0;
        r_start = 1'b1; @(negedge clk); r_start = 1'b0;
        wait (r_done === 1'b1);
        @(negedge clk);
    end endtask

    task run_bat_forward; begin
        @(negedge clk);
        run_bat = 1'b1;
        b_start = 1'b1; @(negedge clk); b_start = 1'b0;
        wait (b_done === 1'b1);
        @(negedge clk);
        run_bat = 1'b0;
    end endtask

    // one X-aware bit-exact bf16 compare (bumps tests/mm like model-q4k's gate)
    task chk16(input [15:0] g, input [15:0] e, input [255:0] label,
               input integer idx); begin
        if (^g === 1'bx) begin
            $display("FAIL scen %0d [%0s %0d]: X output", ci, label, idx); mm = mm + 1;
        end else if (g !== e) begin
            if (mm < 12)
                $display("FAIL scen %0d [%0s %0d]: dut=%04x exp=%04x", ci, label, idx, g, e);
            mm = mm + 1;
        end else tests = tests + 1;
    end endtask

    // safety timeout
    initial begin
        #2000000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end

    initial begin
        tests = 0; mm = 0;
        r_start = 1'b0; b_start = 1'b0; run_bat = 1'b0;
        r_token_id = {TOKW{1'b0}}; b_token_id = {B*TOKW{1'b0}};
        pos = {POSW{1'b0}}; s_len = {(IDXW+1){1'b0}};

        load_weights;
        $readmemh({DIR,"/ncase.hex"}, NCASE_M);
        $readmemh({DIR,"/stim.hex"},  STIM);
        NCASE = NCASE_M[0];

        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        for (ci = 0; ci < NCASE; ci = ci + 1) begin
            c_tok  = STIM[3*ci+0][TOKW-1:0];
            c_pos  = STIM[3*ci+1][POSW-1:0];
            c_slen = STIM[3*ci+2][IDXW:0];
            // numpy-golden anchor for row 0 (whose (tok,pos,s_len) IS case ci)
            $readmemh($sformatf("%s/logits_%0d.hex", DIR, ci), GLOGIT);
            $readmemh($sformatf("%s/xn_%0d.hex",     DIR, ci), GXN);
            $readmemh($sformatf("%s/argmax_%0d.hex", DIR, ci), GARG_M);
            GARG = GARG_M[0][TOKW-1:0];

            // batch tokens: row 0 = case ci's token; row r = case (ci+r)%NCASE's
            // token (distinct embeddings diverge per row through every layer)
            for (rrow = 0; rrow < B; rrow = rrow + 1)
                btoks[rrow*TOKW +: TOKW] = STIM[3*((ci+rrow)%NCASE)+0][TOKW-1:0];
            pos = c_pos; s_len = c_slen;

            // (1) PE_M=1 reference forward per row
            for (rrow = 0; rrow < B; rrow = rrow + 1) begin
                r_token_id = btoks[rrow*TOKW +: TOKW];
                run_ref_forward();
                REFL[rrow] = r_logits; REFA[rrow] = r_argmax; REFH[rrow] = r_h_state;
            end

            // anchor: ref row 0 === numpy golden (the model-q4k contract)
            for (v = 0; v < VOCAB; v = v + 1)
                chk16(REFL[0][16*v +: 16], GLOGIT[v], "ref0 logit", v);
            if (^REFA[0] === 1'bx || REFA[0] !== GARG) begin
                $display("FAIL scen %0d: ref0 argmax dut=%0d golden=%0d", ci, REFA[0], GARG);
                mm = mm + 1;
            end else tests = tests + 1;
            for (d = 0; d < MODEL_DIM; d = d + 1)
                chk16(REFH[0][16*d +: 16], GXN[d], "ref0 h_state", d);

            // (2) batched PE_M=B forward, same weights / pos / s_len
            b_token_id = btoks;
            run_bat_forward();

            // (3) row r of bat === ref run r: logits + argmax + h_state BIT-EXACT
            for (rrow = 0; rrow < B; rrow = rrow + 1) begin
                for (v = 0; v < VOCAB; v = v + 1)
                    chk16(b_logits[16*(VOCAB*rrow + v) +: 16], REFL[rrow][16*v +: 16],
                          "bat logit row", rrow*VOCAB + v);
                if (^b_argmax[TOKW*rrow +: TOKW] === 1'bx ||
                    b_argmax[TOKW*rrow +: TOKW] !== REFA[rrow]) begin
                    $display("FAIL scen %0d: bat argmax row %0d dut=%0d ref=%0d",
                             ci, rrow, b_argmax[TOKW*rrow +: TOKW], REFA[rrow]);
                    mm = mm + 1;
                end else tests = tests + 1;
                for (d = 0; d < MODEL_DIM; d = d + 1)
                    chk16(b_h_state[16*(MODEL_DIM*rrow + d) +: 16], REFH[rrow][16*d +: 16],
                          "bat h_state row", rrow*MODEL_DIM + d);
            end

            // (4) anchor: bat row 0 === numpy golden directly
            for (v = 0; v < VOCAB; v = v + 1)
                chk16(b_logits[16*v +: 16], GLOGIT[v], "bat0 logit vs golden", v);

            $display("scenario %0d: pos=%0d s_len=%0d tok0=%0d -> B=%0d rows %s",
                     ci, c_pos, c_slen, c_tok, B, (mm == 0) ? "MATCH" : "MISMATCH");
        end

        if (mm != 0) begin
            $display("BATCHED-FORWARD MISMATCH: %0d bit differences", mm);
            $fatal(1, "glm_model_q4k PE_M=B diverges from per-row PE_M=1 / golden");
        end
        $display("----------------------------------------------------------------");
        $display("BATCHED glm_model_q4k PE_M=%0d == per-row PE_M=1 (BIT-EXACT): logits+argmax+h_state", B);
        $display("  scenarios             = %0d (row 0 also anchored to the numpy golden)", NCASE);
        $display("  slice                 = MODEL_DIM=%0d L=%0d N_DENSE=%0d VOCAB=%0d", MODEL_DIM, L, N_DENSE, VOCAB);
        $display("----------------------------------------------------------------");
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end
endmodule
