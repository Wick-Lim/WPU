`timescale 1ns/1ps
//============================================================================
// glm_q4k_loopback_rest_tb.v -- C8 LOOPBACK_REST=1 BIT-EXACTNESS proof for
//   glm_q4k_system.  The EXACT mirror of glm_q4k_loopback_tb.v / _fw_tb.v, applied
//   to the THREE remaining die weight-input families that were stub-only: rw (MoE
//   router Q4_K CODES), lw (LM-head bf16 columns), gn (a single bf16 gain/norm).
//----------------------------------------------------------------------------
// WHAT IT PROVES
//   The perf TB (test/glm_q4k_system_perf_tb.v) already proves
//       glm_q4k_system(LOOPBACK_REST=0) committed next_tok == standalone
//       glm_model_q4k fed by the SAME deterministic weight functions.
//   This TB proves the SAME equality with LOOPBACK_REST=1 -- i.e. when the die's
//   rw_q / lw_col / gn_val inputs are physically routed OUT of the die as banked
//   ddr5_xbar reads (TAG_LBRW/LBLW/LBGN) and back IN through the fabric, instead
//   of being served same-cycle by the stub.  If the three round-trips are faithful
//   the committed token stream is BIT-EXACT to the non-loopback (== reference)
//   stream: clock-gating a synchronous die is transparent, and the fed-back bytes
//   equal the stub bytes for the same key.
//
// HOW THE LOOPBACK-REST RESPONDER WORKS (the ONLY change vs the perf TB harness)
//   At LOOPBACK_REST=1 each die pull encodes its exact key into a DDR read address:
//       rw:  cur_rw_addr[0+:R_KW]=rw_k;  [RWA_LY_LO+:LAYW]=db_layer; [24+:8]=8'hC7
//       lw:  cur_lw_addr[0+:DIMW]=lw_k;  [LWA_VT_LO+:VTW]=lw_vtile;  [24+:8]=8'hD8
//       gn:  cur_gn_addr[0+:DIMW]=gn_idx;[GNA_WH_LO+:1]=gn_which;[GNA_LY_LO+:LAYW]=db_layer;
//       (offsets packed end-to-end from the widths -- mirror g_lbrest's localparams)
//            [24+:8]=8'hE9
//   ddr5_xbar carries the FULL address to the per-channel memory unchanged and
//   returns the response paired with its tag.  This TB's DDR5 memory model, on an
//   outstanding read carrying one of the three markers, packs into the response's
//   low bits EXACTLY the bytes the stub presents for that key at LOOPBACK_REST=0:
//       8'hC7 -> low 4*N_EXPERT bits, lane e = f_rwq(layer, e, rw_k)          (rw_q)
//       8'hD8 -> low LM_TN*16 bits,   lane t = gen_bf16((vtile*LM_TN+t)*MODEL_DIM
//                                                        + lw_k + 7603)       (lw_col)
//       8'hE9 -> low 16 bits, gen_bf16(layer*1024 + which*512 + idx + 7411)   (gn_val)
//   Every OTHER read keeps the perf TB's banked-read behaviour (gen_beat).  Only
//   loopback reads carry a nonzero addr[31:24] marker (TAG_LOAD/SLOT/HOT/EFILL all
//   leave the top address bits zero, and aw/fw use 8'hA5/8'hB6), so the per-family
//   decode is unambiguous.
//
//   KEY-FIELD DEPENDENCIES (encode ALL fields each value depends on):
//     f_rwq depends on {db_layer, e(=lane, positional), rw_k}  -> encode db_layer,rw_k
//     lw_col lane depends on {lw_vtile, t(=lane, positional), lw_k, MODEL_DIM}
//                                                             -> encode lw_vtile,lw_k
//     gn_val depends on {db_layer, gn_which, gn_idx}   -> encode all three
//
// TWO BINDINGS (the token binding alone is NOT sufficient for rw/gn):
//   (1) COMMITTED-TOKEN binding: the LOOPBACK_REST=1 committed next_tok stream is
//       BIT-EXACT to the standalone glm_model_q4k reference over the 4-token decode.
//   (2) DIRECT PER-BEAT DIE-INPUT BYTE binding (MANDATORY): rw's Q4_K router codes
//       (the top-k absorbs a full one-code step) and gn's per-dim bf16 gains (normed
//       away) were MEASURED output-INSENSITIVE, so binding (1) does NOT by itself
//       prove the die consumed the CORRECT rw/gn bytes off the fabric.  So on every
//       beat the die pulls a family under LOOPBACK_REST=1, this TB reaches INTO the
//       DUT by hierarchical reference and asserts the die's PRESENTED input --
//       dut.die_rw_q / dut.die_lw_col / dut.die_gn_val (module-scope wires) -- equals
//       the SAME deterministic value function that feeds the stub/reference for the
//       LIVE key, gated by the in-flight-beat-ready flags in the LOOPBACK_REST=1
//       generate branch (dut.g_lbrest.lb{rw,lw,gn}_have).  This is byte-for-byte proof
//       the round-trip delivers the right bytes, independent of output sensitivity.
//       It holds for ALL THREE families and FAILS the gate on any mismatch.
//
// SOUNDNESS (-DLBRESTINJECT): ONE fed-back family's bytes are corrupted on EVERY beat
//   -- the gn value (a single bf16 gain/norm) has its mantissa LSB flipped by the DDR
//   memory model before it is returned.  gn was MEASURED output-insensitive (a single-
//   ULP gain change is normalised away over this short decode), so this is EXACTLY the
//   corruption the committed-token binding alone can miss -- and it is what makes the
//   DIRECT binding load-bearing: die_gn_val no longer equals the expected stub value on
//   any gn beat, so the direct gn byte binding trips and the gate FAILS (the pass banner
//   is only printed when errors==0).  Corrupting the output-INSENSITIVE family (not lw,
//   which also moves the argmax) is the sharper test: it isolates the direct binding as
//   the thing that catches it.  The clean build MUST pass; the injected build MUST NOT
//   print the banner.  (The clean run separately $fatal's if ANY of rw/lw/gn never
//   round-tripped, and if any family's direct binding never sampled -- no vacuous pass.)
//
// STYLE: sync active-high reset; self-checking; on success prints exactly
//   "ALL %0d TESTS PASSED ...".  Mirrors the perf / aw / fw loopback TB idioms.
//============================================================================
module glm_q4k_loopback_rest_tb;

    // ---- small-but-faithful slice (the perf TB geometry, PE_N=2) ----
    parameter integer FLASH_LAT_CFG    = 8;
    parameter integer DDR_NCH_CFG      = 4;
    parameter integer CACHE_SLOTS_CFG  = 4;
    parameter integer N_EXPERT_CFG     = 4;
    parameter integer L_CFG            = 4;
    // EXPERT_STALL / RESIDENT held OFF: at LOOPBACK=1 the §9 loopback branch owns
    // die_clk (the EXPERT_STALL clock-gate lives only in the LOOPBACK==0 branch),
    // and RESIDENT=0 keeps expert refills on the single Flash channel.  Neither
    // is needed to exercise the loopback path.
    parameter integer EXPERT_STALL_CFG = 0;
    parameter integer RESIDENT_CFG     = 0;
    parameter integer DSA_REAL_IDX_CFG = 0;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice params (mirror the perf TB) =================
    parameter integer MODEL_DIM = 16;
    localparam integer L          = L_CFG;
    parameter integer N_DENSE   = 2;
    parameter integer VOCAB     = 16;
    parameter integer H_HEADS   = 2;
    parameter integer NOPE      = 4;
    parameter integer ROPE      = 4;
    parameter integer V_DIM     = 4;
    parameter integer Q_LORA    = 8;
    parameter integer KV_LORA   = 8;
    parameter integer S_MAX     = 4;
    parameter integer TOPK_ATTN = 4;
    localparam integer THETA      = 8000000;
    parameter  integer PE_N        = 2;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = N_EXPERT_CFG;
    parameter integer TOPK      = 2;
    parameter integer INTER_MOE = 16;
    parameter integer INTER_DENSE= 32;
    localparam [31:0]  RSCALE     = 32'h40200000;
    parameter integer TN          = 4;
    localparam integer BLK        = 128;
    parameter  integer LM_TN       = 4;
    localparam integer CACHE_SLOTS = CACHE_SLOTS_CFG;
    localparam integer FLASH_LAT   = FLASH_LAT_CFG;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;
    localparam integer EFIFO_DEPTH = 16;
    localparam integer DDR_NCH     = DDR_NCH_CFG;
    localparam integer DDR_ADDR_W  = 32;
    localparam integer DDR_DATA_W  = 256;
    localparam integer DDR_TAG_W   = 8;
    localparam integer DDR_ROW_LAT = 10;
    localparam integer DDR_RESP_QD = 4;
    localparam integer WL_KMAX     = 256;
    localparam integer WL_ADDR_W   = 24;
    localparam integer LOADER_KLEN = MODEL_DIM;

    // ---- derived (mirror glm_q4k_system / glm_model_q4k) ----
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer IDXW   = (S_MAX<=1)?1:$clog2(S_MAX);
    localparam integer HQK    = H_HEADS*QK_DIM;
    localparam integer HNOPE  = H_HEADS*NOPE;
    localparam integer HV     = H_HEADS*V_DIM;
    localparam integer EIDXW  = (N_EXPERT<=1)?1:$clog2(N_EXPERT);
    localparam integer A_KMAX = (MODEL_DIM>Q_LORA)?
                       ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV):((KV_LORA>HV)?KV_LORA:HV))
                     :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV):((KV_LORA>HV)?KV_LORA:HV));
    localparam integer A_OMAX = (HQK>MODEL_DIM)?
                       ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                     :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV):((HNOPE>HV)?HNOPE:HV));
    localparam integer A_NGMAX = (A_OMAX+PE_N-1)/PE_N;
    localparam integer A_GRPW  = (A_NGMAX<=1)?1:$clog2(A_NGMAX);
    localparam integer A_KCW   = (A_KMAX <=1)?1:$clog2(A_KMAX);
    localparam integer FF_KMAX_D = (INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM;
    localparam integer FF_KMAX_M = (INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM;
    localparam integer FF_GWD = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN+1);
    localparam integer FF_KWD = $clog2(FF_KMAX_D+1);
    localparam integer R_KW   = $clog2(FF_KMAX_M+1);
    localparam integer A_NSB    = (A_KMAX   +255)/256;
    localparam integer FF_NSB_D = (FF_KMAX_D+255)/256;
    localparam integer R_NSB    = (FF_KMAX_M+255)/256;
    localparam integer LAYW   = (L<=1)?1:$clog2(L);
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    // rest loopback key offsets -- MUST mirror glm_q4k_system g_lbrest (packed end-to-end)
    localparam integer RWA_LY_LO = R_KW;
    localparam integer LWA_VT_LO = DIMW;
    localparam integer GNA_WH_LO = DIMW;
    localparam integer GNA_LY_LO = GNA_WH_LO + 1;
    localparam integer ROW_BITS = (KV_LORA+ROPE)*16;
    localparam integer KVPOSW   = (KV_CTX<=1)?1:$clog2(KV_CTX);
    localparam integer CSLOTW   = (CACHE_SLOTS<=1)?1:$clog2(CACHE_SLOTS);
    localparam integer WL_PE_N  = PE_N;
    localparam integer WL_DATA_W= 256;
    localparam integer CH_SEL_W = (DDR_NCH<=1)?1:$clog2(DDR_NCH);

    // ================= deterministic stimulus generators ========================
    function automatic integer f_h; input integer seed; begin
        f_h = (seed*2654435761)^(seed<<13)^(seed*40503);
    end endfunction
    function automatic [15:0] gen_bf16; input integer seed;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = f_h(seed);
        s = h[3];
        e = 8'd124 + {6'b0,h[5:4]};
        m = h[12:6];
        gen_bf16 = {s,e,m};
    end endfunction
    function automatic [15:0] gen_fp16; input integer seed;
        reg [4:0] e; reg [9:0] m; integer h; begin
        h = f_h(seed);
        e = 5'd12 + {4'b0,h[4]};
        m = h[14:5];
        gen_fp16 = {1'b0,e,m};
    end endfunction
    function automatic [3:0] gen_q4; input integer seed; integer h; begin
        h = f_h(seed);
        gen_q4 = h[11:8];
    end endfunction
    function automatic [31:0] gen_s32; input integer seed; begin
        gen_s32 = f_h(seed*97 + 5);
    end endfunction

    // ---- per-family responder value functions (shared DUT/ref) ----
    function automatic [3:0] f_awq; input integer ly; input integer sel;
        input integer fo; input integer kk; begin
        f_awq = gen_q4(ly*7919 + sel*104729 + fo*611953 + kk*13 + 101);
    end endfunction
    function automatic [15:0] f_awd; input integer ly; input integer sel;
        input integer fo; begin
        f_awd = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 211);
    end endfunction
    function automatic [15:0] f_awdm; input integer ly; input integer sel;
        input integer fo; begin
        f_awdm = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 307);
    end endfunction
    function automatic [3:0] f_rwq; input integer ly; input integer e;
        input integer kk; begin
        f_rwq = gen_q4(ly*7919 + e*350377 + kk*13 + 401);
    end endfunction
    function automatic [3:0] f_fwq; input integer ly; input integer sel;
        input integer shr; input integer eidx; input integer fo; input integer kk;
        begin
        f_fwq = gen_q4(ly*7919 + sel*104729 + shr*15485863 + eidx*350377
                       + fo*611953 + kk*13 + 503);
    end endfunction

    integer errors;

    //========================================================================
    // DUT (the Q4_K system) host I/O -- LOOPBACK=1
    //========================================================================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    wire                      busy, done;
    wire [TOKW-1:0]           next_tok;
    wire                      tok_valid;
    wire [VOCAB*16-1:0]       logits;
    wire                      em_req;  wire [TOKW-1:0] em_tok;  wire [DIMW-1:0] em_idx;
    reg  [15:0]               em_val;
    wire [LAYW-1:0]           db_layer;  wire idx_fresh;  wire [LAYW-1:0] idx_win;
    wire                      gn_req, gn_which;  wire [DIMW-1:0] gn_idx;  reg [15:0] gn_val;
    wire                      aw_req;  wire [3:0] aw_sel;  wire [A_GRPW-1:0] aw_grp;  wire [A_KCW-1:0] aw_k;
    reg  [PE_N*4-1:0]         aw_q;
    reg  [16*PE_N*A_NSB-1:0]  aw_d, aw_dmin;
    reg  [96*PE_N*A_NSB-1:0]  aw_scales;
    wire                      rw_req;  wire [R_KW-1:0] rw_k;
    reg  [4*N_EXPERT-1:0]         rw_q;
    reg  [16*N_EXPERT*R_NSB-1:0]  rw_d, rw_dmin;
    reg  [96*N_EXPERT*R_NSB-1:0]  rw_scales;
    wire                      fw_req;  wire [1:0] fw_sel;  wire [FF_GWD-1:0] fw_grp;  wire [FF_KWD-1:0] fw_k;
    wire                      fw_shared;  wire [EIDXW-1:0] fw_eidx;
    reg  [4*TN-1:0]           fw_q, fw_q_up;
    reg  [16*TN*FF_NSB_D-1:0] fw_d_g, fw_dmin_g, fw_d_u, fw_dmin_u;
    reg  [96*TN*FF_NSB_D-1:0] fw_scales_g, fw_scales_u;
    wire                      fn_req;  wire [DIMW-1:0] fn_idx;  reg [15:0] fn_val;
    wire                      lw_req;  wire [VTW-1:0] lw_vtile;  wire [DIMW-1:0] lw_k;  reg [LM_TN*16-1:0] lw_col;
    wire                      kc_req;  wire [IDXW-1:0] kc_idx;  reg [KV_LORA*16-1:0] kc_ckv;  reg [ROPE*16-1:0] kc_krope;
    wire [KVPOSW-1:0]         kv_row_sel;  reg [ROW_BITS-1:0] kv_row_in;
    wire                      flash_req, flash_is_expert;
    wire [EIDXW-1:0]          flash_expert_id;  wire [KVPOSW-1:0] flash_row_idx;
    reg                       flash_done;  reg [ROW_BITS-1:0] flash_row;
    wire [TOKW-1:0]           argmax_o;  wire [MODEL_DIM*16-1:0] h_state;  wire mdl_busy;
    wire                      ec_resp_valid, ec_hit;  wire [CSLOTW-1:0] ec_resp_slot;  wire ec_busy;
    wire [31:0]               ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit;
    wire                      kv_row_valid;  wire [ROW_BITS-1:0] kv_row_out;  wire kv_busy;
    wire [KVPOSW-1:0]         kv_append_count, kv_resident_lo;  wire kv_overflowed;
    wire [31:0]               ec_dropped;
    wire [DDR_NCH-1:0]            mem_req_valid;
    wire [DDR_NCH-1:0]            mem_req_ready;
    wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag;
    reg  [DDR_NCH-1:0]            mem_resp_valid;
    wire [DDR_NCH-1:0]            mem_resp_ready;
    reg  [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data;
    reg  [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag;
    wire                      wl_mem_en;  wire [WL_ADDR_W-1:0] wl_mem_addr;  reg [WL_DATA_W-1:0] wl_mem_data;
    wire [31:0]               xbar_req_count, xbar_resp_count;
    wire                      xbar_resp_valid;  wire [DDR_DATA_W-1:0] xbar_resp_data;
    wire                      loader_busy;  wire [31:0] loader_done_count, loader_beat_count;
    wire [4*WL_PE_N-1:0]      loader_w_q;  wire loader_in_valid;

    glm_q4k_system #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH),
        .LOOPBACK(0), .LOOPBACK_FW(0), .LOOPBACK_REST(1), // <-- the proof knob (rest only)
        .EXPERT_STALL(EXPERT_STALL_CFG), .RESIDENT(RESIDENT_CFG),
        .DSA_REAL_IDX(DSA_REAL_IDX_CFG),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos), .s_len(s_len),
        .busy(busy), .done(done), .next_tok(next_tok), .tok_valid(tok_valid),
        .logits(logits),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_req(kc_req), .kc_idx(kc_idx),
        .kv_row_sel(kv_row_sel), .kv_row_in(kv_row_in),
        .flash_req(flash_req), .flash_is_expert(flash_is_expert),
        .flash_expert_id(flash_expert_id), .flash_row_idx(flash_row_idx),
        .flash_done(flash_done), .flash_row(flash_row),
        .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(wl_mem_data),
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0),
        .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
        .argmax_o(argmax_o), .h_state(h_state), .mdl_busy(mdl_busy),
        .ec_resp_valid(ec_resp_valid), .ec_hit(ec_hit), .ec_resp_slot(ec_resp_slot),
        .ec_busy(ec_busy), .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(ec_demand_stall_cycles),
        .ec_pf_issued(ec_pf_issued), .ec_pf_hit(ec_pf_hit),
        .kv_row_valid(kv_row_valid), .kv_row_out(kv_row_out), .kv_busy(kv_busy),
        .kv_append_count(kv_append_count), .kv_resident_lo(kv_resident_lo),
        .kv_overflowed(kv_overflowed), .ec_dropped(ec_dropped),
        .xbar_req_count(xbar_req_count), .xbar_resp_count(xbar_resp_count),
        .xbar_resp_valid(xbar_resp_valid), .xbar_resp_data(xbar_resp_data),
        .loader_busy(loader_busy), .loader_done_count(loader_done_count),
        .loader_beat_count(loader_beat_count),
        .loader_w_q(loader_w_q), .loader_in_valid(loader_in_valid)
    );

    // ================= system weight/KV responders (stub ports) =====
    //   NOTE: at LOOPBACK_REST=1 the die's rw_q / lw_col / gn_val come from the xbar
    //   round-trip, NOT from these `rw_q`/`lw_col`/`gn_val` inputs (die_rw_q=lbrw_q_q
    //   etc.).  They are still driven (harmlessly) so the port list matches the perf
    //   TB exactly; the rw d/dmin/scales stubs ARE still consumed same-cycle by the
    //   die (only the rw CODES loop, exactly as LOOPBACK keeps aw_d/aw_dmin/aw_scales),
    //   and aw/fw are served same-cycle by the stub (their loopbacks are OFF here).
    integer t, ft, re, sb;
    always @* em_val = gen_bf16(em_tok*MODEL_DIM + em_idx + 7001);
    always @* fn_val = gen_bf16(fn_idx + 7207);
    always @* gn_val = gen_bf16(db_layer*1024 + gn_which*512 + gn_idx + 7411);
    always @* begin
        for (t=0;t<LM_TN;t=t+1)
            lw_col[16*t+:16] = gen_bf16((lw_vtile*LM_TN+t)*MODEL_DIM + lw_k + 7603);
    end
    always @* begin
        for (t=0;t<PE_N;t=t+1) begin
            aw_q[4*t+:4] = f_awq(db_layer, aw_sel, aw_grp*PE_N+t, aw_k);
            for (sb=0;sb<A_NSB;sb=sb+1) begin
                aw_d   [16*(sb*PE_N+t)+:16] = f_awd (db_layer, aw_sel, aw_grp*PE_N+t);
                aw_dmin[16*(sb*PE_N+t)+:16] = f_awdm(db_layer, aw_sel, aw_grp*PE_N+t);
                aw_scales[96*(sb*PE_N+t)   +:32] = gen_s32(db_layer*7919+aw_sel*104729+(aw_grp*PE_N+t)*611953+601);
                aw_scales[96*(sb*PE_N+t)+32+:32] = gen_s32(db_layer*7919+aw_sel*104729+(aw_grp*PE_N+t)*611953+602);
                aw_scales[96*(sb*PE_N+t)+64+:32] = gen_s32(db_layer*7919+aw_sel*104729+(aw_grp*PE_N+t)*611953+603);
            end
        end
    end
    always @* begin
        for (re=0;re<N_EXPERT;re=re+1) begin
            rw_q[4*re+:4] = f_rwq(db_layer, re, rw_k);
            for (sb=0;sb<R_NSB;sb=sb+1) begin
                rw_d   [16*(sb*N_EXPERT+re)+:16] = gen_fp16(db_layer*7919+re*350377+421);
                rw_dmin[16*(sb*N_EXPERT+re)+:16] = gen_fp16(db_layer*7919+re*350377+431);
                rw_scales[96*(sb*N_EXPERT+re)   +:32] = gen_s32(db_layer*7919+re*350377+441);
                rw_scales[96*(sb*N_EXPERT+re)+32+:32] = gen_s32(db_layer*7919+re*350377+442);
                rw_scales[96*(sb*N_EXPERT+re)+64+:32] = gen_s32(db_layer*7919+re*350377+443);
            end
        end
    end
    always @* begin
        for (ft=0;ft<TN;ft=ft+1) begin
            fw_q   [4*ft+:4] = f_fwq(db_layer, fw_sel, fw_shared, fw_eidx, fw_grp*TN+ft, fw_k);
            fw_q_up[4*ft+:4] = f_fwq(db_layer, 3,      fw_shared, fw_eidx, fw_grp*TN+ft, fw_k);
            for (sb=0;sb<FF_NSB_D;sb=sb+1) begin
                fw_d_g   [16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+521);
                fw_dmin_g[16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+531);
                fw_d_u   [16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+521);
                fw_dmin_u[16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+531);
                fw_scales_g[96*(sb*TN+ft)   +:32] = gen_s32(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+541);
                fw_scales_g[96*(sb*TN+ft)+32+:32] = gen_s32(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+542);
                fw_scales_g[96*(sb*TN+ft)+64+:32] = gen_s32(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+543);
                fw_scales_u[96*(sb*TN+ft)   +:32] = gen_s32(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+541);
                fw_scales_u[96*(sb*TN+ft)+32+:32] = gen_s32(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+542);
                fw_scales_u[96*(sb*TN+ft)+64+:32] = gen_s32(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+543);
            end
        end
    end
    integer cd;
    always @* begin
        for (cd=0;cd<KV_LORA;cd=cd+1) kc_ckv  [16*cd+:16] = gen_bf16(db_layer*513 + kc_idx*67 + cd*7 + 8011);
        for (cd=0;cd<ROPE;cd=cd+1)    kc_krope[16*cd+:16] = gen_bf16(db_layer*771 + kc_idx*91 + cd*5 + 8101);
    end

    // ================= INDEPENDENT REFERENCE: standalone glm_model_q4k ==========
    reg                       r_start;
    wire                      r_busy, r_done;
    wire [TOKW-1:0]           r_argmax;
    wire [VOCAB*16-1:0]       r_logits;
    wire                      r_em_req;  wire [TOKW-1:0] r_em_tok;  wire [DIMW-1:0] r_em_idx;  reg [15:0] r_em_val;
    wire [LAYW-1:0]           r_db_layer;  wire r_idx_fresh;  wire [LAYW-1:0] r_idx_win;
    wire                      r_gn_req, r_gn_which;  wire [DIMW-1:0] r_gn_idx;  reg [15:0] r_gn_val;
    wire                      r_aw_req;  wire [3:0] r_aw_sel;  wire [A_GRPW-1:0] r_aw_grp;  wire [A_KCW-1:0] r_aw_k;
    reg  [PE_N*4-1:0]         r_aw_q;
    reg  [16*PE_N*A_NSB-1:0]  r_aw_d, r_aw_dmin;
    reg  [96*PE_N*A_NSB-1:0]  r_aw_scales;
    wire                      r_rw_req;  wire [R_KW-1:0] r_rw_k;
    reg  [4*N_EXPERT-1:0]         r_rw_q;
    reg  [16*N_EXPERT*R_NSB-1:0]  r_rw_d, r_rw_dmin;
    reg  [96*N_EXPERT*R_NSB-1:0]  r_rw_scales;
    wire                      r_fw_req;  wire [1:0] r_fw_sel;  wire [FF_GWD-1:0] r_fw_grp;  wire [FF_KWD-1:0] r_fw_k;
    wire                      r_fw_shared;  wire [EIDXW-1:0] r_fw_eidx;
    reg  [4*TN-1:0]           r_fw_q, r_fw_q_up;
    reg  [16*TN*FF_NSB_D-1:0] r_fw_d_g, r_fw_dmin_g, r_fw_d_u, r_fw_dmin_u;
    reg  [96*TN*FF_NSB_D-1:0] r_fw_scales_g, r_fw_scales_u;
    wire                      r_fn_req;  wire [DIMW-1:0] r_fn_idx;  reg [15:0] r_fn_val;
    wire                      r_lw_req;  wire [VTW-1:0] r_lw_vtile;  wire [DIMW-1:0] r_lw_k;  reg [LM_TN*16-1:0] r_lw_col;
    wire                      r_kc_req;  wire [IDXW-1:0] r_kc_idx;  wire r_kc_seq;
    reg  [KV_LORA*16-1:0]     r_kc_ckv;  reg [ROPE*16-1:0] r_kc_krope;
    reg                       r_kc_valid;
    wire [MODEL_DIM*16-1:0]   r_h_state;

    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .DSA_REAL_IDX(DSA_REAL_IDX_CFG)
    ) u_ref (
        .clk(clk), .rst(rst),
        .start(r_start), .busy(r_busy), .done(r_done),
        .token_id(prompt_tok), .pos(start_pos), .pos_vec({POSW{1'b0}}),
        .s_len_vec({(IDXW+1){1'b0}}), .seq_vec(1'b0), .s_len(s_len),
        .logits(r_logits), .argmax(r_argmax),
        .em_req(r_em_req), .em_tok(r_em_tok), .em_idx(r_em_idx), .em_val(r_em_val),
        .db_layer(r_db_layer), .idx_fresh(r_idx_fresh), .idx_win(r_idx_win),
        .gn_req(r_gn_req), .gn_which(r_gn_which), .gn_idx(r_gn_idx), .gn_val(r_gn_val),
        .aw_req(r_aw_req), .aw_sel(r_aw_sel), .aw_grp(r_aw_grp), .aw_k(r_aw_k),
        .aw_q(r_aw_q), .aw_d(r_aw_d), .aw_dmin(r_aw_dmin), .aw_scales(r_aw_scales),
        .kc_req(r_kc_req), .kc_idx(r_kc_idx), .kc_seq(r_kc_seq),
        .kc_ckv(r_kc_ckv), .kc_krope(r_kc_krope), .kc_valid(r_kc_valid),
        .rw_req(r_rw_req), .rw_k(r_rw_k),
        .rw_q(r_rw_q), .rw_d(r_rw_d), .rw_dmin(r_rw_dmin), .rw_scales(r_rw_scales),
        .fw_req(r_fw_req), .fw_sel(r_fw_sel), .fw_grp(r_fw_grp), .fw_k(r_fw_k),
        .fw_shared(r_fw_shared), .fw_eidx(r_fw_eidx),
        .fw_q(r_fw_q), .fw_q_up(r_fw_q_up),
        .fw_d_g(r_fw_d_g), .fw_dmin_g(r_fw_dmin_g), .fw_scales_g(r_fw_scales_g),
        .fw_d_u(r_fw_d_u), .fw_dmin_u(r_fw_dmin_u), .fw_scales_u(r_fw_scales_u),
        .fn_req(r_fn_req), .fn_idx(r_fn_idx), .fn_val(r_fn_val),
        .lw_req(r_lw_req), .lw_vtile(r_lw_vtile), .lw_k(r_lw_k), .lw_col(r_lw_col),
        .h_state(r_h_state)
    );

    integer rt, rft, rre, rsb, rcd;
    always @* r_em_val = gen_bf16(r_em_tok*MODEL_DIM + r_em_idx + 7001);
    always @* r_fn_val = gen_bf16(r_fn_idx + 7207);
    always @* r_gn_val = gen_bf16(r_db_layer*1024 + r_gn_which*512 + r_gn_idx + 7411);
    always @* begin
        for (rt=0;rt<LM_TN;rt=rt+1)
            r_lw_col[16*rt+:16] = gen_bf16((r_lw_vtile*LM_TN+rt)*MODEL_DIM + r_lw_k + 7603);
    end
    always @* begin
        for (rt=0;rt<PE_N;rt=rt+1) begin
            r_aw_q[4*rt+:4] = f_awq(r_db_layer, r_aw_sel, r_aw_grp*PE_N+rt, r_aw_k);
            for (rsb=0;rsb<A_NSB;rsb=rsb+1) begin
                r_aw_d   [16*(rsb*PE_N+rt)+:16] = f_awd (r_db_layer, r_aw_sel, r_aw_grp*PE_N+rt);
                r_aw_dmin[16*(rsb*PE_N+rt)+:16] = f_awdm(r_db_layer, r_aw_sel, r_aw_grp*PE_N+rt);
                r_aw_scales[96*(rsb*PE_N+rt)   +:32] = gen_s32(r_db_layer*7919+r_aw_sel*104729+(r_aw_grp*PE_N+rt)*611953+601);
                r_aw_scales[96*(rsb*PE_N+rt)+32+:32] = gen_s32(r_db_layer*7919+r_aw_sel*104729+(r_aw_grp*PE_N+rt)*611953+602);
                r_aw_scales[96*(rsb*PE_N+rt)+64+:32] = gen_s32(r_db_layer*7919+r_aw_sel*104729+(r_aw_grp*PE_N+rt)*611953+603);
            end
        end
    end
    always @* begin
        for (rre=0;rre<N_EXPERT;rre=rre+1) begin
            r_rw_q[4*rre+:4] = f_rwq(r_db_layer, rre, r_rw_k);
            for (rsb=0;rsb<R_NSB;rsb=rsb+1) begin
                r_rw_d   [16*(rsb*N_EXPERT+rre)+:16] = gen_fp16(r_db_layer*7919+rre*350377+421);
                r_rw_dmin[16*(rsb*N_EXPERT+rre)+:16] = gen_fp16(r_db_layer*7919+rre*350377+431);
                r_rw_scales[96*(rsb*N_EXPERT+rre)   +:32] = gen_s32(r_db_layer*7919+rre*350377+441);
                r_rw_scales[96*(rsb*N_EXPERT+rre)+32+:32] = gen_s32(r_db_layer*7919+rre*350377+442);
                r_rw_scales[96*(rsb*N_EXPERT+rre)+64+:32] = gen_s32(r_db_layer*7919+rre*350377+443);
            end
        end
    end
    always @* begin
        for (rft=0;rft<TN;rft=rft+1) begin
            r_fw_q   [4*rft+:4] = f_fwq(r_db_layer, r_fw_sel, r_fw_shared, r_fw_eidx, r_fw_grp*TN+rft, r_fw_k);
            r_fw_q_up[4*rft+:4] = f_fwq(r_db_layer, 3,        r_fw_shared, r_fw_eidx, r_fw_grp*TN+rft, r_fw_k);
            for (rsb=0;rsb<FF_NSB_D;rsb=rsb+1) begin
                r_fw_d_g   [16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+521);
                r_fw_dmin_g[16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+531);
                r_fw_d_u   [16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+521);
                r_fw_dmin_u[16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+531);
                r_fw_scales_g[96*(rsb*TN+rft)   +:32] = gen_s32(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+541);
                r_fw_scales_g[96*(rsb*TN+rft)+32+:32] = gen_s32(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+542);
                r_fw_scales_g[96*(rsb*TN+rft)+64+:32] = gen_s32(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+543);
                r_fw_scales_u[96*(rsb*TN+rft)   +:32] = gen_s32(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+541);
                r_fw_scales_u[96*(rsb*TN+rft)+32+:32] = gen_s32(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+542);
                r_fw_scales_u[96*(rsb*TN+rft)+64+:32] = gen_s32(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+543);
            end
        end
    end
    always @* begin
        for (rcd=0;rcd<KV_LORA;rcd=rcd+1) r_kc_ckv  [16*rcd+:16] = gen_bf16(r_db_layer*513 + r_kc_idx*67 + rcd*7 + 8011);
        for (rcd=0;rcd<ROPE;rcd=rcd+1)    r_kc_krope[16*rcd+:16] = gen_bf16(r_db_layer*771 + r_kc_idx*91 + rcd*5 + 8101);
    end
    always @(posedge clk) begin
        if (rst) r_kc_valid <= 1'b0;
        else     r_kc_valid <= r_kc_req;
    end
    reg [TOKW-1:0] r_tok_lat;  reg r_done_seen;
    always @(posedge clk) begin
        if (rst)            begin r_done_seen<=1'b0; r_tok_lat<={TOKW{1'b0}}; end
        else if (r_start)   r_done_seen<=1'b0;
        else if (r_done)    begin r_tok_lat<=r_argmax; r_done_seen<=1'b1; end
    end

    // ---- lint guard ----
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, em_req, aw_req, fw_req, rw_req, gn_req, fn_req,
                     lw_req, idx_fresh, idx_win, mdl_busy, ec_resp_valid, ec_hit,
                     ec_resp_slot, ec_busy, ec_pf_issued, ec_pf_hit, kv_row_out,
                     kv_busy, kv_resident_lo, flash_expert_id, h_state, gn_which,
                     done, kv_overflowed, r_busy, r_em_req, r_aw_req, r_fw_req,
                     r_rw_req, r_gn_req, r_fn_req, r_lw_req, r_idx_fresh,
                     r_idx_win, r_logits, r_h_state, r_gn_which, r_kc_seq,
                     loader_busy, mem_req_tag, ec_hit_count, ec_miss_count,
                     ec_demand_stall_cycles, ec_dropped, kv_append_count,
                     kv_row_valid, aw_q, xbar_resp_data, loader_w_q,
                     loader_in_valid, loader_done_count, loader_beat_count,
                     flash_is_expert, kc_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= KV latent-ROW stub =================
    integer rr;
    always @* begin
        kv_row_in = {ROW_BITS{1'b0}};
        for (rr=0;rr<(KV_LORA+ROPE);rr=rr+1)
            kv_row_in[16*rr+:16] = gen_bf16(kv_row_sel*131 + rr*7 + 3);
    end

    // ================= FLASH PHY STUB (FLASH_LAT-cycle fetch) =================
    reg [31:0] fl_timer; reg fl_active; reg prev_req;
    always @(posedge clk) begin
        if (rst) begin
            fl_timer <= 32'd0; fl_active <= 1'b0; flash_done <= 1'b0; prev_req <= 1'b0;
        end else begin
            flash_done <= 1'b0;
            if (!fl_active) begin
                if (flash_req && !prev_req) begin
                    fl_active <= 1'b1; fl_timer <= FLASH_LAT[31:0];
                end
            end else begin
                if (fl_timer <= 32'd1) begin flash_done <= 1'b1; fl_active <= 1'b0; end
                else fl_timer <= fl_timer - 32'd1;
            end
            prev_req <= flash_req;
        end
    end
    integer cr;
    always @* begin
        flash_row = {ROW_BITS{1'b0}};
        for (cr=0;cr<(KV_LORA+ROPE);cr=cr+1)
            flash_row[16*cr+:16] = gen_bf16(flash_row_idx*977 + cr*13 + 1);
    end

    // ================= LOADER STAGING-TIER RAM (latency-1 registered read) =======
    localparam integer STG_DEPTH = 2048;
    reg [WL_DATA_W-1:0] STAGE [0:STG_DEPTH-1];
    integer si, sj;
    initial for (si=0; si<STG_DEPTH; si=si+1)
        for (sj=0; sj<WL_DATA_W/16; sj=sj+1)
            STAGE[si][16*sj+:16] = gen_bf16(si*61 + sj*17 + 9);
    always @(posedge clk) begin
        if (wl_mem_en) wl_mem_data <= STAGE[wl_mem_addr[10:0]];
        else           wl_mem_data <= {WL_DATA_W{1'b0}};
    end

    // ================= DDR5 PER-CHANNEL MEMORY MODEL + C8 LOOPBACK-REST RESPONDER =
    //   Single in-flight table; each read completes DDR_ROW_LAT cycles later on
    //   its banked channel, held until mem_resp_ready accepts.  For NORMAL reads
    //   the returned beat is gen_beat(tag,addr) (perf TB behaviour).  For a C8
    //   LOOPBACK-REST read -- addr[24 +: 8] in {8'hC7,8'hD8,8'hE9} -- the beat's low
    //   bits are the packed rw / lw / gn bytes decoded from the address (EXACTLY the
    //   bytes the stub rw_q / lw_col / gn_val present for that key at REST=0).
    localparam [7:0] TAG_LBRW_MARK = 8'hC7;     // addr[24 +: 8] on an rw loopback read
    localparam [7:0] TAG_LBLW_MARK = 8'hD8;     // addr[24 +: 8] on a  lw loopback read
    localparam [7:0] TAG_LBGN_MARK = 8'hE9;     // addr[24 +: 8] on a  gn loopback read
    localparam integer NINF = 64;
    reg                  infv  [0:NINF-1];
    reg [DDR_TAG_W-1:0]  inftg [0:NINF-1];
    reg [DDR_ADDR_W-1:0] infad [0:NINF-1];
    reg [15:0]           inftm [0:NINF-1];   // remaining latency (0 => ready)
    integer ii, cc;
    // per-family loopback-path activity counters (non-vacuity: each MUST be >0)
    integer lbrw_req_seen, lbrw_resp_seen;
    integer lblw_req_seen, lblw_resp_seen;
    integer lbgn_req_seen, lbgn_resp_seen;

    // ---- L3 S2: does the die survive a memory that says WAIT? --------------------
    //   EVERY system-level TB in this repo ties mem_req_ready to all-ones, so the
    //   whole system has never been exercised against backpressure.  A real DDR4
    //   controller refuses on most cycles (refresh, bank conflict, arbitration), so
    //   this is a property the board WILL exercise on day one and simulation never has.
    //
    //   -DTB_REQ_STALL drives a deterministic LFSR pattern instead: each channel is
    //   ready roughly half the time, with runs of both accept and refuse.  The
    //   committed token stream must be UNCHANGED -- backpressure may cost cycles, it
    //   may never change a value.
`ifdef TB_REQ_STALL
    reg [15:0] rq_lfsr;
    always @(posedge clk)
        if (rst) rq_lfsr <= 16'hACE1;
        else     rq_lfsr <= {rq_lfsr[14:0],
                             rq_lfsr[15] ^ rq_lfsr[13] ^ rq_lfsr[12] ^ rq_lfsr[10]};
    assign mem_req_ready = rq_lfsr[DDR_NCH-1:0];
`else
    assign mem_req_ready = {DDR_NCH{1'b1}};
`endif

    // ---- S2 evidence: was the stall REAL, and does the request port hold stable? --
    //   (1) stall_cyc: cycles where the system asserted a request and the memory
    //       refused.  If this is 0 the backpressure test proved nothing.
    //   (2) unstable_cyc: cycles where a request was held (valid high, ready low) and
    //       the presented {addr,tag} CHANGED.  glm_q4k_system drives
    //       `xreq_valid = any_pending` over a COMBINATIONAL priority mux, so a refused
    //       request can be replaced by a different one next cycle.  Internally that is
    //       consistent (accept is evaluated in the same cycle), and the tokens above
    //       prove it -- but AXI4 requires ARADDR to stay put until ARREADY, so this is
    //       exactly what a MIG shim must absorb with a registered skid buffer.
    integer s2_stall_cyc, s2_unstable_cyc, s2_i;
    reg [DDR_ADDR_W-1:0] s2_addr_q [0:DDR_NCH-1];
    reg [DDR_TAG_W-1:0]  s2_tag_q  [0:DDR_NCH-1];
    reg [DDR_NCH-1:0]    s2_held;
    initial begin s2_stall_cyc = 0; s2_unstable_cyc = 0; s2_held = 0; end
    always @(posedge clk) if (!rst) begin
        for (s2_i = 0; s2_i < DDR_NCH; s2_i = s2_i + 1) begin
            if (mem_req_valid[s2_i] && !mem_req_ready[s2_i]) begin
                s2_stall_cyc = s2_stall_cyc + 1;
                if (s2_held[s2_i]
                    && ((mem_req_addr[s2_i*DDR_ADDR_W +: DDR_ADDR_W] !== s2_addr_q[s2_i])
                     || (mem_req_tag [s2_i*DDR_TAG_W  +: DDR_TAG_W ] !== s2_tag_q [s2_i])))
                    s2_unstable_cyc = s2_unstable_cyc + 1;
                s2_addr_q[s2_i] <= mem_req_addr[s2_i*DDR_ADDR_W +: DDR_ADDR_W];
                s2_tag_q [s2_i] <= mem_req_tag [s2_i*DDR_TAG_W  +: DDR_TAG_W ];
                s2_held[s2_i]   <= 1'b1;
            end else s2_held[s2_i] <= 1'b0;
        end
    end
    final $display("S2_EVIDENCE stall_cyc=%0d unstable_cyc=%0d", s2_stall_cyc, s2_unstable_cyc);

    reg [31:0] presIdx [0:DDR_NCH-1];
    reg        presV   [0:DDR_NCH-1];

    // NORMAL banked-read beat (perf TB behaviour, X-clean deterministic).
    function automatic [DDR_DATA_W-1:0] gen_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad; integer ln; begin
        gen_beat = {DDR_DATA_W{1'b0}};
        for (ln=0; ln<DDR_DATA_W/16; ln=ln+1)
            gen_beat[16*ln+:16] = gen_bf16(({24'd0,tg}*7 + ad*3 + ln*5 + 1));
        end
    endfunction

    // C8 LOOPBACK-REST rw beat: decode {db_layer,rw_k} from the marked address and
    // pack lane e = f_rwq(layer,e,rw_k) into the low 4*N_EXPERT bits (the die's
    // router CODE lanes).  High bits kept X-clean (gen_beat base) for the whole-beat
    // X-monitor; the die only consumes bits [4*N_EXPERT-1:0].
    function automatic [DDR_DATA_W-1:0] gen_lbrw_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad;
        integer ln; integer ly, kk;
        reg [DDR_DATA_W-1:0] b; begin
        b   = gen_beat(tg, ad);              // X-clean base for the unused high bits
        kk  = ad[0 +: R_KW];
        ly  = ad[RWA_LY_LO +: LAYW];
        for (ln=0; ln<N_EXPERT; ln=ln+1)
`ifdef LBRESTINJECT_RW
            // INJECTION (injection-ONLY): corrupt the fed-back rw router-code lane 0 on
            // every loopback beat.  rw is OUTPUT-INSENSITIVE (a one-code router step is
            // absorbed by top-k selection), so the committed-token binding alone would
            // MISS this -- the DIRECT per-beat die_rw_q byte binding must catch it and
            // fail the gate.  This is the load-bearing demonstration that direct bindings
            // are required for output-insensitive weight families.
            b[4*ln +: 4] = f_rwq(ly, ln, kk) ^ (ln == 0 ? 4'h1 : 4'h0);
`else
            b[4*ln +: 4] = f_rwq(ly, ln, kk);
`endif
        gen_lbrw_beat = b;
        end
    endfunction

    // C8 LOOPBACK-REST lw beat: decode {lw_vtile,lw_k} and pack lane t =
    // gen_bf16((vtile*LM_TN+t)*MODEL_DIM + lw_k + 7603) into the low LM_TN*16 bits.
    function automatic [DDR_DATA_W-1:0] gen_lblw_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad;
        integer ln; integer vtile, kk;
        reg [DDR_DATA_W-1:0] b; begin
        b     = gen_beat(tg, ad);            // X-clean base for the unused high bits
        kk    = ad[0 +: DIMW];
        vtile = ad[LWA_VT_LO +: VTW];
        for (ln=0; ln<LM_TN; ln=ln+1)
            b[16*ln +: 16] = gen_bf16((vtile*LM_TN + ln)*MODEL_DIM + kk + 7603);
        gen_lblw_beat = b;
        end
    endfunction

    // C8 LOOPBACK-REST gn beat: decode {db_layer,gn_which,gn_idx} and pack
    // gen_bf16(layer*1024 + which*512 + idx + 7411) into the low 16 bits.
    function automatic [DDR_DATA_W-1:0] gen_lbgn_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad;
        integer ly, which, idx;
        reg [DDR_DATA_W-1:0] b; begin
        b     = gen_beat(tg, ad);            // X-clean base for the unused high bits
        idx   = ad[0 +: DIMW];
        which = ad[GNA_WH_LO +: 1];
        ly    = ad[GNA_LY_LO +: LAYW];
        b[0 +: 16] = gen_bf16(ly*1024 + which*512 + idx + 7411);
`ifdef LBRESTINJECT
        // SOUNDNESS INJECTION: corrupt the fed-back gn value on EVERY gn beat by
        // flipping its mantissa LSB (a single-ULP bf16 perturbation).  gn is a per-dim
        // gain/norm that was MEASURED output-insensitive (the change is normalised away
        // over this short decode), so the committed-token binding alone can miss it --
        // but the DIRECT per-beat byte binding compares dut.die_gn_val against the
        // expected stub value on every gn beat, so this trips it and the gate FAILS.
        // This is the sharper injection: it targets the output-INSENSITIVE family and
        // thereby isolates the direct byte binding as the check that catches it.
        b[0] = ~b[0];
`endif
        gen_lbgn_beat = b;
        end
    endfunction

    // ---- combinational response presentation (oldest-ready per channel) ----
    always @* begin
        mem_resp_valid = {DDR_NCH{1'b0}};
        mem_resp_data  = {(DDR_NCH*DDR_DATA_W){1'b0}};
        mem_resp_tag   = {(DDR_NCH*DDR_TAG_W){1'b0}};
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            presV[cc]   = 1'b0;
            presIdx[cc] = 32'd0;
        end
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            for (ii=NINF-1; ii>=0; ii=ii-1) begin
                if (infv[ii] && (inftm[ii]==16'd0) &&
                    ((DDR_NCH==1) ? (cc==0)
                                  : (infad[ii][CH_SEL_W-1:0] == cc[CH_SEL_W-1:0]))) begin
                    presV[cc]   = 1'b1;
                    presIdx[cc] = ii;
                end
            end
            if (presV[cc]) begin
                mem_resp_valid[cc] = 1'b1;
                // C8 LOOPBACK-REST decode by the per-family marker in addr[24+:8];
                // every other read keeps the normal banked beat.
                if (infad[presIdx[cc]][24 +: 8] == TAG_LBRW_MARK)
                    mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                        gen_lbrw_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                else if (infad[presIdx[cc]][24 +: 8] == TAG_LBLW_MARK)
                    mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                        gen_lblw_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                else if (infad[presIdx[cc]][24 +: 8] == TAG_LBGN_MARK)
                    mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                        gen_lbgn_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                else
                    mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                        gen_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                mem_resp_tag[cc*DDR_TAG_W +: DDR_TAG_W] = inftg[presIdx[cc]];
            end
        end
    end

    integer freeslot; reg got_free;
    always @(posedge clk) begin
        if (rst) begin
            for (ii=0; ii<NINF; ii=ii+1) begin
                infv[ii]  <= 1'b0; inftg[ii] <= {DDR_TAG_W{1'b0}};
                infad[ii] <= {DDR_ADDR_W{1'b0}}; inftm[ii] <= 16'd0;
            end
            lbrw_req_seen <= 0; lbrw_resp_seen <= 0;
            lblw_req_seen <= 0; lblw_resp_seen <= 0;
            lbgn_req_seen <= 0; lbgn_resp_seen <= 0;
        end else begin
            for (ii=0; ii<NINF; ii=ii+1)
                if (infv[ii] && (inftm[ii]!=16'd0)) inftm[ii] <= inftm[ii]-16'd1;
            for (cc=0; cc<DDR_NCH; cc=cc+1)
                if (presV[cc] && mem_resp_ready[cc]) begin
                    infv[presIdx[cc]] <= 1'b0;
                    // a per-family loopback beat drained
                    if (infad[presIdx[cc]][24 +: 8] == TAG_LBRW_MARK)
                        lbrw_resp_seen <= lbrw_resp_seen + 1;
                    else if (infad[presIdx[cc]][24 +: 8] == TAG_LBLW_MARK)
                        lblw_resp_seen <= lblw_resp_seen + 1;
                    else if (infad[presIdx[cc]][24 +: 8] == TAG_LBGN_MARK)
                        lbgn_resp_seen <= lbgn_resp_seen + 1;
                end
            got_free = 1'b0; freeslot = 0;
            for (ii=NINF-1; ii>=0; ii=ii-1)
                if (!infv[ii]) begin got_free = 1'b1; freeslot = ii; end
            for (cc=0; cc<DDR_NCH; cc=cc+1) begin
                if (mem_req_valid[cc] && mem_req_ready[cc] && got_free) begin
                    infv[freeslot]  <= 1'b1;
                    inftg[freeslot] <= mem_req_tag[cc*DDR_TAG_W +: DDR_TAG_W];
                    infad[freeslot] <= mem_req_addr[cc*DDR_ADDR_W +: DDR_ADDR_W];
                    inftm[freeslot] <= DDR_ROW_LAT[15:0];
                    // a per-family loopback read captured
                    if (mem_req_addr[cc*DDR_ADDR_W+24 +: 8] == TAG_LBRW_MARK)
                        lbrw_req_seen <= lbrw_req_seen + 1;
                    else if (mem_req_addr[cc*DDR_ADDR_W+24 +: 8] == TAG_LBLW_MARK)
                        lblw_req_seen <= lblw_req_seen + 1;
                    else if (mem_req_addr[cc*DDR_ADDR_W+24 +: 8] == TAG_LBGN_MARK)
                        lbgn_req_seen <= lbgn_req_seen + 1;
                end
            end
        end
    end

    // ================= continuous X monitor on the xbar resp beat ==========
    integer xmon_err;
    always @(posedge clk) if (!rst) begin
        if (xbar_resp_valid)
            if (^xbar_resp_data === 1'bx) begin
                $display("FAIL: xbar_resp_data X while valid @%0t", $time); xmon_err=xmon_err+1; end
        if (|mem_req_valid && (^mem_req_addr === 1'bx)) begin
            $display("FAIL: mem_req_addr X while valid @%0t", $time); xmon_err=xmon_err+1; end
    end

    // ================= DIRECT PER-BEAT DIE-INPUT BYTE BINDING (MANDATORY) ========
    //   The committed-token binding (in run_token) proves the OUTPUT is bit-exact, but
    //   rw and gn were MEASURED output-insensitive -- the top-k absorbs a router-code
    //   step; the per-dim gains normalise away -- so a bit-exact token does NOT by itself
    //   prove the die CONSUMED the correct rw/gn bytes off the fabric.  This monitor
    //   closes that gap byte-for-byte: on every posedge where the die is pulling a family
    //   (XX_req high) AND that family's loopback beat for the LIVE key is staged and
    //   ready (dut.g_lbrest.lb{rw,lw,gn}_have -- the exact condition under which the die
    //   is presented, and about to consume, the fed-back bytes), the DUT-internal wire
    //   the die reads -- dut.die_rw_q / dut.die_lw_col / dut.die_gn_val -- MUST equal the
    //   SAME deterministic value function that drives the stub and the reference for the
    //   LIVE key {db_layer,rw_k} / {lw_vtile,lw_k} / {db_layer,gn_which,gn_idx}.  Any
    //   mismatch fails the gate.  rw's expert lanes and lw's column lanes are POSITIONAL
    //   (lane index e/t is not an address field); gn is a single value.  (die_* are
    //   module-scope wires in glm_q4k_system; lb*_have live in the LOOPBACK_REST=1
    //   generate branch g_lbrest; both are reachable by hierarchical reference here.)
    integer rw_derr, lw_derr, gn_derr;   // direct-binding byte mismatches, per family
    integer rw_dchk, lw_dchk, gn_dchk;   // direct-binding samples taken,  per family
    integer de, dtt;
    always @(posedge clk) if (!rst) begin
        // rw : N_EXPERT positional Q4_K router-code lanes, live key {db_layer, rw_k}
        if (rw_req && dut.g_lbrest.lbrw_have) begin
            for (de=0; de<N_EXPERT; de=de+1)
                if (dut.die_rw_q[4*de +: 4] !== f_rwq(db_layer, de, rw_k)) begin
                    $display("FAIL[direct-rw]: die_rw_q lane %0d=%h != f_rwq(ly=%0d,e=%0d,k=%0d)=%h @%0t",
                             de, dut.die_rw_q[4*de +: 4], db_layer, de, rw_k,
                             f_rwq(db_layer, de, rw_k), $time);
                    rw_derr = rw_derr + 1;
                end
            rw_dchk = rw_dchk + 1;
        end
        // lw : LM_TN positional bf16 column lanes, live key {lw_vtile, lw_k}
        if (lw_req && dut.g_lbrest.lblw_have) begin
            for (dtt=0; dtt<LM_TN; dtt=dtt+1)
                if (dut.die_lw_col[16*dtt +: 16] !== gen_bf16((lw_vtile*LM_TN+dtt)*MODEL_DIM + lw_k + 7603)) begin
                    $display("FAIL[direct-lw]: die_lw_col lane %0d=%h != f_lw(vt=%0d,t=%0d,k=%0d)=%h @%0t",
                             dtt, dut.die_lw_col[16*dtt +: 16], lw_vtile, dtt, lw_k,
                             gen_bf16((lw_vtile*LM_TN+dtt)*MODEL_DIM + lw_k + 7603), $time);
                    lw_derr = lw_derr + 1;
                end
            lw_dchk = lw_dchk + 1;
        end
        // gn : a single bf16 gain/norm, live key {db_layer, gn_which, gn_idx}
        if (gn_req && dut.g_lbrest.lbgn_have) begin
            if (dut.die_gn_val !== gen_bf16(db_layer*1024 + gn_which*512 + gn_idx + 7411)) begin
                $display("FAIL[direct-gn]: die_gn_val=%h != f_gn(ly=%0d,which=%0d,idx=%0d)=%h @%0t",
                         dut.die_gn_val, db_layer, gn_which, gn_idx,
                         gen_bf16(db_layer*1024 + gn_which*512 + gn_idx + 7411), $time);
                gn_derr = gn_derr + 1;
            end
            gn_dchk = gn_dchk + 1;
        end
    end

    // ================= per-token bit-exactness check =================
    integer test_count;

    task settle; input integer n; integer c; begin
        for (c=0;c<n;c=c+1) @(negedge clk);
    end endtask

    task run_token; input [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        input [256*8-1:0] label; integer b; begin
        prompt_tok = tk; start_pos = ps; s_len = SL[IDXW:0];
        @(negedge clk); start = 1'b1; r_start = 1'b1;
        @(negedge clk); start = 1'b0; r_start = 1'b0;
        wait (tok_valid === 1'b1);
        @(negedge clk);
        wait (r_done_seen === 1'b1);
        settle(400);   // drain FIFO/cache/Flash + xbar in-flight reads

        test_count = test_count + 1;

        // (a) BINDING: LOOPBACK_REST=1 system committed token == standalone reference
        if (next_tok !== r_tok_lat) begin
            $display("FAIL[%0s]: BINDING next_tok %0d != standalone ref %0d (LOOPBACK_REST=1 diverged)",
                     label, next_tok, r_tok_lat);
            errors=errors+1; end
        // (a') X-cleanliness + internal-argmax consistency
        for (b=0;b<TOKW;b=b+1) if (next_tok[b]===1'bx || next_tok[b]===1'bz) begin
            $display("FAIL[%0s]: next_tok bit %0d X/Z", label, b); errors=errors+1; end
        for (b=0;b<VOCAB*16;b=b+1) if (logits[b]===1'bx || logits[b]===1'bz) begin
            $display("FAIL[%0s]: logits bit %0d X/Z", label, b); errors=errors+1; b=VOCAB*16; end
        if (next_tok !== argmax_o) begin
            $display("FAIL[%0s]: next_tok %0d != system internal argmax %0d", label, next_tok, argmax_o);
            errors=errors+1; end
        // (b) the loopback fabric path was actually exercised for this token
        if (xbar_req_count == 32'd0 || xbar_resp_count == 32'd0) begin
            $display("FAIL[%0s]: ddr5_xbar carried no banked read (req=%0d resp=%0d)",
                     label, xbar_req_count, xbar_resp_count); errors=errors+1; end

        if (xmon_err != 0) begin
            $display("FAIL[%0s]: %0d xbar/req monitor violations", label, xmon_err);
            errors=errors+1; end

        $display("PASS[%0s] tok=%0d(==ref %0d) | xbar req/resp=%0d/%0d | loopback rw=%0d/%0d lw=%0d/%0d gn=%0d/%0d",
                 label, next_tok, r_tok_lat, xbar_req_count, xbar_resp_count,
                 lbrw_req_seen, lbrw_resp_seen, lblw_req_seen, lblw_resp_seen,
                 lbgn_req_seen, lbgn_resp_seen);
    end endtask

    initial begin
        #400000000;
        $display("FAIL: global timeout"); $fatal;
    end

    initial begin
        errors=0; test_count=0; xmon_err=0;
        rw_derr=0; lw_derr=0; gn_derr=0; rw_dchk=0; lw_dchk=0; gn_dchk=0;
        rst=1'b1; start=1'b0; r_start=1'b0;
        prompt_tok={TOKW{1'b0}}; start_pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}};
        wl_mem_data={WL_DATA_W{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // ---- small decode sequence: cold token 0, then feed next_tok back ----
        //   (pos == s_len, the real-decode relation; s_len grows 1..S_MAX)
        run_token(4'd7,     20'd1, 1, "tok0 cold s1");
        run_token(next_tok, 20'd2, 2, "tok1 s2");
        run_token(next_tok, 20'd3, 3, "tok2 s3");
        run_token(next_tok, 20'd4, 4, "tok3 s4");

        // ---- PER-FAMILY soundness gates: EACH of rw/lw/gn MUST have round-tripped ----
        //   (proves the bindings above actually flowed EACH family's bytes through the
        //   real ddr5_xbar, not a bypass -- no vacuous per-family pass).
        test_count = test_count + 1;
        if (lbrw_req_seen == 0 || lbrw_resp_seen == 0) begin
            $display("FAIL: rw loopback path never exercised (rw_req=%0d rw_resp=%0d) -- vacuous",
                     lbrw_req_seen, lbrw_resp_seen);
            errors=errors+1;
        end else
            $display("PASS[rw-loopback-exercised] rw_reads=%0d rw_resps=%0d routed through ddr5_xbar",
                     lbrw_req_seen, lbrw_resp_seen);

        test_count = test_count + 1;
        if (lblw_req_seen == 0 || lblw_resp_seen == 0) begin
            $display("FAIL: lw loopback path never exercised (lw_req=%0d lw_resp=%0d) -- vacuous",
                     lblw_req_seen, lblw_resp_seen);
            errors=errors+1;
        end else
            $display("PASS[lw-loopback-exercised] lw_reads=%0d lw_resps=%0d routed through ddr5_xbar",
                     lblw_req_seen, lblw_resp_seen);

        test_count = test_count + 1;
        if (lbgn_req_seen == 0 || lbgn_resp_seen == 0) begin
            $display("FAIL: gn loopback path never exercised (gn_req=%0d gn_resp=%0d) -- vacuous",
                     lbgn_req_seen, lbgn_resp_seen);
            errors=errors+1;
        end else
            $display("PASS[gn-loopback-exercised] gn_reads=%0d gn_resps=%0d routed through ddr5_xbar",
                     lbgn_req_seen, lbgn_resp_seen);

        // ---- DIRECT PER-BEAT BYTE-BINDING gates: EACH family's presented die input ----
        //   matched its expected value on EVERY sampled beat (rw_derr/lw_derr/gn_derr==0)
        //   AND was actually sampled at least once (rw_dchk/lw_dchk/gn_dchk>0 -- no vacuous
        //   pass).  This is the load-bearing proof for the output-insensitive rw/gn: the
        //   fabric delivered the CORRECT bytes to the die, not merely a bit-exact token.
        test_count = test_count + 1;
        if (rw_dchk == 0) begin
            $display("FAIL: rw DIRECT byte binding never sampled -- vacuous (die never presented a looped rw beat)");
            errors=errors+1;
        end else if (rw_derr != 0) begin
            $display("FAIL: rw DIRECT byte binding mismatched on %0d of %0d sampled beats (die_rw_q != f_rwq for live key)",
                     rw_derr, rw_dchk);
            errors=errors+1;
        end else
            $display("PASS[rw-direct-byte-binding] die_rw_q == f_rwq(db_layer,e,rw_k) on ALL %0d sampled beats",
                     rw_dchk);

        test_count = test_count + 1;
        if (lw_dchk == 0) begin
            $display("FAIL: lw DIRECT byte binding never sampled -- vacuous (die never presented a looped lw beat)");
            errors=errors+1;
        end else if (lw_derr != 0) begin
            $display("FAIL: lw DIRECT byte binding mismatched on %0d of %0d sampled beats (die_lw_col != f_lw for live key)",
                     lw_derr, lw_dchk);
            errors=errors+1;
        end else
            $display("PASS[lw-direct-byte-binding] die_lw_col == f_lw(lw_vtile,t,lw_k) on ALL %0d sampled beats",
                     lw_dchk);

        test_count = test_count + 1;
        if (gn_dchk == 0) begin
            $display("FAIL: gn DIRECT byte binding never sampled -- vacuous (die never presented a looped gn beat)");
            errors=errors+1;
        end else if (gn_derr != 0) begin
            $display("FAIL: gn DIRECT byte binding mismatched on %0d of %0d sampled beats (die_gn_val != f_gn for live key)",
                     gn_derr, gn_dchk);
            errors=errors+1;
        end else
            $display("PASS[gn-direct-byte-binding] die_gn_val == f_gn(db_layer,gn_which,gn_idx) on ALL %0d sampled beats",
                     gn_dchk);

        if (errors!=0) begin
            $display("FAILED: %0d error(s) across %0d tests (LOOPBACK_REST=1 not bit-exact to reference)",
                     errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED  (glm_q4k_system LOOPBACK_REST=1: die rw/lw/gn inputs routed OUT and back IN through ddr5_xbar; committed stream BIT-EXACT to standalone glm_model_q4k reference over a 4-token decode, AND the die's presented die_rw_q/die_lw_col/die_gn_val matched the expected bytes for the live key on EVERY beat -- direct per-beat byte binding)",
                 test_count);
        $finish;
    end
endmodule
