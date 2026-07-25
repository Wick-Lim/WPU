`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_q4k_system_cdc.v  --  TWO-CLOCK (multi-domain) WRAPPER around the verified
//                           single-clock glm_q4k_system GLM-5.2 Q4_K box.
//                           (docs/Q4K_SYSTEM_PLAN.md §1.4 -- the synth-glm-q4k
//                            whole-chip sign-off top.)
//----------------------------------------------------------------------------
// Q4_K RETARGET (vs. the prior glm_fp8_system_cdc on tag 'fp8-verified-baseline'):  the compute
//   box u_core swaps
//   glm_fp8_system -> glm_q4k_system (ONE contract change).  This CDC wrapper
//   only FORWARDS buses across the async-FIFO boundary, so the retarget is a
//   pure WIDTH/FORMAT re-port of the weight-side ports it carries through:
//     aw_col[PE_N*8]/aw_scale[16*PE_N*A_NB] -> aw_q[PE_N*4] +
//         aw_d/aw_dmin[16*PE_N*A_NSB] + aw_scales[96*PE_N*A_NSB]
//     rw_col[8*N_EXPERT]/rw_scale[16*N_EXPERT*R_NB] -> rw_q[4*N_EXPERT] +
//         rw_d/rw_dmin[16*N_EXPERT*R_NSB] + rw_scales[96*N_EXPERT*R_NSB]
//     fw_col*/fw_scale_g/fw_scale_u -> fw_q*/fw_d_*/fw_dmin_*/fw_scales_*
//   The 128-wide FP8 block counts A_NB/FF_NB_D/R_NB become 256-wide Q4_K
//   super-block counts A_NSB/FF_NSB_D/R_NSB (ceil(K/256)).  The loader-code
//   observability lane shrinks 8-bit -> 4-bit (loader_w_row -> loader_w_q) and
//   the staging word WL_DATA_W is 256 (== the ddr5 beat).  EVERY CDC element
//   (the async FIFOs, the two-clock handshake, the reset synchronizers) is
//   FORMAT-AGNOSTIC -- it moves the host request / produced token / busy /
//   done, never weight bytes -- so it retargets by WIDTH PARAMETER ONLY, with
//   ZERO logic change.  No arithmetic is reimplemented.
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   A real GLM-5.2 chip is NOT single-clock: the USB-C device controller that
//   talks to the host runs on the USB SerDes recovered clock, while the compute
//   die (glm_q4k_system: glm_model_q4k + expert_cache_pf + kv_cache_pager +
//   ddr5_xbar + weight_loader_q4k) runs on its own core clock.  The two clocks
//   are ASYNCHRONOUS -- unrelated frequency and phase.  This wrapper keeps the
//   verified compute box UNCHANGED, instantiates it entirely on core_clk, and
//   presents the SAME host-facing interface (start / prompt_tok / start_pos /
//   s_len -> busy / done / next_tok / tok_valid) but now sampled on host_clk.
//   EVERY signal that crosses between the two domains does so ONLY through a
//   cdc_async_fifo (gray-coded pointers + 2-FF synchronizers) or an explicit
//   single-bit 2-FF synchronizer.  NO raw multi-bit value is ever sampled
//   directly across the boundary; there is NO combinational path between the
//   two domains.
//
//============================================================================
// 2-CLOCK BLOCK DIAGRAM  (which signal crosses which way, through what)
//
//   ┌──────────────── host_clk domain (USB-C device) ───────────────┐
//   │  start ─(rising-edge)─┐                                        │
//   │  prompt_tok ┐         │   pack {prompt_tok,start_pos,s_len}    │
//   │  start_pos  ┼─────────┴──▶ REQUEST cdc_async_fifo  ───────────┐│  host_clk ─▶ core_clk
//   │  s_len      ┘             (gray ptrs + 2-FF sync)             ││  (multi-bit, FIFO)
//   │                                                               ││
//   │  busy  ◀── (host_pending | 2-FF sync of sys_busy) ◀───────────┼┼─ sys_busy   (1-bit, 2-FF)
//   │  done  ◀── (edge-detect of 2-FF-synced done TOGGLE) ◀─────────┼┼─ done_tgl_c (1-bit, 2-FF)
//   │  next_tok ◀┐                                                  ││
//   │  tok_valid◀┴── TOKEN cdc_async_fifo read side ◀───────────────┼┼─ sys_next_tok (multi-bit,
//   │                (gray ptrs + 2-FF sync)                        ││   pushed on sys_tok_valid)
//   └───────────────────────────────────────────────────────────────┘│  core_clk ─▶ host_clk
//                                                                      │
//   ┌──────────────── core_clk domain (compute die) ─────────────────┐│
//   │  REQUEST fifo read ─▶ unpack ─▶ 1-cycle sys_start pulse ─▶┐     ││
//   │                                                          ▼     ││
//   │   glm_q4k_system  (UNCHANGED, clk=core_clk, rst=core_rst)       ││
//   │     .busy=sys_busy  .done=sys_done                             ││
//   │     .tok_valid=sys_tok_valid  .next_tok=sys_next_tok           ││
//   │     ...all weight/KV/Flash/DDR5/loader/observability ports.....││─▶ wrapper memory-side
//   │   sys_tok_valid ─▶ TOKEN fifo write side                       ││   ports (core_clk domain)
//   │   sys_done ─▶ done_tgl_c (toggle)                              ││
//   └───────────────────────────────────────────────────────────────┘
//
//   CROSSINGS (every one is gray-FIFO or 2-FF -- NO raw multi-bit crossing):
//     host->core : {prompt_tok,start_pos,s_len}  via REQUEST cdc_async_fifo
//     core->host : next_tok                       via TOKEN   cdc_async_fifo
//     core->host : busy (level)                   via 2-FF synchronizer
//     core->host : done (pulse -> TOGGLE)         via 2-FF synchronizer + edge det
//
//   The memory-side ports (GDDR6/Flash/DDR5/loader/observability) belong wholly to
//   the core_clk domain -- glm_q4k_system runs entirely on core_clk -- so they are
//   passed straight through to the wrapper ports with NO crossing (the host never
//   touches them; in a full SoC their own controllers live in the memory clocks).
//
// STYLE: synchronous ACTIVE-HIGH resets per domain; NO latch (every reg assigned on
//   every path); NO combinational loop; NO combinational path between domains.
//============================================================================
module glm_q4k_system_cdc #(
    // ---- compute-die (glm_model_q4k) slice config -- passed straight through --
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,
    parameter integer N_DENSE    = 3,
    parameter integer VOCAB      = 256,
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    // SWIN: attention union-slot scratch depth, pure pass-through to glm_q4k_system
    //   (which threads it into u_model).  Default = min(S_MAX,TOPK_ATTN), byte-identical
    //   at PE_M=1; forwarded so the wrapper cannot pin the batch's scratch (see below).
    parameter integer SWIN       = (S_MAX < TOPK_ATTN) ? S_MAX : TOPK_ATTN,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    parameter integer ACT_HW     = 0,   // activation HW lanes (0 = full) -- result-invariant knob
    // ---- PE_M / PER_ROW_POS / PER_ROW_SLEN : pure pass-through to glm_q4k_system.
    //   Default PE_M=1 (no speculation) == the committed CDC wrapper, byte-identical.
    //   Forwarded (like DSA_REAL_IDX) so the CDC wrapper cannot SILENTLY PIN the die at
    //   PE_M=1 -- the spec composition (SPEC_COMPOSITION_DESIGN.md step 5) drives the die
    //   at PE_M=K+1, and a hard-wired wrapper would defeat that.  No CDC element sees the
    //   knob; the die's H_IDLE-aligned handshake is PE_M-independent at the CDC boundary.
    parameter integer PE_M       = 1,
    parameter integer PER_ROW_POS = 0,  // 1 = per-row query positions  (pass-through)
    parameter integer PER_ROW_SLEN= 0,  // 1 = per-row causal extents   (pass-through)
    parameter integer DSA_REAL_IDX = 0, // query-dependent DSA top-K (see glm_q4k_system.v)
    // INTRA_CAUSAL: pure pass-through to glm_q4k_system (like PE_M/DSA_REAL_IDX).  Default
    //   0 == the committed CDC wrapper, byte-identical -- no CDC element sees the knob;
    //   the die's H_IDLE-aligned handshake is INTRA_CAUSAL-independent at the boundary.
    //   Forwarded so the wrapper cannot SILENTLY PIN the die at INTRA_CAUSAL=0, the same
    //   dead-port hazard the DSA thread-through closed (SPEC_COMPOSITION_DESIGN.md 5b-sys).
    parameter integer INTRA_CAUSAL = 0,
    // ---- memory-system config ----
    parameter integer CACHE_SLOTS = 4,
    parameter integer FLASH_LAT   = 8,
    parameter integer KV_CTX      = 1024,
    parameter integer KV_RESIDENT = 16,
    parameter integer EFIFO_DEPTH = 16,
    // RESIDENT weight tier (0 = OFF, DEFAULT -- byte-identical chip top).  1 =
    // the full weight image is DDR-tier resident: expert refills are served by
    // a real banked ddr5_xbar read (TAG_EFILL) instead of the Flash channel;
    // runtime decode never touches Flash for weights (KV spill + boot keep it).
    // Pure pass-through to glm_q4k_system -- no CDC element sees the knob.
    parameter integer RESIDENT    = 0,
    // ---- DDR5 fast-tier fabric (ddr5_xbar) config ----
    parameter integer DDR_NCH     = 4,
    parameter integer DDR_ADDR_W  = 32,
    parameter integer DDR_DATA_W  = 256,
    parameter integer DDR_TAG_W   = 8,
    parameter integer DDR_ROW_LAT = 10,
    parameter integer DDR_RESP_QD = 4,
    // ---- weight_loader_q4k (matmul weight-pull DMA) config ----
    parameter integer WL_KMAX     = 256,
    parameter integer WL_ADDR_W   = 24,
    parameter integer LOADER_KLEN = MODEL_DIM,
    // ---- CDC FIFO depths (this wrapper) ----
    parameter integer REQ_AW      = 2,      // request FIFO addr width (depth 2**AW)
    parameter integer TOK_AW      = 3,      // token   FIFO addr width (depth 2**AW)
    // ---- PROTOCOL EXTENSION (USAGE_GAPS §C, findings #19/#26) ------------
    // DEFAULT OFF (PROTO_CTX=0): the host<->device wire format is EXACTLY
    //   {prompt_tok,start_pos,s_len} -> {busy,done,tok_valid,next_tok}, byte-
    //   identical to the shipped top (the new ports are dead: req_ctx_id/
    //   req_opcode ignored, resp_* tied to 0; both CDC FIFOs keep their
    //   original DATA_W; every new gate constant-folds away).
    // PROTO_CTX=1 turns on the multiplexable protocol WITHOUT touching any
    //   arithmetic or CDC primitive:
    //     * the REQUEST carries a CONTEXT/SEQUENCE id + a 2-bit OPCODE
    //       (OP_TOKEN=token-gen, OP_TELEM=telemetry readback), packed in the
    //       HIGH bits of the SAME request FIFO word (low bits unchanged), so
    //       the host can multiplex N contexts;
    //     * the emitted-token RESPONSE carries the run's ctx id back so the
    //       host can DEMUX returned tokens to their context;
    //     * an OP_TELEM request pops a snapshot of registered core-domain
    //       counters (tokens / runs started / runs done / demand-stall cycles)
    //       back through the SAME response FIFO tagged resp_is_telem, so a host
    //       can poll device state.  The ctx id and counters cross domains ONLY
    //       through the existing cdc_async_fifo pattern -- no new raw crossing.
    parameter integer PROTO_CTX   = 0,      // 0 = OFF (byte-identical). 1 = ctx-id + telemetry.
    parameter integer CTX_W       = 8,      // context/sequence id width (host multiplexes 2**CTX_W ctx)
    parameter integer TELEM_W     = 32,     // per-counter width in the telemetry readback
    // ====================================================================
    // derived (do NOT override) -- mirror glm_q4k_system's port-width derivations
    // ====================================================================
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer A_KMAX     = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    parameter integer A_OMAX     = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer A_NGMAX    = (A_OMAX + PE_N - 1) / PE_N,
    parameter integer A_GRPW     = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX),
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1),
    // ---- Q4_K super-block counts (256-elem super-blocks; #super-blocks per K) ----
    //   ceil(K/256), mirroring glm_q4k_system -- these size the d/dmin/scales
    //   buses (was A_NB/FF_NB_D/R_NB = ceil(K/128) prior-FP8 block counts).
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer WL_PE_N    = PE_N,
    // Q4_K staging word: super-block header packs into [127:0] (d/dmin/scales)
    // and a nibble-code beat into [4*PE_N-1:0]; DATA_W stays 256 (== ddr5 beat).
    parameter integer WL_DATA_W  = 256,
    // ---- this wrapper's packed-request width ----
    parameter integer REQ_W      = TOKW + POSW + (IDXW+1)
)(
    //========================== TWO ASYNCHRONOUS CLOCK DOMAINS ===============
    input  wire                          host_clk,   // USB-C device domain
    input  wire                          host_rst,   // sync, active-high (host)
    input  wire                          core_clk,   // compute-die domain
    input  wire                          core_rst,   // sync, active-high (core)

    //========================== HOST interface (sampled on host_clk) ========
    input  wire                          start,
    input  wire [TOKW-1:0]               prompt_tok,
    input  wire [POSW-1:0]               start_pos,
    input  wire [IDXW:0]                 s_len,
    output reg                           busy,
    output reg                           done,
    output reg  [TOKW-1:0]               next_tok,
    output reg                           tok_valid,

    //===== PROTOCOL EXTENSION (PROTO_CTX=1) -- context id + telemetry readback ==
    //  host_clk domain.  DEFAULT OFF: req_ctx_id/req_opcode are ignored and
    //  resp_ctx_id/resp_is_telem/resp_telem are driven to constant 0 (dead
    //  logic -> byte-identical default netlist).  When PROTO_CTX=1:
    //    req_ctx_id/req_opcode   : sampled with `start` (aligned to the request)
    //    resp_ctx_id             : ctx id of the response, valid with tok_valid
    //    resp_is_telem           : 1 => this response is a telemetry snapshot
    //                              (resp_telem valid, next_tok is don't-care),
    //                              0 => this response is a generated token.
    //    resp_telem              : packed counters {stall,done,runs,tokens},
    //                              each TELEM_W wide (LSB-first: tokens in [0]).
    input  wire [CTX_W-1:0]              req_ctx_id,
    input  wire [1:0]                    req_opcode,
    output wire [CTX_W-1:0]              resp_ctx_id,
    output wire                          resp_is_telem,
    output wire [4*TELEM_W-1:0]          resp_telem,

    //====== everything below is the core_clk domain, passed straight through ===
    output wire [VOCAB*16-1:0]           logits,

    //========================== GDDR6 HOT-weight STUBS ======================
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    // ---- attention weights (Q4_K: 4-bit codes + per-super-block d/dmin/scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*4-1:0]             aw_q,
    input  wire [16*PE_N*A_NSB-1:0]      aw_d,
    input  wire [16*PE_N*A_NSB-1:0]      aw_dmin,
    input  wire [96*PE_N*A_NSB-1:0]      aw_scales,
    // ---- MoE router W_g (Q4_K) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [4*N_EXPERT-1:0]         rw_q,
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_d,
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_dmin,
    input  wire [96*N_EXPERT*R_NSB-1:0]  rw_scales,
    // ---- FFN expert weights (Q4_K) ----
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [4*TN-1:0]               fw_q,
    input  wire [4*TN-1:0]               fw_q_up,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_g,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_g,
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_g,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_u,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_u,
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_u,
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,

    //========================== KV append (latent ROW source) ===============
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,

    //========================== SINGLE FLASH CHANNEL (to PHY/TB) ============
    output wire                          flash_req,
    output wire                          flash_is_expert,
    output wire [EIDXW-1:0]              flash_expert_id,
    output wire [KVPOSW-1:0]             flash_row_idx,
    input  wire                          flash_done,
    input  wire [ROW_BITS-1:0]           flash_row,

    //========================== expert prefetch hint (optional) =============
    input  wire                          pf_valid,
    input  wire [EIDXW-1:0]              pf_expert_id,

    //========================== DDR5 fabric channels ========================
    output wire [DDR_NCH-1:0]            mem_req_valid,
    input  wire [DDR_NCH-1:0]            mem_req_ready,
    output wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr,
    output wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag,
    input  wire [DDR_NCH-1:0]            mem_resp_valid,
    output wire [DDR_NCH-1:0]            mem_resp_ready,
    input  wire [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data,
    input  wire [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag,

    //========================== weight_loader staging memory ================
    output wire                          wl_mem_en,
    output wire [WL_ADDR_W-1:0]          wl_mem_addr,
    input  wire [WL_DATA_W-1:0]          wl_mem_data,

    //========================== observability ===============================
    output wire [TOKW-1:0]               argmax_o,
    output wire [MODEL_DIM*16-1:0]       h_state,
    output wire                          mdl_busy,
    output wire                          ec_resp_valid,
    output wire                          ec_hit,
    output wire [CSLOTW-1:0]             ec_resp_slot,
    output wire                          ec_busy,
    output wire [31:0]                   ec_hit_count,
    output wire [31:0]                   ec_miss_count,
    output wire [31:0]                   ec_demand_stall_cycles,
    output wire [31:0]                   ec_pf_issued,
    output wire [31:0]                   ec_pf_hit,
    output wire                          kv_row_valid,
    output wire [ROW_BITS-1:0]           kv_row_out,
    output wire                          kv_busy,
    output wire [KVPOSW-1:0]             kv_append_count,
    output wire [KVPOSW-1:0]             kv_resident_lo,
    output wire                          kv_overflowed,
    output wire [31:0]                   ec_dropped,
    output wire [31:0]                   xbar_req_count,
    output wire [31:0]                   xbar_resp_count,
    output wire                          xbar_resp_valid,
    output wire [DDR_DATA_W-1:0]         xbar_resp_data,
    output wire                          loader_busy,
    output wire [31:0]                   loader_done_count,
    output wire [31:0]                   loader_beat_count,
    output wire [4*WL_PE_N-1:0]          loader_w_q,
    output wire                          loader_in_valid
);
    // ======================================================================
    // PER-DOMAIN RESET SYNCHRONIZERS (async-assert / sync-deassert).
    //   The top-level host_rst / core_rst inputs are treated as ASYNC reset
    //   REQUESTS (a real chip's power-on / SoC reset is asynchronous to each
    //   recovered/PLL clock).  Each is fed through a reset_sync in ITS OWN
    //   destination clock domain so that:
    //     * ASSERT is asynchronous  -- the domain drops into reset the instant
    //       the request goes high, even before its clock toggles;
    //     * DEASSERT is synchronous -- the release walks STAGES flops on the
    //       destination clock, so the reset the domain sees deasserts cleanly
    //       (no reset-recovery metastability).
    //   Active-HIGH polarity is preserved (die-wide convention).  Everything
    //   internal to this wrapper uses the SYNCHRONIZED resets below; the raw
    //   input pins feed ONLY the synchronizers.
    // ======================================================================
    localparam integer RST_SYNC_STAGES = 2;

    // ======================================================================
    // PROTOCOL-EXTENSION derived widths (all fold to the ORIGINAL widths when
    // PROTO_CTX=0, so the default FIFO/instance parameters are unchanged).
    //   REQ_W_EFF  : request FIFO word  = {ctx,opcode, prompt_tok,pos,s_len}
    //   RESP_W_EFF : response FIFO word = {is_telem, ctx, payload}
    //                payload holds EITHER a token (low TOKW bits) OR the packed
    //                telemetry counters (NCTR*TELEM_W), whichever is wider.
    // ======================================================================
    localparam [1:0]   OP_TOKEN   = 2'd0;                 // token-generation request
    localparam [1:0]   OP_TELEM   = 2'd1;                 // telemetry-readback request
    localparam integer REQ_W_EFF  = PROTO_CTX ? (REQ_W + CTX_W + 2) : REQ_W;
    localparam integer NCTR       = 4;                    // tokens, runs, done, stall
    localparam integer TELEM_PAY  = NCTR * TELEM_W;       // packed counter payload
    localparam integer RESP_PAY   = (TOKW > TELEM_PAY) ? TOKW : TELEM_PAY;
    localparam integer RESP_W_EFF = PROTO_CTX ? (1 + CTX_W + RESP_PAY) : TOKW;

    wire host_rst_sync;   // host_clk-domain reset, active-high, sync-deasserted
    wire core_rst_sync;   // core_clk-domain reset, active-high, sync-deasserted

    reset_sync #(.STAGES(RST_SYNC_STAGES)) u_host_rst_sync (
        .clk    (host_clk),
        .arst_in(host_rst),      // async reset request -> host_clk domain
        .rst_out(host_rst_sync)
    );
    reset_sync #(.STAGES(RST_SYNC_STAGES)) u_core_rst_sync (
        .clk    (core_clk),
        .arst_in(core_rst),      // async reset request -> core_clk domain
        .rst_out(core_rst_sync)
    );

    // ======================================================================
    // Per-domain active-LOW resets for the cdc_async_fifo primitive (its reset
    // convention is active-low, sampled synchronously in each clock).  Derived
    // from the SYNCHRONIZED per-domain resets above.
    // ======================================================================
    wire host_rst_n = ~host_rst_sync;
    wire core_rst_n = ~core_rst_sync;

    // ======================================================================
    // ============================ host_clk DOMAIN =========================
    //  Request push : pack {prompt_tok,start_pos,s_len}, push on a rising edge of
    //  `start` (one request per assertion) when the request FIFO is not full.
    // ======================================================================
    reg  start_d;
    always @(posedge host_clk) begin
        if (host_rst_sync) start_d <= 1'b0;
        else               start_d <= start;
    end
    wire start_rise = start & ~start_d;

    wire                 req_wr_full;
    wire                 req_wr_en   = start_rise & ~req_wr_full;
    // request word: ctx/opcode ride in the HIGH bits, so the low REQ_W bits are
    // BIT-IDENTICAL to today's {prompt_tok,start_pos,s_len} and the core-side
    // unpack (which reads the low bits) is unchanged.
    wire [REQ_W_EFF-1:0] req_wr_data;
    generate
        if (PROTO_CTX) begin : g_req_pack
            assign req_wr_data = {req_ctx_id, req_opcode, prompt_tok, start_pos, s_len};
        end else begin : g_req_pack_off
            assign req_wr_data = {prompt_tok, start_pos, s_len};
        end
    endgenerate

    // ======================================================================
    // REQUEST cdc_async_fifo : host_clk (write) -> core_clk (read).
    //  The ONLY path the multi-bit host request takes across the boundary.
    // ======================================================================
    wire                 req_rd_empty;
    wire [REQ_W_EFF-1:0] req_rd_data;
    wire                 req_rd_en;

    cdc_async_fifo #(
        .DATA_W (REQ_W_EFF),
        .ADDR_W (REQ_AW)
    ) u_req_fifo (
        .wclk   (host_clk), .wrst_n (host_rst_n),
        .wr_en  (req_wr_en), .wr_data(req_wr_data), .full(req_wr_full),
        .rclk   (core_clk), .rrst_n (core_rst_n),
        .rd_en  (req_rd_en), .rd_data(req_rd_data), .empty(req_rd_empty)
    );

    // ======================================================================
    // ============================ core_clk DOMAIN =========================
    //  Request pop : single-outstanding pop.  The cdc_async_fifo read is
    //  REGISTERED, so a word popped this cycle is valid on req_rd_data NEXT cycle
    //  (req_rd_d).  Unpack it into holding regs and pulse sys_start for exactly
    //  one core_clk -- start + fields are driven from the SAME edge so they are
    //  aligned at the glm_q4k_system input (which samples start in its H_IDLE).
    // ======================================================================
    reg               req_rd_d;
    reg               sys_start;
    reg  [TOKW-1:0]   sys_prompt_tok;
    reg  [POSW-1:0]   sys_start_pos;
    reg  [IDXW:0]     sys_s_len;

    // ---- launch queueing (2026-07-22) ------------------------------------
    // The box (glm_q4k_system) samples `start` ONLY in H_IDLE and consumes
    // prompt_tok/start_pos/s_len LIVE for the whole run.  Two consequences:
    //   (a) a launch popped while a run is in flight must be HELD, not pulsed
    //       (the pulse lands outside H_IDLE and the request silently vanishes
    //       -- no token, no done; the depth-4 request FIFO invited exactly
    //       that pipelining), and
    //   (b) the holding regs must NOT be rewritten by later pops mid-run (an
    //       OP_TELEM pop rewriting prompt_tok would corrupt the live run).
    // run_inflight tracks fire..done.  launch_pend holds ONE queued launch and
    // blocks further pops until it fires; the FIFO buffers the rest, so no
    // request is ever dropped.  Mid-run OP_TELEM pops still flow freely while
    // no launch is queued (the common poll case).
    reg               run_inflight;
    reg               launch_pend;
    reg  [REQ_W-1:0]  pend_req;    // held launch's fields -- staged SEPARATELY so the
                                   // live run's sys_* regs are never rewritten mid-run
    wire              sys_busy;
    wire              sys_done;

    assign req_rd_en = ~req_rd_empty & ~req_rd_d & ~launch_pend;

    // Decode of the popped request word (core_clk domain).  When PROTO_CTX=0
    // these fold to constants (req_launch==1, req_is_telem==0) so the unpack
    // below is functionally identical to today's unconditional sys_start pulse.
    wire              req_launch;      // launch the compute box for this request
    wire              req_is_telem;    // this request is a telemetry readback
    wire [CTX_W-1:0]  req_ctx_popped;  // ctx id carried by this request
    generate
        if (PROTO_CTX) begin : g_req_dec
            assign req_ctx_popped = req_rd_data[REQ_W_EFF-1 -: CTX_W];
            assign req_is_telem   = (req_rd_data[REQ_W +: 2] == OP_TELEM);
            assign req_launch     = ~req_is_telem;
        end else begin : g_req_dec_off
            assign req_ctx_popped = {CTX_W{1'b0}};
            assign req_is_telem   = 1'b0;
            assign req_launch     = 1'b1;
        end
    endgenerate

    always @(posedge core_clk) begin
        if (core_rst_sync) begin
            req_rd_d       <= 1'b0;
            sys_start      <= 1'b0;
            sys_prompt_tok <= {TOKW{1'b0}};
            sys_start_pos  <= {POSW{1'b0}};
            sys_s_len      <= {(IDXW+1){1'b0}};
            run_inflight   <= 1'b0;
            launch_pend    <= 1'b0;
            pend_req       <= {REQ_W{1'b0}};
        end else begin
            sys_start <= 1'b0;
            req_rd_d  <= req_rd_en;
            if (sys_done) run_inflight <= 1'b0;
            if (req_rd_d & req_launch) begin
                if (~run_inflight) begin
                    // LHS is REQ_W bits wide -> keeps the LOW REQ_W bits of the
                    // (possibly wider) word == {prompt_tok,start_pos,s_len}.
                    // Written ONLY when the launch fires: the box reads these
                    // live all run, so neither an OP_TELEM pop nor a QUEUED
                    // launch pop may touch them while a run is in flight.
                    {sys_prompt_tok, sys_start_pos, sys_s_len} <= req_rd_data;
                    sys_start    <= 1'b1;              // box idle -> fire now
                    run_inflight <= 1'b1;
                end else begin
                    pend_req    <= req_rd_data[REQ_W-1:0]; // box busy -> hold (was: silently lost)
                    launch_pend <= 1'b1;
                end
            end else if (launch_pend & ~run_inflight) begin
                {sys_prompt_tok, sys_start_pos, sys_s_len} <= pend_req;
                sys_start    <= 1'b1;                  // fire the held launch
                run_inflight <= 1'b1;
                launch_pend  <= 1'b0;
            end
        end
    end

    // ----------------------------------------------------------------------
    // THE VERIFIED COMPUTE BOX -- instantiated UNCHANGED, entirely on core_clk.
    // ----------------------------------------------------------------------
    wire [TOKW-1:0]  sys_next_tok;      // (sys_busy/sys_done declared above, at the pop logic)
    wire             sys_tok_valid;

    glm_q4k_system #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .SWIN(SWIN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT),
        .TOPK(TOPK), .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE),
        .TN(TN), .BLK(BLK), .LM_TN(LM_TN), .PE_M(PE_M), .ACT_HW(ACT_HW),
        .PER_ROW_POS(PER_ROW_POS), .PER_ROW_SLEN(PER_ROW_SLEN),
        .DSA_REAL_IDX(DSA_REAL_IDX), .INTRA_CAUSAL(INTRA_CAUSAL),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH), .RESIDENT(RESIDENT),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN)
    ) u_sys (
        .clk(core_clk), .rst(core_rst_sync),
        // host port -- now driven from the core-side request unpack / captured back
        .start(sys_start), .prompt_tok(sys_prompt_tok),
        .start_pos(sys_start_pos), .s_len(sys_s_len),
        .busy(sys_busy), .done(sys_done),
        .next_tok(sys_next_tok), .tok_valid(sys_tok_valid),
        .logits(logits),
        // ---- everything else: pure core-domain pass-through ----
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
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(wl_mem_data),
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

    // ======================================================================
    // TELEMETRY counters + RESPONSE formation (core_clk domain).
    //   PROTO_CTX=0: resp_wr_en/resp_wr_data == today's tok_wr_en/sys_next_tok
    //     (RESP_W_EFF==TOKW) -> the response FIFO is byte-identical.
    //   PROTO_CTX=1: registered counters advance on real activity; a popped
    //     OP_TELEM request pushes a counter SNAPSHOT (tagged is_telem) through
    //     the SAME FIFO; a produced token pushes {is_telem=0, run-ctx, token}.
    //     Token pushes have priority; a pending telemetry push waits for a free
    //     write cycle, so nothing is dropped.
    // ======================================================================
    wire                  tok_wr_full;
    wire                  resp_wr_en;
    wire [RESP_W_EFF-1:0] resp_wr_data;

    generate
        if (PROTO_CTX) begin : g_proto_core
            // registered telemetry counters (core_clk domain)
            reg [TELEM_W-1:0] ctr_tok;    // tokens produced (sys_tok_valid)
            reg [TELEM_W-1:0] ctr_run;    // token-gen runs started (sys_start)
            reg [TELEM_W-1:0] ctr_done;   // runs completed (sys_done)
            reg [CTX_W-1:0]   cur_ctx;    // ctx of the in-flight run (attached to tokens)
            reg [CTX_W-1:0]   pend_ctx;   // ctx of the held (queued) launch
            reg               telem_pend; // an OP_TELEM snapshot is waiting to push
            reg [CTX_W-1:0]   telem_ctx;  // ctx of the pending telemetry response

            wire telem_push = telem_pend & ~sys_tok_valid & ~tok_wr_full;

            always @(posedge core_clk) begin
                if (core_rst_sync) begin
                    ctr_tok    <= {TELEM_W{1'b0}};
                    ctr_run    <= {TELEM_W{1'b0}};
                    ctr_done   <= {TELEM_W{1'b0}};
                    cur_ctx    <= {CTX_W{1'b0}};
                    pend_ctx   <= {CTX_W{1'b0}};
                    telem_pend <= 1'b0;
                    telem_ctx  <= {CTX_W{1'b0}};
                end else begin
                    if (req_rd_d) begin
`ifdef INJECT_CTXTAG
                        // INJECTION (must-FAIL build): the pre-fix behaviour --
                        // cur_ctx follows EVERY pop, OP_TELEM included, so a
                        // mid-run telemetry poll (ctx=B) retags the in-flight
                        // ctx=A run's token as B.  The mid-run-telemetry test
                        // must catch this; if it passes, the ctx binding is not
                        // load-bearing.
                        cur_ctx <= req_ctx_popped;
`endif
                        if (req_launch) begin
                            // ctx follows the RUN, not the pop: a launch that
                            // fires now owns cur_ctx; a held launch parks its
                            // ctx in pend_ctx until it actually fires below.
                            if (~run_inflight) cur_ctx  <= req_ctx_popped;
                            else               pend_ctx <= req_ctx_popped;
                            ctr_run <= ctr_run + 1'b1;
                        end
                        if (req_is_telem) begin
                            telem_pend <= 1'b1;
                            telem_ctx  <= req_ctx_popped;
                        end
                    end else if (launch_pend & ~run_inflight)
                        cur_ctx <= pend_ctx;                // held launch fires this cycle
                    if (sys_tok_valid) ctr_tok  <= ctr_tok  + 1'b1;
                    if (sys_done)      ctr_done <= ctr_done + 1'b1;
                    if (telem_push)    telem_pend <= 1'b0;  // cleared once its snapshot is pushed
                end
            end

            // packed telemetry payload (LSB-first: tokens, runs, done, stall).
            // ec_demand_stall_cycles is a live core-domain observability output.
            wire [TELEM_PAY-1:0] telem_pack =
                { ec_demand_stall_cycles[TELEM_W-1:0], ctr_done, ctr_run, ctr_tok };
            wire [RESP_PAY-1:0]  tok_pay = { {(RESP_PAY-TOKW){1'b0}}, sys_next_tok };

            assign resp_wr_en   = (sys_tok_valid | telem_pend) & ~tok_wr_full;
            assign resp_wr_data = telem_push
                ? { 1'b1, telem_ctx, telem_pack }                        // telemetry snapshot
                : { 1'b0, cur_ctx,   tok_pay    };                       // generated token
        end else begin : g_proto_core_off
            assign resp_wr_en   = sys_tok_valid & ~tok_wr_full;
            assign resp_wr_data = sys_next_tok;
        end
    endgenerate

    // ======================================================================
    // RESPONSE cdc_async_fifo : core_clk (write) -> host_clk (read).
    //  (Named u_tok_fifo -- the token/response path.)  Push the produced token
    //  (and, when PROTO_CTX=1, telemetry snapshots + ctx tags) on resp_wr_en.
    // ======================================================================
    wire                  tok_rd_empty;
    wire [RESP_W_EFF-1:0] tok_rd_data;
    wire                  tok_rd_en;

    cdc_async_fifo #(
        .DATA_W (RESP_W_EFF),
        .ADDR_W (TOK_AW)
    ) u_tok_fifo (
        .wclk   (core_clk), .wrst_n (core_rst_n),
        .wr_en  (resp_wr_en), .wr_data(resp_wr_data), .full(tok_wr_full),
        .rclk   (host_clk), .rrst_n (host_rst_n),
        .rd_en  (tok_rd_en), .rd_data(tok_rd_data), .empty(tok_rd_empty)
    );

    // ----------------------------------------------------------------------
    // host_clk token pop : single-outstanding read; the registered FIFO read
    // makes the popped word valid one host_clk later (tok_rd_d), which we latch
    // into next_tok and surface as a one-cycle tok_valid pulse.
    // ----------------------------------------------------------------------
    reg tok_rd_d;
    assign tok_rd_en = ~tok_rd_empty & ~tok_rd_d;

    always @(posedge host_clk) begin
        if (host_rst_sync) begin
            tok_rd_d  <= 1'b0;
            next_tok  <= {TOKW{1'b0}};
            tok_valid <= 1'b0;
        end else begin
            tok_valid <= 1'b0;
            tok_rd_d  <= tok_rd_en;
            if (tok_rd_d) begin
                next_tok  <= tok_rd_data;   // low TOKW bits == the token
                tok_valid <= 1'b1;
            end
        end
    end

    // ----------------------------------------------------------------------
    // host_clk RESPONSE tag extraction (PROTO_CTX=1): latch ctx id / telemetry
    // fields from the SAME popped FIFO word, aligned to tok_valid (same tok_rd_d
    // timing).  All fields come from tok_rd_data (a host-domain FIFO output) --
    // no cross-domain register is sampled here.  Default OFF: outputs tied to 0.
    // ----------------------------------------------------------------------
    generate
        if (PROTO_CTX) begin : g_proto_host
            reg [CTX_W-1:0]     resp_ctx_r;
            reg                 resp_tel_r;
            reg [TELEM_PAY-1:0] resp_tel_pay_r;
            always @(posedge host_clk) begin
                if (host_rst_sync) begin
                    resp_ctx_r     <= {CTX_W{1'b0}};
                    resp_tel_r     <= 1'b0;
                    resp_tel_pay_r <= {TELEM_PAY{1'b0}};
                end else if (tok_rd_d) begin
                    resp_tel_r     <= tok_rd_data[RESP_W_EFF-1];          // MSB: is_telem
                    resp_ctx_r     <= tok_rd_data[RESP_W_EFF-2 -: CTX_W]; // next CTX_W: ctx id
                    resp_tel_pay_r <= tok_rd_data[TELEM_PAY-1:0];         // low bits: counters
                end
            end
            assign resp_ctx_id   = resp_ctx_r;
            assign resp_is_telem = resp_tel_r;
            assign resp_telem    = resp_tel_pay_r;
        end else begin : g_proto_host_off
            assign resp_ctx_id   = {CTX_W{1'b0}};
            assign resp_is_telem = 1'b0;
            assign resp_telem    = {(4*TELEM_W){1'b0}};
        end
    endgenerate

    // ======================================================================
    // STATUS crossings core_clk -> host_clk (single-bit only).
    //   busy : a LEVEL  -> plain 2-FF synchronizer.
    //   done : a PULSE  -> a TOGGLE in the core domain, 2-FF synced, edge-detected
    //          in the host domain (a level-sync could miss a 1-core-cycle pulse;
    //          a toggle survives because the host domain only needs to see the
    //          single-bit flip eventually -- metastability-safe).
    // ======================================================================
    // ---- busy (level) 2-FF sync ----
    reg busy_s1, busy_s2;
    always @(posedge host_clk) begin
        if (host_rst_sync) begin
            busy_s1 <= 1'b0;
            busy_s2 <= 1'b0;
        end else begin
            busy_s1 <= sys_busy;
            busy_s2 <= busy_s1;
        end
    end

    // ---- done toggle (core) ----
    reg done_tgl_c;
    always @(posedge core_clk) begin
        if (core_rst_sync)   done_tgl_c <= 1'b0;
        else if (sys_done)   done_tgl_c <= ~done_tgl_c;
    end

    // ---- done toggle 2-FF sync + edge detect (host) ----
    reg done_tgl_h1, done_tgl_h2, done_tgl_h3;
    wire done_edge = done_tgl_h3 ^ done_tgl_h2;
    always @(posedge host_clk) begin
        if (host_rst_sync) begin
            done_tgl_h1 <= 1'b0;
            done_tgl_h2 <= 1'b0;
            done_tgl_h3 <= 1'b0;
        end else begin
            done_tgl_h1 <= done_tgl_c;
            done_tgl_h2 <= done_tgl_h1;
            done_tgl_h3 <= done_tgl_h2;
        end
    end

    // ----------------------------------------------------------------------
    // host-facing busy/done.
    //   host_pending : set when a request is accepted into the REQUEST FIFO,
    //   cleared when the run's done edge is observed.  This covers the launch gap
    //   before the (sync-delayed) core busy rises, so busy is asserted immediately
    //   on accept and stays high until the host-visible completion.
    // ----------------------------------------------------------------------
    // A pure telemetry readback (OP_TELEM) does NOT start a run, so it must not
    // raise the run-pending flag (there is no done edge to clear it).  Folds to
    // constant 1 when PROTO_CTX=0 -> byte-identical default behavior.
    wire req_sets_pending;
    generate
        if (PROTO_CTX) begin : g_pending_gate
            assign req_sets_pending = (req_opcode == OP_TOKEN);
        end else begin : g_pending_gate_off
            assign req_sets_pending = 1'b1;
        end
    endgenerate

    reg host_pending;
    always @(posedge host_clk) begin
        if (host_rst_sync) host_pending <= 1'b0;
        else begin
            if (req_wr_en & req_sets_pending) host_pending <= 1'b1;
            else if (done_edge)               host_pending <= 1'b0;
        end
    end

    always @(posedge host_clk) begin
        if (host_rst_sync) begin
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            busy <= host_pending | busy_s2;   // launch gap | synced core busy
            done <= done_edge;                // one host_clk completion pulse
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
