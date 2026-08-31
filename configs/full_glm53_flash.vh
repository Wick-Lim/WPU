//============================================================================
// configs/full_glm53_flash.vh  --  REAL GLM-5.3-Flash full-model config macros
//----------------------------------------------------------------------------
// STATUS: **model config LOCKED. datapath NOT COMPLETE.**
//
// Every model value below is read from the published checkpoint, not guessed:
//   [gguf] = GGUF metadata KV of unsloth/GLM-5.3-Flash-GGUF : UD-Q4_K_XL
//            (the authority for this branch -- the contract is to read THOSE
//             bytes bit-exactly, so the GGUF outranks config.json)
//   [cfg]  = config.json of zai-org/GLM-5.3-Flash (text_config.*)
//   [scan] = derived from the GGUF tensor map by tools/glm53_flash_gguf_scan.py
// Re-derive every number here with:
//   python3 tools/glm53_flash_gguf_scan.py --fetch <dir> && \
//   python3 tools/glm53_flash_gguf_scan.py <dir>
//
// WHY THE GUARD MOVED (this is the important part of this header)
//   The GLM-5.3 scaffold header poisoned the *dimensions*, because the
//   checkpoint was not public and every value was a guess.  That hazard is
//   gone: the checkpoint IS public and every dimension below is a hard
//   citation.  Poisoning a fact would be theatre.
//
//   The hazard is now the opposite one.  GLM-5.3-Flash is NOT a GLM-5.2
//   re-dimension -- it is arch `glm5next`, and 34 of its 45 layers are KDA
//   linear attention, a machine this repo does not have (docs/GLM53_FLASH_PORT.md
//   §2).  So the thing that must fail loudly is no longer "a guessed dimension
//   compiles" but **"a full-model top elaborates while three quarters of the
//   model is missing, and the build calls itself GLM-5.3-Flash"**.
//
//   Therefore: dims are plain, usable, cited defines.  `GLM53F_FULL_TOP_OK`
//   -- required by any *whole-model* wrapper -- expands to a self-describing
//   UNDEFINED identifier until all three missing machines are declared present:
//       GLM53F_KDA_RTL_PRESENT   34/45 layers   (KDA linear attention)
//       GLM53F_HC_RTL_PRESENT    residual path  (hyper-connections, Sinkhorn)
//       GLM53F_Q5K_RTL_PRESENT   34.9% of bytes (the whole Q5_K READ PATH)
//
//   Q5_K STATUS -- read this before assuming the third one is satisfied.  The
//   GEMM arm exists and is gated bit-exact (`WT_Q5K` in glm_matmul_q4k, `make
//   mixedtype` with a must-fail injection).  What does NOT exist is the rest of
//   the read path: weight_loader_q4k cannot lay out a Q5_K tile, because that
//   needs the packer to emit pre-assembled 5-bit codes and a 176 B/super-block
//   geometry (docs/GLM53_FLASH_PORT.md 4.2 item 6).  A whole-model top has to
//   STREAM Q5_K tiles, not just multiply them, so this define stays undefined
//   until the loader path lands -- and the loader $fatal's on a Q5_K descriptor
//   in the meantime rather than silently streaming Q4_K geometry.
//   Defining any of those without landing the RTL is falsifying the ledger.
//   Partial-model studies (the 11 MLA+DSA layers, the MoE, the memory system)
//   do NOT need the gate and keep working -- that is the inherited, proven part.
//============================================================================
`ifndef FULL_GLM53_FLASH_VH
`define FULL_GLM53_FLASH_VH

// ---- top-level model dims ----
`define GLM53F_MODEL_DIM   4096      // [gguf] embedding_length / [cfg] hidden_size   (GLM-5.2: 6144)
`define GLM53F_L           45        // [cfg]  num_hidden_layers                      (GLM-5.2: 78)
`define GLM53F_BLOCKS      46        // [gguf] block_count = L + 1 MTP block
`define GLM53F_N_DENSE     3         // [gguf] leading_dense_block_count / [cfg] first_k_dense_replace
`define GLM53F_VOCAB       154880    // [gguf] vocab_size                             (GLM-5.2: 154880)
`define GLM53F_CTX         1048576   // [gguf] context_length (1M)
`define GLM53F_POSW        20        // 2^20 = 1,048,576 exactly covers CTX

// ---- layer schedule -- THE structural delta vs GLM-5.2 ----
// [gguf] attention.head_count_kv is a per-block list: 0 = KDA linear-attention
// block, 1 = MLA+DSA block.  [scan] confirms the pattern is strictly periodic.
//   full-attention blocks = { 3, 7, 11, ... , 43 }  and the MTP block 45
`define GLM53F_ATTN_PERIOD 4         // [scan] every 4th block is MLA+DSA
`define GLM53F_ATTN_OFFSET 3         // [scan] ...starting at block 3
`define GLM53F_N_KDA       34        // [scan] KDA linear-attention blocks
`define GLM53F_N_MLA       11        // [scan] MLA+DSA blocks (excl. MTP)
`define GLM53F_N_MLA_TOTAL 12        // [scan] ...incl. the MTP block, which is MLA-shaped

// ---- MLA attention (the 11 full-attention blocks + MTP) ----
// NOTE: GLM-5.3-Flash is **NoPE** -- there is no rotary embedding anywhere in
// the attention path.  GLM-5.2's rope unit has no consumer here.
`define GLM53F_H_HEADS     64        // [gguf] attention.head_count                   (GLM-5.2: 64)
`define GLM53F_NOPE        256       // [cfg]  qk_nope_head_dim                       (GLM-5.2: 192)
`define GLM53F_ROPE        0         // [gguf] rope.dimension_count = 0 / [cfg] qk_rope_head_dim (GLM-5.2: 64)
`define GLM53F_QK_DIM      256       // [gguf] attention.key_length_mla / [cfg] qk_head_dim
`define GLM53F_V_DIM       256       // [gguf] attention.value_length_mla / [cfg] v_head_dim
`define GLM53F_KEY_LEN     512       // [gguf] attention.key_length   (kv_lora + nope carry)
`define GLM53F_VALUE_LEN   512       // [gguf] attention.value_length
`define GLM53F_Q_LORA      1536      // [gguf] attention.q_lora_rank                  (GLM-5.2: 2048)
`define GLM53F_KV_LORA     512       // [gguf] attention.kv_lora_rank  -- CONFIRMED here,
                                     //        which retires the standing GLM-5.2 assumption
`define GLM53F_RMS_EPS     32'h3727C5AC // [gguf] attention.layer_norm_rms_epsilon = 1e-5 (fp32)
// attention_bias = false [cfg]; num_key_value_heads = 64 [cfg]; mla_use_nope = true [cfg]

// ---- DSA sparse-attention indexer ----
`define GLM53F_TOPK_ATTN   2048      // [gguf] attention.indexer.top_k                (GLM-5.2: 2048)
`define GLM53F_IDX_HEADS   32        // [gguf] attention.indexer.head_count
`define GLM53F_IDX_DIM     128       // [gguf] attention.indexer.key_length
`define GLM53F_IDX_KPOOL   4         // [gguf] attention.indexer.kpool  -- NEW vs GLM-5.2
// index_kpool_compress = true, index_kpool_always_select_tail = true [cfg]
// index_share_for_mtp_iteration = true [cfg]  (IndexShare, as in GLM-5.2)
// [scan] the indexer gained its own compressor: indexer_compressor_ape [128,4]
//        + indexer_compressor_gate [4096,128] -- no GLM-5.2 counterpart.

// ---- MoE / FFN ----
`define GLM53F_N_EXPERT    288       // [gguf] expert_count                           (GLM-5.2: 256)
`define GLM53F_TOPK        8         // [gguf] expert_used_count                      (GLM-5.2: 8)
`define GLM53F_N_SHARED    1         // [gguf] expert_shared_count
`define GLM53F_INTER_MOE   2048      // [gguf] expert_feed_forward_length             (GLM-5.2: 2048)
`define GLM53F_INTER_SHEXP 2048      // [gguf] expert_shared_feed_forward_length
`define GLM53F_INTER_DENSE 12288     // [gguf] feed_forward_length (dense front, blocks 0-2)
`define GLM53F_RSCALE      32'h40200000 // [gguf] expert_weights_scale = 2.5 (fp32)   (GLM-5.2: 2.5)
`define GLM53F_GATING      2         // [gguf] expert_gating_func = 2 (sigmoid) / [cfg] scoring_func
`define GLM53F_WEIGHTS_NORM 1        // [gguf] expert_weights_norm = true / [cfg] norm_topk_prob
`define GLM53F_N_GROUP     1         // [cfg]  n_group  (topk_group = 1, topk_method = noaux_tc)
// [scan] per-expert routing bias `exp_probs_b.bias` [288] is present on all 43 sparse blocks.

// ---- SwiGLU clamp -- NEW vs GLM-5.2, and it is NOT optional ----
// [gguf] swiglu_clamp_exp / swiglu_clamp_shexp are per-block lists, all 10.0.
// [cfg]  swiglu_limit = 10.0.  GLM-5.2 has no clamp; a GLM-5.3-Flash SwiGLU
// that omits it is numerically wrong, not merely approximate.
`define GLM53F_SWIGLU_CLAMP 32'h41200000 // 10.0 (fp32)

// ---- MTP / speculative decode ----
// [gguf] nextn_predict_layers = 1  -> the MTP head IS present, so the
// spec-decode composition (glm_q4k_spec_system) has a counterpart here.  The
// GLM-5.2-measured A_eff / accept-rate do NOT transfer -- they must be
// re-measured on this model (docs/GLM53_FLASH_PORT.md §4).
`define GLM53F_NEXTN       1
`define GLM53F_MTP_BLOCK   45        // [scan] blk.45.nextn.{eh_proj,enorm,hnorm,shared_head_norm}

// ---- weight quantization: the UD-Q4_K_XL mix, MEASURED [scan] ----
//   Q4_K   84 tensors  114.15 GB  57.2%   ffn_{gate,up}_exps
//   Q5_K   42 tensors   69.76 GB  34.9%   ffn_down_exps          <-- NO KERNEL HERE
//   Q8_0  645 tensors    9.62 GB   4.8%   attention / shared-expert / embed / lm_head
//   Q6_K    3 tensors    5.95 GB   3.0%   UD bump: blk.{11,12,44}.ffn_down_exps
//   F32   638 tensors    0.23 GB   0.1%   norms, routing gates, ssm_a/dt/conv1d
//   TOTAL 1412 tensors  199.70 GB  -- cross-checks against the published shard
//                                     bytes (199.71 GB) to within the 9.52 MB
//                                     of GGUF headers.  That agreement is the
//                                     evidence the tensor parse is correct.
// UD "Dynamic" bumps [scan]: blk.11 gate/up_exps Q4_K->Q5_K; blk.{11,12,44}
// down_exps Q5_K->Q6_K.  A fixed-type assumption would mis-read those blocks.
`define GLM53F_BLK         256       // k-quant super-block (format constant, not model config)

// ---- KDA linear attention -- 34/45 BLOCKS, NO RTL IN THIS REPO ----
// Values are cited and correct; there is simply nothing here that consumes
// them yet.  Tensors per KDA block [scan]: attn_{q,k,v} [4096,8192],
// ssm_conv1d_{q,k,v} [4,1,8192], ssm_{f,g}_{a,b}, ssm_beta, ssm_a [64],
// ssm_dt.bias [8192], ssm_norm [128].
`define GLM53F_KDA_HEADS   64        // [cfg] linear_attn_config.num_heads
`define GLM53F_KDA_DIM     128       // [gguf] kda.head_dim
`define GLM53F_KDA_CONV_K  4         // [gguf] ssm.conv_kernel (short causal conv)
`define GLM53F_KDA_GATE_LB 32'hC0A00000 // [gguf] kda.gate_lower_bound = -5.0 (fp32)

// ---- hyper-connections -- EVERY block, NO RTL IN THIS REPO ----
// [scan] hc_{attn,ffn}_{base [24], fn [16384,24], scale [3]} on all 45 blocks.
// This replaces the plain residual add; it is a structural change to the
// block, not a tweak.
`define GLM53F_HC_MULT     4         // [gguf] hyper_connection.count / [cfg] hc_mult
`define GLM53F_HC_SINKHORN 20        // [gguf] hyper_connection.sinkhorn_iterations
`define GLM53F_HC_EPS      32'h358637BD // [gguf] hyper_connection.epsilon = 1e-6 (fp32)

// ---- vision tower -- OUT OF SCOPE, NO RTL ----
// GLM-5.3-Flash is Glm5NextForConditionalGeneration: config.json carries a
// vision_config (24-layer ViT, hidden 1024, patch 14, image 448, merge 2,
// out_hidden 4096).  The UD-Q4_K_XL GGUF above ships **text weights only**
// [scan: no vision tensor in any of the 1412], so this branch's text-path
// contract is complete without it.  Listed so the omission is deliberate.
`define GLM53F_VIS_DEPTH   24
`define GLM53F_VIS_DIM     1024
`define GLM53F_VIS_PATCH   14
`define GLM53F_VIS_IMG     448

// ---- hardware tiling knobs (accelerator microarch, NOT model config) ----
// Inherited from the GLM-5.2 build and still valid: they are properties of the
// PE array, not of the checkpoint.
`define GLM53F_PE_N        4         // attention/matmul output-lane tile width
`define GLM53F_TN          4         // swiglu output-tile width
`define GLM53F_LM_TN       4         // LM-head GEMV tile width (VOCAB % LM_TN == 0: 154880 % 4 == 0 OK)
`define GLM53F_PE_M        1         // query-token batch B (1 == committed datapath)

// ============================================================================
// FULL-MODEL TOP GATE -- see the header comment for why this, and not the dims
// ============================================================================
`ifdef GLM53F_KDA_RTL_PRESENT
 `ifdef GLM53F_HC_RTL_PRESENT
  `ifdef GLM53F_Q5K_RTL_PRESENT
   `define GLM53F_FULL_TOP_OK 1
  `endif
 `endif
`endif
`ifndef GLM53F_FULL_TOP_OK
 `define GLM53F_FULL_TOP_OK GLM53F_INCOMPLETE_need_KDA_and_hyperconnection_and_Q5K_rtl_see_docs_GLM53_FLASH_PORT_md
`endif

// ============================================================================
// SLICE reference values -- the committed TBs' small-but-faithful config.
// Inherited VERBATIM from the GLM-5.2 build.  They exercise the parameterized
// RTL that this branch carries; they claim nothing about GLM-5.3-Flash, whose
// own slice must wait until the KDA block exists to be sliced.
// ============================================================================
`define GLM53F_SLICE_MODEL_DIM   128
`define GLM53F_SLICE_L           6
`define GLM53F_SLICE_VOCAB       256
`define GLM53F_SLICE_H_HEADS     4
`define GLM53F_SLICE_NOPE        16
`define GLM53F_SLICE_ROPE        16
`define GLM53F_SLICE_V_DIM       32
`define GLM53F_SLICE_Q_LORA      64
`define GLM53F_SLICE_KV_LORA     32
`define GLM53F_SLICE_S_MAX       8
`define GLM53F_SLICE_TOPK_ATTN   8
`define GLM53F_SLICE_N_EXPERT    8
`define GLM53F_SLICE_TOPK        2
`define GLM53F_SLICE_INTER_MOE   64
`define GLM53F_SLICE_INTER_DENSE 256

// ============================================================================
// STRUCTURAL CAVEAT -- S_MAX (unchanged from the GLM-5.2 header)
//   mla_attn_q4k sizes its scores/probs/vstore scratch by S_MAX; a full-context
//   S_MAX would explode elaboration.  S_MAX stays at the latent-ring depth,
//   independent of the 1M POSW field.  Decoupling it is task B7 (SWIN).
//   NOTE: on GLM-5.3-Flash this caveat binds on only 11 of 45 blocks -- the
//   KDA blocks carry a fixed-size recurrent state instead of a growing KV
//   cache, which is the main reason this model's memory profile is not
//   GLM-5.2's.  Quantifying that is docs/GLM53_FLASH_PORT.md §5.
// ============================================================================
`define GLM53F_S_MAX       8         // KEEP SMALL (latent-ring depth)

`endif // FULL_GLM53_FLASH_VH
