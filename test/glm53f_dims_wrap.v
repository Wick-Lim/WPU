//============================================================================
// test/glm53f_dims_wrap.v -- MUST-PASS half of the GLM-5.3-Flash config gate.
//
// configs/full_glm53_flash.vh carries REAL, cited GLM-5.3-Flash dimensions, so
// a partial-model study (the 11 MLA+DSA blocks, the MoE, the memory system --
// i.e. everything this repo actually has RTL for) must be able to read them
// WITHOUT asserting that the missing machines exist.  This wrapper consumes the
// dimension macros with none of the *_RTL_PRESENT defines set and must
// elaborate clean in both iverilog and Verilator.
//
// If this ever fails, the guard has over-reached: it is poisoning facts instead
// of poisoning the unbuilt full-model top.  See the header comment of
// configs/full_glm53_flash.vh.
//
// The checks below are the checkpoint's own structural invariants, each one a
// fact the GGUF tensor map proved (tools/glm53_flash_gguf_scan.py).  A future
// edit that trips one has mis-transcribed the GGUF.
//
// IDIOM (do not "simplify"): generate-scope $error with ONE string literal and
// NO format args -- the elaboration-time form that fires in both simulators,
// and the only form iverilog accepts (it answers format args with "sorry:
// Elaboration tasks currently only support a single string argument").  Same
// rule and rationale as src/mla_attn_q4k.v; this gate runs under -tnull and
// --lint-only, where nothing in an `initial` block would ever execute.
// (A comment line must also not START with the word V-e-r-i-l-a-t-o-r: that
// tool reads such a line as a pragma and fails with BADVLTPRAGMA -- measured.)
//============================================================================
`include "full_glm53_flash.vh"

module glm53f_dims_wrap;
  // top-level shape
  localparam integer MODEL_DIM = `GLM53F_MODEL_DIM;
  localparam integer L         = `GLM53F_L;
  localparam integer BLOCKS    = `GLM53F_BLOCKS;
  localparam integer VOCAB     = `GLM53F_VOCAB;
  localparam integer NEXTN     = `GLM53F_NEXTN;
  // MLA (the blocks this repo can build)
  localparam integer H_HEADS   = `GLM53F_H_HEADS;
  localparam integer NOPE      = `GLM53F_NOPE;
  localparam integer ROPE      = `GLM53F_ROPE;
  localparam integer QK_DIM    = `GLM53F_QK_DIM;
  localparam integer Q_LORA    = `GLM53F_Q_LORA;
  localparam integer KV_LORA   = `GLM53F_KV_LORA;
  // MoE
  localparam integer N_EXPERT  = `GLM53F_N_EXPERT;
  localparam integer TOPK      = `GLM53F_TOPK;
  localparam integer INTER_MOE = `GLM53F_INTER_MOE;
  // layer schedule
  localparam integer N_KDA       = `GLM53F_N_KDA;
  localparam integer N_MLA       = `GLM53F_N_MLA;
  localparam integer N_MLA_TOTAL = `GLM53F_N_MLA_TOTAL;
  localparam integer PERIOD      = `GLM53F_ATTN_PERIOD;
  localparam integer OFFSET      = `GLM53F_ATTN_OFFSET;
  // how many blocks in [0, L) satisfy (i modulo PERIOD == OFFSET)
  localparam integer SCHED_MLA   = ((L - 1 - OFFSET) / PERIOD) + 1;

  generate
    if (N_KDA + N_MLA != L) begin : gen_bad_layer_split
      $error("GLM-5.3-Flash layer split broken: N_KDA + N_MLA must equal L (34 + 11 = 45)");
    end
    if (N_MLA != SCHED_MLA) begin : gen_bad_layer_schedule
      $error("GLM-5.3-Flash attention schedule broken: N_MLA must equal the count of blocks whose index modulo ATTN_PERIOD equals ATTN_OFFSET, i.e. blocks 3,7,...,43");
    end
    // The count alone does NOT pin the offset: for L=45, PERIOD=4 the block
    // count is 11 for OFFSET 2 and for OFFSET 3 alike (measured by injection).
    // These two anchor WHICH blocks are MLA, which is what the tensor map says:
    // the first full-attention block is 3 -- exactly where the dense FFN front
    // ends -- and the last one is 43, with block 44 KDA and block 45 the MTP.
    if (((L - 2) % PERIOD) != OFFSET) begin : gen_bad_last_mla_block
      $error("GLM-5.3-Flash attention schedule broken: the LAST full-attention transformer block must be block 43 (L-2), leaving block 44 KDA");
    end
    if (OFFSET != `GLM53F_N_DENSE) begin : gen_bad_first_mla_block
      $error("GLM-5.3-Flash attention schedule broken: the FIRST full-attention block must be block 3, the same index where the dense FFN front ends and the sparse MoE blocks begin");
    end
    if (N_MLA_TOTAL != N_MLA + NEXTN) begin : gen_bad_mla_total
      $error("GLM-5.3-Flash N_MLA_TOTAL must equal N_MLA + NEXTN -- the MTP block is MLA-shaped");
    end
    if (BLOCKS != L + NEXTN) begin : gen_bad_block_count
      $error("GLM-5.3-Flash BLOCKS must equal L + NEXTN (GGUF block_count 46 = 45 layers + 1 MTP block)");
    end
    if (ROPE != 0) begin : gen_not_nope
      $error("GLM-5.3-Flash is NoPE: GLM53F_ROPE must be 0 (GGUF rope.dimension_count = 0)");
    end
    if (QK_DIM != NOPE + ROPE) begin : gen_bad_qk_dim
      $error("GLM-5.3-Flash QK_DIM must equal NOPE + ROPE (256 = 256 + 0)");
    end
    if (`GLM53F_KEY_LEN != KV_LORA) begin : gen_bad_key_len
      $error("GLM-5.3-Flash is NoPE, so attention.key_length must EQUAL kv_lora_rank (both 512): with rope.dimension_count = 0 there is no rotary tail appended to the compressed latent, unlike GLM-5.2 where the key is kv_lora + rope");
    end
    if (`GLM53F_VALUE_LEN != KV_LORA) begin : gen_bad_value_len
      $error("GLM-5.3-Flash MLA reads keys and values from ONE 512-wide latent: attention.value_length must equal kv_lora_rank");
    end
    if ((VOCAB % `GLM53F_LM_TN) != 0) begin : gen_bad_lm_tiling
      $error("LM-head GEMV tiling requires VOCAB to be a whole multiple of LM_TN");
    end
    if ((N_EXPERT % TOPK) != 0) begin : gen_bad_expert_ratio
      $error("GLM-5.3-Flash routed-expert count must be a whole multiple of top-k (288 / 8 = 36)");
    end
  endgenerate
endmodule
