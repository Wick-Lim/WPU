//============================================================================
// test/glm53f_fulltop_wrap.v -- MUST-FAIL / MUST-PASS half of the gate.
//
// Stands in for any *whole-model* GLM-5.3-Flash wrapper.  It reads
// `GLM53F_FULL_TOP_OK, which configs/full_glm53_flash.vh leaves as an UNDEFINED
// self-describing identifier unless all three machines are declared present:
//     GLM53F_KDA_RTL_PRESENT   34/45 blocks are KDA linear attention
//     GLM53F_HC_RTL_PRESENT    hyper-connections replace the residual add
//     GLM53F_Q5K_RTL_PRESENT   Q5_K is 34.9% of the checkpoint's bytes
//
// As of 2026-09-07 ALL THREE ARE BUILT, so the default direction has flipped:
//     no defines            -> PASS   (all three come from the header)
//     -DGLM53F_NO_HC        -> FAIL   "Unable to bind parameter `GLM53F_INCOMPLETE_...'"
//     -DGLM53F_NO_KDA       -> FAIL   (same)
//     -DGLM53F_NO_Q5K       -> FAIL   (same)
// all eight verified in BOTH iverilog and Verilator by `make glm53f-config-guard`.
// Forcing each machine absent in turn is what keeps the three conditions
// load-bearing now that none of them is missing; a gate that can no longer fail
// proves nothing.
//
// WHAT ELABORATING HERE DOES NOT MEAN.  This wrapper reads a define; it does not
// instantiate a decoder layer, and neither glm53f_hc_block nor glm53f_kda_attn is
// wired into one yet.  The three conditions were always about whether the three
// MISSING MACHINES exist, which is a question this gate can answer.  Whether the
// model is assembled is not, and is tracked in docs/GLM53_FLASH_PORT.md 4.2
// instead -- do not read a passing elaboration as a working model.
//============================================================================
`include "full_glm53_flash.vh"

module glm53f_fulltop_wrap;
  localparam integer FULL_TOP_OK = `GLM53F_FULL_TOP_OK;
  localparam integer L           = `GLM53F_L;
  initial $display("glm53f-fulltop elaborated: OK=%0d L=%0d", FULL_TOP_OK, L);
endmodule
