//============================================================================
// test/glm53f_fulltop_wrap.v -- MUST-FAIL / MUST-PASS half of the gate.
//
// Stands in for any *whole-model* GLM-5.3-Flash wrapper.  It reads
// `GLM53F_FULL_TOP_OK, which configs/full_glm53_flash.vh leaves as an UNDEFINED
// self-describing identifier until all three absent machines are declared
// present:
//     GLM53F_KDA_RTL_PRESENT   34/45 blocks are KDA linear attention
//     GLM53F_HC_RTL_PRESENT    hyper-connections replace the residual add
//     GLM53F_Q5K_RTL_PRESENT   Q5_K is 34.9% of the checkpoint's bytes
//
// Expected results (all four verified in BOTH iverilog and Verilator by
// `make glm53f-config-guard`):
//     no defines        -> FAIL   "Unable to bind parameter `GLM53F_INCOMPLETE_...'"
//     any 2 of the 3    -> FAIL   (same, the gate is an AND)
//     all 3 defined     -> PASS
//
// The point: a build cannot elaborate a "GLM-5.3-Flash full model" and stay
// silent about the fact that three quarters of the model has no RTL.  Defining
// these without landing the RTL is falsifying the ledger, not configuring it.
//============================================================================
`include "full_glm53_flash.vh"

module glm53f_fulltop_wrap;
  localparam integer FULL_TOP_OK = `GLM53F_FULL_TOP_OK;
  localparam integer L           = `GLM53F_L;
  initial $display("glm53f-fulltop elaborated: OK=%0d L=%0d", FULL_TOP_OK, L);
endmodule
