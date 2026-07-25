# GLM-5.2 (UD-Q4_K_XL) accelerator -- Verilog build / simulation / synth / formal
#
#   make all        -> the GLM prove-it gate: unittests + synth-glm + formal
#                      + model-q4k-smoke + resident + resident-equiv + full-elab
#   make unittests  -> build + run EVERY per-unit TB (ALL N TESTS PASSED each)
#   make synth-glm  -> yosys elaborate + `check -assert` the whole product top
#   make formal     -> BMC of the memory-system controllers
#   make expert-cache -> expert-cache prefetch + replacement-policy TBs on the
#                      real GLM decode routing trace (minutes-long; standalone)
#   make full-elab  -> 753B-shape elaboration of glm_model_q4k (iverilog -tnull)
#   make release-gate -> the full pre-release battery (HOURS).  NOT literally every
#                      target: the deliberate exclusions and their reasons are the
#                      NOTE block above the release-gate rule (~line 36).
#   make lint       -> verilator --lint-only on the GLM top (diagnostic; not yet
#                      -Wall clean -- informational, NOT part of `all`)
#   make coverage   -> verilator --coverage-line/-toggle structural coverage of
#                      the clean-verilating TB subset (per-module + merged report)
#   make host-test  -> host-side runtime scaffold self-test (host/test_wpu.py)
#   make clean      -> remove build artifacts and the generated VCDs

IVERILOG  ?= iverilog
VVP       ?= vvp
VERILATOR ?= verilator
YOSYS     ?= yosys

BUILD_DIR  := build
IFLAGS := -g2012 -Wall -I src

.PHONY: all unittests q4k mixedtype model-q4k model-q4k-acthw model-q4k-smoke spec-slow spec-adapt expert-cache full-elab release-gate formal formal-ind lint host-test dsa-thread-equiv full-elab-lanes lane-scaling lane-scaling-ratio lane-scaling-sparse dsa-sparse-correct synth-glm fit-harness cdc coverage resident resident-equiv self-kv-roundtrip self-kv-l6-roundtrip self-kv-equiv dsa-thread-equiv provision-selftest boot-integrity weight-ecc weight-ecc-equiv weight-decomp decomp1-elab cdc-protocol cdc-protocol-equiv clean

# `all` is the GLM-5.2 (UD-Q4_K_XL) prove-it gate (main's product): every per-unit
# TB, the whole-chip structural sign-off, the memory-controller formal proofs, plus
# the assembled-forward smoke, the RESIDENT refill gate + its equivalence proof, and
# the 753B-shape elaboration.  (model-q4k-smoke also runs inside `unittests`; listing
# it here keeps `all` covering it even if the unittests tail changes.)
all: unittests synth-glm formal model-q4k-smoke resident resident-equiv full-elab full-elab-lanes mla-sparse decomp1-elab

# `release-gate` is the full pre-release battery: every simulation, structural,
# CDC and formal gate in the repo.  HOURS-long (model-q4k / spec-slow / expert-cache
# / batched-q4k are minutes-to-hours each in iverilog); run before cutting a release.
# NOTE: dsa-thread-equiv is deliberately NOT here -- it has never completed on any
# machine we have (30m45s / 5.29 GB / unfinished; see its header ~line 543).  The thread
# it guards is covered by dsa-sparse-correct, which runs =0 AND =1 end-to-end against the
# reference.  Every prerequisite below must be a gate that can actually finish.
#
# host-test is here because the RTL gates cannot see host/wpu_device.py at all, and 14
# of its 32 tests are the prefix-cache / KV-reuse / context-capacity logic -- including
# test_context_overflow_refuses_instead_of_aliasing, where the failure mode is a ring
# that silently aliases rather than refuses.  2 s.  It was written, it passes, and until
# now nothing but a human remembering to type `make host-test` ever ran it.
#
# STILL NOT HERE, and why: `lint` (verilator --lint-only -Wall).  It is the check that
# caught the weight-loader SELRANGE iverilog silently zero-filled, so it earns a slot --
# but -Wall yields 116 warnings and verilator exits 2 on them.  That predates any of this
# (bisected against 43de204~1); triaging the 116 is its own job, and wiring it in before
# that would just re-create the unrunnable-gate problem this NOTE exists to record.
#
# ALSO DELIBERATELY NOT HERE (each was previously an UNDOCUMENTED exclusion -- the
# README advertised these proofs as PROVEN while no release-gate member could see a
# regression in them; spec-greedy/intra-batch-verify/self-kv-roundtrip/self-kv-equiv/
# loopback/loopback-fw/loopback-rest are therefore folded IN above as of 2026-07-22):
#   - self-kv-l6-roundtrip : the L=6 deep-config variant of self-kv-roundtrip.  Same
#     mechanism, ~L(x) longer sim; the L=1 roundtrip + the SELF_KV structural equiv
#     already bind the write-back path in-gate.  Run it when touching kv_cache_pager.
#   - provision-selftest : needs two EXTERNAL inputs (a real .gguf file and a
#     llama.cpp gguf-py checkout) that a clean clone does not have -- see the
#     fail-fast check in its recipe for how to point PROVISION_GGUF/PROVISION_GGUF_PY.
#   - mshr : src/expert_cache_mshr.v (the non-blocking MSHR refill,
#     docs/ULTRA_PERF_MODULES.md rank 1) is a STANDALONE module -- it is not
#     instantiated anywhere in the product hierarchy, so no edit to the shipped
#     datapath can break it, and it cannot break the shipped datapath.  Run
#     `make mshr` when touching that file.  It joins release-gate on the day it
#     is wired into glm_q4k_system (which also needs the `awaiting` issuer and
#     the single flash arbiter widened -- see the module header).
#   - dsa-thread-equiv / lint : documented above.
#   - mla-intra : attention-unit-level intra-causal proof; its system-level oracle
#     (intra-batch-verify) is in-gate, and the unit gate is minutes-long standalone.
release-gate: unittests q4k mixedtype model-q4k model-q4k-acthw spec-slow spec-adapt spec-greedy intra-batch-verify self-kv-roundtrip self-kv-equiv loopback loopback-fw loopback-rest resident resident-equiv dsa-sparse-correct expert-cache full-elab full-elab-lanes mla-sparse scale-ops batched-q4k perf-q4k boot-integrity weight-ecc weight-ecc-equiv weight-decomp decomp1-elab weight-loader-lanes cdc-protocol cdc-protocol-equiv synth-glm cdc formal formal-ind host-test
	@echo "release-gate: ALL gates passed"

# release-gate-strict: release-gate PLUS an EXACT per-gate test-count check.  The plain
#   gates grep `ALL [0-9]+ TESTS PASSED` -- a wildcard that still passes if a tb silently
#   ran fewer tests (dropped ifdef, reduced loop, missing build flag).  This runs the full
#   gate once (teeing its output), then tools/check_test_counts.sh compares every gate's
#   ACTUAL printed count against test/expected_test_counts.txt and fails on any mismatch or
#   a manifest gate that never printed.  Adding/removing a test => update one manifest line.
.PHONY: release-gate-strict
release-gate-strict:
	@mkdir -p $(BUILD_DIR)
	@bash -c 'set -o pipefail; $(MAKE) release-gate 2>&1 | tee $(BUILD_DIR)/release_gate.log'
	@tools/check_test_counts.sh $(BUILD_DIR)/release_gate.log


# Build + run every per-unit TB.  attention_unit additionally needs softmax_unit.
unittests:
	@mkdir -p $(BUILD_DIR)
	@# expert_predictor_tb reads the generated routing trace (tools/glm_trace.hex);
	@# regenerate it so `unittests` is self-contained on a fresh clone (deterministic seed).
	@python3 tools/route_trace.py --dump >/dev/null
	@# ---- GLM-5.2 datapath units (bf16/fp32, fp32-golden verified) ----
	@# (the old scalar-TPU per-unit TBs were removed with legacy/.)
	@# rmsnorm_unit: bf16 in/out, fp32 reduce + rsqrt; the FP numerics foundation.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rmsnorm_unit_sim test/rmsnorm_unit_tb.v src/rmsnorm_unit.v
	@printf '[%s] ' "rmsnorm_unit"; $(VVP) $(BUILD_DIR)/rmsnorm_unit_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rmsnorm_unit"; exit 1; }
	@# topk_select: top-K of N fp32 scores (DSA top-2048, MoE router top-8), ref-sort golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/topk_select_sim test/topk_select_tb.v src/topk_select.v
	@printf '[%s] ' "topk_select"; $(VVP) $(BUILD_DIR)/topk_select_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: topk_select"; exit 1; }
	@# glm_matmul: bf16 x bf16 -> fp32-accum -> bf16 GEMM workhorse (QKV/O/FFN/experts/LM head).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_sim test/glm_matmul_tb.v src/glm_matmul.v
	@printf '[%s] ' "glm_matmul"; $(VVP) $(BUILD_DIR)/glm_matmul_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul"; exit 1; }
	@# glm_act: bf16 sigmoid + silu (MoE router gating + SwiGLU experts), fp32 internal.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_act_sim test/glm_act_tb.v src/glm_act.v
	@printf '[%s] ' "glm_act"; $(VVP) $(BUILD_DIR)/glm_act_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_act"; exit 1; }
	@# rope_interleave_unit: decoupled interleaved RoPE (MLA q_rope/k_rope, DSA indexer),
	@# fp32 angle table theta=8e6 to 1M positions. (slow TB: full position sweep.)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rope_sim test/rope_interleave_unit_tb.v src/rope_interleave_unit.v
	@printf '[%s] ' "rope_interleave_unit"; $(VVP) $(BUILD_DIR)/rope_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rope_interleave_unit"; exit 1; }
	@# glm_fp_pipe: pipelined FP modules (mul/add/mac/rsqrt/exp), bit-exact vs glm_fp.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp_pipe_sim test/glm_fp_pipe_tb.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp_pipe"; $(VVP) $(BUILD_DIR)/glm_fp_pipe_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp_pipe"; exit 1; }
	@# glm_matmul_pipe: high-fmax bf16 GEMM on pipelined MACs (L-way interleaved accumulate).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_pipe_sim test/glm_matmul_pipe_tb.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_matmul_pipe"; $(VVP) $(BUILD_DIR)/glm_matmul_pipe_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_pipe"; exit 1; }
	@# spec_decode_top: MTP speculative-decode loop (glm_model_q4k + mtp_head_q4k + spec_decode_seq); spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_top_sim test/spec_decode_top_tb.v src/spec_decode_top.v src/glm_model_q4k.v src/mtp_head_q4k.v src/spec_decode_seq.v src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_decode_top"; $(VVP) $(BUILD_DIR)/spec_decode_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_top"; exit 1; }
	@# spec_batched_top / spec_chain_top run the FULL model forward many times (K x
	@#   scenarios) and are minutes-long in iverilog -- moved to `make spec-slow` so the
	@#   fast `unittests` path completes (same rationale as `bcov`). They still gate on
	@#   spec==greedy; run them via `make spec-slow`.
	@# expert_cache_ctrl: MoE expert-weight HBM cache controller (tag/LRU/miss-DMA), single-package system PoC.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_ctrl_sim test/expert_cache_ctrl_tb.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_cache_ctrl"; $(VVP) $(BUILD_DIR)/expert_cache_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_cache_ctrl"; exit 1; }
	@# expert_predictor: per-(layer,expert) frequency/locality prefetch predictor w/ confidence threshold (fine-grained MoE).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_predictor_sim test/expert_predictor_tb.v src/expert_predictor.v
	@printf '[%s] ' "expert_predictor"; $(VVP) $(BUILD_DIR)/expert_predictor_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_predictor"; exit 1; }
	@# spec_decode_seq: MTP speculative-decode controller (draft/verify/accept-reject; eff tok/pass = 1+alpha).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_seq_sim test/spec_decode_seq_tb.v src/spec_decode_seq.v
	@printf '[%s] ' "spec_decode_seq"; $(VVP) $(BUILD_DIR)/spec_decode_seq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_seq"; exit 1; }
	@# spec_decode_seq K>1: multi-token draft (DRAFT_K=1/2/3/4/6/8), spec==greedy exact + eff-tok/pass vs alpha.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_seq_k_sim test/spec_decode_seq_k_tb.v src/spec_decode_seq.v
	@printf '[%s] ' "spec_decode_seq(K>1)"; $(VVP) $(BUILD_DIR)/spec_decode_seq_k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_seq_k"; exit 1; }
	@# spec_depth_adapt: ADAPTIVE draft depth (runtime k_cur in [1..K] + accept-rate policy) --
	@# spec==greedy under ANY depth schedule (forced + closed-loop), ADAPT=0 default byte-compatible.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_depth_adapt_sim test/spec_depth_adapt_tb.v src/spec_decode_seq.v src/spec_depth_adapt.v
	@printf '[%s] ' "spec_depth_adapt"; $(VVP) $(BUILD_DIR)/spec_depth_adapt_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_depth_adapt"; exit 1; }
	@# kv_cache_pager: MLA latent-KV ring cache (append + DSA-gather + Flash overflow), single-module system.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_sim test/kv_cache_pager_tb.v src/kv_cache_pager.v
	@printf '[%s] ' "kv_cache_pager"; $(VVP) $(BUILD_DIR)/kv_cache_pager_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager"; exit 1; }
	@# kv_cache_pager at FULL RESIDENCY (RESIDENT == S_MAX == 2^POSW): the natural
	@#   "no row ever spills" sizing.  The old POSW-bit RESIDENT slice truncated
	@#   2^POSW to 0 and silently declared every row cold; this config pins the fix.
	@$(IVERILOG) $(IFLAGS) -Pkv_cache_pager_tb.RESIDENT=64 \
	    -o $(BUILD_DIR)/kv_cache_pager_full_sim test/kv_cache_pager_tb.v src/kv_cache_pager.v
	@printf '[%s] ' "kv_cache_pager(RESIDENT=S_MAX)"; $(VVP) $(BUILD_DIR)/kv_cache_pager_full_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager(RESIDENT=S_MAX) -- full-residency ring declared rows cold"; exit 1; }
	@# kv_cache_pager(ECC=1) (task C6-full): lane-partitioned SECDED on the real ring --
	@# single-bit corrected, double-bit detected, across the ragged 768-bit / 100-bit rows.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_ecc_sim test/kv_cache_pager_ecc_tb.v src/kv_cache_pager.v src/ecc_secded.v
	@printf '[%s] ' "kv_cache_pager(ECC)"; $(VVP) $(BUILD_DIR)/kv_cache_pager_ecc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager_ecc"; exit 1; }
	@# kv_cache_pager(NSEQ>1) (P1.3 per-row KV): NSEQ independent ring windows -- no
	@# cross-seq slot collision, independent per-seq eviction, flash_seq cold keying.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_multiseq_sim test/kv_cache_pager_multiseq_tb.v src/kv_cache_pager.v
	@printf '[%s] ' "kv_cache_pager(NSEQ>1)"; $(VVP) $(BUILD_DIR)/kv_cache_pager_multiseq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager_multiseq"; exit 1; }
	@# ddr5_xbar: N-channel banked DDR5 read fabric (address striping -> ~Nx aggregate bandwidth).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ddr5_xbar_sim test/ddr5_xbar_tb.v src/ddr5_xbar.v
	@printf '[%s] ' "ddr5_xbar"; $(VVP) $(BUILD_DIR)/ddr5_xbar_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ddr5_xbar"; exit 1; }
	@# flash_xbar: N-channel banked Flash read fabric -- deep per-channel outstanding queue hides NAND latency (~QDEPTH x), banking ~N x.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/flash_xbar_sim test/flash_xbar_tb.v src/flash_xbar.v
	@printf '[%s] ' "flash_xbar"; $(VVP) $(BUILD_DIR)/flash_xbar_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: flash_xbar"; exit 1; }
	@# boot_loader: power-up Flash->DDR5 resident-set (hot weights) DMA + ready handshake (chip must load model first).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/boot_loader_sim test/boot_loader_tb.v src/boot_loader.v
	@printf '[%s] ' "boot_loader"; $(VVP) $(BUILD_DIR)/boot_loader_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: boot_loader"; exit 1; }
	@# ecc_secded: (72,64) SECDED ECC for the DDR5/Flash path -- exhaustive single-correct + double-detect.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ecc_secded_sim test/ecc_secded_tb.v src/ecc_secded.v
	@printf '[%s] ' "ecc_secded"; $(VVP) $(BUILD_DIR)/ecc_secded_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ecc_secded"; exit 1; }
	@# clk_en_ctrl: work-driven clock-enable gating (die idles ~75% Flash-bound -> ~73% idle-power gated; never gates active work).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_en_ctrl_sim test/clk_en_ctrl_tb.v src/clk_en_ctrl.v
	@printf '[%s] ' "clk_en_ctrl"; $(VVP) $(BUILD_DIR)/clk_en_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_en_ctrl"; exit 1; }
	@# clk_throttle: DVFS/eco frequency prescaler (die runs f/div, byte-identical peak-power cap) + its clk_en_ctrl throttle path.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_throttle_sim test/clk_throttle_tb.v src/clk_throttle.v src/clk_en_ctrl.v
	@printf '[%s] ' "clk_throttle"; $(VVP) $(BUILD_DIR)/clk_throttle_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_throttle"; exit 1; }
	@# swiglu_expert: SwiGLU FFN expert (gate/up/down GEMM + silu*up), dense + MoE modes.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_sim test/swiglu_expert_tb.v src/swiglu_expert.v src/glm_matmul_pipe.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert"; $(VVP) $(BUILD_DIR)/swiglu_expert_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert"; exit 1; }
	@# moe_router: GEMV + sigmoid + top-K + renormalize-then-scale (GLM-5.2 MoE gating).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_sim test/moe_router_tb.v src/moe_router.v src/glm_matmul_pipe.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router"; $(VVP) $(BUILD_DIR)/moe_router_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router"; exit 1; }
	@# glm_softmax: numerically-stable bf16 softmax (MLA attention), full-denominator sum.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_softmax_sim test/glm_softmax_tb.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_softmax"; $(VVP) $(BUILD_DIR)/glm_softmax_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_softmax"; exit 1; }
	@# dsa_indexer: DSA/IndexShare sparse-attention indexer (index-score + top-K + dense fallback).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/dsa_indexer_sim test/dsa_indexer_tb.v src/dsa_indexer.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "dsa_indexer"; $(VVP) $(BUILD_DIR)/dsa_indexer_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: dsa_indexer"; exit 1; }
	@# mla_attn: MLA latent attention orchestrator (low-rank Q/KV + RoPE + DSA + softmax + *V + O proj).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_sim test/mla_attn_tb.v src/mla_attn.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn"; $(VVP) $(BUILD_DIR)/mla_attn_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn"; exit 1; }
	@# sampler: temperature + top-k/top-p + softmax + multinomial(LFSR) token sampling.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/sampler_sim test/sampler_tb.v src/sampler.v src/topk_select.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "sampler"; $(VVP) $(BUILD_DIR)/sampler_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: sampler"; exit 1; }
	@# glm_decoder_block: ONE full GLM-5.2 decoder layer (rmsnorm+mla_attn+residual+rmsnorm+FFN+residual).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_sim test/glm_decoder_block_tb.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block"; $(VVP) $(BUILD_DIR)/glm_decoder_block_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block"; exit 1; }
	@# glm_model: FULL GLM-5.2 forward pass (embed -> 6 layers dense/MoE -> norm -> LM head -> next-token logits).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_sim test/glm_model_tb.v src/glm_model.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model"; $(VVP) $(BUILD_DIR)/glm_model_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model"; exit 1; }
	@# mtp_head: GLM-5.2 multi-token-prediction head (t+2 speculative; num_nextn_predict_layers=1).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_sim test/mtp_head_tb.v src/mtp_head.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head"; $(VVP) $(BUILD_DIR)/mtp_head_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head"; exit 1; }
	@# AXI4-Lite MASTER DMA engine vs an AXI slave-memory BFM.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/axi_master_dma_sim test/axi_master_dma_tb.v src/axi_master_dma.v
	@printf '[%s] ' "axi_master_dma"; $(VVP) $(BUILD_DIR)/axi_master_dma_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: axi_master_dma"; exit 1; }
	@# Async CDC FIFO across two unrelated clocks (7ns vs 11ns).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/cdc_async_fifo_sim test/cdc_async_fifo_tb.v src/cdc_async_fifo.v
	@printf '[%s] ' "cdc_async_fifo"; $(VVP) $(BUILD_DIR)/cdc_async_fifo_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: cdc_async_fifo"; exit 1; }
	@# ---- P2 productization building blocks (ECC / reset-CDC / DFT-MBIST / power-ICG) ----
	@# ecc_mem_wrap: SECDED-protected synchronous RAM (encode on write, decode+correct/detect on read) -- exhaustive single-correct + double-detect.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ecc_mem_wrap_sim test/ecc_mem_wrap_tb.v src/ecc_mem_wrap.v src/ecc_secded.v
	@printf '[%s] ' "ecc_mem_wrap"; $(VVP) $(BUILD_DIR)/ecc_mem_wrap_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ecc_mem_wrap"; exit 1; }
	@# reset_sync: async-assert / sync-deassert reset synchronizer (CDC signoff) -- immediate assert, STAGES-edge clean deassert.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/reset_sync_sim test/reset_sync_tb.v src/reset_sync.v
	@printf '[%s] ' "reset_sync"; $(VVP) $(BUILD_DIR)/reset_sync_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: reset_sync"; exit 1; }
	@# mbist_ctrl: March C- memory BIST for a single-port SRAM (good RAM -> pass; stuck-at-0 cell -> fail + fail_addr).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mbist_ctrl_sim test/mbist_ctrl_tb.v src/mbist_ctrl.v
	@printf '[%s] ' "mbist_ctrl"; $(VVP) $(BUILD_DIR)/mbist_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mbist_ctrl"; exit 1; }
	@# mbist_ctrl_2p: dual-port (1R1W) March C- + concurrent write/read coupling BIST -- the collar for the
	@#   2-port resident stores (kv_cache_pager.ring / mla_attn_q4k.vstore_mem) the single-port engine cannot
	@#   test. good->pass; stuck-at->fail kind=0; concurrent-coupling (invisible to march)->fail kind=1.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mbist_ctrl_2p_sim test/mbist_ctrl_2p_tb.v src/mbist_ctrl_2p.v
	@printf '[%s] ' "mbist_ctrl_2p"; $(VVP) $(BUILD_DIR)/mbist_ctrl_2p_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mbist_ctrl_2p"; exit 1; }
	@# icg_cell: glitch-free integrated clock gate (low-phase enable latch + AND) -- turns clk_en into a real gated clock with no runt pulses.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/icg_cell_sim test/icg_cell_tb.v src/icg_cell.v
	@printf '[%s] ' "icg_cell"; $(VVP) $(BUILD_DIR)/icg_cell_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: icg_cell"; exit 1; }
	@# clk_gate_cluster (task C7): icg_cell + clk_en_ctrl gating a real leaf -- gated == free-running (bit-exact),
	@# idle => clock frozen, scan_enable => transparent, req=1 => enable=1 safety invariant.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_gate_cluster_sim test/clk_gate_cluster_tb.v src/clk_gate_cluster.v src/icg_cell.v src/clk_en_ctrl.v src/clk_gate_leaf.v
	@printf '[%s] ' "clk_gate_cluster"; $(VVP) $(BUILD_DIR)/clk_gate_cluster_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_gate_cluster"; exit 1; }
	@# kv_ecc_ring (task C6): lane-partitioned SECDED ring for wide (ragged, non-64-aligned) KV rows --
	@# single-bit corrected, double-bit detected, across the ragged final lane.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_ecc_ring_sim test/kv_ecc_ring_tb.v src/kv_ecc_ring.v src/ecc_secded.v
	@printf '[%s] ' "kv_ecc_ring"; $(VVP) $(BUILD_DIR)/kv_ecc_ring_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_ecc_ring"; exit 1; }
	@# ---- weight_decomp: on-chip streaming canonical-Huffman weight decompressor (the DECOMP=1
	@#      Flash->DDR5 path).  BIT-EXACT round-trip vs the offline encoder tools/fp8_huff.py.
	@#      Closes the gap where NO gate set DECOMP=1 and the decoder shipped functionally unverified. ----
	@$(MAKE) --no-print-directory weight-decomp
	@# ---- Q4_K local-device track (GGUF UD-Q4_K_XL bring-up; both tracks live -- see docs/Q4K_SYSTEM_PLAN.md) ----
	@$(MAKE) --no-print-directory q4k
	@# ---- ASSEMBLED glm_model_q4k full-forward vs numpy golden (fast SPEC_SLICE smoke;
	@#      the committed-slice `make model-q4k` is the thorough standalone gate) ----
	@$(MAKE) --no-print-directory model-q4k-smoke
	@echo "unittests: all per-unit TBs passed"

# weight-decomp: BIT-EXACT round-trip golden for src/weight_decomp.v, the on-chip
# streaming CANONICAL-Huffman weight decompressor that feeds decoded Q4_K bytes into
# every matmul when glm_q4k_system is built with DECOMP=1 (default 0).  It shipped with
# NO testbench and NO gate ever setting DECOMP=1 -- entirely unverified functionally.
# tools/fp8_huff.py is the matching OFFLINE encoder named in weight_decomp.v's own header
# (length-limited package-merge Huffman + canonical-code assignment; it self-checks
# encode==independent-decode + Kraft-completeness + len<=MAXLEN before emitting, so a
# nonzero exit fails the emit).  test/weight_decomp_tb.v loads the canonical tables,
# streams the compressed bytes through the REAL weight_decomp, and asserts EVERY decoded
# byte === the original byte, EOB fires, and the exact byte-count is produced -- over 7
# blocks spanning code lengths 1..15, single-symbol / near-uniform / skewed data, and
# output back-pressure.  Folded into `unittests`; runnable standalone.  Expected: 34368.
weight-decomp:
	@mkdir -p $(BUILD_DIR)
	@# offline encoder emits + self-checks the goldens (nonzero exit fails the emit).
	@python3 tools/fp8_huff.py >/dev/null            # -> build/weight_decomp_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_decomp_sim test/weight_decomp_tb.v src/weight_decomp.v
	@printf '[%s] ' "weight_decomp"; $(VVP) $(BUILD_DIR)/weight_decomp_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight_decomp"; exit 1; }

# decomp1-elab: ELABORATE glm_q4k_system with DECOMP=1 (the shipped default is 0).
# The leaf gate above proves weight_decomp bit-exact, but NO gate ever elaborated the
# SYSTEM's DECOMP=1 `g_wpath` else-branch -- the compressed-image fetch FSM + the
# weight_decomp instance wired to the system WD_*/RECON_DEPTH params + the byte-reassembly
# -> recon RAM -> loader-refill path (src/glm_q4k_system.v ~790-911).  A width /
# part-select / $clog2 bug there would only surface the day someone flips DECOMP on.
# test/decomp1_elab_wrap.v instantiates the top with .DECOMP(1); iverilog -tnull type/width
# elaborates that branch (NO sim -- like full-elab; a functional DECOMP=1 system run needs a
# full compressed Q4_K super-block image driven through the whole memory system, a separate
# larger task).  GLM_Q4K_SYS_SRCS is the system src list (defined by the `resident` gate).
decomp1-elab:
	@printf '[%s] ' "decomp1-elab(DECOMP=1 branch)"; \
	$(IVERILOG) -g2012 -I src -tnull -pfileline=1 test/decomp1_elab_wrap.v $(GLM_Q4K_SYS_SRCS) \
	    && echo "iverilog -tnull elaboration OK (glm_q4k_system @ DECOMP=1: g_wpath else-branch type/width-checked)" \
	    || { echo "FAILED: decomp1-elab (iverilog -tnull)"; exit 1; }

# Q4_K local-device sub-gate (docs/Q4K_SYSTEM_PLAN.md 4.1): the four verified Q4_K unit TBs
# (q4k_prim / glm_matmul_q4k / swiglu_expert_q4k / moe_router_q4k), bit-exact to ggml goldens.
# Wired into `unittests` above; runnable standalone as `make q4k`. Expected: 18/160/240/40.
q4k:
	@mkdir -p $(BUILD_DIR)
	@# q4k_prim: fp16->fp32 + get_scale_min_k4 primitives (q4k.vh) vs ggml golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/q4k_prim_sim test/q4k_prim_tb.v
	@printf '[%s] ' "q4k_prim"; $(VVP) $(BUILD_DIR)/q4k_prim_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: q4k_prim"; exit 1; }
	@# glm_matmul_q4k: Q4_K GEMM core, bit-exact to ggml dequantize_row_q4_K.
	@python3 tools/q4k_matmul_gen.py >/dev/null            # -> build/q4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_q4k_sim test/glm_matmul_q4k_tb.v src/glm_matmul_q4k.v
	@printf '[%s] ' "glm_matmul_q4k"; $(VVP) $(BUILD_DIR)/glm_matmul_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_q4k"; exit 1; }
	@# swiglu_expert_q4k: MoE expert (gate/up/down + silu) on the Q4_K core.
	@python3 tools/swiglu_q4k_gen.py >/dev/null            # -> build/swiglu_q4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_q4k_sim test/swiglu_expert_q4k_tb.v \
	    src/swiglu_expert_q4k.v src/glm_matmul_q4k.v src/glm_act.v
	@printf '[%s] ' "swiglu_expert_q4k"; $(VVP) $(BUILD_DIR)/swiglu_expert_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_q4k"; exit 1; }
	@# moe_router_q4k: gating GEMV -> sigmoid -> top-K -> renorm on the Q4_K core.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_q4k_sim test/moe_router_q4k_tb.v \
	    src/moe_router_q4k.v src/glm_matmul_q4k.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_q4k"; $(VVP) $(BUILD_DIR)/moe_router_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_q4k"; exit 1; }
	@# ---- mixed-type (Q6_K / Q8_0 / F16) consumer sub-gate ----
	@$(MAKE) --no-print-directory mixedtype
	@echo "q4k: all four Q4_K unit TBs + mixed-type (Q6_K/Q8_0/F16) sub-gate passed"

# model-q4k / model-q4k-smoke: the ASSEMBLED glm_model_q4k FULL-FORWARD golden gate --
#   closes the #1 correctness gap (the assembled Q4_K numeric path had NO functional
#   golden; the model-level TBs ran the generic bf16 twin, not the _q4k product).
#   tools/glm_model_q4k_tb_gen.py drives the assembled numpy golden
#   (tools/glm_model_q4k_ref.py) and emits weights+inputs+expected as $readmemh hex;
#   test/glm_model_q4k_full_tb.v drives the REAL product top glm_model_q4k with those
#   SAME weights and asserts logits+argmax+h_state BIT-EXACT vs the golden.  This is what
#   EXPOSED + LOCKS IN two real fixes: (1) the Phase-1 MLA softmax scale 1/sqrt(qk_head_dim)
#   (src/mla_attn_q4k.v, previously OMITTED), and (2) the glm_matmul_q4k narrow-k_cnt
#   out-of-range sub-block select (src/glm_matmul_q4k.v, latent for KMAX<128).
#
#   model-q4k-smoke : the SPEC_SLICE (MODEL_DIM=16/L=2/VOCAB=16) -- seconds/forward, so
#     it is FOLDED INTO `unittests` as a fast bit-exact assembled-forward regression gate.
#   model-q4k       : the committed slice (MODEL_DIM=128/L=6/VOCAB=256) -- each forward is
#     the full model, minutes-long in iverilog, so it is kept OUT of `unittests` (like
#     `spec-slow`); run explicitly as `make model-q4k`.  Both assert the identical
#     bit-exact contract; the slice only changes the leaf sizes.
MODEL_Q4K_SRCS := test/glm_model_q4k_full_tb.v src/glm_model_q4k.v src/glm_decoder_block_q4k.v \
	src/mla_attn_q4k.v src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v \
	src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v

model-q4k-smoke:
	@mkdir -p $(BUILD_DIR)
	@python3 tools/glm_model_q4k_tb_gen.py --spec >/dev/null   # -> build/mq4k_s/*.hex (SPEC_SLICE golden)
	@$(IVERILOG) $(IFLAGS) -D SPEC_SLICE -o $(BUILD_DIR)/glm_model_q4k_full_s_sim $(MODEL_Q4K_SRCS)
	@printf '[%s] ' "glm_model_q4k_full(spec)"; $(VVP) $(BUILD_DIR)/glm_model_q4k_full_s_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_q4k_full(spec)"; exit 1; }

model-q4k:
	@mkdir -p $(BUILD_DIR)
	@python3 tools/glm_model_q4k_tb_gen.py >/dev/null          # -> build/mq4k/*.hex (committed-slice golden)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_q4k_full_sim $(MODEL_Q4K_SRCS)
	@printf '[%s] ' "glm_model_q4k_full"; $(VVP) $(BUILD_DIR)/glm_model_q4k_full_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_q4k_full"; exit 1; }
	@echo "model-q4k: assembled glm_model_q4k full forward == numpy golden (BIT-EXACT logits+argmax+h_state)"

# Result-invariance gate for the ACT_HW resource knob: the SAME committed-slice
# golden vectors, decoded with the glm_act lane-serialized datapath (ACT_HW=1).
# ALL tests passing == byte-identical tokens/logits with the compact fit config's
# activation serialization (the claim fpga/synth_ku3p.tcl relies on).
model-q4k-acthw:
	@mkdir -p $(BUILD_DIR)
	@python3 tools/glm_model_q4k_tb_gen.py >/dev/null
	@$(IVERILOG) $(IFLAGS) -DTB_ACT_HW=1 -o $(BUILD_DIR)/glm_model_q4k_acthw_sim $(MODEL_Q4K_SRCS)
	@printf '[%s] ' "glm_model_q4k_full(ACT_HW=1)"; $(VVP) $(BUILD_DIR)/glm_model_q4k_acthw_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_q4k_full ACT_HW=1"; exit 1; }
	@echo "model-q4k-acthw: ACT_HW=1 (serialized glm_act) == same golden BIT-EXACT -> knob is result-invariant"

# Mixed-type (Q6_K / Q8_0 / F16) consumer sub-gate: the two per-type dequant-primitive
# TBs (q6k_prim / q8_0_prim) + the integrated mixed-column GEMM (glm_matmul_mixed) that
# drives one tile whose PE_N columns carry different w_type -- exercising the w_type mux,
# all four decoders, the high-precision buses, and accumulator reset.  All bit-exact to
# tools/q4k_ref.py via the tools/q4k_mixed_gen.py goldens, plus weight_loader_q4k (the
# pure-Q4_K loader->GEMM feed) and weight_loader_q4k_mixed which proves the LOADER's
# mixed-type DMA feed (not just the GEMM front-end) bit-exact over a mixed sequence of
# consecutive different-type tiles.  Folded into `q4k` above (hence `unittests`);
# runnable standalone as `make mixedtype`.  Expected: 91220 / 10816 / 32 / 160 / 192.
mixedtype:
	@mkdir -p $(BUILD_DIR)
	@# emit + self-test the goldens (s8 / per-type dequant / mixed GEMM + prim vectors);
	@# the generator's 86,984-check self-test vs q4k_ref gates the emit (nonzero exit fails).
	@python3 tools/q4k_mixed_gen.py >/dev/null            # -> build/{s8_fp32,q4k_deq,q4k_mixed,q6k_prim,q8_0_prim,f16_prim}_vec.txt
	@# q6k_prim: s8_to_fp32 (exhaustive) + Q6_K raw/code + F16 dequant prims (q4k_mixed.vh) vs ggml golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/q6k_prim_sim test/q6k_prim_tb.v
	@printf '[%s] ' "q6k_prim"; $(VVP) $(BUILD_DIR)/q6k_prim_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: q6k_prim"; exit 1; }
	@# q8_0_prim: Q8_0 raw/code dequant prim (q4k_mixed.vh) vs ggml golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/q8_0_prim_sim test/q8_0_prim_tb.v
	@printf '[%s] ' "q8_0_prim"; $(VVP) $(BUILD_DIR)/q8_0_prim_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: q8_0_prim"; exit 1; }
	@# glm_matmul_mixed: one GEMM tile with Q4_K/Q6_K/Q8_0/F16 columns -> the w_type mux +
	@# all four decoders + accumulator reset, bit-exact vs q4k_ref matmul_q4k_col.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_mixed_sim test/glm_matmul_mixed_tb.v src/glm_matmul_q4k.v
	@printf '[%s] ' "glm_matmul_mixed"; $(VVP) $(BUILD_DIR)/glm_matmul_mixed_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_mixed"; exit 1; }
	@# weight_loader_q4k: the LOADER's pure-Q4_K feed -- storage-layout image -> header
	@# (d/dmin/scales) super-block packing + 4-bit code stream -> glm_matmul_q4k pull,
	@# bit-exact vs the ggml Q4_K golden (tools/q4k_ref.py via tools/q4k_matmul_gen.py).
	@python3 tools/q4k_matmul_gen.py 40 2 2 build/wlq4k_vec.txt >/dev/null   # -> build/wlq4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_loader_q4k_sim test/weight_loader_q4k_tb.v src/weight_loader_q4k.v src/glm_matmul_q4k.v
	@printf '[%s] ' "weight_loader_q4k"; $(VVP) $(BUILD_DIR)/weight_loader_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight_loader_q4k"; exit 1; }
	@# weight_loader_q4k_mixed: the LOADER's mixed-type DMA feed (Q6_K/Q8_0/F16 per-type
	@# header packing + code stream + desc_wtype geometry) driving glm_matmul_q4k, over a
	@# MIXED SEQUENCE of consecutive different-type tiles, bit-exact vs q4k_ref matmul.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_loader_q4k_mixed_sim test/weight_loader_q4k_mixed_tb.v src/weight_loader_q4k.v src/glm_matmul_q4k.v
	@printf '[%s] ' "weight_loader_q4k_mixed"; $(VVP) $(BUILD_DIR)/weight_loader_q4k_mixed_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight_loader_q4k_mixed"; exit 1; }
	@echo "mixedtype: Q6_K/Q8_0/F16 prim TBs + mixed-column GEMM + loader->GEMM pure-Q4_K and mixed feeds passed (bit-exact vs q4k_ref)"

# spec-slow: the speculative-decode top harnesses. They run the full model forward
#   many times (K x accept/reject/mixed scenarios) -> minutes-long in iverilog, so they
#   are kept OUT of `unittests`; run explicitly. Both gate on spec==greedy (safety).
spec-slow:
	@mkdir -p $(BUILD_DIR)
	@# spec_batched_top: batched-verify -- K+1 draft positions in ONE PE_M=K+1 model weight-load; spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_batched_top_sim test/spec_batched_top_tb.v src/spec_batched_top.v src/glm_model_q4k.v src/spec_decode_seq.v src/mtp_head_q4k.v src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_batched_top"; $(VVP) $(BUILD_DIR)/spec_batched_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_batched_top"; exit 1; }
	@# spec_chain_top (task B8): K-step MTP chain (K=2 + K=4 engines); committed stream == greedy rollout EXACT (spec==greedy).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_chain_top_sim test/spec_chain_top_tb.v src/spec_chain_top.v src/mtp_head_q4k.v src/glm_model_q4k.v src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/spec_decode_seq.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_chain_top"; $(VVP) $(BUILD_DIR)/spec_chain_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_chain_top"; exit 1; }
	@echo "spec-slow: spec_batched_top + spec_chain_top passed (spec==greedy)"

# spec-adapt: adaptive draft depth (runtime-variable K).  spec_decode_seq(ADAPT=1)
#   honors a per-pass k_cur in [1..DRAFT_K]; spec_depth_adapt picks k_cur from the
#   observed accept results (streak-counter policy).  Gate: spec==greedy under ANY
#   depth schedule (forced, closed-loop-vs-software-model, and ADAPT=0 default-off),
#   K=2/3/4/6/8.  Same TB also runs inside `unittests`; this is the standalone gate.
spec-adapt:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_depth_adapt_sim test/spec_depth_adapt_tb.v src/spec_decode_seq.v src/spec_depth_adapt.v
	@printf '[%s] ' "spec_depth_adapt"; $(VVP) $(BUILD_DIR)/spec_depth_adapt_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_depth_adapt"; exit 1; }

# expert-cache: the two product-relevant expert-cache TBs (recipes restored from
# cbef69d, which removed them with the FP8-only `cache-study` target -- these two
# exercise the KEPT product RTL src/expert_cache_pf.v + src/expert_cache_ctrl.v):
#   expert_cache_pf_tb        -- demand-identical prefetch overlay + GLM-scale
#                                prefetch stall-cut study (623 checks, ~17 min).
#   expert_cache_pf_policy_tb -- LRU vs FREQ replacement on the REAL decode
#                                routing trace tools/glm_trace.hex (~4 min).
# Minutes-long in iverilog -> standalone (in `release-gate`, not `unittests`).
# (test/expert_prefetch_top_tb.v is study-only and ~11 min -- not wired; run by hand.)
expert-cache:
	@mkdir -p $(BUILD_DIR)
	@# regenerate the routing trace (deterministic seed) so the target is self-contained.
	@python3 tools/route_trace.py --dump >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_pf test/expert_cache_pf_tb.v src/expert_cache_pf.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_cache_pf"; $(VVP) $(BUILD_DIR)/expert_cache_pf | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_cache_pf"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_pf_policy test/expert_cache_pf_policy_tb.v src/expert_cache_pf.v
	@printf '[%s] ' "cache-policy(LRU vs FREQ)"; $(VVP) $(BUILD_DIR)/expert_cache_pf_policy | grep -E 'ALL POLICY TESTS PASSED' \
	    || { echo "FAILED: expert_cache_pf_policy"; exit 1; }
	@echo "expert-cache: expert_cache_pf + expert_cache_pf_policy passed"

# Formal (bounded model checking) of the memory-system controllers via yosys write_smt2 +
# yosys-smtbmc -s z3.  Each harness test/formal/<dut>_fv.v instantiates the committed controller
# read-only and asserts safety properties.  The mandatory `async2sync; chformal -lower` lowers
# yosys $check cells so the asserts are NOT silently dropped (a vacuous-pass trap); the assert
# count > 0 guard re-checks non-vacuity per model.  See docs/FORMAL.md.  Bounds kept modest for a
# routine run; deeper bounds (e.g. expert_cache_pf K=55) are in docs/FORMAL.md.
FV_DIR := scratchpad
define run_bmc   # $(1)=dut name  $(2)=extra read deps  $(3)=extra yosys (e.g. chparam)  $(4)=bound K  $(5)=optional artifact/report label (default: $(1); use when one dut runs in several chparam modes)
	@yosys -p "read_verilog -sv -formal -I src src/$(1).v $(2) test/formal/$(1)_fv.v; $(3) \
	          prep -top $(1)_fv -flatten; memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(or $(5),$(1))_fv.smt2" > $(FV_DIR)/$(or $(5),$(1))_fv_build.log 2>&1 \
	    || { echo "FAILED(build): $(or $(5),$(1))"; cat $(FV_DIR)/$(or $(5),$(1))_fv_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(or $(5),$(1))_fv.smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions in smt2): $(or $(5),$(1))"; exit 1; }
	@yosys-smtbmc -s z3 -t $(4) $(FV_DIR)/$(or $(5),$(1))_fv.smt2 > $(FV_DIR)/$(or $(5),$(1))_fv_bmc.log 2>&1 \
	    && printf '[formal %-16s] PASSED  K=%s  (%s asserts)\n' "$(or $(5),$(1))" "$(4)" "`grep -ic assert $(FV_DIR)/$(or $(5),$(1))_fv.smt2`" \
	    || { echo "FAILED(BMC counterexample): $(or $(5),$(1))"; tail -20 $(FV_DIR)/$(or $(5),$(1))_fv_bmc.log; exit 1; }
endef

# Named-harness BMC (harness basename != dut name, and a custom read-source list).
# Used for kv_cache_pager_ecc_fv (proves the ECC=1 lane-SECDED ring datapath).
define run_bmc_named   # $(1)=harness basename  $(2)=read sources  $(3)=bound K
	@yosys -p "read_verilog -sv -formal -I src $(2) test/formal/$(1).v; \
	          prep -top $(1) -flatten; memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(1).smt2" > $(FV_DIR)/$(1)_build.log 2>&1 \
	    || { echo "FAILED(build): $(1)"; cat $(FV_DIR)/$(1)_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(1).smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions): $(1)"; exit 1; }
	@yosys-smtbmc -s z3 -t $(3) $(FV_DIR)/$(1).smt2 > $(FV_DIR)/$(1)_bmc.log 2>&1 \
	    && printf '[formal %-20s] PASSED  K=%s  (%s asserts)\n' "$(1)" "$(3)" "`grep -ic assert $(FV_DIR)/$(1).smt2`" \
	    || { echo "FAILED(BMC counterexample): $(1)"; tail -20 $(FV_DIR)/$(1)_bmc.log; exit 1; }
endef

formal:
	@mkdir -p $(FV_DIR)
	$(call run_bmc,ddr5_xbar,,,12)
	$(call run_bmc,flash_xbar,,,12)
	$(call run_bmc,boot_loader,,,16)
	$(call run_bmc,clk_throttle,,,16)
	$(call run_bmc,spec_decode_seq,,,20)
	$(call run_bmc,kv_cache_pager,,,16)
	$(call run_bmc,expert_cache_pf,src/expert_cache_ctrl.v,chparam -set PF_ENABLE 0 expert_cache_pf_fv;,20)
	@# expert_cache_pf PF_ENABLE=1 (adversarial prefetch): bounded demand-response
	@# liveness P3 -- a demand txn is answered within LIVE_BOUND cycles even while
	@# best-effort prefetch contends for the shared Flash channel (harness header).
	$(call run_bmc,expert_cache_pf,src/expert_cache_ctrl.v,chparam -set PF_ENABLE 1 expert_cache_pf_fv;,20,expert_cache_pf_pf1)
	@# kv_cache_pager ECC=1 datapath (task C6-full followup): the lane-SECDED ring preserves
	@# encode-decode identity + window/in-bounds + no-false-alarm (single-lane; see harness header).
	$(call run_bmc_named,kv_cache_pager_ecc_fv,src/kv_cache_pager.v src/ecc_secded.v,12)
	@echo "formal: 9 BMC runs -- 7 controllers (expert_cache_pf in both PF=0 and PF=1 modes) + the ECC=1 pager datapath -- proven at bound (no counterexample); see docs/FORMAL.md"

# UNBOUNDED proof via temporal k-INDUCTION (yosys-smtbmc -i): base case + induction
# step => the asserts hold on ALL reachable states, not just the first K cycles.
# The step needs the design's reachable state space pinned; harnesses add
# STRENGTHENING INVARIANT asserts (over the DUT's primary I/O + harness shadow
# regs -- this yosys build has no internal observability) until the step closes.
define run_kind  # $(1)=dut name  $(2)=ind-harness basename  $(3)=K  $(4)=extra yosys (e.g. connect-bind)
	@yosys -p "read_verilog -sv -formal -I src src/$(1).v test/formal/$(2).v; \
	          prep -top $(2) -flatten; $(4) memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(2).smt2" > $(FV_DIR)/$(2)_build.log 2>&1 \
	    || { echo "FAILED(build): $(2)"; cat $(FV_DIR)/$(2)_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(2).smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions): $(2)"; exit 1; }
	@yosys-smtbmc -s z3 -i -t $(3) $(FV_DIR)/$(2).smt2 > $(FV_DIR)/$(2)_kind.log 2>&1 \
	    && printf '[k-induction %-20s] PROVEN UNBOUNDED  K=%s  (%s asserts)\n' "$(2)" "$(3)" "`grep -ic assert $(FV_DIR)/$(2).smt2`" \
	    || { echo "FAILED(induction step): $(2)"; tail -20 $(FV_DIR)/$(2)_kind.log; exit 1; }
endef

# flash_xbar response-FIFO / outstanding proof needs the DUT's OWN per-channel counters
# (u_dut.outst[c], u_dut.cnt[c]) in the inductive hypothesis.  yosys 0.66 cannot reference
# them from Verilog (no hierarchical refs), so the harness declares `(* keep *)` UNDRIVEN
# probe wires and we wire them to the flattened DUT registers post-flatten with `connect`.
# The TRAILING SPACE before each `;` is load-bearing: it terminates the escaped bracketed
# id \u_dut.outst[0] (otherwise [0] is parsed as a bit-select of a non-existent wire).
FLASH_IND_CONN := connect -set \dut_outst0 \u_dut.outst[0] ; connect -set \dut_outst1 \u_dut.outst[1] ; connect -set \dut_cnt0 \u_dut.cnt[0] ; connect -set \dut_cnt1 \u_dut.cnt[1] ;
# ddr5_xbar response-FIFO proof (task C5) uses the same connect-bind trick to reach
# the DUT's per-channel response-FIFO occupancy counters cnt[0..N_CH-1] (N_CH=2 slice).
DDR5_IND_CONN := connect -set \dut_cnt0 \u_dut.cnt[0] ; connect -set \dut_cnt1 \u_dut.cnt[1] ;
formal-ind:
	@mkdir -p $(FV_DIR)
	$(call run_kind,boot_loader,boot_loader_ind_fv,8)
	$(call run_kind,kv_cache_pager,kv_cache_pager_ind_fv,16)
	$(call run_kind,spec_decode_seq,spec_decode_seq_ind_fv,2)
	$(call run_kind,ddr5_xbar,ddr5_xbar_ind_fv,12,$(DDR5_IND_CONN))
	$(call run_kind,flash_xbar,flash_xbar_ind_fv,3,$(FLASH_IND_CONN))
	@echo "formal-ind: boot_loader done-gate proven UNBOUNDED; kv_cache_pager append/gather in-bounds + window invariants proven UNBOUNDED; spec_decode_seq token-accounting equality + per-cycle modular increment bounds + step-form (non-decreasing-except-wrap) monotonicity proven UNBOUNDED (k-induction K=2); ddr5_xbar request-path routing safety (exclusive one-hot routing / banked-channel selection / ready coherence / payload integrity) proven UNBOUNDED + response-FIFO no-overflow/underflow (cnt[c]<=RESP_QD conservation form, inflight<=CAP) proven UNBOUNDED via connect-bound internal cnt[] counters (task C5) -- tag-issued stays BOUNDED (FIFO-content data-invariant; the FIFO is a 2-D memory cell, not connect-bindable); strict unsigned monotonicity stays BOUNDED (32-bit counter wrap); flash_xbar per-channel-queue no-overflow (cnt[c]<=QDEPTH) + outstanding<=N_CH*QDEPTH (P3) + inflight<=outstanding (P1a/P1b) proven UNBOUNDED via connect-bound internal counters (k-induction K=3) -- tag-issued (P2) stays BOUNDED (needs FIFO-content data-invariant; FIFO is a 2-D memory cell, not connect-bindable) -- see docs/FORMAL.md"

# Verilator lint of the GLM product top (diagnostic, NOT part of `all`): the GLM
# datapath is not yet -Wall clean (width/pin warnings under review), so this is
# informational.  The fidelity gate is `unittests` + `synth-glm` + `formal`.
lint:
	$(VERILATOR) --lint-only -Wall -Isrc --top-module glm_q4k_system_cdc $(GLM_Q4K_CDC_SRCS)

# Host software scaffold (D2): OpenAI-compatible server + device protocol (stdlib).
host-test:
	@printf '[host-test] '; python3 host/test_wpu.py | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: host-test (a self-test raised; banner absent)"; exit 1; }

GLM_Q4K_CDC_SRCS := src/glm_q4k_system_cdc.v src/glm_q4k_system.v src/cdc_async_fifo.v \
	src/reset_sync.v src/glm_model_q4k.v src/ddr5_xbar.v src/weight_loader_q4k.v \
	src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
	src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v \
	src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v \
	src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
	src/glm_fp_pipe.v

# ---- RESIDENT=1 (LPDDR5X full-residency, rung-3) refill-path gate -----------
# Proves the glm_q4k_system RESIDENT=1 mode behaviorally: expert-cache refills
# are served by ddr5_xbar (TAG_EFILL) and the system flash_req NEVER fires (a
# per-cycle monitor $fatal's on any leak), while the RESIDENT=0 twin in the same
# TB routes the identical stimulus to the single Flash channel with ZERO xbar
# reads (the byte-identical default).  Standalone (not in `unittests`): the
# default-parameter product build is already gated by unittests+synth-glm.
GLM_Q4K_SYS_SRCS := src/glm_q4k_system.v src/glm_model_q4k.v src/ddr5_xbar.v \
	src/weight_loader_q4k.v src/expert_cache_pf.v src/expert_cache_ctrl.v \
	src/kv_cache_pager.v src/glm_decoder_block_q4k.v src/mla_attn_q4k.v \
	src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v \
	src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v \
	src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v \
	src/sampler.v src/glm_fp_pipe.v src/weight_decomp.v src/ecc_secded.v
resident:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_system_resident_sim \
	    test/glm_q4k_system_resident_tb.v $(GLM_Q4K_SYS_SRCS)
	@printf '[%s] ' "glm_q4k_system(RESIDENT)"; $(VVP) $(BUILD_DIR)/glm_q4k_system_resident_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_q4k_system_resident"; exit 1; }

# RESIDENT=0 identity proof: glm_q4k_system with RESIDENT=0 (the shipped default)
# folds to the SAME netlist as the version at git rev RESIDENT_BASE (the
# pre-RESIDENT commit).  Submodules are read -lib (interface blackboxes) so the
# proof is scoped to exactly the edited module.
#
# METHOD (structural netlist identity at the coarse RTLIL level), and WHY NOT SAT.
#   The obvious method -- yosys equiv_make + equiv_simple + equiv_induct, sequential
#   SAT equivalence -- CANNOT run on this design, and the reason is fundamental, not
#   a budget problem:
#     1. equiv_induct's SAT has no model for a MEMORY cell ($mem_v2).  glm_q4k_system
#        has the async-read `efifo` (glm_q4k_system.v:613); prep leaves it as $mem_v2
#        and the proof dies "No SAT model available for cell efifo".  (Fixable with
#        memory_map -- efifo is 16x3b -- but that only exposes the next wall.)
#     2. It also has no model for a STATEFUL BLACKBOX.  The six -lib submodules
#        (glm_model_q4k, ddr5_xbar, expert_cache_pf, kv_cache_pager, weight_loader_q4k,
#        expert_cache_ctrl) are FSM/memory cells; equiv_induct dies "No SAT model
#        available for cell u_ecache".  (weight-ecc-equiv's single blackbox works
#        ONLY because ecc_secded is COMBINATIONAL.)  `-ignore-unknown-cells` treats
#        each side's blackbox outputs as INDEPENDENT free vars -> 6261 unproven
#        $equiv, a false FAIL, measured.
#     3. The sound fix (expose -cut the blackbox boundary into shared cut-point ports)
#        FAILS to match: RESIDENT renamed a boundary net (`arb_ec_req`, absent in base),
#        so the exposed ports differ between the two revisions -- equiv_make: "Can't
#        match gate port arb_ec_req.i_gate".  Closing that needs hand-written interface
#        adapters per changed signal (exactly what cdc-protocol-equiv's header records
#        was done INTERACTIVELY) -- brittle, and wrong for an automated gate.
#   Full whitebox SAT (elaborate glm_model_q4k for real) is intractable: it does not
#   even finish `proc` in 6 min at this config.
#
#   So this gate proves netlist identity STRUCTURALLY, the same class of proof
#   cdc-protocol-equiv uses, but at the COARSE RTLIL level (prep -flatten; memory_map;
#   opt -full) instead of the gate level.  The coarse level is deliberate: at the
#   gate level, abc picks a different-but-equivalent decomposition for RESIDENT=0 vs
#   base (876 vs 873 cells, MUX<->NAND trades -- a mapping artifact, NOT a logic diff),
#   which would false-FAIL.  Before abc, every RESIDENT=0 fold (ef_*=0, the ternaries
#   collapsing to the original wiring, the tied-off S10 generate) constant-folds away
#   and the coarse cell netlist is BYTE-FOR-BYTE the base's.  For a change that is
#   nothing but default-off constant folding, coarse-histogram identity is structural
#   identity: any surviving RESIDENT logic shows up as extra coarse cells.
#
#   SELF-VALIDATING.  A structural-identity gate is worthless if the comparison can't
#   tell things apart, so this gate ALSO asserts RESIDENT=1 does NOT match base (it
#   differs by 20 histogram lines).  Both must hold: RESIDENT=0==base AND
#   RESIDENT=1!=base.  A pass therefore means the netlist is unchanged AND the check
#   that established it is live.
RESIDENT_BASE ?= 05639bf
RESIDENT_LIBS := src/glm_model_q4k.v src/ddr5_xbar.v src/weight_loader_q4k.v \
	src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v
# coarse-RTLIL cell histogram of glm_q4k_system: $1=source $2=chparam-or-empty $3=out
define RESIDENT_COARSE
$(YOSYS) -q -p "read_verilog -lib -sv -I src $(RESIDENT_LIBS); \
	read_verilog -sv -I src $(1); hierarchy -top glm_q4k_system; $(2) \
	prep -top glm_q4k_system -flatten; memory_map; opt -full; opt_clean -purge; \
	tee -o $(3) stat" >/dev/null 2>&1; \
	sed -n '/[0-9] cells$$/,$$p' $(3)
endef
resident-equiv:
	@mkdir -p $(BUILD_DIR)
	@git show $(RESIDENT_BASE):src/glm_q4k_system.v > $(BUILD_DIR)/glm_q4k_system_base.v
	@$(call RESIDENT_COARSE,$(BUILD_DIR)/glm_q4k_system_base.v,,$(BUILD_DIR)/res_base.txt) > $(BUILD_DIR)/res_base_cells.txt
	@$(call RESIDENT_COARSE,src/glm_q4k_system.v,chparam -set RESIDENT 0 glm_q4k_system;,$(BUILD_DIR)/res_r0.txt) > $(BUILD_DIR)/res_r0_cells.txt
	@$(call RESIDENT_COARSE,src/glm_q4k_system.v,chparam -set RESIDENT 1 glm_q4k_system;,$(BUILD_DIR)/res_r1.txt) > $(BUILD_DIR)/res_r1_cells.txt
	@diff $(BUILD_DIR)/res_base_cells.txt $(BUILD_DIR)/res_r0_cells.txt >/dev/null \
	    || { echo "FAILED: resident-equiv (RESIDENT=0 netlist != $(RESIDENT_BASE))"; \
	         diff $(BUILD_DIR)/res_base_cells.txt $(BUILD_DIR)/res_r0_cells.txt; exit 1; }
	@if diff $(BUILD_DIR)/res_base_cells.txt $(BUILD_DIR)/res_r1_cells.txt >/dev/null; then \
	    echo "FAILED: resident-equiv self-test (RESIDENT=1 == base: the check is blind)"; exit 1; fi
	@echo "[resident-equiv] PROVEN: glm_q4k_system(RESIDENT=0) coarse netlist == $(RESIDENT_BASE); RESIDENT=1 differs (check is live)"

# ---------------------------------------------------------------------------
# self-kv-roundtrip : KV latent WRITE-BACK round-trip (KV_WRITEBACK_DESIGN step 2).
#   glm_q4k_system(SELF_KV=1) closes the KV loop -- the die's committed latent is
#   appended to kv_cache_pager and gathered back into the model's kc_ckv/kc_krope.
#   A multi-token self-attending decode must match an INDEPENDENT standalone
#   glm_model_q4k whose KV is delivered by a TB shadow (same latents, transport
#   by a TB array instead of the pager).  BIT-EXACT committed-token binding, every
#   token; $fatal on mismatch; terminal banner "ALL N TESTS PASSED".
# ---------------------------------------------------------------------------
self-kv-roundtrip:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_self_kv_roundtrip_sim \
	    test/glm_q4k_self_kv_roundtrip_tb.v $(GLM_Q4K_SYS_SRCS)
	@printf '[%s] ' "glm_q4k_system(SELF_KV=1,kv-roundtrip)"; \
	    $(VVP) $(BUILD_DIR)/glm_q4k_self_kv_roundtrip_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: self-kv-roundtrip"; exit 1; }

# ---------------------------------------------------------------------------
# self-kv-l6-roundtrip : PER-(LAYER,POSITION) KV keying at L>1 (step 3).
#   glm_q4k_system(SELF_KV=1) at L=6 (3 dense + 3 MoE layers): each token runs 6
#   layers; every layer m gathers + appends its OWN KV, keyed (layer,position) by
#   the pager's per-layer windows (NSEQ=L, seq=db_layer).  A 6-token self-attending
#   decode must match an INDEPENDENT standalone glm_model_q4k whose KV is delivered
#   by a TB shadow keyed by (LAYER,pos) -- so a layer that read another layer's
#   window would DIVERGE.  BIT-EXACT full-logit + committed-token binding, every
#   token; $fatal on mismatch; terminal banner "ALL N TESTS PASSED".
#   INJECTION (proves the check catches layer-aliasing): drop the layer term on the
#   gather side in glm_q4k_system (pg_gather_seq = {KV_SEQW{1'b0}}); the L>1 binding
#   FAILS at the first real-KV gather of a layer m>0.  Revert after confirming.
# ---------------------------------------------------------------------------
self-kv-l6-roundtrip:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_self_kv_l6_roundtrip_sim \
	    test/glm_q4k_self_kv_l6_roundtrip_tb.v $(GLM_Q4K_SYS_SRCS)
	@printf '[%s] ' "glm_q4k_system(SELF_KV=1 L=6 per-(layer,pos) KV round-trip)"; \
	    $(VVP) $(BUILD_DIR)/glm_q4k_self_kv_l6_roundtrip_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: self-kv-l6-roundtrip"; exit 1; }

# self-kv-equiv : SELF_KV=0 BYTE-IDENTICAL core proof (resident-equiv method).
#   Coarse-RTLIL CELL histogram of glm_q4k_system(SELF_KV=0) == the pre-write-back
#   base at $(SELF_KV_BASE) -- the additive latent-observation ports (kv_lat_row,
#   kv_lat_valid) add wires/ports only, ZERO logic cells; SELF_KV=1 differs (the
#   gate is live, not blind).  Submodules read -lib (interface blackboxes) so the
#   proof is scoped to the edited top exactly like resident-equiv.
SELF_KV_BASE ?= f1a436a
self-kv-equiv:
	@mkdir -p $(BUILD_DIR)
	@git show $(SELF_KV_BASE):src/glm_q4k_system.v > $(BUILD_DIR)/skv_base.v
	@$(call RESIDENT_COARSE,$(BUILD_DIR)/skv_base.v,,$(BUILD_DIR)/skv_base.txt) > $(BUILD_DIR)/skv_base_cells.txt
	@$(call RESIDENT_COARSE,src/glm_q4k_system.v,chparam -set SELF_KV 0 glm_q4k_system;,$(BUILD_DIR)/skv_s0.txt) > $(BUILD_DIR)/skv_s0_cells.txt
	@$(call RESIDENT_COARSE,src/glm_q4k_system.v,chparam -set SELF_KV 1 glm_q4k_system;,$(BUILD_DIR)/skv_s1.txt) > $(BUILD_DIR)/skv_s1_cells.txt
	@diff $(BUILD_DIR)/skv_base_cells.txt $(BUILD_DIR)/skv_s0_cells.txt >/dev/null \
	    || { echo "FAILED: self-kv-equiv (SELF_KV=0 core cell histogram != $(SELF_KV_BASE))"; \
	         diff $(BUILD_DIR)/skv_base_cells.txt $(BUILD_DIR)/skv_s0_cells.txt; exit 1; }
	@if diff $(BUILD_DIR)/skv_base_cells.txt $(BUILD_DIR)/skv_s1_cells.txt >/dev/null; then \
	    echo "FAILED: self-kv-equiv self-test (SELF_KV=1 == base: the check is blind)"; exit 1; fi
	@echo "[self-kv-equiv] PROVEN: glm_q4k_system(SELF_KV=0) coarse CELL histogram == $(SELF_KV_BASE) (core byte-identical; +2 additive latent-observation ports, 0 cells); SELF_KV=1 differs (check is live)"

# ---------------------------------------------------------------------------
# dsa-thread-equiv : DSA_REAL_IDX threaded system -> model -> decoder -> mla_attn.
#
#   WHY THIS EXISTS.  DSA_REAL_IDX used to be a parameter of mla_attn_q4k ONLY: no
#   parent passed it, so the production hierarchy hard-wired it to its default 0 and
#   =1 was UNREACHABLE from any top.  At 0 the indexer is fed zero key-index vectors,
#   so every key scores 0 and top-K keeps keys 0..min(S,TOPK)-1 by tie-break --
#   QUERY-INDEPENDENT (mla_attn_q4k.v:155-158).  That is invisible at the committed
#   S_MAX=8/TOPK_ATTN=8 (dense: min(8,8)=8 = every key, and mla_attn_q4k.v:165-169
#   says the dense path never pulls keys at all, so it is a no-op for ANY value) --
#   and it is catastrophic the moment S_MAX > TOPK_ATTN, where every query at every
#   position would attend ONLY to the first TOPK tokens of the sequence.  Fluent
#   output, frozen prefix, and green tests: nothing asserts WHICH keys were selected.
#   So raising the context window must be a DECISION about attention, not an accident
#   of an unthreaded default.  =1 is already proven bit-exact at the leaf by
#   `make mla-sparse` (PE_M=3, per-row q-dependent DSA); this gate proves the
#   THREADING changed nothing at the default.
#
#   WHAT THIS GATE WOULD PROVE IF IT PASSED: glm_decoder_block_q4k at DSA_REAL_IDX=0
#   (default) is BYTE-IDENTICAL to the pre-threading netlist -- with mla_attn_q4k
#   elaborated as a REAL module, not a blackbox.
#
#   STATUS (2026-07): NOT GREEN, AND OPT-IN -- deliberately NOT in release-gate.
#   Measured on this machine (M-series, 20 cores, yosys 0.66), three ways, none finished:
#     equiv_simple+equiv_induct, real mla_attn   30m45s, 5.29 GB RSS, still growing
#     the same, earlier attempt                  >13 min, killed
#     RTLIL diff, hierarchy+proc, NO memory pass  >9 min per side, still in proc
#   `prep`/`proc` alone blow up on a decoder-sized module here; the SAT layer never even
#   gets a fair run.  This is a property of the machine + tool, not of the design.
#
#   WHY THAT IS NOT A HOLE.  The load-bearing claim is that the parameter TRAVERSES
#   system -> model -> decoder -> mla_attn, and that is machine-proven by
#   `make dsa-sparse-correct` (below), which did not exist when this gate was written:
#     DSA=0  tokens 12,14,14,14   system == standalone ref
#     DSA=1  tokens 12,14, 2, 2   system == standalone ref
#   If ANY link dropped the value, =1 would produce =0's tokens.  It does not.  The =0
#   row is the same "threading changed nothing at the default" claim, checked
#   behaviourally against the reference simulator instead of structurally.
#   The residue this gate would add is BYTE-IDENTICAL NETLIST at =0 -- which no claim in
#   docs/R3_APPLIANCE_SPEC.md rests on.  The area question ("what does =1 cost?") is
#   answered by `make lane-scaling-sparse` with yosys stat: +0.2%.  Structurally,
#   DSA_REAL_IDX occurs in glm_decoder_block_q4k.v exactly three times -- comment,
#   declaration (default 0), and the one forwarding connection at line 344.  It sizes
#   nothing and gates no generate, so at 0 forwarded to a callee that already defaulted
#   to 0, the LRM gives identical elaboration.
#
#   So: a sound argument plus a behavioural machine check, with the structural check
#   parked as opt-in.  Run it somewhere with a bigger budget before writing the words
#   "byte-identical" anywhere.  Do NOT re-add it to release-gate without first showing
#   it completes -- an unrunnable gate does not raise the bar, it makes the whole suite
#   unrunnable, which is exactly what it did from 43de204 until this commit.
DSA_EQUIV_BASE ?= d8f8f8f
DSA_EQUIV_DEPS := src/mla_attn_q4k.v src/swiglu_expert_q4k.v src/moe_router_q4k.v \
	src/glm_matmul_q4k.v src/rmsnorm_unit.v src/rope_interleave_unit.v \
	src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v \
	src/glm_matmul_pipe.v src/glm_fp_pipe.v
dsa-thread-equiv:
	@mkdir -p $(BUILD_DIR)
	@git show "$(DSA_EQUIV_BASE):src/glm_decoder_block_q4k.v" > $(BUILD_DIR)/dblk_base.v
	@printf '[dsa-thread-equiv] '; $(YOSYS) -q -p "\
	    read_verilog -sv -I src $(DSA_EQUIV_DEPS); \
	    read_verilog -sv -I src $(BUILD_DIR)/dblk_base.v; \
	    prep -top glm_decoder_block_q4k; opt_clean -purge; \
	    rename glm_decoder_block_q4k gold; design -stash gdes; \
	    read_verilog -sv -I src $(DSA_EQUIV_DEPS); \
	    read_verilog -sv -I src src/glm_decoder_block_q4k.v; \
	    prep -top glm_decoder_block_q4k; opt_clean -purge; rename glm_decoder_block_q4k gate; \
	    design -copy-from gdes -as gold gold; \
	    equiv_make gold gate equiv; prep -top equiv; \
	    equiv_simple -undef; equiv_induct -undef; equiv_status -assert" \
	    && echo "PROVEN: glm_decoder_block_q4k(DSA_REAL_IDX=0) == $(DSA_EQUIV_BASE) (pre-threading)" \
	    || { echo "FAILED: dsa-thread-equiv"; exit 1; }


# ---- FULL-CONFIG (753B-shape) elaboration gate (PRODUCT_ROADMAP P1.2) -------
# Elaborates glm_model_q4k with EVERY parameter at the REAL 753B GLM-5.2
# (UD-Q4_K_XL) production shape (configs/full_glm52.vh: MODEL_DIM=6144, L=78,
# VOCAB=154880, N_EXPERT=256, ...) via the documented iverilog -tnull invocation
# in test/full_config_elab_wrap.v -- type/width elaboration only, NO simulation
# (a full-config functional sim is intractable; see docs/FULL_CONFIG_ELAB.md).
# Catches width overflow / out-of-range part-selects / $clog2 edges at true scale.
# The doc's yosys `hierarchy -check` path (C) is NOT wired: yosys 0.66 is
# documented-blocked on derived-module re-elaboration and a fresh attempt ran
# >13 min CPU without completing.  verilator --lint-only (path B) remains the
# authoritative independent cross-check per docs/FULL_CONFIG_ELAB.md.
FULL_ELAB_SRCS := test/full_config_elab_wrap.v src/glm_model_q4k.v \
	src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v \
	src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v \
	src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v

full-elab:
	@printf '[%s] ' "full-elab(753B shape)"; \
	$(IVERILOG) -g2012 -I src -I configs -tnull -pfileline=1 $(FULL_ELAB_SRCS) \
	    && echo "iverilog -tnull elaboration OK (glm_model_q4k @ MODEL_DIM=6144/L=78/VOCAB=154880)" \
	    || { echo "FAILED: full-elab (iverilog -tnull)"; exit 1; }

# ---------------------------------------------------------------------------
# lane-scaling : does adding MAC lanes actually reduce cycles/token?
#
#   WHY.  R3_APPLIANCE_SPEC §3 specs the array from a ROOFLINE (bandwidth /
#   bytes-per-token), a model in which lanes are the lever and the array is the
#   thing standing between you and the memory. Nobody had checked that against
#   real RTL cycles. This does, on the cycle-accurate perf harness.
#
#   MEASURED (RESIDENT=1; the numbers §3 quotes):
#     TN=4  PE_N=2  LM_TN=4   10,902 cyc/tok   1.00x
#     TN=16 PE_N=2  LM_TN=4    8,244           1.32x
#     TN=16 PE_N=16 LM_TN=4    5,860           1.86x
#     TN=16 PE_N=16 LM_TN=16   5,731           1.90x   <- every lane knob 4x
#   cycles = 7,358 + 14,176/TN fits all four TN points to 0.00%.
#   State histogram at max lanes: T_ATTN 66.7% (only -38% for 8x PE_N),
#   T_ACC 1.7%, T_ESCAN 0 (dead at PE_M=1). So lanes cap at ~1.9x here and
#   attention -- not the accumulate, not the weight bus -- is what is left.
#
#   RATIO-FAITHFUL RE-MEASUREMENT (the default config's ratios were wrong).
#   The TB default is MODEL_DIM=16 / INTER_MOE=16 / TOPK=2, which misses 6 of the
#   7 ratios that set the split -- MODEL_DIM/INTER_MOE is 1.0 there vs 3.0 real,
#   H*NOPE/MODEL_DIM 0.5 vs 2.0, H*V_DIM/MODEL_DIM 0.5 vs 2.67 (up to 5.3x off).
#   `make lane-scaling-ratio` re-runs on a config that reproduces the REAL ratios
#   EXACTLY at 1/128 the size (MODEL_DIM=48 INTER_MOE=16 INTER_DENSE=96 TOPK=8
#   H_HEADS=4 NOPE=24 ROPE=8 V_DIM=32 Q_LORA=16 KV_LORA=4):
#     TN=4  PE_N=4  LM=4    53,961 cyc/tok  1.00x   attn 50.9%  expw 27.2%  acc 1.6%
#     TN=16 PE_N=4  LM=4    36,591          1.47x   attn 75.0%  expw 10.6%  acc 2.4%
#     TN=16 PE_N=16 LM=16   23,798          2.27x   attn 62.6%  expw 16.3%  acc 3.6%
#   So at the REAL ratios: every lane knob 4x buys 2.27x, ATTENTION is 62.6% at max
#   lanes (and rises to 75% if you widen only the expert path), and T_ACC is 3.6% --
#   not the bottleneck at either the wrong ratios (1.7%) or the right ones.
#
#   STILL [측정필요]: absolute size (6144 vs 48). Ratios transfer, so the SPLIT does;
#   absolute cycles and tok/s do not. A true full-shape run is impractical even under
#   Verilator -- the generated C++ is 6,977 files / 4.4 GB (22 h at -Os single-thread;
#   ~30 min at -j16 -O0) -- and it would run with a broken weight path anyway (the
#   loader caps PE_N at 16; see R3 §3).
#
#   The histogram probe is pure observation (samples u_block.state; drives
#   nothing), so it cannot perturb the timing it reports.
LANE_SCALE_SRCS := test/glm_q4k_system_perf_tb.v src/glm_q4k_system.v \
	src/glm_model_q4k.v src/ddr5_xbar.v src/weight_loader_q4k.v \
	src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
	src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v \
	src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v \
	src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
	src/glm_fp_pipe.v src/weight_decomp.v src/ecc_secded.v
lane-scaling:
	@mkdir -p $(BUILD_DIR)
	@for cfg in "4 2 4" "16 16 16"; do \
	  set -- $$cfg; \
	  $(IVERILOG) -g2012 -I src -o $(BUILD_DIR)/lane_$$1_$$2_$$3 \
	    -P glm_q4k_system_perf_tb.TN=$$1 -P glm_q4k_system_perf_tb.PE_N=$$2 \
	    -P glm_q4k_system_perf_tb.LM_TN=$$3 -P glm_q4k_system_perf_tb.RESIDENT_CFG=1 \
	    $(LANE_SCALE_SRCS) 2>/dev/null \
	    || { echo "FAILED: lane-scaling compile (TN=$$1 PE_N=$$2 LM_TN=$$3)"; exit 1; }; \
	  printf '[lane-scaling] TN=%-2s PE_N=%-2s LM_TN=%-2s ' "$$1" "$$2" "$$3"; \
	  vvp $(BUILD_DIR)/lane_$$1_$$2_$$3 2>/dev/null \
	    | grep -oE 'cycles/token=[0-9]+' | head -1 \
	    || { echo "FAILED: lane-scaling run"; exit 1; }; \
	done

# TOPK_ATTN is deliberately NOT here: it selects the regime, and each gate must state
# its own (dense = TOPK_ATTN==S_MAX, sparse = TOPK_ATTN<S_MAX). Leaving it in the shared
# block meant lane-scaling-sparse passed -GTOPK_ATTN twice and relied on Verilator's
# last-wins behaviour to pick the right one -- correct by accident, so it is now explicit.
RATIO_CFG := -GMODEL_DIM=48 -GINTER_MOE=16 -GINTER_DENSE=96 -GTOPK=8 -GQ_LORA=16 \
	-GKV_LORA=4 -GH_HEADS=4 -GNOPE=24 -GROPE=8 -GV_DIM=32 -GS_MAX=8 \
	-GN_DENSE=2 -GVOCAB=16 -GRESIDENT_CFG=1 -GN_EXPERT_CFG=16 -GL_CFG=4 -GTIMING_ONLY=1
lane-scaling-ratio:
	@command -v $(VERILATOR) >/dev/null 2>&1 || { echo "lane-scaling-ratio: needs verilator 5.x"; exit 1; }
	@for cfg in "4 4 4" "16 16 16"; do \
	  set -- $$cfg; \
	  $(VERILATOR) --binary --timing -Isrc -Mdir $(BUILD_DIR)/vr_$$1_$$2_$$3 -o vr \
	    --top-module glm_q4k_system_perf_tb --build-jobs 16 -CFLAGS -O0 \
	    -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE -Wno-PINMISSING \
	    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE \
	    $(RATIO_CFG) -GTOPK_ATTN=8 -GTN=$$1 -GPE_N=$$2 -GLM_TN=$$3 \
	    $(LANE_SCALE_SRCS) >/dev/null 2>&1 \
	    || { echo "FAILED: lane-scaling-ratio build (TN=$$1 PE_N=$$2 LM_TN=$$3)"; exit 1; }; \
	  printf '[lane-scaling-ratio] TN=%-2s PE_N=%-2s LM_TN=%-2s ' "$$1" "$$2" "$$3"; \
	  $(BUILD_DIR)/vr_$$1_$$2_$$3/vr 2>/dev/null | grep -oE 'cycles/token=[0-9]+' | head -1 \
	    || { echo "FAILED: lane-scaling-ratio run"; exit 1; }; \
	done

# lane-scaling-sparse : the same sweep in the SPARSE regime, which is the real-use
#   condition (a real context is far longer than TOPK_ATTN, so attention selects
#   rather than keeping every key). The ratio gate above runs dense
#   (TOPK_ATTN=8 = S_MAX=8): nothing to select, every key kept. Measured here with
#   TOPK_ATTN=2 < S_MAX=8 and DSA_REAL_IDX=1:
#     TN=4  PE_N=4    49,363 cyc/tok  1.00x   attn 46.3%  KV/score 21.8%
#     TN=16 PE_N=16   20,586          2.40x   attn 56.7%  KV/score 36.5%
#   vs dense at max lanes: 23,798 / attn 62.6% / KV-score 47.1%.
#   So §3's lane story SURVIVES the real-use regime and only softens: attention still
#   dominates at 56.7%, 36.5% of all cycles is still lane-invariant, and 4x the lanes
#   still buys 2.40x -- against a roofline that promises linear.
lane-scaling-sparse:
	@command -v $(VERILATOR) >/dev/null 2>&1 || { echo "lane-scaling-sparse: needs verilator 5.x"; exit 1; }
	@for cfg in "4 4 4" "16 16 16"; do \
	  set -- $$cfg; \
	  $(VERILATOR) --binary --timing -Isrc -Mdir $(BUILD_DIR)/vsp_$$1_$$2 -o vsp \
	    --top-module glm_q4k_system_perf_tb --build-jobs 16 -CFLAGS -O0 \
	    -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE -Wno-PINMISSING \
	    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE \
	    $(RATIO_CFG) -GTOPK_ATTN=2 -GDSA_REAL_IDX_CFG=1 \
	    -GTN=$$1 -GPE_N=$$2 -GLM_TN=$$3 \
	    $(LANE_SCALE_SRCS) >/dev/null 2>&1 \
	    || { echo "FAILED: lane-scaling-sparse build (TN=$$1 PE_N=$$2)"; exit 1; }; \
	  printf '[lane-scaling-sparse] TN=%-2s PE_N=%-2s ' "$$1" "$$2"; \
	  $(BUILD_DIR)/vsp_$$1_$$2/vsp 2>/dev/null | grep -oE 'cycles/token=[0-9]+' | head -1 \
	    || { echo "FAILED: lane-scaling-sparse run"; exit 1; }; \
	done

# dsa-sparse-correct : glm_q4k_system stays consistent with the standalone reference
#   with DSA_REAL_IDX=1 -- the query-dependent key selection actually ON.
#
#   WHY THIS DID NOT EXIST.  `make mla-sparse` proves DSA_REAL_IDX=1 bit-exact at the
#   LEAF (mla_attn_q4k, PE_M=3, dense+sparse). But nothing ever ran =1 through the whole
#   glm_q4k_system (expert cache + KV pager + xbar + weight loader): the perf TB did not
#   pass the parameter, so =1 was unreachable in any system-level simulation. 43de204
#   threaded it to the system top and said =1 was "reachable"; it was reachable by a
#   parameter override that no harness supplied. This gate closes that.
#
#   MEASURED (TOPK_ATTN=2 < S_MAX=8, i.e. the sparse regime where DSA does anything;
#   at TOPK_ATTN==S_MAX it is a no-op for any value -- mla_attn_q4k.v:165-169):
#     DSA=0  tokens 12,14,14,14   system == standalone ref
#     DSA=1  tokens 12,14, 2, 2   system == standalone ref
#   Both agree with the reference, and the TOKENS DIVERGE from each other at exactly
#   s_len>TOPK_ATTN=2 -- where selection starts. That is the point: =1 keeps consistency
#   while genuinely selecting different, query-dependent keys (S_DSAPF 0 -> 84).
#   Cost measured separately (`make lane-scaling-sparse` config): +0.2%.
#
#   Runs BOTH values: =0 is the shipped default and must stay green; =1 is the one this
#   gate exists for. iverilog, self-check ON (TIMING_ONLY=0) -- this is a CORRECTNESS
#   gate, so it uses the numeric reference simulator, not Verilator.
#
#   IN RELEASE-GATE since 2026-07, taking over the slot dsa-thread-equiv could never
#   finish (see that gate's header ~line 543).  This one IS the thread's machine check:
#   the value has to survive system -> model -> decoder -> mla_attn for the =1 row to
#   diverge from the =0 row, and both rows are held against the reference.
#   RUNTIME: 1045 s wall (17.4 min), EXIT=0, both values green -- measured 2026-07 on
#   M-series/20-core.  Slow because it is four full system sims, but FINITE: the whole
#   point of the swap is that release-gate can now run to completion.
#   CONTEXT SCALING (Verilator, same sparse config, DSA=1, TOPK_ATTN=2, S_MAX 8->64):
#     S_MAX= 8  20,586 cyc/tok  system==ref   S_KEY 12,236 SOFT 8,064 CTX 9,728
#     S_MAX=16  20,602 (+0.1%)  system==ref   identical
#     S_MAX=32  20,634 (+0.2%)  system==ref   identical
#     S_MAX=64  20,698 (+0.5%)  system==ref   identical
#   An 8x window costs +0.5% and the KV/score cycles do not move at all -- TOPK_ATTN caps
#   the key count, so window SIZE does not enter attention cost. Consistency holds at every
#   point. The real cost of a bigger context is KV capacity (87.8 KB/token, R3 §5c), not
#   compute.
#
#   Runtime ~15 min: =1 is several times slower than =0 under iverilog because it
#   actually walks the DSA prefetch path (=0 never pulls a key). Verilator does the
#   same run in seconds but is not the numeric reference (docs/COVERAGE.md).
DSA_CORR_CFG := -Pglm_q4k_system_perf_tb.MODEL_DIM=48 -Pglm_q4k_system_perf_tb.INTER_MOE=16 \
	-Pglm_q4k_system_perf_tb.INTER_DENSE=96 -Pglm_q4k_system_perf_tb.TOPK=8 \
	-Pglm_q4k_system_perf_tb.Q_LORA=16 -Pglm_q4k_system_perf_tb.KV_LORA=4 \
	-Pglm_q4k_system_perf_tb.H_HEADS=4 -Pglm_q4k_system_perf_tb.NOPE=24 \
	-Pglm_q4k_system_perf_tb.ROPE=8 -Pglm_q4k_system_perf_tb.V_DIM=32 \
	-Pglm_q4k_system_perf_tb.S_MAX=8 -Pglm_q4k_system_perf_tb.TOPK_ATTN=2 \
	-Pglm_q4k_system_perf_tb.N_DENSE=2 -Pglm_q4k_system_perf_tb.VOCAB=16 \
	-Pglm_q4k_system_perf_tb.TN=16 -Pglm_q4k_system_perf_tb.PE_N=16 \
	-Pglm_q4k_system_perf_tb.LM_TN=16 -Pglm_q4k_system_perf_tb.RESIDENT_CFG=1 \
	-Pglm_q4k_system_perf_tb.N_EXPERT_CFG=16 -Pglm_q4k_system_perf_tb.L_CFG=4 \
	-Pglm_q4k_system_perf_tb.TIMING_ONLY=0
dsa-sparse-correct:
	@mkdir -p $(BUILD_DIR)
	@for d in 0 1; do \
	  $(IVERILOG) -g2012 -I src -o $(BUILD_DIR)/dsacorr_$$d $(DSA_CORR_CFG) \
	    -Pglm_q4k_system_perf_tb.DSA_REAL_IDX_CFG=$$d $(LANE_SCALE_SRCS) 2>/dev/null \
	    || { echo "FAILED: dsa-sparse-correct compile (DSA_REAL_IDX=$$d)"; exit 1; }; \
	  printf '[dsa-sparse-correct(DSA_REAL_IDX=%s)] ' "$$d"; \
	  vvp $(BUILD_DIR)/dsacorr_$$d 2>/dev/null | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    | sed 's/  (.*//' \
	    || { echo "FAILED: system != standalone ref at DSA_REAL_IDX=$$d"; exit 1; }; \
	done

# ---------------------------------------------------------------------------
# full-elab-lanes : the 753B shape at a SILICON-SCALE lane width, not the toy tile.
#
#   WHY.  R3_APPLIANCE_SPEC §3 specs 13,176 lanes (expert 4 engines x 1,647 +
#   hot 6,588), but what is committed is PE_N = TN = 4 -- in the slice AND in the
#   real GLM-5.2 config (configs/full_glm52.vh). `full-elab` therefore only ever
#   proved the 753B SHAPE at a 4-wide tile; nobody had elaborated the RTL at the
#   lane width the spec actually asks for, so "the array scales" was an assumption.
#
#   WHAT THIS PROVES.  Both lane knobs reach the spec'd width at the true 753B shape:
#     PE_N -> the attention path (mla_attn_q4k's matmuls)
#     TN   -> the expert/dense path (swiglu_expert_q4k's, which passes TN as its
#             matmul's PE_N -- src/swiglu_expert_q4k.v:25,104; the expert pool's
#             lane knob is TN, NOT PE_N)
#   Measured here: elaboration is near-flat in lane width (PE_N 4->1647: 12->31 s;
#   TN 4->1647: 11->53 s), which is what you expect -- widening PE_N/TN adds
#   parallel independent multipliers (glm_matmul_q4k.v:264,270), it does not
#   lengthen the critical path.
#
#   WHAT THIS DOES *NOT* PROVE -- and one of these turned out to matter a lot:
#     * SCOPE: FULL_ELAB_SRCS is the MODEL (glm_model_q4k and below). It does NOT
#       include weight_loader_q4k / glm_q4k_system. So this gate says the MODEL
#       scales; it says NOTHING about the weight path feeding it. It does not.
#       weight_loader_q4k.v:231,237 part-select rd_data[4*PE_N-1:0] and
#       rd_data[16*PE_N-1:0] out of a DATA_W=256 bus (glm_q4k_system.v:296), so the
#       loader caps at PE_N=64 (w_q) / PE_N=16 (w_hp) -- 128x short of the spec'd
#       2,048. iverilog ZERO-FILLS an out-of-range part-select without a peep, which
#       is exactly why this gate went green at 1,647; Verilator flags it (SELRANGE).
#       Registered in R3 §3.
#     * synthesis, area, or timing at that width. Timing at TN=1,647 stays [측정필요]
#       -- the FPGA campaign needed a 4.6x repipelining to reach 46.5 MHz at TN=4.
#     * the 4 parallel expert engines (the RTL still has ONE u_moe scanning the
#       expert axis sequentially -- glm_decoder_block_q4k.v:422, T_ESCAN).
LANE_ELAB_PE_N ?= 1647
LANE_ELAB_TN   ?= 1647
full-elab-lanes:
	@mkdir -p $(BUILD_DIR)/lanecfg
	@sed -e 's|^`define GLM52_PE_N .*|`define GLM52_PE_N $(LANE_ELAB_PE_N)|' \
	     -e 's|^`define GLM52_TN .*|`define GLM52_TN $(LANE_ELAB_TN)|' \
	     configs/full_glm52.vh > $(BUILD_DIR)/lanecfg/full_glm52.vh
	@printf '[%s] ' "full-elab-lanes(PE_N=$(LANE_ELAB_PE_N) TN=$(LANE_ELAB_TN))"; \
	$(IVERILOG) -g2012 -I src -I $(BUILD_DIR)/lanecfg -I configs -tnull -pfileline=1 \
	    $(FULL_ELAB_SRCS) \
	    && echo "elaboration OK at silicon lane width (753B shape)" \
	    || { echo "FAILED: full-elab-lanes"; exit 1; }

# ---- Whole-chip structural gate for the GLM-5.2 (UD-Q4_K_XL) product top ----
# Elaborates the ENTIRE product hierarchy -- the 2-clock chip top
# `glm_q4k_system_cdc` and every Q4_K compute + memory-system + CDC leaf beneath
# it -- and runs `check -assert`, which FAILS (non-zero exit) on any unresolved
# hierarchy, combinational loop, multiple driver, or inferred latch anywhere in
# the Q4_K datapath.  `stat` prints the flattened gate-level cell count.
synth-glm:
	$(YOSYS) -q -p "read_verilog -sv -I src $(GLM_Q4K_CDC_SRCS); \
	                hierarchy -top glm_q4k_system_cdc -check; proc; opt; check -assert; stat"

# ---- FPGA-fit bring-up harness gate (keeps fpga/bringup_harness.v in sync) ----
# The P&R harness (fpga/bringup_harness.v) wraps glm_q4k_system_cdc and buries its
# thousands of wide memory-side ports so a real routed fit is possible. This gate
# fails if the harness's ~110 named port connections drift from the product top's
# ports (a DUT port rename/resize breaks elaboration here) and if the harness's own
# LFSR/CRC feedback ever forms a comb loop / latch. Fast: iverilog elaboration +
# yosys top-only check -assert (DUT treated as a box; its internals are gated by
# synth-glm above).  Not part of `all` -- run in the fpga fit path.
fit-harness:
	@$(IVERILOG) $(IFLAGS) -s bringup_harness -o $(BUILD_DIR)/bringup_harness_elab \
	    fpga/bringup_harness.v $(GLM_Q4K_CDC_SRCS) >/dev/null 2>&1 \
	    && echo "[fit-harness] iverilog elaboration OK" \
	    || { echo "FAILED: fit-harness iverilog elaboration (harness/DUT port drift?)"; exit 1; }
	@$(YOSYS) -q -p "read_verilog -sv -I src fpga/bringup_harness.v $(GLM_Q4K_CDC_SRCS); \
	                 hierarchy -top bringup_harness -check; select bringup_harness; \
	                 proc; check -assert" \
	    && echo "[fit-harness] yosys check -assert: 0 problems (no comb loop / latch)" \
	    || { echo "FAILED: fit-harness yosys structural check"; exit 1; }

# ---- CDC structural sign-off for the 2-clock product top (task C8) ----------
# Asserts every host_clk<->core_clk crossing in glm_q4k_system_cdc flows through a
# recognized synchronizer (async FIFO / 2-FF / reset_sync) and that no raw
# multi-bit register is captured across the boundary.  A targeted structural
# checker (not a commercial CDC tool -- see tools/cdc_check.py header for limits);
# constraints/glm_q4k_system_cdc.sdc carries the matching false-path/async SDC.
cdc:
	@python3 tools/cdc_check.py src/glm_q4k_system_cdc.v \
	    || { echo "FAILED: cdc structural sign-off"; exit 1; }


# ---- Structural code coverage (Verilator --coverage-line/-toggle) ---------
# Verilate + run the SUBSET of behavioral TBs that build+run cleanly under
# Verilator 5.x (--binary), each with --coverage-line --coverage-toggle, then
# report per-module line/toggle/branch coverage of the module's OWN source and
# a merged design-source summary.  Artifacts land in build/cov/ (gitignored);
# the merged database is build/cov/merged.dat.  This is STRUCTURAL coverage of
# the committed slice config -- the bit-fidelity proof remains the byte-
# identical iverilog suite (`make unittests`).  Scope + the TBs that don't
# verilate (and why) are documented in docs/COVERAGE.md.
coverage:
	@command -v $(VERILATOR) >/dev/null 2>&1 || { echo "coverage: needs verilator (5.x)"; exit 1; }
	@mkdir -p $(BUILD_DIR)/cov
	@VERILATOR="$(VERILATOR)" bash tools/cov_run.sh

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.vcd

# ============================================================================
# laguna-config-check : Phase-1 config-consistency gate for the Laguna-S-2.1
#   port (branch laguna-s-2.1).  Elaborates configs/full_laguna_s21.vh and
#   asserts every derived per-layer count / GQA group / dim against the config
#   locked from poolside/Laguna-S-2.1 config.json.  NO datapath -- it guards the
#   config header itself.  Paired with a must-FAIL injection (a corrupted count
#   MUST be caught) so the gate is proven live, same discipline as every gate.
# ============================================================================
.PHONY: laguna-config-check
laguna-config-check:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) -g2012 -I configs -o $(BUILD_DIR)/laguna_config_check test/laguna_config_check.v
	@printf '[%s] ' "laguna_config_check"; $(VVP) $(BUILD_DIR)/laguna_config_check | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: laguna-config-check (config header inconsistent with the locked counts)"; exit 1; }
	@# INJECTION -- force a WRONG full-layer schedule (full iff i%3==0 -> 16 full, not 12):
	@#   the layer-count asserts MUST catch it -> the gate MUST FAIL.
	@$(IVERILOG) -g2012 -I configs -DLAGUNA_INJECT_BADSCHED -o $(BUILD_DIR)/laguna_config_check_inject test/laguna_config_check.v
	@printf '[%s] ' "laguna_config_check_INJECT_badsched"; \
	    if $(VVP) $(BUILD_DIR)/laguna_config_check_inject 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: BADSCHED injection NOT caught (a wrong per-layer schedule still passed -- the count asserts are not load-bearing)"; exit 1; \
	    else echo "injection correctly FAILED (the per-layer schedule rules are pinned against the locked 12-full/36-sliding counts)"; fi

# ============================================================================
# laguna-moe : Phase-2 MoE/FFN gate for the Laguna-S-2.1 port.  Proves the MoE
#   path carries to Laguna's exact config -- the leaf modules need NO change,
#   only re-parameterization (hidden_act=silu == GLM SwiGLU; sigmoid router +
#   norm_topk_prob + bias=0 no-op == GLM router; the ONLY delta is the scale-
#   placement, handled at assembly by SCALE=1.0 + combine x2.5, see
#   docs/LAGUNA_S21.md).  Three checks:
#     (1) the Laguna MoE numeric-order reference self-test (block order:
#         sigmoid router -> top-10 -> norm -> routed x2.5 + unscaled shared);
#     (2) moe_router_q4k at Laguna's 256 experts / TOP-10 (renorm invariant --
#         the Laguna-specific width: top-10, not GLM's top-8);
#     (3) swiglu_expert_q4k at Laguna's expert FFN (INTER_MOE=1024), Q4_K
#         bit-exact vs the ggml reference.
# ============================================================================
.PHONY: laguna-moe
laguna-moe:
	@mkdir -p $(BUILD_DIR)
	@# (1) Laguna MoE numeric-order reference (block-level order + scale placement)
	@printf '[%s] ' "laguna_moe_ref"; python3 tools/laguna_moe_ref.py --selftest | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: laguna_moe_ref self-test"; exit 1; }
	@# (2) moe_router_q4k at Laguna 256 / TOP-10 -- sigmoid -> top10 -> renorm invariant
	@$(IVERILOG) $(IFLAGS) -DTB_N_EXPERT=256 -DTB_TOPK=10 -DTB_HIDDEN=128 -DTB_NTEST=4 -DTB_TIMEOUT_NS=600000000 \
	    -o $(BUILD_DIR)/laguna_moe_router_sim test/moe_router_q4k_tb.v \
	    src/moe_router_q4k.v src/glm_matmul_q4k.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_q4k(256/top-10 Laguna)"; $(VVP) $(BUILD_DIR)/laguna_moe_router_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: laguna moe_router_q4k (256/top-10)"; exit 1; }
	@# (3) swiglu_expert_q4k at Laguna expert FFN INTER_MOE=1024 (Q4_K bit-exact vs ggml ref)
	@python3 tools/swiglu_q4k_gen.py 4 64 1024 4 $(BUILD_DIR)/laguna_swiglu_vec.txt >/dev/null
	@$(IVERILOG) $(IFLAGS) -DTB_HIDDEN=64 -DTB_INTER=1024 -DTB_KMAX=1024 -DTB_VEC='"$(BUILD_DIR)/laguna_swiglu_vec.txt"' \
	    -DTB_DONE_GUARD=400000 -DTB_TIMEOUT_NS=600000000 \
	    -o $(BUILD_DIR)/laguna_swiglu_sim test/swiglu_expert_q4k_tb.v \
	    src/swiglu_expert_q4k.v src/glm_matmul_q4k.v src/glm_act.v
	@printf '[%s] ' "swiglu_expert_q4k(INTER=1024 Laguna)"; $(VVP) $(BUILD_DIR)/laguna_swiglu_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: laguna swiglu_expert_q4k (INTER=1024)"; exit 1; }
	@echo "laguna-moe: Laguna MoE path carries -- ref order OK; router 256/top-10 renorm-invariant; SwiGLU expert INTER_MOE=1024 Q4_K bit-exact. Scale placement (SCALE=1.0 + combine x2.5) is the assembly-phase combine, NOT a leaf change."

# ============================================================================
# laguna-attn / laguna-model : Phase 3-6 REFERENCE gates for the attention
#   machine and the full forward.  These are the executable Laguna spec (numpy),
#   self-tested -- the golden the Laguna RTL checks against.  Attention covers
#   Phases 3-5 (GQA + q/k RMSNorm + rotate_half dual YaRN/plain partial RoPE +
#   sliding-window/causal mask + softplus per-head gating); model covers Phase 6
#   (embed -> 48x(attn + MoE/dense) -> norm -> LM head -> argmax, dense@0 + the
#   full/sliding schedule).  See docs/LAGUNA_S21.md SS6 for the verification-level
#   ledger (what is bit-exact-RTL vs reference-verified).
# ============================================================================
.PHONY: laguna-attn laguna-model laguna-datapath-elab laguna
laguna-attn:
	@printf '[%s] ' "laguna_attn_ref"; python3 tools/laguna_attn_ref.py --selftest | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: laguna_attn_ref self-test"; exit 1; }

laguna-model:
	@printf '[%s] ' "laguna_model_ref"; python3 tools/laguna_model_ref.py --selftest | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: laguna_model_ref self-test"; exit 1; }

# laguna-datapath-elab : ELABORATION study -- the leaf blocks a GQA attention
#   composes from (Q4_K GEMM for Q/K/V/O, RMSNorm for q/k-norm at head_dim=128
#   eps=1e-6, softmax for the attention window) are main's verified leaves; this
#   type/width-checks them at Laguna's attention shape (iverilog -tnull, no sim).
laguna-datapath-elab:
	@mkdir -p $(BUILD_DIR)
	@printf '[%s] ' "laguna_datapath_elab"; \
	    $(IVERILOG) -g2012 -I src -I configs -tnull test/laguna_datapath_elab_wrap.v \
	    src/glm_matmul_q4k.v src/rmsnorm_unit.v src/glm_softmax.v src/glm_fp_pipe.v \
	    && echo "ELABORATED (GQA datapath leaves type/width-check at Laguna attention shape: head_dim=128, eps=1e-6)" \
	    || { echo "FAILED: laguna-datapath-elab (a leaf does not parameterize to Laguna attention shape)"; exit 1; }

# laguna : the aggregate Laguna-port gate (branch laguna-s-2.1).  Runs every
#   Laguna check: config-consistency, the MoE path, the attention + full-forward
#   references, and the datapath elaboration.  This is the branch's release gate.
laguna: laguna-config-check laguna-attn laguna-model laguna-datapath-elab laguna-moe
	@echo "laguna: ALL Laguna-port gates passed (config + attn/model refs + datapath elab + MoE). See docs/LAGUNA_S21.md SS6 for the verification-level ledger."
# mshr : NON-BLOCKING (MSHR) expert-cache refill -- docs/ULTRA_PERF_MODULES.md
#   rank 1, the top fetch-path lever.  The committed cache (expert_cache_ctrl /
#   expert_cache_pf) services a miss with a BLOCKING S_FETCH, so a token's top-k
#   misses cost ~k*FLASH_LAT and demand bandwidth is capped at 1/FLASH_LAT.
#   src/expert_cache_mshr.v overlaps them with an M-entry MSHR file.
#
#   The gate MEASURES the lever (it does not assert it): K cold misses issued
#   back-to-back are timed in cycles and compared against the blocking baseline
#   K*FLASH_LAT.  It also proves the three hazards MSHR introduces -- victim
#   collision, duplicate fetch, in-flight-is-not-a-hit -- with a must-FAIL
#   injection for each of the first two, and exercises OUT-OF-ORDER completion.
#
#   SCOPE: this module is NOT wired into the product top (the verified datapath
#   is untouched).  Integration additionally needs the two other single-
#   outstanding seams widened -- the `awaiting` request issuer and the single
#   untagged flash arbiter in glm_q4k_system.v -- see the module header.
# ============================================================================
.PHONY: mshr
mshr:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_mshr_sim \
	    test/expert_cache_mshr_tb.v src/expert_cache_mshr.v
	@printf '[%s] ' "expert_cache_mshr"; $(VVP) $(BUILD_DIR)/expert_cache_mshr_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mshr"; exit 1; }
	@# INJECTION 1 -- drop the victim-exclusion so two concurrent misses may pick
	@#   the SAME slot: the directory/response checks MUST catch it.
	@$(IVERILOG) $(IFLAGS) -DINJECT_NORESV -o $(BUILD_DIR)/expert_cache_mshr_noresv \
	    test/expert_cache_mshr_tb.v src/expert_cache_mshr.v
	@printf '[%s] ' "expert_cache_mshr_INJECT_noresv"; \
	    if $(VVP) $(BUILD_DIR)/expert_cache_mshr_noresv 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: NORESV injection NOT caught (concurrent misses may share a victim slot and nothing noticed)"; exit 1; \
	    else echo "injection correctly FAILED (victim reservation across in-flight misses is load-bearing)"; fi
	@# INJECTION 2 -- refetch an id that is already in flight instead of merging:
	@#   the fetch-count check MUST catch the duplicate Flash traffic.
	@$(IVERILOG) $(IFLAGS) -DINJECT_NOMERGE -o $(BUILD_DIR)/expert_cache_mshr_nomerge \
	    test/expert_cache_mshr_tb.v src/expert_cache_mshr.v
	@printf '[%s] ' "expert_cache_mshr_INJECT_nomerge"; \
	    if $(VVP) $(BUILD_DIR)/expert_cache_mshr_nomerge 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: NOMERGE injection NOT caught (a duplicate fetch for an in-flight id went unnoticed)"; exit 1; \
	    else echo "injection correctly FAILED (MSHR merging of an in-flight id is load-bearing)"; fi
# ============================================================================
# mla-sparse : mla_attn_q4k SPARSE / PER-ROW batching oracle (standalone gate)
# ----------------------------------------------------------------------------
#   DUT-vs-DUT EXACT (===) oracle for the PE_M-batched Q4_K MLA attention:
#   one batched DUT (PE_M=3, PER_ROW_POS/PER_ROW_SLEN, per-row q-dependent DSA
#   DSA_REAL_IDX=1) vs PE_M=1 re-runs per row on that row's own (x,pos,s_len)
#   -- BIT-EXACT rows across dense + sparse (S_MAX > TOPK) cases, per-row DSA
#   divergence proven live, fetch-sharing asserted (exact / union bounds), a
#   dense-vs-sparse full-window cross-check (TOPK=4 vs TOPK=S_MAX machine,
#   bit-identical when selection covers the window), and PER_ROW_SEQ per-row
#   KV windows (TOPK_SEQ=2; kc==sum-of-rows, weights shared).  The fp8 track
#   carried the sibling gate (on branch fp8, not on main); this is the
#   Q4_K-product sibling closing that audit gap.
# ============================================================================
.PHONY: mla-sparse
mla-sparse:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_q4k_sparse_perrow_sim test/mla_attn_q4k_sparse_perrow_tb.v src/mla_attn_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_q4k_sparse_perrow"; $(VVP) $(BUILD_DIR)/mla_attn_q4k_sparse_perrow_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_q4k_sparse_perrow"; exit 1; }
#============================================================================
# mla-intra : mla_attn_q4k INTRA-BATCH CAUSAL leaf oracle (5b-leaf; standalone).
# ----------------------------------------------------------------------------
#   The correctness gate for the new INTRA_CAUSAL feature.  One batched PE_M=B
#   pass (INTRA_CAUSAL=1, PER_ROW_POS, real DSA) over B tokens at consecutive
#   causal positions p..p+B-1 -- row r attends 0..p-1 (shared cache) PLUS the
#   current-token keys of rows 0..r-1 (this pass) -- must be BIT-EXACT (full
#   MODEL_DIM out AND the widened kv_lat_row_all egress) to a SERIAL single-row
#   (PE_M=1, INTRA off) chain of the same tokens, where each serial decode's OWN
#   kv_lat_row is appended to the cache for the next (the die-internal KV write-
#   back).  DENSE / MIXED / SPARSE regimes, B=2,3,4.  Two INJECTIONS prove the
#   oracle bites: dropping the per-row causal mask (row r attends its own current
#   token) and skipping the intra key's RMSNorm before W_uk must BOTH make the
#   oracle FAIL.  NOT wired into release-gate (a standalone leaf gate).
# ============================================================================
INTRA_SRCS := test/mla_attn_q4k_intra_causal_tb.v src/mla_attn_q4k.v src/glm_matmul_q4k.v \
	src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v \
	src/glm_softmax.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
.PHONY: mla-intra
mla-intra:
	@mkdir -p $(BUILD_DIR)
	@# (1) the LEAF ORACLE : batched intra-causal pass === serial chain, bit-exact.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_q4k_intra_sim $(INTRA_SRCS)
	@printf '[%s] ' "mla_attn_q4k_intra_causal"; $(VVP) $(BUILD_DIR)/mla_attn_q4k_intra_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_q4k_intra_causal (batched != serial)"; exit 1; }
	@# (2) INJECTION -- drop the per-row causal mask : the oracle MUST FAIL.
	@$(IVERILOG) $(IFLAGS) -DINTRA_INJECT_NOMASK -o $(BUILD_DIR)/mla_attn_q4k_intra_nomask $(INTRA_SRCS)
	@printf '[%s] ' "mla_attn_q4k_intra_INJECT_nomask"; \
	    if $(VVP) $(BUILD_DIR)/mla_attn_q4k_intra_nomask 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: NOMASK injection NOT caught (oracle passed a non-causal batch)"; exit 1; \
	    else echo "injection correctly FAILED (causal mask is load-bearing)"; fi
	@# (3) INJECTION -- skip RMSNorm on the intra key before W_uk : MUST FAIL.
	@$(IVERILOG) $(IFLAGS) -DINTRA_INJECT_SKIPNORM -o $(BUILD_DIR)/mla_attn_q4k_intra_skipnorm $(INTRA_SRCS)
	@printf '[%s] ' "mla_attn_q4k_intra_INJECT_skipnorm"; \
	    if $(VVP) $(BUILD_DIR)/mla_attn_q4k_intra_skipnorm 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: SKIPNORM injection NOT caught (oracle passed an un-normed intra key)"; exit 1; \
	    else echo "injection correctly FAILED (RMSNorm+W_uk on the intra key is load-bearing)"; fi
#============================================================================
# intra-batch-verify : SYSTEM-LEVEL position-accurate BATCHED-VERIFY oracle (5b-sys).
# ----------------------------------------------------------------------------
#   ONE PE_M=K+1 batched pass through glm_q4k_system (SELF_KV=1, INTRA_CAUSAL=1,
#   PER_ROW_POS=1) over K+1 tokens at consecutive positions 0..K -- row r attends
#   the CURRENT-token keys of rows 0..r-1 computed IN THAT SAME PASS, through the
#   whole multi-layer model (L=3: 2 dense + 1 MoE) -- must be BIT-EXACT (full per-row
#   logit vector + per-row argmax) to a SERIAL single-row (PE_M=1, SELF_KV=1)
#   glm_q4k_system decode chain of the same tokens, where SELF_KV=1 makes the pager
#   APPEND each decode's committed latent and GATHER it back for the next (the
#   die-internal KV write-back).  K=1,2,3 -> separate PE_M=2/3/4 batched instances
#   (literal "one PE_M=K+1 pass" per K).  This is the composition of the leaf oracle
#   (mla-intra) across the layers + the pager per-(layer,pos) round-trip (self-kv-l6):
#   batched intra-causal MLA brought up into the SYSTEM.  (The identical harness at
#   L=6/N_DENSE=3 also prints ALL 9 TESTS PASSED -- layer-count independent.)
#   INJECTION: driving the batched pos_vec with ONE shared position (every row ropes
#   at 0) breaks intra-batch causal indexing -> rows attending >=2 keys (K>=2)
#   diverge and the batch==serial oracle MUST FAIL.  (K=1 row 1 attends ONE key, so
#   softmax=1 -> position-insensitive; the injection first bites at K=2.)
#   Standalone gate; NOT in release-gate (multi-instance system sim -- minutes-long).
# ============================================================================
INTRA_BATCH_SRCS := test/glm_q4k_intra_batch_verify_tb.v $(GLM_Q4K_SYS_SRCS)
.PHONY: intra-batch-verify
intra-batch-verify:
	@mkdir -p $(BUILD_DIR)
	@# (1) the SYSTEM ORACLE : batched intra-causal pass === serial decode chain, bit-exact.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_intra_batch_sim $(INTRA_BATCH_SRCS)
	@printf '[%s] ' "glm_q4k_intra_batch_verify"; $(VVP) $(BUILD_DIR)/glm_q4k_intra_batch_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_q4k_intra_batch_verify (batched system != serial decode chain)"; exit 1; }
	@# (2) INJECTION -- one shared position for every batched row : the oracle MUST FAIL.
	@$(IVERILOG) $(IFLAGS) -DINJECT_SHARED_POS -o $(BUILD_DIR)/glm_q4k_intra_batch_sharedpos $(INTRA_BATCH_SRCS)
	@printf '[%s] ' "glm_q4k_intra_batch_INJECT_sharedpos"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_intra_batch_sharedpos 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: SHARED_POS injection NOT caught (batched verify passed with wrong per-row positions)"; exit 1; \
	    else echo "injection correctly FAILED (per-row position wiring is load-bearing)"; fi

# ============================================================================
# spec-greedy : SPEC==GREEDY for the CLOSED speculating top (5c).
# ----------------------------------------------------------------------------
#   glm_q4k_spec_system closes the batched speculative-decode loop around
#   glm_q4k_system (PE_M=K+1, SELF_KV=1, INTRA_CAUSAL=1, PER_ROW_POS=1) + the
#   committed spec_decode_seq (accept/reject) + the per-pass KV WRITE-BACK of the
#   accepted prefix.  Driven pass-by-pass over K=1,2,3 x {ALL-ACCEPT, ALL-REJECT,
#   MIXED} draft schedules, its committed token STREAM must EQUAL a plain PE_M=1
#   greedy decode of the same prompt (only the tokens-per-pass count changes), AND
#   -- the load-bearing binding -- every committed batch row's FULL pre-argmax
#   LOGIT vector must be BIT-EXACT to the greedy reference's logit at that position.
#   The full-logit bind is what proves the KV write-back placed the accepted prefix
#   at the RIGHT (layer,position): a pass at committed length t reads the shared
#   prefix 0..t-1 the prior write-backs left in the pager, so a mis-placed or
#   phantom-after-a-reject KV row would change the logits and FAIL.  ALL-REJECT
#   (K tokens/pass -> 1) and MIXED (partial accept p<K) exercise t>0 and the
#   partial-accept write-back hazard directly.
#   INJECTION (-DINJECT_RAW_DRAFT): the loop commits the RAW DRAFTS (fed as
#   spec_decode_seq's truth_vec) instead of the model's argmaxes; the ALL-REJECT
#   schedule's deliberately-wrong drafts then diverge from greedy -> this gate MUST
#   FAIL.  This is exactly what spec_decode_seq must prevent (never commit a raw
#   draft).  Standalone gate (multi-instance system sim, minutes-long); NOT in
#   release-gate.
# ============================================================================
SPEC_GREEDY_SRCS := test/glm_q4k_spec_greedy_tb.v src/glm_q4k_spec_system.v \
	src/spec_decode_seq.v $(GLM_Q4K_SYS_SRCS)
.PHONY: spec-greedy
spec-greedy:
	@mkdir -p $(BUILD_DIR)
	@# (1) spec==greedy : committed stream + full-logit bind == PE_M=1 greedy, K=1,2,3 x {ACCEPT,REJECT,MIXED}.
	@$(IVERILOG) $(IFLAGS) -DSPEC_FULL -o $(BUILD_DIR)/glm_q4k_spec_greedy_sim $(SPEC_GREEDY_SRCS)
	@# EXACT count (31) -- pins BOTH that -DSPEC_FULL ran (default subset prints a different
	@#   number) AND that no config was silently dropped.  A wildcard [0-9]+ would pass either.
	@printf '[%s] ' "glm_q4k_spec_system(spec==greedy,K=1-3)"; $(VVP) $(BUILD_DIR)/glm_q4k_spec_greedy_sim | grep -E 'ALL 31 TESTS PASSED' \
	    || { echo "FAILED: spec-greedy (committed stream != greedy, OR test count != 31 -- a config was dropped)"; exit 1; }
	@# (2) INJECTION -- commit RAW DRAFTS instead of the model's argmaxes : the gate MUST FAIL.
	@#     (K=2 subset -- the ALL-REJECT wrong drafts diverge; enough to bind the mechanism.)
	@$(IVERILOG) $(IFLAGS) -DINJECT_RAW_DRAFT -o $(BUILD_DIR)/glm_q4k_spec_greedy_inject $(SPEC_GREEDY_SRCS)
	@printf '[%s] ' "glm_q4k_spec_greedy_INJECT_rawdraft"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_spec_greedy_inject 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: RAW_DRAFT injection NOT caught (loop committed raw drafts and still matched greedy)"; exit 1; \
	    else echo "injection correctly FAILED (spec_decode_seq committing only the model argmaxes is load-bearing)"; fi
	@# (3) INJECTION -- ALSO write back the KV of the REJECTED row p+1 (phantom KV after a
	@#     partial accept) : greedy never wrote that key, so the next pass's attention set
	@#     differs and the committed stream / full-logit binding diverges -> MUST FAIL.
	@#     (-DSPEC_FULL so the K=1,2,3 MIXED partial-accept schedules exercise it.)
	@$(IVERILOG) $(IFLAGS) -DSPEC_FULL -DINJECT_PHANTOM_KV -o $(BUILD_DIR)/glm_q4k_spec_greedy_phantom $(SPEC_GREEDY_SRCS)
	@printf '[%s] ' "glm_q4k_spec_greedy_INJECT_phantomkv"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_spec_greedy_phantom 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: PHANTOM_KV injection NOT caught (a rejected row's KV was appended and the stream still matched greedy)"; exit 1; \
	    else echo "injection correctly FAILED (only ACCEPTED rows may write back KV -- a phantom row after a reject diverges from greedy)"; fi

# ============================================================================
# loopback : C8 LOOPBACK==1 committed stream == LOOPBACK==0 (== reference).
# ----------------------------------------------------------------------------
#   glm_q4k_system has a LOOPBACK param (default 0) that PHYSICALLY routes the
#   die's attention-weight Q4_K CODE lanes OUT of the die as a banked ddr5_xbar
#   read (TAG_LBAW, addr encodes {layer,sel,grp,k} + an 8'hA5 marker) and back IN
#   through the fabric, clock-gating the die until each beat returns -- instead of
#   the same-cycle stub.  test/glm_q4k_loopback_tb.v reuses the perf TB's full
#   weight/KV/expert responder harness + the standalone glm_model_q4k reference,
#   and EXTENDS the ddr5_xbar memory model so a marked (8'hA5) read returns the
#   packed f_awq(layer,sel,grp*PE_N+t,k) lanes in the response's low PE_N*4 bits
#   -- the exact bytes the stub aw_q serves at LOOPBACK=0.  A 4-token decode must
#   commit next_tok BIT-EXACT to the reference at every step (clock-gating a
#   synchronous die is transparent), proving the fed-back bytes round-trip
#   faithfully.  N=5 (4 token bindings + a loopback-exercised soundness gate that
#   $fatal's if lb_req/lb_resp never fired -- no vacuous pass).
#   INJECTION (-DLBINJECT): corrupt the fed-back lane 0 (flip bit 0) on EVERY
#   loopback beat.  Because the die actually CONSUMES those lanes the committed
#   stream then DIVERGES from the reference -> this gate MUST FAIL (no pass
#   banner).  Standalone gate (system sim, ~2 min/build); NOT in release-gate.
# ============================================================================
LOOPBACK_SRCS := test/glm_q4k_loopback_tb.v $(GLM_Q4K_SYS_SRCS)
.PHONY: loopback
loopback:
	@mkdir -p $(BUILD_DIR)
	@# (1) CLEAN : LOOPBACK=1 committed stream == standalone glm_model_q4k reference.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_loopback_sim $(LOOPBACK_SRCS)
	@printf '[%s] ' "glm_q4k_system(LOOPBACK=1==ref)"; $(VVP) $(BUILD_DIR)/glm_q4k_loopback_sim | grep -E 'ALL 5 TESTS PASSED' \
	    || { echo "FAILED: loopback (LOOPBACK=1 committed stream != reference, OR test count != 5)"; exit 1; }
	@# (2) INJECTION -- corrupt the fed-back attention-weight lanes : the gate MUST FAIL.
	@$(IVERILOG) $(IFLAGS) -DLBINJECT -o $(BUILD_DIR)/glm_q4k_loopback_inject $(LOOPBACK_SRCS)
	@printf '[%s] ' "glm_q4k_loopback_INJECT_awlanes"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_loopback_inject 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: LBINJECT NOT caught (corrupted loopback bytes still matched reference -- die ignores the fed-back lanes)"; exit 1; \
	    else echo "injection correctly FAILED (the die CONSUMES the fed-back ddr5_xbar bytes -- loopback is load-bearing)"; fi

# ============================================================================
# loopback-fw : C8 LOOPBACK_FW==1 committed stream == LOOPBACK_FW==0 (== reference).
# ----------------------------------------------------------------------------
#   The EXACT mirror of `loopback` for the bandwidth-dominant FFN routed-expert
#   (fw) family.  glm_q4k_system has a LOOPBACK_FW param (default 0) that
#   PHYSICALLY routes the die's FFN Q4_K CODE lanes -- BOTH buses fw_q (GATE/DOWN)
#   and fw_q_up (UP) -- OUT of the die as ONE banked ddr5_xbar read (TAG_LBFW,
#   addr encodes {layer,eidx,sel,shared,grp,k} + an 8'hB6 marker, distinct from
#   aw's 8'hA5) and back IN through the fabric, clock-gating the die until the beat
#   returns -- instead of the same-cycle stub.  test/glm_q4k_loopback_fw_tb.v
#   reuses the perf/loopback TB's full weight/KV/expert responder harness + the
#   standalone glm_model_q4k reference, and EXTENDS the ddr5_xbar memory model so a
#   marked (8'hB6) read returns the packed f_fwq(...) lanes: low 4*TN bits =
#   f_fwq(layer,sel,shared,eidx,grp*TN+t,k) (fw_q), next 4*TN bits =
#   f_fwq(layer,3,shared,eidx,grp*TN+t,k) (fw_q_up) -- the exact bytes the stub
#   fw_q/fw_q_up serve at LOOPBACK_FW=0 (8*TN=32 bits pack into one 256b beat).  A
#   4-token decode must commit next_tok BIT-EXACT to the reference at every step.
#   N=5 (4 token bindings + an fw-loopback-exercised soundness gate that $fatal's
#   if lbfw_req/lbfw_resp never fired -- no vacuous pass).
#   INJECTION (-DLBFWINJECT): corrupt the fed-back fw lane 0 (flip bit 0) on EVERY
#   loopback beat.  Because the die actually CONSUMES those lanes the committed
#   stream then DIVERGES from the reference -> this gate MUST FAIL (no pass
#   banner).  Standalone gate (system sim, ~2 min/build); opt-in, NOT in release-gate.
# ============================================================================
LOOPBACK_FW_SRCS := test/glm_q4k_loopback_fw_tb.v $(GLM_Q4K_SYS_SRCS)
.PHONY: loopback-fw
loopback-fw:
	@mkdir -p $(BUILD_DIR)
	@# (1) CLEAN : LOOPBACK_FW=1 committed stream == standalone glm_model_q4k reference.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_loopback_fw_sim $(LOOPBACK_FW_SRCS)
	@printf '[%s] ' "glm_q4k_system(LOOPBACK_FW=1==ref)"; $(VVP) $(BUILD_DIR)/glm_q4k_loopback_fw_sim | grep -E 'ALL 5 TESTS PASSED' \
	    || { echo "FAILED: loopback-fw (LOOPBACK_FW=1 committed stream != reference, OR test count != 5)"; exit 1; }
	@# (2) INJECTION -- corrupt the fed-back FFN-expert lanes : the gate MUST FAIL.
	@$(IVERILOG) $(IFLAGS) -DLBFWINJECT -o $(BUILD_DIR)/glm_q4k_loopback_fw_inject $(LOOPBACK_FW_SRCS)
	@printf '[%s] ' "glm_q4k_loopback_fw_INJECT_fwlanes"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_loopback_fw_inject 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: LBFWINJECT NOT caught (corrupted loopback bytes still matched reference -- die ignores the fed-back fw lanes)"; exit 1; \
	    else echo "injection correctly FAILED (the die CONSUMES the fed-back ddr5_xbar fw bytes -- fw loopback is load-bearing)"; fi

# ============================================================================
# loopback-rest : C8 LOOPBACK_REST==1 committed stream == LOOPBACK_REST==0 (== ref).
# ----------------------------------------------------------------------------
#   The mirror of `loopback` / `loopback-fw` for the THREE remaining die weight-input
#   families that were stub-only: rw (MoE-router Q4_K CODES), lw (LM-head bf16 columns)
#   and gn (a single bf16 gain/norm).  glm_q4k_system has a LOOPBACK_REST param
#   (default 0) that PHYSICALLY routes die_rw_q / die_lw_col / die_gn_val OUT of the die
#   as banked ddr5_xbar reads (TAG_LBRW/LBLW/LBGN; addr markers 8'hC7/8'hD8/8'hE9, all
#   distinct from aw's 8'hA5 and fw's 8'hB6; each addr encodes that family's exact pull
#   key -- rw {db_layer,rw_k}, lw {lw_vtile,lw_k}, gn {db_layer,gn_which,gn_idx}) and
#   back IN through the fabric, clock-gating the die until each beat returns -- instead
#   of the same-cycle stub.  test/glm_q4k_loopback_rest_tb.v reuses the perf/loopback TB
#   harness + the standalone glm_model_q4k reference, and EXTENDS the ddr5_xbar memory
#   model so a marked read returns the packed bytes the stub serves at REST=0 (low
#   4*N_EXPERT = f_rwq codes; low LM_TN*16 = lw bf16 columns; low 16 = the gn bf16).  A
#   4-token decode must commit next_tok BIT-EXACT to the reference at every step.  N=10:
#   4 committed-token bindings + a PER-FAMILY loopback-exercised soundness gate for EACH
#   of rw/lw/gn (each $fatal's if that family never round-tripped) + a PER-FAMILY DIRECT
#   PER-BEAT BYTE binding for EACH of rw/lw/gn.  The direct binding is the key check: rw
#   and gn were MEASURED output-insensitive, so a bit-exact token does NOT prove their
#   byte consumption -- so the TB reaches into the DUT by hierarchical reference and
#   asserts the die's PRESENTED input (dut.die_rw_q / die_lw_col / die_gn_val) equals the
#   expected value for the live key on EVERY beat the die pulls that family (no vacuous
#   pass: each direct binding must also sample at least once).
#   INJECTION (-DLBRESTINJECT): flip the mantissa LSB of the fed-back gn value on EVERY
#   gn beat.  gn is output-INSENSITIVE, so the committed-token binding stays green -- but
#   the DIRECT gn byte binding sees die_gn_val != the expected stub value on every gn beat
#   and FAILS the gate (no pass banner).  Corrupting the output-insensitive family isolates
#   the direct binding as the check that catches it.  Standalone gate (system sim,
#   ~2 min/build); opt-in, NOT in release-gate.
# ============================================================================
LOOPBACK_REST_SRCS := test/glm_q4k_loopback_rest_tb.v $(GLM_Q4K_SYS_SRCS)
.PHONY: loopback-rest
loopback-rest:
	@mkdir -p $(BUILD_DIR)
	@# (1) CLEAN : LOOPBACK_REST=1 committed stream == standalone glm_model_q4k reference.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_q4k_loopback_rest_sim $(LOOPBACK_REST_SRCS)
	@printf '[%s] ' "glm_q4k_system(LOOPBACK_REST=1==ref)"; $(VVP) $(BUILD_DIR)/glm_q4k_loopback_rest_sim | grep -E 'ALL 10 TESTS PASSED' \
	    || { echo "FAILED: loopback-rest (LOOPBACK_REST=1 committed stream != reference OR direct byte binding mismatched, OR test count != 10)"; exit 1; }
	@# (2) INJECTION -- flip the fed-back gn value's mantissa LSB : the DIRECT gn byte
	@#     binding MUST catch it and the gate MUST FAIL (gn is output-insensitive, so this
	@#     is precisely what the committed-token binding alone would miss).
	@$(IVERILOG) $(IFLAGS) -DLBRESTINJECT -o $(BUILD_DIR)/glm_q4k_loopback_rest_inject $(LOOPBACK_REST_SRCS)
	@printf '[%s] ' "glm_q4k_loopback_rest_INJECT_gn"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_loopback_rest_inject 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: LBRESTINJECT NOT caught (corrupted fed-back gn bytes still passed -- the direct byte binding is not load-bearing)"; exit 1; \
	    else echo "injection correctly FAILED (the DIRECT byte binding caught the corrupted fed-back gn value -- the fabric round-trip is proven to deliver the die's bytes)"; fi
	@# (3) INJECTION -- corrupt the fed-back rw ROUTER-CODE lane 0.  rw is OUTPUT-INSENSITIVE
	@#     (a one-code router step is absorbed by top-k selection), so the committed-token
	@#     binding alone would MISS this: the DIRECT per-beat die_rw_q byte binding must
	@#     catch it -> MUST FAIL.  The load-bearing demo that output-insensitive families
	@#     need direct bindings.
	@$(IVERILOG) $(IFLAGS) -DLBRESTINJECT_RW -o $(BUILD_DIR)/glm_q4k_loopback_rest_inject_rw $(LOOPBACK_REST_SRCS)
	@printf '[%s] ' "glm_q4k_loopback_rest_INJECT_rw"; \
	    if $(VVP) $(BUILD_DIR)/glm_q4k_loopback_rest_inject_rw 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: LBRESTINJECT_RW NOT caught (corrupted router codes passed -- the rw direct binding is not load-bearing)"; exit 1; \
	    else echo "injection correctly FAILED (the DIRECT rw byte binding caught the corrupted router codes that top-k absorption hides from the token stream)"; fi

#============================================================================
# ---- PERF (Q4_K): cycle-accurate throughput harness (audit #15) -----------
#   The Q4_K port of the fp8 track's perf/cycle-emulation harness
#   (docs/CYCLE_EMULATION.md [PENDING] item): test/glm_q4k_system_perf_tb.v
#   decodes a 4-token sequence on glm_q4k_system with the die clock-gated on
#   expert-cache demand misses (EXPERT_STALL=1), binding every committed token
#   against a standalone glm_model_q4k, and emits machine-readable
#   'PERF q4k ... cycles/token=... stall/token=... compute/token=...' lines.
#   tools/perf_sweep.sh compiles/runs it per config (FLASH_LAT x cache-hit vs
#   thrash x RESIDENT=0/1 x EXPERT_STALL on/off) and tabulates the measured
#   memory-stall fraction.  MEASUREMENT harness, not part of `all`
#   (minutes-long: ~8 iverilog runs).  SWEEP=full adds DDR_NCH/CACHE_SLOTS.
.PHONY: perf-q4k
perf-q4k:
	@mkdir -p $(BUILD_DIR)
	@bash tools/perf_sweep.sh > $(BUILD_DIR)/perf_q4k_sweep.log 2>&1 \
	    || { cat $(BUILD_DIR)/perf_q4k_sweep.log; echo "FAILED: perf-q4k (sweep run failed)"; exit 1; }

# ============================================================================
# ---- SCALE-FUNCTIONAL gates (docs/SCALE_FUNCTIONAL.md items 2 + 3) ---------
# ----  NEW delimited section: `make scale-ops` + `make batched-q4k`  --------
# ============================================================================
.PHONY: scale-ops batched-q4k

# scale-ops (item 2): the REAL-DIMS Q4_K operator sweep.  Re-runs the EXISTING
# operator TBs -- same goldens / same check contracts -- at the real GLM-5.2
# operator magnitudes via TB `define overrides (the vector generators grew
# parameters, they were NOT forked; all slice defaults stay byte-identical):
#   * glm_matmul_q4k   K in {512,2048,6144}, KMAX=6144 -> NSB=24 super-blocks
#                      (the real per-projection K) -- BIT-EXACT vs the ggml
#                      Q4_K golden tools/q4k_ref.py (~40 s)
#   * moe_router_q4k   N_EXPERT=256 TOPK=8 (the real expert count / top-K),
#                      HIDDEN=128 -- renorm invariant Sum(w)=SCALE (~5 min)
#   * swiglu_expert_q4k INTER=2048 (real INTER_MOE; down proj = 8 Q4_K
#                      super-blocks/column), HIDDEN=64 -- tolerance golden
#   * glm_softmax      LEN=2048 (the real DSA window index_topk), committed
#                      logit envelope (shifted args >= -63; larger magnitudes
#                      are outside the exp pipe's documented range) (~3 min)
#   * rmsnorm_unit     LEN=6144 campaign (the real MODEL_DIM) -- the committed
#                      TB already sweeps it; re-run here so the gate is
#                      self-contained (~20 s)
#   * rope_interleave_unit ROT_DIM=64 (the real qk_rope_head_dim), positions
#                      to ~1M -- likewise already real-dim; re-run here (~1 s)
# NOT covered (stated, not implied): router/swiglu/softmax GEMV K stays below
# the real 6144 reduction (that mechanism is what the K=6144 GEMM leg proves),
# and there is no standalone mla_attn_q4k real-geometry TB on main (the MLA
# datapath is gated at the slice by `make model-q4k`).
scale-ops:
	@mkdir -p $(BUILD_DIR)
	@# glm_matmul_q4k at real per-projection K (NSB up to 24), bit-exact.
	@python3 tools/q4k_matmul_gen.py 30 2 2 build/q4k_real_vec.txt 512,2048,6144,6144,6144 >/dev/null
	@$(IVERILOG) $(IFLAGS) -DTB_KMAX=6144 -DTB_VEC='"build/q4k_real_vec.txt"' -DTB_TIMEOUT_NS=20000000 \
	    -o $(BUILD_DIR)/glm_matmul_q4k_real_sim test/glm_matmul_q4k_tb.v src/glm_matmul_q4k.v
	@printf '[%s] ' "glm_matmul_q4k(K=6144)"; $(VVP) $(BUILD_DIR)/glm_matmul_q4k_real_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_q4k real-dims"; exit 1; }
	@# rmsnorm_unit at the real MODEL_DIM (LEN=6144 campaign in the committed TB).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rmsnorm_unit_real_sim test/rmsnorm_unit_tb.v src/rmsnorm_unit.v
	@printf '[%s] ' "rmsnorm_unit(LEN=6144)"; $(VVP) $(BUILD_DIR)/rmsnorm_unit_real_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rmsnorm_unit real-dims"; exit 1; }
	@# rope_interleave_unit at the real qk_rope_head_dim (ROT_DIM=64, pos to ~1M).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rope_real_sim test/rope_interleave_unit_tb.v src/rope_interleave_unit.v
	@printf '[%s] ' "rope_interleave_unit(ROT_DIM=64)"; $(VVP) $(BUILD_DIR)/rope_real_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rope_interleave_unit real-dims"; exit 1; }
	@# swiglu_expert_q4k at the real INTER_MOE=2048 (down proj NSB=8).
	@python3 tools/swiglu_q4k_gen.py 4 64 2048 4 build/swiglu_q4k_real_vec.txt >/dev/null
	@$(IVERILOG) $(IFLAGS) -DTB_HIDDEN=64 -DTB_INTER=2048 -DTB_KMAX=2048 -DTB_VEC='"build/swiglu_q4k_real_vec.txt"' \
	    -DTB_DONE_GUARD=400000 -DTB_TIMEOUT_NS=600000000 \
	    -o $(BUILD_DIR)/swiglu_expert_q4k_real_sim test/swiglu_expert_q4k_tb.v \
	    src/swiglu_expert_q4k.v src/glm_matmul_q4k.v src/glm_act.v
	@printf '[%s] ' "swiglu_expert_q4k(INTER=2048)"; $(VVP) $(BUILD_DIR)/swiglu_expert_q4k_real_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_q4k real-dims"; exit 1; }
	@# glm_softmax at the real DSA attention window (LEN=2048 rows).
	@$(IVERILOG) $(IFLAGS) -DTB_SOFTMAX_LEN=2048 -o $(BUILD_DIR)/glm_softmax_real_sim \
	    test/glm_softmax_tb.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_softmax(LEN=2048)"; $(VVP) $(BUILD_DIR)/glm_softmax_real_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_softmax real-dims"; exit 1; }
	@# moe_router_q4k at the real expert count / top-K (256 / 8).
	@$(IVERILOG) $(IFLAGS) -DTB_N_EXPERT=256 -DTB_TOPK=8 -DTB_HIDDEN=128 -DTB_TIMEOUT_NS=600000000 \
	    -o $(BUILD_DIR)/moe_router_q4k_real_sim test/moe_router_q4k_tb.v \
	    src/moe_router_q4k.v src/glm_matmul_q4k.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_q4k(256/top-8)"; $(VVP) $(BUILD_DIR)/moe_router_q4k_real_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_q4k real-dims"; exit 1; }
	@echo "scale-ops: real-dims Q4_K operator sweep passed (GEMM K=6144/NSB=24 bit-exact; router 256/top-8; SwiGLU INTER_MOE=2048; softmax LEN=2048; rmsnorm LEN=6144; rope ROT_DIM=64)"

# batched-q4k (item 3): the BATCHED PE_M>1 assembled-model golden.  Two
# glm_model_q4k instances share the model-q4k weight set (the SAME
# tools/glm_model_q4k_tb_gen.py vectors): a PE_M=2 batched forward's row r must
# equal a standalone PE_M=1 forward on that row's token -- logits + argmax +
# h_state BIT-EXACT -- and row 0 is ALSO compared against the assembled numpy
# golden directly (chaining the batch proof to the reference, not just
# DUT-vs-DUT).  Committed slice; each scenario = B+1 extra full forwards ->
# minutes-long in iverilog (like model-q4k), so standalone, not in `unittests`.
BATCHED_Q4K_SRCS := test/glm_model_q4k_pem_tb.v src/glm_model_q4k.v src/glm_decoder_block_q4k.v \
	src/mla_attn_q4k.v src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v \
	src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v

batched-q4k:
	@mkdir -p $(BUILD_DIR)
	@python3 tools/glm_model_q4k_tb_gen.py >/dev/null          # -> build/mq4k/*.hex (committed-slice golden)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_q4k_pem_sim $(BATCHED_Q4K_SRCS)
	@printf '[%s] ' "glm_model_q4k_pem(PE_M=2)"; $(VVP) $(BUILD_DIR)/glm_model_q4k_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_q4k_pem"; exit 1; }
	@echo "batched-q4k: glm_model_q4k PE_M=2 rows == per-row PE_M=1 runs (BIT-EXACT logits+argmax+h_state; row 0 anchored to the numpy golden)"



# provision-selftest -- REAL streaming GGUF -> on-device model image + manifest
# ----------------------------------------------------------------------------
# docs/USAGE_GAPS.md finding #1 ("real provisioning").  tools/ckpt_pack_q4k.py
# only round-trips a SYNTHETIC tiny GGUF and emits ASCII $readmemh hex (which
# for the 467 GB checkpoint explodes to ~950 GB of text).  tools/provision_image.py
# is the real one: it STREAMS a real GGUF in bounded chunks (O(1) RAM -- a
# hand-rolled header parser + seek()/read(), never memmap-faulting the blob),
# lays the raw Q4_K/Q6_K/Q8_0/... tensor blocks into a BINARY block image, emits
# a JSON MANIFEST (per-tensor {name,type,offset,length,sha256} + top-level
# {model,total_bytes,image_sha256,format_version,...}) and a resident-hot vs
# demand-streamed-expert flash_base/len SEGMENT list a boot loader can consume.
# The selftest builds the image + manifest from a small REAL GGUF, then VERIFIES
# the round-trip: each tensor's image bytes recompute to its manifest sha256 AND
# equal the original GGUF tensor bytes read via the gguf-py oracle (the same
# reader tools/gguf_crosscheck.py uses -- an INDEPENDENT parser from build's).
# It also proves the verifier can FAIL (1-byte tamper must be detected) and that
# the hot/expert flash-segment split is real logic (synthetic MoE-name check).
#
# Override the model / gguf-py location:
#   make provision-selftest PROVISION_GGUF=/path/to.gguf PROVISION_GGUF_PY=/path/to/gguf-py
# ============================================================================
.PHONY: provision-selftest
PROVISION_GGUF    ?= $(HOME)/.cache/wpu/smollm2-135m-q8_0.gguf
PROVISION_GGUF_PY ?= $(HOME)/.cache/wpu/gguf-py

provision-selftest:
	@mkdir -p $(BUILD_DIR)
	@[ -f "$(PROVISION_GGUF)" ] || { echo "provision-selftest needs a real GGUF file:"; \
	    echo "  make provision-selftest PROVISION_GGUF=/path/to/model.gguf PROVISION_GGUF_PY=/path/to/llama.cpp/gguf-py"; \
	    echo "  (any small GGUF works, e.g. SmolLM2-135M Q8_0; gguf-py = the package dir inside a llama.cpp checkout)"; exit 1; }
	@[ -d "$(PROVISION_GGUF_PY)" ] || { echo "provision-selftest: gguf-py dir not found: $(PROVISION_GGUF_PY)"; \
	    echo "  pass PROVISION_GGUF_PY=/path/to/llama.cpp/gguf-py"; exit 1; }
	@python3 tools/provision_image.py --gguf-py $(PROVISION_GGUF_PY) \
	    selftest $(PROVISION_GGUF) 2>&1 | tee $(BUILD_DIR)/provision_selftest.log
	@grep -q 'provision_image ALL [0-9]\+ TESTS PASSED' $(BUILD_DIR)/provision_selftest.log \
	    || { echo "FAILED: provision-selftest (no ALL-PASSED line)"; exit 1; }
	@grep -q 'PROVISION .* OK:' $(BUILD_DIR)/provision_selftest.log \
	    || { echo "FAILED: provision-selftest (no PROVISION OK line)"; exit 1; }
	@echo "provision-selftest: real GGUF -> binary image + sha256 manifest + flash-segment list; round-trip verified (image == manifest == original GGUF bytes), O(1) RAM, tamper detected"



# boot-integrity : boot-time INTEGRITY + VERSION manifest gate
# ----  NEW delimited section: `make boot-integrity`  ------------------------
# ----------------------------------------------------------------------------
#   USAGE_GAPS LOCK-IN-NOW A (findings #2/#3/#4/#35/#40).  src/boot_loader.v
#   grew an ADDITIVE, DEFAULT-OFF `INTEGRITY` parameter:
#     * INTEGRITY=0 (default): byte-identical to the pre-change, BMC/k-induction
#       -proven power-up model-load sequencer -- `done` releases unconditionally
#       on copy-complete; the manifest ports are sunk, boot_fail/err_code stay 0.
#     * INTEGRITY=1: before `done` is asserted, the LATCHED manifest header
#       (MAGIC / format-model VERSION / total-length / a running CRC-32 folded
#       over the loaded words) must match.  On ANY mismatch the engine FAILS
#       CLOSED -- it registers boot_fail + an err_code (MAGIC/VER/LEN/CRC) and
#       NEVER asserts `done` (inference stays gated).  A partial (truncated),
#       corrupt (bad-CRC), or wrong-version image can never be DMA'd and
#       silently released as a working model.
#
#   This gate proves BOTH halves:
#     (1) DEFAULT UNCHANGED -- a yosys SEQUENTIAL EQUIVALENCE (equiv_simple +
#         equiv_induct, with the skid-FIFO memory_map'd to flops) of the
#         INTEGRITY=0 module against the pre-change source at $(BOOT_INTEG_BASE).
#         Both sides are wrapped in an IDENTICAL `u`-instance shell exposing only
#         the pre-gate port set, so the FIFO/cursor registers match by name and
#         the induction step closes; `equiv_status -assert` fails (non-zero) on
#         any unproven point.  (The new manifest logic is dead at INTEGRITY=0 and
#         is const-folded away by `opt`, so the proof scope is the released FSM.)
#     (2) FAIL-CLOSED BEHAVIOUR -- test/boot_loader_manifest_tb.v runs the
#         INTEGRITY=1 and INTEGRITY=0 engines SIDE BY SIDE on the same images
#         (good / truncated / wrong-version / bad-CRC / bad-magic / re-run after
#         a fail / empty).  ON: good->done (no fail); each bad image->boot_fail
#         with the right err_code and `done` NEVER asserts.  OFF: releases every
#         image and stays inert (a live "identical to today" check).  A mutation
#         that releases a bad image is caught by the done/boot_fail exclusion
#         monitor -- the TB cannot vacuously pass.
# ============================================================================
.PHONY: boot-integrity
BOOT_INTEG_BASE ?= 9907504
# printf template for the equivalence wrapper (single `u` instance, pre-gate
# ports only); the two %s are the extra instance connections -- empty for the
# pre-change GOLD, the sunk manifest ports for the INTEGRITY=0 GATE.
BL_WRAP_FMT := module bl_equiv_top (\n input wire clk, input wire rst, input wire start,\n input wire [2:0] seg_count,\n input wire [127:0] seg_flash_base, input wire [127:0] seg_ddr_base,\n input wire [63:0] seg_len,\n output wire flash_req, output wire [31:0] flash_addr,\n input wire flash_ready, input wire flash_rvalid, input wire [63:0] flash_rdata,\n output wire ddr_we, output wire [31:0] ddr_addr, output wire [63:0] ddr_wdata,\n input wire ddr_ready,\n output wire busy, output wire done, output wire [18:0] words_done);\n boot_loader u (\n .clk(clk),.rst(rst),.start(start),.seg_count(seg_count),\n .seg_flash_base(seg_flash_base),.seg_ddr_base(seg_ddr_base),.seg_len(seg_len),%s\n .flash_req(flash_req),.flash_addr(flash_addr),\n .flash_ready(flash_ready),.flash_rvalid(flash_rvalid),.flash_rdata(flash_rdata),\n .ddr_we(ddr_we),.ddr_addr(ddr_addr),.ddr_wdata(ddr_wdata),.ddr_ready(ddr_ready),\n .busy(busy),.done(done),.words_done(words_done)%s);\nendmodule\n

boot-integrity:
	@mkdir -p $(BUILD_DIR)
	@# ---- (1) default-OFF (INTEGRITY=0) sequential-equivalence vs pre-change ----
	@git show $(BOOT_INTEG_BASE):src/boot_loader.v > $(BUILD_DIR)/boot_loader_base.v
	@printf '$(BL_WRAP_FMT)' '' '' > $(BUILD_DIR)/bl_gold_wrap.v
	@printf '$(BL_WRAP_FMT)' ".mf_magic(32'b0),.mf_version(16'b0),.mf_len(19'b0),.mf_crc(32'b0)," ",.boot_fail(),.err_code()" > $(BUILD_DIR)/bl_gate_wrap.v
	@$(YOSYS) -q -p "read_verilog -sv -I src $(BUILD_DIR)/boot_loader_base.v $(BUILD_DIR)/bl_gold_wrap.v; \
	    prep -top bl_equiv_top -flatten; memory_map; opt -full; opt_clean -purge; rename bl_equiv_top gold; design -stash gdes; \
	    read_verilog -sv -I src src/boot_loader.v $(BUILD_DIR)/bl_gate_wrap.v; \
	    prep -top bl_equiv_top -flatten; memory_map; opt -full; opt_clean -purge; rename bl_equiv_top gate; \
	    design -copy-from gdes -as gold gold; \
	    equiv_make gold gate equiv; prep -top equiv; \
	    equiv_simple -undef; equiv_induct -undef; equiv_status -assert" \
	    && echo "[boot-integrity] EQUIV PROVEN: boot_loader(INTEGRITY=0) == $(BOOT_INTEG_BASE) (default netlist UNCHANGED)" \
	    || { echo "FAILED: boot-integrity default-off equivalence"; exit 1; }
	@# ---- (2) fail-closed manifest behaviour (INTEGRITY=1 vs INTEGRITY=0 side by side) ----
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/boot_loader_manifest_sim test/boot_loader_manifest_tb.v src/boot_loader.v
	@printf '[%s] ' "boot_loader_manifest"; $(VVP) $(BUILD_DIR)/boot_loader_manifest_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: boot_loader_manifest"; exit 1; }
	@echo "boot-integrity: INTEGRITY=0 proven byte-equivalent to $(BOOT_INTEG_BASE); INTEGRITY=1 fail-closes on truncated/wrong-version/bad-CRC/bad-magic (done never releases a bad model)"



# WEIGHT-PATH SECDED ECC  (USAGE_GAPS §B / finding #32)
#------------------------------------------------------------------------------
# The resident ~467 GB weights have NO bit-error protection.  weight_loader_q4k
# gains a DEFAULT-OFF (WEIGHT_ECC=0) SECDED ECC read path reusing src/ecc_secded.v:
# with WEIGHT_ECC=1 each DATA_W read word is modelled as ECC_LANE_W-wide SECDED
# lanes (the weight memory stores no check bits, so the codec is a DECODE STAGE
# on the read data with an injectable fault port) -- single-bit errors CORRECTED,
# double-bit errors FLAGGED (registered sticky ecc_uncorrectable + a corrected-
# error counter for scrub/telemetry).
#
#   weight-ecc        : the new self-checking fault-injection TB (test/weight_ecc_tb.v)
#                       -- default-off transparency, SBU correction, DBU detection.
#   weight-ecc-equiv  : PROVES the default (WEIGHT_ECC=0) module is unchanged --
#                       yosys sequential equivalence (equiv_simple + equiv_induct)
#                       of the modified default-param module vs its pre-change
#                       version at git rev WEIGHT_ECC_BASE.  The 3 added ECC-only
#                       ports are stripped from the gate so the port sets match;
#                       every retained state bit / output is proven equivalent.
.PHONY: weight-ecc weight-ecc-equiv
weight-ecc:
	@mkdir -p $(BUILD_DIR)
	@# self-checking SECDED read-path proof: two loaders (WEIGHT_ECC 0 vs 1) on the
	@# SAME memory -- clean streams byte-identical (ECC transparent), a 1-bit fault is
	@# CORRECTED (stream still clean + counter increments), a 2-bit fault FLAGGED.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_ecc_sim test/weight_ecc_tb.v src/weight_loader_q4k.v src/ecc_secded.v
	@printf '[%s] ' "weight_ecc"; $(VVP) $(BUILD_DIR)/weight_ecc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight_ecc"; exit 1; }
	@echo "weight-ecc: SECDED weight read path -- default-off transparent, single-bit corrected, double-bit flagged"

# REFERENCE for the byte-identical proof: an ECC-PHYSICALLY-REMOVED loader, NOT a
# frozen commit.  It was originally `git show 9907504` (the pre-ECC commit).  That
# broke the moment the loader's CORE changed for a reason unrelated to ECC -- the
# pj_iss column-index width fix (a real bug: pj_iss was sized by super-block width
# SBW, wrapping for PE_N > 2^SBW; see src/weight_loader_q4k.v).  A frozen pre-ECC
# commit cannot carry that fix, so the reference is now a checked-in file =
# (9907504's loader) + (the identical pj_iss fix), with the ECC ports/params/generate
# genuinely absent.  The proof therefore stays honest: WEIGHT_ECC=0 elaborates to the
# same netlist as a loader that never had ECC at all.  When the loader's non-ECC core
# changes again, this reference must be updated in lockstep -- and this gate FAILS
# until it is, which is the point (it guards the core netlist).
WEIGHT_ECC_REF := test/ref/weight_loader_q4k_noecc_ref.v
weight-ecc-equiv:
	@mkdir -p $(BUILD_DIR)
	@$(YOSYS) -q -p "read_verilog -lib -sv -I src src/ecc_secded.v; \
	    read_verilog -sv -I src $(WEIGHT_ECC_REF); \
	    prep -top weight_loader_q4k; opt_clean -purge; rename weight_loader_q4k gold; design -stash gdes; \
	    read_verilog -lib -sv -I src src/ecc_secded.v; \
	    read_verilog -sv -I src src/weight_loader_q4k.v; \
	    prep -top weight_loader_q4k; \
	    delete weight_loader_q4k/w:ecc_err_inject weight_loader_q4k/w:ecc_corr_count weight_loader_q4k/w:ecc_uncorrectable; \
	    opt_clean -purge; rename weight_loader_q4k gate; \
	    design -copy-from gdes -as gold gold; \
	    equiv_make gold gate equiv; prep -top equiv; \
	    equiv_simple -undef; equiv_induct -undef; equiv_status -assert" \
	    && echo "[weight-ecc-equiv] PROVEN: weight_loader_q4k(WEIGHT_ECC=0) == ECC-removed reference (default read path byte-identical)" \
	    || { echo "FAILED: weight-ecc-equiv"; exit 1; }

# weight-loader-lanes: the SAME loader->GEMM mixed golden as the default (PE_N=4) gate,
# but at a SCALED lane count (PE_N=32, DATA_W=512) -- the config the pj_iss column-width
# fix opened.  Without this nothing in CI runs the loader past PE_N=16, so the fix (and
# the DATA_W>=16*PE_N guard) could silently regress.  Golden regenerated by emit_wlmixed
# at PE_N=32; own vector file so it never races the default gate's build/wlmixed_vec.txt.
weight-loader-lanes:
	@mkdir -p $(BUILD_DIR)
	@python3 -c "import sys;sys.path.insert(0,'tools');import q4k_mixed_gen as g;g.emit_wlmixed('$(BUILD_DIR)/wlmixed_vec_p32.txt',PE_M=2,PE_N=32)" >/dev/null
	@$(IVERILOG) $(IFLAGS) -s weight_loader_q4k_mixed_tb \
	    -Pweight_loader_q4k_mixed_tb.PE_N=32 -Pweight_loader_q4k_mixed_tb.DATA_W=512 \
	    -Pweight_loader_q4k_mixed_tb.VEC_FILE=\"$(BUILD_DIR)/wlmixed_vec_p32.txt\" \
	    -o $(BUILD_DIR)/wl_lanes_sim test/weight_loader_q4k_mixed_tb.v src/weight_loader_q4k.v src/glm_matmul_q4k.v
	@printf '[%s] ' "weight_loader_lanes(PE_N=32)"; $(VVP) $(BUILD_DIR)/wl_lanes_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight-loader-lanes"; exit 1; }
	@echo "weight-loader-lanes: loader->GEMM bit-exact at PE_N=32/DATA_W=512 -- lane ceiling open, pj_iss fix guarded"



# HOST<->DEVICE PROTOCOL EXTENSION  (USAGE_GAPS §C, findings #19/#26)
#   glm_q4k_system_cdc gains a DEFAULT-OFF parameter PROTO_CTX.  With PROTO_CTX=1
#   the 2-clock host<->device frame carries a CONTEXT/SEQUENCE id (so the host
#   can multiplex N contexts and demux returned tokens) plus a TELEMETRY-readback
#   opcode that returns registered device counters (tokens / runs / done / stall).
#   The ids + counters cross host_clk<->core_clk ONLY through the existing
#   cdc_async_fifo pattern -- no new unsynchronized crossing (still passes `cdc`).
#   With PROTO_CTX=0 the top is byte-identical to the shipped device (proven by
#   `cdc-protocol-equiv`), so `cdc`/`synth-glm` are unaffected.
# ============================================================================
.PHONY: cdc-protocol cdc-protocol-equiv

# ---- functional gate: ctx-id round-trip + telemetry readback (PROTO_CTX=1) ---
# Drives the 2-clock top across its two ASYNCHRONOUS clocks with the same
# faithful weight/KV/Flash/DDR5/loader backing the perf TB uses (so the compute
# box emits REAL tokens), tags each token-gen request with a DISTINCT ctx id and
# checks the matching id rides BACK on every response (round-trip/demux), then
# polls telemetry and checks the returned counters EXACTLY (and that they
# advance between two readbacks -- a constant-returning stub fails).
cdc-protocol:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/cdc_protocol_ctx_sim \
	    test/cdc_protocol_ctx_tb.v $(GLM_Q4K_CDC_SRCS)
	@printf '[%s] ' "cdc_protocol_ctx"; $(VVP) $(BUILD_DIR)/cdc_protocol_ctx_sim | grep -E '\[cdc_protocol_ctx\] ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: cdc-protocol"; exit 1; }
	@# INJECTION -- pre-fix ctx tagging (cur_ctx follows EVERY pop, OP_TELEM included):
	@#   the mid-run-telemetry test must catch the retagged token -> MUST FAIL.
	@$(IVERILOG) $(IFLAGS) -DINJECT_CTXTAG -o $(BUILD_DIR)/cdc_protocol_ctx_inject \
	    test/cdc_protocol_ctx_tb.v $(GLM_Q4K_CDC_SRCS)
	@printf '[%s] ' "cdc_protocol_ctx_INJECT_ctxtag"; \
	    if $(VVP) $(BUILD_DIR)/cdc_protocol_ctx_inject 2>/dev/null | grep -qE 'ALL [0-9]+ TESTS PASSED'; then \
	        echo "FAILED: CTXTAG injection NOT caught (a mid-run OP_TELEM pop retagged the run's token and no test noticed)"; exit 1; \
	    else echo "injection correctly FAILED (cur_ctx follows the RUN, not the pop -- the mid-run-telemetry test is load-bearing)"; fi

# ---- default-unchanged proof: PROTO_CTX=0 netlist == the pre-change top -------
# Synthesizes the wrapper (compute/CDC submodules blackboxed via -lib, so the
# proof is SCOPED to glm_q4k_system_cdc's own logic) at PROTO_CTX=0 and at the
# pre-change git rev CDC_PROTO_BASE, under an IDENTICAL yosys script, and asserts
# the two CELL histograms are byte-identical -- same gate types, same counts,
# same 54 flops, same {glm_q4k_system, cdc_async_fifo x2, reset_sync x2} instances.
# The PROTO_CTX ports add ONLY dead wires (0 cells), so any real logic change to
# the default netlist makes the diff non-empty and fails.  (Interactively this
# was also cross-checked with a full yosys equiv_simple+equiv_induct sequential
# equivalence -- 7988/7988 $equiv cells proven -- via matched interface adapters.)
CDC_PROTO_BASE ?= 0d2a2e5
CDC_PROTO_LIBS := src/glm_q4k_system.v src/cdc_async_fifo.v src/reset_sync.v
cdc-protocol-equiv:
	@mkdir -p $(BUILD_DIR)
	@git show $(CDC_PROTO_BASE):src/glm_q4k_system_cdc.v > $(BUILD_DIR)/cdc_proto_base.v
	@# SYMMETRY: the base run must apply the SAME chparam as the new run below --
	@#   yosys elaborates an explicitly-chparam'd top as a parametrized derivative
	@#   and can optimize it marginally differently from the default elaboration
	@#   (observed: 144 vs 143 cells on IDENTICAL source), so a default-vs-chparam
	@#   comparison flags false netlist drift.
	@$(YOSYS) -q -p "read_verilog -lib -sv -I src $(CDC_PROTO_LIBS); \
	    read_verilog -sv -I src $(BUILD_DIR)/cdc_proto_base.v; \
	    chparam -set PROTO_CTX 0 glm_q4k_system_cdc; \
	    synth -top glm_q4k_system_cdc -flatten; tee -o $(BUILD_DIR)/cdc_proto_stat_base.txt stat"
	@$(YOSYS) -q -p "read_verilog -lib -sv -I src $(CDC_PROTO_LIBS); \
	    read_verilog -sv -I src src/glm_q4k_system_cdc.v; \
	    chparam -set PROTO_CTX 0 glm_q4k_system_cdc; \
	    synth -top glm_q4k_system_cdc -flatten; tee -o $(BUILD_DIR)/cdc_proto_stat_new.txt stat"
	@sed -n '/[0-9] cells$$/,$$p' $(BUILD_DIR)/cdc_proto_stat_base.txt > $(BUILD_DIR)/cdc_proto_cells_base.txt
	@sed -n '/[0-9] cells$$/,$$p' $(BUILD_DIR)/cdc_proto_stat_new.txt  > $(BUILD_DIR)/cdc_proto_cells_new.txt
	@# LIVENESS: if a yosys upgrade changes `stat`'s table format the sed above matches
	@#   nothing and empty==empty would pass VACUOUSLY (its equiv siblings assert the
	@#   =1 variant differs; here the cheap direct guard is: extraction must be non-empty).
	@[ -s $(BUILD_DIR)/cdc_proto_cells_base.txt ] && [ -s $(BUILD_DIR)/cdc_proto_cells_new.txt ] \
	    || { echo "FAILED: cdc-protocol-equiv (stat extraction EMPTY -- yosys stat format changed; the diff below would be vacuous)"; exit 1; }
	@diff $(BUILD_DIR)/cdc_proto_cells_base.txt $(BUILD_DIR)/cdc_proto_cells_new.txt \
	    && echo "[cdc-protocol-equiv] PROVEN: glm_q4k_system_cdc(PROTO_CTX=0) cell netlist == $(CDC_PROTO_BASE) (byte-identical default; PROTO_CTX adds only dead ports)" \
	    || { echo "FAILED: cdc-protocol-equiv (default netlist changed)"; exit 1; }

