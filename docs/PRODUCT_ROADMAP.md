# Product Roadmap — GLM-5.2 accelerator (product, not research)

> **TRACK NOTE (Q4_K-native).** The current / `main` track is **Q4_K-native**, targeting the published
> [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF) checkpoint (467 GB,
> ~38% smaller than the 753 GB FP8 checkpoint). The active datapath is `glm_model_q4k`,
> `glm_decoder_block_q4k`, `mla_attn_q4k`, `moe_router_q4k`, `swiglu_expert_q4k`, `mtp_head_q4k`,
> with system tops `glm_q4k_soc(_ms)` → `glm_q4k_system` → `glm_q4k_system_cdc`, the Q4_K GEMM core
> `glm_matmul_q4k`, the loader `weight_loader_q4k.v`, and the ggml-Q4_K reference `tools/q4k_ref.py`
> (checkpoint packer `tools/ckpt_pack_q4k.py`). **FP8 is the PRIOR / PRESERVED track** on branch
> **`fp8`** + tag **`fp8-verified-baseline`** (and the older research prototype on branch `prototype`,
> frozen at `47fb7f8`); it is referenced here as prior/preserved, never current. See
> [`Q4K_RETARGET.md`](Q4K_RETARGET.md).
>
> **Honest scope of the Q4_K proof** (per the evidence ledger, consistent with the
> [README](../README.md) *What's proven* table). `make q4k` proves the Q4_K **GEMM core**
> (`glm_matmul_q4k`) **bit-exact to the team's own ggml-Q4_K reference** (`tools/q4k_ref.py`) — **not**
> "bit-exact to the real UD-Q4_K_XL file": the RTL now also consumes **Q6_K/Q8_0/F16** (per-column
> `w_type` routing, `make mixedtype`, bit-exact to the same reference), and that reference's **dequant
> layer is now proven bitwise-equal to real GGUF bytes** (376,586,240 weights — Q4_K/Q6_K/Q8_0, two
> real published GGUFs — vs llama.cpp's own dequant — [`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md));
> llama.cpp **whole-runtime** arithmetic stays out-of-contract. The **assembled** `glm_model_q4k` now has an **end-to-end
> numeric golden** — `make model-q4k`: full forward vs the numpy reference `tools/glm_model_q4k_ref.py`,
> ALL 1155 tests bit-exact (logits+argmax+h_state; plus `make model-q4k-acthw` through the ACT_HW=1
> datapath) — but that golden is still our own numpy reimpl, NOT llama.cpp/GGUF. The **FPGA fit is
> measured** (Vivado ML 2026.1 synth + full P&R on XCKU3P — see P3.2); every tok/s, BOM/TCO, and LOI
> figure remains **[EST]**, not measured. The remaining OPEN item — **real-checkpoint validation vs the
> real GGUF bytes / llama.cpp** — is exactly what the gates below are framed around.

The `prototype` branch (frozen at `47fb7f8`) holds the **prior FP8 research prototype**: the full FP8
datapath + memory system + ultra-perf batching stack, bit-exact and mechanism-proven at a
small-but-faithful slice, with honest gaps documented. It answered *"does the architecture work, and how
fast can it go?"* — for FP8, at a slice. It is **not** the current product.

`main` now develops exactly one thing — the **GLM-5.2 Q4_K local-inference accelerator at rung ① — the
FPGA prove-it track**: a manufacturable design whose near-term goal is a **working FPGA demo** proving the
published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` runs **reliably on real low-end FPGA silicon, offline and
bit-exact-to-our-ggml-Q4_K-reference**. The **full product** — the funded custom board (rung ②) and the
volume ASIC/SoC (rung ③) — is the **roadmap documented below**, not the code `main` develops right now.
The mindset shifts from *demonstrate + measure a mechanism* to *run the real model correctly, at full
scale, robustly, and ship it.*

> **What the product IS: a LOCAL, single-user personal box that works with the ethernet unplugged.**
> One box, one user, running the full 753 B model **locally** — it **works fully offline / air-gapped:
> nothing leaves because there's no path out**. The audit is literally *"does it still answer with the
> ethernet cable unplugged?"* — **yes**, the strongest, binary form of 'non-egress'; it categorically
> excludes *every* cloud option (in-VPC / zero-retention / confidential-computing TEE all still need a
> connection and fail the unplugged test). Lead with the **capability** that unlocks, not the defense:
> finally run a **full frontier** model in the disconnected places you're locked out of today (SCIF /
> classified, isolated OT & critical-infra, field/edge, denied-connectivity), and **own it outright** —
> no per-token API fees, no vendor that can rate-limit, deprecate, or cut you off. *Offline alone is
> table-stakes (a 70 B laptop model is offline too); the moat is the **combination** — offline + full
> frontier (753 B) + appliance/seat price.* (Honest: the ~467 GB Q4_K checkpoint is loaded **once** —
> itself doable offline — and model updates are physical re-provisioning; after that, fully disconnected.)
> The performance metric that matters is **single-user interactive throughput**, and it is
> **rung-dependent** (set by the silicon's memory bandwidth — see
> [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)): **~5–8 tok/s [EST] on the near-term prove-it FPGA (rung ①),
> ~15–40 tok/s [EST] on the funded custom board (rung ②)**, and ~40+ at manufacturing volume (rung ③) —
> the **same bit-exact Q4_K RTL** on every rung, only the memory interface changes. *(Update — a
> measured-roofline design-point menu now refines these [EST] ranges, using h/U measured on an OLMoE
> proxy trace, [`H_MEASUREMENT.md`](H_MEASUREMENT.md): 1–2 NVMe, no multipliers ~0.5–1 tok/s; 90 GB
> DRAM + 100 GB/s → 13–24; 90 GB + 200 GB/s (ONFI 64-ch) → 25–47; 225 GB + 200 GB/s → 54–127 — the
> "100 tok/s" design point. Spec-decode amortization must be read as A/U(K) ≈ 1.1–1.3× at K=4
> (measured U(4)=2.25–2.64), not ~2×. All still [EST]; h/U are proxy measurements — see also
> [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md).)* *(Updated 2026-07: the **rung-③ primary
> design point is now FULL RESIDENCY** — 512 GB LPDDR5X (~1.1 TB/s) holds the whole ~467 GB
> checkpoint, h=1 by construction, design point **≈80 tok/s [measured-inputs EST]** (U(K) **and**
> the MTP accept rate r both GLM-family measured — job B's vLLM MTP sweep: r₁=0.87, per-position
> decay, memory-bound optimum K=1–2; ~95 if GLM-5.2's deeper MTP hits its published accept depth —
> [`H_MEASUREMENT.md`](H_MEASUREMENT.md)) — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md). The
> streaming design-point menu above stays true and active for rung ①, the hybrid upside SKU, and
> \>512 GB checkpoints — but "54–127" is no longer the rung-③ design point. U(K) is now
> **GLM-family measured** — GLM-4.5-Air traced via MoE-gate hooks
> ([`H_MEASUREMENT.md`](H_MEASUREMENT.md) 2nd measurement): U(4)=2.60–2.71, U(8)=4.19–4.41 —
> superseding the OLMoE first-pass proxy values.)* The design is
> deliberately **NVMe/PCIe-bandwidth-bound to be cheap** (an NVMe SSD holds the whole model; **fast DDR** —
> DDR4 on rung ①, DDR5 or HBM on rung ② — caches the hot working set). Any **aggregate / datacenter-batch**
> numbers in these docs (B≈256, per-user ~0.14 tok/s) are a **secondary analysis of a different,
> non-target deployment** of the same silicon — the RTL supports it, but it is **not this product**, and
> its per-user latency never describes the box you plug in.

> **Two tracks.** This doc is the **RTL / silicon track** (make the chip correct, full-scale, robust,
> synthesizable). The **device / appliance track** — the USB-C external box (form factor, power, thermal,
> host software, enclosure, manufacturing, pricing) — is in
> [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md). Its first gates (real-model fidelity + FPGA fit) are the
> P1 fidelity chain and the FPGA vendor-flow measurement (P3.2 / [`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md)).

---

## The research → product gap (what must change)

| Dimension | Have (Q4_K, verified in-repo) | Product (need) |
|---|---|---|
| Correctness scope | **Q4_K GEMM core bit-exact to the ggml-Q4_K reference** (`glm_matmul_q4k` 160/160, `q4k_prim` 18/18 vs `tools/q4k_ref.py`); `swiglu_expert_q4k` functional (240); `moe_router_q4k` structural invariants (40); the **assembled** `glm_model_q4k` full forward **bit-exact vs the numpy reference** `tools/glm_model_q4k_ref.py` (`make model-q4k`, ALL 1155 tests: logits+argmax+h_state; `make model-q4k-acthw` same golden through the ACT_HW=1 datapath) — plus spec==greedy self-consistency | the **full 467 GB UD-Q4_K_XL checkpoint** producing the real model's tokens **at full depth** end-to-end (real-checkpoint validation — the assembled golden is our own numpy reimpl, not llama.cpp/GGUF) |
| Format coverage | **Mixed-type DONE**: Q6_K/Q8_0/F16 dequant primitives (`src/q4k_mixed.vh`), per-column `w_type` routing in `glm_matmul_q4k`, `desc_wtype` in `weight_loader_q4k` — `make mixedtype` (`q6k_prim`, `q8_0_prim`, `glm_matmul_mixed` 32/32, `weight_loader_q4k_mixed` 192/192 incl. a 24-tile mixed sequence), bit-exact vs `q4k_ref.py` — the chip CAN now consume a real UD-Q4_K_XL type mix | bit-verify the mixed-type path **vs the real GGUF bytes** (still unchecked against llama.cpp arithmetic) |
| Scale | small faithful slice (128/6/8); the **full 753 B UD-Q4_K_XL config elaborates clean** (`test/full_config_elab_wrap.v`, verilator 0 errors; `make synth-glm` yosys `check -assert` exit 0) — **structure only, not a sim** | full config *functionally simulated/run* (6144, 78 layers, 256 experts, vocab 154880, 1 M ctx) — currently intractable (LM-head GEMV alone ~2.4e8 K-beats/token) |
| Batching/KV | per-row position/extent/sequence (`PER_ROW_POS/SLEN/SEQ`) threaded model→decoder→mla; `kv_cache_pager` `NSEQ` independent ring windows; multi-sequence batched attention (`PER_ROW_SEQ`) end-to-end through `glm_model_q4k`; batched multi-seq top `glm_q4k_soc_ms` with a top-owned per-layer KV store (`kv_mem`) + `N_STEPS` continuous-batching decode loop; expert-union-skip MoE batching **folded inline into `glm_decoder_block_q4k`** (standalone `batched_moe.v` removed); `spec_chain_top` MTP-chain draft — **validated as per-row / spec==greedy self-consistency (DUT-vs-DUT vs a standalone PE_M=1 `glm_model_q4k`), not a numeric golden**; byte-identical at `PER_ROW_SEQ=0`. **Scope of that "validated" (2026-07 correction): it covers the MODEL-level per-row path that gates exercise (`make mla-sparse`, `make batched-q4k`). The top `glm_q4k_soc_ms` itself is in NO gate and has NO testbench — repo-wide it is referenced only by docs — so its bit-exactness is the prior-FP8 result with the Q4_K re-run PENDING, exactly as `OPERATION_FLOW.md:331` / `SCALE_FUNCTIONAL.md:157` / `ULTRA_PERF.md:125` already state. This row previously read "all validated", which claimed more than the ledger proves. Also note no gate anywhere sets `PER_ROW_POS=1` at model level, and neither spec top enables it (`spec_batched_top.v` drives one shared scalar `.pos(cur_pos)`) — see that file's header.** | a resident dense DRAFT model (K_eff↑, needs a trained draft's weights); real-checkpoint validation on a GPU/large host |
| Memory | DDR5/NVMe/USB-C **PHYs stubbed** (TB) | licensed **PHY IP** integrated + signed off |
| Verification | bounded BMC (7 controllers + 1 ECC-ring) + 5 lifted to unbounded k-induction (`make formal`/`formal-ind`); directed TBs at slice; verilator line/toggle/branch coverage (`make coverage`) | coverage *closure*, constrained-random regression, gate-level sim, production-width formal |
| Reliability | ECC foundations (`ecc_mem_wrap` SECDED scrub, `kv_ecc_ring`), CDC/reset hardening (`reset_sync` wired), DVFS (`clk_throttle`); the die already carries the inline `die_clk` ICG (`glm_q4k_system.v:1307-1311`); `mbist_ctrl` is the verified single-port March **reference** (per-macro BIST collars are the physical-flow insertion, [`P2_MEMORY_MAP.md`](P2_MEMORY_MAP.md) §4 — not a hand-wire-in-top task) | full ECC/recovery, CDC sign-off, reset/init hardening, dual-port BIST collars + DFT/scan closed |
| Physical | **measured FPGA fit**: Vivado ML 2026.1 real synth + full P&R of `glm_q4k_system_cdc` on XCKU3P (compact + ACT_HW=1) — 142,320 LUT (87.5%) synth-stage (routed 141,298 LUT, `fpga/results/util_routed_ku3p_acthw1.rpt`), ~100K FF, 421 DSP, 0 BRAM, hold met; routed Fmax 10.2 → 17.2 → 46.5 MHz over bit-exact repipeline rounds, **campaign CLOSED at 4.6×** — the worst path is now route-dominated (wide-bus wiring at 87% utilization), physical work not arithmetic (see [`fpga/`](../fpga/README.md) + `fpga/results/`); **prior-FP8 sky130 realizability** on tag `fp8-verified-baseline` (see below) | **bitstream** on a board — the Fmax campaign is closed at 46.5 MHz (in the bring-up demo's target band; 200 MHz-class is rung-②/③ work) (rungs ①②; ASIC/tapeout is the rung-③ volume endgame — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) |
| Software | weight-pack tools (`ckpt_pack_q4k.py`/`flash_layout.py`); **host scaffold built** — OpenAI-compatible server + device protocol + **real GLM BPE tokenizer** + chat template + sampling ([`host/`](../host/README.md); simulator backend targets the on-`main` `glm_model_q4k` slice via `vvp` and returns real RTL argmax tokens — parse/protocol covered by `make host-test`, 32 tests) | production host **driver** (real USB backend), runtime/scheduler, quant-layout pipeline |
| Manufacturing | — | PCB, BOM, assembly, qualification |

---

## The #1 product gate (do this FIRST) — the Q4_K fidelity chain

**Real-checkpoint full-model fidelity.** Until the actual GLM-5.2 Q4_K weights produce the actual model's
next tokens through our datapath, there is no product. On the Q4_K track that gate is **not one step but a
short chain** (per the evidence ledger — steps 1–2 are now **DONE**, step 3 remains **OPEN**):

1. **Assembled-Q4_K numeric golden** *(`NEXT_STEPS_PLAN.md` B9)*. Previously the assembled
   `glm_model_q4k` (including the `mla_attn_q4k` `1/sqrt(d_head)` softmax scale) was checked **only** as
   spec==greedy self-consistency — a real lossless-speculation safety property, but **DUT-vs-DUT**, whose
   reference is itself a `glm_model_q4k`. Nothing asserted the assembled Q4_K forward pass matches a ggml /
   reference numeric golden. **Build a TB that compares a 1-token `glm_model_q4k` forward against an
   independent assembled ggml-Q4_K reference** (an extension of `tools/q4k_ref.py` beyond the GEMM core).
   **(DONE — `make model-q4k`: `glm_model_q4k` full forward — embed → L×(MLA+DSA+MoE) → final norm → LM
   head → argmax — vs the numpy reference `tools/glm_model_q4k_ref.py`, ALL 1155 tests bit-exact on
   logits+argmax+h_state; plus `make model-q4k-acthw`, the same golden through the ACT_HW=1
   serialized-activation datapath, also 1155. The golden is still our own numpy reimpl, NOT
   llama.cpp/GGUF — that gap is step 3.)**
2. **Mixed-type Q6_K/Q8_0/F16 consumer** *(B10)*. The RTL was Q4_K-only, so it **could not consume a real
   UD-Q4_K_XL checkpoint as-is** — the dynamic mix keeps sensitive tensors at Q6_K/Q8_0/F16, for which
   `q4k_ref.py` had Python goldens with no RTL consumer. Add an RTL dequant path (per-tensor `w_type`
   routing) so those tensors are consumed at their real precision (not re-quantized to Q4_K, which would
   void the fidelity of the moat).
   **(DONE — `make mixedtype`: Q6_K/Q8_0/F16 dequant primitives in `src/q4k_mixed.vh`, per-column
   `w_type` routing in `src/glm_matmul_q4k.v`, `desc_wtype` in `src/weight_loader_q4k.v`; `q6k_prim` /
   `q8_0_prim`, `glm_matmul_mixed` 32/32, `weight_loader_q4k_mixed` 192/192 incl. a 24-tile mixed
   sequence — all bit-exact to the same `tools/q4k_ref.py` golden. The chip CAN now consume a real
   UD-Q4_K_XL checkpoint's type mix; still not bit-verified vs the real GGUF bytes — step 3.)**
3. **Real-checkpoint validation** *(the standing OPEN #1 fidelity gate — `NEXT_STEPS_PLAN.md` D1)*. On a
   GPU / large-memory host: run the real 753 B GLM (llama.cpp / the real GGUF) on a prompt → golden
   next-token argmax; run our datapath (assembled-Q4_K golden from step 1, mixed-type from step 2) on the
   same weights + prompt; assert token match over a corpus. Any divergence = a per-Linear
   scale-orientation / bf16-tail / RoPE/KV/MoE plumbing bug to fix in the RTL contract. **No in-repo tool
   exists** — the prior FP8 `modal_validate.py` was on the deleted track and is **not** a Q4_K validator.

This chain is gating: it validates the *assembly* of hundreds of operators on *trained* weights — the one
thing the slice and the spec==greedy self-consistency cannot.

---

## Product phases (main)

### P1 — Full-scale + real-model correctness  *(gates everything)*
- **P1.1 The Q4_K fidelity chain (above). Blocking.**
  - P1.1a Assembled-Q4_K numeric golden (B9) — **DONE** (`make model-q4k` + `model-q4k-acthw`: 1155
    bit-exact vs `tools/glm_model_q4k_ref.py`).
  - P1.1b Mixed-type Q6_K/Q8_0/F16 consumer (B10) — **DONE** (`make mixedtype`, bit-exact vs `q4k_ref.py`).
  - P1.1c Real-checkpoint validation vs llama.cpp / the real GGUF (D1) — **OPEN**, needs a GPU/large host;
    its prerequisites P1.1a+P1.1b are now done.
- **P1.2 Scale the RTL/params to the full config.** **ELABORATED (done — structure only, not a sim):** the
  full 753 B UD-Q4_K_XL shape (DIM 6144 / L=78 / 256-expert top-8 / VOCAB 154880 / Q_LORA 2048 / KV_LORA
  512 **[PENDING safetensors]**) elaborates clean via `test/full_config_elab_wrap.v` (verilator, 0 errors —
  type/width only) and the whole-chip `glm_q4k_system_cdc` passes `make synth-glm` (yosys `hierarchy -check`
  + `check -assert`, exit 0). **NOT-YET:** full-config *functional* sim is intractable at this shape (the
  LM-head GEMV alone is ~2.4e8 K-beats/token), so full-scale correctness rides on the P1.1 fidelity chain
  at the slice + the elaboration study, not a full-config run. *(Prior-FP8 note: the O(NB²)→O(1) sequential
  block-scale dequant fold that first made the product KMAX buildable — one `fp32_mul_pipe` + one
  `fp32_add_pipe` reused over all blocks — was established and measured **bit-exact on the prior FP8 track**
  `glm_matmul_fp8` (tag `fp8-verified-baseline`); `glm_matmul_q4k` inherits the same sequential fp32-accumulate structure,
  and its Q4_K bit-exactness is the `make q4k` 160/160 result. The specific FP8 buildability counts are a
  prior-track measurement — no fabricated Q4_K equivalent.)*
- **P1.3 Close the batched-decode / per-position-KV correctness gaps for product.** **Largely closed on
  `main` (Q4_K modules), all as self-consistency, not a numeric golden:**
  - **Per-row position/extent/sequence** threaded model→decoder→mla (`PER_ROW_POS` / `PER_ROW_SLEN` /
    `PER_ROW_SEQ`), all safe-defaulting to 0 (byte-identical when off).
  - **KV pager storage side** carries `NSEQ` independent per-seq ring windows (per-seq counter/window/
    eviction + cold keying; NSEQ=1 byte-identical; directed multi-seq TB + BMC/k-induction).
  - **Multi-sequence batched attention (`PER_ROW_SEQ`) end-to-end through `glm_model_q4k`** — `mla_attn_q4k`
    replaces the shared-seq union-skip with a per-row-slot union tagged by `union_seq` and routes each
    fetch to its sequence's window; verified in both the DENSE and the **SPARSE** (S>TOPK_ATTN, the real
    long-context/DSA) regime. Full-model B=2 and B=4 batch different sequences in one forward, each row's
    argmax/logits **per-row self-consistent** vs a standalone PE_M=1 `glm_model_q4k` decoding that sequence
    alone, sharing the query-side weight fetch (fewer attn-weight beats than N separate decodes). *(These
    per-seq PE_M=1 references are themselves `glm_model_q4k` — DUT-vs-DUT self-consistency, not a ggml
    numeric golden; see P1.1a.)*
  - **Batched multi-sequence TOP** `glm_q4k_soc_ms` — `glm_model_q4k` at PE_M=B with a real `NSEQ`-window
    `kv_cache_pager` + `expert_cache_pf`, a host FSM that prefills B sequences into their windows, runs ONE
    batched forward, and commits B tokens; a **REAL per-layer KV store** (`kv_mem`, L×NSEQ windows ×
    resident positions) owned by the top replaces the TB stub; dense + sparse per-row self-consistent.
  - **`DSA_REAL_IDX=1` (query-dependent IndexShare) under multi-seq** via a per-sequence `kidx_buf` prefetch.
  - **Expert-union-skip MoE batching folded inline into `glm_decoder_block_q4k`** (PE_M>1 union scan; the
    standalone `batched_moe.v` module was removed from `main`) — B token rows share one fetch of each
    distinct union expert, accumulating in ascending-index order for byte-exact batch-invariance.
  - **Multi-step continuous-batching DECODE LOOP** (`glm_q4k_soc_ms` `N_STEPS>1`) — one `start` decodes N
    tokens/seq: the host FSM streams the B argmax, writes each decode token's latent into `kv_mem` at its
    growing position for every layer, feeds the argmax back, and advances pos/extent; each row's step-k
    token is **per-row self-consistent** vs a standalone PE_M=1 `glm_model_q4k` decoding that sequence alone
    N steps. N_STEPS=1 is byte-identical to the single-step top.
  - **Real draft chaining** (`spec_chain_top`) — the MTP head runs recurrently on its chain hidden-state
    `h_mtp` to mint K drafts, then a PE_M=K+1 batched-verify commits the accepted prefix; committed==greedy
    (`make spec-slow`). *(Updated 2026-07: **adaptive draft depth adopted and RTL-landed** —
    `spec_decode_seq` gains an `ADAPT` param (default 0; yosys sequential-equivalence PROVEN
    unchanged for existing consumers) + the new `spec_depth_adapt` saturating-streak policy module,
    K adaptive in [1..5], output-invariant by construction (spec==greedy for ANY depth schedule);
    gate: `make spec-adapt`.)*
  - **Remains (beyond RTL):** a **RESIDENT DENSE DRAFT model** for higher accept rate (K_eff 3–5 vs
    self-draft's ~1.7–2.2 — needs a trained small draft's weights; measured caveat: the *bandwidth*
    amortization of spec decode is A/U(K) ≈ 1.1–1.3× at K=4, A≈3 — the K+1 verify rows union their
    experts, measured U(4)=2.25–2.64 on the OLMoE proxy, superseded 2026-07 by the GLM-family
    measurement (GLM-4.5-Air: U(4)=2.60–2.71) — see [`H_MEASUREMENT.md`](H_MEASUREMENT.md)); and the
    standing P1.1 fidelity chain.

### P2 — Productize the RTL (robustness)
- P2.1 ECC on DDR5 + NVMe read path; error detection / correction / retry / recovery paths. *(Built:
  `ecc_mem_wrap` SECDED scrub-write-back + sticky `serr`/`derr`, `kv_ecc_ring` lane-SECDED + a
  `kv_cache_pager_ecc_fv` formal harness. Remains: DDR5/NVMe payload-byte ECC + committed-BMC
  re-parameterization for the widened word.)*
- P2.2 Full CDC sign-off across USB / memory / compute clock domains; reset/init/boot-load hardening.
  *(Built: `reset_sync` wired at both host/core boundaries of `glm_q4k_system_cdc`; `make cdc` structural
  crossing check; the PHY-closure loopback feeds ddr5_xbar's returned weight bytes back into the die,
  **proven bit-exact** for **all five die weight-input families**: attention (`make loopback`), the
  bandwidth-dominant routed-expert `fw` codes (`make loopback-fw`), and the `rw`/`lw`/`gn` families
  (`make loopback-rest`, ALL 10 — committed bit-exact PLUS a direct per-beat byte binding that asserts the
  die's presented input equals the expected byte every beat, needed because `rw`/`gn` are output-insensitive;
  all with a corruption-injection build that FAILS). Remains: the real vendor PHY IP transport.)*
- P2.3 Reliability (FPGA path): the built ECC (SECDED scrub) + KV-ring ECC + CDC/reset hardening carry
  over; MBIST maps to a BRAM self-test. ASIC-specific DFT (scan-chain insertion, boundary scan) is **not
  needed on the FPGA rungs (①②)** — the vendor JTAG/config handles device test — and is **deferred to the
  rung-③ ASIC endgame**, where it becomes required (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). *(Status
  2026-07, corrected: `mbist_ctrl` is the unit-verified **single-port March C- reference algorithm**, not a
  block to hand-instantiate in `glm_q4k_system*` — a verified sweep of the production memories found the two
  arrays that actually hold model state (`kv_cache_pager.ring`, `mla_attn_q4k.vstore_mem`) are concurrent-R/W
  2-port, which a single-port March engine cannot test as-is, so wiring one in would be theatre. The DFT
  insertion **contract** — per-macro BIST collar matching each macro's ports — is documented in
  [`P2_MEMORY_MAP.md`](P2_MEMORY_MAP.md) §4. The **dual-port BIST collar** now has a verified RTL reference —
  `src/mbist_ctrl_2p.v` (`[mbist_ctrl_2p] ALL 11 TESTS PASSED`: March C- + concurrent write/read coupling,
  fault kinds 0/1); the memory compiler still emits the real per-macro collar in the physical flow. Top
  `scan_enable` stitch is still an ASIC-endgame step.)*
- P2.4 Power: real ICG clock-gating cells, power domains, DVFS hooks, thermal budget. *(Status 2026-07,
  corrected: the production top **already** gates the entire compute die with an inline glitch-free ICG —
  `die_clk = clk & die_en_lat`, the `icg_cell` low-phase-latch pattern hand-coded at `glm_q4k_system.v:1307-1311`,
  driving `u_model`; it freezes the die on weight-stream stall / expert miss (byte-identical when off). So the
  biggest clock gate is instantiated; finer-grain ICGs are **synthesis-inferred** from enable-qualified register
  banks (the correct flow), not hand-placed. `clk_throttle` DVFS prescaler unit + BMC verified; `icg_cell`/
  `clk_en_ctrl`/`clk_gate_cluster` are the unit-verified reference/demonstrator blocks. See [`LOW_POWER.md`](LOW_POWER.md) §5.)*
- P2.5 Verification closure: functional + code coverage targets, constrained-random regression, gate-level
  (post-synth) sim, production-width controller formal + k-induction for unboundedness.

### P3 — Vendor IP + physical implementation
- P3.1 License + integrate the PHYs: the DDR controller+PHY (**DDR4 on rung ①, DDR5 multi-channel or HBM on
  rung ②** — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)), PCIe/NVMe host controller, USB-C device.
- P3.2 Target — **FPGA-card product for the near/mid term** (rungs ①②, the committed path — see
  [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); **ASIC is the rung-③ volume endgame, not out of scope.** A
  data-center-class FPGA + on-board DDR + NVMe (M.2/PCIe) runs the real model streamed; bitstream via the
  vendor flow. This is a **staged ladder**: rung ① (low-end FPGA, Kintex US+ KU3P-class + DDR4, ~5–8 tok/s
  [EST]) proves it cheap, rung ② (custom mid-FPGA board, DDR5/HBM, ~15–40 tok/s [EST]) is the funded
  interactive product. **The routed FPGA fit is MEASURED** — Vivado ML 2026.1 real synth + full
  place&route of `glm_q4k_system_cdc` on XCKU3P (compact config + ACT_HW=1): 142,320 LUT (87.5%)
  synth-stage (routed 141,298 LUT),
  ~100K FF, 421 DSP, 0 BRAM, hold met; routed Fmax 10.2 → 17.2 → 46.5 MHz through BIT-EXACT fmax
  repipeline rounds, every round re-proven on the 1155-test assembled golden (`rope_interleave_unit`
  10-stage; `glm_act` 20-stage + rmsnorm reduce/rsqrt; `glm_matmul_q4k` dequant+MAC 5-stage) —
  **campaign CLOSED at 4.6×**: the worst path is now ROUTE-dominated (a wide-bus route, ~59% wire
  delay at 87% utilization) — physical work, not arithmetic; 46.5 MHz sits in the bring-up demo's
  target band, 200 MHz-class is rung-②/③ work, and the stage decompositions carry to the ASIC
  unchanged — see [`fpga/`](../fpga/README.md) + `fpga/results/` and
  [`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md). (The old Gowin GW5AT-138 / Tang Mega /
  nextpnr scaffold was removed, superseded by the Vivado flow. ACT_HW is a new result-invariant resource
  knob — `glm_act` HW_LANES serialization.)
  - **Why FPGA first, ASIC at volume:** the workload is **memory-bandwidth-bound** — performance is set by
    memory bandwidth, which is set by the chip's IO pins + hard PHYs, which is set by budget. The FPGA rungs
    prove the RTL on real silicon and reach product-market fit *without* a multi-million NRE bet. An **ASIC
    is exactly what breaks the FPGA's IO/PHY ceiling**: it integrates **HBM stacks + many-channel
    controllers + near-memory compute** that no FPGA package offers, at **~TB/s** with **lower $/seat +
    lower power** once the NRE amortizes over manufacturing volume (rung ③). So the earlier "ASIC out of
    scope" — argued from *"compute-bound → ASIC's compute edge is wasted"* — is **superseded**: the real
    bottleneck is bandwidth, and ASIC is precisely how a bandwidth-bound product scales past the FPGA
    package. Sequence: **FPGA (①②) to prove + fund → ASIC (③) when volume justifies the NRE and demands
    lower $/seat + higher tok/s + lower power.** Not now (no volume, no capital); the endgame later, for
    cost-down + performance + power at volume.
- P3.3 Full-scale STA (SDC), power sign-off, signal/power integrity.

### P4 — System, software, manufacturing
- P4.1 PCB: multi-layer controlled-impedance board (FPGA + DDR5/HBM + NVMe via M.2/PCIe + USB-C), BOM,
  assembly. (This is the **rung ② custom product board**; the rung-① KU3P-class dev board is
  bring-up/reduced-demo only, not the shipping hardware — the prior Gowin Tang Mega 138K Pro target was
  removed, superseded by the Vivado/XCKU3P flow — [`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md).)
- P4.2 Software stack: USB-C host driver, tokenizer, the checkpoint→NVMe quant-layout pipeline
  (productionize `ckpt_pack_q4k.py`/`flash_layout.py`), inference runtime + continuous-batch scheduler.
- P4.3 Qualification: reliability (temp/voltage/aging), compliance (USB-C, EMI), yield/binning.

---

## What stays / what changes from the prior tracks

- **Keep (the core IP):** the Q4_K datapath (`glm_matmul_q4k` bit-exact to the ggml-Q4_K reference,
  `swiglu_expert_q4k`, `moe_router_q4k`, `mla_attn_q4k`), MLA/DSA/MoE, the memory-system controllers
  (`expert_cache_pf`, `kv_cache_pager`, `ddr5_xbar`, `flash_xbar`, `weight_loader_q4k`, `boot_loader`), the
  batching stack (PE_M batch-widening on all four Q4_K wrappers — swiglu/router/mla/mtp — + model +
  `spec_batched_top`/`spec_chain_top`, and the PE_M>1 grouped MoE **union-skip folded inline into
  `glm_decoder_block_q4k`**, byte-identical), the optimizations (BFP accumulator, parallel indexer,
  `weight_decomp` decompressor), the formal harnesses, and the **ggml-Q4_K reference kit** (`tools/q4k_ref.py`).
  These carry forward. **The memory system is byte-agnostic** (it moves addresses/slots/IDs, never weight
  bytes), so it carried over from the FP8 track by parameter/doc, not logic.
  *(One-time naming note: `flash_xbar` — like `FLASH_LAT`, `flash_req`, `flash_seq` — is a **committed RTL
  identifier**, kept as-is. It is the medium-agnostic storage-read fabric (address → weight bytes, with
  read-request issue + latency-hiding). In the product its NAND-specific backend is a labeled placeholder
  swapped for an **NVMe/PCIe host controller**; the crossbar abstraction and everything above it — compute
  die, `weight_loader_q4k`, `expert_cache_pf`, `kv_cache_pager` — are unchanged.)*
- **Change (the mindset):** every prototype "demonstrates the mechanism / honest gap / TB-driven" becomes a
  closed, full-scale, covered, signed-off product feature. Correctness is now *the real model on real
  weights, numerically checked* — not a slice, and not spec==greedy self-consistency.
- **Out of pure RTL (but on the product critical path):** PHY IP, physical implementation, software, PCB,
  manufacturing. These dominate product cost/time and are mostly vendor/EDA/board work, not algorithm design.
- **Prior FP8 track (tag `fp8-verified-baseline`).** The compute-side die-shrink / accumulator / fold-pipeline PPA wins,
  the sky130 place-and-route realizability, and the real-checkpoint bit-accuracy runs were established on
  the **prior FP8 track** and are **FP8-specific measurements** — they are **not** re-run for Q4_K and are
  **not** claims about the current Q4_K `main`. To inspect them: `git checkout fp8-verified-baseline` . No Q4_K equivalent is fabricated here.

## Immediate next step (product)

**Finish the P1.1 Q4_K fidelity chain** — the standing gate (1, 2, and 4 below are now closed; 3 remains):
1. **P1.1a Assembled-Q4_K numeric golden** (`NEXT_STEPS_PLAN.md` B9) — was the top OPEN item: move the
   assembled `glm_model_q4k` from spec==greedy self-consistency to an actual numeric match vs an assembled
   ggml-Q4_K reference (including the MLA `1/sqrt(d_head)` scale). This is the one thing the slice and the
   spec loops cannot do. **(DONE — `make model-q4k` / `model-q4k-acthw`: full forward vs
   `tools/glm_model_q4k_ref.py`, 1155 tests bit-exact.)**
2. **P1.1b Mixed-type Q6_K/Q8_0/F16 consumer** (B10) — so the chip can ingest a real UD-Q4_K_XL checkpoint
   as-is. **(DONE — `make mixedtype`, bit-exact vs `q4k_ref.py`.)**
3. **P1.1c Real-checkpoint validation** (D1, needs a GPU host) — the #1 fidelity gate, gated on 1+2
   (both now done).
4. **FPGA fit** (P3.2 / [`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md)) — the routed LUT/DSP/Fmax that sets
   device size / thermal / BOM / price. **(MEASURED / DONE — Vivado ML 2026.1 full P&R of
   `glm_q4k_system_cdc` on XCKU3P: 142,320 LUT (87.5%) synth-stage (routed 141,298 LUT), 421 DSP, 0 BRAM, routed Fmax 46.5 MHz after
   bit-exact repipeline rounds; the Fmax campaign is CLOSED at 4.6× — the worst path is now
   route-dominated, physical not arithmetic; see [`fpga/`](../fpga/README.md).)**

The per-position causal-KV / batched-decode gap (the prototype's PE_M shared-pos limitation) is **largely
closed** on `main` — per-row position/extent/sequence (`PER_ROW_SEQ`) and the multi-step
continuous-batching decode loop (`glm_q4k_soc_ms` `N_STEPS>1`) make batched decode position-accurate and
per-row self-consistent; the remaining P1.3 item is a resident dense DRAFT model (needs weights).
Everything else (P2–P4) sequences behind a green P1.
