# ULTRA_PERF — Ranked Ultra-High-Performance Opportunity Report

> **Companion:** [`ULTRA_PERF_MODULES.md`](ULTRA_PERF_MODULES.md) is the **bottom-up module-level**
> microarchitecture pass (critical-path/Fmax limiters, per-cycle throughput, internal serialization,
> fabric-width ceilings — every item cited to the RTL). This document is the **top-down system-lever**
> view. Read them together: the module pass makes several one-line levers here (stripe-across-N, exact
> prefetch, higher-K) concrete in the RTL, and surfaces the fetch-path depth + Fmax seams that gate them.

> **Current track: Q4_K local-inference.** This is a **perf-optimization study** for the current
> product — the **Q4_K** GLM-5.2 datapath on `main` (target weight store: the published
> `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`, **467 GB**). See [`Q4K_RETARGET.md`](Q4K_RETARGET.md) /
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md) and the honest verification ledger in the
> [`README`](../README.md). The RTL/tests referenced are the `*_q4k` units on `main`
> (`glm_model_q4k`, `glm_decoder_block_q4k`, `mla_attn_q4k`, `swiglu_expert_q4k`, `moe_router_q4k`,
> `mtp_head_q4k`, `glm_matmul_q4k`, `glm_q4k_soc(_ms)`, `glm_q4k_system(_cdc)`, `weight_loader_q4k`).
>
> **Prior FP8 track (tag `fp8-verified-baseline`).** Several *measured* perf multipliers
> quoted below (flash_xbar latency-hide, weight_decomp ratio, MTP throughput, idle-gate %, flash_layout
> balance) were measured on the **prior FP8 datapath**. The mechanisms are format-agnostic, but the
> numbers are **prior-FP8 measurements** — a **Q4_K re-run is PENDING** and no Q4_K equivalent is
> fabricated here. They are labelled **[prior-FP8]** wherever they appear.

### GLM-5.2 single-module accelerator (Q4_K die + DDR4/DDR5/HBM per rung + 1–N NVMe SSD — the memory tier is rung-dependent, [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md))

**Scope.** Opportunities *beyond* the already-built levers (flash_xbar latency-hide, weight_decomp,
MTP K=2, clk_en_ctrl idle-gate, flash_layout balance, fmax fixes, predictor-prefetch [measured no-op]).
The perf multipliers for those levers are **[prior-FP8]** (tag `fp8-verified-baseline`; Q4_K re-measure PENDING) — see the
banner above. Numbers marked **[EST]** are roofline model estimates, not measured silicon. **Numeric
honesty:** the Q4_K GEMM core is **bit-exact to the team's own ggml reference `tools/q4k_ref.py`** —
whose **dequant layer is now proven on real GGUF bytes** (376,586,240 weights — Q4_K/Q6_K/Q8_0,
two real published GGUFs — bitwise-equal to llama.cpp's own dequant — [`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)) — see
[`COVERAGE.md`](COVERAGE.md) / [`FORMAL.md`](FORMAL.md) and the README verification table. The
**real 467 GB UD-Q4_K_XL file itself / the llama.cpp whole runtime are NOT validated**: the big file
has never been downloaded, and the RTL uses **bf16 activations + fp32 accumulate** while llama.cpp
uses **Q8_K-quantized activations + integer dot** — a *different arithmetic contract*
(out-of-contract by design). Levers that change outputs are flagged **NOT bit-exact**.

> **Q4_K vs FP8 (why the retarget helps the roofline).** UD-Q4_K_XL is **467 GB** (~38% smaller than the
> 753 GB FP8 checkpoint; ~0.6 B/param avg vs ~1.0). The workload is **memory-bandwidth-bound**, so fewer
> bytes/token ⇒ more tok/s at the same bandwidth — Q4_K is ~**1.6× faster** than FP8 on the same rung
> **[EST]**. Every per-token byte figure below is the Q4_K figure (~0.6× the prior-FP8 count). **Honest
> gaps (ledger):** mixed-type support is **DONE** — Q6_K/Q8_0/F16 dequant primitives (`src/q4k_mixed.vh`),
> per-column `w_type` routing in `glm_matmul_q4k`, and `desc_wtype` in `weight_loader_q4k` (gates:
> `make mixedtype` — q6k_prim, q8_0_prim, glm_matmul_mixed 32/32, weight_loader_q4k_mixed 192/192 incl. a
> 24-tile mixed sequence; bit-exact to the same `tools/q4k_ref.py` reimpl golden), so the chip **can** now
> consume a real UD-Q4_K_XL checkpoint's dynamic type mix (sensitive tensors at Q6_K/Q8_0/F16). The
> Q4_K/Q6_K/Q8_0 dequant layer is **bit-verified against real GGUF bytes**
> ([`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)); the 467 GB GLM file itself has not been consumed
> end-to-end.

> **Naming note (storage backend).** The committed RTL identifiers — `flash_xbar`, `FLASH_LAT`,
> `flash_req`, `flash_seq`, `flash_layout`, `flash_is_expert`, `flash_expert_id`, … — are kept
> **as-is** and are *not* renamed. `flash_xbar` is the medium-agnostic **storage-read fabric**
> (address → weight bytes, with latency-hiding) that in the product sits in front of the
> **NVMe/PCIe host-controller backend**: the NAND-specific backend is swapped for NVMe, while the
> crossbar's read-request/latency abstraction and the compute die / `weight_loader_q4k` /
> `expert_cache_pf` / `kv_cache_pager` are **unchanged**. **Honest caveat (ledger):** `ddr5_xbar` /
> `flash_xbar` as committed are **single-lane** (1 beat/cycle, ~32 GB/s @1 GHz) and in the integrated top
> the fabric is **observation-only** (the die pulls weights combinationally from a TB stub). The
> ~100 GB/s+ figures below are **rung-dependent bandwidth targets** that require added lanes/channels +
> real PHYs — **out of scope for the current RTL**, set by the silicon you buy
> ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).

> **Product identity — a LOCAL, single-user box that works fully offline / air-gapped (read this
> first).** The headline is **frontier AI with the ethernet unplugged**: this accelerator is a
> **personal appliance — one box, one user, running the full 753 B GLM-5.2 model locally (Q4_K weights,
> 467 GB), no internet and no cloud, ever**. Lead with the *capability* that unlocks — run a frontier
> model in the classified / regulated / disconnected places you're locked out of today, and own it
> outright — not the defense. **Nothing leaves because there is no path out**; non-egress is the *proof*,
> not the pitch, and the audit is literally *"does it still work with the ethernet cable unplugged?"*
> (yes). That bar is one **no cloud can clear — including "secured cloud" (in-VPC / zero-retention / TEE
> enclaves), which all still need a connection**. Honest caveats: the **467 GB** of weights are
> provisioned **once** (itself doable offline) and model updates are **physical** re-provisioning; and
> offline *alone* is table-stakes for any local box — the moat is the **combination of offline + full
> frontier (753 B) + appliance/seat price** (see [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md)). The
> performance number that matters is therefore **single-user interactive throughput**, and it is
> **rung-dependent** — set by the memory bandwidth the silicon can feed, which is set by the rung you
> build ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)): **~5–8 tok/s [EST] on the near-term prove-it FPGA
> (rung ①), ~15–40 tok/s [EST] on the funded custom board (rung ②)** with the faithful levers stacked
> *(measured-proxy design-point update in §4 — [`H_MEASUREMENT.md`](H_MEASUREMENT.md))*. Any
> **aggregate-serving / datacenter-batch** figures below (B≈256, ~50 tok/s *aggregate*, **per-user
> ~0.14 tok/s**) describe a **DIFFERENT, non-target deployment** kept here only as analysis of what the
> same silicon *could* do batched — that per-user latency does **not** describe the box you plug in. When
> in doubt, the single-user numbers are the product.

**The one equation.** The workload is NVMe/PCIe-bandwidth-bound:
`tok/s ≈ NVMe_BW / [(1−h)·footprint] · K`. **`NVMe_BW`/`DDR_BW` are themselves set by the rung's IO pins +
PHY** ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)), so the absolute tok/s scales with the hardware rung; the
levers below move the *other* terms. **Measured correction ([`H_MEASUREMENT.md`](H_MEASUREMENT.md),
OLMoE proxy):** the spec multiplier `K` must be read as **A/U(K)** — the K-position expert *union* grows
with K (U(2)=1.51–1.65, U(4)=2.25–2.64), so spec-chain amortization is ~**1.1–1.3× at K=4 (A≈3)**, not
×K. *(Updated 2026-07: U(K) is now **GLM-family measured** — GLM-4.5-Air MoE-gate trace on Modal H100:
U(2)=1.60–1.64, U(4)=2.60–2.71, U(6)=3.46–3.62, U(8)=4.19–4.41, EOR 0.36–0.39 — superseding the OLMoE
first-pass values above; [`H_MEASUREMENT.md`](H_MEASUREMENT.md) 2nd measurement.)* Only three classes of idea move this wall:
**(i) move fewer expert bytes** (sparsity / dedup / stronger decomp), **(ii) compute at the data** (near-storage),
**(iii) raise speculative K** (better drafts + batched verify). Everything else is incremental — die-side
fmax/area work does **not** move the wall (the die is already ~75% idle behind the NVMe/storage read).

---

## 1. Headline table — TOP opportunities (ranked by impact × feasibility)

> **Status legend.** A **✅ RTL-present** mechanism means the Q4_K module carries the structure. The
> **assembled end-to-end numeric golden for `glm_model_q4k` is now DONE**: `make model-q4k` runs the full
> forward (embed → Lx(MLA+DSA+MoE) → final norm → LM head → argmax) against the numpy reference
> `tools/glm_model_q4k_ref.py` — **ALL 1155 TESTS bit-exact** (logits+argmax+h_state), plus
> `make model-q4k-acthw` (the same golden through the ACT_HW=1 serialized-activation datapath, also 1155).
> Caveat that stays: the golden is our **own numpy reimpl, NOT llama.cpp/GGUF**. The batched
> `PE_M`/union/paged-KV paths were verified **bit-exact on the prior FP8 track (tag `fp8-verified-baseline`)**, not on
> Q4_K. Where a row says a mechanism was "verified bit-exact", that is a **[prior-FP8]** result unless
> stated otherwise. Also checked on Q4_K end-to-end: `spec_decode_top` **18/18 spec==greedy**
> (DUT-vs-DUT self-consistency, the "greedy golden" is itself a `glm_model_q4k` — a lossless-speculation
> safety property, **not** a numeric golden), + `spec_batched/chain` via `make spec-slow`.

| # | Opportunity | Mechanism (1-line) | Quantified impact **[EST]** | Where | Effort | Ceiling? |
|---|-------------|--------------------|------------------------------|-------|--------|----------|
| 1 | **Expert-grouped layer-synchronous batched MoE** — union-fetch ✅ **RTL-present in `glm_decoder_block_q4k`** | Fetch the per-layer expert *union* once from the NVMe SSD, reuse across B token-rows (the PE_M>1 grouped MoE scans the expert axis + skips non-union experts) | Aggregate **6–8×**: ~36–50 tok/s @B≈256 vs ~6 single-user; multi-seq batched top (`glm_q4k_soc_ms`, `PER_ROW_SEQ`) + paged KV. **Union-skip verified bit-exact [prior-FP8]; Q4_K B=1 assembled golden DONE (`make model-q4k`, 1155 bit-exact); batched (PE_M>1) Q4_K golden = NOT-YET** | rtl-here | high | ✅ |
| 2 | **PE_M batch-widening of the Q4_K wrappers** — ✅ **RTL-present (4/4)** | **All four** wrappers (`swiglu_expert_q4k`/`moe_router_q4k`/`mla_attn_q4k`/`mtp_head_q4k`) carry a `PE_M` param + per-row buffers → one weight fetch serves B rows. Weight-share property (*"PE_M=B issues the same weight beats as PE_M=1 → B rows, 1 fetch stream"*) verified bit-exact **[prior-FP8]** | Silicon enabler for #1; 0 extra dequant muls, 0 extra weight BW. **Q4_K unit TBs: functional/invariant, not a PE_M golden** | **rtl-here** | RTL done | (enabler) |
| 3 | **Stronger weight decompressor (context-modeled)** — ✅ **RTL-present (`weight_decomp2`)** | Spend idle die on an order-1 context-modeled Huffman decoder in the NVMe→DDR5 refill path | ~1.3–1.5× fewer streamed bytes **[prior-FP8, on the FP8 byte stream]** → **Q4_K applicability PENDING** (Q4_K is *already* 4-bit — an extra entropy coder gains less; re-measure needed) | **rtl-here** | RTL done | ✅? |
| 4 | **Resident dense draft model → high-K spec decode** | ~1–3B DDR5-resident draft proposes K=4–8; target verifies in one pass | K_eff 1.7→**3–5** → ~2–3× single-user, **bit-exact** | rtl-here* | high | ✅ |
| 5 | **Batched single-pass verification** — ✅ **RTL-present (`spec_batched_top`/`spec_chain_top`)** | Forward {base + D draft} positions *together* as a PE_M=K+1 batch in ONE weight-load (spec NVMe ÷(K+1)) | The gate for spec NVMe gain, **built**; **spec==greedy holds on Q4_K** (`spec_decode_top` 18/18 DUT-vs-DUT; `make spec-slow`). Remaining lift is draft α (#4) | **rtl-here** | RTL done | ✅ |
| 6 | **Contextual activation sparsity in SwiGLU** | Low-rank predictor → fetch only active W_up cols / W_down rows | ~1.5–3× fewer routed bytes (~14→~5–9 GB); **NOT bit-exact** | **rtl-here** | high | ✅ |
| 7 | **Dynamic top-k expert pruning (k_eff<8)** | Threshold-mask tiny renormalized gates in the router FSM | ~1.3–1.6× fewer routed bytes; cheapest big lever; **NOT bit-exact** | **rtl-here** | low | ✅ |
| 8 | **Exact router-driven prefetch + K-token union** | Run cheap router GEMV for K spec tokens → exact union prefetch | Demand-stall 81%→~99% + ~1.5–2× byte-dedup; **bit-exact** | **rtl-here** | med | ✅ |
| 9 | **Unmask + compress the hot-weight DDR5 read** | Point weight_decomp at hot path; amortize K×; +DDR5 channels | The *next* wall: the ~11 GB hot-set touch becomes DDR-bound (rung-② ~15–40 band; canonical: R3_APPLIANCE_SPEC §2) | **rtl-here** | med | ✅ |
| 10 | **IndexShare (DSA index once / 4 layers)** | Cache index-list; skip dsa_indexer on 3 of 4 layers (model-faithful) | At 1M ctx: index-read cut ~4× → keeps long-ctx NVMe-bound | **rtl-here** | med | ✅ |
| 11 | **Parallel/pipelined DSA indexer** | Replace in-order 1-MAC dot with 128-lane reduction tree | At 1M ctx: ~0.05→~6 tok/s (kills O(S)·7-cyc drain); **bit-exact** | **rtl-here** | high | ✅ |
| 12 | **MLA weight absorption (attend in 512-dim latent)** | Fold W_uk into q, W_uv into W_o; drop per-key up-projection | Removes ~3.2e5 per-key GEMMs/tok; **bit-exact** (matmul reassoc) | **rtl-here** | high | ✅ |
| 13 | **Near-storage / computational-storage expert compute** — the **rung-③ endgame** *(2026-07 pivot: now the hybrid/streaming-SKU + >512 GB endgame — the primary rung-③ SKU is full-residency LPDDR5X, see §2a / [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md))* | Move the Q4_K dequant+MACs into the NVMe drive (in-SSD / near-NAND); stream 12 KB act down, 12 KB result up | ~1000× fewer bus bytes/expert → **10×+** ceiling (breaks the FPGA IO/PHY wall). **No J/tok win** | **rung ③** | high | ✅ |
| 14 | **Batch × MTP multiply** | Run all B streams' MTP drafts in the same grouped pass | ~1.7× *on top of* batch → ~60–85 tok/s aggregate @B≈256 | **rtl-here** | low | (compose) |
| 15 | **Paged multi-sequence KV cache** — ✅ **RTL-present (`kv_cache_pager` NSEQ windows + `glm_q4k_soc_ms` `kv_mem`)** | Per-seq ring windows + a real per-(layer,seq) KV store; `PER_ROW_SEQ` attention lets each row attend its OWN sequence | Enabler for B>1 *distinct users* in one forward. **Verified per-row bit-exact [prior-FP8] at B=2/4; Q4_K multi-seq assembled golden = NOT-YET (B=1 `glm_model_q4k` golden DONE)** | **rtl-here** | RTL done | (enabler) |

\* #4 RTL substrate (g_kn verifier) exists; the draft *weights* are a training task.

---

## 2. CEILING-CHANGERS vs INCREMENTAL/CONDITIONAL

### 2a. CEILING-CHANGERS — flip NVMe/storage-bound → compute-bound, or 10×-class

These touch `(1−h)·footprint`, `K`, or the bus itself.

- **Near-storage compute (#13)** — the single biggest lever *and the **rung-③ endgame*** ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)):
  ~1000× fewer bus bytes/expert lifts the ceiling to the SSD's internal NAND-sense limit (**10×+ tok/s [EST]**).
  It needs custom CSD/PIM / near-memory silicon — an ASIC/SoC with HBM + near-memory compute is **exactly what
  breaks the FPGA's IO/PHY bandwidth ceiling**, so it is **not "out of scope forever" but the volume endgame
  (rung ③)**: not now (no volume, no capital), real later for cost-down + performance + power at manufacturing
  volume. It does **not** cut J/token (the sense energy is the cost). The compute core is reusable Q4_K RTL.
  *(Updated 2026-07: the **primary rung-③ design point has pivoted to full residency** — 512 GB LPDDR5X
  holds the whole ~467 GB checkpoint, [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) — so near-storage
  compute now applies to the hybrid/streaming upside SKU and >512 GB checkpoints, not the primary SKU.)*
- **Aggregate batching (#1/#2/#15)** — ⚠️ **NOT the product; a secondary "what-if" for a datacenter
  deployment of the same silicon.** Reframes batching from the naive "~1.5×" (a B=32, LRU-hit-rate artifact)
  to a **6–8× aggregate** lever via expert-union reuse — the **union-fetch mechanism is present in
  `glm_decoder_block_q4k`** (PE_M>1 fetches only the selected-expert union; the FP8 `batched_moe.v` union-skip
  logic was **folded inline** into the decoder, so there is no separate module on `main`), and a
  multi-sequence batched top (`glm_q4k_soc_ms`, `PER_ROW_SEQ`) decodes B distinct users in one forward with
  paged per-seq KV. **The union-skip + per-row bit-exactness were verified on the prior FP8 track (branch
  `fp8`); on Q4_K the assembled *batched* (PE_M>1) path is NOT-YET checked against any golden** (the B=1
  assembled `glm_model_q4k` forward IS now golden-checked — `make model-q4k`, 1155 bit-exact vs the numpy
  ref). New knee at **B≈256** (all 256 experts active: `E[distinct]=256·(1−0.96875^B)`), new ceiling =
  the compute roofline **~50 tok/s aggregate @100 GB/s NVMe [EST]**, reached near B≈355. **But per-user
  latency floors at ~0.14 tok/s at that batch — so THIS regime is offline/throughput serving, NOT the local
  personal box.** The personal box runs at B=1 (single-user row above); this bullet just documents that the
  RTL *also supports* batched serving if a datacenter product is ever wanted.
- **Stronger weight decomp (#3)** — the idea that uses the 75%-idle die to cut the *actual* wall. RTL present
  (`weight_decomp2`, order-1 context-modeled Huffman). Its **~1.3–1.5× ratio is a [prior-FP8] measurement on
  the FP8 byte stream** — and **Q4_K is already 4-bit**, so an extra lossless entropy coder is expected to gain
  *less* on Q4_K codes; **the Q4_K re-measure is PENDING** and no Q4_K ratio is claimed. When it does help it is
  a direct multiplier on **both** single-user tok/s **and** J/token (the NVMe/PCIe storage read ≈ 80% of
  per-token energy). **rtl-here**, faithful, stacks multiplicatively.
- **Higher speculative K (#4 + #5 + #8)** — the faithful single-user lever. #5 batched-verify is **RTL-present**
  (`spec_batched_top`/`spec_chain_top`: PE_M=K+1 verify in one weight-load → NVMe ÷(K+1)); **spec==greedy holds
  on Q4_K** as DUT-vs-DUT self-consistency (`spec_decode_top` 18/18; `spec_batched/chain` via `make spec-slow`).
  The self-draft MTP chain reaches K_eff ~1.7–2.2; #4 a resident dense draft (needs trained weights) would
  raise α (K_eff → 3–5); #8 exact-router-union dedups the K-token expert set. **All bit-exact** (target
  verifies every token). Honest cap: the MoE union penalty (#22 below) keeps K_eff_nvme well below the
  dense-model ×K — **now measured (proxy):** U(2)=1.51–1.65, U(4)=2.25–2.64 → realized NVMe amortization
  **A/U(K) ≈ 1.1–1.3× at K=4, A≈3** ([`H_MEASUREMENT.md`](H_MEASUREMENT.md) /
  [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md)). *(2026-07: superseded by the GLM-4.5-Air
  measured values — U(2)=1.60–1.64, U(4)=2.60–2.71, U(8)=4.19–4.41; OLMoE stays as the first-pass
  history.)*
- **Activation sparsity (#6) + dynamic top-k (#7)** — shrink `footprint` directly (~1.5–3× and ~1.3–1.6×).
  Both **NOT bit-exact** (quality knobs; must be validated against the accuracy contract). #7 is the cheapest
  big lever (a comparator+mask in the router).
- **Hot-weight DDR5 second wall (#9)** — not today's wall, but bites once the first 3–4 NVMe/storage levers land;
  the ~11 GB Q4_K hot-set touch (attention/dense/shared, DDR-resident; canonical: R3_APPLIANCE_SPEC §2) then becomes DDR-bound, set by DDR BW
  (rung-② ~15–40 band). Decomp+K-amortize+channels keep the post-NVMe-fix regime from re-stalling.
- **Long-context faithful set (#10 IndexShare, #11 parallel indexer, #12 MLA absorption)** — at 1M ctx the
  O(S) indexer and per-key up-projection, *not* NVMe/storage, become the wall (in-order indexer ≈ 0.05 tok/s). These
  restore the NVMe-bound ~6 tok/s at extreme context. All **model-faithful / bit-exact** (matmul reassoc).
  *(Update: the 1/√(qk_head_dim) softmax scale is now **applied** in `mla_attn_q4k` — the earlier
  missing-scale caveat is resolved.)*

### 2b. INCREMENTAL / CONDITIONAL — smaller or regime-specific

- **Batch × MTP (#14)** — free ~1.7× *compose* on top of batching; multiplicative, not a new ceiling.
- **Enablers (#2 PE_M widen, #15 paged KV, continuous-batch scheduler)** — required to *realize* #1, but
  add no ceiling by themselves. Scheduler defends the peak (a half-full batch pays full union for half the
  tokens → half aggregate).
- **Attention hot-path batch reuse + DDR5 realloc** — defensive; prevents attention/hot-weight becoming the
  secondary bottleneck at high B; frees DDR5 for KV.
- **Pipelined KV gather / latent-KV / widen attention engines** — long-ctx *latency/footprint*, ~1% of
  bytes; not ceiling movers.
- **PE-array scaling (wider PE_N), fmax tail fixes, output-stationary SRAM, pipeline MLA softmax** — **~0 on
  single-user decode** (compute already hidden 4× under the NVMe/storage read). Value is **prefill/TTFT** (linear in array
  size) and energy/voltage headroom. *(Now grounded by the MEASURED KU3P fit: Vivado ML 2026.1, 142,320 LUT
  / 87.5 %, routed Fmax **46.5 MHz** after the closed 4.6× repipeline campaign — worst path route-dominated,
  i.e. physical, not arithmetic. Clock↔area trade: compute-side stream consumption = dequant lanes × clock —
  at 46.5 MHz, 7 GB/s needs ~300 lanes and 100 GB/s ~4,300 (infeasible on KU3P); ~1,000 at 200 MHz-class;
  ~200 at ASIC 1 GHz+. A higher clock buys a smaller/cheaper die, **not** more tok/s.)*
- **Pipeline draft into NVMe/storage shadow / reuse accepted KV / deeper layer-pipeline** — latency-only on the
  already-idle die; **0% on the bandwidth ceiling**.

### 2c. HONEST NEGATIVES (do not re-propose)

- **Multiple compute dies sharing one DDR5+NVMe** — **0** within the module (shared bus). Linear only as
  N *separate* modules (N× cost). The bottleneck is the bus, not die count.
- **Deeper prefetch / layer-pipeline** — already bandwidth-bound (81% demand-stall removed); residual is the
  irreducible DDR5 read floor (~5% utilization, not throughput).
- **Predictor-prefetch** — **measured no-op** (ledger-confirmed); hit-rate is entropy-capped by fine-grained
  routing (the routed experts change every token). Not a Q4_K-vs-FP8 artifact — it is the routing entropy.
- **Hierarchical block-max pruned indexing** — could cut indexer 5–20× at 1M but is **off the faithful path**
  (changes outputs); escape hatch only.

---

## 3. "What to build" — the single biggest RTL-here lever

### Build A — Expert-grouped batched MoE (#1+#2+#15) — the aggregate ceiling
The only thing that needs to be *invented*; the math die is ready. **What's RTL-present today vs what needs
a Q4_K golden is stated per step.**

1. **PE_M batch-widen the wrappers (#2, the keystone) — RTL-present (4/4).** `glm_matmul_q4k` supports
   PE_M>1 (8·PE_M a_shift port, per-row accumulator banks, PE_M·PE_N dequant walk; the scarce 24×24 dequant
   muls stay pinned at NB **regardless of PE_M**). **All four wrappers — `swiglu_expert_q4k` / `moe_router_q4k`
   / `mla_attn_q4k` / `mtp_head_q4k` — carry a `PE_M` parameter** with `[0:PE_M-1][…]` per-row buffers,
   per-row a_shift/hsh, and the **same** w_col streamed to all rows. The *"B rows == 1 fetch stream"*
   weight-share property was verified **bit-exact on the prior FP8 track [prior-FP8]**; on Q4_K the unit TBs
   (`swiglu_expert_q4k` 240 functional/self-labeled, `moe_router_q4k` 40 invariant, `glm_matmul_q4k` 160
   bit-exact-vs-ggml) do **not** include a PE_M weight-share golden — that verification is **NOT-YET on Q4_K**.
   `mtp_head_q4k` (the composite: 3 RMSNorms + combine-proj + a full `glm_decoder_block_q4k` + LM head) threads
   PE_M through per-row and enables **Batch × MTP (#14)**.
2. **Grouped dispatcher.** Per MoE layer: (a) route all B tokens, histogram top-8 picks into a per-expert
   token-list (reuse scatter_gather.v + topk_select); (b) for each *distinct* active expert, gather its rows
   into a PE_M tile, fetch the expert from NVMe/DDR5 **once**, run the grouped GEMM, scatter back;
   (c) advance all B tokens to L+1 in lockstep. The union is the **only** NVMe/storage traffic, shared across
   all B rows; it caches in the rung-② board's DDR5 (DDR size is rung-dependent —
   [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md), not a fixed spec).
   **(b) is RTL-present.** `glm_decoder_block_q4k` (PE_M>1) fetches **only** the union of experts any row
   selected — a `T_ESCAN` scan over the expert axis with a combinational `any_has` membership test skips every
   non-union expert (the FP8 `batched_moe.v` union-skip logic was **folded inline** here). This was verified
   **BYTE-IDENTICAL on the prior FP8 track [prior-FP8]** (union_tb / `_pem` / decoder TB, and `make bcov`
   B∈{1,2,3,5,8} — *all FP8-track, removed from `main`, see tag `fp8-verified-baseline`*). **On Q4_K the assembled decoder
   path is now covered at B=1 by the `glm_model_q4k` end-to-end golden** (`make model-q4k`, 1155 bit-exact
   vs `tools/glm_model_q4k_ref.py`), **but the PE_M>1 union fetch has no Q4_K golden (NOT-YET)** — its
   Q4_K numeric correctness at batch is unproven. Effect *when realized*: up to **~32×** fewer NVMe/storage
   expert fetches at small batch (real 256-expert config); ~no benefit at B≈256 (union≈all). What still needs
   inventing is scaling the dispatcher/scheduler (a)/(c) to the datacenter B≈256 regime.
3. **Paged KV (#15) — substrate RTL-present.** `kv_cache_pager` carries `NSEQ` independent ring windows
   (per-seq counter/window/eviction) and `glm_q4k_soc_ms` OWNS a real per-(layer,seq) KV store (`kv_mem`) the
   multi-seq model reads combinationally; `PER_ROW_SEQ`/`kc_seq` route each row to its own sequence's window.
   MLA latent KV ~1 KB/tok/layer → B=256 at few-K ctx fits the DDR5 freed by the shrunken (one-layer-union)
   expert cache. Per-row bit-exactness at B=2/4 was verified **[prior-FP8]**; the Q4_K *multi-seq* assembled golden is
   **NOT-YET** (the B=1 `glm_model_q4k` golden is DONE — `make model-q4k`). Remaining: scale the window count to datacenter B and a vLLM-style shared-pool block table.

**Payoff [EST]:** per-token routed footprint `75·256·(1−0.96875^B)·(~23 MB)/B` (75 MoE layers; ~23 MB per
expert-per-layer at Q4_K = ~0.6× the prior-FP8 ~37 MB) → at B=256 ≈ **~1.7 GB/tok** (~8× vs the ~14 GB
single-user routed footprint) → ~36 tok/s aggregate @100 GB/s NVMe; ~50 tok/s compute-roofline cap near
B≈355; ×1.7 more with MTP (#14). *(All [EST]; the aggregate regime is non-target.)*

### Build B — Faithful high-K speculation (#5+#4+#8) — the single-user ceiling
1. **Batched verify (#5) — RTL-present.** `spec_batched_top` / `spec_chain_top` forward {base + D draft}
   positions as a PE_M=K+1 batch through ONE `glm_model_q4k` weight-load; per-layer expert streaming serves all
   rows; `spec_decode_seq` commits the accepted prefix (longest-accepted = greedy = bit-exact). **spec==greedy
   is verified on Q4_K** (`spec_decode_top` 18/18 in `make unittests`; `spec_batched/chain` via `make
   spec-slow`) — a DUT-vs-DUT self-consistency safety property (the "greedy golden" is itself a `glm_model_q4k`
   sharing the weight ROMs), **not** a numeric golden.
2. **Resident dense draft (#4) — still PENDING (needs trained weights).** The self-draft path exists today
   (`spec_chain_top` chains the MTP head, K_eff ~1.7–2.2); a ~1–3B DDR5-resident draft (attention + shared
   expert + heads only → **zero NVMe**) proposing K=4–8, or Medusa heads (no chain decay), would raise α to
   K_eff 3–5. Output stays bit-exact (target verifies).
3. **Exact union prefetch (#8).** Run the cheap router GEMV for all K spec tokens during draft compute →
   exact top-8 union → prefetch before needed. Hides NVMe/storage latency (81%→~99%) and dedups the K-token set.
4. **Geometry rule (#22):** on a NVMe/storage-bound MoE, prefer **deep-narrow chains** over wide trees — each branch
   drags in (1−r) divergent experts you may reject; a naive 4-wide tree at r=0.35 → K_eff_nvme≈0.85
   (a **regression**). Depth keeps K_eff_nvme>1.

*(Updated 2026-07 — **adaptive spec-chain ADOPTED + RTL-landed** (commit 6c5332f): the GLM-U K-sweep
shows tok/s at r=0.9 **plateaus at K=4–5** (~93–95; K>5 adds nothing) and at r=0.8 the optimum is
K=2–3 (~78) → adaptive range **K∈[1..5]**, so a fixed K=6–8 buys nothing. RTL: `spec_decode_seq`
gains an `ADAPT` param (default 0 — yosys sequential-equivalence PROVEN unchanged for existing
consumers) + `k_cur` port + `pass_*` taps; new `src/spec_depth_adapt.v` saturating-streak policy;
**output-invariant by construction** (spec==greedy for ANY depth schedule). Gates: `spec_depth_adapt`
31,522; `spec_decode_seq`(K>1) 3,702 (K now 1/2/3/4/6/8); K=1 exact 621; `spec_chain_top` 4/4 incl.
a new DRAFT_K=4 engine; `spec_batched_top` 8/8; `spec_decode_top` 18/18; new `make spec-adapt`
Makefile gate. The accept rate r has since been **measured** (job B vLLM MTP sweep, GLM-4.5-Air:
r₁=0.87 with steep per-position decay 0.87/0.60/0.32/0.13/0.04, A_eff plateau ~2.9 → the
memory-bound optimum is **K=1–2** and the adaptive controller settles there on its own —
[`H_MEASUREMENT.md`](H_MEASUREMENT.md) 3rd measurement; K≤5 stays as headroom for GLM-5.2's
deeper-trained MTP).)*

---

## 4. Honest roofline — the three regimes

The **product is the first row** (local, single-user box). The other two are *analyses of the same silicon*
under deployments this product does not target — kept for honesty, not as the roadmap. All figures are
**[EST]** roofline projections (per-token Q4_K footprint ~25 GB = 40B active × ~0.6 B/param, of which the
**~14 GB routed-expert bytes are the NVMe wall** and the **~11 GB hot-set touch caches in DDR** —
canonical byte constants: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2).

| Regime | Deployment | Bound by | tok/s (today → with levers) **[EST]** | Levers that apply |
|--------|---------|----------|----------------------------------------|-------------------|
| **Single-user ← THE PRODUCT** | **local personal box, interactive** | storage/DDR BW (~14 GB routed/tok) | **rung ① ~5–8** (~100 GB/s striped NVMe) → **rung ② ~15–40** (~300–600 GB/s DDR5/HBM working set), [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) | decomp #3, sparsity #6, top-k #7, draft-K #4+5+8, hot-weight #9 |
| Aggregate-serving *(not the product)* | datacenter batch, offline | NVMe/storage union then compute roofline | ~6 (B=1) → **~36–50** (×1.7 MTP → 60–85) *aggregate; per-user ~0.14* | batched MoE #1, PE_M #2, paged KV #15, B≈256 knee, scheduler |
| Compute-bound *(hypothetical)* | full-resident HBM (**rung ③** class) | compute roofline (~80 GFLOP/tok ÷ ~4 TFLOP/s ≈ 20 ms) | **~40–50** ceiling | only if experts free (near-storage #13 or HBM — **rung ③**); array-scale for prefill |

**On the ~100 GB/s storage figure [EST].** This is an *aggregate* NVMe/PCIe target, **not a single drive**,
and — per the naming-note caveat — **beyond the current single-lane fabric** (`flash_xbar`/`ddr5_xbar` as
committed do ~32 GB/s @1 GHz; more lanes/channels + real PHYs are rung-② hardware, not RTL). One PCIe Gen4
×4 NVMe delivers ~7 GB/s (Gen3 ×4 ~3.5, Gen5 ×4 ~14), so ~100 GB/s implies a **multi-drive / many-lane array**
(order ~14× Gen4 or ~7× Gen5 M.2 drives striped, or an equivalent many-lane PCIe fan-out) on a **custom
board**, and it scales *with lanes/drives* exactly as the old NAND story scaled with channels. Per the ledger
roofline: 1–2 NVMe (~7–14 GB/s) → **~0.5–1 tok/s**; 4 NVMe (~28) → **~2**; striped ~14 drives (~100 GB/s) →
**~5–8 (rung ①)**; DDR5/HBM feeding the working set (~400 GB/s–1 TB/s) → **~15–40 (rung ②)**; an HBM3 ceiling
(~3 TB/s) → ~120 but **467 GB won't fit HBM (≤192 GB) so aspirational**. Treat 100 GB/s as an upper-bound
target, not a single-M.2 spec; the tok/s scales roughly linearly with the storage BW actually deployed (a
single Gen5 x4 ~14 GB/s ≈ ~1 tok/s at ~14 GB routed/tok, before decomp/sparsity/spec-K multipliers).

**Update — measured-proxy design-point menu ([`H_MEASUREMENT.md`](H_MEASUREMENT.md); h/U proxy-measured
on an OLMoE trace — U(K) since GLM-Air-measured, see the pivot note below; all tok/s still [EST]):** 1–2 NVMe, no multipliers → **~0.5–1 tok/s**;
90 GB DRAM cached (~20 % of the expert pool, bandwidth-h 0.36–0.60) + 100 GB/s → **13–24**; 90 GB +
200 GB/s (ONFI 64ch) → **25–47**; 225 GB (~50 % pool, h 0.72–0.88) + 200 GB/s → **54–127** (the
"100 tok/s" design point). LRU collapses to ~0 below a 10 % cache. The roofline formula stands, but read
the spec multiplier `K` as **A/U(K)** (measured U above). See also
[`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md).

*(Updated 2026-07 — design-point pivot, [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)* The **primary
rung-③ design point is now FULL RESIDENCY**: 512 GB LPDDR5X (16×32 GB, 1024-bit on-package, ~1.1 TB/s)
holds the whole ~467 GB checkpoint — **h=1 by construction**, cold storage is one commodity M.2 NVMe
(boot-load ~70 s), and the ONFI-64ch streaming tier is deleted from the primary SKU (pads stay on-die
for the hybrid upside SKU). The residency-box design point is **≈80 tok/s [measured-inputs EST]**
(U(K) **and** the accept rate r both GLM-family measured — job B's vLLM MTP sweep put the
memory-bound optimum at K=1–2; ~95 if GLM-5.2's deeper MTP hits its published accept depth —
[`H_MEASUREMENT.md`](H_MEASUREMENT.md)). The h-curve menu above therefore
applies to **rung ① / the hybrid upside SKU / >512 GB checkpoints**, not the primary rung-③ SKU; the
**54–127** figure survives only as the hybrid-SKU-if-h≥0.75 note. U(K) is now **GLM-family measured**
(GLM-4.5-Air: U(2)=1.60–1.64, U(4)=2.60–2.71, U(8)=4.19–4.41 — [`H_MEASUREMENT.md`](H_MEASUREMENT.md)
2nd measurement; OLMoE stays as the first-pass history).

**Reading it.** The product — a **single-user local box** — stacking the faithful RTL levers projects
**~15–40 tok/s [EST] on the funded custom board (rung ②) and ~5–8 tok/s [EST] on the near-term prove-it FPGA
(rung ①)** — the tok/s a box hits is set by the memory bandwidth of the rung you build
([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)), not by the RTL alone (the RTL is the same, ggml-bit-exact Q4_K
core on every rung). The rung-② working-set band **needs expert-cache hit-rate — which routing entropy caps
(predictor-prefetch is a measured no-op) — or non-bit-exact pruning (#6/#7)** to reach the upper end. Rung ②
is comfortably interactive, approaching the ~50 tok/s compute ceiling — at which point the answer is a
bigger/cheaper compute die (compute is **not** the BOM cost). *(The aggregate-serving row is a separate,
non-target deployment: batching many users reaches the **same** ~50 tok/s ceiling but as pooled throughput, so
each of the ~355 users floors at ~0.14 tok/s — that per-user figure belongs to a datacenter box, never to the
personal appliance.)* The **only** way to raise the ceiling *itself* above ~50 is near-storage compute (#13) —
the **rung-③ ASIC/SoC endgame** (HBM stacks + near-memory compute at manufacturing volume, for cost-down +
performance + power). Cache cleverness (predictor) and extra dies are provably capped. *(2026-07: per the
pivot note above, the primary rung-③ SKU is now full-residency LPDDR5X — near-storage compute stays the
endgame for the hybrid/streaming SKU and >512 GB checkpoints.)*

---

## 5. Dedup map (overlapping ideas collapsed)

- **PE_M widening** appeared 3× (wrapper-widen / expert-path-widen / attention-widen) → one keystone **#2**,
  with attention/hot-path as a defensive sub-case.
- **Draft model / raise-α** appeared 4× (on-die draft, resident draft, native multi-head, raise-α) →
  consolidated into **#4** (resident dense/Medusa draft, bit-exact).
- **Batched verify** appeared in both SPECULATIVE and CROSS-CUTTING → **#5** (the gate), with the
  geometry-rule (deep-narrow #22) and union-aware scheduling (#8) as its design constraints.
- **Exact router prefetch + K-token union** appeared in SPECULATIVE and CROSS-CUTTING → **#8**.
- **B≈256 knee / continuous-batch scheduler** are framing+defense of **#1**, not separate ceilings.
- **fmax tails / array-scale / softmax-pipeline / SRAM-stationary** all collapse to one note: **prefill &
  energy only, ~0 on single-user decode** (§2b).

---

*Estimates [EST] are first-order model projections (master eq + roofline), not silicon measurements.
**[prior-FP8]** multipliers were measured on the prior FP8 datapath (tag `fp8-verified-baseline`); their Q4_K re-measure is
PENDING and no Q4_K equivalent is fabricated. Faithful levers preserve the Q4_K arithmetic contract
(bit-exact to the ggml reference `tools/q4k_ref.py` — not the real GGUF / llama.cpp, which is OPEN); #6/#7
(and latent-KV, hierarchical indexing) are quality knobs and must be validated against the accuracy contract
before shipping.*
