# Single-Package GLM-5.2 (Q4_K) Inference System — design note

> **Current datapath is Q4_K; FP8 is the prior track (tag `fp8-verified-baseline`).** This single-package
> design was first drafted for the FP8 datapath; the **current product datapath is Q4_K**
> (`glm_q4k_soc` / `glm_q4k_soc_ms` over `glm_model_q4k`, a **Q4_K compute die**, wrapped by
> `glm_q4k_system_cdc`), and the weight store is the **~467 GB `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`**
> (~38% smaller than the prior ~753 GB FP8 checkpoint). **The byte counts, per-token footprints,
> cache sizes, and BOM numbers below are now the Q4_K figures** — re-derived from the prior FP8
> track by the deterministic quant ratio (Q4_K mix ≈ **0.6 B/param** vs FP8's ~1.0, i.e. **×0.6**),
> with FP8 shown only as the **prior comparison**. Per-bit **energy ratios are format-agnostic** and
> carry over unchanged, as does the **memory-hierarchy / streaming / expert-cache thesis**. The
> **FPGA fit is now MEASURED** (Vivado ML 2026.1 routed fit of `glm_q4k_system_cdc` on **XCKU3P**,
> compact config + ACT_HW=1: 142,320 LUT / 87.5 % synth-stage (routed 141,298 LUT,
> `fpga/results/util_routed_ku3p_acthw1.rpt`), 421 DSP, 0 BRAM, routed Fmax **46.5 MHz** after a
> closed bit-exact repipelining campaign — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); **board
> bring-up is not done**, so system-level perf/BOM numbers stay **[EST]**. **Not bit-exact to the
> published GGUF:** the mixed-type (Q6_K/Q8_0/F16) RTL consumers are now **DONE** (`make mixedtype`),
> but bit-exactness to the *real downloaded GGUF bytes / llama.cpp runtime* remains **OPEN**;
> the moat is **offline + full-frontier (753B) + appliance price**, *not* bit-exactness to the GGUF
> (see [`README.md`](../README.md)). RTL/test names of the form `*_fp8` below map to their `*_q4k`
> equivalents on main (tag `fp8-verified-baseline` preserves the FP8 track).

> **Scope.** A system design for running the *published* `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`
> weights on **one module** — a custom Q4_K compute die + **64 GB DDR5** (the fast working
> memory) + a **1–4 TB NVMe SSD** (the whole model) — instead of a multi-chip HBM cluster. It targets
> "the real 753B model runs, at interactive-ish speed," e.g. as a **local, single-user** USB-C
> external accelerator — a **fully offline / air-gapped** box that runs the full 753B frontier model
> **with the ethernet unplugged** (nothing leaves because there is **no path out** — see §3),
> not datacenter-scale real-time serving.
>
> **This doc details the *rung-2* build; the fast-memory tier is rung-dependent.** The concrete
> **64 GB DDR5 / ~400–600 GB/s** config described throughout is the **funded custom-board (rung 2)**
> point on the [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) — *not* the only spec. Performance is set by
> **memory bandwidth**, memory bandwidth by the silicon's IO pins + hard PHYs, and that by the build
> budget, so the product ships as a **3-rung ladder** running the *same* bit-exact Q4_K RTL, only the
> memory interface changing: **rung 1** (prove-it, now) a low-end FPGA + **DDR4 ~4 ch (~100 GB/s)** +
> 1 NVMe → **~5–8 tok/s [EST]**; **rung 2** (post-seed) this **DDR5 8–12 ch (or HBM), ~300–600 GB/s**
> custom board → **~15–40 tok/s [EST]**; **rung 3** (at volume) a SoC/ASIC with HBM stacks (~TB/s) →
> **~40+ tok/s [EST]**. *(Updated 2026-07 — the **rung-3 primary design point pivoted to full
> residency**: 512 GB LPDDR5X on-package (~1.1 TB/s) holds the whole ~467 GB checkpoint, cold store =
> one M.2 NVMe, design point **≈80 tok/s [measured-inputs EST]** — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md). The
> measured-proxy roofline design points ([`H_MEASUREMENT.md`](H_MEASUREMENT.md); h/U first measured
> on the OLMoE trace, U now superseded by the GLM-4.5-Air measurement): NVMe 1–2 drives, no
> multipliers ~0.5–1 tok/s; 90 GB DRAM + 100 GB/s ~13–24; 90 GB + 200 GB/s ~25–47; 225 GB +
> 200 GB/s ~54–127 — all [EST]; these **streaming** points now apply to **rung 1 / the hybrid upside
> SKU / >512 GB checkpoints**, not the rung-3 primary; the spec multiplier reads as **A/U(K)**,
> not ×K.)* Read every "64 GB DDR5"
> below as the **rung-2** spec — on rung 1 the fast tier
> is DDR4, on rung 3 it is **512 GB LPDDR5X on-package** (full residency; HBM stays the
> long-range ceiling).
>
> **Fast-memory choice: multi-channel DDR5, not HBM/GDDR6.** This workload is **NVMe/PCIe-bandwidth-
> bound** (the wall is reading cold experts from the NVMe SSD), so the fast tier only needs ~300–600 GB/s.
> An **8–12-channel DDR5** subsystem delivers that (DDR5-6400 ≈ 51 GB/s/ch → ~410 GB/s at 8 ch,
> ~615 at 12), making HBM's multi-TB/s — and even GDDR6's ~400–600 GB/s (8–12 ch) — more than required.
> DDR5 is the **cheapest (~$2–4/GB → ~$150–300 for 64 GB)** and **lowest-power per bit** of the
> three, and **densest by capacity** (64 GB = a few **DIMMs**, upgradeable — vs ~32 soldered GDDR6
> chips or an in-package HBM stack); the prefetch controller (`expert_cache_pf`) hides DDR5's
> access latency behind compute. The trade is a **wide multi-channel memory controller**
> (server-class, 8–12 ch) — that is DDR5's engineering cost, in exchange for the lowest BOM and
> power. *(Earlier drafts targeted HBM, then GDDR6; "the fast tier" below is now DDR5.)*
>
> Numbers tagged **[EST]** are system-level estimates (market-/physics-derived), not measured
> RTL results. The compute datapath this wraps is the verified RTL in this repo (see
> [`ACCEL_GLM52.md`](ACCEL_GLM52.md) and the `*_q4k` units); the memory/streaming system here
> is **designed, not built**.
>
> **Compute die → FPGA card now, ASIC at volume (the [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)
> rungs).** The near-term product realizes this "compute die" on an **FPGA** (FPGA + on-board DDR5/DDR4
> + an NVMe SSD via M.2/PCIe on one card) — rungs ①②. A **custom ASIC is the rung-③ endgame, not
> "out of scope."** The earlier "an ASIC's compute-density edge is wasted" call reasoned from
> *compute-bound*; but the real bottleneck is **memory bandwidth (IO pins + PHY)**, and an ASIC is
> **exactly what breaks the FPGA's IO/PHY ceiling** — HBM stacks + many-channel controllers +
> near-memory Q4_K compute at **~TB/s**, at **lower $/seat and lower power once amortized over volume**
> (see [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) P3.2). Its multi-million NRE + long lead time only
> pay off **at manufacturing volume**, so it is **sequenced *after* the FPGA proves product-market
> fit** — *not now* (no volume, no capital), but the deliberate endgame **for cost-down + performance
> + power** at scale. The single-package memory-hierarchy analysis below is agnostic to the choice —
> read "the die" as "the FPGA fabric (rungs ①②) or the ASIC (rung ③)."

---

## 1. Goal

One module that runs GLM-5.2 (UD-Q4_K_XL, 753B params, ~40B active/token, 1M context) by **storing
the whole model on a cheap NVMe SSD and streaming the per-token working set through fast DDR5
into a Q4_K compute die** — exploiting MoE sparsity (8/256 experts/layer) so only a small
fraction of the 467 GB is touched per token. Optimize for **cost + interactive speed** (a
USB-C external accelerator) over peak throughput.

## 2. The problem

| | Size [EST] | Consequence |
|---|---|---|
| Weights (Q4_K mix, ~0.6 B/param) | **~467 GB** (~450 GB are cold routed experts; vs the prior FP8 ~753 GB) | No chip holds it on-die or in HBM |
| Latent-KV cache @ 1M ctx | **~94 GB** (MLA; an MHA cache would be 5.36 TB) | Also too big for SRAM/HBM alone |
| Compute / token | **~80 GFLOP** (~40B active × 2) | *Small* — a modest die does it in ~80 ms |

The model is **memory-bandwidth-bound, not compute-bound**: ~40B params are active per token
(**≈25 GB** of weight at Q4_K's ~0.6 B/param). The pace-setting cost is the **read**, and the wall
is the **~14 GB of routed experts** (canonical derivation: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2) that change every token and stream from NVMe; the ~17 GB hot
non-routed set stays resident in DDR5 (per-token touch ~11 GB) and is re-read from fast memory. That read time dwarfs the
~80 GFLOP of math, so the design problem is a **memory hierarchy + streaming** problem. (The KV
cache is bf16 latent-MLA, unaffected by weight quant — still ~94 GB at 1M ctx.)

## 3. Architecture

```
                 ┌──────────────────────────── MODULE ─────────────────────────┐
                 │                                                              │
   token ──▶     │   ┌───────────────┐  weight-pull   ┌────────────────────┐    │
                 │   │ Q4_K COMPUTE  │◀──(w_req/w_col)─│ 64 GB DDR5        │    │
                 │   │     DIE       │   bf16 acts     │ (~400–600 GB/s, 8–12ch)    │    │
   logits ◀──    │   │ MLA·MoE·SwiGLU│──▶ bf16 out     │  • hot weights ~17GB│    │
                 │   │ + MTP + bf16  │                 │  • KV working window│    │
                 │   │   tail        │                 │  • EXPERT CACHE ~20GB│   │
                 │   └───────────────┘                 └─────────┬──────────┘    │
                 │                                      miss ▲   │ refill         │
                 │                                            │   ▼                │
                 │                              ┌──────────────────────────────┐  │
                 │                              │  1–4 TB NVMe (~10s GB/s agg) │  │
                 │                              │  • full ~450 GB cold experts  │  │
                 │                              │  • KV overflow (cold pages)    │ │
                 │                              └──────────────────────────────┘  │
                 └──────────────────────────────────────────────────────────────┘
   (USB-C to a host PC carries only token IDs in/out — the model never crosses it.)
```

**Why this is an offline / air-gapped appliance.** Because every byte of weights and KV lives
on-module and the host link carries **only token IDs**, the box runs the full 753B frontier model
**with the ethernet unplugged** — the crispest, binary form of "nothing leaves": your data *cannot*
leave because there is **no path out**. Lead with that capability (finally running a frontier model
where the cloud is barred — SCIFs, isolated OT / critical-infra, field/edge, or simply data you won't
hand to a vendor); strongest-possible non-egress is its *proof* (the audit is literally "does it still
work with the cable unplugged?" — yes). It also ends the "secured cloud" debate: in-VPC /
zero-retention / TEE deployments all still require connectivity and fail the unplugged test. Honest
caveats — the 467 GB model is **provisioned once** (itself doable offline / in a secure facility) and
model/weight updates are **physical re-provisioning**; and "offline" *alone* is table-stakes for any
local box, so the moat is the **combination: offline + full-frontier (753B) + appliance price** (§11) —
a 70B laptop model fails frontier quality, an 8×H100 rig fails price/form-factor, and secured cloud
fails the unplugged test.

Three components, three roles:
- **Q4_K compute die** — the verified RTL (MLA attention, MoE, SwiGLU, RoPE, RMSNorm, LM head,
  MTP) with GGML Q4_K weight matmuls (dequant → fp32 MAC) + bf16 tail. Pulls weights via a streaming interface.
- **64 GB DDR5** — the *fast working memory*: everything reused every token (hot weights), the
  KV working window, and the **routed-expert cache**. (~400–600 GB/s at 8–12 channels ≈ exactly the ~300–600 GB/s this workload needs, since it
  is NVMe/PCIe-bound — HBM's/GDDR6's higher BW would be wasted.)
- **1–4 TB NVMe SSD** — the *cheap bulk store*: the entire 467 GB Q4_K model + KV overflow.

## 4. Memory tiering & map

| Tier | Size | Bandwidth [EST] | Contents | Reused every token? |
|---|---|---|---|---|
| On-die SRAM | MBs | ~10s TB/s | activations, GEMM tiles, DSA index scratch, double-buffers | — |
| **DDR5** | **64 GB** | **~400–600 GB/s (8–12 ch)** | **hot weights ~17 GB resident** (attention all layers, shared expert, dense FFN, router, embed/LM-head, norms; per-token touch ~11 GB — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2) + a wide KV working window + **expert cache ~20 GB** (~900 experts; the ~11 GB freed vs the FP8 hot-set widens the KV window) | hot: yes |
| **NVMe SSD** | **1–4 TB** | **~10s GB/s agg [EST]** (per-drive ~3.5 GB/s Gen3 x4 / ~7 Gen4 x4; scale via PCIe lanes / multiple drives) | **full ~450 GB cold routed-expert pool** + KV cold pages | no (streamed on demand) |

The split that makes it work: **non-routed params (~17 GB) are a *fixed* set used every
token → resident in DDR5.** The **~450 GB routed experts are a *data-dependent* set (8/256 per
layer, chosen at runtime) → live on the NVMe SSD, streamed/cached on demand.**

## 5. Per-token dataflow

1. **Embed** token (bf16, DDR5) → residual `x`.
2. For each of 78 layers:
   a. RMSNorm(x) (bf16, DDR5).
   b. **MLA attention** — weight projections (W_dq..W_o) pulled Q4_K from **DDR5 (hot)**; q·K
      score + softmax + weighted-V in bf16; KV append + DSA-gather of 2048 rows from the **KV
      window (DDR5)** / overflow (NVMe).
   c. **FFN** — dense layers (first 3): SwiGLU from DDR5. MoE layers (75): **router** picks
      top-8 experts → for each, **check the DDR5 expert cache → hit: read DDR5; miss: stream
      the ~22 MB expert from the NVMe SSD into DDR5, evict LRU** → SwiGLU; + shared expert (DDR5).
   d. Residual adds (bf16).
3. Final RMSNorm + **LM-head GEMV** (bf16, DDR5) → next-token logits → argmax/sample.
4. **Prefetch**: while layer L computes, DMA layer L+1's likely experts NVMe→DDR5 (double-buffer).

Hot reads (~11 GB touched of the ~17 GB resident) come from DDR5 (fast). The **routed-expert reads (~14 GB) are the
bottleneck** — DDR5-cache hits are fast, misses hit the NVMe SSD.

## 6. The bottleneck — routed-expert streaming

Per token the MoE layers need **75 × 8 = 600 expert blocks** (~23 MB each at Q4_K — 37.75M params ×
~0.62 B/param) = **~14 GB [EST]** (the wall — canonical derivation:
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2; vs the prior FP8's ~22 GB), scattered
data-dependently across the ~450 GB pool. Speed is set by:

```
  t_token ≈ max( t_compute≈80ms , t_hot_DDR5 , t_routed )
  t_routed ≈ (miss_rate × 14 GB) / NVMe_BW   +   (hit × 14 GB) / DDR5_BW
```

With a ~20 GB expert cache (≈900 of 19,200 expert-instances) and expert-popularity skew +
batch reuse, the **miss rate** — not raw compute — governs throughput.

## 7. Performance model [EST]

### 7.1 Measured cache hit rate (calibrated GLM-scale trace, RTL-confirmed)

The make-or-break unknown — the expert-cache hit rate — was simulated at GLM scale (256
experts × 75 layers, top-8) with a routing trace calibrated to a *trained* MoE router
(load-balanced → mild popularity skew, weak temporal locality), fed through the **real
`expert_cache_ctrl` RTL** (hit/miss bit-exact vs a python LRU model). Tools:
`tools/route_trace.py`, `tools/glm_cache_confirm_tb.v`.

**batch=1 (interactive decode) hit rate vs DDR5 cache size:**

The RTL cache is **slot-based**, so the hit rates are a function of *slot count* and routing,
**independent of the quant format**; only the *bytes per slot* change (Q4_K expert ~22 MB vs FP8
~37 MB), so each slot count maps to a **~0.6×** DDR5 cache size vs the prior FP8 track:

| DDR5 cache | slots | uniform-ish (realistic) | skewed (optimistic) |
|---|---|---|---|
| ~3.3 GB | 150 | **0 %** | **0 %** |
| ~13 GB | 600 | 26 % | 51 % |
| **~20 GB** (the 64 GB config: hot ~17 + cache ~20 + wide KV window) | 900 | **27 %** | 53 % |
| ~40 GB | 1800 | 31 % | 58 % |

Two non-obvious findings the sim revealed:
- **Hard threshold at ~13 GB (= one token's 600-expert Q4_K footprint; ~22 GB on the prior FP8
  track).** Below it the hit rate is **0 %** — in batch=1 the decoder sweeps all 75 layers per
  token, so an expert is evicted long before the *next* token revisits its layer unless the cache
  holds a full token's footprint. **The 64 GB DDR5 (~20 GB cache / 900 slots) sits just past this knee.**
- **Trained routers are load-balanced → less cacheable than a naive Zipf.** The realistic
  ("uniform-ish") hit rate at ~20 GB (900 slots) is **~27 %**, not the ~67 % a synthetic skewed trace
  suggested.

**Batching is the real lever, not cache size.** With layer-major batched access (experts reused
within a layer across the batch) the hit rate is ~28–50 % (batch 8) to ~47–66 % (batch 32)
**even at a ~3.3 GB cache** — cache size becomes nearly irrelevant.

### 7.2 Combined throughput model (batching + prefetch)

With **prefetch** the NVMe *latency* is hidden behind compute (§8: 99 % of stall removed when
the compute window ≥ NVMe latency), so the machine runs at the NVMe/PCIe **bandwidth** wall. The
master equation (per-token routed footprint = 600 experts; **Q4_K ≈ 14 GB** — canonical:
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2; vs the prior FP8's ~22 GB):

> **aggregate tokens/s ≈ NVMe_BW / [ (1 − h) × footprint ]**  ·  (then × K for speculative/MTP)

> **Measured correction ([`H_MEASUREMENT.md`](H_MEASUREMENT.md)):** the "× K" term must be read as
> **A/U(K)** — the K speculative tokens route to *overlapping but not identical* experts, and the
> measured union factor (U(2)=1.51–1.65, U(4)=2.25–2.64, U(8)=3.25–3.92, OLMoE proxy) caps the
> amortization at **~1.1–1.3× at K=4 (A≈3)**, not ~×K. h now also has measured-proxy values
> (bandwidth-h 0.36–0.60 at a ~90 GB / 20 % cached pool; 0.72–0.88 at ~225 GB / 50 %).
> *(Updated 2026-07: U(K) is now **GLM-family measured** — GLM-4.5-Air, superseding the OLMoE
> first pass: U(2)=1.60–1.64, U(4)=2.60–2.71, U(6)=3.46–3.62, U(8)=4.19–4.41, ±0.05. And on the
> full-residency rung-3 primary ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)) **h=1 by
> construction** — the h values matter only for rung 1 / the hybrid upside SKU.)*

where **h** is the *batched* cache hit rate measured through the real RTL (§7.1, §8):
batch 1 = 26.5 %, batch 8 = 29.7 %, batch 32 = 50.5 %. Each lever moves one term:

| Lever | What it changes | Measured effect |
|---|---|---|
| **Prefetch** | latency-bound → **bandwidth-bound** | required to reach the wall; 99 % stall cut |
| **Batching** | raises **h** → lowers (1 − h) | h 27 %→50 % (batch 1→32) ⇒ only **~1.5×** here |
| **Sub-Q4 re-quant** (off-path) | shrinks the **footprint** below Q4_K | modest — Q4_K already banked the FP8→Q4 **~1.6×** |
| **Speculative/MTP** | **÷K** weight passes (K tokens/pass) | measured **A/U(K) ≈ 1.1–1.3× at K=4** ([`H_MEASUREMENT.md`](H_MEASUREMENT.md)), not ~×K |
| **NVMe/PCIe bandwidth** (hardware) | raises **NVMe_BW** (more lanes / drives) | linear |

**This project runs the published Q4_K weights** (`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`),
faithfully, no further re-quantization — the byte counts here **are the Q4_K figures** (re-derived
from the prior FP8 track by the deterministic **×0.6** quant ratio; the store is ~38% smaller). So
the throughput levers are the faithful ones; further **sub-Q4** re-quant is *off the faithful path*
(it means re-quantizing the model ourselves and owning the quality risk) and is listed only as an
escape hatch — and Q4_K already banked most of the old FP8→INT4 gain. Aggregate tokens/s on the
**Q4_K path** (NVMe 50 / 100 GB/s **aggregate** — striped across many PCIe lanes / multiple NVMe
drives, **not** a single M.2 [EST]; prefetch on):

| Config (Q4_K path) | h | (1−h)×13 GB | @50 GB/s | @100 GB/s |
|---|---|---|---|---|
| batch 1 (single-user) | 27 % | ~9.5 GB | ~5 | ~11 |
| batch 1 + **MTP ×2** | — | — | ~10 | ~21 |
| batch 32 | 50 % | ~6.5 GB | ~8 | ~15 |
| **batch 32 + MTP ×2** | — | — | **~15** | **~31** |
| *(off-path)* sub-Q4 re-quant batch 32 + MTP ×2 | — | ~4.3 GB | ~23 | ~46 |

**The multiplier that matters is speculative / MTP decoding — but its measured value is A/U(K),
not ×K.** GLM-5.2 ships an MTP head (`num_nextn_predict_layers=1`) and we built it (`mtp_head_q4k`):
verifying K tokens per weight-load pass amortizes the NVMe traffic **without leaving Q4_K** — but the
K drafts route to overlapping-not-identical experts, so the measured amortization is
**A/U(K) ≈ 1.1–1.3× at K=4, A≈3** ([`H_MEASUREMENT.md`](H_MEASUREMENT.md), OLMoE proxy — since
superseded by the GLM-4.5-Air measurement, U(4)=2.60–2.71; GLM-5.2's own routing still
unmeasured), not the ideal ~K×. Read the "MTP ×2" rows above as ideal-K upper bounds.

**Batching is not a free Nx** in this NVMe/PCIe-bandwidth-bound regime — it only helps through the
hit rate, and trained-router entropy caps the reuse: batch 32 gives **~1.5× aggregate**, split
across the B streams (**per-user = aggregate ÷ B**), i.e. it trades single-user latency for
aggregate throughput — that **batched/aggregate regime is a non-target datacenter deployment of the
same silicon, not this single-user (B=1) product**.

**Bottom line (Q4_K):** this section's **conservative** model (prefetch + batch-hit-rate + MTP only)
puts single-user at **~5–11 tokens/s**, **~10–21 with MTP ×2**. Stacking the *full* faithful lever set
(`flash_xbar` N-way read banking across PCIe lanes / drives + `weight_decomp` + activation-sparsity + draft-K + hot-weight) raises
the single-user ceiling to **~25–40 tok/s [EST]** — the top of the **rung-2 (funded custom board)**
range (~15–40) and the fuller-stack product headline ([`ULTRA_PERF.md`](ULTRA_PERF.md) §4). Stage
that to the [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md): the old flat "~25–40" was implicitly this
**rung-2** number and it is **not** reachable on the cheap near-term hardware — the **prove-it FPGA
(rung 1, DDR4 ~4 ch) is ~5–8 tok/s [EST]** (real + bit-exact, slow-but-honest), and the **rung-3
SoC/ASIC runs ≈80 [measured-inputs EST]** at volume (2026-07 full-residency primary — 512 GB LPDDR5X on-package,
~1.1 TB/s; [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)). The **~15–31 aggregate** at batch 32 + MTP ×2 (~100 GB/s
aggregate NVMe — many PCIe lanes / drives striped [EST]) is the **non-target batched/datacenter regime of the same silicon, not the product's
speed** (the box runs B=1). Prefetch is required (hides
latency → reach the bandwidth wall); MTP and raw NVMe/PCIe bandwidth are the real multipliers;
batching is a modest, latency-costing aggregate boost. Interactive, not datacenter-real-time.
Compute and the single die are *not* the limit (the die idles on NVMe reads); the wall is moving
~14 GB of routed-expert weights per token across the on-module NVMe/PCIe bus. (Further sub-Q4
re-quant would push further but is a different, re-quantized model — outside the "run the published
Q4_K" goal, and Q4_K already banked the FP8→Q4 ~1.6×.) *(Measured update: the "MTP ×2" rows are
ideal-K upper bounds — the measured spec amortization is A/U(K) ≈ 1.1–1.3× at K=4 — and the
measured-proxy design-point menu (NVMe-only ~0.5–1 · 90 GB+100 GB/s ~13–24 · 90 GB+200 GB/s ~25–47 ·
225 GB+200 GB/s ~54–127 tok/s [EST]) is in [`H_MEASUREMENT.md`](H_MEASUREMENT.md) — updated 2026-07:
that streaming menu now applies to rung 1 / the hybrid upside SKU / >512 GB checkpoints; the rung-3
primary is full residency, design point **≈80 tok/s [measured-inputs EST]** ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)).)*

## 8. MoE expert-cache subsystem (the heart of it)

Because the active expert set is **data-dependent and changes every token**, routed experts
can't be statically placed — it's a **caching + scheduling** problem:

- **Cache** (DDR5, ~20 GB / ~900 Q4_K experts): LRU/LFU of expert blocks; exploits expert-popularity skew.
- **Batching**: many tokens/sequences route to overlapping experts → load once, reuse across
  the batch (biggest throughput lever; costs latency). **RTL-measured** through the committed
  `expert_cache_ctrl` at ~20 GB cache (900 slots): batch 1 / 8 / 32 → **26.5 % / 29.7 % / 50.5 %** hit rate
  (same router picks, only access order changes — isolating batching as the lever). **The stronger
  batching lever is expert-*union* reuse** — fetch each layer's union of selected experts once and
  share it across B rows — now realized in RTL: `glm_decoder_block_q4k`'s PE_M>1 MoE loop fetches
  **only** the union (`T_ESCAN` scan + `any_has` skip), and the PE_M batch-widen is **DONE 4/4**
  (swiglu/router/mla/mtp), all bit-exact. This reframes aggregate batching from the ~1.5×
  hit-rate view (§7.2) to a **6–8× aggregate** lever near B≈256 (a **non-target batched/datacenter regime**; the product itself runs B=1); see
  [`ULTRA_PERF.md`](ULTRA_PERF.md) #1 and [`FLASH_STRIPING.md`](FLASH_STRIPING.md) §4.
- **Prefetch/predict**: speculate next experts (the next layer's router is cheap and runs ahead)
  and DMA into DDR5 during the current layer's compute → hide the **big NVMe fetch latency**.
  Built + measured as **`src/expert_cache_pf.v`** (a prefetch hint port + demand-priority
  background NVMe fetch + a `demand_stall_cycles` counter; demand path bit-exact to
  `expert_cache_ctrl` with prefetch off; a `CACHE_HIT_LAT` parameter models the DDR5 read).
  Honest result (compute-window model, FLASH_LAT=20, DDR5 `CACHE_HIT_LAT=4`): prefetch
  **trades the big NVMe-miss stall (≈22 cyc) for the small DDR5 read** — demand stall cut
  **~81 %** (4400 → 818), not the ~99 % an idealized zero-latency cache (CACHE_HIT_LAT=0) shows.
  The residual is the irreducible DDR5 read floor (`+CACHE_HIT_LAT` per resident hit); a
  second-level DDR5→die read-ahead could hide that too (future work).
- **Layout**: store co-activated experts contiguously / aligned for sequential NVMe reads
  (bandwidth- not IOPS-bound, since each expert is a ~22 MB contiguous Q4_K block).
- **Speculative / MTP decoding**: GLM-5.2 ships an MTP head (built here as `mtp_head_q4k`) — verify
  K tokens per weight-load pass → cut weight traffic by the measured **A/U(K) ≈ 1.1–1.3× at K=4**
  (the drafts' expert unions overlap only partially — [`H_MEASUREMENT.md`](H_MEASUREMENT.md)), not ~K×.

## 9. Hardware ceiling vs software leverage

| | Sets it | Knobs |
|---|---|---|
| **Hardware ceiling** | raw NVMe/DDR5 bandwidth, bus width, **energy/bit**, compute rate | more PCIe lanes / NVMe drives + DDR5 channels, wider bus, faster die |
| **Software leverage** | how much you *actually* move + when | batching, expert cache policy, prefetch, **quantization**, speculative/MTP, storage layout, scheduling/overlap |

Software can't beat the bandwidth/energy ceiling but **gets you close to it and cuts demand** —
exactly how today's stacks (vLLM, llama.cpp/KTransformers MoE offload, DeepSpeed ZeRO-Infinity,
FlexGen) run 600B+ MoE models on a single GPU + RAM/SSD. The **hardware ceiling is itself a ladder**,
not a fixed number: more channels / PCIe lanes = a bigger, newer chip = more money, so the reachable
tok/s climbs rung by rung (FPGA+DDR4 → FPGA+DDR5/HBM → ASIC+HBM) — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)
for the per-rung tok/s [EST].

## 10. Power / heat

The dominant dynamic energy is **moving ~14 GB/token of routed weights** (Q4_K; ~22 GB on the
prior FP8 track). Keeping the whole model
**on-module** (DDR5 + NVMe next to the die) is the key win — vastly less energy than streaming
weights from a host over USB/PCIe. Among the fast-tier options DDR5 is the **lowest-power**
choice (mainstream, not the high-speed/high-power GDDR6; its per-bit energy is above an
in-package HBM stack but it uses far fewer, slower devices than GDDR6). On the compute side
Q4_K's dequant→fp32 MAC (the weights arrive as 4-bit codes; the prior-track FP8 `glm_matmul_fp8`
measured 18× 7-bit multipliers vs fp32's 24×24) keeps the die's dynamic power and DSP/area down. Net: tens of W (the v3-volume residency box floor is **≥50–78 W [EST]** — a floor, not a budget,
the SoC term is UNVERIFIED; the old ~40–60 W is retired, never derived — canonical
in [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §4) — needs a
heatsink/fan (a small box, not a thin USB stick), powered over its own DC/USB-C PD input (R3 recommends a ~100–140 W adapter; whether PD 100 W suffices depends on the UNVERIFIED SoC term — R3 §4).

> **Honest energy caveat (research-backed).** Prefetch + caching hide NVMe *latency* but **cannot
> remove its energy-per-bit penalty** — an NVMe SSD is still a NAND array behind a PCIe controller, so
> the storage-read energy (NAND array **plus** the PCIe SerDes/host controller) is at least the
> **~24–26× DRAM** the NAND term alone costs, and for offloaded decode the per-token energy can be
> **up to ~12× an HBM-resident baseline**. So **NVMe offload is a *capacity/throughput* tool, not an
> energy win** — the lever is to **minimize byte traffic** (expert caching + batching to reuse fetched
> experts, and `weight_decomp` to stream fewer bytes), not the NVMe/PCIe bus itself. (Sources via the
> deep-research pass; this corrects any "storage = low power" reading.)

## 11. Cost — memory BOM [EST, 2025–26, volatile]

| Chip | $/GB | Qty | Cost |
|---|---|---|---|
| **DDR5** | ~$2–4 | 64 GB (a few DIMMs, e.g. 8× 8 GB across 8 channels) | **~$150–300** |
| NVMe SSD (M.2 / PCIe) | ~$0.05–0.10 | 1–4 TB | **~$60–300** |
| **Memory chips total** | | | **≈ $200–400** |

(DDR5 is the cheapest fast tier — ~5–8× under 64 GB HBM (~$1–1.5k) and below GDDR6 (~$200–500),
at no performance loss since the workload is NVMe/PCIe-bound. The trade is a **wide 8–12-channel
memory controller** (server-class) to reach the bandwidth, not extra chips — DIMMs are dense and
upgradeable. The NVMe SSD that holds the entire 467 GB model is cheap either way — a small fraction of the DDR5 BOM.)

*Not included:* the board (a few DDR5 DIMMs across 8–12 channels + the die + an NVMe SSD via M.2/PCIe on a PCB —
DDR5 needs **no CoWoS / interposer**, a real simplification vs HBM, and far fewer devices than
GDDR6), the **wide multi-channel DDR5 controller IP** (the real engineering cost), the **NVMe/PCIe
host controller**, and the custom compute-die NRE + die cost. For context, an H100's 80 GB *HBM* alone
is ~$2k of its BOM — the DDR5 here is ~$150–300 for the same capacity, the payoff for being
NVMe/PCIe-bound.

## 12. Mapping to the committed RTL

**What this repo already provides (the compute die):**
- The full GLM-5.2 (UD-Q4_K_XL) operator datapath: `glm_matmul_q4k`, `swiglu_expert_q4k`,
  `mla_attn_q4k`, `moe_router_q4k`, `glm_decoder_block_q4k`, and the capstone **`glm_model_q4k`**
  (full forward pass, next-token argmax; the Q4_K GEMM core is bit-exact to the ggml Q4_K
  reference `tools/q4k_ref.py`, and the assembled model now has an end-to-end golden —
  `make model-q4k` 1155 + `model-q4k-acthw` 1155; bit-exactness to the real GGUF bytes /
  llama.cpp remains open).
- **Streaming weight-pull interfaces** (`w_req`/`w_col` + per-[128,128]-block bf16 scales) on
  every unit — the weight *source is abstracted*, so DDR5/NVMe/host can drive them.
- The **`mtp_head_q4k`** for speculative decoding.
- **PE_M batch-widening (4/4)** on all Q4_K wrappers (`swiglu_expert_q4k` / `moe_router_q4k` /
  `mla_attn_q4k` / `mtp_head_q4k`) — B token-rows share one weight fetch, verified bit-exact — and
  **union-skip grouped MoE** in `glm_decoder_block_q4k` (PE_M>1 fetches only the selected-expert
  union: `T_ESCAN` scan + `any_has` skip), the batch-axis footprint-reduction lever (ULTRA_PERF #1).
- A small-scale DMA append/gather streaming datapath (`tpu_soc`/`axi_master_dma`/
  `scatter_gather`/`cdc_async_fifo`) exercising the control logic.
- The **MoE expert-cache controller** in RTL — `expert_cache_ctrl` (tag/LRU; hit/miss bit-exact
  vs a python LRU model) and the prefetching **`expert_cache_pf`** (prefetch-hint port +
  demand-priority background NVMe fetch + a `demand_stall_cycles` counter). The DDR5/NVMe it
  caches from is still a model/stub.
- The **KV-cache pager** `kv_cache_pager` (append + DSA-gather window, `NSEQ` independent ring
  windows, optional SECDED-ECC); its backing memory is still a model/stub. A batched
  multi-sequence SoC top (`glm_q4k_soc` / `glm_q4k_soc_ms`) wires the model + pager + expert
  cache + a host prefill/decode FSM together.

**What this design adds (not built — the system layer):**
- DDR5 PHY + an **NVMe/PCIe host controller** (PCIe root complex + PHY) and USB-C device controller (the licensed vendor IP + real backing store,
  vs the stubbed `ddr5_xbar`/`flash_xbar` crossbars and cache/pager models above). **Note:** `flash_xbar` (with `FLASH_LAT`, `flash_req`/`flash_seq`/`flash_is_expert`/`flash_expert_id`) is the committed RTL name for the **storage-read fabric**; its address→weight-bytes read-request / latency-hiding abstraction is **medium-agnostic**, so in the product the NAND-specific backend is swapped for the NVMe/PCIe host controller with the module names left unchanged.
- The runtime/scheduler (batching, prefetch, speculative-decode loop) — largely software.

## 13. Open questions / honest limits

- **Expert-cache hit rate** — now estimated (§7.1) on a *calibrated* GLM-scale trace through
  the real `expert_cache_ctrl` RTL: ~27 % at batch=1 / ~20 GB cache (900 slots), with a hard 0 %
  floor below ~13 GB (Q4_K; the floor is slot-based), and batching as the dominant lever. Still
  **calibrated, not captured** — the actual
  numbers need a *real* GLM-5.2 routing trace (can't run 753B here); the trained-router balance
  assumption could be off in either direction. **Update:** h/U now have **measured-proxy** values
  from a real MoE trace (OLMoE-1B-7B-Instruct — EOR 0.35–0.49, U(2)=1.51–1.65 / U(4)=2.25–2.64,
  bandwidth-h 0.36–0.60 at a ~90 GB / 20 % cached pool; LRU collapses to ~0 below 10 % cache) —
  see [`H_MEASUREMENT.md`](H_MEASUREMENT.md) and [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md).
  **The GLM-family rerun is now DONE** (2026-07 2nd measurement: GLM-4.5-Air traced on an H100 via
  MoE-gate hooks — EOR 0.36–0.39 (~6× random), U(2)=1.60–1.64, U(4)=2.60–2.71, U(6)=3.46–3.62,
  U(8)=4.19–4.41, workload variance ±0.05 — superseding the OLMoE proxy, which stays as the
  first-pass history); **GLM-5.2's own routing remains unmeasured** (Air is GLM-family, not the
  flagship). With the rung-3 full-residency pivot ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md))
  **h is no longer product-deciding on the primary SKU** (h=1 by construction); h-curves stay
  relevant for rung 1 / the hybrid upside SKU.
- **NVMe/PCIe bandwidth** (~10s GB/s **aggregate**) is assumed; this is **not** one M.2's figure — a
  single PCIe Gen3 x4 NVMe is ~3.5 GB/s and Gen4 x4 ~7 GB/s [EST], so ~10s GB/s means **striping many
  PCIe lanes / several NVMe drives**, which the custom board must actually deliver. PCIe/NVMe read BW
  still caps well below DDR5 (which is exactly why the NVMe tier, not DDR5, is the wall).
- **64 GB DDR5 is comfortable for Q4_K** (hot ~17 GB + ~20 GB cache / ~900 slots + a wide KV
  window); the batch=1 knee is now **~13 GB / 600 slots** (Q4_K experts are ~22 MB), so the DDR5
  sizing is **far more forgiving than on FP8** — **48 GB still clears the knee** (hot 17 + ≥13 GB
  cache, with room to spare) at full single-user performance, and only well under ~40 GB does the
  cache risk dropping below the 600-slot floor. Batched serving is insensitive to cache size, so a
  smaller DDR5 is fine there too. (On the prior FP8 track the same knee sat at ~22 GB, so 48 GB FP8
  fell short → ~27 % slower single-user; Q4_K's smaller experts remove that constraint.)
- **Wide memory controller** (an 8–12-channel DDR5 subsystem to reach ~400–600 GB/s, server-class
  routing/signal-integrity) is DDR5's real engineering cost — but it needs no advanced packaging
  (no CoWoS/interposer) and far fewer devices (a few DIMMs vs ~32 GDDR6 chips), and the DIMMs are
  upgradeable.
- This is **interactive, not datacenter-real-time**; high tokens/s/user at scale still wants
  multi-chip HBM (bandwidth), which the **rung-2** DDR5 board here deliberately trades away for cost.
  Reclaiming that bandwidth is precisely the **rung-3 SoC/ASIC** endgame (~TB/s, many-channel PHY,
  near-memory compute) — sequenced after the FPGA proves PMF, at volume; see
  [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md). *(Updated 2026-07: the rung-3 **primary** design point
  is now **full residency** — 512 GB LPDDR5X, 1024-bit on-package, ~1.1 TB/s, the whole ~467 GB
  checkpoint DRAM-resident, one commodity M.2 NVMe as cold store, design point **≈80 tok/s [measured-inputs EST]** —
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md); HBM stays the long-range ceiling, and this doc's
  NVMe-streaming analysis applies to rung 1 / the hybrid upside SKU / >512 GB checkpoints.)*
