# Low-power design — the output-preserving path to a cool single box

> **This doc describes the CURRENT Q4_K product track.** The energy analysis below (bandwidth
> roofline, cache/paging, DVFS, clock-gating) is **format-agnostic** and has been reframed FP8 → Q4_K
> (753 GB → **~467 GB**, ~1.0 → **~0.6 B/param**). A handful of numbers were *measured on the prior
> **FP8** track* (now preserved on tag `fp8-verified-baseline`) — lossless-compression
> ratios, the BFP-accumulator cell count, and the `FLASH_LAT` stall table — and are called out inline
> as **prior-FP8, Q4_K re-run PENDING**. They are **not** relabelled as Q4_K results. See
> [`Q4K_RETARGET.md`](Q4K_RETARGET.md) / [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md) and the honest
> [`../README.md`](../README.md). RTL/test names of the form `*_q4k` are the current modules; the BFP
> accumulator survives in-tree only as the FP8 track, while `weight_decomp` (order-0) is now **wired
> on `main`** into `glm_q4k_system` behind the default-off `DECOMP` parameter and release-gated
> (`make weight-decomp`, `decomp1-elab`) — only its measured *ratio* is FP8-era.

**Requirement:** the accelerator must be low-power. **Decision (this project):** every power lever must
be **output-preserving** — it must not move the decoded token relative to the un-restructured
`glm_model_q4k` reference (the invariant is self-consistency, §8; the Q4_K **GEMM core** itself is
bit-exact to the independent ggml reference `tools/q4k_ref.py`, **not** to the real downloaded GGUF or
llama.cpp). Push a **single box as low as possible** on that floor, and add **fidelity-trade** levers
(§6) only if the output-preserving floor misses the target. This doc is the honest energy budget, the
lever ladder, and what is built vs. staged.

> **The assembled end-to-end numeric golden is now DONE** (`make model-q4k` — 1155 tests bit-exact vs
> the numpy reference `tools/glm_model_q4k_ref.py`, plus `make model-q4k-acthw` through the ACT_HW=1
> datapath). The output-preservation of every lever here is checked as *self-consistency* (spec ==
> greedy, DUT-vs-DUT) plus that golden and the GEMM-core bit-exactness vs `q4k_ref.py` — still **not**
> against the real downloaded GGUF / llama.cpp (that external validation remains OPEN; see README
> *What's proven*). "Output-preserving" below always means *relative to
> the reference `glm_model_q4k`*, never "byte-identical to the file people download."

## 1. The one fact that governs power: it's mostly storage read (NVMe)

> **(Updated 2026-07 — rung-③ design-point pivot, [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)**
> The primary rung-③ design point is now **full residency**: 512 GB LPDDR5X (16×32 GB, 1024-bit
> on-package, ~1.1 TB/s) holds the WHOLE ~467 GB checkpoint — **h=1 by construction**, cold storage is
> one commodity M.2 NVMe used only for the ~70 s boot-load, box **≥50–78 W [EST] floor** (R3 §4 —
> the old ~40–60 W is retired, never derived). On that box the dominant
> per-token energy term moves from the NVMe read to the (much cheaper/bit) LPDDR5X read. The
> NVMe-streaming energy analysis below stays TRUE and ACTIVE for the **rung-① FPGA demo, the hybrid
> upside SKU, and >512 GB checkpoints** — it is re-scoped, not deleted.

`J/token ≈ bytes_moved × energy/bit`. Two anchors decide everything:

- **Storage-read energy per bit ≫ DRAM/bit.** The NVMe/PCIe read path (NAND cell + SSD
  controller + PCIe PHY) costs far more energy/bit than a DDR5 access. The earlier NAND-cell
  anchor was ~24–26× DRAM/bit ([`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md)); an
  NVMe read adds controller + link overhead on top, so treat the exact multiplier as **[EST]**
  (can't be cleanly re-derived without a real drive — but storage-read still dominates). This
  anchor is about the *medium*, not the numeric format, so it is unchanged by FP8 → Q4_K.
- The die is **storage-bandwidth-bound (NVMe/PCIe)** — it sits idle waiting on the routed-expert
  stream ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md), [`ULTRA_PERF.md`](ULTRA_PERF.md)). This is a
  roofline property of a 753B MoE reading experts from storage, format-agnostic.

So per-token energy splits roughly (Q4_K, **~25 GB active bytes/token** = ~40B active × ~0.6 B/param):

| bucket | bytes/token | why | can we cut it? |
|---|---|---|---|
| **NVMe routed-expert bytes** | **~14 GB (the wall)** | top-8/256 experts/MoE-layer **change every token** and are streamed from the NVMe SSD on the streaming rungs (467 GB ≫ those rungs' DDR — can't all reside there; the rung-③ residency box holds all 467 GB in 512 GB LPDDR5X, so this row becomes a DRAM read — see the pivot note above) | **only** by moving fewer bytes or moving them less often |
| DDR / cache / KV | **~11 GB hot-set touch** + latent-KV (canonical: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2 — 25 − 14 GB; the earlier "~9 GB" was an undercount) | attention/dense/shared working set that *can* cache in fast DDR (DDR4 rung-① / DDR5·HBM rung-②) | HBM (energy/bit), smaller footprint |
| **compute die** | ~80 GFLOP/token (bf16) | on a die that is mostly stalled on the expert stream | DVFS, gating, die-shrink (all done/free — see §4) |

Although the routed experts are only ~56 % of the ~25 GB active bytes, they **dominate the energy**
because they come from **NVMe** (≫ energy/bit than the DDR-resident hot-set) **and** must be re-read
**every token**. Q4_K already banks a **~1.6× byte reduction vs the prior FP8 checkpoint** (0.6 vs
~1.0 B/param, 467 vs 753 GB) — that quantization *is* a ~40 %-off energy/token win at the format level,
already in the baseline below.

**The blunt conclusion: DVFS, die-shrink and clock-gating only touch the small, mostly-idle compute
slice. Real low power is won on the ~14 GB/token of NVMe routed-expert bytes — cut them, or amortize
each fetch across more tokens** (spec-decode K — the single-user box's lever). Amortizing across B
*users* instead is the **non-target datacenter-batch** regime (the same silicon batched, kept for
analysis but not this personal box, which runs B=1). Everything below is ranked by that truth.

> **Rung note — per-token power is hardware-rung-dependent ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).**
> Every output-preserving lever in this doc is the **same RTL on every rung**, so it cuts J/token
> identically on the rung-① prove-it FPGA (DDR4) and the rung-② funded board (DDR5/HBM). But the
> **structural** power win — driving down the dominant storage-read **energy/bit itself** — is the
> **rung-③ ASIC endgame**: HBM stacks + many-channel PHY + **near-memory Q4_K dequant/compute** at
> ~TB/s move the ~14 GB expert bytes far cheaper/bit, for **lower $/seat + lower power once the NRE
> amortizes over volume**. ASIC here is **not** a compute play (compute is already ~free, §4) — it is
> the **volume power/cost win**, sequenced *after* the FPGA rungs prove PMF, not "out of scope."
> *(2026-07 pivot: the rung-③ **primary** SKU is now the full-residency 512 GB LPDDR5X box (≥50–78 W
> [EST] floor, [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §4); the near-memory/HBM path stays the endgame for the
> hybrid/streaming SKU and >512 GB checkpoints.)*

## 2. The irreducible floor

The active experts **must** be re-read from the NVMe SSD every token *on the streaming rungs* (the
model is fixed; the ~467 GB routed-expert set can't reside in those rungs' DDR — tens of GB,
rung-dependent, DDR4 on the prove-it rung and DDR5/HBM on the funded board;
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). *(2026-07: on the primary rung-③ **residency** box the
whole 467 GB resides in 512 GB LPDDR5X — h=1 by construction — so the floor there is the per-token
LPDDR5X re-read, far cheaper/bit; [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)* That sets a hard
J/token floor. The *only* output-preserving ways under it are: **(a) fewer fetches per token** (amortize
one weight-load across K tokens via spec decode — the single-user box's lever; the across-**B-users**
batch variant is the non-target datacenter regime, kept for analysis, not this product), **(b) more of
the hot working set resident** (bigger/faster DDR → higher hit-rate → fewer NVMe reads; a hardware-$
lever — literally climbing the ladder: more DDR channels / HBM = a bigger chip = a higher rung,
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md); now proxy-MEASURED — [`H_MEASUREMENT.md`](H_MEASUREMENT.md),
OLMoE trace: residency-only bandwidth-h ≈ 0.36–0.60 with ~20 % of the expert pool cached (~90 GB at GLM
scale), 0.72–0.88 at ~50 % (~225 GB), LRU ~0 below a 10 % cache — h-curves that, post-pivot, matter only
for the **hybrid-SKU decision**: the primary rung-③ box is full-residency h=1). Compute tricks cannot
touch the floor.

> **What about lossless weight compression?** On the **prior FP8 track** the trained E4M3 weight *bytes*
> had entropy well under 8 bits/symbol (~5–6.5), so a streaming lossless decompressor (`weight_decomp`)
> bought **~1.34× fewer NVMe bytes** — a real "(c) fewer bytes per fetch" lever **[ratio measured on
> the prior-FP8 track]**. **Q4_K weights are already 4-bit k-quantized** (near entropy-dense super-blocks: 256 weights
> in 144 B — fp16 `d`/`dmin` + 6-bit scales + 4-bit codes), so the lossless headroom is largely **gone**
> — the byte win is already banked in the quantization itself. The order-0 `weight_decomp` RTL is
> **wired on `main`** (default-off `DECOMP`, release-gated); order-1 `weight_decomp2` is **DEAD / NOT
> IN PRODUCT**. There is **no Q4_K re-run** and no claim that either
> ratio carries. For Q4_K, treat "fewer bytes per fetch" as **already spent by the format**, and lever
> (a) — spec-decode amortization — as the live one.

## 3. Output-preserving lever ladder (single-user)

The **absolute J/token is an [EST] roofline** (`bytes/token × energy/bit`, NVMe ≫ DRAM/bit) — **not**
silicon watts. A useful Q4_K anchor is the prior FP8 model's ~9 J/token roofline scaled by Q4_K's
~1.6× fewer bytes → **~5–6 J/token [EST]**; the *relative* levers below are the defensible part. The
compression rung of the old FP8 ladder is **omitted** here (prior-FP8, §2). Effects are expressed as
**relative** to keep them honest — the dominant one divides the ~14 GB NVMe term, which no compute
trick can touch.

| lever | mechanism | effect | status |
|---|---|---|---|
| `flash_xbar` ×N + deep queue | N× storage BW (PCIe lanes / multi-NVMe) + latency-hide | BW, not J/token | ✅ built (latency-hide + N× read fan-out); fabric BMC-proven |
| MTP/spec **K=2** | verify 2 tokens per weight-load (K_eff≈1.7) | **÷~1.7** on the NVMe term | ✅ built, spec==greedy exact (`spec_decode_top` 18/18) |
| grouped MoE **union-skip** batch **[non-target: B>1 multi-user datacenter regime; the box runs B=1]** | B rows share 1 expert fetch (÷ up to B) | ↓ at B>1 | ✅ built, output-preserving |
| **DVFS freq** (`clk_throttle`) | run die f/div in the ~4–5× stall slack (§4) | **peak-power only** (not J/token) | ✅ **RTL built, BMC-proven + byte-identical** — the eco/thermal knob |
| **DVFS voltage** | lower supply at the reduced f | −~15 % total **energy** | vendor/physical (the J/token half) |
| **spec high-K verify** (÷(K+1) weight-loads) | verify K+1 draft positions in ONE model weight-load (PE_M=K+1 batch) → **÷(K+1) weight-loads on the ~14 GB term** | scales with K_eff | ✅ **HW built + output-preserving** (`spec_batched_top` / `spec_chain_top`, spec==greedy via `make spec-slow`; PE_M weight-share on `glm_model_q4k`) |
| ↳ raise K_eff 1.7 → **3–5** | resident ~1–3 B dense draft (vs the chained MTP self-draft) proposes K=4–8 with higher acceptance | approaches the floor | ⏳ **draft-quality, not RTL** — needs a real 1–3 B draft-model artifact ([`ULTRA_PERF.md`](ULTRA_PERF.md) #4) |
| `weight_decomp` lossless (order-0) | fewer NVMe bytes (lossless pack) | **ratio unmeasured on Q4_K** — the 1.34× is an FP8-era measurement, not transferable [EST at best] | ✅ RTL **wired on `main`** behind the default-off `DECOMP` parameter (`glm_q4k_system.v`), release-gated (`make weight-decomp`, `decomp1-elab`); Q4_K is already 4-bit → little lossless headroom expected (§2) |

> **Measured U(K) cap on the ÷K rows ([`H_MEASUREMENT.md`](H_MEASUREMENT.md); OLMoE first-pass, now
> superseded by the **GLM-4.5-Air measurement** — U(2)=1.60–1.64, U(4)=2.60–2.71, U(6)=3.46–3.62,
> U(8)=4.19–4.41, EOR 0.36–0.39, workload variance ±0.05):** the K spec positions' expert *union*
> grows with K (OLMoE first-pass: U(2)=1.51–1.65, U(4)=2.25–2.64) — so on
> the routed-expert NVMe term the realized amortization is **A/U(K) ≈ 1.1–1.3× at K=4 (A≈3)**, not
> ÷K_eff or ÷(K+1). Raising K_eff without beating U(K) buys less than the ideal division suggests
> (see also [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md)).

**Projected output-preserving floor [EST]: dominated by spec high-K amortization** — it divides the
~14 GB NVMe term, which no compute trick can. Its **hardware is already built and output-preserving**
(§4a); what remains is *draft acceptance* (α), a **model-quality** property, not an RTL gap. The
absolute J/token stays [EST] until the vendor flow + a real board give watts.

### What is measured vs modeled (firming the [EST])
Be clear which inputs are which:
- **Measured/verified relative factors (Q4_K):** MTP K=2 **spec==greedy exact** (`spec_decode_top`
  18/18 — DUT-vs-DUT self-consistency, the "greedy golden" is itself a `glm_model_q4k`) → K_eff≈1.7
  (self-draft, α decays past K=2 because GLM ships **one** MTP layer); clock-gating **~73 %** of
  idle-dynamic gated (`clk_en_ctrl_tb`, a gated-*cycle* fraction); die storage-bound roofline
  ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)).
- **Prior-FP8 measured — NOT Q4_K (Q4_K re-run PENDING):** `weight_decomp` **1.34×** fewer
  NVMe bytes (5.97 bits/sym on FP8 *byte* entropy — the order-0 RTL itself is on `main`, wired
  default-off + release-gated; only the *ratio* is FP8-era) + order-1 `weight_decomp2` ~1.4–1.5×
  (module **DEAD / NOT IN PRODUCT**); the BFP
  accumulator **−87.6 %** cells; the `FLASH_LAT` stall table and the **~4–5× compute-slowdown budget**
  (the FP8 `FLASH_LAT` table). The stall mechanism is **since re-characterized on Q4_K**
  (`make perf-q4k`, 2026-07-11 — see the box below and [`CYCLE_EMULATION.md`](CYCLE_EMULATION.md));
  the compute-slowdown budget and BFP/`weight_decomp2` cell counts remain FP8-datapath results cited
  as prior-track.
- **Modeled (the [EST] part):** the **absolute J/token** = `bytes/token (~25 GB active, ~14 GB routed)
  × energy/bit` with the **NVMe/PCIe read path ≫ DRAM/bit** — a datasheet/roofline model, **not** a
  silicon wattmeter. So the *relative* Q4_K improvements are measured/verified; the *absolute* J/token
  is [EST] until the vendor P&R flow + a board (with the real NVMe drive) give real watts.

> **Note — `flash_xbar` fronts the NVMe/PCIe backend.** `flash_xbar`, `FLASH_LAT`, `flash_req`,
> `flash_seq`, `flash_is_expert`, `flash_expert_id`, … are **committed RTL identifiers** for the
> **storage-read fabric**: a medium-agnostic read-request / latency-hiding crossbar (address →
> weight bytes), unchanged by FP8 → Q4_K. In the product its NAND-specific backend is a labeled
> placeholder — swapped for an **NVMe/PCIe host controller** — while the crossbar abstraction,
> `weight_loader_q4k`, `expert_cache_pf` and `kv_cache_pager` (and the compute die) are **unchanged**.
> The names still read "flash"; the storage medium underneath is **NVMe**. The "more devices → more
> bandwidth" idea survives via **PCIe lanes / multiple NVMe drives** (e.g. Gen3 x4 ~3.5 GB/s, Gen4 x4
> ~7 GB/s; ~100 GB/s needs many lanes/drives — [EST], not a single NAND-die array).

## 4. DVFS — the free, byte-identical compute-power lever

A storage-bound die **should run slow and cool.** Because the token window is mostly storage-stall
(NVMe/PCIe), the compute has a **~4–5× frequency-reduction budget**: drop the die clock (and, at lower
f, the supply voltage) until compute just fills the storage-stall shadow — **zero throughput loss**
(throughput is set by the NVMe stream, not compute). `P_dyn ∝ C·V²·f`, so a 4–5× f cut is a 4–5×
compute-dynamic cut at constant V, and more with V scaling. It is **result-invariant** (same math,
slower clock) — the same "compute is nearly free" slack that shrinks the die
([`MINIATURIZATION.md`](MINIATURIZATION.md)).

> **The stall-shadow mechanism is now measured on Q4_K (2026-07-11, `make perf-q4k`).**
> `test/glm_q4k_system_perf_tb.v` on `glm_q4k_system`, with `EXPERT_STALL=1` and every token held ==
> a standalone `glm_model_q4k` golden, shows exposed `stall/token` scaling linearly with read latency
> (`FLASH_LAT` 8 → 11; 1024 → **2,567** at RESIDENT=0), and — the residency confirmation —
> **35 cyc/token independent of `FLASH_LAT` at RESIDENT=1** (expert refills bypass the storage tier),
> a ~73× cut. The prior FP8 table (`FLASH_LAT` 256→777, 2048→6153) is retained on tag `fp8-verified-baseline` as
> the historical mechanism reference. The *conclusion* it supports — **storage-stall shadow ∝ read
> latency → a ~4–5× DVFS budget** — is a **format-agnostic roofline** property: at real 753B scale the
> die is memory-bound regardless of numeric format, so compute is ≈ 20–25 % of the token window.

**Two halves of DVFS — and only one is RTL.** `P_dyn ∝ C·V²·f`:
- **Frequency (f) — RTL-realized here.** `src/clk_throttle.v` runs the die at effective **f/div** by
  feeding `clk_en_ctrl` a `throttle` term (the die takes one active slot per `div` cycles),
  byte-identical (reuses the proven stall-gate path; `div<=1` = off). This scales **peak power**
  (fewer active edges per unit time) — the knob that lets the USB-C box hold a lower **power envelope /
  thermal cap** (the product plan's "eco mode", [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md) §6).
  **But frequency scaling does NOT cut energy-per-token**: the switching-event *count* is unchanged,
  only spread over more time. Verified: `test/clk_throttle_tb.v` (f/div incl. f/4, f/3, hold-safe;
  passes in `make all`); `clk_en_ctrl_tb` regression identical with `throttle=0`; and `clk_throttle`
  is **BMC-proven** in the formal suite.
- **Voltage (V) — vendor/physical.** The **J/token** win is the `V²` term (lower energy *per* event),
  which needs the vendor flow to actually lower the supply at the reduced f. Not RTL.

So DVFS's contribution is: **peak-power/thermal cap now (RTL, `clk_throttle`)**, and the ~15 %-of-total
**energy** win only once voltage is scaled on the vendor flow. Free in throughput while
`div ≤` the ~4–5× budget.

## 4a. Spec high-K amortization — the ÷K hardware is already built (in Q4_K)

The single biggest output-preserving energy lever is **amortizing one NVMe weight-stream across K
verified tokens**, and its **Q4_K hardware exists and is output-preserving**:

- **`spec_batched_top.v`** ("storage ÷K"): the K+1 verify positions `{cur_tok, d_0..d_{K-1}}` are
  pushed through **one** `glm_model_q4k` as a **PE_M=K+1 batch** — one weight fetch per (layer,
  projection, expert) feeds **all** K+1 rows (the PE_M weight-share contract). So a K+1-position
  verify costs **ONE** model weight-load, not K+1 → **weight-loads ÷ (K+1)** on the dominant ~14 GB
  NVMe term.
- **`spec_chain_top.v`**: mints the K drafts by running the one shipped MTP layer (`mtp_head_q4k`)
  **recurrently** (the P1.3b `h_mtp` chain state), then does the PE_M=K+1 verify and commits the
  **longest accepted prefix**. Committed stream **== greedy rollout, EXACT** (spec==greedy).
- **Verification:** the full `spec==greedy` sims run via **`make spec-slow`** (`spec_batched_top` +
  `spec_chain_top`, both on `glm_model_q4k` / `mtp_head_q4k`) — the committed CI gate (minutes-long in
  iverilog; not re-run to completion in every session). The K=2 spec==greedy exactness is also
  covered fast by `spec_decode_top` (18/18). *(The `glm_model_fp8_pem` "ALL 3 PASSED" weight-share
  corroboration was a **prior-FP8** result and is not restated as a Q4_K number.)*

**So the ÷K amortization is done in RTL, in Q4_K.** The only thing between K_eff≈1.7 (built, self-draft
chain — acceptance decays because GLM ships **one** MTP layer) and K_eff 3–5 (the low-J/token floor) is
**draft acceptance α** — raised by a separate resident ~1–3 B dense draft model. That is a
**model-quality / artifact** step, **not** an RTL one. (And note the measured union cap: the realized
NVMe amortization is **A/U(K) ≈ 1.1–1.3× at K=4** — [`H_MEASUREMENT.md`](H_MEASUREMENT.md) — so higher
K_eff must also beat U(K).)

*(Updated 2026-07 — **adaptive draft depth is ADOPTED + RTL-landed**, commit 6c5332f: `spec_decode_seq`
gains an `ADAPT` param (default 0 — yosys sequential-equivalence PROVEN unchanged for existing
consumers) + a `k_cur` port and `pass_*` taps, driven by the new `src/spec_depth_adapt.v`
saturating-streak policy — **output-invariant by construction** (spec==greedy for ANY depth schedule),
so it stays on the output-preserving floor. GLM-U K-sweep: at r=0.9 tok/s plateaus at **K=4–5** (K>5
adds nothing), at r=0.8 the optimum is K=2–3 → adaptive range **K∈[1..5]**. Gates: `spec_depth_adapt`
31,522; `spec_decode_seq`(K>1) 3,702 (K 1/2/3/4/6/8); K=1 exact 621; `spec_chain_top` 4/4 incl. a new
DRAFT_K=4 engine; `spec_batched_top` 8/8; `spec_decode_top` 18/18; new `make spec-adapt` Makefile gate.
The accept rate r has since been **measured** (job B, vLLM MTP sweep on GLM-4.5-Air: r₁=0.87 with
per-position decay 0.87/0.60/0.32/0.13/0.04, A_eff plateau ~2.9 → memory-bound optimum **K=1–2**,
residency-box design point ≈80 tok/s [measured-inputs EST] —
[`H_MEASUREMENT.md`](H_MEASUREMENT.md) 3rd measurement; the adaptive controller settles at that
optimum on its own).)*

## 5. Already-built power wins (compute + idle)

- **Idle-die clock gating**: **~73 % of idle-dynamic power gated** (a measured gated-*cycle*
  fraction, `clk_en_ctrl_tb`), sim-verified never to gate an advancing cluster. **Mechanism, as
  actually shipped:** the production top `glm_q4k_system` gates the *entire compute die* with an
  inline glitch-free ICG — `die_clk = clk & die_en_lat`, the enable latched on the low phase of
  `clk` (`glm_q4k_system.v:1307-1311`, the `icg_cell.v` pattern hand-coded), driving `u_model`
  (`:618 .clk(die_clk)`). It freezes the die on each aw-beat weight-stream stall and on expert-cache
  demand miss; default (`LOOPBACK=EXPERT_STALL=0`) → `die_clk===clk` (byte-identical). `clk_en_ctrl`
  is the **standalone cluster-level model** that produced the ~73 % fraction (via `clk_en_ctrl_tb`
  and `clk_gate_cluster`), NOT a block instantiated in the production hierarchy — the shipping gate
  is the inline `die_clk`. This is **format-agnostic** (gates on the weight stream, FP8/Q4_K alike).
  Finer-grain gating (matmul accumulator banks, the model's activation/logit buffers, the decoder
  residual/FFN accumulators, the attention datapath banks) is left to **synthesis-inferred ICG** on
  their enable-qualified writes — the correct flow, not hand-instantiated cells. In `make all` today.
- **Die-shrink L0/L1**: compact config + `swiglu_expert_q4k` engine-share (6→4 GEMM/block) →
  area-proportional static+dynamic ↓, output-preserving ([`MINIATURIZATION.md`](MINIATURIZATION.md)).
- **Q4_K weight transport**: weights are **4-bit codes** (÷2 the bytes of the FP8 E4M3 track for the
  same param), dequantized on-chip to bf16 (`d`/`dmin` + 6-bit scales); activations stay **bf16** (no
  activation quant) with **fp32 accumulate**. The 4-bit storage/transport *is* the ~1.6× byte/energy
  win already counted in §1; the bf16 MAC is cheaper than an fp32-weight MAC but **not** silicon-
  characterized.

> **Prior-FP8 compute-energy results [tag `fp8-verified-baseline`, Q4_K re-characterization PENDING]:** the **BFP
> fixed-point accumulator** measured **−87.6 % cells** vs fp32-accumulate, and **FP8 E4M3 (4×4
> mantissa)** freed DSPs at far less energy/MAC than fp32. Both are **FP8-datapath** results — the
> Q4_K path uses bf16 activations + fp32 accumulate (a different arithmetic contract) and has **no
> BFP module in-tree**, so these figures are **not** carried over as Q4_K numbers.

These are real but bounded — they act on the small, mostly-idle compute slice.

## 6. Explicitly OFF the table (per the output-preserving decision)

Bigger single-box wins exist but **change the output** — deferred unless §3's floor misses the target,
then revisited as a separate fidelity decision:

- **Dynamic top-k expert pruning** (k_eff<8): ÷1.3–1.6× fewer routed bytes — *cheapest big lever*,
  low effort ([`ULTRA_PERF.md`](ULTRA_PERF.md) #7). **NOT output-preserving.**
- **Contextual SwiGLU activation sparsity**: fetch only active W_up cols / W_down rows, ÷1.5–3×
  (#6). **NOT output-preserving.**
- **Lossy sub-4-bit (below Q4_K) for cold experts**: large, but a fidelity trade.

## 7. Honest gaps

- **Every J/token here is [EST]** — a `bytes × energy/bit` roofline, **not** a silicon/P&R wattmeter.
- **No real watt number exists** — it needs the vendor power flow on the routed netlist plus a board
  with a real NVMe drive. (The FPGA fit + Fmax are now **MEASURED** — Vivado ML 2026.1 on XCKU3P,
  142,320 LUT / 87.5 % synth-stage (routed 141,298 LUT), routed Fmax 46.5 MHz, campaign closed route-dominated — but board bring-up is
  not done; [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) / README.)
- **The ~73 % gating** is a measured *cycle* fraction, not a power meter.
- **Assembled end-to-end Q4_K numeric golden: DONE** (`make model-q4k`, 1155 bit-exact vs our own
  numpy reimpl `tools/glm_model_q4k_ref.py`) — but every lever's output-preservation is still checked
  against our own references (spec==greedy self-consistency + that golden + `q4k_ref.py`), **not** vs
  the real GGUF / llama.cpp (external validation OPEN; README *What's proven*).
- **spec high-K:** the ÷K *hardware* is **built + output-preserving in Q4_K** (§4a); what is not built
  is the **resident ~1–3 B dense draft model** that raises K_eff 1.7→3–5 — a model artifact, not RTL.
- **Prior-FP8 numbers** (compression 1.34×/1.4–1.5×, BFP −87.6 % cells, the `FLASH_LAT` stall table
  and its ~4–5× compute-slowdown budget) are **tag `fp8-verified-baseline`** measurements; the Q4_K re-runs are
  [PENDING] and none of them is restated as a Q4_K result.

## 8. Verification invariant

Every output-preserving lever is a *result-invariant restructuring*: the system TB token must stay
`== standalone glm_model_q4k` (the same gate as the MoE union-skip; spec==greedy for the spec levers).
DVFS, die-shrink, gating, MTP and batching all pass it today; spec high-K passes it via `make
spec-slow`. Crucially this invariant is **self-consistency** (DUT-vs-DUT — the "greedy golden" is
itself a `glm_model_q4k`), **not** a check against a real GLM-5.2 golden — the assembled Q4_K numeric
path is now golden-checked end-to-end vs our own numpy reimpl (`make model-q4k`, 1155 bit-exact), and
the dequant layer is sealed against the **real GGUF bytes** (376,586,240 weights bitwise-equal vs
llama.cpp's kernels — `docs/GGUF_CROSSCHECK.md`); llama.cpp *whole-runtime* numeric equality remains
out-of-contract by design (op orders differ), and the 467 GB checkpoint has not been run end-to-end. Power is optimized **without
ever moving the decoded token relative to the reference.**

## Status
- **Energy budget + DVFS budget: characterized** (this doc; DVFS mechanism carried from the prior-FP8
  perf harness, real-scale [EST]; Q4_K perf re-run PENDING).
- **Built (Q4_K):** `flash_xbar` (BMC-proven fabric), MTP K=2 (`spec_decode_top` 18/18), union-skip,
  clock-gating ~73 % (`clk_en_ctrl`), die-shrink L0/L1 (`swiglu_expert_q4k`), the spec high-K ÷K-weight-
  load hardware (`spec_batched_top` / `spec_chain_top`, spec==greedy via `make spec-slow`), **and the
  DVFS/eco frequency prescaler** (`clk_throttle` → `clk_en_ctrl.throttle`, f/div peak-power cap,
  byte-identical, BMC-proven, `clk_throttle_tb` in `make all`).
- **Prior-FP8 measurements, NOT carried to Q4_K:** the `weight_decomp` lossless-compression ratio
  (1.34×; the order-0 RTL itself is on `main`, default-off + release-gated) and the ~1.4–1.5× of the
  **DEAD / NOT IN PRODUCT** `weight_decomp2` (Q4_K is already 4-bit), the BFP accumulator (−87.6 %
  cells), FP8 E4M3 MAC, and the `FLASH_LAT` stall table. Q4_K re-characterization [PENDING].
- **Near ceiling / gated:** higher spec K_eff needs a **resident ~1–3 B draft model** (artifact, not
  RTL); DDR5 self-refresh is a **PHY/vendor** feature; the DVFS **voltage** (J/token) half and the real
  watt number are **vendor flow** + a board.
- **The J/token *efficiency* levers are therefore built or model/vendor-gated;** the RTL-doable
  remainder was the peak-power prescaler (done) + this measured-vs-modeled firming (done).
