# GLM-5.2 Q4_K accelerator — performance / power improvement plan

The governing model (from `docs/SYSTEM_SINGLE_PACKAGE.md` §7):

```
  tokens/s  ≈   NVMe_BW   /  [ (1 − h) × footprint ]   ×   K
  J/token   ≈   bytes_moved × energy/bit   (NVMe read bytes dominate, ~24–26× DRAM/bit [EST])
```

For UD-Q4_K_XL the per-token `footprint` is **~25 GB** (~40B active params × ~0.6 B/param), and the
**wall** is the **~14 GB of routed experts** (top-8, they change every token) streamed from NVMe/Flash;
the ~11 GB per-token hot-set touch of attention/dense/shared weights caches in DDR (canonical byte constants: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2). So every lever is one of: **raise
NVMe_BW**, **raise hit-rate h**, **shrink footprint**, **raise K (speculative)**, or **cut
bytes_moved** (which helps both tok/s and J/token). The compute die is *not* the bottleneck
(~20–25 % utilized) — die-side optimization (accumulator area, fmax, formal) improved
cost/thermals/correctness but does **not** move tok/s or J/token until the NVMe/storage tier is
unblocked. This plan targets the real bottleneck. **All tok/s figures below are [EST]** (roofline-
modeled) until a running board exists — board bring-up is still open, but the **Vivado routed fit +
Fmax are now MEASURED** (Vivado ML 2026.1 on XCKU3P, compact config + ACT_HW=1: 141,298 LUT routed (142,320 / 87.5 % synth-stage)
post-synthesis, 141,298 LUT routed — `fpga/results/util_routed_ku3p_acthw1.rpt`;
421 DSP, 0 BRAM; routed Fmax **46.5 MHz** after a 4.6× repipeline campaign, every round re-proven
bit-exact on the 1155-test assembled golden; campaign CLOSED — the worst path is route-dominated,
physical work not arithmetic).

> **Status update — most of the old correctness gaps are now CLOSED.** The assembled Q4_K end-to-end
> numeric golden is **DONE** (`make model-q4k` — 1155 tests bit-exact vs the numpy reference
> `tools/glm_model_q4k_ref.py`, plus `make model-q4k-acthw` through the ACT_HW=1 serialized-activation
> datapath); **mixed-type Q6_K/Q8_0/F16 consumers are DONE** (`make mixedtype`), so a real UD-Q4_K_XL
> checkpoint's dynamic type mix can be consumed; and the MLA `1/√qk_head_dim` softmax scale is now
> **applied**. **CLOSED (2026-07-10/11): bit-exactness vs the *real* published GGUF bytes** —
> `tools/gguf_crosscheck.py` fed two real published GGUFs' raw block bytes to `q4k_ref` and to
> llama.cpp's own `dequantize_row_q4_K/_q6_K/_q8_0` (built libggml, direct-linked):
> **376,586,240 weights (Q4_K + Q6_K + Q8_0), all bitwise-equal** — so by transitivity the RTL
> dequant ≡ the real files' ([`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)). Still out-of-contract:
> llama.cpp *whole-runtime* numeric equality (attention/accumulation orders differ by design), and
> the 467 GB checkpoint has not been run end-to-end — tracked in `README.md`.

**The memory-stall mechanism this plan rests on is emulatable on real RTL cycles.** `glm_q4k_system`
carries an `EXPERT_STALL` parameter (default 0 = byte-identical to the committed system) that
clock-gates the compute die for exactly the cycles a demand-miss is being serviced by the
NVMe/storage backend, so `cyc_per_tok` **grows with `FLASH_LAT`** as a direct count — exposed
demand-stall `= 3·FLASH_LAT + 9`, slope = miss count — while the committed token stays identical
(control math is format-agnostic). The published cycle sweep (flat 7947 → 8724 @ `FLASH_LAT=256`) is
the **prior FP8-track measurement** (tag `fp8-verified-baseline`). The sweep is **since re-run on Q4_K**
(2026-07-11, `make perf-q4k` — `test/glm_q4k_system_perf_tb.v`): slice `cyc_per_tok` ≈ **10,896**,
exposed demand-stall linear in `FLASH_LAT` (11 @ 8 → 2,567 @ 1024, RESIDENT=0), and the residency
pivot confirmed on real cycles (RESIDENT=1 → 35 cyc/token regardless of latency, ~73× cut) —
see [`CYCLE_EMULATION.md`](CYCLE_EMULATION.md). This upgrades the roofline's stall term from
*assumed* to *counted*, on the Q4_K die. (With the measured 46.5 MHz routed Fmax the *slice* demo wall-clock is now
computable: cyc_per_tok ~8.0–11.0 K → **~170–240 µs/token ≈ ~4,200–5,800 slice tok/s** — the
correctness-demo speed of the tiny slice, **not** a GLM-5.2 product number.)

Baseline (measured h, [EST] BW): NVMe ~50 GB/s [EST] (aggregate across many PCIe lanes/drives — a
single Gen4 x4 NVMe ~7 GB/s), h≈27 % (measured, see 2.3), K=1 → **~2–3 tok/s single-user [EST]**,
~8–10 J/token [EST].

> **Update — h/U are now proxy-MEASURED ([`H_MEASUREMENT.md`](H_MEASUREMENT.md), OLMoE-1B-7B trace;
> U(K) since GLM-Air-measured — see the 2026-07 note below):** residency-only bandwidth-h ≈ **0.36–0.60** with ~20 % of the expert pool cached
> (~90 GB at GLM scale), **0.72–0.88** at ~50 % (~225 GB); LRU collapses to ~0 below a 10 % cache.
> Measured-roofline design points (all [EST], MEASURED-PROXY h/U inputs): 1–2 NVMe (no multipliers)
> ~0.5–1 tok/s; 90 GB DRAM + 100 GB/s → 13–24; 90 GB + 200 GB/s (ONFI 64ch) → 25–47; 225 GB +
> 200 GB/s → **54–127** (the "100 tok/s" design point). See also
> [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md).
>
> *(Updated 2026-07:* **(i)** U(K) is now **GLM-family measured** — GLM-4.5-Air MoE-gate trace:
> U(2)=1.60–1.64, U(4)=2.60–2.71, U(8)=4.19–4.41, EOR 0.36–0.39 — superseding the OLMoE first-pass
> ([`H_MEASUREMENT.md`](H_MEASUREMENT.md) 2nd measurement). **(ii)** The **primary rung-③ design point
> pivoted to FULL RESIDENCY** ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)): 512 GB LPDDR5X
> (~1.1 TB/s on-package) holds the whole 467 GB checkpoint → **h=1 by construction** (h is no longer a
> product-deciding variable there), design point **≈80 tok/s [measured-inputs EST]** (accept rate r measured, job B); the ONFI-64ch streaming tier
> is deleted from the primary SKU. The h-curve/ONFI design points above stay TRUE for **rung ① / the
> hybrid upside SKU / >512 GB checkpoints**, and **54–127** survives only as the
> hybrid-SKU-if-h≥0.75 note.)*

Legend: 🟢 RTL-doable in this repo · 🟡 system/architecture (design + vendor IP) · 🔴 out of RTL scope.

---

## P1 — NVMe/storage bandwidth (the linear lever; biggest single win)

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 1.1 | **`flash_xbar`** — N-channel banked storage-read fabric | 🟢 | Same pattern as `ddr5_xbar`: stripe expert fetches across N back-end channels → ~N× aggregate NVMe_BW. `flash_xbar` is a committed RTL identifier: the medium-agnostic storage-read fabric (address → weight bytes); in the product its NAND back-end is a labeled placeholder swapped for an **NVMe/PCIe host controller**, so the N channels map to **PCIe lanes / multiple NVMe drives**, not NAND dies. A 1–4 TB NVMe store reaches "10s of GB/s" [EST] by striping reads across lanes/drives (one Gen4 x4 NVMe ~7 GB/s; scale with lanes/drives). **Built + BMC-proven** (flash_xbar is in the formal controller set — see [`FORMAL.md`](FORMAL.md)): the standalone fabric adds a deep per-channel outstanding queue (QDEPTH) that hides storage read latency (Little's law, BW = outstanding/latency; ~8× on the fabric in isolation at QDEPTH≈FLASH_LAT [EST]) + N_CH banking. **⚠️ RESCOPED (2026-07-22) — the die-feedback half is DONE and gated:** all **five die weight-input families (aw / fw / rw / lw / gn)** are loopback-proven through the banked `ddr5_xbar` — the die's weight bytes routed OUT as fabric reads and back IN, committed stream bit-exact, each with a corruption-injection build that must FAIL (`make loopback` / `loopback-fw` / `loopback-rest`, all in `release-gate`) — and the weight-loader's PE_N≤16 lane cap is root-caused and fixed (loader→GEMM bit-exact through PE_N=64; `make weight-loader-lanes` gates PE_N=32 — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)'s weight-path note). **Still open:** integrating the FLASH/storage tier (`flash_xbar`) into that proven path, and the Vivado fit of the banked fabric. | **~N× tok/s** [EST] (linear) — DDR-side die feedback loopback-proven; flash-tier integration + fabric Vivado fit still open |
| 1.2 | **NVMe expert layout** — co-activated experts on different channels | 🟢 | Offline placement so a token's top-8 experts spread across channels/drives (avoid channel hotspots), mirroring the DDR5 stripe. **Built: `tools/flash_layout.py` (balanced greedy least-conflict packer).** Measured N_CH=8 (format-agnostic — depends only on routing, not on the weight format): optimized 55 % of 8× peak BW vs naive round-robin 39 % (~+40 %) — kills the 4/5/6-on-one-channel collision tail (99.5 % of fetches ≤2/ch). Honest: 8× is unreachable (top-8 ≥ 8ch pigeonhole + popularity skew); the win is removing hotspots, not reaching peak. 8ch is the top-8 sweet spot. The pigeonhole cap and the sub-expert-striping (RAID-0) option that removes it are in [`FLASH_STRIPING.md`](FLASH_STRIPING.md). | **~+40 %** [EST] of flash_xbar's realizable BW (sustains 1.1) |
| 1.3 | **Deeper NVMe read pipeline** — more outstanding requests | 🟢 | Raise the fetch queue depth so the NVMe/storage tier stays saturated despite per-read latency (BW = outstanding / latency). Wire into `expert_cache_pf` + `flash_xbar`. | recovers the latency-bound gap to peak BW |
| 1.4 | Faster NVMe medium (PCIe5 NVMe / more drives / more lanes) | 🟡 | Vendor/board choice; the NVMe host controller / PCIe root is vendor IP. The bandwidth a board can actually feed is **rung-dependent** — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md): the near-term prove-it rung is a low-end FPGA dev board + DDR4 + one NVMe; the funded product is a custom board (mid FPGA + DDR5-multi-channel **or** HBM + multi-NVMe over more PCIe lanes). DDR5 is the **rung-2** memory, not a hard-asserted single spec. Document the BW target the RTL fabric must feed. | linear, but $ + board |

## P2 — Cut bytes moved (raises tok/s AND lowers J/token — the dual win)

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 2.1 | **Expert decompressor** in the NVMe→DDR5 path | 🟢 | Store expert weights losslessly-compressed on the NVMe SSD and decompress on-chip during fetch → cut NVMe bytes/expert → effective NVMe_BW ↑ and NVMe read energy ↓ by the same factor. A `weight_decomp` unit (order-0 canonical-Huffman, bit-exact lossless) is **wired into the production `glm_q4k_system` on main** behind the default-off `DECOMP` parameter (DECOMP=0 byte-identical to the pre-DECOMP module — `src/glm_q4k_system.v` §DECOMP) and **release-gated** (`make weight-decomp` round-trip golden + `decomp1-elab`). **⚠️ Ratio [PENDING] for Q4_K:** the **1.34× (5.97 bits/sym; 8 tests)** is an **FP8-era measurement (tag `fp8-verified-baseline`, E4M3 byte stream)**, where raw FP8 weight *bytes* have entropy well under 8 bits. **Q4_K does not inherit that number**: a Q4_K super-block is *already* ~4.5 bpw block-entropy-packed (fp16 `d`/`dmin` + 6-bit scales + 4-bit codes), so the lossless headroom left for a second Huffman pass is **much smaller and unmeasured** — [EST] at best until re-measured on real Q4_K blocks. **Measure the real Q4_K ratio** before claiming any win. Do **not** relabel the FP8 1.34× as a Q4_K result. | **[PENDING]** for Q4_K (prior FP8 track: ~1.34×) |
| 2.2 | **MTP K>1 / better draft** — verify more tokens per weight-load | 🟢 | Extend `spec_decode_seq` to a K>1 multi-token draft (longest-accepted-prefix). **Built + gated: `DRAFT_K>1` (g_kn batch verifier), spec==greedy** on the Q4_K model (`spec_decode_top` 18/18; K=1 byte-identical; `spec_batched_top` / `spec_chain_top` via `make spec-slow`). Eff tok/pass under a chained-decay acceptance model [EST] (the shipped 1-MTP-layer reality) at α=0.7: K=1→1.69, K=2→2.08, K=3→2.17. HONEST: the shipped model has 1 MTP layer, so K>1 must chain the head autoregressively (acceptance decays) → **K=2 is the sweet spot**; K=3+ is marginal unless a deeper MTP stack lands. Realized in RTL by `spec_chain_top`: the MTP head runs recurrently on chain hidden-state `h_mtp` to mint K drafts, then a `PE_M=K+1` batched-verify in one weight-load commits the accepted prefix (spec==greedy; self-draft K_eff ~1.7–2.2 [EST]). **Measured U(K) correction ([`H_MEASUREMENT.md`](H_MEASUREMENT.md)):** the K positions' expert *union* grows with K (proxy-measured U(2)=1.51–1.65, U(4)=2.25–2.64), so the realized NVMe-byte amortization is **A/U(K) ≈ 1.1–1.3× at K=4 (A≈3)** — read the spec multiplier through that cap, not as ×K_eff. *(Updated 2026-07: U(K) is now GLM-family measured — GLM-4.5-Air: U(2)=1.60–1.64, U(4)=2.60–2.71, U(8)=4.19–4.41 — superseding the OLMoE proxy values; and **adaptive spec-chain is ADOPTED + RTL-landed** (commit 6c5332f): the GLM-U K-sweep shows the r=0.9 optimum plateaus at **K=4–5** (K>5 adds nothing) and r=0.8 at K=2–3 → adaptive range **K∈[1..5]** via `spec_decode_seq` `ADAPT` param (default 0 — yosys sequential-equivalence proven unchanged) + `k_cur`/`pass_*` taps + new `src/spec_depth_adapt.v` saturating-streak policy; output-invariant (spec==greedy for ANY depth schedule); gates incl. `spec_depth_adapt` 31,522, `spec_decode_seq`(K>1) 3,702 (K 1/2/3/4/6/8), `spec_chain_top` 4/4 incl. a new DRAFT_K=4 engine, `spec_decode_top` 18/18, and the new `make spec-adapt` gate. Accept rate r has since been **measured** (job B vLLM MTP sweep, GLM-4.5-Air: r₁=0.87 with per-position decay 0.87/0.60/0.32/0.13/0.04, A_eff plateau ~2.9 → memory-bound optimum **K=1–2**, residency-box design point ≈80 tok/s [measured-inputs EST] — [`H_MEASUREMENT.md`](H_MEASUREMENT.md) 3rd measurement; the adaptive controller settles at that optimum on its own).)* | **K=2 ≈ +23 %** [EST] over K=1 (chained, fixed-K model); adaptive K∈[1..5] landed; NVMe-byte win capped by measured U(K) |
| 2.3 | **Higher cache hit-rate h** — bigger cache + predictor-driven prefetch | 🟢 | Wire `expert_predictor` into `expert_cache_pf` prefetch. **❌ MEASURED NO-OP at GLM cache size (`expert_prefetch_top`, format-agnostic):** at SLOTS=900 (> reuse distance) the predictor hints the most-popular expert, which LRU **already keeps resident** → 0 prefetches issued, hit-rate byte-identical to baseline; deep LOOKAHEAD 1/2/3 identical. Only helps in the narrow regime just *below* the reuse-distance knee (SLOTS=550: 0 %→4.6 %). **Conclusion: hit-rate is capped by fine-grained-routing entropy, not prefetch cleverness — this lever does not move the real config.** (Matches the ledger: predictor-prefetch is a measured no-op.) | **~0 %** at real cache size (honest) |
| 2.4 | **Union-skip grouped MoE** (batch axis) — fetch only the union of selected experts | 🟢 | The PE_M>1 grouped MoE in `glm_decoder_block_q4k` scans the expert axis (`T_ESCAN`) and fetches **only** the union of experts any of the B rows selected (combinational `any_has` membership), not all N_EXPERT. **Union-skip is folded INLINE** into `glm_decoder_block_q4k` (the `T_ESCAN` scan + `any_has` cursor mirror the now-removed standalone `batched_moe.v` reference pattern). The PE_M batch-widen enabler is DONE across swiglu/router/mla/mtp (`_q4k`). *(The full B-coverage cross-check `make bcov` — B∈{1,2,3,5,8} × routing {same,distinct,random,overlap}, batched(PE_M=B) == B per-row runs bit-exact, union fetched once — was a **prior FP8-track** gate, removed from `main`; see tag `fp8-verified-baseline`.)* On the single-user box the batch axis is exercised only by speculative decode's small `PE_M=K+1` self-verify batch (`spec_chain_top`, B=1 per user); the large-batch B≈256 case is the **non-target datacenter-serving** regime (kept as analysis, not this product). | up to **~32×** fewer NVMe expert fetches at small batch (the single-user spec-decode verify batch) [EST]; ~0 at B≈256 (union≈all — the non-target datacenter-batch regime) |

## P3 — Hide latency / raise utilization

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 3.1 | **Predictor-driven deep prefetch loop** | 🟢 | Use `expert_predictor` confidence to prefetch ahead of the demand cache (L+2..) so the NVMe/storage tier stays saturated and the die stalls less; double/triple-buffer experts. Note the tension with 2.3: at the real cache size the popular experts are already resident, so the payoff is in keeping NVMe *busy* (latency hiding), not raising h. | sustains P1/P2 (keeps NVMe busy) |
| 3.2 | **Idle-die clock gating** (`clk_en_ctrl`) | 🟢 | The die idles ~75 % waiting on NVMe/storage; gate the compute lanes during fetch stalls + boot. **Built: work-driven clock-ENABLE controller** (`clk_en_ctrl.v`; synthesis infers the ICG cell), same-cycle wake + hysteresis, **formally-safe** (never gates an advancing cluster — ICG is in the verified building-block set, see [`FORMAL.md`](FORMAL.md)). Gated-cycle fraction ≈ 73.75 % [EST] — this is ~ the idle fraction of the ~25 %-duty NVMe/storage-bound die minus a 4-cycle wake margin, an arithmetic consequence of the duty cycle (format-agnostic); a Q4_K-config re-measure is folded into the pending `EXPERT_STALL` sweep. | **~74 %** [EST] of compute idle-dynamic power gated |

## P4 — Energy-specific (J/token)

> **Full low-power design + the bit-exact roadmap to ~1.5–3 J/token [EST] is in
> [`LOW_POWER.md`](LOW_POWER.md)** (energy is ~80 % NVMe/storage read bytes → amortize the fetch; DVFS on the
> 75–80 %-idle die is a free, byte-identical compute-power lever; spec high-K is the staged floor-setter).

The NVMe/storage read byte movement is ~80 % of per-token energy. P2 (decompress, MTP, hit-rate) is the main
energy lever — it directly cuts NVMe bytes. P3.2 (clock gating) trims idle. Beyond RTL:

| # | Item | Type | Note |
|---|---|---|---|
| 4.1 | HBM instead of DDR5 (if energy ≫ cost) | 🟡 | HBM is the lower-energy-per-bit fast tier; the DDR5 choice trades energy for cost. A build-time / **rung-2** board option (DDR5-multi-channel vs HBM — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); HBM stacks are also what anchor the **rung-3** ASIC's on-package memory. (Note the 467 GB Q4_K checkpoint does **not** fit in HBM (≤192 GB) — HBM is the fast-tier cache for the hot-set, not the whole model. *2026-07 pivot: the primary rung-3 design point instead achieves **full residency** with 512 GB LPDDR5X, ~1.1 TB/s on-package — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).*) |
| 4.2 | Computational storage / near-NVMe compute | 🔴 | Moves compute to the data to avoid moving bytes — out of RTL scope (RTL can't add IO pins/PHY or near-memory silicon). This **near-memory compute** is exactly a **rung-3 SoC/ASIC** capability (HBM stacks + many-channel PHY + near-memory low-precision at ~TB/s) that breaks the FPGA's IO/PHY ceiling once amortized over volume — the endgame, not a dead-end (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) rung ③). *(2026-07 pivot: the primary rung-③ SKU is now full-residency 512 GB LPDDR5X — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md); near-memory compute stays the endgame for the hybrid/streaming SKU and >512 GB checkpoints.)* |

---

## Projected combined effect (single-user, [EST])

Stacking the 🟢 RTL items on the baseline (~2–3 tok/s, ~8–10 J/token). **Every row is [EST]** — the
built RTL items are verified *as RTL* here (and the Vivado routed fit + 46.5 MHz Fmax are now
MEASURED — see above), but the absolute tok/s awaits a running board; the assembled Q4_K numeric path
now has its 1155-test end-to-end golden (`make model-q4k`, vs our own numpy reimpl; the dequant layer
beneath it is since **PROVEN bit-exact on real published GGUF bytes** —
[`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md) — while llama.cpp *whole-runtime* equality stays
out-of-contract):

| Step | Lever | tok/s | J/token | status |
|---|---|---|---|---|
| baseline | — | ~2–3 | ~9 | — |
| + flash_xbar Nch (1.1) | NVMe_BW ×N | ~×N | ~9 | built + BMC-proven; DDR-side die feedback loopback-proven through the banked `ddr5_xbar` (all five weight families, in `release-gate` 2026-07-22); **flash-tier integration + fabric Vivado fit still open** |
| ~~+ expert decompress (2.1)~~ | ~~bytes ÷~~ | — | — | wired on main (default-off `DECOMP`, release-gated); **Q4_K ratio [PENDING]** (prior FP8 track measured 1.34×; Q4_K already block-packed → headroom unknown) |
| + MTP K_eff ~1.7–2.1 (2.2) | ÷ traffic | +~20–25 % | ↓ | built + spec==greedy (1 MTP layer → K=2 sweet spot) |
| ~~+ hit-rate (2.3)~~ | ~~(1−h)↓~~ | — | — | ❌ measured NO-OP at real cache |
| + clock gating (3.2) | idle ↓ | (tok/s ~flat) | ↓ | built, enable-side RTL (ICG vendor); ~74 % idle-power gated [EST] |

**Honest summary (post-measurement):** the real single-user gains come from **NVMe/storage bandwidth
(P1)** plus **fewer verified weight-loads (MTP K>1, P2.2)** — *not* from cache cleverness (2.3 is a
measured no-op) and *not*, on Q4_K, from a further expert decompressor (2.1's 1.34× was FP8-specific;
Q4_K is already ~4.5 bpw block-packed, so its headroom is unmeasured/PENDING). The dominant P1 win is
**partly realized** (2026-07-22): the banked `ddr5_xbar` feeds the die's five weight-input families in
proven, release-gated loopback; the flash/storage tier and the banked fabric's Vivado fit remain open.
All numbers [EST].

> **These are RTL-lever multipliers on the storage roofline — not a hardware headline.** The absolute
> tok/s each multiplier lands on is set by the board's **memory bandwidth**, which is **rung-dependent**
> (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). Staged to the ladder: **rung ① (near-term prove-it —
> low-end FPGA + DDR4 + 1 NVMe) ~5–8 tok/s [EST]**, the honest cheap demo; **rung ② (funded custom board
> — DDR5-multi-channel / HBM + multi-NVMe) ~15–40 tok/s [EST]**, where the lever-stacked figures here
> live; **rung ③ (volume SoC/ASIC — HBM + near-memory compute) ~40+ tok/s [EST]**. "Built today" means
> the RTL is built + verified *as RTL* — **not** that rung-② speed is reachable on rung-① cheap
> hardware. (The assembled Q4_K numeric path now **has** its end-to-end golden — `make model-q4k`,
> 1155 bit-exact vs our own numpy reimpl.) The same
> Q4_K RTL runs on every rung; only the bandwidth it is fed changes. (Q4_K's GEMM core is bit-exact to
> `tools/q4k_ref.py` — and `q4k_ref` itself is since **PROVEN bit-identical to llama.cpp's own dequant
> on real published GGUF bytes**, 376,586,240 weights, [`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md);
> llama.cpp *whole-runtime* equality stays out-of-contract.) Clock↔area at
> the measured fit: compute-side stream consumption = dequant lanes × clock — at the measured
> 46.5 MHz, 7 GB/s (1 NVMe) needs ~300 lanes and 100 GB/s ~4,300 (infeasible on KU3P); ~1,000 at
> 200 MHz-class; ~200 at ASIC 1 GHz+. A higher clock buys a smaller/cheaper die, **not** more tok/s
> (memory stays the wall past saturation).

## Execution order (RTL, by impact-per-effort)

1. **`flash_xbar` integration** (P1.1) — biggest single win; the fabric is built + BMC-proven, and
   the DDR-side die feedback is now loopback-proven through the banked `ddr5_xbar` (all five weight
   families + the PE_N lane-cap fix, in `release-gate` as of 2026-07-22); the remaining work is the
   flash/storage-tier integration into that proven path and Vivado-fitting the banked fabric.
2. **MTP K>1** (P2.2) — extend the speculative loop, keep spec==greedy exact; K=2 sweet spot on the
   shipped 1-MTP-layer model. *(2026-07: DONE one better — adaptive depth K∈[1..5] landed
   (`spec_depth_adapt` + `spec_decode_seq` `ADAPT`, `make spec-adapt`); the measured accept sweep
   (job B) then put the memory-bound optimum at **K=1–2** (A_eff plateaus ~2.9 while U keeps
   growing) — the adaptive controller settles there on its own; K≤5 stays as headroom for
   GLM-5.2's deeper-trained MTP — [`H_MEASUREMENT.md`](H_MEASUREMENT.md).)*
3. **predictor-driven deep prefetch** (P3.1) — wire the built predictor in for latency-hiding (not for
   hit-rate: 2.3 is a measured no-op at the real cache size), measure NVMe occupancy.
4. **idle clock-gating** (P3.2) — power, low-risk (built; re-measure fraction under the Q4_K config).
5. **Q4_K expert decompressor** (P2.1) — *only if* a real Q4_K-block compressibility measurement shows
   headroom; `weight_decomp` is already wired on main (default-off `DECOMP`, release-gated) but its
   1.34× is an FP8-era measurement that does **not** transfer to already-packed Q4_K. Measure first.
6. Document the 🟡 system items (NVMe layout, faster medium, HBM option) as build/board choices.
