# Cycle-accurate emulation — measuring the throughput mechanism on real RTL cycles

Moves the perf story off the pure `[EST]` roofline in one specific, defensible way: the
**memory-stall mechanism the roofline assumes is measured on real RTL cycles** by running the
*assembled* system top `glm_q4k_system` (compute die `glm_model_q4k` + `expert_cache_pf` +
`kv_cache_pager` + storage-read arbiter (`flash_xbar`) + `ddr5_xbar` + `weight_loader_q4k`) under a
cycle-accurate testbench with the storage read-latency model (NVMe/PCIe) engaged, sweeping the
knobs, and counting cycles.

> **Status / provenance (read this first).** The cycle-emulation *instrumentation* is **current and
> Q4_K**: `src/glm_q4k_system.v` carries the `EXPERT_STALL` param and the `ec_hit_count` /
> `ec_miss_count` / `ec_demand_stall_cycles` counters used throughout this doc. **The harness is now
> ported to Q4_K and re-run** (2026-07-11): `test/glm_q4k_system_perf_tb.v` + `tools/perf_sweep.sh`
> drive `glm_q4k_system` + a standalone `glm_model_q4k` reference (every committed token bit-exact,
> not just timed), gated by `make perf-q4k`. The **measured Q4_K sweep is in [§Measured — Q4_K](#measured-sweep--q4k-make-perf-q4k)
> below**; the prior FP8 table is retained beneath it as the historical mechanism reference. Key
> Q4_K result: the compute-die baseline is **~10,896 cyc/token** (slice), and the **residency pivot
> is confirmed on real cycles** — at `FLASH_LAT=1024`, `RESIDENT=0` exposes **2,567 stall
> cyc/token** (19%) while `RESIDENT=1` exposes **35** (0.3%), a ~73× cut, matching the mechanism
> (exposed stall ∝ latency × miss count). The FP8 numbers further below remain FP8-die-specific.

> **Wall-clock update (2026-07 — routed Fmax now MEASURED).** The XCKU3P routed-Fmax campaign is
> closed at **46.5 MHz** ([`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md)), so the slice demo's wall clock
> is now computable from the table below: `EFF_CYC` ≈ 8.0–11.0 K (FP8-era absolute number; the Q4_K
> die is a similar ballpark plus a few hundred cycles from the Fmax-campaign repipeline latencies)
> → **~170–240 µs/token ≈ ~4,200–5,800 *slice* tok/s**. That is the correctness-demo speed of the
> tiny slice, **not** a GLM-5.2 product number — never conflate the two.

> **Storage note (one-time).** The `flash_xbar` / `FLASH_LAT` storage-read fabric and its latency
> model are **medium-agnostic** (address → weight bytes, with read-request issue and latency
> hiding). In the product this fabric fronts an **NVMe/PCIe (M.2) backend** — the model store is
> an **NVMe SSD (1–4 TB)**, tiered **NVMe (bulk, slow) → fast DDR (hot set; DDR4 rung-1 / DDR5 or HBM rung-2, see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) → die** — and
> the NAND-specific PHY is swapped for an NVMe host controller. The committed RTL keeps the
> `flash_*` identifiers (`flash_xbar`, `FLASH_LAT`, `flash_req`, `flash_seq`, …); below, a "Flash"
> in a param / sweep-label / PERF-output string is that committed name, while the storage *concept*
> it models is NVMe/PCIe read latency and bandwidth. *(2026-07 design-point update: on the primary
> rung-③ **residency** box the whole ~467 GB checkpoint resides in 512 GB LPDDR5X (~1.1 TB/s
> on-package) and the single M.2 NVMe is boot-load only — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md);
> the NVMe-streaming latency model here remains the live mechanism for the rung-① FPGA demo, the
> hybrid upside SKU, and >512 GB checkpoints.)*

This is the software-emulation core of "FPGA emulation" — **measured cycles**, not a board run.
It does **not** produce an absolute real-753B tok/s (that needs full-scale weights / a real FPGA);
it validates the *mechanism and scaling* the roofline rests on.

The throughput measured here is the product's own number: the **LOCAL, SINGLE-USER box** running the
full GLM-5.2 753B (Q4_K / UD-Q4_K_XL, ~467 GB) locally at **B=1** (one box, one user — see
`docs/USBC_PRODUCT_PLAN.md`). `cyc_per_tok` / `EFF_CYC` are single-stream per-token latencies, so
single-user tok/s = clock ÷ `EFF_CYC`. (The same silicon's aggregate/datacenter-batch throughput is
a separate, non-target analysis, not this product's speed.)

## Harness

- **Instrumentation (current, Q4_K — `src/glm_q4k_system.v`).** Overridable params `FLASH_LAT`,
  `DDR_NCH`, `CACHE_SLOTS` (plus `L` / `N_EXPERT` to size a larger cache-thrashing config and
  `EXPERT_STALL` — see §Faithful integration), wired into both the DUT and the TB storage-read
  (NVMe/PCIe) latency responder; the DUT exposes the counters `ec_demand_stall_cycles` /
  `ec_hit_count` / `ec_miss_count` read by the perf line.
- **Driver (prior FP8 track — tag `fp8-verified-baseline`).** `test/glm_fp8_system_perf_tb.v` — the functional
  system perf TB (`token == standalone glm_model_fp8`, X-aware) with TB-level `*_CFG` overrides
  (`FLASH_LAT_CFG`, `DDR_NCH_CFG`, `CACHE_SLOTS_CFG`, `L_CFG`, `N_EXPERT_CFG`, `EXPERT_STALL_CFG`),
  a free-running cycle counter (`cyc_per_tok` = start → `tok_valid`), and a machine-readable
  `PERF …` line reading the DUT counters. `tools/perf_sweep.sh` compiles once per config via
  `iverilog -P glm_fp8_system_perf_tb.<P>=<v>`, runs, and tabulates. **The equivalent Q4_K perf TB
  (against `glm_q4k_system` / `glm_model_q4k`) is since PORTED** (`make perf-q4k`, 2026-07-11) — see
  the measured Q4_K sweep above.

**On the prior FP8 harness the functional binding held at every valid config** (`ALL 3 TESTS
PASSED`, token == standalone reference) — so every cycle number below is from a *correct-token*
run. The FLASH sweep was **independently re-run** (two separate runs produced byte-identical
numbers).

## Measured sweep — Q4_K (`make perf-q4k`)

Measured 2026-07-11 by `test/glm_q4k_system_perf_tb.v` on `glm_q4k_system` (compute die
`glm_model_q4k`, every committed token bit-exact vs a standalone `glm_model_q4k` reference), 4-token
decode, slice config (L=4, n_expert=8, cache_slots=2, ddr_nch=4). `cyc_per_tok` is the slice
compute-die baseline, **not** a GLM-5.2 product number.

| FLASH_LAT | RESIDENT | EXPERT_STALL | cyc/token | stall/token | compute/token | exposed stall |
|---|---|---|---|---|---|---|
| 8 | 0 | 1 | 10,899 | 11 | 10,888 | 0.1% |
| 1024 | 0 | 1 | 13,463 | 2,567 | 10,896 | **19%** |
| 8 | 1 | 1 | 10,931 | 35 | 10,896 | 0.3% |
| 1024 | 1 | 1 | 10,931 | 35 | 10,896 | **0.3%** |
| 1024 | 0 | 0 (no stall model) | 10,896 | 2,467* | 8,428 | — (thrash: 32 dropped) |

**What this proves on real Q4_K cycles:** (1) the compute-die slice baseline is **~10,896
cyc/token**; (2) exposed storage stall is **linear in `FLASH_LAT`** (11→2,567 as latency 8→1024 at
RESIDENT=0) and **∝ miss count** — the roofline's memory-bound assumption, now measured not assumed;
(3) the **residency pivot is confirmed**: RESIDENT=1 makes stall independent of `FLASH_LAT` (35
cyc/token at both 8 and 1024) because expert refills no longer traverse the storage tier — a ~73×
cut vs RESIDENT=0 at FLASH_LAT=1024. *(The last row uses the no-stall accounting mode where misses
drop rather than stall; its stall column is the counter, not exposed latency.)* Independently re-run
(byte-identical). See `make perf-q4k` and `build/perf_q4k_sweep.log`.

## Measured sweep — prior FP8 track (tag `fp8-verified-baseline`), historical mechanism reference

> **These are prior-FP8 cycle counts**, produced by the fp8-branch `tools/perf_sweep.sh` on
> `glm_fp8_system` + `glm_model_fp8`. Retained because the *mechanism* (stall linear in `FLASH_LAT`,
> ∝ miss count) is format-agnostic — the Q4_K table above now confirms it directly. The absolute
> FP8 `cyc_per_tok` differs from the Q4_K baseline; do **not** read these as Q4_K.

`cyc_per_tok` = cold-token compute latency (start→tok_valid). `stall` = `ec_demand_stall_cycles`
(cycles the cache/storage subsystem was in a demand-miss refill) over 3 tokens. **`EFF_CYC` =
`cyc_per_tok + stall`** — the *integrated* latency (see §Integration). `MEM%` = `stall / EFF_CYC`.

| sweep | FLASH_LAT | DDR_NCH | CACHE_SLOTS | cyc/tok | stall | **EFF_CYC** | **MEM%** | hit | miss |
|---|---|---|---|---|---|---|---|---|---|
| FLASH | 8 | 4 | 4 | 7947 | 33 | **7980** | 0.4% | 93 | 3 |
| FLASH | 64 | 4 | 4 | 7947 | 201 | **8148** | 2.4% | 93 | 3 |
| FLASH | 256 | 4 | 4 | 7947 | 777 | **8724** | 8.9% | 93 | 3 |
| FLASH | 1024 | 4 | 4 | 7947 | 3081 | **11028** | 27.9% | 93 | 3 |
| DDRNCH | 256 | 1 | 4 | — | — | — | — | — | — (degenerate: NCH must be ≥2; TB fabric-liveness fails) |
| DDRNCH | 256 | 2 | 4 | 7947 | 777 | 8724 | 8.9% | 93 | 3 |
| SLOTS | 256 | 4 | 2 | 7947 | 1554 | **9501** | 16.3% | 90 | 6 |
| SLOTS | 256 | 4 | 4 | 7947 | 777 | 8724 | 8.9% | 93 | 3 |

## Three measured findings (mechanism — format-agnostic)

These are the format-agnostic takeaways; the numbers are the prior-FP8 run above, but the mechanism
holds regardless of quant format and rests on RTL (`expert_cache_pf`, `EXPERT_STALL`) that is
current and Q4_K on `main`.

1. **The memory-stall mechanism is exactly the roofline's, measured on real cycles.** `stall`
   rises strictly linearly: `stall = 3·FLASH_LAT + 9`, where the slope **3 = the number of
   demand-miss storage fetches** across the 3 tokens. This *is* the roofline term *exposed stall =
   (#misses) × (storage read latency)* — no longer assumed, but counted on the RTL.
2. **Stall ∝ miss count (cache-capacity is a real, measured signal).** Halving `CACHE_SLOTS`
   (4→2) raised misses 3→6 and **doubled** the stall (777→1554) → `EFF_CYC` 8724→9501, MEM%
   8.9→16.4. The cache size moves the exposed stall exactly as the miss count moves.
3. **Effective latency is memory-bound as latency grows.** With the exposed stall integrated,
   the memory fraction climbs 0.4% → 2.4% → 8.9% → **27.9%** as FLASH_LAT goes 8 → 1024 — at the
   *slice's* tiny miss count. At real scale (§Extrapolation) it dominates.

## Integration — why EFF_CYC, and the honest topology note

Raw `cyc_per_tok` is **FLASH_LAT-invariant (flat 7947 on the prior FP8 run)** — by construction, not
because the knob is dead. In the TB, the compute die `u_model` receives expert weights
*combinationally* from the weight-source stub, and the host FSM waits only on `mdl_done`; the
`expert_cache_pf` that accumulates `ec_demand_stall_cycles` sits on a **parallel observer path** that
models the storage-refill (NVMe/PCIe) stall but does not gate `mdl_done`. So `cyc_per_tok` measures
pure compute latency and `stall` measures the refill cost *concurrently*.

**Real hardware cannot compute a missed expert's GEMM until its weights arrive from the NVMe SSD.**
The faithful cost is therefore **compute + the *exposed* (un-prefetch-hidden) demand-stall** — exactly
`ec_demand_stall_cycles`, which counts only demand misses the prefetcher did *not* hide. Hence
**`EFF_CYC = cyc_per_tok + stall`** is the honest integrated latency.

### Faithful integration (`EXPERT_STALL`) — measured directly, not just modeled

`glm_q4k_system` has an **`EXPERT_STALL`** parameter (default **0** = byte-identical to the
committed system) that makes the model faithful: it **clock-gates the compute die**
(`glm_model_q4k`) for exactly the cycles `expert_cache_pf` holds `ec_busy` — i.e. every cycle a
demand-miss is being serviced by the NVMe/storage backend — using the same glitch-free
negedge-latched clock gate the C8 loopback path already proves bit-exact (the cache / FIFO /
`flash_xbar` storage-read arbiter keep running on the ungated clock, so the fetch always completes —
no deadlock). With it enabled, **`cyc_per_tok` itself GROWS with `FLASH_LAT`** as a direct
measurement (no longer flat) while the token stays byte-identical. This param and its clock-gate are
current on `main` (`src/glm_q4k_system.v`); the numeric demonstration below is from the prior FP8
harness.

**Verified on the prior FP8 perf harness** (tag `fp8-verified-baseline`; perf TB, `EXPERT_STALL=1`,
`FLASH_LAT=256`): `ALL 3 TESTS PASSED` (token `== standalone glm_model_fp8`) + `PERF flash_lat=256 …
cyc_per_tok=8607 stall=777 … expert_stall=1` — i.e. `cyc_per_tok` rose from the flat **7947** to
**8607** (the exposed demand-stall now inside the measured token window). The growth **equals the
independently-counted exposed demand-stall** (a second counter, `in_window_stall`, agrees with the
cache's cumulative `ec_demand_stall`), so the number is measured, not fabricated. This upgrades
`EFF_CYC` from a modeling statement to a **directly-measured faithful token latency**. **These are
prior-FP8 cycle counts; the Q4_K re-run on `glm_q4k_system` (which retains the identical
`EXPERT_STALL` gate and `in_window_stall` counter) is since DONE** (`make perf-q4k`). (The default `EXPERT_STALL=0`
keeps the committed system byte-identical — regression `ALL 3 TESTS PASSED`.)

## Extrapolation to real scale (measured mechanism → projection)

The slice has **miss = 3 per 3 tokens** (tiny weights → almost everything fits the cache). At the
real config the per-token miss count is large: ~75 MoE layers × ~8 routed experts × `(1 − hit)`.
With the GLM-trace hit rate `h ≈ 27%` (measured on the FP8 prior track, `make cache-study` [removed
from `main` — see tag `fp8-verified-baseline`] / [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md)), that is ~**hundreds
of demand misses per token**. (**Update — newer proxy measurement:** [`H_MEASUREMENT.md`](H_MEASUREMENT.md),
OLMoE-1B-7B-Instruct trace — U(K)/EOR have since been **GLM-family measured** on GLM-4.5-Air
([`H_MEASUREMENT.md`](H_MEASUREMENT.md) 2nd measurement), superseding the OLMoE first-pass —
bandwidth-h is cache-size-dependent: **0.36–0.60** with
a 20 % expert pool cached (~90 GB at GLM scale), **0.72–0.88** at 50 % (~225 GB), collapsing to ~0
below 10 % under LRU; see also [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md). Read `h` from
that measurement; the mechanism and formula here are unchanged. **2026-07 pivot:** on the primary
rung-③ **residency** box h=1 by construction — the whole checkpoint sits in 512 GB LPDDR5X
([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)) — so this h-driven miss extrapolation applies to the
rung-① / hybrid-SKU streaming regime.) Using the **measured** per-miss cost (each miss exposes ≈ `FLASH_LAT`
cycles, from finding 1) and the **measured** hit rate:

```
real exposed stall / token  ≈  (layers · experts · (1−h)) · FLASH_LAT_exposed   ≫  compute
```

so `EFF_CYC` becomes **memory-dominated** — the slice's 0.4–28% memory fraction is a *floor*, and
real scale pushes it toward the roofline's ~75–80% NVMe/PCIe-bound regime. This is a
**measured-mechanism projection** (per-miss cost and hit rate are measured on the prior FP8 track;
only the miss *count* is scaled by the known config), which is stronger than the pure first-order
roofline. The hit rate `h` and per-miss cost are themselves format-agnostic (they are a function of
routing entropy and storage latency, not the weight format), so they carry into the Q4_K product.

## What this does / does not establish

- **Establishes (measured, prior FP8 top):** the memory-stall term is linear in storage read
  latency and proportional to miss count on real RTL cycles; the assembled system produces a
  *correct* token while these hold; the integrated `EFF_CYC` decomposition (compute vs exposed
  storage/NVMe stall). The RTL that produces these signals (`expert_cache_pf`, `EXPERT_STALL`) is
  current and Q4_K on `main`.
- **Does NOT establish:** an absolute real-753B tok/s (needs full-scale weights / a real FPGA run);
  the Q4_K product's own absolute real-753B `cyc_per_tok` (the SLICE `cyc_per_tok` ≈ 10,896 IS
  measured — `make perf-q4k` — but full-scale is not); achievable-vs-peak NVMe/PCIe
  bandwidth on silicon; or overlap efficiency at production widths. Those remain the FPGA-prototype
  step ([`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md)).

All of this is at the committed **slice** dims. The value is the *mechanism and scaling*, not the
absolute magnitude — see the extrapolation for how it maps to the real regime.

## Reproduce

**Current branch (Q4_K).** The cycle-emulation *instrumentation* is in `src/glm_q4k_system.v` — the
`EXPERT_STALL` clock-gate and the `FLASH_LAT` / `CACHE_SLOTS` / `DDR_NCH` knobs feeding the
`ec_hit_count` / `ec_miss_count` / `ec_demand_stall_cycles` counters. The Q4_K perf TB + sweep that
drive them into a `PERF …` table are **since ported** (`make perf-q4k`).

**Prior FP8 harness (tag `fp8-verified-baseline`)** — the exact commands that produced the table above:

```sh
git checkout fp8-verified-baseline
bash tools/perf_sweep.sh             # faithful (EXPERT_STALL=1) SLICE + SCALE(thrashing cache) + decoupled BASELINE
SWEEP=full bash tools/perf_sweep.sh  # additionally sweeps DDR_NCH and CACHE_SLOTS
# single point:
iverilog -g2012 -I src -P glm_fp8_system_perf_tb.FLASH_LAT_CFG=1024 \
    -o build/perf test/glm_fp8_system_perf_tb.v <glm_fp8_system_sim source list from the Makefile>
vvp build/perf     # -> ALL 3 TESTS PASSED + PERF flash_lat=1024 ... stall=3081 ...
```
