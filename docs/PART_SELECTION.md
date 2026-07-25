# Part selection — the minimum info to lock the rung-① / ② board BOM

> **This is the current Q4_K product's part-selection bridge.** The part-selection *logic*
> below (fit → FPGA device → DDR/NVMe/power) is **format-agnostic** and current. **UPDATE:
> the Q4_K routed fit is MEASURED** — Vivado ML 2026.1 synth + full place&route of
> `glm_q4k_system_cdc` (compact config + ACT_HW=1) on **XCKU3P**: **142,320 LUT (87.5%),
> ~100 K FF, 421 DSP, 0 BRAM**, hold met, routed **Fmax 46.5 MHz** (bit-exact repipeline
> campaign 10.2 → 17.2 → 46.5 MHz, closed — worst path now route-dominated; see
> [`../fpga/README.md`](../fpga/README.md)). The older FP8-era figures kept below are
> historical scale indicators only (tag `fp8-verified-baseline` / tag `fp8-verified-baseline`). Do not
> read any FP8 utilization figure below as a Q4_K result. Current-track context:
> [`README.md`](../README.md), [`Q4K_RETARGET.md`](Q4K_RETARGET.md),
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md). RTL/test names of the form `*_fp8` map to their
> `*_q4k` equivalents on `main` (`glm_matmul_fp8`→`glm_matmul_q4k`,
> `glm_fp8_system_cdc`→`glm_q4k_system_cdc`).

*How the physical parts (FPGA, DDR4, NVMe, power) get **confirmed** for the offline
single-user 753B-param box (GLM-5.2, ~467 GB at UD-Q4_K_XL). This is the bridge between the
FPGA-fit measurement and a real PCB design/BOM.* Companions: [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)
(why performance = bandwidth = the silicon's IO/PHY), [`BOM.md`](BOM.md) (per-rung cost),
[`MINIATURIZATION.md`](MINIATURIZATION.md) (the compact config + the fit-measurement plan),
[`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md).

> **Nothing here is a committed *part* choice yet — but the gating measurement has landed.**
> The vendor place-and-route fit of the **Q4_K** die is **MEASURED** (XCKU3P, 87.5% LUT,
> routed 46.5 MHz — see above); the DDR/NVMe/power numbers below remain **[DERIVED]**
> targets from the bandwidth roofline until distributor/board quotes lock them.

---

## The single gating input: the FPGA-fit measurement

Part selection cannot start from a guess. **FPGA size = fit** — and the Q4_K fit is now
**measured** (step 1 below, DONE). The order is fixed:

```
1. Vendor flow measures fit for the compact config
     (PE_N / DDR_NCH / KV_RESIDENT / EFIFO_DEPTH / CACHE_SLOTS, all result-invariant)
       -> real LUT / DSP / BSRAM / URAM utilization           [MEASURED for Q4_K -- XCKU3P]
2. Smallest FPGA that holds that utilization + >=30% headroom  -> FPGA device locked
3. That device's I/O banks + transceivers set DDR4 channel count + PCIe lanes
4. DDR4 (BW + capacity) and NVMe (capacity + PCIe gen) follow from steps 2-3
5. Power tree (rails x max current) follows mechanically from the three datasheets
```

**Step 1 has produced Q4_K numbers — downstream selection is unblocked.** The
compact config is set by the result-invariant knobs (`PE_N` / `DDR_NCH` / `KV_RESIDENT` /
`EFIFO_DEPTH` / `CACHE_SLOTS`) on `glm_q4k_system_cdc`; beyond the yosys elaboration
(`make synth-glm`), the Vivado flow has now **synthesized and fully placed-and-routed** it on
XCKU3P: **142,320 LUT (87.5%), ~100 K FF, 421 DSP, 0 BRAM**, hold met, routed **46.5 MHz**.
Against the ≥30%-headroom rule, 87.5% means the compact config *fits* KU3P (demo-viable) but a
product board points one size up (KU5P-class) or at config trims [DERIVED]. (The *full* 753B
shape remains elaboration-only.)

---

## 1. FPGA candidates

Repo-designated class: **low-end Kintex UltraScale+ (KU3P-class)**. Hard requirements: a
**multi-channel DDR4 interface**, a **PCIe hard block** (for NVMe), and enough fabric +
DSP + block RAM to hold the **Q4_K compute die** at the compact config.

| Candidate | Role | DDR4 channels (approx) | PCIe | Note |
|---|---|---|---|---|
| **Gowin GW5AT-138** (Tang Mega 138K) | **(retired)** | DDR3 only, 1 GB soldered | Gen3 x8 (1 core, ~8 GB/s) | **Not a part-selection target** — the Gowin/nextpnr fit scaffold is **removed**, superseded by the Vivado/XCKU3P flow; cannot hold 467 GB, no DDR4/NVMe, so it can't run the ①b streaming demo. |
| **Kintex US+ KU3P** (XCKU3P) | if fit is small | HP-bank limited, ~1–2 ch | Gen3/4 hard block | the repo-designated minimum class |
| **KU5P / KU11P / KU15P** | if fit is larger or needs 3–4 ch DDR4 | 2–4 ch | Gen3/4 | more banks/resources + headroom, higher cost |

**Decision rule:** the ①a fit number picks among KU3P / KU5P / KU11P / KU15P. The **Q4_K fit is
measured**: the compact config routes on **XCKU3P at 87.5% LUT** (46.5 MHz) — demo-viable on
KU3P; a product board wanting ≥30% headroom points at KU5P-class or config trims [DERIVED].

> ①a (fit) vs ①b (demo): the ①a fit is a synthesis/P&R resource count — **measured** on the
> Vivado/XCKU3P flow (the retired Gowin scaffold was an earlier stand-in). But the ①b demo
> (offline, bit-exact, real tokens streamed from NVMe) needs DDR4 + NVMe on a physical
> KU3P-class board — **board bring-up is still open**.
> See [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md).

## 2. DDR4 — bandwidth + capacity

- **Bandwidth target.** Every token reads the **~11 GB hot-set touch** from DDR (resident hot
  partition ~17 GB; canonical byte constants: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2 —
  the earlier "~19 GB / ~28 GB raw" figures here were FP8-era counts; the bit-exact levers
  `weight_decomp` + MLA weight-absorption reduce the touched bytes further ~1.3–1.5×),
  while the **~14 GB of routed experts** (top-8, *change every token*) stream from NVMe/PCIe.
  `tok/s ≈ DDR_BW / hot_footprint`, capped by `NVMe_BW / routed_footprint`
  ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md); all tok/s **[EST]**).
  - DDR4-3200 = **25.6 GB/s per 64-bit channel**.
  - 3 channels ≈ **~77 GB/s**  ·  4 channels ≈ **~102 GB/s** (the doc's ~100 GB/s rung-① target).
- **Capacity.** The ~17 GB resident hot partition must be **resident** → **≥ 24 GB**; leave room for an
  expert cache → **32–48 GB** (routing entropy caps expert-cache benefit for bit-exact — see
  the ledger — so the cache trades capacity for a bounded hit-rate, not a guaranteed speedup).
  - e.g. **16 GB × 3 ch = 48 GB** (~77 GB/s)  or  **8 GB × 4 ch = 32 GB** (~102 GB/s).
- **Form.** On a custom board, mount **DDR4 BGA devices directly** (better SI than SO-DIMM);
  a 64-bit channel = 4× x16 devices. Length-matched, impedance-controlled routing.

*(This is the tier — DDR4-many-channels — that rung ② may swap for DDR5-fewer-channels or
HBM to reach ~15–40 tok/s (measured-proxy design points: ~13–47 —
[`H_MEASUREMENT.md`](H_MEASUREMENT.md); the ~54–127 @225 GB-cache band now belongs to the
rung-③ hybrid SKU, the rung-③ primary being full residency at ≈80 tok/s [measured-inputs EST] —
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)); same bit-exact Q4_K RTL, only the bandwidth
changes.)*

## 3. NVMe — the model store

- **Interface:** PCIe **Gen3 x4 (~3.5 GB/s)** or **Gen4 x4 (~7 GB/s)** M.2 connector.
- **Capacity:** rung-① reduced-config image → **1 TB** is plenty; the full **467 GB**
  UD-Q4_K_XL target fits in **1 TB** with room to spare, **1–2 TB** for KV overflow / headroom.
- **Constraint:** one FPGA PCIe hard block = one NVMe. Multiple drives (rung ②, striping the
  routed-expert stream) need bifurcation or a PCIe switch + more lanes.

## 4. Power tree — designed on the custom board

| Rail | For | Approx |
|---|---|---|
| VCCINT | FPGA core | 0.85–0.9 V, **high current** (scales with die size) |
| VCCAUX / VCCBRAM | FPGA aux | 1.8 V / 0.9 V |
| MGTAVCC / MGTAVTT | transceivers (PCIe) | 0.9 V / 1.2 V |
| VCCO (DDR banks) | DDR4 I/O | 1.2 V |
| DDR4 VDD / VPP / VTT | memory | 1.2 / 2.5 / 0.6 V (VDDQ/2 tracking) |
| NVMe | M.2 | 3.3 V |

→ a **multi-rail PMIC + several DC-DCs**. Power integrity (decoupling, planes) is, with
DDR routing, the core of the layout difficulty.

---

## Minimum-info checklist to LOCK each part

| Part | Minimum info required to confirm | Source |
|---|---|---|
| **FPGA** | LUT/DSP/BSRAM/URAM utilization + ≥30% headroom → smallest device; DDR4 channels + PCIe lanes = bank/transceiver budget | **①a Q4_K fit measurement — [MEASURED] (XCKU3P: 87.5% LUT, 421 DSP, 46.5 MHz)** |
| **DDR4** | target tok/s → required BW → channels × speed; resident hot partition ~17 GB → capacity ≥ 24–32 GB | roofline ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) |
| **NVMe** | reduced-config image size → capacity; Gen3 vs Gen4 → bandwidth | FPGA PCIe generation |
| **Power** | after the three above: each rail's voltage × max current → PMIC selection | FPGA/DDR/NVMe datasheets |

## Fit-measurement status (what is measured vs still blocked)

> **All utilization numbers in this section are from the PRIOR FP8 datapath** (tag `fp8-verified-baseline` /
> tag `fp8-verified-baseline`) — the only routed/fabric fit data that exists. They are **NOT
> Q4_K results.** The Q4_K datapath (4-bit codes + 6-bit sub-scales + fp16 `d`/`dmin`, bf16
> activations, fp32 accumulate) has a **different multiply/dequant structure** than FP8 e4m3,
> so these FP8 counts indicate *scale only*, not a Q4_K prediction. **The Q4_K routed fit is
> now MEASURED (Vivado ML 2026.1 on XCKU3P — see the DONE block and table below).**

**[PRIOR FP8] Measured — per compute unit (open flow, DSP-inferred)** — `glm_matmul_fp8` maps to
**~17.8 K LUT4-equiv + 20 DSP (MULT18X18/MULT9X9) + ~5.4 K DFF** (yosys 0.66 `synth_gowin`
with DSP inference; the `abc -lut4` path that "timed out" was a tool limitation, not a
design one — see [`../fpga/README.md`](../fpga/README.md)). *On `main` this module is
`glm_matmul_q4k`; it is bit-exact to the ggml Q4_K reference in sim (`make q4k`), but has
**not** been fit-measured on any device.*

**[PRIOR FP8] Measured — compact whole-system coarse demand (open flow, pre-map)** — `glm_fp8_system_cdc`
at the compact config (PE_N=2, DDR_NCH=2, KV_RESIDENT=8, EFIFO_DEPTH=8, CACHE_SLOTS=2):

| Coarse cell | Count | Reads as |
|---|---|---|
| `$mul` (multipliers) | **346** | multiply ops = **DSP demand ceiling** (small FP8 mants pack; not 1 DSP each — the Q4_K dequant-multiply will pack differently) |
| `$mem_v2` (memories) | **54** | BSRAM demand (count; sizes set the block total) |
| `$dff` (registers) | 4004 cells (multi-bit) | FF demand |
| `$add` / `$sub` / `$mux` | 7558 / 2063 / 112657 | LUT/carry logic demand |

*On `main` the whole-chip top is `glm_q4k_system_cdc`; it **elaborates** cleanly
(`make synth-glm`: yosys `hierarchy -check` + `check -assert`, 0 unresolved), and its
**routed resource count is now MEASURED** (Vivado/XCKU3P — see the DONE block below).*

**Which device to compare against — NOT GW5AST-138.** The Gowin **GW5AST-138** (Tang Mega,
138,240 LUT4 / 298 DSP) is **DDR3-only, no NVMe** → it **cannot be the product board** (the ①b
streaming demo needs DDR4 + NVMe). It was only ever a *fabric-fit curiosity*, and comparing 346
mul ops to its 298 DSP ("DSP tight") is measuring against a **disqualified device** — an artifact,
not a real constraint. The real target is a **KU3P-class part (DDR4-capable)** with **~1,368 DSP48E2**
(XCKU3P — confirm vs datasheet), several × the Gowin part → on the actual target **DSP is not the
binding resource** (this device-comparison reasoning is format-agnostic; the Q4_K multiply/dequant
count will differ from the FP8 346, but not by a factor that changes the KU3P-class conclusion).

**DONE — the authoritative routed fit of the Q4_K die on the real (KU3P-class) device is MEASURED**
(Vivado ML 2026.1, XCKU3P): synth + full place&route of `glm_q4k_system_cdc` (compact config +
ACT_HW=1) → **142,320 LUT (87.5%), ~100 K FF, 421 DSP, 0 BRAM**, hold met; routed **Fmax
10.2 → 17.2 → 46.5 MHz** across a bit-exact repipeline campaign (every round re-proven on the
1155-test assembled golden), **closed at 4.6×** — the worst remaining path is route-dominated
(wide-bus wiring at 87% utilization): physical work, not arithmetic. 46.5 MHz sits in the bring-up
demo's target band; 200 MHz-class is rung-2/3 work, and the stage decompositions carry to the ASIC
unchanged. The old Gowin/nextpnr scaffold is **removed** (superseded by this Vivado flow). See
[`../fpga/README.md`](../fpga/README.md).

| Field | Value | Status |
|---|---|---|
| Per-unit fit (`glm_matmul_fp8`, Gowin DSP-inferred) | ~17.8K LUT4-eq + 20 DSP + 5.4K DFF | **[PRIOR FP8, MEASURED]** — indicative scale only (wrong-device fabric; not Q4_K) |
| Whole-system multiply demand (`glm_fp8_system_cdc`, device-agnostic) | 346 `$mul` ops (compact) | **[PRIOR FP8, MEASURED, coarse]** |
| Whole-system memory demand (`glm_fp8_system_cdc`, device-agnostic) | 54 arrays (compact) | **[PRIOR FP8, MEASURED, coarse]** |
| Routed LUT / DSP / BSRAM of the **Q4_K** die on a **KU3P-class** part | 142,320 LUT (87.5%) / 421 DSP / 0 BRAM (+ ~100 K FF) on XCKU3P | **[MEASURED — Vivado ML 2026.1]** |
| Routed Fmax (Q4_K) | **46.5 MHz** (10.2 → 17.2 → 46.5; campaign closed — worst path route-dominated) | **[MEASURED]** |
| → FPGA device chosen (KU3P / 5P / 11P class) | compact config fits KU3P at 87.5% (demo-viable); ≥30%-headroom product rule → KU5P-class or config trims | [DERIVED — lock pending quotes] |
| → DDR4 channels × size | — | [PENDING — unblocked by the fit] |
| → NVMe (gen × capacity) | — | [PENDING — unblocked by the fit] |
| → power rails / PMIC | — | [PENDING — unblocked by the fit] |

> **Correction of record:** an earlier draft treated GW5AST-138's 298 DSP as the fit budget and
> read "DSP tight." That device is DDR4-disqualified; the device-agnostic demand (346 mul / 54 mem,
> *measured on the prior FP8 die*) is comfortable on the KU3P-class target. The authoritative routed
> fit of the **Q4_K** die has since been **measured** with Vivado on XCKU3P (421 DSP used of 1,368 —
> DSP indeed not the binding resource; LUT is, at 87.5%).

---

**Bottom line.** The prerequisite **Q4_K fit measurement (①a) has landed** (XCKU3P: 87.5% LUT,
421 DSP, 0 BRAM, routed 46.5 MHz). The FPGA device class is bounded (KU3P holds the compact
config for the demo; ≥30% product headroom points at KU5P-class or trims), the DDR4
channels/capacity follow from the roofline, and NVMe + power fall out of those datasheets
mechanically — which is exactly the input a PCB schematic/layout needs to begin.
