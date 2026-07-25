# Chip miniaturization — plan (Q4_K die; whole-die fit MEASURED on Vivado/XCKU3P — per-lever deltas still prior-FP8)

> **What this doc is.** A die-**minimization study** for the current **Q4_K** compute die. The *analysis*
> (an NVMe/PCIe-bandwidth-bound die should be minimal; the "compute is nearly free" serialization budget;
> the lever catalog) is **format-agnostic** and carries over to the Q4_K die unchanged. What does **not**
> carry over is the *per-lever measured area*: every concrete LUT/cell delta below (the L1
> ~12K-LUT4 saving, the yosys `stat` deltas) was measured on the **prior FP8 die** (preserved on branch
> **`fp8`**). The **whole-die Q4_K fit is now MEASURED** on the vendor flow — **Vivado ML 2026.1**,
> real synth + full place&route of `glm_q4k_system_cdc` on **XCKU3P** (compact config + ACT_HW=1):
> **142,320 LUT (87.5 %) synth-stage** (routed **141,298 LUT**,
> `fpga/results/util_routed_ku3p_acthw1.rpt`), **~100K FF, 421 DSP, 0 BRAM**, hold met, routed Fmax **46.5 MHz** after a
> closed bit-exact repipelining campaign (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md); the old
> Gowin/nextpnr scaffold is removed). The **per-lever** Q4_K deltas are now *measurable* on that flow
> but have not been re-run — the per-lever figures are still presented **as prior-FP8 measurements,
> never relabeled as Q4_K**. The FP8-only doc that carried the raw `stat` output (`PPA_FP8.md`) now
> lives on tag `fp8-verified-baseline`.
>
> RTL/test names of the form `*_fp8` in the prior-track measurements below map to their `*_q4k`
> equivalents on `main` (`glm_matmul_fp8`→`glm_matmul_q4k`, `swiglu_expert_fp8`→`swiglu_expert_q4k`,
> `glm_model_fp8`→`glm_model_q4k`). The current-track numerics are in [`Q4K_RETARGET.md`](Q4K_RETARGET.md) /
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md).

> **Scope note (FPGA-fit now; rung-③ ASIC groundwork later).** See the hardware ladder
> ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) for the staged framing. The near-term product path is
> an **FPGA card** ([`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) P3.2), so "shrink the die" here means
> **fit a smaller / cheaper FPGA** (fewer LUT/DSP/BSRAM) on the **prove-it/custom-board rungs ①–②** —
> *not* a tapeout today. But this die-minimization is **not a dead-end**: the same by-construction area
> cuts are the **die-area groundwork for the rung-③ ASIC**, where a smaller die is exactly what lowers
> **$/unit + power at manufacturing volume** (the ASIC is the *endgame*, not out of scope — it is
> sequenced after the FPGA proves PMF). The analysis stays valid — every lever is correctness-invariant
> (byte-identical token) — and the *study itself* is **deprioritized behind a green P1** (real-model
> fidelity + full-scale correctness): compute is NVMe/PCIe-starved and already cheap, so shrinking it
> buys cost/power headroom, not throughput. The vendor flow (E1) is **now available** — the Vivado
> XCKU3P fit measured the whole die at **87.5 % LUT**, which gives the shrink levers a concrete
> motivation (headroom is tight).

How to make the Q4_K compute die dramatically smaller (for a smaller / cheaper FPGA on rungs ①–②, and
lower $/unit + power for the rung-③ ASIC at volume — i.e. a cheaper, cooler **local single-user box**),
ranked and phased. Grounded in the architecture's defining property. (Every lever cuts BOM/power; **none
change the product's single-user interactive tok/s** — that speed is set by memory bandwidth and is
therefore *rung-dependent* [~5–8 rung ① / ~15–40 rung ② / **≈80 rung ③** (measured-inputs; updated 2026-07 — the
rung-③ primary design point is now full residency, 512 GB LPDDR5X on-package; see
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)), all **[EST]**; see
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md), and the measured-proxy design-point menu in
[`H_MEASUREMENT.md`](H_MEASUREMENT.md) — the streaming points there now apply to rung ① / the
hybrid upside SKU] — the die is NVMe/PCIe-bandwidth-bound on rungs ①–②, so compute is
nearly free.)

## Thesis — an NVMe/PCIe-bandwidth-bound die should be *minimal*

The workload is **NVMe/PCIe-bandwidth-bound**: the die sits ~75–80 % idle behind the NVMe→DDR
expert stream ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md), [`ULTRA_PERF.md`](ULTRA_PERF.md); the DDR
tier is rung-dependent — DDR4 on rung ①, DDR5/HBM on rung ②, per
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). So
**compute speed is nearly free** — you can make the die much *slower* (more serial, more shared)
with **zero throughput loss**, up to the point where compute time exceeds the exposed NVMe/PCIe stall.
*(Measured corollary of the same thesis — the clock↔area trade: compute-side stream consumption =
dequant lanes × clock, so at the routed 46.5 MHz ~300 lanes cover a 1-NVMe 7 GB/s stream, ~1,000 at
200 MHz-class, ~200 at ASIC 1 GHz+ — a higher clock buys a **smaller/cheaper die, not higher tok/s**;
memory stays the wall past saturation.)*
Yet the die still carries **parallel / duplicated compute hardware sized for a throughput the
NVMe/PCIe-starved workload cannot use**: after the L1 within-module merge (below), the remaining
duplication is **cross-module** — `mla_attn_q4k`, `moe_router_q4k`, `swiglu_expert_q4k` and
`mtp_head_q4k` each instance their **own** `glm_matmul_q4k` engine (plus the bf16 tail units). That
unused parallelism is the miniaturization target.

### The "compute is nearly free" budget (the enabling constraint)
Die utilization ≈ 20–25 % → a **~4–5× compute-slowdown budget** before compute becomes the
bottleneck. Every serialization/sharing lever spends from this budget; stay under it and
throughput is unchanged. **Measurable now** (no vendor tools): the cycle-accurate emulation
(`EXPERT_STALL`, [`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)) reports `cyc_per_tok` and the exposed
stall — run a serialized variant and confirm the token window is still stall-dominated
(`compute_cyc` still < `exposed_stall`). This is **enabler E2** below.

## Lever catalog (ranked by die-area impact)

Savings are **estimates**, and the concrete LUT/cell figures below were measured on the **prior FP8
die** (tag `fp8-verified-baseline`) — the exact **Q4_K** per-lever LUT delta is now **measurable** on the vendor
flow (E1 = **Vivado ML 2026.1**, which routed the whole die on XCKU3P; yosys 0.66 still can't map
the compute datapaths through ABC — see [`PHYSICAL_SKY130.md`](PHYSICAL_SKY130.md)) but has not been
re-run per-lever. The reduction is by-construction, and every lever is designed to keep the
**decoded token byte-identical** (the prior-FP8 byte-identical evidence is quoted as such; the Q4_K
byte-identical re-check is part of E1/E2).

| # | lever | mechanism | est saving | time cost | budget-safe? | effort | risk / status |
|---|---|---|---|---|---|---|---|
| **L0** ✅ | compact config | right-size PE_N/DDR_NCH/KV_RESIDENT/EFIFO/CACHE_SLOTS (result-invariant) | PE array halved + smaller fabric | more cycles | ✔ | done | **Q4_K compact fit MEASURED** — the routed Vivado XCKU3P fit *is* the compact config (+ ACT_HW=1): 142,320 LUT / 87.5 % synth-stage (routed 141,298), 421 DSP, 0 BRAM |
| **L1** ◑ | **cross-op matmul sharing** | *(structure present in Q4_K RTL)* swiglu gate/up GEMMs run at different times → one shared `u_mm` via a 1-bit `up_pass` arbiter + 2:1 weight mux (`swiglu_expert_q4k.v` carries exactly one `glm_matmul_q4k`) | **prior-FP8 measured ~12K LUT4** (2 swiglu × 6186 FP8 matmul core, 6→4 engines/block; −1519 generic cells/expert) — **Q4_K re-measure not yet run (now measurable on the Vivado flow)** | **≈ 0** (already sequential) | ✔ (free) | large refactor | **structure in Q4_K RTL, passes `make q4k` (swiglu 240/240 functional); area is prior-FP8 (tag `fp8-verified-baseline`)** |
| **L2** ❌ | tail vector-ALU sharing | *(assessed — NOT bounded-viable)* only `glm_softmax` instances the pipelined primitives, and its 4 pipes are **distinct ops** (exp/add/mul/rsqrt — nothing to merge); RMSNorm/RoPE/act use **inline `glm_fp.vh` fp32 macros**, not shareable module instances | small (fp32 tail ≪ Q4_K GEMM) | small | ✔ | high (cross-module scheduler) | **skip — reward≪risk** |
| **L3** ◐ | intra-op serialization | swiglu gate/up → 1 **captured by L1**; the remaining piece is the **cross-module 3-way hoist** (mla+router+swiglu → one engine) | further PE-array cut | 2×+ that op | ✔ *within budget* | high | **deferred** (needs PE_N=8 + top-level ports + arbiter) |
| **L4** ✅ | shared dequant/fold | after L1 each `glm_matmul_q4k` already carries a single fp32-accumulate + Q4_K block-dequant/scale fold; **nothing further** | small | — | ✔ | falls out of L1 | **subsumed by L1** |
| **L5** ✅ | memory-fabric trim | *(assessed — already spent)* the 4 byte-agnostic controllers (ddr5_xbar, kv_cache_pager, expert_cache_ctrl/pf) were **already trimmed** (6b2c82f, 899ea64): minimal-width regs, verilator-clean; QDEPTH off-limits (NVMe/PCIe latency-hide) | small | none | ✔ (except QDEPTH) | med | **no change (already minimal; shared unchanged from prior track — byte-agnostic)** |
| **L6** ⚠ | bit-serial Q4_K MAC | 1-bit/cycle multiply → tiny multiplier | large per-PE | **16–32×** | ✖ **OVERSHOOTS budget** (compute becomes the bottleneck) | high | **high — skip** unless a deeper-idle regime is proven |
| **L7** ⚠ | tail precision trade | bf16 tail → fp16/bf12 | moderate | none | n/a | med | **NOT byte-identical** (fidelity trade) — separate decision |
| **L8** ◐ | repo dead-code quarantine | *(legacy TPU + `batched_moe` DONE)* the legacy scalar **TPU v2.0** core (16 src + 19 TBs: `tpu_top`/`soc`/`axi`, decoder, regfile, `gemm_systolic`, `conv2d_unit`, `vector_alu`, …) and the redundant **`batched_moe`** (its union-skip logic folded **inline** into `glm_decoder_block_q4k`) have been **removed** — **`make all` is GLM-Q4_K-only** (the prove-it gate). Remainder: the generic bf16/fp32 twins (the structural siblings of the Q4_K units) still in-tree | **0 on the chip** | — | ✔ | low (done) | none (hygiene only) |

## Phased roadmap

- **Phase A — resource right-sizing (DONE on Q4_K).** L0 compact config is what the **measured
  Vivado XCKU3P fit** built (`glm_q4k_system_cdc`, compact config + ACT_HW=1 — 142,320 LUT / 87.5 % synth-stage, routed 141,298),
  and the bit-exact gate now exists: every Fmax-campaign round was **re-proven bit-exact on the
  1155-test assembled golden** (`make model-q4k`). (The prior-FP8 `synth-glm-compact` /
  `sim-glm-compact` build, byte-identical token `{0,11,11}`, was removed with the FP8 system top.)
- **Phase B — shared compute (structure in Q4_K; area prior-FP8).** L1 cross-op matmul sharing is
  **present in the Q4_K RTL** (swiglu gate/up → one `u_mm` via the `up_pass` arbiter, 6→4 engines/block)
  and passes the functional gate (`make q4k`, swiglu 240/240); its **measured area saving is prior-FP8**
  (≈12K LUT4, tag `fp8-verified-baseline`) — the per-lever Q4_K re-measure is now unblocked (E1 = Vivado) but not yet
  run. L2 tail-ALU sharing **assessed and
  skipped** (fp32 tail is inline-macro'd + distinct ops → not a bounded win; reward ≪ Q4_K-GEMM area).
- **Phase C — bounded serialization (mostly subsumed / deferred).** L3's swiglu part was captured by
  L1; L4 (shared fold) is subsumed (one fold per engine); L5 (fabric trim) was already spent. The
  **only remaining lever is the cross-module 3-way hoist** (mla+router+swiglu → one engine) — invasive
  (PE_N=8 + top-level ports + arbiter), **deferred to after E1** so the LUT payoff can justify the risk.
- **Phase D — validate & FPGA-fit (whole-die DONE).** E1's whole-die measurement is **DONE** —
  Vivado ML 2026.1 routed `glm_q4k_system_cdc` on **XCKU3P**: 142,320 LUT (87.5 %) synth-stage (routed 141,298 LUT), 421 DSP, 0 BRAM,
  hold met, routed Fmax 46.5 MHz (the Gowin/nextpnr scaffold is removed). Remaining: re-run the
  **per-lever** deltas on that flow, and E2 pin the exact serialization budget.
- **Out of scope / caution.** L6 (bit-serial — overshoots the budget), L7 (precision trade — not
  byte-identical, a fidelity decision), and cutting QDEPTH (hurts the NVMe/PCIe latency-hide).

## Enablers (unblock the above)

- **E1 — measurement (AVAILABLE — Vivado).** yosys 0.66 cannot map the compute datapaths (ABC wall),
  but the vendor flow is now **Vivado ML 2026.1** (it produced the routed XCKU3P whole-die fit; the
  old Gowin EDA / nextpnr path is removed). Concrete Q4_K cell/LUT numbers are measurable on that
  flow; the **per-lever** deltas on record are still prior-FP8 until re-run.
- **E2 — the serialization budget.** Use the `EXPERT_STALL` cycle-emulation to measure
  `compute_cyc` vs `exposed_stall` per token; every Phase-B/C lever must keep
  `compute_cyc < exposed_stall` (else throughput drops). This is buildable now.

## Verification methodology (how each lever stays honest)

1. **Byte-identical token** — the lever is a *result-invariant restructuring* (same math, different
   scheduling/sharing), so a system TB (`token == standalone glm_model_q4k`) must print the SAME token
   stream. This gate now **exists on the Q4_K track**: the assembled `glm_model_q4k` has an
   **end-to-end golden** (`make model-q4k` 1155 + `model-q4k-acthw` 1155), and every round of the
   routed-XCKU3P Fmax campaign was re-proven **bit-exact on that 1155-test golden** — the same gate
   any future lever must pass.
2. **Budget check (E2)** — the cycle-emulation must still show the token window stall-dominated.
3. **Area** — whole-die measured on the vendor flow (Vivado/XCKU3P); per-lever deltas measurable
   there but not yet re-run — the concrete per-lever numbers on record are prior-FP8.

## Honest constraints
- **Big refactors.** L1–L4 restructure the die datapath (arbitrated shared engines), unlike the
  parametric L0. Higher engineering + verification cost.
- **Per-lever area not yet re-measured.** The whole-die Q4_K fit is measured (Vivado/XCKU3P); the
  per-lever Q4_K LUT deltas are estimates until re-run on that flow — the concrete per-lever figures
  quoted are prior-FP8 (tag `fp8-verified-baseline`), not current-die numbers.
- **The budget is a ceiling.** Serialization is free only while `compute_cyc < exposed_stall`;
  past that (e.g. L6 bit-serial) throughput drops — the levers are ordered to respect it.

## Status
- **Phase A: DONE on Q4_K.** The routed Vivado XCKU3P fit *is* the Q4_K compact config (+ ACT_HW=1 —
  142,320 LUT / 87.5 % synth-stage, routed 141,298, 421 DSP, 0 BRAM), and the bit-exact gate exists: every Fmax-campaign round
  was re-proven on the 1155-test assembled golden. (The prior-FP8 L0 build, token `{0,11,11}`, was
  removed with the FP8 system top.)
- **Phase B: structure present in Q4_K; area is prior-FP8.** The L1 gate/up merge is **live in the Q4_K
  RTL** — `swiglu_expert_q4k.v` holds exactly one `glm_matmul_q4k` (`u_mm`) driven by the `up_pass`
  arbiter, 6→4 GEMM engines/block — and passes its functional TB (`make q4k`, swiglu 240/240). The
  **measured area** (≈12K LUT4 = 2 × 6186 FP8 matmul core; −1519 generic cells/expert via yosys `stat`)
  and the byte-identical evidence (token `{4,31,20}` gworst_rel 0.00689655; swiglu 1024 err/tol 0.1004;
  swiglu_pem 513; decoder 9) are **prior-FP8 track** (tag `fp8-verified-baseline`; the raw `stat` lived in the
  now-branch-only `PPA_FP8.md`). **The per-lever Q4_K re-measurement has not been run** (now
  measurable on the Vivado flow) — do **not** read the ≈12K LUT4
  as a Q4_K number. **L2 assessed and skipped** (only softmax uses shareable primitive modules and its
  4 pipes are distinct ops; RMSNorm/RoPE/act use inline `glm_fp.vh` macros — a cross-module fp32
  scheduler is high-risk for a fp32-tail reward that is ≪ the Q4_K-GEMM area).
- **All L1-style bounded merges are EXHAUSTED in the Q4_K RTL:** every chip module holds exactly
  **one** `glm_matmul_q4k` (mla `u_mm` already time-shares its 7 weight projections; router `u_gemv`;
  swiglu `u_mm`; mtp `u_proj` — verified by inspection of the `*_q4k` sources). L4 is subsumed (one
  fold per engine); L5 was already trimmed.
- **Phase C remaining = the cross-module 3-way hoist (L3).** Hoist mla+router+swiglu onto a
  *single* shared engine at the decoder-block level. This is the last real area lever but is
  **invasive**: PE_N reconcile (mla=4, router tile=8, swiglu TN=4 → PE_N=8), new top-level operand
  ports, and a 3-way arbiter — high byte-identical risk. **Deferred:** its payoff (4→3 engines/block)
  should be gated on a per-lever **E1 (Vivado)** measurement proving the Q4_K LUT delta justifies the
  risk, rather than done blind.
- **Phase D: whole-die numbers MEASURED** — Vivado ML 2026.1 on **XCKU3P**: 142,320 LUT (87.5 %)
  synth-stage (routed 141,298 LUT), 421 DSP, 0 BRAM, routed Fmax 46.5 MHz (campaign closed, bit-exact each round). Remaining: the
  per-lever delta re-run on that flow.
