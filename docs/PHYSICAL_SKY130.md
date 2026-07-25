# Physical characterization — REAL sky130 standard cells (prior FP8 track)

> **⚠ PRIOR FP8 TRACK — this is *not* the current product's physical run.** Everything in the
> **Results** and **Post-placement P&R** sections below is a place-and-route of **`glm_matmul_fp8`**,
> the FP8 GEMM tile of the **prior / datacenter track**. That module is **deleted from `main`** and
> preserved on tag **`fp8-verified-baseline`**. The current product is **Q4_K**
> local-device inference — the Q4_K GEMM tile is `glm_matmul_q4k`, the whole-chip top is
> `glm_q4k_system_cdc` (see [`Q4K_RETARGET.md`](Q4K_RETARGET.md),
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md), [`../README.md`](../README.md)).
>
> **No Q4_K sky130 / ORFS physical run has been done — it is OPEN / [PENDING].** The numbers here are
> **prior-FP8 measurements**; they are presented *as* prior-FP8 work and are **not** relabeled as
> Q4_K, and **no Q4_K physical result has been fabricated** to stand in for them. RTL/test names of
> the form `*_fp8` map to their `*_q4k` equivalents on `main` (`glm_matmul_fp8`→`glm_matmul_q4k`,
> `glm_fp8_system_cdc`→`glm_q4k_system_cdc`, `weight_loader`→`weight_loader_q4k`).
>
> **What still carries across both tracks** (format-agnostic, updated framing only): the
> ASIC-vs-FPGA **bandwidth** reasoning below, the shared fp32 MAC pipe (`glm_fp_pipe.v`, still on
> `main` and used by the Q4_K dequant→MAC path), and the **memory-system controller** ECP5 fit — the
> crossbars / KV pager / expert cache are pure control fabric with no FP8-vs-Q4_K datapath in them.

> **Scope note — ASIC is the rung-③ volume endgame, not "out of scope"** (see
> [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). The **near-term** product path is an **FPGA card**
> (ladder rungs ①②, [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) P3.2); a custom **ASIC** is the
> **rung-③ endgame at manufacturing volume** — precisely what breaks the FPGA's IO/PHY *bandwidth*
> ceiling (HBM stacks + many-channel controllers + near-memory Q4_K compute at ~TB/s) for lower
> $/seat + higher tok/s + lower power once the multi-million NRE amortizes over volume. (The earlier
> "ASIC out of scope" was argued from *compute*-bound reasoning; the real bottleneck is **memory
> bandwidth**, which is exactly what an ASIC's PHY/HBM breaks — so ASIC is sequenced *after* FPGA
> proves PMF, not abandoned.) This sky130 (an ASIC PDK) characterization therefore does **double
> duty**: (1) near-term **realizability evidence** — proof that a real compute-tile RTL synthesizes
> and *places* to real standard cells with timing met, i.e. it is physically sound and will map
> cleanly to FPGA fabric; and (2) **groundwork for that rung-③ ASIC** — a real-PDK PPA basis for the
> eventual volume tapeout. **Caveat:** the tile actually run through this flow is the **prior FP8**
> `glm_matmul_fp8`. The current product's PPA basis is the **Q4_K** tile `glm_matmul_q4k`, whose real
> sky130 / ORFS run is **[PENDING]** (see banner). The FP8 numbers stand as real-cell PPA anchors for
> a *closely related* fixed-point-accumulate GEMM; they are directional evidence for the Q4_K tile,
> not a measurement of it.

Moves the **prior FP8** compute tile's (`glm_matmul_fp8`) area/timing from **[EST]** (market/physics
models) to **real synthesized numbers on a real open-source PDK**. *(The equivalent run for the
current **Q4_K** tile `glm_matmul_q4k` is **[PENDING]**.)* Flow: yosys 0.66 `synth` → `dfflibmap` →
`abc -liberty` mapping to the **SkyWater sky130 high-density** standard-cell library
(`sky130_fd_sc_hd`, typical corner `tt_025C_1v80`, via `volare` PDK
`c6d73a35…`). `stat -liberty` gives real cell area (µm²); `abc … stime` gives the mapped
register-to-register critical-path delay (→ fmax). This is **pre-P&R** (gate-level synth,
no routing parasitics), so post-route fmax/area would be somewhat worse — but these are
*real cells on a real PDK*, not estimates.

## Results — prior FP8 tile (sky130_fd_sc_hd, tt corner)

*(Prior-FP8 measurements. `glm_matmul_fp8` is deleted from `main`; see banner. `fp32_mac_pipe`
lives in `glm_fp_pipe.v`, which is **still on `main`** and shared by the Q4_K dequant→MAC path, so
that row is directly relevant to the Q4_K tile.)*

| Module | Real area (µm²) | of which sequential | Critical path | fmax (pre-route) |
|---|---|---|---|---|
| `glm_matmul_fp8` (prior-FP8, block-scaled FP8 GEMM, PE 4×4, K=256) | **262,689** | 39.1 % (102,719) | **35.4 ns** | ~28 MHz |
| `fp32_mac_pipe` (shared pipelined fp32 MAC, `glm_fp_pipe.v`) | **48,121** | — | **7.6 ns** | **~131 MHz** |

## Post-placement P&R (real ORFS flow, sky130hd) — prior FP8 tile

Beyond the pre-route synthesis above, the **prior-FP8** `glm_matmul_fp8` was pushed through the
**OpenROAD flow (ORFS, `openroad/orfs` image, sky130hd platform)** — real floorplan + legalized
placement + timing-driven resizing + post-placement STA:

| Flow stage | Result |
|---|---|
| ORFS synthesis (sky130hd) | instance area **330,259 µm²** |
| Floorplan + PDN + tapcells | die/core/power-grid built |
| Global + detailed placement | **legal, 0 violations** (edge-spacing / padding / placement) |
| Resizer (RSZ) timing repair | placed **design area 357,320 µm² @ 38 % utilization** |
| **Post-placement STA @ 40 ns clk** | **WNS = 0, TNS = 0, worst slack +15.89 ns**, **0 setup / 0 hold** |

**What this adds over the pre-route table:** these are **post-floorplan, post-legalized-
placement** numbers with the timing-driven resizer engaged — most of the P&R flow, not just
synthesis. Post-placement timing is **fully met at 40 ns (25 MHz) with +15.89 ns slack** →
critical path ≈ 24 ns → **~41 MHz post-placement**, *better* than the pre-route 35.4 ns
because the resizer buffers/upsizes the dequant/fold path — a direct, measured confirmation of
the "pipeline the fold stage → higher fmax" thesis (all on the **prior FP8** tile; the fold-drain
pipeline it validates was applied in `src/glm_matmul_fp8.v` on tag `fp8-verified-baseline`, item 2 below).

**Where it stops, and why (environment, not design).** The flow died at **CTS** with
`illegal instruction`: `openroad/orfs` ships an **amd64-only** OpenROAD binary, and on this
**ARM (Apple Silicon) host it runs under emulation** where TritonCTS hits an instruction
qemu can't emulate (SIGILL). There is **no arm64 ORFS image** to switch to. So CTS → routing →
parasitic extraction → post-route STA/power need an **x86-64 Linux host (or cloud runner)** to
finish — the design passed synthesis, floorplan and placement cleanly; the block is a pure
host-architecture limit.

## What this establishes (prior FP8 track)

1. **The −87.6% accumulator claim has a real-cell anchor — on the prior FP8 tile.**
   `glm_matmul_fp8`'s BFP fixed-point accumulator maps to real sky130 cells; the 262,689 µm²
   figure (39% sequential) is a concrete area for the FP8 GEMM tile, replacing the cell-count
   `[EST]` of the prior FP8 PPA analysis (`PPA_FP8.md`, now on tag `fp8-verified-baseline`). The −87.6%
   fixed-point-accumulate win is an **FP8-specific, prior-track** result; the Q4_K tile
   (`glm_matmul_q4k`) has its own dequant→fp32-accumulate path whose real-cell area is **[PENDING]**.

2. **The fmax-limiting path was REAL, and the pipeline fix was applied — on the prior FP8 tile.**
   The shared pipelined MAC closes at ~131 MHz (7.6 ns), but `glm_matmul_fp8`'s own
   register-to-register path is 35.4 ns (~28 MHz) — i.e. the **block-dequant / accumulate-fold
   logic AROUND the pipelined MAC, not the MAC itself, was the fmax limiter**. This was flagged by
   the prior FP8 PPA analysis and then measured on real cells — and **fixed** (on tag `fp8-verified-baseline`,
   `src/glm_matmul_fp8.v`): the dequant/fold stage (the accumulator drain) was pipelined, registering
   `acc_sel_r`/`wf_sel_r` before the block-scale `fp32_mul_pipe` so `fixed_to_fp32` (48-bit
   leading-one + barrel-shift + RNE) no longer shares a stage with the mul's 24×24 mantissa multiply.
   `DEQ_LAT +1`, latency-transparent via `out_valid`; data **bit-identical** (prior-FP8
   `glm_matmul_fp8_tb`, ALL 224, err/tol 0.4317 unchanged). Real sky130_fd_sc_hd (tt) timing
   confirmed the win: the isolated fold segment `accx → fixed_to_fp32 → block-scale mul` dropped
   **15,576 → 12,390 ps (−20.5%, 64.2 → 80.7 MHz)** and the full 2×2/K256 module
   **22,186 → 14,189 ps (45 → 70 MHz)**. *(All prior-FP8 numbers.)* The Q4_K tile has an
   analogous per-column-dequant → fp32-MAC drain in `src/glm_matmul_q4k.v`; whether it hits the same
   fold-stage limit and needs the same pipeline is **not yet characterized on cells — [PENDING].**

## Honest scope (what would take it to full physical sign-off)

- **This is the prior FP8 tile, not the Q4_K product.** The single biggest open item: **no Q4_K
  sky130 / ORFS physical run exists.** `glm_matmul_q4k` (and the whole `glm_q4k_system_cdc`) has been
  elaborated and structurally checked (`make synth-glm`, see below) but **not** synth-mapped to real
  cells or placed. Everything above characterizes `glm_matmul_fp8` on tag `fp8-verified-baseline`.
- **Through placement, not yet routed (prior FP8).** Synthesis, floorplan and legalized placement are
  done on the real ORFS flow (see the post-placement section — 357,320 µm², timing met at
  40 ns). The remaining **CTS → routing → parasitic extraction → post-route STA** did not run
  here because `openroad/orfs` is **amd64-only** and TritonCTS SIGILLs under ARM emulation on
  this host; there is no arm64 image. Those steps need an **x86-64 Linux host / cloud runner**
  — a host-architecture limit, not a design one. (An earlier OpenLane Docker attempt also hit
  image-pull friction; ORFS got much further — all the way through placement.)
- **No power yet.** Dynamic/leakage power needs the post-route netlist + activity (VCD) →
  a power analysis pass. The clock-gating (`clk_en_ctrl`/`icg_cell`) is in place to exploit
  the ~75%-idle die, but the J/token figure stays a model estimate until a routed power run.
- **Per-module, not full-chip.** The whole `glm_q4k_system_cdc` is elaboration- and
  structurally-clean (`make synth-glm`: yosys `hierarchy -check` + `check -assert`, 0 unresolved),
  but a full-chip sky130 map/P&R (with the memory macros as real SRAM) is a larger effort; the prior
  FP8 GEMM + the shared fp32 MAC above are the compute-core datapoints.

## Reproduce (prior FP8 tile — on tag `fp8-verified-baseline`)

`glm_matmul_fp8.v` is deleted from `main`; the command below reproduces the prior-FP8 result on
tag `fp8-verified-baseline` (`git checkout fp8-verified-baseline`). The analogous **Q4_K** run — swapping in `src/glm_matmul_q4k.v` /
`-top glm_matmul_q4k` — has **not yet been done ([PENDING])**.

```sh
python3 -m volare enable --pdk sky130 c6d73a35f524070e85faff4a6a9eef49553ebc2b
LIB=~/.volare/volare/sky130/versions/c6d73a35*/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
# tag fp8-verified-baseline:
yosys -p "read_verilog -sv -I src src/glm_matmul_fp8.v src/glm_fp_pipe.v; \
          synth -top glm_matmul_fp8 -flatten; dfflibmap -liberty $LIB; \
          abc -liberty $LIB; stat -liberty $LIB"          # -> Chip area
# timing: append an abc -script ending in `topo; stime`  # -> Delay = <ps>
```

## FPGA resource fit (ECP5) — memory-system controllers, partial & honest

The memory-system controllers below are **format-agnostic** (pure control fabric — crossbars, KV
pager, expert cache, boot/weight loaders — with **no** FP8-vs-Q4_K datapath), so this ECP5 fit
carries across both tracks. Since the **near-term** product is an **FPGA card** (ladder rungs ①②;
the ASIC is the rung-③ volume endgame, [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)), this is a first
look at whether the *full system* fits a real FPGA. Partitioned `synth_ecp5` (yosys 0.66) of each
controller **standalone** completed cleanly:

| block | LUT4 | FF | CCU2C | MULT18 | EBR |
|---|---|---|---|---|---|
| `ddr5_xbar` | 18,137 | 8,507 | 0 | 0 | 0 |
| `flash_xbar` | 26,112 | 17,011 | 0 | 0 | 0 |
| `kv_cache_pager` | 25,249 | 25,367 | 17 | 0 | 0 |
| `expert_cache_pf` | 744 | 287 | 80 | 0 | 0 |
| `weight_loader_q4k` † | 302 | 202 | 63 | 0 | 0 |
| `boot_loader` | 931 | 399 | 74 | 0 | 0 |
| **sum (6 controllers)** | **71,475** | **51,773** | 234 | 0 | 0 |

† `weight_loader_q4k` was retargeted from the prior `weight_loader` to Q4_K super-block decode; the
row is the **prior-track partitioned synth** figure, a Q4_K re-synth refresh is **[PENDING]**. The
other five controllers are datapath-independent and unchanged.

vs an **ECP5-85** (`LFE5UM5G-85`: ~84k LUT4, ~84k FF, 156 MULT18, 208 EBR): the **memory-system
controllers alone are ~85% of LUT4 / 62% of FF** — so the full system (controllers + the compute
die) **does not fit an ECP5-85**; a larger FPGA is needed. (The 0 EBR reflects that the actual
RAM is external/TB-modeled here, so these are the control-fabric + QDEPTH-queue costs; the crossbars'
deep outstanding queues are the LUT/FF drivers.)

**Honest correction — the compute die's ECP5 size was NOT obtained (prior FP8 exploration).** The
prior-FP8 exploratory pass could not `synth_ecp5` the FP8 die (`glm_model_fp8`, now deleted): yosys
0.66 is prohibitively slow / artifact-prone elaborating it. That pass *reported* the die as ~10–17×
over an ECP5-85 due to a `glm_matmul_fp8` at `KMAX=16384 → NB=128` accumulator banks — **this was a
synth artifact, not the real design.** It was independently disproven at the time: the FP8 die
simulated and passed, and every in-die matmul had its `KMAX` **overridden** to `FF_KMAX_D/M`
(= 256/128 at the slice → **NB ≤ 2**), never the module-default 16384 — so the die was small
(NB ≤ 2), its ECP5-mapped size simply **unmeasured** (a yosys-0.66 `synth_ecp5` tooling limit),
not a design over-provisioning. **For the current Q4_K product this question is since ANSWERED
by measurement, not expectation:** the whole Q4_K product top `glm_q4k_system_cdc` is now
**placed & routed on a real XCKU3P** (Vivado ML 2026.1, compact config + ACT_HW=1: **142.3K LUT /
87.5%**, ~100K FF, 421 DSP, hold met, routed Fmax **46.5 MHz** — see
[`../fpga/README.md`](../fpga/README.md)), and the assembled-Q4_K end-to-end golden exists
(`make model-q4k`, 1155/1155, per [`../README.md`](../README.md)); the ECP5 mapping specifically
is superseded/moot.

**Takeaway:** ECP5-85 is too small for the full system (memory-system controllers ~85% alone) →
a larger FPGA was needed — and that larger-FPGA fit is since **DONE / MEASURED**: the Vivado-fit
step of [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) / [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)
ran for real — Vivado ML 2026.1 synth + full place&route of `glm_q4k_system_cdc` on **XCKU3P**
(compact config + ACT_HW=1: 142.3K LUT / 87.5%, 0 BRAM, hold met, routed-Fmax campaign closed at
**46.5 MHz**, every round re-proven bit-exact on the 1155-test assembled golden —
[`../fpga/README.md`](../fpga/README.md)). The ECP5 rows above stand as the earlier
partitioned-controller data point only; a full-system ECP5 map is moot. Board bring-up (a running
board) is still not done.
