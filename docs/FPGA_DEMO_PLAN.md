# FPGA measured-demo plan (D0.2 — fit gate MEASURED; board demo still open)

*Turns the [`fpga/`](../fpga/README.md) "FPGA fit" scaffold into a measured result and a hardware
demo. **Status (2026-07): the fit half is DONE — MEASURED** (Vivado routed fit + Fmax, see below);
the on-board token demo (L3) is still open. **Why it matters:** the FPGA class sets the box's size / thermal / **BOM / per-seat price**,
and the per-seat price is what makes the [`ICP.md`](ICP.md) real. Everything downstream is bounded
by the number this track returns.*

**Product frame:** local, single-user box (B=1). This **is** what `main` develops right now — the
**rung-① "prove-it"** plan for the **GLM-5.2 Q4_K accelerator** (GGML UD-Q4_K_XL, ~467 GB) on the
staged [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) — the cheap, near-term proof that the *same* verified
RTL runs the real model's tokens on real FPGA silicon (**~5–8 tok/s [EST]** — re-banded by the
measured-proxy design-point note in §The demo ladder; slow but memory-bound).
Two boards, two jobs (see [`PART_SELECTION.md`](PART_SELECTION.md) §the ①a-vs-①b split):

- **①a fit-measurement / bring-up board** — the **Sipeed Tang Mega 138K Pro** (**Gowin GW5AT-138**,
  ~138 K LUT, on-board DDR3). Cheap way to exercise the synth + P&R flow and read a fit. It is
  **DDR3-only, no NVMe**, so it can *measure the fit* but **cannot** run the streaming token demo.
  *(Superseded for fit-measurement: the fit gate was measured directly on the ①b part with Vivado;
  the old Gowin/nextpnr scaffold has been removed from `fpga/`.)*
- **①b demo target** — a **low-end Kintex UltraScale+ (KU3P-class, XCKU3P)** board with **DDR4 + 1
  NVMe** (~$1–2 k box). The DDR4/NVMe memory system is what sets the rung-① tok/s (see the
  measured-proxy note below); this is also the **repo-designated part** whose **routed fit + Fmax
  on Vivado is now MEASURED** (see below).

See [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) for the staged context, [`PART_SELECTION.md`](PART_SELECTION.md)
for the board BOM, and [`MINIATURIZATION.md`](MINIATURIZATION.md) for the compact config.

---

## What is DONE vs what remains

The honest status splits cleanly: the Q4_K RTL is **structurally signed-off at product scale**, and
the **routed fit + Fmax is now MEASURED** (Vivado ML 2026.1, XCKU3P) — the number that sizes the box
is in. What remains open is the *board* (bring-up + the L3 token demo), not the fit.

**DONE — structural / elaboration sign-off (no LUT/Fmax, no functional golden):**

| Check | What it proves | Target |
|---|---|---|
| **Whole 2-clock Q4_K product top** `glm_q4k_system_cdc` (+ every Q4_K compute/memory/CDC leaf) elaborates and passes `hierarchy -check` + `check -assert` (exit 0 — no unresolved hierarchy, comb loop, multiple driver, or inferred latch) | the assembled Q4_K chip is a **structurally sound netlist** — *structural sign-off, not a sim* | `make synth-glm` |
| **Full 753B UD-Q4_K_XL-shape elaboration** of `glm_model_q4k` (DIM 6144 / L=78 / 256-expert / VOCAB 154880) | the RTL **elaborates cleanly at the true product shape** — type/width / `$clog2` / part-select only; *no stimulus, no golden, no run* | `test/full_config_elab_wrap.v` ([`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md)) |
| **Core GEMM is buildable at product scale by construction** — `glm_matmul_q4k` is a **sequential streaming-K fp32-accumulate** fold (one Q4_K super-block = 256 weights; `NSB = ceil(KMAX/256)` super-blocks along K), so it uses **O(1) FP pipes** regardless of K, with the only K-scaling being the small block-accumulator memory (BRAM-able) | the matmul does **not** blow up in FP pipes / FFs at the LM-head K depth — no unrolled per-block explosion | bit-exact gate `make q4k` (`glm_matmul_q4k` **160/160** vs the ggml Q4_K ref `tools/q4k_ref.py` — **not** the real GGUF) |

**MEASURED — the gate that actually sizes the box (DONE, 2026-07):**

> The **Q4_K routed fit + Fmax on Vivado (XCKU3P)** is **MEASURED**: Vivado ML 2026.1, real synth +
> full place & route of `glm_q4k_system_cdc` (compact config + `ACT_HW=1`) — **141,298 LUT routed** (142,320 / 87.5 % at the synth stage),
> ~100 K FF, **421 DSP**, 0 BRAM, hold met. Routed **Fmax 10.2 → 17.2 → 46.5 MHz** through a
> repipelining campaign, every round re-proven bit-exact on the 1155-test assembled golden
> (round 4 baseline rope cone 98.4 ns/382 levels; round 5 rope 10-stage; round 6 `glm_act` 20-stage +
> rmsnorm reduce/rsqrt; round 7 `glm_matmul_q4k` dequant+MAC 5-stage). The **campaign is CLOSED at
> 4.6×**: the worst path is now **route-dominated** (`u_moe/y_out` → hbuf wide bus, 21.2 ns, 59 % wire
> at 87 % utilization) — physical work, not arithmetic. 46.5 MHz sits in the bring-up demo's target
> band; 200 MHz-class is rung-②/③ work, and the stage decompositions carry to the ASIC unchanged.
> Flow: [`../fpga/synth_ku3p.tcl`](../fpga/synth_ku3p.tcl) (part `xcku3p-ffvb676-2-e`, compact
> config); reports in `fpga/out/` ([`../fpga/README.md`](../fpga/README.md)).

The Q4_K LUT/DSP numbers above are the **routed Vivado measurement on the target part**. Structural
cell histograms from `make synth-glm` remain a *sanity* cross-check, **not** a routed fit.

> **Prior FP8 track (tag `fp8-verified-baseline`) — methodology carried forward; numbers are NOT Q4_K.** The prior
> FP8 datacenter track ran an open **Gowin `synth_gowin`** synth exploration that established two
> transferable methodology points: (1) **DSP inference** (`MULT18X18`/`MULT9X9`) maps a block-scaled
> quant datapath and **sidesteps the `abc -lut4` accumulator timeout**, and (2) a **sequential
> block-scaled dequant fold is O(1) in FP pipes** (the pattern `glm_matmul_q4k` uses today). Those
> runs produced **FP8-specific** figures — e.g. the `glm_matmul_fp8` leaf @ KMAX=256 mapped in ~77 s
> to **~17.8 K LUT4-equivalent + 20 DSP mults + ~5.4 K DFF** — which live on tag `fp8-verified-baseline` and are
> **prior-track measurements, not Q4_K, and not routed** (no P&R Fmax). **The Q4_K Vivado run above
> is DONE and supersedes them**; do not read the FP8 numbers as the current product's fit. (The
> Gowin/nextpnr scaffold itself has been removed from `fpga/`, superseded by the Vivado flow.)

## L3 integration status (2026-07) -- the sim side is BUILT; the board side is named, not pretended

Everything between the verified RTL and a live XCKU3P board was mapped, and every piece that can
exist WITHOUT the physical board now exists, is default-off, and sits in `release-gate` with a
paired must-fail injection:

| piece | artefact | proven in sim | gate |
|---|---|---|---|
| product top could not reach its own proven paths | 6 params forwarded (`LOOPBACK`×3, `SELF_KV`, `EXPERT_STALL`, `SYS_REQ_LANES`) | liveness: `LOOPBACK=1` DIFFERS at the system level | commit `0959277` |
| die vs memory backpressure | `-DTB_REQ_STALL` LFSR on `mem_req_ready` | **7,688 stall cycles, tokens bit-exact; 60 cycles of held-request payload churn measured** (the AXI-illegal waveform) | `loopback-rest` variant |
| runtime reads → MIG | `src/ddr4_mig_shim.v` (per-channel registered skid slot → one AXI AR/R master) | AR payload stable under a requester churning a held request EVERY cycle (14,692 churn cycles absorbed) | `mig-shim` |
| boot writes → MIG | `src/axi_boot_writer.v` (AW/W/B; read/write channels are independent, so no arbiter) | 400 hostile-requester writes retired intact against a slave refusing ~half of AW and W | `boot-writer` |
| storage | `src/spi_flash_reader.v` (SPI-NOR 0x03) + `boot_loader` | end-to-end boot copy over REAL mode-0 SPI waveforms, byte-for-byte into DDR | `spi-boot` |
| weight image | `tools/ckpt_pack_q4k.py` header-order **BUG FIXED** (sb-outer vs the loader's pj-outer -- coincides at nb==1, silently diverges at real K>256) | the packer's own file consumed by the real `weight_loader_q4k` at nb=3: every header slot + every code nibble == the pre-pack source | `packer-rtl-crosscheck` |
| host | `src/uart_host_bridge.v` ('T'/'K' frames, 8N1) | real waveforms both directions, independent TB-side decode | `uart-host` |
| the board top itself | `fpga/l3_top.v` -- all of the above + em/fn LUTRAM stores wired to `glm_q4k_system_cdc` with the PHY-closure params ON | ELABORATES (that is all elaboration proves) | `l3-elab` |

Two geometry facts discovered on the way, both now encoded rather than remembered:
* the fitted geometry needs a **25-bit fw loopback key**, which the elaboration guards correctly
  reject at the default marker position -- `LB_MARKER_LSB` (default 24 = byte-identical, boards use
  32 with `DDR_ADDR_W=40`);
* `em`/`fn` are the two weight families with **no loopback path** and a SAME-CYCLE combinational
  contract, so `l3_top` serves them from boot-filled LUTRAM (`[EST]` ~8K LUTs at the fitted config;
  a BRAM migration needs a die clock-gate term and is a named optimisation, not assumed).

**CORRECTION (found immediately after the S2-S7 commit, recorded before anything else):**
`l3_top` as committed has a FUNCTIONAL GAP the elaboration gate cannot see.  The LOOPBACK
parameters loop only the **CODE lanes** (`aw_q`, `fw_q`/`fw_q_up`, `rw_q`/`lw_col`/`gn_val`); the
dequant **HEADERS** (`aw_d`/`aw_dmin`/`aw_scales`, `fw_d_g`/`fw_dmin_g`/`fw_scales_g`,
`fw_d_u`/…, `rw_d`/…) are explicitly "served same-cycle by the stub" (glm_q4k_system.v:249-251)
and have NO production path.  `l3_top` ties them to zero, so on a board every dequantised weight
would be 0 and the tokens garbage.  The S2-S7 commit message's "every stub input they replace is
tied off here" was therefore an overclaim.  The header path is the top open item below; the
architecturally consistent fix is a boot-filled BRAM header store + a 1-cycle die clock-gate per
header pull (the exact freeze mechanism LOOPBACK already proved bit-exact), since headers are
small (~tens of KB at the fitted config) and BRAM sits at 0%.

**Still open, in dependency order** (1-2 are sim-verifiable; 3-6 need the board):
0. **the dequant-header path** — HALF DONE: `HDR_LATE` (commit `c54224a`) moved the matmul's
   header consumption from the start-latch to accept-time wire reads, removing the one-cycle
   deadline a sync-read BRAM cannot meet (and which die clock-gating cannot fix -- a global freeze
   preserves internal latch-vs-settle order).  Remaining, all OUTSIDE frozen RTL: the `l3_top`
   stores + boot segments + the packer's header images + an end-to-end sim gate.
   **Sizing at the fitted config (measured key spaces, not guesses):**
   - `rw`: keyed by `db_layer` alone (one router pass per layer) — 6 × (16+16+96)·8 = **6 Kb**;
   - `aw`: keyed `{db_layer, aw_sel, aw_grp}`, dense 6×16×64 × 256 b = **1.5 Mb ≈ 43 RAMB36**;
   - `fw`: the word must pair GATE+UP headers (both buses live in one pass) = 1024 b.  A dense
     `{ly,sel,shared,eidx,grp}` address is **~24 Mb — exceeds the whole 12.6 Mb BRAM budget**, so
     the store SPLITS: routed `{ly,sel,eidx,grp≤32}` 4096×1024 b = 4 Mb ≈ 114 RAMB36, plus
     shared/dense `{ly,sel,grp≤64}` 1024×1024 b = 1 Mb ≈ 29 RAMB36;
   - total ≈ **190 of 360 RAMB36** (BRAM is at 0% in the measured fit) — fits, with pure
     concatenation addressing (no index adders).

1. a packer mode that lays the DDR image out at the **loopback address encoding**
   (`packer-rtl-crosscheck` covers the `wl_mem` staging layout, which is observability-only);
2. MIG IP instantiation + pins + XDC for the chosen board; `l3_top` exposes exactly one AXI port;
3. the Vivado re-fit: the measured fit is 87.5% LUT with **BRAM/URAM at 0%**, and `l3_top` adds the
   shim/writer/UART/SPI/em-store on top -- whether it closes is a MEASUREMENT, not a claim;
4. flash the compact-config image, `T`-frame over UART, and read tok/s off the wire -- the actual L3.

## The demo ladder (each rung is cheap and de-risks the next)

| Rung | What | Tooling | Proves | Status |
|---|---|---|---|---|
| **L0** | **Synth fit** — structural cell histogram of the compact top | `make synth-glm` (yosys `hierarchy -check` + `check -assert`, have it) | the design is a **sound netlist** that resolves at product scale → sanity area band | **DONE** (structural sign-off, exit 0) — *not* a routed fit |
| **L0′** | **Routed fit + Fmax on the real part** — LUT/FF/DSP/BRAM/URAM + Fmax on **XCKU3P** | **Vivado** (`fpga/synth_ku3p.tcl`) | the design *fits a class of FPGA* → **BOM band + per-seat price** | **DONE — MEASURED** (142,320 LUT 87.5 %, 421 DSP, 0 BRAM; see above) |
| **L1** | **P&R one leaf** on the board → real **Fmax** | Vivado | real clock → real tok/s extrapolation (tok/s = Fmax ÷ `cyc_per_tok`) | **subsumed by L2** — the full top routed at 46.5 MHz; the old Gowin/nextpnr leaf harnesses are removed (`fpga/bringup_harness.v` supersedes) |
| **L2** | **P&R the compact top** → full routed fit + Fmax | Vivado (`fpga/`) | the whole product top places & routes | **DONE — MEASURED**: `bringup_harness.v` buries the wide memory-side ports; full P&R, hold met, routed Fmax **46.5 MHz** (campaign closed at 4.6×; worst path route-dominated) |
| **L3** | **Reduced-config forward on the board** — a few real GLM-5.2 layers, Q4_K weights streamed from on-board Flash/SD → **measured tok/s** | board + a Q4_K weight image (`tools/ckpt_pack_q4k.py`) | *real silicon token at a measured tok/s* — **THE demo** | needs board |

**tok/s is already grounded in RTL, not hand-waved:** the workload is **memory-bandwidth-bound**
(tok/s ≈ sustained weight bandwidth ÷ **~25 GB per token** = ~40 B active params × ~0.6 B/param), and
the memory-stall mechanism + `cyc_per_tok` are measured on real RTL cycles
([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md), `make perf-q4k`: Q4_K slice `cyc_per_tok` ≈ 10,896,
exposed stall linear in `FLASH_LAT` — 2,567 cyc/token at 1024 (RESIDENT=0) vs 35 at RESIDENT=1). (`FLASH_LAT` and the `flash_xbar` read path are committed RTL names for a
**medium-agnostic storage-read abstraction** — address → weight bytes, latency-hidden — that in the
product fronts an **NVMe/PCIe** backend, not a NAND die.) L0′/L2 now give the routed **Fmax —
46.5 MHz MEASURED**, so the *slice* demo's wall clock is computable: `cyc_per_tok` ≈ 8.0–11.0 K
([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)) → **~170–240 µs/token ≈ ~4,200–5,800 *slice* tok/s** —
the correctness-demo speed of the tiny slice, **not** a GLM-5.2 product number (never conflate the
two). On this rung the target is
**~5–8 tok/s [EST]**, set by the sustained weight bandwidth the board can feed (~100 GB/s): the ~14 GB
of per-token **routed experts** stream from NVMe (the wall — they change every token) while the ~11 GB
per-token hot-set touch (attention / dense / shared; canonical byte constants:
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2) caches in **DDR4**. The funded custom board (rung-②, DDR5/HBM) is
where the **~15–40 tok/s [EST]** interactive product lands, and rung-③ (SoC/ASIC at volume) reaches
**~40+ [EST]** — all the *same* verified RTL, only the memory bandwidth the silicon can feed it changes.

> **Measured-proxy update (2026-07 — [`H_MEASUREMENT.md`](H_MEASUREMENT.md) /
> [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md)).** The rung tok/s bands above pre-date the
> h/U proxy measurement (OLMoE-1B-7B-Instruct trace; U(K)/EOR since re-measured GLM-family — see the
> updated note below). The measured-roofline
> design-point menu [EST, MEASURED-PROXY inputs] re-bands them: **NVMe ×1–2 (no multipliers)
> ~0.5–1 tok/s; 90 GB DRAM + 100 GB/s → 13–24; 90 GB + 200 GB/s (ONFI 64-ch) → 25–47; 225 GB +
> 200 GB/s → 54–127**. Spec-decode amortization must be read as
> **A/U(K) ≈ 1.1–1.3× at K=4, A~3** (measured U(4)=2.25–2.64 OLMoE; GLM-4.5-Air measured
> U(4)=2.60–2.71 supersedes), not a ~2× K multiplier. Also note the
> compute-clock ↔ area trade: at the measured 46.5 MHz, sustaining 100 GB/s of dequant consumption
> would need ~4,300 lanes (not feasible on KU3P; ~1,000 at 200 MHz-class, ~200 at ASIC 1 GHz+) — a
> higher clock buys a smaller/cheaper die, **not** higher tok/s (memory stays the wall past
> saturation).
>
> **Updated 2026-07:** the **rung-③ primary design point pivoted to FULL RESIDENCY** — 512 GB
> LPDDR5X (~1.1 TB/s) holds the whole ~467 GB checkpoint, design point **≈80 tok/s [measured-inputs
> EST]** (U(K) **and** the MTP accept rate r both GLM-family measured — the vLLM MTP sweep, job B,
> put the memory-bound optimum at K=1–2; ~95 if GLM-5.2's deeper MTP hits its published accept
> depth — [`H_MEASUREMENT.md`](H_MEASUREMENT.md)) — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md). The
> streaming menu above — including the 54–127 point — now applies to **rung ① (this demo box: NVMe
> streaming IS how it works), the hybrid upside SKU, and >512 GB checkpoints**, not the rung-③
> primary SKU; h matters only for the hybrid-SKU decision (residency ⇒ h=1 by construction). U(K)
> is now **GLM-family measured** — GLM-4.5-Air traced via MoE-gate hooks
> ([`H_MEASUREMENT.md`](H_MEASUREMENT.md) 2nd measurement): U(4)=2.60–2.71, U(8)=4.19–4.41 —
> superseding the OLMoE first-pass (kept above as history).

## Cost / what a real demo needs

- **L0 is free and DONE** (`make synth-glm`, yosys). **L0′/L2 are DONE — measured with Vivado ML
  2026.1 Standard (free, covers XCKU3P)** via `fpga/run_fit.sh`. The old **Gowin EDA /
  `nextpnr-himbaechel`** path for the Tang Mega bring-up board is **removed** (superseded by the
  Vivado flow); the Tang Mega 138K Pro (**~$200–300**) would only be needed to *program* a bring-up
  run.
- **L3 (the money shot)** needs a **KU3P-class board (DDR4 + NVMe)** + a Flash/SD-resident Q4_K weight
  image (`tools/ckpt_pack_q4k.py` produces the RTL weight-memory layout `weight_loader_q4k.v` reads)
  and a reduced config (a few layers) — **not** the product's full **1–4 TB NVMe model store** or the
  **DDR5/HBM + NVMe/PCIe (M.2) host controller** (those are **rung-② custom-board / vendor-IP items** —
  see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md); DDR5/HBM is a funded-board spec, not the near-term
  proof). The demo is a **reduced-config proof that real weights produce real tokens on real silicon at
  a measured rate**, not the shippable box.

> **Weight-image honesty (per [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md)):** `tools/ckpt_pack_q4k.py`
> round-trips its gen → pack → unpack against a **synthetic tiny GGUF it fabricates in-memory**, proven
> bit-exact vs the ggml dequant mirrors in `tools/q4k_ref.py` — and those mirrors are now themselves
> **proven bitwise-equal to real GGUF bytes at the dequant layer** (376,586,240 weights —
> Q4_K/Q6_K/Q8_0 across two real published GGUFs — vs llama.cpp's own `dequantize_row_*` —
> [`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)). The RTL weight path has **mixed-type Q6_K/Q8_0/F16
> consumers (DONE — `make mixedtype`)**. Still honest and open: the real 467 GB UD-Q4_K_XL file has
> not been downloaded/consumed end-to-end, and llama.cpp **whole-runtime** numeric equality is
> out-of-contract.

## Why this is the investable lever

An investor discounts every `[EST]`. The demo ladder converts three of them to fact, cheaply:
1. **Fit/BOM** (L0′ routed Vivado fit on XCKU3P): "it fits a $X FPGA" → the per-seat price the
   [`ICP.md`](ICP.md) economics need. *(**DONE — MEASURED**: 141,298 LUT routed (142,320 / 87.5 % synth-stage) / 421 DSP / 0 BRAM,
   routed Fmax 46.5 MHz.)*
2. **Single-user tok/s** (L1/L2 Fmax × measured `cyc_per_tok`): the rung-① proof speed (**~5–8 tok/s
   [EST]**, DDR4 + NVMe), measured not modeled — the funded rung-② board is where **~15–40 tok/s [EST]**
   lands ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). *(Fmax half now measured — 46.5 MHz, slice
   wall clock ~170–240 µs/token; the rung tok/s bands are re-banded by the measured-proxy note in
   §The demo ladder.)*
3. **Real tokens on silicon** (L3): the whole thesis, demonstrable on a desk.

Paired with **one signed design-partner LOI** from the primary ICP (a law-firm innovation team) —
**[PENDING]** — that is the pre-seed package: *a measured box + a customer who wants it.*
