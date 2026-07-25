# `fpga/` — XCKU3P fit (Vivado) for the GLM-5.2 **Q4_K** accelerator

The **hardware-fit gate — MEASURED, closed** (see [Results](#results--measured-vivado-ml-20261-2026-07-compact-config--act_hw1)
below): the verified Q4_K RTL is a **real, routed FPGA fit** — actual LUT / DSP / BRAM
utilization and routed **Fmax** on a Kintex UltraScale+ **XCKU3P** — the number iverilog
(sim) and yosys (`check-assert`, structural) cannot give. This number sets the FPGA
class → the box's size, thermal budget, BOM, and per-seat price.

> **Track note.** This is the **current Q4_K / XCKU3P / Vivado** flow. A prior scaffold
> targeted **FP8 on Gowin GW5AT-138** (Gowin/nextpnr); that scaffold is **removed**
> (superseded by this Vivado flow) — it referenced the FP8 datapath also removed from
> `main` (preserved on tag `fp8-verified-baseline`).

## What it synthesizes
`glm_q4k_system_cdc` — the whole 2-clock product top (compute die `glm_model_q4k` +
memory system + CDC) — in the **compact** config (result-invariant resource params:
`PE_N 4→2, DDR_NCH 4→2, KV_RESIDENT 16→8, EFIFO_DEPTH 16→8, CACHE_SLOTS 4→2`). The
decoded token is byte-identical to the default config (proven in sim); only the
parallelism/capacity shrinks to target a small dev-board part.

## Files
| File | What |
|---|---|
| `synth_ku3p.tcl` | Vivado batch: read the 24 Q4_K sources → **(1)** `synth_design` of the raw product top `glm_q4k_system_cdc` → `util_synth.rpt` (pure resource fit, no pins); **(2)** `synth_design` + place + route of `bringup_harness` → `util_routed.rpt` + `timing.rpt` (routed Fmax) |
| `bringup_harness.v` | SYNTHESIZABLE P&R harness: wraps the exact product top but buries its thousands of wide memory-side ports (DDR/flash/KV/dequant buses + `logits`) behind an on-chip LFSR/CRC, so I/O collapses to 5 clock/control pins + 1 output → **routable**. Non-constant LFSR drivers keep the datapath from being pruned, so the routed fit is real. Verified: iverilog elaborates + yosys `hierarchy -check`/`check -assert` clean (no comb loop from the registered feedback) |
| `constraints.xdc` | `create_clock` for `host_clk` + `core_clk` + async CDC groups (timing/CDC only — **pin locations are board-specific**, add from your board's master XDC). Same clock/reset port names on both the system and the harness, so it applies to either top |
| `run_fit.sh` | runs the tcl in the Vivado Docker container (pins the container MAC for the license hostid, mounts `Xilinx.lic`, installs the tool's runtime libs, sources `settings64.sh`) |
| `out/` | reports land here (`util_synth.rpt`, `util_routed.rpt`, `timing.rpt`, checkpoints) |

## Run
Vivado ML must be installed into the `xilinx_install` docker volume first. Then from
the repo root:
```bash
bash fpga/run_fit.sh
```

### License (one-time, free)
Vivado ML **2026.1 refuses to launch without a valid license** — not even a trivial
batch script (verified: a bare `puts` script errors `valid license was not found`).
The **free "Vivado ML Standard"** edition covers XCKU3P (KU3P/KU5P are the free-tier
UltraScale+ parts), but you must generate a $0 license once:
1. <https://www.xilinx.com/getlicense> → sign in → **Vivado ML Standard** (Node-Locked, free).
2. Host ID: `run_fit.sh` pins the container MAC to `02:42:c0:a8:64:02`, so give the
   FlexLM host id **`0242c0a86402`** when the portal asks.
3. Save the generated `Xilinx.lic` to `/Users/Shared/xilinx/Xilinx.lic` (or export
   `XILINX_LICENSE=/path/to/Xilinx.lic`), then re-run `bash fpga/run_fit.sh`.

`run_fit.sh` auto-mounts the `.lic` and sets `XILINXD_LICENSE_FILE`; the fixed MAC
keeps the node-locked license valid across container runs.

> The portal now names the free tier **"Vivado Basic Tier License, Node Locked"**
> (the ML-Standard/WebPACK successor); it grants `Vivado_Synthesis` +
> `Vivado_Implementation` + `Vivado_Simulation`, which covers this whole flow.

### Docker Desktop resource footguns (learned the hard way)
Docker Desktop **validates** `settings.json` at boot. A resource value it rejects
(memory or disk **exceeding what the host can back** — e.g. a 150 GB VM disk on a
111 GB-free host) does not get clamped: Docker **factory-resets every resource
setting** (mem → 8 GB, disk → 60 GB) **and recreates the VM disk — wiping all
volumes**, including the 57 GB Vivado install. Rules that keep it stable:
1. Edit `settings.json` only with Docker **fully stopped** (wait for
   `com.docker.backend` to exit) — a shutdown flush overwrites live edits.
2. Keep `memoryMiB` ≥ 32768 (the compact-chip synth OOM-kills below that even
   with `maxThreads 4`) and `diskSizeMiB` comfortably **under** host free space.
3. Keep the extracted installer + `install_config.txt` (`/Users/Shared/xilinx_setup/`)
   — if the volume is ever wiped, the reinstall is one ~1 h batch command
   (`xsetup -a XilinxEULA,3rdPartyEULA -b Install -c install_config.txt`).

### Docker crash workaround (handled automatically by `run_fit.sh`)
With a valid license, Vivado 2026.1 in Docker then **aborts at launch** with
`realloc(): invalid pointer` — the FlexLM checkout (`libXil_lmgr11.so`) scans
network devices through `libudev`, and Vivado's bundled `libtcmalloc.so.4`
(loaded as `NEEDED` by the main binary, pure malloc interposition — zero direct
`tc_*` references) mixes allocators with glibc-internal allocations inside that
scan. Verified on Ubuntu 20.04 and 22.04 images; `LD_PRELOAD`/`/etc/ld.so.preload`
shims do **not** help because lmgr `dlopen`s libudev directly. The fix that works:
bind-mount an **empty stub** `.so` over the bundled `libtcmalloc.so.4` (removes the
interposition; everything runs on plain glibc). `run_fit.sh` builds and mounts the
stub automatically. Feature libs additionally need `libpixman-1-0 libcairo2
libpango-1.0-0 libglib2.0-0 libfreetype6 libfontconfig1` (also installed by the
script).

## Adjust for your board
- **Part**: edit `PART` in `synth_ku3p.tcl` (package/speed grade, e.g. `xcku3p-ffvb676-2-e`).
- **Fmax sweep**: tighten `core_clk` period in `constraints.xdc`; achieved Fmax = `1/(period − WNS)` (WNS from `out/timing.rpt`).
- **Pins**: add `set_property PACKAGE_PIN`/`IOSTANDARD` from your dev board's master XDC.

## Honest notes
- **First real fit** (DONE — see Results below): Vivado did surface issues the structural
  yosys gate could not (long combinational cones); iterated through the repipeline rounds,
  each re-proven bit-exact.
- **Wide top-level ports** (resolved): `glm_q4k_system_cdc` exposes thousands of
  memory-/logits-side bits (`logits`=VOCAB·16, `h_state`, DDR/Flash/KV/dequant buses).
  Full P&R would fail on **I/O count** on a real package. `bringup_harness.v` buries all
  of them behind an on-chip LFSR/CRC (expose only host pins), so the tcl's step (2) routes
  the harness for the real Fmax; step (1) still reports pure-product resources synth-only.
- **RAM**: compact-system P&R is memory-heavy; if `route_design` OOMs on the Docker
  allocation, raise Docker Desktop memory or run synth-only first.

## Results — MEASURED (Vivado ML 2026.1, 2026-07, compact config + `ACT_HW=1`)
Product top `glm_q4k_system_cdc`, `xcku3p-ffvb676-2-e`, synth-only utilization
(`util_synth.rpt`); harness P&R **completed** (place + route clean), timing from
`timing.rpt`.

| Resource | Used | Avail (XCKU3P) | Util % |
|---|---|---|---|
| CLB LUT | **142,320** | 162,720 | **87.5%** |
| CLB Register (FF) | ~100K+ (pipeline rounds added FFs) | 325,440 | ~33% |
| Block RAM (36Kb) | 0 | 360 | 0% (all storage in LUTRAM/FF) |
| DSP48E2 | 421 | 1,368 | 30.8% |
| **Fits XCKU3P?** | **YES** — places and routes, hold met | | |
| **Routed Fmax `core_clk`** | **46.5 MHz** (WNS −16.52ns @5ns target) | | |

**The fmax repipeline campaign (every round bit-exact, re-proven on the same
1155-test assembled golden + the full unit-TB battery):**

| Round | Change | Critical cone | WNS @5ns | Fmax |
|---|---|---|---|---|
| 4 (baseline) | first routable fit (`ACT_HW=1`) | rope trig+rotate, 98.4ns / 382 levels | −93.1 | 10.2 MHz |
| 5 | `rope_interleave_unit` → 10 stages | glm_act rsqrt, 58.5ns / 236 levels | −53.3 | 17.2 MHz |
| 6 | `glm_act` → 20 stages + `rmsnorm` reduce/rsqrt | **route-dominated** wide-bus, 21.2ns (59% wire) | **−16.5** | **46.5 MHz** |
| 7 | `glm_matmul_q4k` dequant+MAC → 5 stages | same route-dominated path (matmul was already sub-critical) | −16.5 | 46.5 MHz |

**Campaign closed at 4.6×.** The wall is no longer arithmetic: the worst path is
`u_moe/y_out → hbuf` — a wide expert-output bus crossing the die, 59% wire delay
at 87% LUT utilization. Moving past it needs physical work (floorplanning,
register duplication along the route, a bigger part at lower utilization), with
steep effort per MHz. 46.5 MHz sits in the bring-up demo's target band (the
compact die saturates the demo's NVMe-class stream), so the campaign stops here;
the 200 MHz-class number the full-bandwidth product needs is rung-②/③ work, and
the round-5/6/7 stage decompositions carry over to the ASIC unchanged.

**Which commit these numbers were measured at, and why they stay valid.** The
final fit run synthesized the tree as of `05639bf` (pre-RESIDENT). RTL edits
since then keep the fit-configuration netlist identical *by proof, not by
promise*: the harness elaborates `glm_q4k_system` at default parameters
(`RESIDENT=0`), and `make resident-equiv` formally proves (yosys sequential
equivalence, `equiv_simple`+`equiv_induct`) that the default-parameter module
equals the `05639bf` version; the CDC wrapper / harness diffs are parameter
plumbing at value 0, and the other touched sources are comment-only. The rule
going forward: an RTL change that alters the default-parameter netlist (like
every fmax-repipeline round did) re-runs the fit; a change proven
netlist-identical at the fit configuration carries the measured numbers, with
the proof named here.
