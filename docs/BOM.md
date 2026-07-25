# BOM & per-seat economics — the box across the 3 rungs

*What the box actually costs to build, at each rung of [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md).
The point: the FPGA is **not** the dominant cost — **memory + storage + board** are. Every figure is an
**order-of-magnitude [EST]**; exact FPGA quotes need a distributor, exact board cost needs a PCB-house
quote. Prices are single-unit / low-volume unless noted.*

> **Why this exists.** The FPGA-fit track ([`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md)) sets the FPGA class;
> the FPGA class + memory + storage set the BOM; the BOM sets the per-seat price; the per-seat price is
> what makes the [`ICP.md`](ICP.md) economics real. This doc closes that chain with actual numbers.
> For **how the individual parts get locked** (FPGA → DDR4/NVMe/power combos, and the fit
> measurement that gates them), see [`PART_SELECTION.md`](PART_SELECTION.md).

> **Local-device retarget (Q4_K).** The local track targets the published `unsloth/GLM-5.2-GGUF :
> UD-Q4_K_XL` (**467 GB**, ~38% smaller than the 753 GB FP8 checkpoint), so the memory/storage lines below
> are **conservative**: the hot-set and routed bytes scale down ~proportionally, which eases the DDR and
> NVMe sizing (a smaller hot-set cache and a 467 GB model still fits ~1 TB with room to spare). FP8 is
> preserved on tag **`fp8-verified-baseline`** ([`Q4K_RETARGET.md`](Q4K_RETARGET.md)).

---

## The cost shape (read this first)

The workload is memory-bandwidth-bound, so the box is **memory- and storage-dominated, not compute-dominated**:

- **Compute (FPGA)** — cheap relative to the rest, and cheaper still at ASIC volume. Even a mid FPGA is a
  minority of the BOM.
- **Fast DDR (bandwidth)** — the real cost driver, because *bandwidth* (channels/PHY) is what performance
  needs, and more channels = a bigger chip + more DRAM + a harder board.
- **NVMe (capacity)** — cheap per TB; the 467 GB Q4_K model fits ~1 TB with room for KV / overflow.
- **Board + power + enclosure** — rises steeply with signal speed (DDR5/PCIe Gen4/HBM = 8–12-layer
  controlled-impedance PCB, outsourced design + assembly).

So "make the box cheaper" ≈ "need less memory bandwidth" ≈ "accept lower tok/s" — the ladder, in cost form.

---

## Rung ① — prove-it box (now, ~$1–2 k)

Low-end **Kintex UltraScale+ (KU3P-class)** dev board + DDR4 + one NVMe. **Reduced-config demo** (a dev
board's DDR/storage can't hold the 467 GB Q4_K model); goal is *"real 753B-family RTL runs on real FPGA
silicon, offline, bit-exact to the ggml-Q4_K reference (`tools/q4k_ref.py`)"*. The FPGA fit / Fmax is
**MEASURED — DONE** (Vivado routed `glm_q4k_system_cdc` on XCKU3P: 142,320 LUT / 87.5% synth-stage (routed 141,298 LUT), 421 DSP, 0 BRAM,
46.5 MHz; [`../fpga/`](../fpga/README.md)); at 46.5 MHz the demo slice computes ~4,200–5,800 slice tok/s
(the correctness-demo speed of the reduced config, **not** a GLM-5.2 product number), while GLM-scale
NVMe-only streaming sits at ~0.5–1 tok/s [EST, measured-proxy — [`H_MEASUREMENT.md`](H_MEASUREMENT.md)].

| Line | Part (example) | ~Cost | Note |
|---|---|---|---|
| FPGA (on dev board) | KU3P-class board (e.g. a KCU-class or KU3P eval) | **~$1,000–2,500** | dev board = FPGA + power + clocks + JTAG bundled; buy the board, not the raw chip |
| DDR | on-board (dev board's own DDR4) | (incl.) | dev board ships with some DDR4 |
| NVMe | 1× M.2 NVMe 1 TB | ~$60–100 | holds a reduced-config weight image |
| Vivado | ML edition (KU3P may need paid) | ~$0–3,000/yr | WebPACK covers small parts; confirm KU3P tier |
| **Prove-it total** | | **~$1,000–2,500 + tool** | one unit, for the demo — not a product |

> This rung is **capex for the demo**, not a per-seat product cost. Its job is to convert `[EST]` → a
> measured Fmax ÷ cyc_per_tok (**DONE** — 46.5 MHz routed ÷ ~8.0–11.0 K cyc/tok → ~170–240 µs/token on
> the demo slice) and a real "it runs" video (board bring-up — **still open**). Cheapest path to the
> fundable proof.

---

## Rung ② — custom product board (post-seed, ~$3–6 k/box)

Custom PCB (outsourced artwork + assembly) carrying a **mid FPGA with DDR5 multi-channel or HBM** + big
DDR + multi-NVMe. This is the actual **shippable single-user box** at ~15–40 tok/s [EST]
(measured-proxy design points: ~13–24 at 90 GB DRAM + 100 GB/s, ~25–47 at 200 GB/s —
[`H_MEASUREMENT.md`](H_MEASUREMENT.md); the ~54–127 @ 225 GB + 200 GB/s cache band now belongs to
the rung-③ hybrid SKU — the rung-③ primary is full residency at a design point of ≈80 tok/s [measured-inputs EST],
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)).

| Line | Part | ~Cost | Note |
|---|---|---|---|
| **FPGA** | Versal / Agilex / HBM-class US+ (DDR5 or HBM, multi-PCIe) | **~$1,500–5,000** | the bandwidth-capable chip; a minority of BOM |
| **Fast DDR** | 64 GB DDR5 (multi-channel) *or* HBM (on-package, 16–32 GB) | ~$300–700 (DDR5) / (HBM in chip) | the hot-set cache; bandwidth is the cost, not GB |
| **NVMe** | 1–4 TB (1–2 drives over PCIe) | ~$100–400 | full 467 GB Q4_K model + KV overflow |
| **PCB** | 8–12-layer controlled-impedance, outsourced design | ~$300–800 (proto/unit; NRE separate) | DDR5/PCIe Gen4 signal integrity = many layers |
| **Assembly** | BGA reflow + PnP (turnkey) | ~$200–600/unit | vendor does it; BGA can't be hand-soldered |
| **Power / clock / connectors / enclosure / USB-C** | PMIC, oscillators, M.2/PCIe conn, case | ~$150–400 | |
| **Vivado/Quartus** | paid (amortized) | ~$3,000/yr / N units | tool, not per-box |
| **Rung-② box BOM** | | **~$2,500–6,000/unit** | + one-time NRE (PCB design ~$10–30 k, paid once) |

**One-time (NRE, not per-box):** custom PCB design/artwork outsourced **~$10,000–30,000+**, plus a few
board revisions. Amortized over units, negligible per-seat at any real volume.

---

## Rung ③ — SoC/ASIC (at volume, endgame)

> **(updated 2026-07: the rung-③ primary design point pivoted to FULL RESIDENCY** — 512 GB LPDDR5X
> (16×32 GB, 1024-bit on-package, ~1.1 TB/s) holds the whole ~467 GB checkpoint; cold storage = one
> commodity M.2 NVMe (boot-load ~70 s); design point **≈80 tok/s [measured-inputs EST]**; box **≥50–78 W [EST] floor** (R3 §4 — old ~40–60 W retired, never derived); board
> 120×80 mm; **BOM ~$1.8–2.4 k** — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md). The
> HBM/streaming shape below survives as the **hybrid upside SKU** (if GLM h≥0.75) and the
> >512 GB-checkpoint fallback.)

Custom silicon (HBM + many-channel PHY + near-memory Q4_K dequant). **~40+ tok/s, lower power, lower
$/seat** — but only after volume justifies the NRE.

| Line | ~Cost | Note |
|---|---|---|
| **ASIC NRE** (masks, tapeout, IP) | **~$1 M–10 M+** one-time | mature node (not bleeding-edge — bandwidth-bound, not compute-bound); dominant risk |
| **Per-unit silicon** (at volume) | ~$50–300/chip | far below FPGA once amortized |
| **HBM** (on-package) | ~$100–400 | the bandwidth source |
| **NVMe + board + assembly** | ~$300–800 | simpler board than FPGA (integration on-die) |
| **Rung-③ box BOM (at volume)** | **~$1,000–2,000/unit** + amortized NRE | the cost-down + perf + power win the user flagged |

ASIC only makes sense once **unit volume × (FPGA-cost − ASIC-cost) > NRE** — i.e. at product-market fit /
Series-B scale. Sequenced last, on purpose.

---

## Per-seat economics — does it sell?

The pitch is **not** "cheap tokens" — it's *"the only turnkey way to run a frontier model where the cloud
can't go, offline, at a seat price"* ([`ICP.md`](ICP.md)). So the comparison that matters:

| Option | Frontier 753B? | Offline / air-gapped? | ~Cost per seat |
|---|---|---|---|
| Cloud frontier API | ✅ | ❌ (disqualifies the ICP) | ~$20–200/mo — *but banned* |
| Mac/GPU + 70 B local | ❌ (quality gap) | ✅ | ~$3–6 k one-time |
| **Big-RAM workstation — the *same* `UD-Q4_K_XL` GGUF on llama.cpp** | ✅ (identical file) | ✅ | **~$5–15 k [EST]** one-time (~512–768 GB DDR5 to hold 467 GB) |
| 8×H100 self-host 753 B | ✅ | ✅ | **~$250–400 k** (shared, + power + MLOps) |
| **This box — rung ②** | **✅** | **✅** | **~$3–6 k/box (one seat) + support** |
| **This box — rung ③ (volume)** | ✅ | ✅ | **~$1.8–2.4 k/box** (full-residency SKU [EST] — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)) |

**Honest comparison — the workstation, not the H100.** The real nearest alternative is **not** 8×H100; it is
a **big-RAM workstation running the identical `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` on llama.cpp** (~512–768 GB
DDR5 to hold the 467 GB model, **~$5–15 k [EST]**). That box is *also* full-frontier and *also* fully
offline — on raw capability it is a **tie** (it runs the same file this project targets). What the appliance
sells against it is **turnkey seat-price + support** (no MLOps, no llama.cpp tuning), a **purpose-built
memory/streaming datapath** (on rungs ①/② and the hybrid SKU, NVMe-streamed experts instead of paying to
keep all 467 GB in expensive DRAM; the rung-③ primary SKU pivoted to full 512 GB-LPDDR5X residency —
[`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)) and — on the funded rungs — **lower power / form factor**. Note also both are
running our-own-scoped Q4_K arithmetic vs the ggml reference, **not** a validated bit-match to llama.cpp's
runtime.

**The number that sells** (against the *datacenter* alternative): a rung-② box at **~$3–6 k** vs **8×H100 at
~$250–400 k** = **~50–100× cheaper** for the offline-753B use case. That gap is real but it flatters us — the
$5–15 k workstation above is the tighter comparison. Still, for a buyer whose alternative is a $400 k
datacenter build (or *nothing*, because the cloud is barred), a **$5 k provably-offline frontier box** is a
trivial line item — legal already pays $100–500/seat/mo for Westlaw-class tools; a per-seat appliance fits.
All figures **[EST]**.

## Honest limits

- All prices are **order-of-magnitude [EST]** — FPGA needs a distributor quote, board needs a PCB-house
  quote, ASIC NRE is a wide band. Treat as ranges, not commitments.
- Rung-② tok/s (~15–40) is the funded number (measured-proxy design points ~13–47 —
  [`H_MEASUREMENT.md`](H_MEASUREMENT.md); the ~54–127 225 GB-cache band is now the rung-③ hybrid-SKU
  case, the rung-③ primary being full residency at ≈80 [measured-inputs EST] —
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)); the **near-term demo (rung ①) is
  reduced-config** (GLM-scale NVMe-only streaming ~0.5–1 tok/s [EST]).
- BOM is **memory/storage/board-dominated**; the FPGA is a minority. "Cheaper box" means "less bandwidth"
  means "lower tok/s" — the ladder, in money.
- Software / host / support / margin are **on top** of these hardware BOMs (a product sells above BOM).
- The honest floor competitor is a **~$5–15 k big-RAM workstation running the same `UD-Q4_K_XL` GGUF** — same
  model, same offline property. The appliance's edge is turnkey/seat-price/power/form-factor, **not** a
  capability the workstation lacks; price and pitch must be argued on that basis, not on exclusivity.
