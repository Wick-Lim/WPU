# Hardware performance ladder — prove cheap, scale on funding

*The honest hardware plan for the local single-user box. **Every rung below was sized against GLM-5.2**
(753B-param MoE, ~40B active/token, 467 GB weights + ~94 GB KV at 1M context). This branch's target,
GLM-5.3-Flash, changes both of those terms — and not by the same factor — so read the
[GLM-5.3-Flash re-sizing](#glm-53-flash-re-sizing-what-changes-and-what-does-not) section before quoting
any capacity here for it. The tok/s figures are a separate matter and do **not** carry over at all: they
divide by a speculative `A_eff` that is unmeasured for GLM-5.3-Flash.*

*Original framing:
It replaces the earlier flat "64 GB DDR5 / ~100 GB/s / 25–40 tok/s" assumption with a **staged ladder**:
the performance you get is set by **memory bandwidth** — specifically the **NVMe/PCIe bandwidth that
streams the routed experts** — memory bandwidth is set by the **FPGA/silicon's IO + PHY**, and that is
set by **how much money is in the build**. So the plan is: **prove it works cheap → raise → scale.***

> All tok/s here are **[EST]** — first-order projections from the bandwidth roofline
> (`tok/s ≈ sustained streaming BW / [(1−h)·routed footprint] · K`), **not** measured silicon. Read the
> spec multiplier `K` as **A/U(K) ≈ 1.1–1.3× at K=4** per the measured union factor U(K), not ~2× —
> and `h` now has measured-proxy values (OLMoE trace) — see [`H_MEASUREMENT.md`](H_MEASUREMENT.md).
> *(Updated 2026-07: U(K) is now **GLM-family MEASURED** — GLM-4.5-Air MoE-gate trace, U(4)=2.60–2.71
> — superseding the first-pass OLMoE-proxy U; the **adaptive spec-chain is adopted and RTL-landed**
> (K∈[1..5]); and r is now **MEASURED** too (job B vLLM MTP sweep, GLM-4.5-Air: r₁=0.87 with steep
> per-position decay, A_eff plateau ~2.9 → memory-bound optimum **K=1–2**, residency design point
> ≈80 tok/s [measured-inputs EST] — [`H_MEASUREMENT.md`](H_MEASUREMENT.md) 3rd measurement). And the **rung-③ primary design point pivoted to full residency** — h=1 by construction
> there; h-curves stay relevant only for the hybrid upside SKU. See the pivot section below +
> [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)*
> The **FPGA fit + routed Fmax are MEASURED**: Vivado ML 2026.1 full place&route of
> `glm_q4k_system_cdc` on XCKU3P (compact config + ACT_HW=1) — **141,298 LUT routed**
> (`fpga/results/util_routed_ku3p_acthw1.rpt`; 142,320 / 87.5% at the synth stage), 421 DSP, routed
> Fmax **46.5 MHz** after a closed 4.6× repipelining campaign, every round re-proven bit-exact on the
> 1155-test assembled golden. But there is still **no running board**, so every tok/s below stays
> **[EST]** until bring-up measures a real Fmax ÷ cyc_per_tok on hardware. Only rung ① is a
> near-term buildable proof; ②③ are funding-gated projections.

> **Local-device retarget (Q4_K).** `main` develops the **Q4_K local-inference track** — the target
> weight store is the published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` (**~467 GB**, ~38% smaller than the
> 753 GB FP8 checkpoint). At ~0.6 B/param avg vs FP8's ~1.0, Q4_K reads **~1.6× fewer bytes/token**, so on
> a bandwidth-bound box it is **~1.6× faster than FP8** at the same interface [EST]. "Bit-exact" on this
> track means **bit-exact to our ggml-Q4_K reference `tools/q4k_ref.py`** — mixed-type Q6_K·Q8_0·F16
> RTL consumers are **DONE** (`make mixedtype`), and the reference's **dequant layer is proven on
> real GGUF bytes — Q4_K/Q6_K/Q8_0** ([`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)); the
> whole-runtime llama.cpp check is out-of-contract and the real 467 GB file has not been consumed
> end-to-end (see [`../README.md`](../README.md)). FP8 is preserved on tag **`fp8-verified-baseline`**
> ([`Q4K_RETARGET.md`](Q4K_RETARGET.md)).

---

## The one thing that sets performance: **memory bandwidth, not compute**

The workload is **NVMe/PCIe-bandwidth-bound**. The Q4_K compute die is small and sits largely idle
(~75–80% starved) behind the memory system (documented throughout — [`ULTRA_PERF.md`](ULTRA_PERF.md),
[`MINIATURIZATION.md`](MINIATURIZATION.md)). So **compute is cheap; the wall is reading weights** — every
token streams **~25 GB** of weights (~40B active params × ~0.6 B/param at Q4_K), split into two very
different byte pools:

- **~11 GB hot-set touch** (MLA projections all layers + shared expert + dense FFN + router +
  embed/LM-head; resident hot partition ~17 GB — canonical byte constants:
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2). These are the **same** bytes every token, so
  they **cache in the DDR working set** — read once, reused.
- **~14 GB routed-expert bytes** (top-8 of 256 experts per layer). These **change every token** with the
  router's choice, so they **cannot be cached** at a useful hit rate — routing entropy caps it, and the
  predictor-prefetch lever is a **MEASURED no-op** on the trace harness. They must be **streamed fresh
  from NVMe/Flash every token**. **This ~14 GB is the wall.**

So the bit-exact roofline is `tok/s ≈ sustained NVMe/Flash streaming BW / ~14 GB`. The ~17 GB hot partition
lives in DDR and is free after the first token; **DDR bandwidth only starts to matter once routed experts
also hit in the DDR cache** — which the routing entropy above limits. **The streaming bandwidth is set by
the number of PCIe lanes / NVMe drives (and, on the upper rungs, DDR5 channels / HBM stacks), which is set
by the chip's IO pins + hard PHYs** — a physical property of the silicon you buy, *not* something RTL can
add (our `ddr5_xbar`/`flash_xbar` already parameterize the channel count; the ceiling is the chip's pins).
**More bandwidth = a bigger/newer chip and more drives = more money.** That single fact produces the ladder.

---

## The ladder

| Rung | Silicon | Streaming path | tok/s [EST] | Bit-exact? | ~Box BOM | Funding | When |
|---|---|---|---|---|---|---|---|
| **① Prove-it (cheap)** | low-end FPGA (Kintex US+ **KU3P** class) + DDR4 hot-set cache | 1–2 NVMe (~7–14 GB/s) … striped ~14 drives (~100 GB/s) | **~0.5–1 … ~5–8** | **yes** (bit-exact throughout) | ~$1–2 k (floor) → higher w/ drive array | self / minimal | **now (the demo)** |
| **② Custom board** | mid FPGA (Versal / Agilex / HBM-class US+) DDR5 multi-ch or HBM | DDR5 8–12 ch or HBM (~400 GB/s–1 TB/s) feeding the working set | **~15–40** | **contingent** — needs expert-cache hit rate *or* non-bit-exact pruning | ~$3–6 k | seed | post-raise |
| **③ SoC / ASIC** | custom silicon — **primary (2026-07 pivot): 512 GB LPDDR5X full residency** (16×32 GB, 1024-bit on-package, ~1.1 TB/s; [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)); HBM stays the long-range ceiling | whole ~467 GB checkpoint DRAM-resident; cold store = one M.2 NVMe (~70 s boot-load; no streaming tier) | **≈80 residency design point [measured-inputs EST]** (~95 if GLM-5.2 MTP hits published depth; ~120 HBM-ceiling aspirational) | **yes** (residency ⇒ h=1 by construction) | ~$1.8–2.4 k | Series B+ / volume | at scale |

> **Update — measured-proxy design-point menu ([EST], MEASURED-PROXY h/U inputs;
> [`H_MEASUREMENT.md`](H_MEASUREMENT.md), [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md)):**
> NVMe 1–2 (no multipliers) ~0.5–1 tok/s; 90 GB DRAM expert cache + 100 GB/s → 13–24;
> 90 GB + 200 GB/s (ONFI 64ch) → 25–47; 225 GB + 200 GB/s → 54–127 (formerly the "100 tok/s" design
> point — now the **hybrid-upside-SKU** case only, contingent on GLM h ≥ 0.75; the primary rung-③
> point is **full residency, design point ≈80 tok/s [measured-inputs EST]** — see the pivot section below).
> Measured residency-only h (OLMoE proxy): ~20% of pool cached → h=0.36–0.60; ~50% → 0.72–0.88
> (LRU collapses below ~10%) — with the residency pivot these h-curves matter only for the hybrid SKU.
> Spec-chain amortization is A/U(K), not ~2× — U(K) is now **GLM-Air MEASURED** (U(4)=2.60–2.71,
> superseding the OLMoE-proxy U), and the adaptive spec-chain (K∈[1..5]) is adopted in RTL.

*Per-rung parts, box BOM, and per-seat economics: [`BOM.md`](BOM.md) — all cost/economics figures are
**[EST]/[PENDING]** planning numbers, not quotes. Short version — the BOM is memory/storage/board-dominated,
the FPGA is a minority, and a rung-② ~$3–6 k box is order-of-magnitude cheaper than an 8×H100 self-host
for the offline-753B use case [EST].*

Each rung is **the same RTL** — whose **Q4_K GEMM core is bit-exact vs the ggml reference**
(`glm_matmul_q4k` 160/160), and the **assembled end-to-end model golden is DONE**
(`make model-q4k` 1155 + `make model-q4k-acthw` 1155; [`../README.md`](../README.md)). Only the memory interface it drives changes; nothing about the model
changes across rungs — **just the bandwidth the silicon can feed it.**

### ① Prove-it — the near-term goal
A **low-end Kintex UltraScale+ (KU3P-class)** dev board, DDR4 holding the **~17 GB resident hot partition**, and an
NVMe/Flash array streaming the **~14 GB routed experts**. Throughput is set by that storage array, and it
is **bit-exact at every point on the band**:

- **BOM floor — 1–2 NVMe (~7–14 GB/s) → ~0.5–1 tok/s [EST].** The honest cheap-box number. Slow, but real.
- **4 NVMe (~28 GB/s) → ~2 tok/s [EST].**
- **Striped ~14 drives (~100 GB/s) → ~5–8 tok/s [EST]** — the top of rung ①, still bit-exact; the striping
  strategy is [`FLASH_STRIPING.md`](FLASH_STRIPING.md).

**Slow, but real and bit-exact** — the point of this rung is not speed, it's proving *"the full GLM-5.2
runs on real FPGA silicon, offline, producing the reference model's tokens."* It is a **reduced-config**
demo first, since a dev board's small DDR/Flash can't hold 467 GB — see
[`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md). That working demo is what makes rung ② fundable.
**Prove cheap, then raise.**

### ② Custom board — post-seed
A **custom PCB** (artwork + assembly outsourced) carrying a **mid-tier FPGA with DDR5 multi-channel or
HBM** (Versal / Agilex / an HBM-class UltraScale+) + multiple NVMe over more PCIe lanes. DDR5 doubles
per-channel BW vs DDR4; HBM gives ~460 GB/s/stack — either reaches **~400 GB/s–1 TB/s to the working
set**. But note the **honesty knob**: that bandwidth feeds the DDR/HBM-resident **hot partition (~17 GB; ~11 GB touched/token)** at full
rate; the **~14 GB routed experts still change every token**, so **~15–40 tok/s [EST] is *contingent***,
not free. You reach it only by either

- **(bit-exact) landing routed experts in the DDR/HBM cache** at a real hit rate — now **measured
  (proxy)**: residency-only h=0.36–0.60 with ~20% of the pool cached (~90 GB GLM-scale), 0.72–0.88 at
  ~50% ([`H_MEASUREMENT.md`](H_MEASUREMENT.md)); the predictor-prefetch path remains a
  **MEASURED no-op** ([`ULTRA_PERF.md`](ULTRA_PERF.md)); or
- **(NOT bit-exact) activation-sparsity / expert pruning** — a separate model-quality decision that trades
  the bit-exact guarantee for fewer streamed experts.

Absent either, the box stays on the rung-① NVMe wall even with fast DDR. The two hardware routes
(DDR5-many-channels vs newer-FPGA-DDR5 vs HBM) **converge near the same ~$3–6 k build** — the cost is the
memory-bandwidth silicon, whichever way you buy it. This is the interactive product rung.

### ③ SoC / ASIC — at volume
**Reframe of the earlier "ASIC out of scope".** That call was made under *"compute-bound → ASIC's compute
edge is wasted"*. But the real bottleneck is **memory bandwidth (IO pins + PHY)**, and **an ASIC is
exactly what breaks the FPGA's IO/PHY ceiling** — it can integrate **HBM stacks + many-channel controllers
+ near-memory Q4_K compute** that no FPGA package offers, at **~TB/s**, with **lower per-unit cost and
power** once amortized over volume. An HBM3 ceiling (~3 TB/s) roofs at **~120 tok/s** — but **aspirational**:
the ~467 GB Q4_K checkpoint **does not fit** in an HBM budget (≤192 GB), so an *HBM* ASIC still needs the
tiered NVMe→HBM streaming path, not an all-HBM resident model *(updated 2026-07: the primary rung-③
design point sidesteps this with **512 GB LPDDR5X full residency** — see the pivot below; the HBM path
here is the long-range ceiling)*. So ASIC is **not off the table — it is the endgame**:
its multi-million NRE and months–years lead time only pay off **at manufacturing volume**, exactly where a
shipping product lives. **Sequence: FPGA (rungs ①②) to prove + reach product-market fit → ASIC (③) when
volume justifies the NRE and demands the lower $/seat + higher tok/s + lower power.** Not now (no volume,
no capital); real later (that's how bandwidth-bound silicon products scale).

---

### Rung-③ memory-tier decision — v2 PIVOT (2026-07-10): 512 GB LPDDR5X FULL RESIDENCY

**The v1 fix below (256 GB hybrid) was re-decided after the min(NAND, DRAM)
correction** ([`H_MEASUREMENT.md`](H_MEASUREMENT.md) v2): cache HITS also cross
the DRAM tier, so the hybrid's honest numbers are ~42 tok/s at 512-bit and
~84 at 1024-bit *only if GLM h ≥ 0.75* (an unmeasured bet — at h=0.6 it falls
to ~45). The new primary design point:

- **LPDDR5X 512 GB (16×32 GB, 1024-bit, ~1.1 TB/s), whole checkpoint resident**
  → design point **≈80 tok/s [measured-inputs EST]** (directly `~1.1 TB/s ÷ 13.87 GB/token ≈ 80` —
  the canonical per-token constant already folds in the measured spec-chain amortization
  A_eff=1.87 at K=1; the old "~71 base × spec-chain" derivation is retired with the 15.4 GB/tok
  constant, [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2. U(K) **and** the accept rate r both
  GLM-family measured — job B's vLLM MTP sweep put the memory-bound optimum at K=1–2; ~95 if
  GLM-5.2's deeper MTP hits its published accept depth), **deterministic — no h dependence at all.**
- **Deletes** the ONFI 64ch controller RTL (LDPC/bad-block) from the critical
  path and the 40–90 W NAND-read power (box power: honest floor **≥50–78 W [EST]**
  v3-volume / **≥64–99 W [EST]** v3-proto — the old "~40–60 W" figure is retired,
  never derived; [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §4). Cold storage = one
  commodity M.2 NVMe (boot-loads 467 GB in ~70 s; no streaming RTL).
- Costs: +$800–1,700 memory vs the hybrid; 1024-bit = Apple-M-Ultra-class
  packaging (16 packages double-sided, on-substrate routing — proven practice,
  our hardest packaging item); capacity ceiling ~512 GB (next-gen bigger
  checkpoints fall back to the hybrid).
- **The 1024-bit hybrid (~84 tok/s) survives as the upside SKU** if the GLM h
  measurement lands ≥0.75 — keep the ONFI pads on-die, unbonded in the
  residency SKU, so both SKUs share one die.
- Full appliance concept spec (board 120×80 mm, on-substrate packaging, power
  v1→v3 history, clock/node/lane derivation, competitive bracket):
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).
- **Execution order re-decided (2026-07-11): prototype-first.** A retail-parts
  supply audit found 32 GB LPDDR5X packages are OEM/NDA-only while **24 GB
  (9.6 Gbps) is buyable retail** — so the rung-③ build sequence now leads with
  **v3-proto: 24 GB ×20, 1280-bit, 480 GB resident, PCB-HDI direct mount
  (~130×110 mm board), ~1.54 TB/s → ~110 tok/s [EST]** (which presumes a MAC array
  sized to consume 1.54 TB/s — ~12.7K lanes @490 MHz (dedicated per-phase engines), re-derived in
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §3; a smaller array, not bandwidth,
  becomes the bottleneck), deferring both the
  NDA procurement and the on-substrate-16-package packaging (the two hardest
  non-silicon items) to the volume SKU. One die serves both (1280-bit superset;
  volume SKU bonds 1024). The honest gate that does NOT move: the SoC tapeout
  itself (LPDDR5X PHY IP + 12–16 nm NRE). Details:
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §5c.

<details><summary>v1 decision (2026-07, superseded — kept for the reasoning trail)</summary>

### Rung-③ memory-tier decision (2026-07, v1 — SUPERSEDED by the pivot above)

The rung-③ SoC's memory system is **decided**: **LPDDR5X 256 GB (8×32 GB packages,
512-bit, ~550 GB/s) as the h-cache tier + ONFI-direct NAND 64ch (~200 GB/s, raw
1 TB) as the stream tier** — the measured-proxy 54–127 tok/s design point
([`H_MEASUREMENT.md`](H_MEASUREMENT.md)). The reasoning trail, kept honest:

- **CXL 512 GB — REJECTED.** CXL solves capacity (which NAND already gives for
  ~$0.05–0.1/GB) and not bandwidth (which is our wall): one x16 Gen5 link is
  ~50 GB/s effective → ~4–5 tok/s, *below* the NAND array at ~30× the $/GB, and
  a multi-link CXL root drags in server-class host silicon. Same PCIe wall the
  offloading literature sits behind ([`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md)).
- **512 GB full residency — REJECTED.** Counter-intuitive but measured-informed:
  full residency forces the capacity onto the *slow* medium (DDR5 DIMMs,
  ~350 GB/s at 8ch) → ~25–35 tok/s [EST], **below** the 225-GB-class hybrid,
  because the hybrid serves the hot 72–88 % (measured h at 50 % pool) from
  *faster* LPDDR5X and lets $100 of NAND absorb the miss tail. Hierarchy beats
  hoarding.
- **LPDDR5X sizing**: packages top out at 32 GB (x64), so 8 packages = 256 GB on
  a 512-bit bus is the practical ceiling class (Apple M-Max/Strix Halo/DGX Spark
  prove the packaging + consumer price point). The 512-bit width is REQUIRED,
  not a luxury: at the 54–127 tok/s point the cache tier itself carries several
  hundred GB/s (hits + hot set), ~70–90 % of 550 GB/s. A 4-package/128-GB SKU
  lands in the 25–47 tok/s class — the two SKUs share one die.
- FPGA rungs keep DDR4/DDR5 DIMMs (no LPDDR5X hard controllers in FPGAs);
  LPDDR5X enters with the ASIC PHY IP at rung ③.

</details>

## Why this ordering is the right bet

- **De-risk before spend.** A ~$1–2 k FPGA proves the RTL on real silicon before any ~$3–6 k custom board
  or multi-million ASIC. Skipping straight to expensive hardware bets money on unverified silicon behavior.
- **The demo, not the spec, raises the money.** "GLM-5.2 runs bit-exact (vs the ggml reference) on an FPGA,
  offline" (rung ①) is a stronger pitch than a datasheet promise of 40 tok/s. Investors discount `[EST]`;
  they fund a working box.
- **Each rung funds the next.** Prove-it → seed → custom board → PMF → Series B → ASIC. Standard
  bootstrapped-hardware sequencing; no rung asks for money the previous rung hasn't justified.
- **Same moat throughout.** Offline / air-gapped, full-frontier, Q4_K-native — unchanged on every rung.
  Only the tok/s the silicon can feed goes up. The [`ICP.md`](ICP.md) buyer (offline is *mandatory*) values
  *"it runs at all, provably local"* on rung ①, and pays more for rung ②'s speed.

## GLM-5.3-Flash re-sizing: what changes, and what does not

Reproduce every number in this section:

```sh
python3 tools/glm53_flash_memory_budget.py
```

It parses the model constants straight out of `configs/full_glm53_flash.vh`, so it cannot drift from
the locked config, and it keeps the three evidence classes apart: `[measured]` weight bytes from the
GGUF census, `[derived]` KV/state arithmetic under stated precision assumptions, `[EST]` everything
vendor-side.

### The two terms move by different factors

| | GLM-5.2 (what the rungs were sized for) | GLM-5.3-Flash | ratio |
|---|---|---|---|
| weights | 467 GB | **199.7 GB** `[measured]` | 2.3× |
| KV @ 1M context | ~94 GB | **11.8 GB** `[derived]` | **7.6×** |
| DSA indexer keys @ 1M | — | 0.37 GB `[derived]` | |
| KDA recurrent state | — | 0.148 GB, **constant in context** | |
| **total resident @ 1M** | **~561 GB** | **212 GB** | 2.6× |

The KV term collapses for two independent reasons, and both are structural rather than a tuning
choice: **only 11 of 45 layers cache at all** (the other 34 are KDA linear-attention blocks carrying a
fixed-size recurrent state), and **NoPE removes the rotary tail** — `attention.key_length` equals
`kv_lora_rank` exactly, where GLM-5.2's cached key was `kv_lora 512 + rope 64`. Per token that is
`11 × 512 × 2 B = 11 KiB`, against GLM-5.2's `78 × 576 × 2 B = 87.8 KiB`.

**A 1M-context memory budget carried over from GLM-5.2 is wrong for this model by about 2.6×, in the
favourable direction.** Stating it that way round matters: the error is not conservative, it is just
wrong, and it would mis-size a board.

### The trap: capacity fell, unit count must not

Bus width and aggregate bandwidth follow the **number of packages or stacks**, not the total capacity.
Fitting the model into fewer units is the obvious move once capacity drops 2.6×, and it is the wrong
one — units are carried for bandwidth.

All tok/s below use the **full 1 M-context denominator, 14.795 GB/token** (weights
+ KDA state + DSA KV + indexer) — the same one the tok/s section uses, not the
weights-only 14.118, which would print two different numbers for one config.
`data pins` is the die's memory PHY width, and it is the point of the table: for
LPDDR5X it equals `units × 64`, so halving the packages halves the PHY *and* the
bandwidth together.

| tier | units | capacity | data pins | bandwidth | tok/s `[EST]` | |
|---|---|---|---|---|---|---|
| LPDDR5X | 16 × 16 GB | 256 GB | 1024-bit | 1.10 TB/s | **74** | full rung-③ speed at half the capacity |
| LPDDR5X | 8 × 32 GB | 256 GB | **512-bit** | 0.55 TB/s | **37** | same capacity, **half the PHY and half the throughput** — the low-cost variant |
| LPDDR5X | 16 × 32 GB | 512 GB | 1024-bit | 1.10 TB/s | 74 | rung-③ as originally specced |
| HBF | 1 × 256 GB | 256 GB | — | 1.60 TB/s | 108 | model fits one stack — but one stack is one stack's bandwidth |
| HBF | 2 × 256 GB | 512 GB | — | 3.20 TB/s | **216** | rung-④ design point, re-derived |
| HBM3E | 6 × 36 GB | 216 GB | — | 7.20 TB/s | 487 | newly reachable — see below |
| HBM4 | 4 × 64 GB | 256 GB | — | 8.00 TB/s | 541 | newly reachable — see below |

So rung ③ for GLM-5.3-Flash has **two honest answers, not one**, and which is right depends on the
target:

- **16 × 16 GB** — 256 GB, 1024-bit, ~74 tok/s. Full rung-③ speed at half the capacity.
- **8 × 32 GB** — 256 GB, 512-bit, ~37 tok/s. The same capacity from **half the packages**, and with
  it **half the die's memory PHY**: fewer pins, less PHY area, less IO power, a simpler board. Half
  the throughput is the price.

An earlier revision of this document called the first one "the" answer. That was wrong. 8 × 32 GB is
not a mistake to be corrected — it is a legitimate low-cost design point, and it is the one to pick if
the target is a cheap compact box rather than ~74 tok/s. **Note it is the package COUNT that is the
lever, not the capacity: 16 × 32 GB (512 GB) and 16 × 16 GB (256 GB) run at the same speed.**

The same structure holds at rung ④: 199.7 GB fits a **single** HBF stack by capacity, and a 1-stack
box is a real, cheaper, ~108 tok/s design point — the second stack buys bandwidth, not capacity. In
every tier the capacity headroom is a genuine *choice*, not a free win: spend it on fewer units and
you spend throughput with it.

### What the collapsed KV does to the two-store split

Rung ④ specs a **~96 GB HBM** tier because GLM-5.2's KV reaches ~94 GB at 1M context. GLM-5.3-Flash
needs **11.8 GB** — that tier is oversized by ~8×.

This is worth flagging as an architectural question rather than a sizing tweak, because **the size of
the KV was the motivation for splitting the stores in the first place.** The split is justified in this
document by KV being "random-access and latency-sensitive — the one pattern flash latency cannot
serve". That argument is about *access pattern* and it still holds at 11.8 GB. But the *capacity*
argument for a dedicated HBM stack largely evaporates, and 11.8 GB is small enough that other
placements become worth pricing. This document does not resolve that; it records that the premise moved.

### Newly reachable: all-HBM residency

At 212 GB resident, the whole model fits in HBM — 6 HBM3E stacks (216 GB) or 4 HBM4 stacks (256 GB).
GLM-5.2 at ~561 GB could not: it would need 16 HBM3E or 9 HBM4 stacks *just to fit*. This is a genuinely
new option that GLM-5.3-Flash opens.

It is **not** a faster version of the appliance, and should not be quoted as one:

- **Power.** 8 TB/s at DRAM-class 4–7 pJ/bit is **256–448 W for the memory rail alone** `[EST]`. The
  appliance envelope in this document is ≥50–78 W *total*. This is a workstation/server bracket.
- **Volatility.** HBM is volatile, so the boot-load and the persistent store come back — which is
  precisely what HBF was chosen to eliminate ("no ~70 s boot-load → instant-on").
- **Die.** 567 tok/s is 9.5 T MAC/s — ~9,500 lanes at 1 GHz, ~4,700 at 2 GHz. Fewer than GLM-5.2 needs
  for the same rate (16.7B active vs ~40B), but measured lane scaling here is **sublinear** (4× lanes →
  ~2.40×), so that is a floor on the silicon, not a bill of materials.

For a desktop appliance HBF remains the right answer for the weight tier. All-HBM is a different product.

### The tok/s estimate, with the full denominator

The roofline in this document divides **weight** bytes by bandwidth. On
GLM-5.3-Flash that under-counts: two terms are read every token that GLM-5.2 does
not have, and one of them grows with context.

| context | weights | KDA state | DSA KV | indexer | **total** |
|---|---|---|---|---|---|
| 32 K | 14.118 | 0.285 | 0.023 | 0.012 | **14.438 GB/tok** |
| 256 K | 14.118 | 0.285 | 0.023 | 0.092 | **14.519 GB/tok** |
| 1 M | 14.118 | 0.285 | 0.023 | 0.369 | **14.795 GB/tok** |

- **KDA state** — the 34 linear-attention layers read *and write* their
  `[64, 128, 128]` fp32 recurrent state once per token: `2 × 142.6 MB`. It does
  not grow with context, but it is not free.
- **DSA KV** — only the top-2048 selected latents, so this term is *flat* past
  2048 tokens. That is the sparse-attention win.
- **indexer** — the pooled key cache must be *read* to be scored, so this one
  grows linearly and is the largest context-dependent term at 1 M.

Weights still dominate at every context, so the bandwidth-bound model holds — but
the honest denominator is 2–5 % above the weights-only figure.

**Resulting tok/s `[EST]`, UNAMORTIZED** (no speculative decode — this model's
accept rate is unmeasured):

| rung | bandwidth | 32 K | 256 K | 1 M |
|---|---|---|---|---|
| ① prove-it FPGA (KU3P + NVMe) | ~4 GB/s | 0.3 | 0.3 | 0.3 |
| ② custom board (DDR5/HBM) | ~0.6 TB/s | 41.6 | 41.3 | 40.6 |
| ③ SoC, 256 GB LPDDR5X as 16 × 16 GB | 1.1 TB/s | **76.2** | **75.8** | **74.3** |
| ④ HBF 2-stack | 3.2 TB/s | 221.6 | 220.4 | 216.3 |
| ④′ all-HBM (HBM4 4-stack) | 8.0 TB/s | 554.1 | 551.0 | 540.7 |

For scale: GLM-5.2 needs `A_eff = 1.87` to reach ~79 tok/s at 1.1 TB/s (43 raw).
GLM-5.3-Flash reaches **74–76 with no speculation at all**. If its MTP head turns
out to amortize like GLM-5.2's — **not measured, and not a number this document
adopts** — rung ③ would land near 139 and rung ④ near 404. Those two are shown
only to size the prize, and must not be quoted as this design's figures until the
accept rate is measured on this model.

Reproduce the whole table: `python3 tools/glm53_flash_memory_budget.py`.

### Two reasons the high-bandwidth rows are optimistic

**The DSA indexer stops being free at long context.** Every tok/s in this document divides weight bytes
by bandwidth — a model that assumes the weight stream is the whole story. Modelling the indexer as
scoring each pooled position once per token:

| context | indexer MAC/token | vs the weight MACs |
|---|---|---|
| 32K | 0.37 G | 2.2 % |
| 128K | 1.48 G | 8.8 % |
| 256K | 2.95 G | 17.6 % |
| **1M** | **11.81 G** | **70.5 %** |

Parity lands at ~1.49 M tokens. Below ~128K this is noise. At the full 1M context the machine is no
longer purely weight-bandwidth-bound, and every tok/s figure here is derived from exactly the model
that stops applying there. (GLM-5.2 had an `index_topk_freq` to amortize this; GLM-5.3-Flash's config
publishes no such field, so no amortization is assumed and these are an upper bound.)

**No speculative amortization is included.** Every GLM-5.3-Flash tok/s above is the *unamortized*
figure. The MTP head exists (`nextn_predict_layers = 1`), so the mechanism carries over and the real
numbers should be better — but `A_eff` and the accept rate are GLM-5.2 properties that do not transfer,
and quoting a divisor this model has not been measured for would be inventing the input. See
[`GLM53_FLASH_PORT.md`](GLM53_FLASH_PORT.md) §4.3.

For scale: GLM-5.2 needs `A_eff = 1.87` to reach ~79 tok/s at 1.1 TB/s (43 tok/s raw). GLM-5.3-Flash
reaches **78 tok/s at the same bandwidth with no speculation at all**.

---

## Rung ④ (future, memory-tech-dependent): HBF weights + HBM KV — a two-store box

**Status: forward-looking architecture note `[EST]`, NOT designed or verified. Contingent on an
emerging memory technology.** Captured because it maps unusually well onto what the RTL already is.

This machine is a **weight-streaming, bandwidth-bound** design: the tok/s limiter is streaming the
~14 GB/token of routed expert weights (`tok/s ≈ BW ÷ 13.87 GB/token`). Two memory technologies split
the two access patterns cleanly:

- **Weights → HBF (High Bandwidth Flash).** HBF (3D-NAND stacked HBM-style, announced 2025; not
  shipping) offers **HBM-class bandwidth at flash capacity + non-volatility**. Because it is
  non-volatile *and* high-bandwidth, one HBF store does **both** jobs the current design splits across
  tiers: the persistent bulk store (today's **NVMe**) and the high-bandwidth stream source (today's
  **DDR/LPDDR working cache**). Consequences:
  - **No NVMe tier** — HBF is the persistent store.
  - **No 467 GB DRAM copy / no ~70 s boot-load** — weights are already resident in non-volatile HBF;
    instant-on. (The residency box's ~70 s DRAM fill disappears.)
  - Flash's higher *read latency* is **hideable for the weight stream** (it is sequential/predictable;
    `flash_xbar`'s deep-queue latency-hiding already does this), and **write endurance is a non-issue**
    (weights are written once at provisioning, then read-only).
- **KV cache → HBM.** The MLA latent KV is **87.8 KB/token** (`(kv_lora 512 + rope 64) × 2 B × 78
  layers`), accumulating with context: **~90 GB at the full 1M context** — so a **96 GB HBM is
  sufficient** (~6 GB headroom — a tight fit for full 1M context; less capacity if context is capped),
  against **467 GB of weights in HBF**.
  KV is small relative to the weight stream but **random-access and latency-sensitive** — the one pattern
  flash latency cannot serve, so it lives in low-latency HBM. Moving KV to HBM is about latency/capacity,
  not raw tok/s; the tok/s win comes from the **weight-stream BW (HBF)**.

> **GLM-5.3-Flash note.** Everything in this rung-④ section is sized against GLM-5.2. For this
> branch's target the weight tier falls to 199.7 GB and the KV tier to 11.8 GB — the ~96 GB HBM below
> is oversized by ~8×, and the *capacity* argument for a dedicated KV stack largely goes with it (the
> *access-pattern* argument survives). The 2-stack HBF default is unchanged, because the second stack
> buys bandwidth, not capacity. See
> [GLM-5.3-Flash re-sizing](#glm-53-flash-re-sizing-what-changes-and-what-does-not) above.

**Two independent stores, both feeding the die directly** — HBF streams weights straight to the compute
die (no staging copy through HBM), and HBM holds only the KV. There is no HBF→HBM path; the asymmetry is
the point. Concretely: a **512 GB HBF** (467 GB weights + ~45 GB / ~10 % headroom — a tight exact fit;
~1 TB for larger / future models) at **~1.6 TB/s**, + a **96 GB HBM** (90 GB max-context KV + ~6 GB) —
a large cheap non-volatile weight store and a modest low-latency KV store, sized to their very different jobs.

**Speed `[EST]`:** the announced **~1.6 TB/s is per-stack**, and HBF is stacked like HBM, so the rung-④
design point assumes **multi-stack by default: 2 stacks (3.2 TB/s aggregate; a single 512 GB stack
already holds the model, so the second stack buys pure bandwidth) → ~200+ tok/s `[EST]`**. A 1-stack
entry configuration lands at ~100--115 `[EST]`. The engineering cost that comes WITH the default: the
binding constraint is no longer memory but the die (consuming 1.54 TB/s already needs ~12.7K MAC lanes
@490 MHz; 3.2 TB/s needs ~26K) and power (the memory rail alone passes 100 W), and measured lane scaling
is **sublinear** (4× lanes → ~2.40×), so the 2-stack point requires roughly doubling the compute silicon.
Higher stack counts are arithmetically possible but push into chiplet/kW territory — a different product
bracket — so they are deliberately not quoted as this design's numbers.

**RTL fit.** `flash_xbar` is a **medium-agnostic** address→weight-bytes crossbar ("the NAND-specific
backend is the swapped part, not the abstraction"), so fronting HBF is a backend swap, and
`kv_cache_pager`/`weight_loader_q4k` already separate KV from the weight stream — the two-store split
maps onto the existing byte-agnostic memory system by **re-parameterization**. What is NOT there:
the DDR-tier removal + re-tiering and the **vendor HBF/HBM PHY + controllers** (external IP). All `[EST]`.

## Honest caveats

- Every tok/s is **[EST]** — roofline projections. The **fit + routed Fmax are now MEASURED** (Vivado
  ML 2026.1 PnR on XCKU3P, 46.5 MHz, campaign closed at 4.6× — the worst path is route-dominated, not
  arithmetic), but there is still **no running board**, so even rung ① is projected until the FPGA demo
  measures a real Fmax ÷ cyc_per_tok on hardware.
- **Bit-exact scope.** All of the above is bit-exact to our ggml reimpl `tools/q4k_ref.py` — whose
  dequant layer is now **proven on real GGUF bytes** ([`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md));
  mixed-type Q6_K/Q8_0/F16 RTL consumers are **DONE** (`make mixedtype`). Still open/honest: the
  llama.cpp **whole-runtime** check is out-of-contract, and the real 467 GB file has not been
  consumed end-to-end ([`../README.md`](../README.md)).
- **Separate the bit-exact band from the knobs.** Rung ① (~0.5–8 tok/s) is bit-exact throughout. Rung ②'s
  ~15–40 needs *either* a DDR-cached-expert hit rate (routing-entropy-limited; prefetch is a measured
  no-op) *or* non-bit-exact pruning — state which one when quoting it.
- Rung ① is **reduced-config** (dev-board DDR/storage can't hold 467 GB); the full model needs the custom
  board (②). The demo proves the *mechanism on real silicon*, not the full box.
- Compute-side levers (e.g. the order-0 `weight_decomp` lossless pack — **wired on `main` behind a
  default-off parameter and release-gated** (`make weight-decomp`, `decomp1-elab`); its **1.34× ratio is
  an FP8-era measurement, not transferable to Q4_K until re-measured** [EST at best]) improve
  area/power/timing but **do not move an NVMe-bound roofline** —
  only more streaming bandwidth (drives / channels / HBM) moves tok/s.
- Chip/board prices are order-of-magnitude; exact FPGA quotes need a distributor, exact board cost needs
  the PCB-house quote. BOM is memory/storage-dominated, not FPGA-dominated. All economics **[EST]/[PENDING]**.
