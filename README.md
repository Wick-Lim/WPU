# WPU — GLM-5.3-Flash · `UD-Q4_K_XL`

> **Model branch `glm5.3-flash/UD-Q4_K_XL`** — the port to
> [`unsloth/GLM-5.3-Flash-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
> Project overview, site and paper are on the hub:
> [**`main`**](https://github.com/Wick-Lim/WPU) · sibling targets:
> [`glm5.2/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.2/UD-Q4_K_XL) (the proven build) ·
> [`laguna-s-2.1/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/laguna-s-2.1/UD-Q4_K_XL).
>
> WPU = **W**eight **P**rocessing **U**nit — a Q4_K local-inference accelerator in Verilog.

> **Everyone else named their chip after the math** — Tensor, Neural, Language *Processing Unit*.
> **This one is named after the bottleneck: the weights.** Frontier LLM inference is not
> compute-bound, it is *weight-bandwidth*-bound — `tok/s ≈ memory bandwidth ÷ weight bytes per
> token`, and for this checkpoint that denominator is **14.118 GB/token**, measured from the GGUF
> tensor map (`tools/glm53_flash_gguf_scan.py`; top-8-of-288 routed experts + every dense weight,
> `token_embd` counted as the one row a token actually reads, no speculative amortization). So the
> die here is sized to *consume a weight stream*, not to maximize FLOPS, and it reads the published
> weight files (GGUF k-quants) **bit-exactly, with no conversion**.

## ⚠️ Port status: config LOCKED, datapath NOT COMPLETE

This branch forked at the `glm5.2/UD-Q4_K_XL` tip and carries every proof that branch
earned. **Those are GLM-5.2 proofs.** GLM-5.3-Flash is *not* a re-dimensioned GLM-5.2 —
its arch id is **`glm5next`**, and **34 of its 45 layers are KDA linear attention, a
machine this repo does not have.** A third of the checkpoint's bytes are **Q5_K**, for
which there is no kernel here either.

| | status |
|---|---|
| model config (all dims, cited from the GGUF) | **LOCKED** — `configs/full_glm53_flash.vh`, gated by `make glm53f-config-guard` |
| MLA + DSA blocks (11 / 45) + MTP | inherited, proven at GLM-5.2 shapes |
| MoE, dequant (Q4_K/Q6_K/Q8_0), memory system | inherited, proven at GLM-5.2 shapes |
| KDA linear attention (34 / 45 blocks) | **no RTL** |
| hyper-connections (every block's residual) | **no RTL** |
| Q5_K dequant (34.9 % of bytes) | **no kernel** |
| clamped SwiGLU, indexer k-pool compressor | **not implemented** |

**No claim is made that this branch runs GLM-5.3-Flash.** The full scope, the measured
tensor census and the reproduction commands are in
[`docs/GLM53_FLASH_PORT.md`](docs/GLM53_FLASH_PORT.md).

> **🙏 Looking for an arXiv endorsement (cs.AR).** The preprint of this work —
> *Bit-Exact by Construction: A Verification-First RTL Accelerator that Inherits the
> GGUF k-Quant Checkpoint Ecosystem* ([source + compiled PDF on the hub](https://github.com/Wick-Lim/WPU/tree/main/paper))
> — needs a first-time-author endorsement for arXiv
> **cs.AR**. If you are qualified to endorse in cs.AR and, after looking at the paper
> and this repository's verification ledger, consider the work credible, you can
> endorse here: **<https://arxiv.org/auth/endorse?x=7L4XXQ>**
> (contact: <wicklim90@gmail.com>). Every proven/measured claim in the paper is
> reproducible from this repository's `make` gates.

**🌐 Project site:** [**Overview**](https://wick-lim.github.io/WPU/) (status ledger + product
concept) · [**Board**](https://wick-lim.github.io/WPU/board.html) (measured FPGA fit + the
rung-③ 512 GB LPDDR5X design point, with the concept floorplan) ·
[**Roadmap**](https://wick-lim.github.io/WPU/roadmap.html) (the 3-rung hardware ladder + the
future HBF/HBM tier) — all figures info-only, every projection tagged `[EST]`.

**The target of this branch** is the published GGUF k-quant of GLM-5.3-Flash,
[`unsloth/GLM-5.3-Flash-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) —
a **320.8B-param** hybrid MoE (**16.7B active/token**, arch `glm5next`) in **199.70 GB**. Those
three figures are measured from the checkpoint's own tensor map, not quoted:
`python3 tools/glm53_flash_gguf_scan.py --fetch /tmp/glm53f && python3 tools/glm53_flash_gguf_scan.py /tmp/glm53f`
reads ~30 MB of GGUF headers and re-derives all of them, cross-checking the byte total against
the published shard sizes to within the 9.52 MB the headers themselves occupy.

**The machine this branch inherits** was built and proven against GLM-5.2. The Q4_K GEMM core is
**bit-exact to an independent ggml-Q4_K reference** (`tools/q4k_ref.py`, itself proven bitwise-equal
to real GGUF bytes at the dequant layer). The full operator datapath is assembled in Q4_K, has an
end-to-end numeric golden against a numpy reference, and elaborates clean at the true 753B GLM-5.2
shape. It is wrapped by a single-module memory system (multi-channel DDR5 + NVMe expert cache +
weight/boot loaders + multi-clock CDC) whose controllers are bounded-model-checked and unbounded-k-
induction-proven. The whole product top is placed & routed on a real FPGA.

**What is not done is stated up front.** For GLM-5.3-Flash specifically: 34 of 45 layers (KDA linear
attention), the hyper-connection residual path, and the Q5_K dequant that covers 34.9 % of the
checkpoint's bytes are **all unimplemented** — see the port-status table above and
[`docs/GLM53_FLASH_PORT.md`](docs/GLM53_FLASH_PORT.md). Carried over from the GLM-5.2 build:
llama.cpp *whole-runtime* numeric equality is out-of-contract by design (attention/accumulation
orders differ), no full checkpoint has been run end-to-end, and every throughput / cost figure is
`[EST]` (roofline-modeled, not measured on silicon) — and for this model even the speculative-decode
acceptance input is borrowed from GLM-5.2 until it is re-measured. See
[*What's proven*](#whats-proven) for the exact status of every claim.

> **The product** is a single-user box that runs with the ethernet unplugged — the full model,
> fully offline / air-gapped, provisioned once (**~199.7 GB** of UD-Q4_K_XL weights for
> GLM-5.3-Flash, against ~467 GB for GLM-5.2) then disconnected. No per-token API
> fees, no vendor that can rate-limit or cut you off. The number that matters is single-user interactive
> throughput; it is set by the hardware rung (memory bandwidth / IO / PHY budget) — see
> [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md).

> **Where things live.** The repo is organised as a hub plus one branch per model target:
> [**`main`**](https://github.com/Wick-Lim/WPU) is the hub (project README, the
> [site](https://wick-lim.github.io/WPU/), the [paper](https://github.com/Wick-Lim/WPU/tree/main/paper));
> **this branch** ports GLM-5.3-Flash (config locked, datapath incomplete — see the port-status
> table above); [`glm5.2/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.2/UD-Q4_K_XL) is the
> proven GLM-5.2 build this one forked from; and
> [`laguna-s-2.1/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/laguna-s-2.1/UD-Q4_K_XL) ports the
> same accelerator to a second model (dequant inherited unchanged, MoE bit-exact in RTL, the GQA
> attention machine reference-verified — orchestrator RTL scoped, not written). Prior work is kept as
> **tags**, never as current: `fp8-verified-baseline` (the earlier FP8 datacenter track) and
> `compression-study-baseline`. The full product (rungs ②③) is the roadmap, not this branch's current
> code ([`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md), [`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)).

---

## What's proven

> **⚠️ Every row below was measured on GLM-5.2 shapes**, and is inherited by this branch as a
> *machine*, not as a claim about GLM-5.3-Flash. They are the reason the port starts from a strong
> base — the dequant core, the MoE, the memory system and the whole verification harness are
> model-independent — but none of them has been re-run at GLM-5.3-Flash's shape, and the blocks
> GLM-5.3-Flash needs that do not exist yet (KDA, hyper-connections, Q5_K) have no row here at all.
> See [`docs/GLM53_FLASH_PORT.md`](docs/GLM53_FLASH_PORT.md) §4.

Each row is tagged for the kind of evidence behind it: **PROVEN** (a gated bit-exact / functional sim),
**FORMAL** (a solver proof over the memory/control plane only), **MEASURED** (real RTL cycles or a real
silicon fit), **ELABORATED** (structural, no functional golden), or **NOT-YET** (a real, open gap). Every
"bit-exact" means bit-exact to the ggml-Q4_K reference `tools/q4k_ref.py`; llama.cpp whole-runtime equality
is a separate, out-of-contract question (last two rows).

Every gate prints `ALL <N> TESTS PASSED` on success and `$fatal`s on the first mismatch. The full gate is
`make release-gate-strict`: **every release gate green with its exact per-gate test count pinned** (a manifest
check that catches a testbench silently running fewer tests than intended). The spec-greedy / intra-batch /
SELF_KV / loopback proofs below are release-gate members, no longer opt-in-only.

| What | Status | Evidence |
|---|---|---|
| **Q4_K GEMM core** (`glm_matmul_q4k`) — block dequant → fp32 MAC → bf16 | **PROVEN — bit-exact vs ggml** | `make q4k` · 160/160 |
| **Assembled full forward** (`glm_model_q4k`: embed → L×(MLA+DSA+MoE) → norm → LM-head → argmax) | **PROVEN — bit-exact vs numpy ref** (logits + argmax + h_state) | `make model-q4k` · 1155/1155 |
| **Mixed-type path** (Q6_K / Q8_0 / F16 tensors of the dynamic UD-Q4_K_XL mix) | **PROVEN — bit-exact** — all four types incl. a 24-tile mixed sequence | `make mixedtype` |
| **Dequant vs real GGUF bytes** | **PROVEN** — `q4k_ref.py` vs llama.cpp's own `dequantize_row_*` on two real published GGUFs: **376,586,240 weights (Q4_K + Q6_K + Q8_0) all bitwise-equal** → by transitivity the RTL dequant ≡ the real files' | `tools/gguf_crosscheck.py` |
| **Speculative-decode composition** (`glm_q4k_spec_system`: memory system + `PE_M=K+1` batched verify + accept/reject loop in one top) | **PROVEN — spec==greedy**: committed stream === a `PE_M=1` greedy decode, K=1,2,3 × {ACCEPT,REJECT,MIXED}. **A_eff MEASURED** from a hardware `weight_loads` counter (ALL-ACCEPT hits the K+1/load ceiling) | `make spec-greedy` · 31/31 |
| **Intra-batch causal MLA** (`INTRA_CAUSAL`) — batched verify == serial single-row chain | **PROVEN — full-logit bit-exact** | `make intra-batch-verify` · 9/9 |
| **Die-internal KV write-back** (`SELF_KV`) — the die attends its own written per-(layer,pos) KV | **PROVEN — bit-exact** vs an independent (layer,pos) reference; byte-identical when off | `make self-kv-roundtrip` / `self-kv-equiv` |
| **PHY-closure loopback** — the die's weight bytes routed OUT as a banked `ddr5_xbar` read and back IN through the fabric, committed stream bit-exact | **PROVEN — all five weight-input families** (aw, fw, rw, lw, gn); output-insensitive rw/gn add a direct per-beat die-input byte binding; each with a corruption-injection build that FAILS | `make loopback` / `loopback-fw` / `loopback-rest` |
| **L3 board-boot chain** (`fpga/l3_top.v`) — SPI-NOR boot image → em/fn LUTRAM + 3 dequant-header BRAM stores + DDR weight seg, then 4 greedy tokens decoded over the real UART framing against a shadow-fed reference | **PROVEN — bit-exact end to end** (the module a board instantiates, at a tiny config); corrupting one boot-image byte or one DDR beat each diverges the tokens | `make l3-e2e` · 11/11 + 2 must-fail injections · `make l3-hash-mirror` 704/704 |
| **Batched MLA / batched assembled model** (`PE_M>1`) | **PROVEN — DUT-vs-DUT bit-exact** | `make mla-sparse` / `batched-q4k` |
| **Memory-system controllers** — routing/FIFO/token-accounting/ECC/done-gates | **FORMAL — BMC** (7 controllers + 1 ECC-ring), + **unbounded k-induction** on 5 | `make formal` / `formal-ind` |
| **Whole 2-clock product top** (`glm_q4k_system_cdc`) | **ELABORATED** — yosys `hierarchy -check` + `check -assert` exit 0 (no unresolved hierarchy / comb loop / multiple driver / inferred latch); structural sign-off, not a sim | `make synth-glm` |
| **Full 753B UD-Q4_K_XL-shape** (`glm_model_q4k` at DIM 6144 / L=78 / 256-expert / VOCAB 154880) | **ELABORATED** — type/width check only, no stimulus | `test/full_config_elab_wrap.v` |
| **FPGA fit** — real Vivado synth + place & route on Kintex UltraScale+ **XCKU3P** | **MEASURED** — compact config + `ACT_HW=1`: **141,298 LUT routed** (142,320 / 87.5% at the synth stage), ~100K FF, **421 DSP, 0 BRAM**, hold met, routed Fmax **46.5 MHz** (bit-exact repipeline campaign closed at 4.6×) | `bash fpga/run_fit.sh` · `fpga/results/util_routed_ku3p_acthw1.rpt` |
| **Cycle-accurate stall harness** | **MEASURED** (tokens held bit-exact): the residency pivot on real cycles — RESIDENT=1 exposes 35 stall cyc/token vs 2,567 at RESIDENT=0/FLASH_LAT=1024 | `make perf-q4k` |
| **DFT / power** | 2-port BIST **reference** (`mbist_ctrl_2p`, dual-port March + concurrent-coupling, 11/11); inline glitch-free `die_clk` ICG in the top; SECDED weight/KV ECC (`make weight-ecc`) | see `docs/P2_MEMORY_MAP.md`, `docs/LOW_POWER.md` |
| **llama.cpp full-runtime numeric equality** | **NOT-YET / out-of-contract** — attention/accumulation orders differ by design; the 467 GB file has not been run end-to-end | — |
| **Throughput / energy / BOM / TCO / LOI** | **NOT-YET `[EST]`** — roofline-modeled (with measured model-side inputs: A_eff hardware-measured, accept rate r measured on GLM-4.5-Air); no silicon | — |

---

## How it works

```
  1 TB NVMe    ──►  flash_xbar    ──►  DDR working ──►  ddr5_xbar  ──►   Q4_K compute die   ──►  token
 (~467 GB Q4_K   N-channel banked    set / cache       N-channel     (MLA + DSA + MoE, ggml
  GGUF weights)  + deep queue        (LRU+freq+pf)     banked read    Q4_K dequant → fp32 MAC → bf16 tail)
```

The workload is **NVMe/PCIe-bandwidth-bound** by design (to keep it cheap): MoE experts stream from the SSD
through a DDR working cache into a mostly-idle Q4_K die (tier: NVMe bulk/slow → DDR hot-set/fast → die).
Every Q4_K weight matmul dequantizes per ggml (256-weight super-block: 4-bit quants scaled by a 6-bit
per-sub-block scale/min, fp16 `d`/`dmin`) → fp32 MAC → bf16; norms, softmax, rope, residual and the
activation×activation attention matmuls stay bf16. The whole memory system (`expert_cache_pf`,
`kv_cache_pager`, `ddr5_xbar`, FIFO/arbiter/CDC) is **byte-agnostic** — it moves addresses/slots/IDs, never
weight bytes — so it carried over from the FP8 track by parameter, not logic.

**Verification methodology.** Every unit is checked against an independent golden (the generic bf16/fp32
twins against fp64/fp32 goldens, the Q4_K core against the ggml reference). Regression is **byte-identical**;
the memory controllers add bounded model checking, some lifted to unbounded k-induction. No formal proof
touches the numeric datapath — that plane is held by the bit-exact sim goldens. **Every "it passes" claim is
paired with an injection that a real bug FAILS** (a corrupted latent, a committed raw draft, a wrong-index
byte) — a test that cannot tell a correct result from a broken one proves nothing.

---

## The target: `unsloth/GLM-5.3-Flash-GGUF : UD-Q4_K_XL`

A dynamic k-quant mix: the routed experts carry the model at 4–6 bits, everything else is kept at Q8_0.
Each type dequantizes exactly per ggml, then the same GEMM contract runs (dequant → fp32 MAC → bf16).
The mix below is **measured from all 1412 tensors** of the published build, not assumed — and it is how
the Q5_K gap was found:

| Type | Tensors | Bytes | Share | Where | Dequant | RTL consumer |
|---|---|---|---|---|---|---|
| **Q4_K** | 84 | 114.15 GB | 57.2 % | `ffn_{gate,up}_exps` | `w = (d·sc)·q − (dmin·m)` | ✅ `q4k.vh` + `glm_matmul_q4k.v` (160/160) |
| **Q5_K** | 42 | 69.76 GB | **34.9 %** | `ffn_down_exps` | — | ❌ **no kernel in this repo** |
| **Q8_0** | 645 | 9.62 GB | 4.8 % | attention, shared expert, `output` | `w = d·q` | ✅ `q4k_mixed.vh` + `w_type` arm |
| **Q6_K** | 3 | 5.95 GB | 3.0 % | UD bump on `blk.{11,12,44}.ffn_down_exps` | `w = (d·sc)·(q−32)` | ✅ `q4k_mixed.vh` + `w_type` arm |
| **F32** | 638 | 0.23 GB | 0.1 % | norms, routing gates, `ssm_a`/`dt`/`conv1d` | — | n/a |
| **total** | **1412** | **199.70 GB** | | | | |

That byte total cross-checks against the published shard sizes (199.71 GB) to within the 9.52 MB of GGUF
headers — the agreement is what makes the parse trustworthy rather than merely plausible.

The **UD "Dynamic" bumps are per tensor, not per family**: `blk.11`'s gate/up experts are promoted
Q4_K→Q5_K and `blk.{11,12,44}`'s down experts Q5_K→Q6_K. A loader that assumes one type per tensor family
will mis-read those blocks; the type must be read from the GGUF per tensor.

**Architecture** (all values cited in `configs/full_glm53_flash.vh`, gated by `make glm53f-config-guard`):
arch `glm5next`, hidden 4096, **45 layers + 1 MTP block**, split **34 KDA linear-attention + 11 MLA+DSA**
(full attention at blocks 3, 7, …, 43), `first_k_dense_replace=3`, 64 heads, MLA (`qk_nope 256`,
**`qk_rope 0` — NoPE, no rotary anywhere**, `v 256`, `kv_lora 512` *confirmed in the GGUF*, `q_lora 1536`),
DSA indexer (32 heads × 128, `index_topk 2048`, k-pool 4 with compression), MoE **288 experts** top-8 +
1 shared, `moe_intermediate 2048`, dense `intermediate 12288`, **clamped SwiGLU (limit 10.0)**,
**hyper-connection residual (mult 4, Sinkhorn 20 iters)**, vocab 154880, 1M context, RMSNorm `eps 1e-5`,
MTP (`nextn_predict_layers 1`). **320.759 B params total, 16.742 B active/token, 14.118 GB read/token.**

Re-derive every figure in this section from the checkpoint itself (~30 MB of headers, not 199.7 GB):

```sh
python3 tools/glm53_flash_gguf_scan.py --fetch /tmp/glm53f
python3 tools/glm53_flash_gguf_scan.py /tmp/glm53f
```

---

## Die-side cycle work (2026-07) — measured, and honestly bounded

Three parameter-gated, default-off levers cut the measured decode window by **19.1%**
(43,724 → 35,364 cycles at the profiled slice): concurrent SwiGLU gate‖up (−3,680), 1-issue/cycle
softmax (−864), and a (layer,key) K/V projection cache (−3,816, later shelved for a correctness
gap its own oracle caught — the write-up in `docs/ULTRA_PERF_MODULES.md` keeps both the measured
ceiling and the five defects). Each realised its predicted ceiling exactly; one earlier lever
(multi-port issue) measured exactly zero, and both outcomes are recorded.

**None of this changes the tok/s table below.** The design is bandwidth-bound and the die is
~20–25% utilized — cycles are not the binding constraint, so cycle savings buy Fmax/batch headroom,
not tokens. Net tok/s = cycle win × (Fmax_new / Fmax_base), and no re-fit has measured the right
factor yet; every lever therefore ships default-off.

## Throughput — `[EST]`, an optimistic ceiling

> **⚠️ Every number in this section is GLM-5.2's and has NOT been re-measured for GLM-5.3-Flash.**
> The raw denominator for this model *is* known — **14.118 GB/token**, derived from its own tensor
> map — but the amortized figure below divides that kind of number by `A_eff`, and `A_eff` is a
> property of a specific model's draft quality. GLM-5.3-Flash does have an MTP head
> (`nextn_predict_layers = 1`), so the mechanism carries over, but its accept rate is unmeasured.
> Until it is, no amortized tok/s figure for GLM-5.3-Flash is published here.
> See [`docs/GLM53_FLASH_PORT.md`](docs/GLM53_FLASH_PORT.md) §4.3.

The design is **bandwidth-bound**, so `tok/s ≈ memory BW ÷ 13.87 GB/token`, where 13.87 GB is `A_eff=1.87`
(the amortization mechanism now **hardware-measured**) at the **measured** accept rate r₁=0.87 (GLM-4.5-Air).
The denominator is well-grounded; the numerator is the external hardware's bandwidth.

| Rung / config | Memory BW | tok/s `[EST]` |
|---|---|---|
| ① Prove-it FPGA (KU3P + DDR4 hot-set) | 1–2 NVMe … striped | **~0.5–1 … ~5–8** · bit-exact |
| ② Custom board (mid FPGA, DDR5/HBM) | ~400 GB/s–1 TB/s | **~15–40** · contingent |
| ③ SoC — 512 GB LPDDR5X full residency (primary) | ~1.1 TB/s (up to 1.54 TB/s … HBM) | **≈80** `[measured-inputs EST]` (~95 if GLM-5.2 MTP is deeper; ~111–120 aspirational HBM ceiling) |
| ④ **future** — HBF weights + HBM KV (two-store box) | HBF ~1.6 TB/s **per stack** (2-stack base) + 96 GB HBM KV | **~200+** `[EST]` (stack expansion required; die/power become the binding constraint) |

**Rung ④ (future, memory-tech-dependent).** Once HBF (High Bandwidth Flash — 3D-NAND stacked HBM-style,
announced 2025) matures, the two jobs the current design splits — persistent store (NVMe) and weight-stream bandwidth (LPDDR5X) —
> **Sizing note.** The ladder below is dimensioned for GLM-5.2's ~467 GB checkpoint plus ~94 GB of KV.
> GLM-5.3-Flash needs **199.7 GB + 11.8 GB = 212 GB** at the full 1M context — 2.6× less — because only
> 11 of its 45 layers cache at all and NoPE strips the rotary tail off the cached latent. The
> **capacity** side is re-derived in [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)
> §"GLM-5.3-Flash re-sizing" (`python3 tools/glm53_flash_memory_budget.py`), including the trap that
> the package/stack count must *not* fall with the capacity, since bandwidth follows units, and the
> all-HBM residency that 212 GB newly makes reachable. The **speed** side stays GLM-5.2's: every
> rung's tok/s divides by an `A_eff` this model has not been measured for.

collapse into **one non-volatile, high-bandwidth store**: ~512 GB HBF holds the 467 GB weights *resident and
non-volatile* (no NVMe tier, no ~467 GB DRAM copy, no ~70 s boot-load → instant-on), while a separate ~96 GB HBM
holds only the KV cache. The announced ~1.6 TB/s is **per stack** and HBF stacks like HBM, so the 2-stack base
(~3.2 TB/s against weights) is the natural entry point → **~200+ tok/s `[EST]`**; a 1-stack config is the ~100–115
entry. Because lane scaling is sublinear (4× → ~2.40×), the binding constraint shifts from bandwidth to the **die
(≈26K lanes) and power** — higher stack counts push into a chiplet / kW bracket that is deliberately not quoted.
The RTL abstraction already supports the swap (`flash_xbar` is a medium-agnostic storage-read fabric — "the NAND
backend is the swapped part, not the abstraction"); what rung ④ adds is the DDR-tier removal + re-tiering and the
**vendor HBF/HBM PHY + controllers** (external IP). See [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md) § Rung ④.

**Read these as an optimistic ceiling, not a precise number.** The BW is *peak*, not sustained (real DRAM
~70–85% of peak); lane scaling is **sublinear** (measured 4× lanes → ~2.40×, because attention and MoE run
in sequential phases), so the die needs overprovisioned lanes to consume high BW and can otherwise become
the bottleneck; and nothing is measured on silicon. The real number more likely lands *below* the estimate
than above. What is validated on real RTL cycles is the memory-stall *mechanism* (`make perf-q4k`), not the
absolute tok/s. **Rung ④ is a further step out** — it depends on a memory technology (HBF) that is announced
but not yet shipping, so its `~200+` is the softest `[EST]` in the table. See
[`docs/R3_APPLIANCE_SPEC.md`](docs/R3_APPLIANCE_SPEC.md),
[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md), [`docs/H_MEASUREMENT.md`](docs/H_MEASUREMENT.md).

---

## What's NOT done (open, honest)

**GLM-5.3-Flash port gaps — the ones specific to this branch, in dependency order:**

- **KDA linear attention — 34 of 45 layers have no RTL.** Short causal conv (k=4) on q/k/v, gated
  delta-rule state update, `f`/`g` low-rank gates, decay from `ssm_a` + `ssm_dt`, per-head norm. This
  is a new datapath with new state memory and a new golden reference; it is the item that sets the
  schedule.
- **Q5_K dequant — no kernel anywhere in this repo**, and Q5_K is 34.9 % of the checkpoint's bytes
  (all 42 `ffn_down_exps`). Without it the model cannot be read at all. Smallest of the three gaps:
  it is a sibling of the proven Q4_K/Q6_K path and the existing `gguf_crosscheck` harness extends to it.
- **Hyper-connections — no RTL.** Sinkhorn normalization (20 iterations) over a width-4 connection
  matrix replaces the plain residual add on every block; needs a fixed-point study first.
- **Clamped SwiGLU (limit 10.0) and the indexer k-pool compressor** — not implemented; both small.
- **Speculative-decode inputs not re-measured.** The MTP head exists here
  (`nextn_predict_layers = 1`), but GLM-5.2's `A_eff = 1.87` / accept rate 0.87 are properties of a
  different model and do not transfer, so no amortized tok/s figure is published for GLM-5.3-Flash.
- **Packer / flash layout not re-targeted** to `glm5next` tensor names and the per-tensor UD mix; and
  `gguf_crosscheck` has not been re-sealed on GLM-5.3-Flash bytes.

Full scope and reproduction commands: [`docs/GLM53_FLASH_PORT.md`](docs/GLM53_FLASH_PORT.md).

**Carried over from the GLM-5.2 build (unchanged by this branch):**

- **llama.cpp whole-runtime numeric equality** — out-of-contract by design; the 467 GB checkpoint has not
  been consumed end-to-end (needs a GPU / large-memory host).
- **Board bring-up** — the FPGA P&R is done in-repo, and the whole board-side digital chain now exists and
  is gated in simulation (SPI flash reader → boot loader → em/fn + header stores → AXI shim → UART host;
  `make l3-e2e` decodes real tokens through it). What is still missing is physical: the board itself, the
  vendor **MIG IP** + pin **XDC**, and the re-fit that measures whether the added stores close on the part
  (the measured fit is 87.5% LUT with BRAM at 0%, so there is room — but room is not a measurement).
- **Vendor PHY hard-IP** (DDR5 / NVMe / USB-C) — TB-stubbed; the digital PHY-closure loopback is proven, the
  analog hard-IP is licensed IP.
- **ASIC scan insertion + compiled SRAM macros + their BIST collars** — tool/vendor steps on the RTL (the
  RTL is scan-ready and carries verified BIST *references*), not hand-RTL.
- **Full-scale functional sim** — infeasible (LM-head GEMV ~2.4e8 K-beat/token); the model is verified at a
  small-but-faithful slice and elaboration-clean at the real 753B GLM-5.2 shape (a GLM-5.3-Flash
  full-shape elaboration waits on the KDA block — `make glm53f-config-guard` keeps any whole-model
  top from elaborating until it exists). HBM-scale lane consumption
  (~12,732 lanes) is parameterized + sublinear-measured but functionally verified only at modest lane counts.
- **Economics** (BOM / TCO / LOI) — planning-doc `[EST]`, not evidence; no signed LOI exists.

---

## Build / test

```sh
brew install icarus-verilog verilator yosys     # iverilog 13.0, verilator 5.048, yosys 0.66
python3 -m pip install numpy                    # required by the golden-reference generators (make all / q4k / model-q4k)

make glm53f-config-guard # GLM-5.3-Flash config gate: dims usable, whole-model top poisoned (8/8, both tools)
make all                 # the rung-① FPGA prove-it gate: unittests + synth-glm + formal + more
make release-gate-strict # the full release gate + exact per-gate test-count manifest check
make q4k                 # the Q4_K sub-gate (q4k_prim / glm_matmul_q4k / swiglu_expert_q4k / moe_router_q4k)
make model-q4k           # assembled full-forward numeric golden (1155 tests)
make mixedtype           # Q6_K / Q8_0 / F16 mixed-type path
make spec-greedy         # composed speculating top: spec==greedy + A_eff measured (in release-gate)
make loopback            # PHY-closure loopback (aw); loopback-fw / loopback-rest for the other families
make l3-e2e              # the L3 board top end to end: SPI boot image -> tokens over UART (in release-gate)
make formal / formal-ind # BMC / unbounded k-induction of the memory controllers
make synth-glm           # yosys whole-chip structural gate on glm_q4k_system_cdc
make host-test           # host OpenAI-server + device-protocol + tokenizer scaffold (32 tests)
```

Re-derive the GLM-5.3-Flash target figures straight from the published checkpoint (reads ~30 MB of
GGUF headers, not the 199.7 GB of weights):

```sh
python3 tools/glm53_flash_gguf_scan.py --fetch /tmp/glm53f
python3 tools/glm53_flash_gguf_scan.py /tmp/glm53f
python3 tools/glm53_flash_memory_budget.py        # 212 GB resident @1M + what it does to the ladder
```

The one true bit-exact datapath result, standalone (zsh does not word-split — list sources explicitly):

```sh
mkdir -p build
python3 tools/q4k_matmul_gen.py >/dev/null
iverilog -g2012 -Wall -I src -o build/glm_matmul_q4k_sim test/glm_matmul_q4k_tb.v src/glm_matmul_q4k.v
vvp build/glm_matmul_q4k_sim     # -> ALL 160 TESTS PASSED (bit-exact vs ggml Q4_K)
```

**Slice.** The RTL is built at a small-but-faithful slice keeping every operator and ratio (MODEL_DIM=128,
6 layers [3 dense + 3 MoE], 4 heads, 8-expert top-2 + 1 shared, VOCAB=256, S_MAX=8). Running the real 753B
model adds the memory/streaming system + array scaling.

---

## Documents

- **[Project site](https://wick-lim.github.io/WPU/)** — the one-page status
  [Overview](https://wick-lim.github.io/WPU/), the
  [Board](https://wick-lim.github.io/WPU/board.html) design point, and the
  [Roadmap](https://wick-lim.github.io/WPU/roadmap.html) ladder.
- **[`docs/GLM53_FLASH_PORT.md`](docs/GLM53_FLASH_PORT.md)** — **this branch's port ledger**: what
  GLM-5.3-Flash actually is, the measured tensor census and quant mix, what is inherited from the
  GLM-5.2 build, and the ordered list of what is missing. Start here for "what does this branch
  actually run."
- **[`docs/Q4K_RETARGET.md`](docs/Q4K_RETARGET.md)** — the Q4_K dequant math, GEMM contract, per-type status.
  Start here for "what is Q4_K-exact and what isn't."
- **[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)** — the hardware plan: rungs ①–③ (prove-it FPGA →
  custom board → 512 GB LPDDR5X SoC) plus the future rung ④ (HBF weights + HBM KV, ~200+ tok/s `[EST]`). Start
  here for "how fast, on what hardware."
- **[`docs/R3_APPLIANCE_SPEC.md`](docs/R3_APPLIANCE_SPEC.md)** — the rung-③ 512 GB LPDDR5X residency-box
  design point (≈80 tok/s `[measured-inputs EST]`), power / BOM / lane derivation.
- **[`docs/SPEC_COMPOSITION_DESIGN.md`](docs/SPEC_COMPOSITION_DESIGN.md)** / **[`docs/KV_WRITEBACK_DESIGN.md`](docs/KV_WRITEBACK_DESIGN.md)** — the spec-chain × memory-system composition (the tok/s critical path) and the die-internal KV write-back.
- **[`docs/OPERATION_FLOW.md`](docs/OPERATION_FLOW.md)** / **[`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)** — end-to-end operational flow and accelerator architecture.
- **[`docs/P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md)** / **[`docs/LOW_POWER.md`](docs/LOW_POWER.md)** / **[`docs/FORMAL.md`](docs/FORMAL.md)** — DFT/ECC memory map, power levers, formal scope.
- **[`fpga/`](fpga/README.md)** — the measured Vivado fit on XCKU3P. **[`host/`](host/README.md)** — the host
  software scaffold (OpenAI-compatible server, real GLM tokenizer, RTL-sim backend).
- **[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)** / **[`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)** — product direction and the honest open items.

---

## Appendix — prior FP8 track (tag `fp8-verified-baseline`)

Before the Q4_K retarget, `main` developed a datacenter-native **FP8 E4M3** accelerator targeting
[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8). It is preserved at tag **`fp8-verified-baseline`**
(every `*_fp8.v`, TB, and evidence doc), referenced here as prior/preserved
— none of it is on `main`. At that tag: operator bit-accuracy 9216/9216 vs the real FP8 safetensors,
exhaustive FP8 E4M3 arithmetic (66069 cases), real sky130 place-and-route of `glm_matmul_fp8`, and
compute-side PPA wins — all FP8-specific. To inspect: `git checkout fp8-verified-baseline`. The memory-system controllers,
CDC, ECC/BIST, and clock-gating blocks are shared byte-agnostic logic across both tracks.

---

## License

**Apache-2.0** — the repo-level [`LICENSE`](LICENSE) governs every file in this repository. Per-file
SPDX / Apache headers are deliberately omitted by policy; if you copy a file out individually, it carries
the repository's Apache-2.0 terms.
