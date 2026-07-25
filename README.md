# WPU — **W**eight **P**rocessing **U**nit

> A GLM-5.2 Q4_K local-inference accelerator in Verilog.
>
> **Everyone else named their chip after the math** — Tensor, Neural, Language *Processing Unit*.
> **This one is named after the bottleneck: the weights.** Frontier LLM inference is not
> compute-bound, it is *weight-bandwidth*-bound — `tok/s ≈ memory bandwidth ÷ 13.87 GB of weights
> per token` — so the die here is sized to *consume a weight stream*, not to maximize FLOPS, and it
> reads the published weight files (GGUF k-quants) **bit-exactly, with no conversion**.

> **🙏 Looking for an arXiv endorsement (cs.AR).** The preprint of this work —
> *Bit-Exact by Construction: A Verification-First RTL Accelerator that Inherits the
> GGUF k-Quant Checkpoint Ecosystem* ([`paper/wpu.tex`](paper/wpu.tex),
> [compiled PDF](paper/wpu.pdf)) — needs a first-time-author endorsement for arXiv
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

A synthesizable Verilog accelerator that runs one real model on a local, offline box: the
published GGUF k-quant of GLM-5.2,
[`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF) — a 753B-param
MoE (~40B active/token, `GlmMoeDsaForCausalLM`) in ~4-bit **Q4_K**, ~467 GB.

The Q4_K GEMM core is **bit-exact to an independent ggml-Q4_K reference** (`tools/q4k_ref.py`, itself
proven bitwise-equal to real GGUF bytes at the dequant layer). The full operator datapath is assembled
in Q4_K, has an end-to-end numeric golden against a numpy reference, and elaborates clean at the true
753B shape. It is wrapped by a single-module memory system (multi-channel DDR5 + NVMe expert cache +
weight/boot loaders + multi-clock CDC) whose controllers are bounded-model-checked and unbounded-k-
induction-proven. The whole product top is placed & routed on a real FPGA.

**What is not done is stated up front:** llama.cpp *whole-runtime* numeric equality is out-of-contract by
design (attention/accumulation orders differ), the 467 GB checkpoint has not been run end-to-end, and every
throughput / cost figure is `[EST]` (roofline-modeled, not measured on silicon). See
[*What's proven*](#whats-proven) for the exact status of every claim.

> **The product** is a single-user box that runs with the ethernet unplugged — the full 753B model,
> fully offline / air-gapped, provisioned once (~467 GB Q4_K weights) then disconnected. No per-token API
> fees, no vendor that can rate-limit or cut you off. The number that matters is single-user interactive
> throughput; it is set by the hardware rung (memory bandwidth / IO / PHY budget) — see
> [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md).

> **Branches.** `main` develops the GLM-5.2 Q4_K accelerator (this README). **`laguna-s-2.1`** ports the
> same accelerator to a **second model** — [Laguna-S-2.1](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)
> (118B MoE, `LagunaForCausalLM`): the Q4_K dequant contract is inherited unchanged, the MoE path is
> bit-exact in RTL at Laguna's config, and the (different) GQA attention machine is specified +
> reference-verified end to end (`make laguna`) — the bit-exact orchestrator RTL is scoped, not yet
> written. See that branch's [`docs/LAGUNA_S21.md`](https://github.com/Wick-Lim/WPU/blob/laguna-s-2.1/docs/LAGUNA_S21.md).
> The prior **FP8 datacenter track** is preserved on tag `fp8-verified-baseline`; a
> compression-research study on tag `compression-study-baseline`. All referenced as prior/preserved, never
> current. The full product (rungs ②③) is the roadmap, not main's current code
> ([`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md), [`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)).

---

## What's proven

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

## The target: `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`

A dynamic k-quant mix: most tensors Q4_K; sensitive ones kept at higher precision. Each type dequantizes
exactly per ggml, then the same GEMM contract runs (dequant → fp32 MAC → bf16). All four types have RTL
consumers, bit-exact to the same reimpl golden.

| Type | Dequant | Golden | RTL consumer |
|---|---|---|---|
| **Q4_K** | `w = (d·sc)·q − (dmin·m)` | ✅ bit-exact | `q4k.vh` + `glm_matmul_q4k.v` (160/160) |
| **Q6_K** | `w = (d·sc)·(q−32)` | ✅ bit-exact | `q4k_mixed.vh` + `w_type` arm |
| **Q8_0** | `w = d·q` | ✅ bit-exact | `q4k_mixed.vh` + `w_type` arm |
| **F16** | `w = fp16→fp32` | ✅ (exact) | `w_type` passthrough |

Architecture (model dims, independent of quant): hidden 6144, 78 layers (`first_k_dense_replace=3`), 64
heads (`head_dim=192`), MLA (`qk_nope 192 + qk_rope 64`, `v 256`, `kv_lora 512`, `q_lora 2048`), MoE 256
experts top-8 + 1 shared, dense `intermediate 12288`, DSA sparse attention (`index_topk 2048`), vocab
154880, 1M context, `rope_theta 8e6`, RMSNorm `eps 1e-5`, MTP (`num_nextn_predict_layers 1`). `q_lora 2048`
confirmed vs the real safetensors; `kv_lora 512` is `[PENDING safetensors]` (DeepSeek-standard).

---

## Throughput — `[EST]`, an optimistic ceiling

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

- **llama.cpp whole-runtime numeric equality** — out-of-contract by design; the 467 GB checkpoint has not
  been consumed end-to-end (needs a GPU / large-memory host).
- **Board bring-up** — the FPGA P&R is done in-repo; running on a physical dev board needs the board + pin
  XDC + a MIG bridge.
- **Vendor PHY hard-IP** (DDR5 / NVMe / USB-C) — TB-stubbed; the digital PHY-closure loopback is proven, the
  analog hard-IP is licensed IP.
- **ASIC scan insertion + compiled SRAM macros + their BIST collars** — tool/vendor steps on the RTL (the
  RTL is scan-ready and carries verified BIST *references*), not hand-RTL.
- **Full-scale functional sim** — infeasible (LM-head GEMV ~2.4e8 K-beat/token); the model is verified at a
  small-but-faithful slice and elaboration-clean at the real 753B shape. HBM-scale lane consumption
  (~12,732 lanes) is parameterized + sublinear-measured but functionally verified only at modest lane counts.
- **Economics** (BOM / TCO / LOI) — planning-doc `[EST]`, not evidence; no signed LOI exists.

---

## Build / test

```sh
brew install icarus-verilog verilator yosys     # iverilog 13.0, verilator 5.048, yosys 0.66
python3 -m pip install numpy                    # required by the golden-reference generators (make all / q4k / model-q4k)

make all                 # the rung-① FPGA prove-it gate: unittests + synth-glm + formal + more
make release-gate-strict # the full release gate + exact per-gate test-count manifest check
make q4k                 # the Q4_K sub-gate (q4k_prim / glm_matmul_q4k / swiglu_expert_q4k / moe_router_q4k)
make model-q4k           # assembled full-forward numeric golden (1155 tests)
make mixedtype           # Q6_K / Q8_0 / F16 mixed-type path
make spec-greedy         # composed speculating top: spec==greedy + A_eff measured (in release-gate)
make loopback            # PHY-closure loopback (aw); loopback-fw / loopback-rest for the other families
make formal / formal-ind # BMC / unbounded k-induction of the memory controllers
make synth-glm           # yosys whole-chip structural gate on glm_q4k_system_cdc
make host-test           # host OpenAI-server + device-protocol + tokenizer scaffold (32 tests)
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
