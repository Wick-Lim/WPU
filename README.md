# WPU — **W**eight **P**rocessing **U**nit

**A Verilog inference accelerator that runs published open-weight LLMs on a local, offline box —
bit-exactly, with no conversion.**

> **Everyone else named their chip after the math** — Tensor, Neural, Language *Processing Unit*.
> **This one is named after the bottleneck: the weights.** Frontier LLM inference is not
> compute-bound, it is *weight-bandwidth*-bound — `tok/s ≈ memory bandwidth ÷ GB of weights read
> per token` — so the die is sized to *consume a weight stream* rather than to maximize FLOPS, and
> it reads the published weight files (GGUF k-quants) **bit-exactly, with no conversion step**.

**🌐 Project site:** [**Overview**](https://wick-lim.github.io/WPU/) (status ledger + product
concept) · [**Board**](https://wick-lim.github.io/WPU/board.html) (measured FPGA fit + the
512 GB LPDDR5X design point) ·
[**Roadmap**](https://wick-lim.github.io/WPU/roadmap.html) (the hardware ladder + the future
HBF/HBM tier) — all figures info-only, every projection tagged `[EST]`.

> **🙏 Looking for an arXiv endorsement (cs.AR).** The preprint of this work —
> *Bit-Exact by Construction: A Verification-First RTL Accelerator that Inherits the
> GGUF k-Quant Checkpoint Ecosystem* ([`paper/wpu.tex`](paper/wpu.tex),
> [compiled PDF](paper/wpu.pdf)) — needs a first-time-author endorsement for arXiv **cs.AR**.
> If you are qualified to endorse in cs.AR and, after looking at the paper and the verification
> ledger, consider the work credible, you can endorse here:
> **<https://arxiv.org/auth/endorse?x=7L4XXQ>** (contact: <wicklim90@gmail.com>).
> Every proven/measured claim is reproducible from the `make` gates on the model branches below.

---

## Model targets

**This branch (`main`) is the hub** — the project overview, the site, and the paper. **The RTL and
its verification gates live on one branch per model target**, because each model is a different
compute graph even though the memory system, the Q4_K datapath and the whole verification harness
are shared.

| Model | Checkpoint | Branch | Status |
|---|---|---|---|
| **GLM-5.3-Flash** | [`unsloth/GLM-5.3-Flash-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)<br>320.8B hybrid MoE (16.7B active/token), 199.70 GB<br>212 GB resident at 1M context | [`glm5.3-flash/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.3-flash/UD-Q4_K_XL) | **Current target. Config locked; a complete decoder layer exists for all 45 blocks.** Arch `glm5next` — not a re-dimensioned GLM-5.2. Every machine it needed is built and gated; what is left is model-level assembly. [**Details below.**](#glm-53-flash-where-the-port-stands) |
| **GLM-5.2** | [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF)<br>753B MoE (~40B active/token), ~467 GB | [`glm5.2/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.2/UD-Q4_K_XL) | **The proven build.** Full datapath bit-exact vs an independent ggml reference, memory-system controllers formally verified, whole product top placed & routed on a real FPGA. The paper is about this build, and the GLM-5.3-Flash branch forked from its tip. |
| **Laguna-S-2.1** | [`unsloth/Laguna-S-2.1-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)<br>118B MoE (~8B active/token) | [`laguna-s-2.1/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/laguna-s-2.1/UD-Q4_K_XL) | **Port in progress.** Dequant inherited unchanged; MoE path bit-exact in RTL at Laguna's config; the (different) GQA attention machine is specified and reference-verified end to end — the bit-exact orchestrator RTL is scoped, not yet written. |

**Branch naming:** `<model>/<quantization>`, e.g. `glm5.2/UD-Q4_K_XL`. A second quantization of the
same model is a sibling branch under the same model prefix.

---

## GLM-5.3-Flash: where the port stands

`glm5next` is not a re-dimensioned GLM-5.2. **34 of its 45 layers are KDA linear attention**, every
block carries **hyper-connections** with a Sinkhorn-projected residual mix, and its k-quant mix
includes a format this repo had never built. Every machine it needed now exists and is gated, and
one decoder block now walks all 45 layers on the checkpoint's own schedule. What is left is the
model *around* that stack.

### Landed

Each row is a `make` target with its own must-fail injections, and the release gate pins its exact
test count.

| Machine | Gate | The claim |
|---|---|---|
| **Q5_K dequant** | `mixedtype`, `q5k-loader` | 34.9 % of the checkpoint's bytes, bit-exact on real published bytes — reference, GEMM arm, and the loader that feeds it |
| **Clamped SwiGLU** | `q4k` | the clamp is **asymmetric** (gate upper-only, `up` both ways); a symmetric guess is numerically wrong, not approximate |
| **fp32 sigmoid** | `fp-sigmoid` | the repo had mul/add/rsqrt/exp and **no divide**; Newton reciprocal plus saturation to *exactly* 1.0 and *exactly* 0.0 — neither reachable in bf16 |
| **Hyper-connections** | `mhc-*`, `hc-block` | four parallel residual streams and the 4×4 doubly-stochastic mix, map *and* residual path |
| **KDA attention** | `kda*`, `kda-attn` | the whole sublayer, fetching its own Q8_0 projections |
| **MoE** | `moe-router`, `swiglu-mt`, `moe-ffn` | sigmoid gating with `exp_probs_b`, the mixed-type expert, the expert loop, and the always-on shared expert at weight 1 |
| **MLA attention** | `mla-*`, `glm53f-mla-attn` | **NoPE** — no rotary anywhere in the path; bitwise end to end |
| **The decoder layer** | `dec-block` | **all 45 blocks**, in every attention × FFN combination the checkpoint uses. The arms are runtime-selectable, and the gate proves it by **equivalence**: the same vectors and golden, rebuilt with both arms in silicon and the selectors pointing at the ones the golden describes, must give the identical result |
| **Per-tensor descriptors** | `wdesc` | `(kind, layer, expert) → (base, klen, nsblk, wtype)`. The system had never had one for *any* type — it drove the loader with a hardcoded tile and left `desc_wtype` undriven |
| **The 45-layer walk** | `layers` | one block run 45 times, at the real L: streams loaded once, the checkpoint's own MLA/dense schedule per layer, every pull annotated with its layer |

### Open — and none of it is a missing machine

- **A layer stack is not a model.** The walk runs 45 blocks; there is still no **embedding, final
  norm, LM head or sampler** around it, and nothing drives it from a token stream.
- **The descriptor's real byte offsets.** The machine is built and gated; turning the census into
  actual offsets needs the GGUF **tensor map**, which needs the checkpoint — see below. That is a
  data step, not an RTL step.
- **State placement is decided but not plumbed.** Traffic, not capacity, settles it: the KDA state
  is a full read-modify-write of 148 MB that never grows, the KV cache is 11.8 GB read by a *sparse
  gather* of 2048 latents, and the DSA index is a **full scan every token** — 16× the KV's traffic
  on 3 % of its capacity. None of it belongs on-die, and every piece is already a *port* rather than
  storage, so what is missing is the per-layer addressing, not a memory.
- **The llama.cpp seal** is physically blocked: no checkout of the 199.7 GB model here, so every
  tok/s figure for this target stays unmeasured.

A passing elaboration is not a working model, and this list is the difference.

---

## Why a second model was cheap — and what it actually cost

Porting to Laguna-S-2.1 measured how much of this design is model-independent:

- **Inherited unchanged (~70–80%).** The Q4_K / Q6_K / Q8_0 dequant contract is *format-level*, so
  it carries to any GGUF k-quant with no work at all. The Q4_K GEMM core, RMSNorm, softmax, the MoE
  router / expert path, the whole memory system (multi-channel DDR5 + expert cache + KV pager +
  weight/boot loaders + multi-clock CDC), and the entire verification harness are
  dimension-parameterized.
- **Genuinely new per model: the attention machine.** GLM-5.2 uses MLA + DSA sparse attention;
  Laguna uses GQA with per-layer head counts, sliding-window layers, dual YaRN/plain RoPE and
  per-head softplus output gating. That is what each model branch actually builds.

That split is why the model branches exist — and why shared-core work is worth merging across them
rather than forking outright.

**GLM-5.3-Flash is the case that tests the 70–80% figure, and partly breaks it.** Being a
same-family successor was expected to make it the *cheapest* port yet. It is not:

- The attention machine changed again, and this time it **doubled**: the model is a hybrid, so the
  branch needs **two** of them — 34 of 45 layers are KDA linear attention, which did not exist
  here, and the other 11 are MLA+DSA. The inherited MLA machine did not carry over either: this
  model is **NoPE**, so its rotary path had to come out, and a zero-width rotary tail is not
  expressible in Verilog. Both are now built, as siblings — but that is two attention machines
  for one port, where the 70–80 % figure assumed one.
- The residual path changed — hyper-connections with Sinkhorn normalization on every block. Nothing
  in the "attention is the only per-model part" split anticipated that.
- **The dequant contract was the one thing assumed to be free**, because it is format-level rather
  than model-level. GLM-5.3-Flash's UD-Q4_K_XL mix uses **Q5_K**, which this repo had never
  implemented, for 34.9% of its bytes (it has since landed on the port branch — as
  Q4_K with a wider code on an existing bus, which is the cheap case; it still
  had to be *built*, and its gate still had to be made non-vacuous). Format-level portability holds only across the *set of
  formats already built*; a k-quant mix is a per-checkpoint fact, and it must be read from the GGUF
  rather than assumed.

It also moves the hardware plan, in the other direction. Only 11 of 45 layers cache KV and NoPE strips
the rotary tail off the cached latent, so the 1M-context KV falls from ~94 GB to **11.8 GB** and the
whole resident footprint from ~561 GB to **212 GB**. That re-sizes rung ③ (256 GB — but built as
16 × 16 GB, because the 1024-bit bus follows package count, not capacity), leaves rung ④'s HBM tier
~8× oversized, and newly puts an all-HBM residency in reach that GLM-5.2 could not touch. None of it
changes the tok/s, which still divide by an `A_eff` this model has not been measured for.

Details: [`docs/GLM53_FLASH_PORT.md`](https://github.com/Wick-Lim/WPU/blob/glm5.3-flash/UD-Q4_K_XL/docs/GLM53_FLASH_PORT.md)
and [`docs/HARDWARE_LADDER.md`](https://github.com/Wick-Lim/WPU/blob/glm5.3-flash/UD-Q4_K_XL/docs/HARDWARE_LADDER.md)
§"GLM-5.3-Flash re-sizing" on the port branch.

---

## The discipline

Every claim carries the *kind* of evidence behind it, and the words are not interchangeable:
**PROVEN** (a gated bit-exact / functional simulation), **FORMAL** (a solver proof), **MEASURED**
(real RTL cycles or a real silicon fit), **ELABORATED** (structural only), **[EST]**
(roofline-modeled, *not* measured on silicon), **NOT-YET** (a real, open gap, stated as one).

Two habits keep that honest:

- **Every load-bearing gate is paired with a must-fail injection build.** A test that cannot fail
  proves nothing, so each one is also run against a deliberately broken variant it *must* catch.
- **The release gate pins the exact test count of every gate.** A testbench that silently runs
  fewer tests than intended is a regression, and the manifest turns that into a build failure —
  `ALL <n> GATE COUNTS MATCH`, or the build fails.

The gate is one command (`make release-gate-par`) and takes a couple of hours: it runs the whole
ladder six targets at a time, each into its own log, then concatenates them in declared order so the
count check reads a log identical to a serial run. It also holds the machine awake, which is not a
nicety — two earlier runs measured 18 h and 49 h wall, of which 16 h and 41 h were the machine
asleep.

What is *not* done is stated as plainly as what is: no silicon exists, no throughput figure has been
measured on hardware, and llama.cpp whole-runtime numeric equality is out-of-contract by design.
The per-claim ledger lives in each model branch's README.

---

## Repository layout

| Branch | Contents |
|---|---|
| `main` (this) | Project hub: this README, the [project site](https://wick-lim.github.io/WPU/), and the [paper](paper/). |
| [`glm5.3-flash/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.3-flash/UD-Q4_K_XL) | The GLM-5.3-Flash port: the locked config and its two-sided guard, Q5_K, the KDA and MLA attention machines, hyper-connections, the MoE path, the complete decoder layer, the executable specs, the GGUF census and memory-budget tools, and the port ledger. Forked at the GLM-5.2 tip, so it carries every gate that branch has. |
| [`glm5.2/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.2/UD-Q4_K_XL) | The GLM-5.2 accelerator: RTL, testbenches, `make` gates, docs, host runtime, FPGA flow. |
| [`laguna-s-2.1/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/laguna-s-2.1/UD-Q4_K_XL) | The Laguna-S-2.1 port: locked config, executable references, gates. |

Preserved history, referenced as prior work and never as current: **`fp8-verified-baseline`** (the
earlier FP8 datacenter track) and **`compression-study-baseline`** (a weight-compression research
study) — both **tags**, inspectable with `git checkout fp8-verified-baseline`.

## License

[Apache-2.0](LICENSE). The repository-level license governs all files; there are no per-file SPDX
headers by policy.
