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
| **GLM-5.2** | [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF)<br>753B MoE (~40B active/token), ~467 GB | [`glm5.2/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.2/UD-Q4_K_XL) | **Flagship.** Full datapath bit-exact vs an independent ggml reference, memory-system controllers formally verified, whole product top placed & routed on a real FPGA. The paper is about this build. |
| **Laguna-S-2.1** | [`unsloth/Laguna-S-2.1-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)<br>118B MoE (~8B active/token) | [`laguna-s-2.1/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/laguna-s-2.1/UD-Q4_K_XL) | **Port in progress.** Dequant inherited unchanged; MoE path bit-exact in RTL at Laguna's config; the (different) GQA attention machine is specified and reference-verified end to end — the bit-exact orchestrator RTL is scoped, not yet written. |

**Branch naming:** `<model>/<quantization>`, e.g. `glm5.2/UD-Q4_K_XL`. A second quantization of the
same model is a sibling branch under the same model prefix.

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
  fewer tests than intended is a regression, and the manifest turns that into a build failure.

What is *not* done is stated as plainly as what is: no silicon exists, no throughput figure has been
measured on hardware, and llama.cpp whole-runtime numeric equality is out-of-contract by design.
The per-claim ledger lives in each model branch's README.

---

## Repository layout

| Branch | Contents |
|---|---|
| `main` (this) | Project hub: this README, the [project site](https://wick-lim.github.io/WPU/), and the [paper](paper/). |
| [`glm5.2/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/glm5.2/UD-Q4_K_XL) | The GLM-5.2 accelerator: RTL, testbenches, `make` gates, docs, host runtime, FPGA flow. |
| [`laguna-s-2.1/UD-Q4_K_XL`](https://github.com/Wick-Lim/WPU/tree/laguna-s-2.1/UD-Q4_K_XL) | The Laguna-S-2.1 port: locked config, executable references, gates. |

Preserved history, referenced as prior work and never as current: **`fp8-verified-baseline`** (the
earlier FP8 datacenter track) and **`compression-study-baseline`** (a weight-compression research
study) — both **tags**, inspectable with `git checkout fp8-verified-baseline`.

## License

[Apache-2.0](LICENSE). The repository-level license governs all files; there are no per-file SPDX
headers by policy.
