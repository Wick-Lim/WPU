# Porting WPU to Laguna-S-2.1 (UD-Q4_K_XL)

> **Status (branch `laguna-s-2.1`).** Config LOCKED + verified · dequant inherited
> (format-level) · **MoE path bit-exact in RTL** · attention + full forward
> **reference-verified end to end** · GQA-orchestrator RTL scoped, not yet written.
> One command runs every Laguna gate: **`make laguna`** (config-check + MoE + attn/
> model references + datapath elaboration). Per-piece evidence levels are in §6 —
> **no claim of a running / RTL-bit-exact Laguna accelerator is made.** `main` untouched.

Target: **`unsloth/Laguna-S-2.1-GGUF : UD-Q4_K_XL`** — a 118B MoE (~8B active/token)
in the same GGUF k-quant ecosystem as the GLM-5.2 build that `main` targets.

This branch (`laguna-s-2.1`) adds a second model target **without disturbing the
verified GLM-5.2 datapath on `main`**. The rule stays: bit-exact or it doesn't ship —
every new block gets a golden gate + a must-fail injection pair, same discipline as
the GLM path.

> **Provenance.** The config below is **CONFIRMED** from `poolside/Laguna-S-2.1`
> `config.json` (arch `LagunaForCausalLM`, base model of the unsloth GGUF). Items
> still to read from the **GGUF metadata / reference kernels** at build time are
> marked **[VERIFY]** — do NOT hardcode a guessed value there.

---

## 1. What the model is (confirmed from config.json)

| Property | Laguna-S-2.1 | GLM-5.2 (`main`) |
|---|---|---|
| Total / active params | 118B / ~8B per token | 753B / ~40B |
| Arch id / dev | `LagunaForCausalLM` / poolside | `GlmMoeDsaForCausalLM` |
| hidden_size | **3072** | 6144 |
| vocab_size | **100,352** | 154,880 |
| layers | **48** (layer 0 dense FFN, layers 1–47 MoE) | 78 (3 dense-front) |
| MoE | 256 experts, **top-10**, +1 shared; `norm_topk_prob`, `moe_routed_scaling_factor 2.5`, no logit softcap | 256, top-8, +1 shared |
| MoE expert FFN / shared FFN | **1024 / 1024**; dense FFN **12288** | — |
| Attention | **GQA**, head_dim **128**, **8 KV heads (const)**, Q heads **48 on full-attn layers / 72 on sliding layers**, `attention_bias=false` | MLA + DSA |
| Output gating | **per-head, all 48 layers** (`gating=per-head`) | none |
| Layer layout | `[full, sliding×3] × 12` → **12 full + 36 sliding**, window **512** | uniform MLA+DSA |
| RoPE | **dual**: full = **YaRN** (θ=500k, factor 128, orig_max 8192, β_slow 1 / β_fast 32, attn_factor 1.4852, **partial_rotary 0.5**); sliding = **plain** (θ=10k, **partial_rotary 1.0**) | single partial-rope |
| RMSNorm eps / dtype | **1e-6** / bf16 | 1e-5 / bf16 |
| Context | 1,048,576 | 1,048,576 |
| Tokens | bos 2, eos [2,24], pad 9; `tie_word_embeddings=false` | — |
| Quant | UD-Q4_K_XL (Q4_K + Q6_K + Q8_0 + F16 mix) | UD-Q4_K_XL (same mix) |

**Consequence of "smaller":** hidden is only 3072 and just ~8B params stream per
token. At Q4 the whole checkpoint is **~70 GB** (vs 467 GB), so on the same rung-③
512 GB / ~1.1 TB/s box it fits with huge headroom and the roofline denominator
shrinks with the active footprint → **several× faster than GLM-5.2** [EST]. **Caveat:**
no MTP head (§4) means the speculative-decode multiplier does **not** transfer, so
that "several×" is the *no-spec* roofline, not an apples-to-apples spec number.

---

## 2. What carries over unchanged (the foundation — already verified on `main`)

These are format-level or fully parameterized, so they inherit directly:

- **Q4_K / Q6_K / Q8_0 dequant** — bit-exact, model-agnostic (the moat). This is
  exactly why "이거도 Q4_K_XL로" works: the dequant contract is inherited, not rebuilt.
  → re-run `tools/gguf_crosscheck.py` against the Laguna GGUF to re-seal it on the
  real bytes (Phase 1).
- **Q4_K GEMM core** (`glm_matmul_q4k`), **RMSNorm**, **softmax** — dimension-parameterized.
- **MoE router + top-k selector** (`moe_router_q4k`, `topk_select`) + **expert path**
  (`swiglu_expert_q4k`) — `TOPK` and `N_EXPERT` are parameters; **top-10 is a config
  change** (`TOPK=8 → 10`), 1 shared expert already modeled.
- **Memory system** — `ddr5_xbar`, `weight_loader_q4k`, `expert_cache_pf`,
  `kv_cache_pager`, boot loader, CDC. Medium-agnostic; re-sized by config.
- **The entire verification harness** — bit-exact gates, injection pairing, the
  68→79-gate manifest, PHY-closure loopback, cycle-stall emulation. New Laguna
  blocks plug into the same `make` gate discipline.

---

## 3. What is genuinely new (attention is a different machine)

The current attention orchestrator is MLA+DSA-specific. Laguna is GQA, and config.json
surfaced **four** new elements the repo does **not** have today (grep-confirmed on `src/`):

1. **GQA attention orchestrator** — head_dim 128, **8 KV heads**, grouped-query. No latent
   c_kv/k_rope compression, no DSA indexer. Reuses the inner QK^T / softmax / V blocks;
   the orchestrator is new. **Extra wrinkle:** the **Q-head count is per-layer** — 48 on
   full-attention layers (group 6), 72 on sliding layers (group 9) — so the orchestrator
   must be parameterized per-layer, not once globally.
2. **Sliding-window attention (window 512)** — NOT implemented today (`src/` "window"
   hits are all clock-gating). Needs windowed causal masking in the KV gather **plus a
   per-layer selector** driving the `[full, sliding×3]×12` layout (12 full / 36 SWA).
3. **Dual RoPE with YaRN** — full-attention layers use **YaRN** scaling (θ=500k, factor
   128, orig_max 8192, β_slow 1 / β_fast 32, attention_factor 1.4852) at **partial_rotary
   0.5**; sliding layers use **plain** rope (θ=10k) at **partial_rotary 1.0**. Our
   `rope_interleave_unit` does a single partial-rope — **YaRN's per-dim frequency ramp +
   attention_factor scaling, and the per-layer-type rope select, are new.**
4. **Per-head output gating (softplus), all 48 layers** — new datapath on the attention
   output: per-head `out *= softplus(gate)`. softplus differs from the SiLU/sigmoid in
   `glm_act.v` — add/verify the exact form **[VERIFY]** against the reference kernel.

Config-only deltas (parameterized paths, no new logic): `TOPK=8→10`, `norm_topk_prob`,
`moe_routed_scaling_factor=2.5`, hidden 3072, vocab 100352, expert/shared FFN 1024,
dense FFN 12288, layer-0-dense / 1–47-MoE, RMSNorm eps 1e-6.

---

## 4. Open items — RESOLVED from config.json (2026-07), with what remains

Resolved from `poolside/Laguna-S-2.1/config.json`:

- **Q heads / group:** 48 (full) / 72 (sliding), KV 8 → group 6 / 9. head_dim 128. ✅
- **Dims:** hidden 3072, dense FFN 12288, MoE expert FFN 1024, shared FFN 1024, vocab
  100352, RMSNorm eps 1e-6. ✅
- **Dense-front:** exactly **one** dense layer (layer 0); layers 1–47 are MoE. ✅
- **RoPE:** dual scheme — full = YaRN / partial 0.5, sliding = plain θ10k / partial 1.0. ✅
- **MoE:** top-10, `norm_topk_prob=true`, `moe_routed_scaling_factor=2.5`, no logit
  softcap, 1 shared expert. ✅
- **Speculative decoding:** **NO MTP / draft head in config** (no `mtp`/`nextn`/`predict`
  keys). → `glm_q4k_spec_system` composition and the GLM-measured A_eff (1.87) / accept
  rate (0.87) do **NOT** transfer. Laguna runs at the **no-spec roofline** unless an
  external drafter (e.g. a small Laguna, or n-gram) is added later. This is the single
  biggest throughput consequence of the port.

Still **[VERIFY]** at build time (need GGUF metadata / reference kernels, not just config):

- Exact **softplus output-gating** formula (scale/bias, applied pre- or post-`V`) — read
  `modeling_laguna.py` / a reference forward.
- **YaRN** application details: the per-dim wavelength ramp between β_fast/β_slow and how
  `attention_factor` multiplies the scores — match the reference bit-for-bit.
- Whether the **GGUF tensor names / metadata** expose the per-layer head counts, layer
  types, and gating so `tools/gguf_crosscheck.py` can map them (vs config-driven).
- Global (full-attention) layers: confirm plain full-causal (no window) + YaRN only.

---

## 4b. Laguna MoE forward — exact order (from modeling_laguna.py) and the GLM delta

Extracted from `poolside/Laguna-S-2.1/modeling_laguna.py` (`tools/laguna_moe_ref.py`
is the numpy port). `hidden_act` defaults to **silu** → SwiGLU, identical to `main`.

```
router (LagunaTopKRouter):
  logits = x @ Wg.T            (fp32)         # softcap 0.0 -> off
  scores = sigmoid(logits)                    # SIGMOID scoring, NOT softmax
  sel    = topk(scores + bias, k=10)          # bias = e_score_correction_bias = 0 in
                                              #   the GGUF (zero-init) -> topk by score
  w      = scores.gather(sel)                 # gather UNBIASED scores
  if norm_topk_prob: w /= w.sum(-1)           # normalize the top-k weights
experts:  down( silu(gate(x)) * up(x) ) * w_e                       # SwiGLU
block:    out = (Sigma_e experts_e) * 2.5  +  MLP_shared(x)         # scale routed SUM,
                                                                   #   add UNSCALED shared
```

**The ONE delta vs `main`'s `moe_router_q4k`:** GLM folds the scale into each routed
weight (`w_j=(gate_j/s)*2.5`, so `Sum(w_j)=2.5`); Laguna keeps `w` normalized
(`Sum=1.0`) and scales the routed **sum** (then adds the unscaled shared expert).
Algebraically identical, fp-rounding differs. **No leaf-module change is needed:**
`moe_router_q4k` already has a `SCALE` parameter — Laguna runs it at `SCALE=1.0`
(normalize only) and the decoder-block combine applies `x2.5` to the routed
accumulator before adding shared. Everything else — sigmoid, top-k, renorm, SwiGLU
experts, the 1-shared-expert structure — is `main`'s modules re-parameterized.
`tools/laguna_moe_ref.py --selftest` exhibits the fold-vs-sum-scale fp delta.

---

## 5. Phased plan (each phase is bit-exact-gated before the next)

1. **Phase 1 — inherit + config.** ✅ DONE. `configs/full_laguna_s21.vh` locks the shape;
   `make laguna-config-check` (195 asserts + injection) proves it. Dequant is format-level:
   `tools/gguf_crosscheck.py` runs unchanged on the Laguna GGUF (deferred to download).
2. **Phase 2 — MoE + FFN at Laguna shape.** ✅ DONE. `make laguna-moe`: the Laguna MoE
   numeric-order reference self-test + `moe_router_q4k` at Laguna **256/top-10** (renorm
   invariant) + `swiglu_expert_q4k` at Laguna **INTER_MOE=1024** (Q4_K bit-exact vs the
   ggml ref). Confirms the leaf modules carry unchanged; the scale-placement (SCALE=1.0 +
   combine x2.5, §4b) is deferred to the Phase-6 decoder-block combine, not a leaf change.
3. **Phase 3 — GQA attention.** ✅ Reference DONE. `make laguna-attn`
   (`tools/laguna_attn_ref.py`, 16 self-tests): GQA (per-layer Q heads), q/k per-head
   RMSNorm, causal mask. The composing leaves (Q4_K GEMM / RMSNorm / softmax) type/width-
   check at Laguna's attention shape — `make laguna-datapath-elab`.
4. **Phase 4 — sliding-window + dual RoPE/YaRN.** ✅ Reference DONE (in `laguna-attn`):
   `rotate_half` (not GLM interleave) partial RoPE, YaRN (full layers) + plain (sliding),
   sliding-window mask + the `[full, sliding×3]×12` schedule — all self-tested (a token
   outside the window provably cannot affect the output).
5. **Phase 5 — softplus output gating.** ✅ Reference DONE (in `laguna-attn`): per-head
   `out *= softplus(x @ Wg)` before o_proj, self-tested.
6. **Phase 6 — assemble.** ✅ Reference DONE. `make laguna-model`
   (`tools/laguna_model_ref.py`, 11 self-tests): embed → 48×(attn + MoE/dense) → norm →
   LM head → argmax, dense@0 + full/sliding schedule, **end-to-end causal**. **No
   spec-decode** — Laguna has no MTP head (§4), so the box runs the no-spec roofline.
7. **Phase 7 — gate.** ✅ `make laguna` aggregates every Laguna gate (config-check + MoE +
   attn/model references + datapath elaboration). Manifest pins for the RTL leaf gates
   land when the bit-exact orchestrator does (below).

Foundation (dequant + MoE + memory + verification infra) carries ~70–80% of the work; the
new surface is the attention machine — **GQA (per-layer head count) + sliding-window +
dual-RoPE/YaRN + softplus output gating**.

## 6. Verification-level ledger — what is proven, and how (no overclaim)

The repo's discipline is to state the EVIDENCE LEVEL of every claim. For the Laguna port:

| Piece | Level | Evidence |
|---|---|---|
| Dequant (Q4_K/Q6_K/Q8_0) carries | **format-level, inherited** | `tools/gguf_crosscheck.py` runs unchanged; re-seal on real bytes deferred to download |
| Config (layers/heads/experts/rope) | **VERIFIED vs config.json** | `make laguna-config-check` — 195 asserts + injection; audited vs `docs/laguna_s21_config.json` |
| MoE path (router/expert) at Laguna dims | **bit-exact RTL (leaf)** | `make laguna-moe` — `moe_router_q4k` 256/top-10 + `swiglu_expert_q4k` INTER=1024, Q4_K bit-exact vs ggml ref |
| Attention math (GQA, q/k-norm, rotate_half+YaRN, SWA mask, softplus gate) | **reference-verified (numpy)** | `make laguna-attn` — 16 self-tests |
| Full forward (schedule, dense/MoE dispatch, causality) | **reference-verified (numpy)** | `make laguna-model` — 11 self-tests |
| GQA datapath leaves at Laguna shape | **elaboration** | `make laguna-datapath-elab` — type/width-check, no sim |

**Not yet done (the remaining silicon-adjacent RTL), stated plainly:** a bit-exact RTL
GQA *orchestrator* that streams the leaves (Q/K/V/O GEMM → q/k-norm → rotate_half/YaRN
RoPE → GQA gather + causal/SWA mask → softmax → ×V → softplus gate → o_proj) against a
weight-DMA harness, checked bit-exact vs `laguna_attn_ref` with an injection pair; a new
**softplus** activation leaf (main's `glm_act` has SiLU, not softplus) and a **YaRN /
rotate_half** RoPE leaf (main's `rope_interleave_unit` is interleaved plain-rope); the
full-model RTL assembly + a `laguna-release-gate-strict` manifest. These are the same
class of work the GLM-5.2 attention (`mla_attn_q4k`) represents on `main` — the Laguna
port has the executable golden (references) and the config/MoE/datapath proofs to build
them against, but the orchestrator RTL itself is not written. **No claim of a running or
bit-exact-in-RTL Laguna accelerator is made here** — only that the port is fully
specified, its MoE path is bit-exact in RTL, and its attention/forward are reference-
verified end to end.
