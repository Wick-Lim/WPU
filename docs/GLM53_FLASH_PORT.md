# Porting WPU to GLM-5.3-Flash (`UD-Q4_K_XL`)

> **Branch `glm5.3-flash/UD-Q4_K_XL`, forked at the `glm5.2/UD-Q4_K_XL` tip.**
>
> **Model config: LOCKED.** Every dimension is now a hard citation from the
> published checkpoint — `unsloth/GLM-5.3-Flash-GGUF : UD-Q4_K_XL` and
> `zai-org/GLM-5.3-Flash` — not an assumption. See `configs/full_glm53_flash.vh`.
>
> **Datapath: NOT COMPLETE, and not by a small margin.** GLM-5.3-Flash is not a
> re-dimensioned GLM-5.2. It is a different architecture (`glm5next`), and
> **34 of its 45 layers are a machine this repo does not have.** Nothing on this
> branch claims a running GLM-5.3-Flash accelerator. The hub (`main`) states the
> same split.

This document supersedes `docs/GLM53_PORT.md` from the deleted `glm5.3/UD-Q4_K_XL`
scaffold branch (commit `5576383`), which was written on 2026-08-19 when no
GLM-5.3 checkpoint existed. That scaffold's central prediction — "expected
config-only deltas: dims, expert count, scaling factors" — **was wrong**, and its
own §2 item 2 said what to do about it: *"Confirm the arch id. If it is NOT
`GlmMoeDsa*`-family, stop: the MLA/DSA orchestrator is model-specific work,
scope it like the Laguna GQA port."* That is exactly the branch we are on.

---

## 1. What the checkpoint turned out to be

`general.architecture` in the GGUF is **`glm5next`**; `config.json` says
`Glm5NextForConditionalGeneration`. GLM-5.2 was `GlmMoeDsaForCausalLM`. The
family assumption the scaffold was built on does not hold.

The structural change is the layer stack. GGUF `attention.head_count_kv` is a
per-block list, and it reads `[0,0,0,1, 0,0,0,1, …]` — `0` marks a block with no
KV heads at all, i.e. a **linear-attention (KDA) block**; `1` marks a
**MLA + DSA block** of the kind this repo has built and proven:

| block class | count | what it is | do we have RTL? |
|---|---|---|---|
| KDA linear attention | **34 / 45** | gated delta-rule recurrence: short causal conv (k=4) on q/k/v, decay `ssm_a`, `ssm_dt` bias, `f`/`g` low-rank gates, per-head norm | **no** |
| MLA + DSA | 11 / 45 | the GLM-5.2 attention machine, NoPE, with a k-pooled indexer | yes, inherited |
| MTP / nextn | block 45 | MLA-shaped speculative head | yes, inherited |

Full-attention blocks are `{3, 7, 11, …, 43}` — strictly every 4th block
starting at 3, which is also exactly where the 3-block dense-FFN front ends.
Both facts are asserted at elaboration in `test/glm53f_dims_wrap.v`.

Three further changes are not dimensions either:

- **NoPE.** GGUF `rope.dimension_count = 0`, `config.json` `qk_rope_head_dim = 0`,
  `mla_use_nope = true`. There is no rotary embedding anywhere in the attention
  path. `src/rope_interleave_unit.v` has no consumer in this model. A direct
  consequence, asserted in the gate: `attention.key_length == kv_lora_rank`
  (both 512) — on GLM-5.2 the key was the latent *plus* a 64-wide rotary tail.
- **Hyper-connections.** Every block carries `hc_attn_{base,fn,scale}` and
  `hc_ffn_{base,fn,scale}` (`hyper_connection.count = 4`,
  `sinkhorn_iterations = 20`). This replaces the plain residual add — a change to
  the block's structure, not a coefficient.
- **Clamped SwiGLU.** `swiglu_clamp_exp` / `swiglu_clamp_shexp` = 10.0 on every
  block. GLM-5.2 has no clamp. An unclamped SwiGLU here is numerically wrong,
  not merely approximate.

The model is also `ForConditionalGeneration` with a `vision_config` (24-layer
ViT, 448px, patch 14). **The UD-Q4_K_XL GGUF ships text weights only** — there is
no vision tensor among its 1412 — so the text-path contract is complete without
it. The omission is deliberate, not an oversight.

## 2. The shape, as published

Sources: `[gguf]` GGUF metadata KV, `[cfg]` `config.json` `text_config`,
`[scan]` derived from the tensor map by `tools/glm53_flash_gguf_scan.py`.

| field | GLM-5.2 | **GLM-5.3-Flash** | note |
|---|---|---|---|
| arch id | `GlmMoeDsaForCausalLM` | **`glm5next`** | family change |
| hidden_size | 6144 | **4096** | |
| num_hidden_layers | 78 | **45** | + 1 MTP block = 46 `[gguf]` |
| layer stack | 78 × MLA+DSA | **34 × KDA + 11 × MLA+DSA** | the port |
| first_k_dense_replace | 3 | 3 | |
| vocab_size | 154880 | 154880 | unchanged |
| context | 1 M | 1 M | POSW = 20 unchanged |
| num_attention_heads | 64 | 64 | |
| qk_nope_head_dim | 192 | **256** | |
| qk_rope_head_dim | 64 | **0** | NoPE |
| rope_theta | 8e6 | **absent** | no rotary at all |
| v_head_dim | 256 | 256 | |
| q_lora_rank | 2048 | **1536** | |
| kv_lora_rank | 512 *(assumed)* | **512 (confirmed)** | see below |
| index_topk | 2048 | 2048 | |
| indexer heads / dim | — | 32 / 128 | + kpool 4, compress, tail-select |
| n_routed_experts | 256 | **288** | |
| num_experts_per_tok | 8 | 8 | |
| moe_intermediate_size | 2048 | 2048 | |
| intermediate_size | 12288 | 12288 | |
| routed_scaling_factor | 2.5 | 2.5 | |
| swiglu clamp | none | **10.0** | new |
| hyper-connections | none | **mult 4, Sinkhorn 20** | new |
| MTP head | yes | **yes** (`nextn_predict_layers = 1`) | |
| total params | 753 B | **320.759 B** | `[scan]` |
| active / token | ~40 B | **17.377 B** | `[scan]`, top-8/288 + dense |
| checkpoint size | ~467 GB | **199.70 GB** | `[scan]`, UD-Q4_K_XL |

**A GLM-5.2 debt this retires:** `configs/full_glm52.vh` carried `kv_lora_rank =
512` as the *DeepSeek-MLA standard assumption*, PENDING safetensors
confirmation. GLM-5.3-Flash publishes `attention.kv_lora_rank = 512` explicitly
in its GGUF metadata. That confirms the value for 5.3-Flash; it remains an
assumption for GLM-5.2 itself, which is a different checkpoint.

## 3. The quantization mix, measured

`[scan]`, all 1412 tensors of the UD-Q4_K_XL build:

| ggml type | tensors | bytes | share | where | kernel here? |
|---|---|---|---|---|---|
| Q4_K | 84 | 114.15 GB | 57.2 % | `ffn_{gate,up}_exps` | yes, bit-exact |
| **Q5_K** | **42** | **69.76 GB** | **34.9 %** | `ffn_down_exps` | **no — none in this repo** |
| Q8_0 | 645 | 9.62 GB | 4.8 % | attention, shared expert, embed, lm_head | yes, bit-exact |
| Q6_K | 3 | 5.95 GB | 3.0 % | UD bump on `blk.{11,12,44}.ffn_down_exps` | yes, bit-exact |
| F32 | 638 | 0.23 GB | 0.1 % | norms, routing gates, `ssm_a`/`dt`/`conv1d` | n/a |
| **total** | **1412** | **199.70 GB** | | | |

That total cross-checks against the published shard sizes (199.71 GB) to within
the 9.52 MB of GGUF headers. The agreement is the evidence that the tensor parse
and the k-quant block arithmetic are right — an unverified parse would not land
on the file sizes.

**Q5_K is a hard blocker, and it is new.** This repo implements Q4_K, Q6_K and
Q8_0 (`tools/q4k_ref.py` + the RTL dequant), all proven bit-exact against real
ggml on real GGUF bytes. Q5_K appears nowhere in `src/`, `tools/`, `test/` or the
`Makefile`. It covers **a third of this checkpoint's bytes**, so without it the
model cannot be read at all, let alone run.

The **UD "Dynamic" bumps** matter for the packer: `blk.11` has its
`ffn_{gate,up}_exps` promoted Q4_K → Q5_K, and `blk.{11,12,44}.ffn_down_exps` are
promoted Q5_K → Q6_K. Any loader that assumes a fixed type per tensor family will
mis-read those blocks. The type must be taken per tensor, from the GGUF.

## 4. Port scope

### 4.1 Inherited and still valid (evidence carried by fork)

These are **GLM-5.2 proofs**, measured at GLM-5.2's shape. They transfer as
*machines*, not as claims about GLM-5.3-Flash:

- Q4_K / Q6_K / Q8_0 dequant + GEMM core, RMSNorm, softmax — bit-exact vs ggml,
  dimension-parameterized.
- The MLA + DSA + MoE + MTP datapath — bit-exact vs a numpy reference at the
  slice. It is the right machine for **11 of 45** blocks here.
- Memory system (DDR5 xbar, expert cache + MSHR, KV pager, loaders, CDC) —
  formally verified controllers; FPGA fit measured on XCKU3P.
- Verification harness: `make` gates, must-fail injection pairs, pinned test
  counts, the L3 board-boot E2E chain.

### 4.2 New work, in dependency order

| # | item | why it blocks | rough shape |
|---|---|---|---|
| 1 | **Q5_K dequant** — reference + RTL + bit-exact gate | 34.9 % of bytes unreadable | smallest item; a sibling of the proven Q4_K/Q6_K path, same super-block structure, and the existing `gguf_crosscheck` harness extends to it directly |
| 2 | **KDA linear-attention block** | 34 / 45 layers | the real work: short causal conv (k=4) on q/k/v, gated delta-rule state update, `f`/`g` low-rank gates, decay from `ssm_a` + `ssm_dt`, per-head norm. New datapath, new state memory, new golden reference |
| 3 | **Hyper-connections** | every block's residual path | Sinkhorn normalization (20 iters) over a width-4 connection matrix; needs a numerically-careful fixed-point study before RTL |
| 4 | **Clamped SwiGLU** | every FFN | small: a clamp on the existing unit, plus a golden re-run |
| 5 | **Indexer compressor** | the 11 DSA blocks | k-pool 4 with compression and always-select-tail; `indexer_compressor_{ape,gate}` have no GLM-5.2 counterpart |
| 6 | **Re-target packer / flash layout** | boot path | `tools/ckpt_pack_q4k.py`, `tools/flash_layout.py` against `glm5next` tensor names and the per-tensor UD mix |
| 7 | **Re-seal `gguf_crosscheck`** | the dequant trust row | run against real GLM-5.3-Flash GGUF bytes, including Q5_K |
| 8 | **Re-run the whole gate ladder** | everything above | slice → full-elab at the true shape → `release-gate-strict` with counts re-pinned |

Item 2 is the one that decides the schedule. Items 1, 4 and 6 are mechanical.

### 4.3 What the MTP answer changes

`nextn_predict_layers = 1`, and the MTP block is MLA-shaped, so
`glm_q4k_spec_system` has a real counterpart here. But the **measured GLM-5.2
`A_eff` (1.87) and accept-rate (0.87) do not transfer** — they are properties of
a specific model's draft quality and must be re-measured on GLM-5.3-Flash before
any roofline number is quoted. Until then, every throughput figure for this model
is `[EST]` with a borrowed acceptance input, and is labelled as such.

## 5. Why the memory profile is not GLM-5.2's

Worth stating because it is the most likely thing to be quietly assumed wrong:
**only 11 of 45 blocks hold a growing KV cache.** The 34 KDA blocks carry a
fixed-size recurrent state instead — it does not grow with context length. A
1M-context memory budget derived from GLM-5.2's "every layer pages KV" model
would be badly wrong for this model, in the favourable direction.

That also softens the S_MAX / SWIN caveat (task B7): the attention-scratch
constraint now binds on 11 blocks, not 78. Quantifying the real budget is a
follow-up, and no number for it is published yet.

## 6. Reproducing every number here

Nothing above is asserted from memory. The GGUF headers are ~30 MB of the
199.70 GB checkpoint, so a full re-derivation is cheap:

```sh
python3 tools/glm53_flash_gguf_scan.py --fetch /tmp/glm53f   # ~30 MB
python3 tools/glm53_flash_gguf_scan.py /tmp/glm53f
```

That prints the metadata KV, the layer schedule, the full tensor census, the
quant mix, the byte cross-check against the published shard sizes, the active
parameter count, and the RTL coverage gap.

The config header itself is gated:

```sh
make glm53f-config-guard      # 8/8: 4 cases x 2 tools
```

## 7. What is NOT claimed on this branch

- **No running GLM-5.3-Flash.** 34/45 layers have no RTL, a third of the bytes
  have no dequant kernel, and the residual path is unimplemented.
- **No GLM-5.3-Flash numeric proof.** Every bit-exactness result on this branch
  was measured on GLM-5.2 shapes and is inherited as a *machine*, not as a claim
  about this model.
- **No throughput or cost figure.** All `[EST]`, and until §4.3 is answered even
  the acceptance-rate input is borrowed.
- **No vision path.** Out of scope, and absent from this GGUF.
- **The 199.70 GB checkpoint has not been run end-to-end.** Only its headers
  have been read.
