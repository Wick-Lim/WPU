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
- **Hyper-connections (mHC).** Every block carries `hc_attn_{base,fn,scale}` and
  `hc_ffn_{base,fn,scale}` (`hyper_connection.count = 4`,
  `sinkhorn_iterations = 20`). Reading the reference implementation showed this is
  larger than "the residual add changed": **the block carries `hc_mult = 4`
  parallel residual streams**, and the mHC map produces three things per site —
  `pre` (collapse the 4 streams into the sublayer input), `post` (place the
  sublayer output, range `[0,2]` because it is `2·sigmoid`), and `comb`, a 4×4
  matrix Sinkhorn-projected onto the doubly-stochastic manifold to re-mix the
  streams. The block interface itself is `[4, D]`, not `[D]`.
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
| active / token | ~40 B | **16.742 B** | `[scan]`, top-8/288 + dense; `token_embd` is a row lookup, not a per-token GEMV |
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
| **Q5_K** | **42** | **69.76 GB** | **34.9 %** | `ffn_down_exps` | **yes, bit-exact** — landed on this branch |
| Q8_0 | 645 | 9.62 GB | 4.8 % | attention, shared expert, embed, lm_head | yes, bit-exact |
| Q6_K | 3 | 5.95 GB | 3.0 % | UD bump on `blk.{11,12,44}.ffn_down_exps` | yes, bit-exact |
| F32 | 638 | 0.23 GB | 0.1 % | norms, routing gates, `ssm_a`/`dt`/`conv1d` | n/a |
| **total** | **1412** | **199.70 GB** | | | |

That total cross-checks against the published shard sizes (199.71 GB) to within
the 9.52 MB of GGUF headers. The agreement is the evidence that the tensor parse
and the k-quant block arithmetic are right — an unverified parse would not land
on the file sizes.

**Q5_K was the hard blocker. It is now CLOSED.** Q5_K covers a third of this
checkpoint's bytes (all 42 `ffn_down_exps`) and nothing in this repo implemented
it, so the model could not be read at all. It now can be:

- **Reference** — `q4k_ref.dequantize_block_q5_K`, a verbatim reimplementation of
  ggml's `dequantize_row_q5_K`, including the `u1`/`u2` mask walk (where a
  "simplified" version silently goes wrong for 64-groups 1–3).
- **RTL** — `WT_Q5K` in `glm_matmul_q4k`. Q5_K costs **one type code and no new
  bus**: arithmetically it is Q4_K with a wider code,
  `w = (d·sc)·(q4 + 16·h) − (dmin·m)`, so it reuses the Q4_K header multiplies and
  the min subtract verbatim and rides the existing `w_hp` bus that already carries
  Q6_K's 6-bit code. `w_type` widened 2 → 3 bits/lane; the four existing encodings
  are unchanged and Q4_K stays 0, so the "undriven `w_type` reads Q4_K" safety
  property still holds.
- **Gate** — `make mixedtype`. The Q5_K columns are bit-exact vs the golden, a
  cross-tile coverage assertion requires all five types to appear, and a
  **must-fail injection** (`-DINJ_Q5K_NOMIN`) lets `WT_Q5K` join the P3
  pass-through list and skip the `dmin·m` subtract. Widths still match and no
  error is raised — the weights are just wrong — and the gate fails, which is what
  makes the Q5_K columns evidence rather than passengers.
- **Real published bytes** — the reference was run on actual GLM-5.3-Flash Q5_K
  bytes (range-fetched from shard 3) and cross-validated against the
  already-proven Q6_K kernel on the *same tensor role in adjacent layers*:
  std 0.01844 (Q5_K `blk.13`) vs 0.01850 (Q6_K `blk.11`), ratio **0.9966**. A
  wrong field layout does not land there. `qh`'s high bit is set on 49.5 % of
  weights, as a symmetric 5-bit code requires.

**Two things Q5_K does NOT yet include, stated plainly:**

1. **The llama.cpp seal has NOT been run.** `tools/gguf_crosscheck.py` and
   `tools/dequant_dump.c` now carry a `q5_k` arm, so the gold-standard comparison
   against ggml's own `dequantize_row_q5_K` runs as soon as a llama.cpp build is
   available — but no checkout was present here. So Q5_K's status is *bit-exact to
   our ggml reimplementation, and consistent with real published bytes*, not yet
   *bitwise-equal to llama.cpp on real bytes* the way Q4_K / Q6_K / Q8_0 are.
2. **`weight_loader_q4k` cannot lay out a Q5_K tile.** That needs the packer to
   emit pre-assembled 5-bit codes plus a 176 B/super-block geometry (§4.2 item 6).
   A Q5_K descriptor would otherwise take the Q4_K geometry — same widths, wrong
   bytes, no error — so the loader now `$fatal`s on it in simulation rather than
   streaming garbage.

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
| 1 | ~~**Q5_K dequant**~~ — **DONE**: reference + RTL + gate + must-fail injection | was: 34.9 % of bytes unreadable | landed, see §3. Residual: the llama.cpp seal (needs a checkout) and the loader/packer tile geometry (item 6) |
| 2 | **KDA linear-attention block** — all four non-GEMV units DONE; layer wrapper open | 34 / 45 layers | `src/kda_recur.v` (state update, `make kda`, §4.3c), `src/kda_conv_step.v` (k=4 causal conv + SiLU, `make kda-conv`, §4.3d), `src/kda_gate_step.v` (forget gate → `exp(g)` and beta, `make kda-gate`, §4.3e) and `src/kda_onorm_step.v` (gated RMSNorm on the recurrence output, `make kda-onorm`, §4.3f). What remains is **composition**: the q/k/v/beta/f/g/o projections are ordinary `glm_matmul_q4k` GEMVs, and nothing new is needed *inside* any block — the layer wrapper sequences the existing GEMV engine and these four units, and the state has to live in BRAM/DDR at the real shape (4.19 MB/layer). **Finding carried from §4.3e: the repo's only sigmoid is bf16, priced at ~1–3 % output error per KDA layer; fp32 sigmoid or accept — a decision.** |
| 3 | **Hyper-connections (mHC)** — spec DONE, precision study DONE (§4.3i), map RTL DONE (§4.3k), **residual path RTL DONE** (§4.3l: `mhc_fn_gemv` + `mhc_stream_ops` + `mhc_block_site`, 17 injections across the five units). **block skeleton DONE** (§4.3m: `glm53f_hc_block`, two sites per block). `GLM53F_HC_RTL_PRESENT` is now **defined**; composing it with real sublayers is assembly work | every block's residual path | `tools/glm53_flash_ref.py: hyper_connection`, gated. The block carries **4 parallel residual streams** (interface `[4,D]`). The study's answer: **not fixed-point** — fp32 Sinkhorn (residual @20 iters: median 1.0e-6, worst-of-200 4.8e-4; bf16 is 93× worse on the same inputs; `x·recip(y)` costs 2.7e-7, so no divider is needed), `comb` entries reach 5e-8 so a float exponent is required, and `pre = σ+1e-6` is a third case that needs an fp32 sigmoid |
| 4 | ~~**Clamped SwiGLU**~~ — **DONE** (RTL + 4-leg gate) | every FFN | `SWIGLU_CLAMP` in `swiglu_expert_q4k`, default 0 so the GLM-5.2 path stays byte-identical. Gated by `make q4k`: the feature leg, a **vacuity leg** (the clamp golden must FAIL against an unclamped DUT), and a **must-fail injection** that clamps the gate symmetrically |
| 5 | **DSA indexer — RE-SCOPED: a new indexer front-end, not a compressor bolt-on** | the 11 DSA blocks | Reading `Glm5NextTextIndexer` (§4.3h) showed the GLM-5.2 `dsa_indexer.v` (single index vector, token-level scoring, IndexShare freq-4 reuse) shares only `topk_select` with what GLM-5.3-Flash needs: a k-pool compressor (per-channel 4-way softmax over `gate + APE` logits), **LayerNorm with bias** on `k` (the repo has only RMSNorm), **32 index heads × 128 with ReLU and a `weights_proj` head combination**, scoring over `S/4` pools, top-512 pools → ×4 token expansion + tail append, and no freq-4 reuse. Medium-large, previously mis-scoped as "small" |
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

### 4.3b Clamped SwiGLU — DONE

`SWIGLU_CLAMP` (default **0**) in `src/swiglu_expert_q4k.v`. Off, the committed
GLM-5.2 datapath is byte-identical — the clamp lives on *wires* at the two
consumption points and is never folded into the FSM, because this module's own
`GU_CONC` note records that folding a parameter term into the FSM has broken
default netlist identity three times in this repo.

On, it implements the asymmetry exactly:

```
gate = min(gate, +10.0)          // upper bound ONLY
up   = clip(up, -10.0, +10.0)    // both bounds
h    = silu(gate) * up
```

Four legs in `make q4k`, because a clamp gate is unusually easy to write
vacuously:

| leg | what it rules out |
|---|---|
| `swiglu_expert_q4k(CLAMP)` vs a `--clamp` golden | the feature simply not working |
| the **same golden** vs `SWIGLU_CLAMP=0` must **FAIL** | a golden that passes against *any* DUT — i.e. operands that never actually cross ±10 |
| `-DINJ_SWIGLU_SYMCLAMP` (gate clamped symmetrically) must **FAIL** | the asymmetry going unchecked. This is the plausible wrong reading, and a *small* error — `silu` is near zero for large negative gates — so it is exactly the kind that survives a loose tolerance |
| the generator asserts both clamp directions fired | vectors where the lower bound is never exercised |

Measured clamp activity in the committed vectors: upper bound fired 100×, lower
45×.

**bf16 NaN caveat**, recorded rather than papered over: the clamp compares
magnitudes as unsigned 15-bit integers (exact for finite bf16), so NaN/Inf
saturate to ±limit where `torch.clamp` propagates NaN. Real activations are never
NaN, and handling it would cost gates on the merge path. Same policy as the
`f16_deq` NaN note in `src/q4k_mixed.vh`.

### 4.3c KDA recurrence core — DONE, and what it surfaced

`src/kda_recur.v` implements one decode token of Kimi Delta Attention for `H`
heads over a `[DK, DV]` state, in the golden's exact operation order:

```
S[d][e] *= exp(g[d])                 -- decay BEFORE the kv read
kv[e]    = Σ_d S[d][e]·kn[d]          -- ascending d, sequential
delta[e] = (v[e] − kv[e])·beta
S[d][e] += kn[d]·delta[e]            -- delta rule, not a plain write
out[e]   = Σ_d S[d][e]·qn[d]          -- ascending d, sequential
```

`RECOMPUTE=1` (default) re-applies the decay in the second pass instead of
storing the decayed state: 2 reads + **1 write** of `S` per token rather than
2 + 2. At the real shape the state is 4.19 MB/layer and 285 MB/token of traffic
across 34 layers, so the write is the half worth saving.

**Gate (`make kda`) — three legs and an injection, kept apart on purpose:**

| leg | claim | why it is shaped that way |
|---|---|---|
| generator self-test (80) | the pre-normed operands reproduce the in-kernel-norm recurrence **bitwise** | so the two DUT legs provably check the *same* math |
| `kda_recur(EXACT)` (5184) | fp32 mul/add only; **bounded at 64 ULP**, worst observed 32 | see the `fp32_add` finding below — this is *not* an inherent limit |
| `kda_recur(RSQRT)` (5184) | DUT runs its own l2norm through the Quake `fp32_rsqrt` | a **tolerance** check — the same status `swiglu_expert_q4k` has |
| `-DINJ_KDA_NODECAY` | drops the pass-B decay; **must fail** | a small, compounding error — exactly what a loose tolerance swallows |

**Three things building this surfaced, each now pinned rather than latent:**

1. **`src/glm_fp.vh fp32_add` is not exactly IEEE round-to-nearest-even.**
   Measured: 4/10,000 random pairs land 1 ULP low, concentrated at exponent gaps
   4–5; `fp32_mul` is exactly conformant (0/10,000). This had been invisible
   because **every proven path in this repo ends in bf16**, and a 1-ULP fp32
   difference survives `bf16_round` in only 2/200,000 cases — so the Q4_K core's
   bit-exactness claim is intact and unaffected. `kda_recur` is the first
   consumer whose *output* is fp32, which is why it showed here. `make fp-ieee`
   now measures both primitives over exponent gaps 0–24 and pins a **ceiling**
   (10 ppt10k for add, 0 for mul); an exactly-rounded adder would score 0 and
   still pass. **Fixing the adder is a repo-wide change** (it perturbs every
   pinned netlist) and was deliberately not done here.
2. **Why the EXACT leg is 64 ULP and not 1.** The recurrence *amplifies* the
   adder's gap through cancellation: in `delta = (v − kv)·beta`, when `v ≈ kv`
   a 1-ULP error in `kv` becomes a large *relative* error in `delta`, which the
   update and the output reduction carry forward. Replaying the RTL's exact
   operation order in numpy reproduces the golden **bit for bit**, so the ceiling
   tracks a known, measured, fixable defect — and should drop to 0 if the adder
   is fixed.
3. **numpy `.sum()` is pairwise, not sequential.** Measured: for length-8 fp32
   vectors it differs from a sequential accumulate in **159/300** cases. A
   streaming datapath cannot match a pairwise reference bitwise, so the golden's
   reduction order is pinned sequential (`_seq_sum`) and that order is this
   repo's contract. torch/FLA reduce in their own blocked order; whole-runtime
   equality with them is out of contract — the stance this repo already takes for
   llama.cpp.

Also: `INV_SQRT_DK` is a **parameter**, not computed from `fp32_rsqrt` — deriving
the q scale from the Quake approximation would put an approximation inside the
leg that claims to be exact.

### 4.3d KDA causal conv step — DONE

`src/kda_conv_step.v`: the depthwise K-tap causal conv + SiLU that every KDA
layer runs its q, k and v through before the recurrence
(`causal_conv1d_update`, seq_len = 1). GLM-5.3-Flash: K = 4 (GGUF
`ssm.conv_kernel`), weights `ssm_conv1d_{q,k,v}.weight [4, 1, 8192]` F32, no
bias. **This repo had no conv unit of any kind before this.**

```
window[c] = [ state[c][0..K−2], x[c] ]        oldest → newest
state'[c] =   window[c][1..K−1]                shift in x
conv[c]   = bf16_RNE( Σ_k w[c][k]·window[c][k] )   fp32, taps ascending
y[c]      = silu(conv[c])
```

**Contract, stated up front:** torch runs this conv in bf16 with an
implementation-defined accumulation order, which no fixed datapath can match
bitwise. So the order is **pinned here** — fp32 mul/add, taps ascending
oldest→newest, one RNE round — and the pre-activation bf16 is **exposed on
`conv_out`** so that leg is checked bitwise. Same stance as the KDA reductions
and the llama.cpp comparison.

| leg (`make kda-conv`) | claim |
|---|---|
| generator self-test (400) | the pinned ascending-tap dot equals the reference's to fp32 reassociation, and a **flipped**-tap dot never does on the corpus — so the injection below is live |
| `conv_out` (bitwise) | fp32 mul/add + one RNE round, all exact primitives (modulo `fp32_add`'s pinned gap, which the corpus shows did not move a bf16 rounding boundary) |
| `s_out` (bitwise) | the history shift — pure wiring |
| `y_out` (tolerance) | `silu(conv)` through `glm_act`'s polynomial — the same status `swiglu_expert_q4k` has |
| `-DINJ_CONV_FLIP` must fail | **orientation**: `F.conv1d` *correlates*, it does not flip the kernel, so `w[K−1]` multiplies the *newest* sample. Reversing the taps is the plausible misreading; measured 255/256 `conv_out` mismatches with it |

Verilator-clean (one `TIMESCALEMOD` warning fixed by giving the module the same
`` `timescale `` as the `glm_act` it instantiates).

### 4.3e KDA forget gate + beta step — DONE, with a precision finding

`src/kda_gate_step.v`: the elementwise stage between the gate projections and
the recurrence (`Glm5NextTextForgetGate.forward` + the beta line):

```
t[h,d]  = decay[h] · (f[h,d] + dt_bias[h,d])      decay = exp(A_log[h])
g[h,d]  = −5.0 · sigmoid(t[h,d])                   lower_bound branch
ge[h,d] = exp(g[h,d])                              → kda_recur g_in
beta[h] = sigmoid(b[h])                            → kda_recur beta_in
```

Two design decisions: `decay = exp(A_log)` is a function of a **static weight**
(`ssm_a [64]` F32), so it is host-precomputed once per layer and enters as an
fp32 input — no exp unit is spent on it; and the pipe is chained on **valid
handshakes** (`glm_act` → `fp32_exp_pipe`), not latency constants, so a
re-timed sub-pipe cannot silently desynchronise it.

| leg (`make kda-gate`) | claim |
|---|---|
| generator self-test (5) | on a **deliberately** saturating corpus: `g ∈ [−5, 0]`, both endpoints attained, every saturated zero is `−0.0`, `exp(g)` finite |
| saturation (5 checks) | where the golden's fp32 sigmoid saturated to exactly 0 (`g = −0.0`), the DUT's `g` must be **negative and within `glm_act`'s rail floor** — see the finding |
| `g` / `ge` / `beta` | **tolerance**, rel 0.03 + abs 0.002, both stages approximate |
| `-DINJ_GATE_DECAY_AFTER` must fail | applies `decay` **after** the sigmoid — the plausible misreading of `decay_rate * g`; measured 932 mismatches with it |

**Finding — the bf16 activation unit cannot honour the reference's saturation
contract.** The fp32 reference's sigmoid reaches exactly `0.0` only for
`t < ≈ −104`, and `−5.0 · +0.0 = −0.0` (the signed-zero contract pinned in
`glm53_flash_ref`; `fp32_mul(−5.0, +0.0)` was probed to return `0x80000000`).
But `glm_act` **rails its input at ±16**, and `sigmoid(−16) = 1.13e-7` *is*
representable in bf16 — so the DUT lands at `g = −5·σ(−16) = −5.63e-7`
(measured bits `b5174000`), never at `−0.0`. The effect on what the recurrence
consumes: `ge = exp(−5.63e-7) = 0.99999944` instead of `1.0` — about **5 fp32
ULP at 1.0**, in exactly the regime where the model wants *no* decay. The leg
therefore requires *negative and ≤ the rail floor* (a positive value would be a
real bug) and **reports the floor**, rather than either failing forever or
quietly accepting anything.

The larger precision term is the **full bf16 path**, measured per output by the
TB (relative where `|golden| ≥ 1e-3`):

| output | consumed by | worst abs | worst rel |
|---|---|---|---|
| `g = −5·σ(t)` | (intermediate) | 1.24e-2 | 3.23 % (in the steep σ transition, where `|g|` is small) |
| **`ge = exp(g)`** | **the recurrence's decay** | 2.50e-3 | **1.24 %** |
| **`beta = σ(b)`** | **the recurrence's write gate** | 2.02e-3 | **1.34 %** |

The generator separately attributes **3.66e-3** of `ge`'s relative error to
rounding the sigmoid *argument* to bf16 alone; the polynomial and the bf16
*output* rounding supply the rest. All of it traces to one cause: **this repo's
only sigmoid is bf16**, and the KDA gate path is fp32 in the reference.

A ~1.2–1.3 % error on *both* gates the recurrence consumes — does it compound?
`tools/kda_gate_compound_study.py` answers that on the reference recurrence
itself (no RTL): T = 2048 tokens at the real DK = DV = 128, exact gates vs the
same tokens with the gates perturbed by the measured error.

| perturbation | `out` rel err, T=1 | T=16 | T=256 | T=2048 | state-norm ratio @2048 |
|---|---|---|---|---|---|
| saturation floor only (ge 0.99999944 where 1.0 wanted) | 0 | 3.4e-7 | 4.3e-7 | 5.3e-7 | 1.000000 |
| random ±1.24 % / ±1.34 % per element per token | 4.1e-3 | 9.5e-3 | 8.9e-3 | **7.8e-3** | 1.0003 |
| systematic +1.24 % / +1.34 % every token | 1.3e-2 | 2.0e-2 | 2.6e-2 | **2.9e-2** | 1.019 |

**Reading it.** The recurrence is contractive (decay < 1), so the per-token gate
error does **not** accumulate without bound — it settles at a steady state set by
the decay horizon. The saturation floor is a non-issue (5e-7). Uncorrelated bf16
error settles at ~0.8–1.0 % on `out`, about the per-token input error.
Systematic bias — the plausible structure if `glm_act`'s polynomial is biased —
settles at ~2.9 %, about 2.3× the input error. That rules out catastrophic
compounding and prices the bf16 gate path: **1–3 % output error per KDA layer,
across 34 layers.** Whether *that* is acceptable is a model-quality question
(perplexity-level), not one the recurrence math settles — and the study uses
random q/k/v/gates, not the model's activations, so it is decision evidence, not
proof. The decision it informs: build an fp32 sigmoid, or accept 1–3 % per layer.
The TB's bound (rel 0.03 + abs 0.002) is the measured envelope with headroom, so
a regression shows as a number moving.

### 4.3f KDA output norm step — DONE

`src/kda_onorm_step.v`: `Glm5NextTextRMSNormGated` on the recurrence output,
per head over `DV`:

```
y[h][i] = weight[i] · ( x[h][i] · rsqrt( mean_i x[h]² + ε ) ) · σ(gate[h][i])
out     = bf16(y)                                     one rounding, at the end
```

`x` is already bf16-valued at o_norm entry in the reference (the recurrence
returns `.to(bf16)`), so a bf16 `x` port is faithful, not a shortcut. `ε = 1e-5`
= `rmsnorm_unit`'s default; `weight = ssm_norm.weight [128]`.

**Composition, and why the gate is folded into gamma.** The proven `rmsnorm_unit`
is bf16-in / fp32-reduce / bf16-out and applies gamma *inside* its normalize
pass. Multiplying `σ(gate)` onto its bf16 *output* would round twice where the
reference rounds once. So the module computes `gamma_eff[i] = bf16(weight[i] ·
σ(gate[h][i]))` per head and streams `(x, gamma_eff)` through the unit
**unmodified** — one final rounding, faithful — at the cost of rounding
`gamma_eff` to bf16 where the reference keeps `weight·σ` in fp32. Measured:
that rounding alone costs **3.87e-3** relative worst-case. The generator's
self-test proves the fold is an exact identity (300/300) and that the plausible
misreading — gating `x` *before* the norm, which changes the variance — never
coincides, so the injection is live.

| leg (`make kda-onorm`) | claim |
|---|---|
| generator self-test (300) | gate folds into gamma exactly; gate-first never matches |
| `kda_onorm` (1152) | **tolerance**: rel 0.03 + abs 0.004 — measured worst rel **1.89 %** (`|golden| ≥ 1e-3`), worst abs 1.56e-2; three approximation sources (Quake rsqrt, bf16 polynomial σ, bf16 gamma_eff) |
| `-DINJ_ONORM_GATE_FIRST` must fail | "norm of the gated input" — measured 1097/1152 mismatches with it |

The handshake mirrors `glm_decoder_block_q4k`'s idiom: the unit pulls
(`in_req`/`g_req`), the producer answers with a registered one-cycle-later
`valid` and a beat counter. `LANES = 1` at the slice. Verilator-clean (two
`WIDTHEXPAND`s on the narrow head/beat counters fixed with explicit
zero-extension wires, not a lint pragma).

**With this, every non-GEMV unit of a KDA layer exists and is gated** — recurrence,
conv, gates, output norm. The layer wrapper is now a sequencing problem over the
existing `glm_matmul_q4k` engine plus these four, not new numerics.

### 4.3g KDA layer wrapper — SCOPED; and the "blocker" was my own framing error

With the four non-GEMV units gated, the wrapper is a composition problem. Reading
how the existing attention module is driven turned it into three sub-problems:

1. **Sequencing (pattern exists).** `mla_attn_q4k` drives ONE shared
   `glm_matmul_q4k` through an explicit FSM, selecting each projection with
   `w_sel` on the external weight-request stream. `swiglu_expert_q4k` is the same
   pattern in 403 lines rather than 2041, and is the better template — KDA has no
   RoPE, no KV paging, no DSA indexer. A KDA layer is **nine** GEMVs —
   `attn_{q,k,v,output}`, `ssm_{beta,f_a,f_b,g_a,g_b}` [scan, all Q8_0] — with the
   four units interleaved.
2. ~~**Q8_0 weight plumbing — a gap.**~~ **CORRECTED 2026-09-06: there is no gap.**
   The claim was that the decoder block's fan-out to the attention slot carries
   only Q4_K header buses, "and that fan-out has to widen, and it lives in
   `glm_decoder_block_q4k`, which sits under pinned netlist baselines". Both halves
   are true and the conclusion still does not follow: it assumed GLM-5.3-Flash
   would **reuse** that block. It will not — §4.3m already established the sibling
   pattern (`glm53f_hc_block` beside `glm_decoder_block_q4k`, itself beside
   `glm_decoder_block.v`), and a sibling declares its own port widths. The GEMV
   engine already takes `w_type` (2 = Q8_0), `w_hp` (code in [7:0]) and `w_q8_d`
   (fp16 d per 32-block) as inputs, and `weight_loader_q4k` already emits them.
   **Nothing shared has to change.**
3. ~~**Recurrent-state ownership — a gap.**~~ **Same correction.** The point that
   KDA is not a drop-in for `mla_attn_q4k` stands — that slot's contract carries
   KV-cache ports KDA cannot use and nothing that threads a `[H, DK, DV]` state
   plus a `[3·H·DK, K−1]` conv history. But a GLM-5.3-Flash layer module is not
   trying to fit that slot; it declares the state ports it needs, exactly as
   `mhc_block_site` declares the four residual streams. What remains real is the
   BRAM/DDR **residency** decision at 4.19 MB/layer — a memory question, not a
   plumbing one.

**So the KDA wrapper is not blocked on a decision about shared RTL; it is work.**
I had recorded it as "two gaps in shared, baseline-pinned RTL" and repeated that
until building `glm53f_hc_block` made the sibling route obvious in the other
direction. The cost of the error was direction, not rework: nothing was built
against the wrong assumption.

**Composition, pinned from the reference rather than inferred.**
`causal_conv_step` is ONE depthwise conv over the **concatenated** q,k,v
(`C = 3·qkv_dim`) with SiLU on its output; `forget_gate` takes
`decay = exp(A_log)` from `ssm_a` [64], which is a per-layer constant and so is
legitimately precomputed (`kda_gate_step` already expects it that way);
`g = f_b(f_a(h)) + dt_bias` is `[H, DK] = [64, 128]`; and `beta = σ(ssm_beta @ h)`
is `[64]`, one per head. Dimensions check against the census: 64 heads × 128
head_dim = 8192 = the q/k/v projection width.

### 4.3h DSA indexer — SCOPED, not started (and re-scoped up)

The ledger had this as "indexer compressor — small". Reading the reference
(`Glm5NextTextIndexer`) says otherwise. Per DSA layer, per query token:

```
q[h]        = wq_b(q_resid)                       32 heads × 128      (q_lora 1536 → 4096)
k           = LayerNorm(wk(h_tok), eps=1e-6)      128, ONE k shared by all heads; has a BIAS
gate[tok]   = h_tok @ compress_gate^T             128
-- k-pool compression, pools of 4 consecutive keys from the first valid key --
logit[p][j][c] = gate[key_j][c] + ape[j][c]       j = position in pool (0..3), c = channel
prob           = softmax_j(logit)                 per CHANNEL, over the 4 positions; -inf for invalid keys
pool_key[p][c] = Σ_j prob[p][j][c] · k[key_j][c]
-- scoring over POOLS, not tokens --
score[h][p]    = relu( (q[h] · pool_key[p]) · 128^-0.5 )
w[h]           = weights_proj(h_tok)[h] · 32^-0.5
index_score[p] = Σ_h w[h] · score[h][p]           masked to -inf where the pool's last key is not visible
selected       = top-(2048/4 = 512) pools  →  each expands to its 4 token indices  →  2048
tail           = the current incomplete pool's ≤3 raw indices, appended (index_kpool_always_select_tail)
```

**What GLM-5.2's `dsa_indexer.v` has and does not have.** It scores ONE index
vector against every token's index vector and keeps the top 2048, with the
IndexShare freq-4 / offset-3 reuse driven from the block. It has no pooling, no
head dimension, no ReLU, no head-weight combination, no LayerNorm, no
pool→token expansion, no tail. GLM-5.3-Flash also has **no `index_topk_freq`**
(the sharing that exists, `index_share_for_mtp_iteration`, is across MTP
iterations, not layers), so the indexer runs on every DSA layer every token —
which is exactly what `tools/glm53_flash_memory_budget.py` already assumes for
its indexer cost (`S/4` pooled candidates × 32 heads × 128).

**What carries over:** `topk_select` (top-512 over pools instead of top-2048 over
tokens — a *smaller* select), the fp32 MAC pipes for the dot products, and the
pull-handshake style. **What is new:** the compressor (a per-channel 4-way
softmax — `glm_softmax` is a vector softmax over a length, so this is a reshaped
use or a small new unit), a **LayerNorm-with-bias unit** (only `rmsnorm_unit`
exists), the 32-head ReLU + weighted head sum, and the pool→token expansion +
tail append (index arithmetic, not numerics).

**Executable reference: DONE, self-tested, not yet in the ladder.**
`tools/dsa_indexer_ref.py` transcribes the decode step — LayerNorm-with-bias
`k_norm`, the gate projection, k-pool compression (per-channel softmax over the
4 positions, pooling from the first *valid* key), the 32-head ReLU scores with
the `D^-0.5` / `H^-0.5` scales, the head-weighted sum, top-(TOPK/KPOOL) pool
selection, ×KPOOL expansion, and the incomplete-tail append — and pins six traps
a plausible transcription gets wrong. Its self-test checks the invariants the
math must satisfy (probabilities sum to 1 per channel; every pool key lies in the
convex hull of its keys; pooling starts at the first valid key; selected pools are
exactly the top-k over valid pools; no duplicate/invalid/out-of-range index; the
tail is appended; a loop-form compressor agrees with the vectorised one). It is
deliberately **not** wired into `make release-gate` yet: a new pinned gate needs a
full ~7 h ladder run to re-pin, so it is batched with the next RTL change. Until
then its status is "reference exists, self-tested locally" — the same standing
`tools/glm53_flash_ref.py` had before `make glm53f-ref` existed.

### 4.3i mHC precision study — DONE (the "fixed-point study before RTL")

`tools/mhc_precision_study.py`, on the transcribed reference with random weights
(decision evidence about precision budgets, not a claim about the model):

| question | result |
|---|---|
| **Q1** how doubly stochastic is `comb`, fp32 | over **200 draws** of (`fn`, `base`, streams), residual (max `|row/col sum − 1|`) @20 iters = **median 1.0e-6** (the `eps` floor), p90 5.9e-6, **worst 4.8e-4**; @40 the worst is still 2.3e-6. Swept over the comb-logit spread, the worst residual @20 grows 1.1e-6 (std 0.7) → 5.9e-3 (std 1.4) → 6.7e-2 (std 11) |
| **Q2** the map in bf16 | `comb` moves by up to **2.8e-3** (entries ~0.25), and a bf16 `comb`'s residual is **2.8e-3 vs 3.0e-5 fp32 on the same inputs — 93×** (an earlier version of this row divided by fp32's *median* and claimed three orders; 93× is the paired number) |
| **Q3** 45-block compounding, bf16 map vs fp32 | RMS rel divergence **1.3–3.1e-3**, settling (comb is stochastic, so the mix is contractive) |
| **Q4** dynamic range | `pre ∈ (2.0e-4, 0.99997)`, `post ∈ [0, 1.99999]`, smallest `comb` entry **1.8e-8** |
| **Q5** no divider: `x·recip(y)` for `x/y` | Sinkhorn's 20 iterations are **40 divisions** and this repo has no fp32 divide. With `glm_fp_recip.vh`'s worst case (+1 ULP on every one of the 40), `comb` moves by **2.7e-7** and the residual is **unchanged** (5.440e-3 both ways). The substitution is safe — mHC needs no divider |

**Conclusion — not fixed-point.** The eps-floored normalisations put `comb`
entries down to ~5e-8: that needs a floating-point exponent, and a naive
fixed-point mantissa cannot carry it. The map (`fn` GEMV → `pre/post/comb`
→ Sinkhorn) should be **fp32**, and the stream mix `comb @ streams` is a 4×4 fp32
matmul over `D = 4096` per block — twice per block (attention and FFN sites).
**Correction, 2026-09-04 — why 20 iterations, restated.** The first version of
this row read "20 is past the plateau — correct and sufficient", from a study
that measured **one** random draw. 1.0e-6 is that draw's residual and is the
population *median*, but not a bound: the tail reaches 4.8e-4 at 20 iterations
and grows with the comb-logit spread, whose trained value is not published. So
20 is **not** a convergence criterion that happens to be met — it is a published
constant (`hc_sinkhorn_iters = 20`), and the matrix the model uses is whatever 20
iterations produce, doubly stochastic or not. **Consequence for RTL:** run
exactly 20 on a fixed schedule and never early-exit on a convergence test — an
early exit would be both faster and less faithful. That also makes the mHC
latency data-independent, which is the easier thing to build.

**Third sighting of the bf16-sigmoid limit.** `pre = σ(·) + 1e-6` and `post =
2σ(·)` are sigmoids whose *fine structure near saturation* matters (`pre` must
resolve `1 + 1e-6`; bf16 cannot distinguish it from 1). After the KDA forget gate
(§4.3e) and the o_norm gamma (§4.3f), this is the third GLM-5.3-Flash path where
the repo's only sigmoid — bf16 `glm_act` — is the binding precision limit. The
decision "build an fp32 sigmoid" is no longer about one unit.

### 4.3j fp32 sigmoid — DONE, and the exp ceiling it uncovered

The bf16-sigmoid limit was sighted three times (§4.3e KDA gate, §4.3f o_norm
gamma, §4.3i mHC `pre`), and the mHC study made it blocking: that map **must** be
fp32, and `pre = σ + 1e-6` is not representable in bf16 at all. So mHC RTL cannot
start without an fp32 sigmoid. `src/fp32_sigmoid_pipe.v` is it.

The repo had **no fp32 divide**, so `src/glm_fp_recip.vh` adds a Newton
reciprocal (`r ← r(2 − yr)`, exponent-trick seed) in its own header — not in
`glm_fp.vh`, because every pinned netlist baseline depends on that file being
untouched. Measured on 4003 vectors over the range `1+exp(−x)` actually spans:

| Newton iters | not bit-exact | worst |
|---|---|---|
| 1 | 3995/4003 | 42804 ULP |
| 2 | 3523/4003 | 109 ULP |
| 3 | 1687/4003 | 2 ULP |
| **4** | 1251/4003 | **1 ULP** ← plateau |
| 5 | 1251/4003 | 1 ULP |

**What makes the unit worth building is saturation, not average accuracy.**
`fp32_exp_pipe` is FTZ and overflows to `+inf`, so `σ` reaches **exactly 1.0**
(x ≳ 17, once `1+e` rounds to 1.0) and **exactly 0.0** (x ≲ −88). Those are the
two things bf16 cannot do and the two the callers need: mHC needs `σ = 1.0` so
that `σ + 1e-6` differs from 1.0, and the KDA forget gate needs `σ = +0.0` so
that `−5.0·σ = −0.0`. Both are checked **bitwise**, each with a vacuity check.
The saturation-to-1.0 mux is not cosmetic: without it Newton on `y = 1.0` lands
1 ULP low (measured `0x3F7FFFFF` at x = 21.6), which would break the mHC caller
outright.

**The finding: `fp32_exp_pipe` is the ceiling, not the sigmoid.** Three spot
checks had suggested ~10 ULP. Swept over `x ∈ [−40, 40]`, the pipe is **1899 ULP
= 2.3e-4 relative** — two orders worse. The sigmoid built on it measures **790
ULP (≈9.4e-5)**, *better* than its own exp, because `σ = 1/(1+e)` compresses the
error (`dσ/σ = −(e/(1+e))·de/e`, and `e/(1+e) < 1`). `make fp-sigmoid` pins both
numbers as ceilings.

| leg (`make fp-sigmoid`) | claim |
|---|---|
| `fp32_exp_acc` (3000) | `fp32_exp_pipe` vs correctly-rounded exp: worst **1899 ULP**, ceiling 4096 |
| `fp32_sigmoid` (1087) | 130 bitwise σ = 1.0, 34 bitwise σ = 0.0, 6 subnormal goldens correctly FTZ-flushed, 917 normal points ≤ **790 ULP** (ceiling 1024) |

**What this changes for the callers, and what it does not.** Against the bf16
path's ~1.2e-2 this is ~130×. But it is *not* "fp32-exact": a 2.3e-4 exp inside
the mHC softmax sets `comb`'s accuracy floor well above the `eps`-floor residual
of §4.3i's median draw. **Improving mHC further
means improving `fp32_exp_pipe`'s polynomial**, not the sigmoid wrapper — a
separate, now-quantified piece of work. The KDA gate and o_norm retrofits (their
FSMs are built around `glm_act`'s latency and bf16 ports) are the next
increment, not done here.

### 4.3k mHC RTL — DONE for the map, NOT for the residual path

The precision study (§4.3i) said fp32 and no fixed-point; Q5 there said no
divider is needed. Both mHC units are now built and gated.

**`src/mhc_sinkhorn.v` — `make mhc-sinkhorn`, 1088 checks.** The 4×4 projection:
one column normalise then `ITERS−1` (row, column) pairs = **39 passes**, every
normalise dividing by `sum + 1e-6`. 14 cycles per pass, so **548 cycles**, and the
TB pins `done` to that exact cycle — which makes `NPASS = 39` a structural check
independent of the numerics. Three must-fail injections: `INJ_SINK_SYMM` (40 symmetric passes),
`INJ_SINK_ROWFIRST`, `INJ_SINK_NOEPS`.

**`src/mhc_map_step.v` — `make mhc-map`, 800 checks.** `pre = σ(·)+ε`,
`post = 2σ(·)`, `comb = softmax(·)+ε` → Sinkhorn. Streams 8 sigmoids through one
`fp32_sigmoid_pipe` and 16 exps through one `fp32_exp_pipe`. **700 cycles, and the
TB requires every vector to take exactly that** — mHC has no convergence early
exit, so data-independent latency is a design claim worth testing. Four must-fail
injections: `INJ_MAP_POST_NO2`, `INJ_MAP_PRE_NOEPS`, `INJ_MAP_COMB_NOEPS`,
`INJ_MAP_SOFTMAX_NOMAX`.

| | measured |
|---|---|
| `pre` / `post` vs the float64 reference | worst **493 / 719 ULP**, bound 1024 |
| `comb` (softmax + 39 Sinkhorn passes) | worst **1681 ULP ≈ 2.0e-4**, bound 16384 |
| predicted worst-case envelope | pre/post 790 ULP, comb **9886 ULP ≈ 1.2e-3** |
| Sinkhorn alone, `x·recip(y)` vs true division | worst **18 ULP** (12 predicted; the rest is `fp32_add`) |

**Where the bounds come from, and why it matters.** They are not the DUT's own
output rounded up. The generator perturbs every exp by `fp32_exp_pipe`'s measured
2.3e-4 and every non-railed sigmoid by `fp32_sigmoid_pipe`'s measured 790 ULP, in
the worst-case direction, and pushes that through the renormalise and all 39
passes. So the test constrains the implementation instead of describing it, and
the actual 1681 ULP being 5.9× inside the 9886 envelope is a real result — the
adversarial model assumes uncorrelated exp errors, and a softmax row's errors come
from one polynomial and partly cancel in the ratio.

**A consequence worth stating plainly.** `comb`'s envelope is ~1.2e-3 against the
**2.8e-3** a bf16 map costs (§4.3i Q2). The fp32 map is better, but by ~2.4×, not
by the orders §4.3i's original phrasing implied — because the binding term is
`fp32_exp_pipe`'s polynomial, not the sigmoid wrapper and not the reciprocal.
**Improving mHC now means improving that polynomial**, and nothing else in this
subsystem will move the number.

**Two things this surfaced.**
* `fp32_sigmoid_pipe`'s exposed `LAT` said `LAT_EXP + 2 + RECIP_ITERS` = 52; the
  real valid-in→valid-out latency is **53** (`LAT_EXP + 3 + RECIP_ITERS`: exp,
  stage A, `RECIP_ITERS+1` Newton stages, output mux). It is documentation only —
  no logic reads it, and `make fp-sigmoid` is unchanged — but it is a parameter
  labelled "exposed for callers", and the first caller to schedule on it hit the
  off-by-one. Now measured rather than counted by eye.
* `INJ_SINK_PAIRWISE` is **deliberately not a must-fail injection.** At H = 4 a
  pairwise reduction moves the result ≤ 40 ULP while the reciprocal substitution
  already moves it ≤ 49, so the reduction order is below the noise floor of a
  divider-free datapath and no tolerance the DUT can meet would separate them. The
  RTL still reduces sequentially (it matches the reference and costs nothing), but
  that is not a gated claim here — unlike KDA, where the reduction is over 128+
  terms and the order is decisive. A must-fail entry that cannot fail is worse
  than none.
* `INJ_MAP_SOFTMAX_NOMAX` needed the corpus fixed before it fired. The softmax's
  max subtraction is invariant in exact arithmetic and only a rounding difference
  at ordinary logit spreads — measured, it does **not** fail on such a corpus. Past
  |logit| ≈ 88 `fp32_exp_pipe` overflows and the shift stops being cosmetic, so
  every 8th vector is now a wide-logit saturation case. That case also drives the
  sigmoids onto their exact-0/exact-1 rails, which the clean run passes.

**What is still missing, and why `GLM53F_HC_RTL_PRESENT` stays undefined.** The
map is the numerically hard part; the *residual path* the define names is not
built: the unweighted RMSNorm over `H·D = 16384`, the `[(2+H)·H, H·D]` `fn` GEMV
that produces `mixed`, the collapse `Σ_h pre[h]·streams[h]`, the mix
`comb @ streams + post ⊗ sublayer_out`, and storage for **four** parallel D-wide
streams per block instead of one residual. Defining the flag now would be exactly
the overclaim the guard exists to catch, so `configs/full_glm53_flash.vh` carries
an HC STATUS note instead.

**Cost, and whether it is hideable.** 700 cycles per invocation, twice per block
over 45 blocks = **63,000 cycles/token** (49,320 of it Sinkhorn) = **63 µs at
1 GHz**. Against the token times the memory tiers imply (§5's 14.795 GB/token
denominator):

| tier | tok/s | token | mHC share | block weight fetch | 2 invocations |
|---|---|---|---|---|---|
| LPDDR5X ×16, 1.10 TB/s | 74 | 13.51 ms | 0.47 % | 294 µs | 1.4 µs |
| HBF ×2, 3.20 TB/s | 216 | 4.63 ms | 1.36 % | 101 µs | 1.4 µs |
| HBM3E ×6, 7.20 TB/s | 487 | 2.05 ms | 3.07 % | 45 µs | 1.4 µs |
| HBM4 ×4, 8.0 TB/s | 567 | 1.76 ms | 3.57 % | 38 µs | 1.4 µs |

**It is hideable, and the tightest case is the *fastest* memory.** A block's
weights do not depend on that block's residual streams, so the map can run
underneath the weight fetch. The binding comparison is therefore 1.4 µs of mHC
against the *shortest* per-block fetch — 38 µs at the HBM4 tier, so 3.7 % — not
against the longest. **But it is only hideable if the scheduler prefetches block
N's weights while block N's map runs**; serialised, mHC costs the full 0.5–3.6 %
of every token, which is a real number at the HBM tiers and a scheduling
requirement worth writing down rather than discovering later.

The H lanes run in parallel because H is a small fixed constant; serialising them
is ¼ the adders and ~4× the cycles if area ever binds — which the table says there
is room for at the LPDDR5X tiers and not much at HBM4.

### 4.3l mHC residual path — DONE in RTL; nothing instantiates it yet

§4.3k built the map. This is the path around it, and it is what makes a block
carry **four** residual streams instead of one.

**The structure, settled from the census rather than inferred.** `attn_norm[4096]`
and `ffn_norm[4096]` exist on all 46 blocks *alongside* the `hc_*` tensors, so mHC
**wraps** a sublayer and does not replace its norm:

```
flat      = unweighted_rmsnorm(streams.flatten())     [16384]  -- map input only
mixed     = hc_{attn,ffn}_fn @ flat                   [24],  Q8_0 weights
pre,post,comb = map(mixed, base, scale)
collapsed = Σ_h pre[h]·streams[h]                     [D],  NOT normalised
sub_out   = SUBLAYER( attn_norm(collapsed) )                  -- contract unchanged
streams'  = comb @ streams + post ⊗ sub_out           [4,D]
```

The sublayer still sees `[D]` in and `[D]` out. **That is why hyper-connections and
the KDA layer wrapper do not block each other** — a fact worth having, because the
earlier scoping assumed they might.

| unit | gate | checked against |
|---|---|---|
| `mhc_fn_gemv` | `make mhc-gemv`, 300 + 51 | bit-exact emulation, worst **11 ULP** |
| `mhc_stream_ops` | `make mhc-ops`, 15392 + 244 | the pinned `hc_collapse`/`hc_mix`, worst **0 / 4 ULP** |
| `mhc_block_site` | `make mhc-site`, 2568 + 39 | the composed spec, worst **7.7e-5 / 2.5e-4** abs |

**Five measurements decided this design before any RTL was written.**

| question | measured | decision |
|---|---|---|
| `hc_*_fn` dtype | **Q8_0** `[16384,24]` | not F32 — the F32 bucket's size made F32 a tempting, wrong guess |
| streams in bf16? | 5.9e-3 RMS after 90 sites | **fp32** — the streams are ONE 64 KB buffer, so bf16 saves nothing |
| `fn` GEMV on the existing GEMM? | bf16 activations cost **2.9e-3 – 6.0e-3** | **no** — ~40× the map's bound, worse than the bf16 map fp32 replaced |
| reduction order at K = 16384 | 1.05e-4 on `mixed`, **6e-6** propagated | not decisive; the reference is left on BLAS |
| fold `rms` past the GEMV | 7.4e-5 on `mixed`, **7e-6** propagated | **yes** — removes a pass and a 64 KB `flat` buffer |

**A new unit the earlier scoping missed.** `glm_matmul_q4k` is bf16-in/bf16-out, so
the `fn` GEMV needed its own fp32-activation × Q8_0-weight engine. It is a small
dedicated unit, deliberately not a widening of `glm_matmul_q4k`: that GEMM is
netlist-pinned and used by every top, and widening it for a 24-output GEMV would
re-pin baselines repo-wide for nothing. This is the **fourth** time the repo's
bf16 convention has collided with GLM-5.3-Flash's precision needs — after the
sigmoid, the map, and the residual streams — which is a pattern, not three
coincidences.

**Two metric traps, both of which produced a wrong number first.**
* `comb` is doubly stochastic, so the mix's four terms nearly cancel (this corpus
  reaches `Σ|terms| / |result|` = **1239×**). A plain relative error divides by the
  cancelled result and reported **1.8e-2** for what is a 1-ULP difference; against
  the output RMS it is **5.0e-7**. The site's bound is therefore **absolute** — a
  ULP bound derived from the same envelope comes out at 1.4e6 ULP, ~17 % at these
  magnitudes, which gates nothing.
* numpy's `comb @ streams` does **not** reduce sequentially — 300/300 differing.
  `tools/glm53_flash_ref.py` now pins the order in `hc_mix`, and `make glm53f-ref`
  (15 → **21** checks) asserts that numpy does **not** match, so the pin stays live.

**Also fixed here:** the `mhc_stream_ops` TB claimed the gate was bitwise "because
the reference performs the same fp32 adds in the same order". Same order is not
the same adder — `fp32_add` is 1 ULP low on ~0.04 % of pairs, and under the mix's
cancellation that surfaces as 4 ULP. Collapse *is* bitwise (0 ULP, `pre[h] > 0`,
no systematic cancellation); mix is not.

**`GLM53F_HC_RTL_PRESENT` still stays undefined — and now for exactly one reason.**
The machine is built and gated; **nothing instantiates it.**
`glm_decoder_block_q4k.v` still computes `h = x + attn(rmsnorm(x))` — one residual,
added. The define flips when the decoder block instantiates two `mhc_block_site`
per block. That is the entire remaining gap, it is decoder-block work rather than
mHC work, and the sublayer contract does not change.

### 4.3m Block residual skeleton — DONE, and the first guard condition closes

`src/glm53f_hc_block.v` — `make hc-block`, 2310 + 20 checks, 4 must-fail
injections. Two mHC sites (attention, FFN) wrapped around two sublayers it does
not own. This is the thing `glm_decoder_block_q4k.v` cannot express: that block
computes `h = x + attn(rmsnorm(x))` — **one residual, added**.

**A sibling, not an edit.** The repo already keeps `glm_decoder_block.v` (bf16)
and `glm_decoder_block_q4k.v` (Q4_K) as siblings; this is the GLM-5.3-Flash one.
Editing the Q4_K block would re-pin every netlist baseline depending on it for a
change no GLM-5.2 build wants.

**One site instance, run twice.** `mhc_block_site` owns the streams, so muxing the
weights and running it twice keeps **one** `[H,D]` buffer per block rather than two
plus a copy. The second pass must see what the first wrote — which is exactly what
`INJ_HCB_STALE_STREAMS` checks.

**What this gate is for.** `mhc-site` already pins one site's numerics; what a
block adds is *wiring*, so all four injections target that: per-site weights
(`SAME_WEIGHTS`), per-site learned norm (`NORM_SWAP`), the norm being applied at
all (`SKIP_NORM`), and the stream threading (`STALE_STREAMS`). `rmsnorm_unit` is
modelled bit for bit in the generator (LANES=1: bf16 in, sequential fp32 sumsq,
`mean·1/LEN`, `+eps`, the same Quake rsqrt, `bf16(x·inv·γ)`), leaving
`mhc_map_step`'s polynomial exp as the only non-bitwise term in the whole chain.
Measured, **bf16 rounding absorbs it entirely — worst normed error 0.0**, i.e. the
normalised sublayer input is bitwise; the mixed streams land at 7.0e-5.

**Where fp32 stops — and the right lesson from four bf16 collisions.** mHC's
gating math is fp32 because its ε-floored maps demand it. The *sublayer* path is
bf16, exactly like every other activation in this repo, and that is faithful
rather than a concession: `collapsed` is converted to bf16 for the block's own
RMSNorm, the sublayer works in bf16 throughout, and its output widens back to fp32
only to re-enter the mix. So all four collisions (§4.3j sigmoid, §4.3k map, §4.3l
streams and GEMV activations) were **inside mHC's gating**, not in the main
activation path. "GLM-5.3-Flash needs fp32 activations" would be the wrong
generalisation.

**`GLM53F_HC_RTL_PRESENT` is now DEFINED — the first of the three to close.** The
residual path is built end to end at block level and gated by six targets with 21
must-fail injections. What the define does **not** claim: `glm53f_hc_block` reaches
its sublayers through a handshake and does not instantiate them, so composing it
with `mla_attn_q4k` and the MoE/dense FFN into a real decoder layer is assembly
work that remains open — and for 34 of 45 layers the sublayer *is* the KDA machine
that `GLM53F_KDA_RTL_PRESENT` gates. The whole-model top therefore stays poisoned
by the other two conditions.

The guard grew a case for this (8 → **10** checks): the top must elaborate with
only `KDA` and `Q5K` on the command line, which is true only if `HC` really comes
from the header. The header's `` `define `` is wrapped in `` `ifndef `` — not
decoration, since the guard drives these from the command line to test both
directions, and an unconditional define would make a must-fail case fail for a
*redefinition error* rather than for the guard, i.e. pass for the wrong reason.

### 4.3n KDA layer — DONE as a unit; the guard does NOT flip, and why

`src/glm53f_kda_layer.v` — `make kda-layer`, 968 + 84 checks, 4 must-fail
injections. One decode step of Kimi Delta Attention: nine projections sequenced,
ONE depthwise conv over the **concatenation** of q,k,v with SiLU, the forget gate,
the delta-rule recurrence and the gated output norm. It owns the two pieces of
state that make KDA not a drop-in for `mla_attn_q4k`: the `[H, DK, DV]` recurrence
(fixed size — **not** a growing KV cache, which is why only 11 of 45 layers page
KV) and the `[3·H·DK, K−1]` conv history.

**Why `GLM53F_KDA_RTL_PRESENT` stays undefined even though the layer passes.** The
nine GEMVs arrive through a `proj_req`/`proj_sel` handshake, answered
behaviourally by the TB. That is *not* the same situation as `glm53f_hc_block`,
whose sublayers are genuinely separate machines: these projections are the
layer's **own weights**, and a layer that cannot fetch them is incomplete.
Driving `glm_matmul_q4k` directly is the remaining step and it is mechanical —
`w_type = 2`, code on `w_hp`, fp16 `d` on `w_q8_d`, all already engine inputs and
all already emitted by `weight_loader_q4k`, with `swiglu_expert_q4k` (403 lines)
as the template rather than `mla_attn_q4k` (2041).

**Two bugs this surfaced, one of them not mine.**

*`kda_recur`'s accuracy contract was wrong.* Its header said "EXACT=0: the module
computes l2norm and exp itself", and its port comment called `g_in` "log-decay
(EXACT=0)". The code does neither — a note inside `S_PREP` says plainly that
"EXACT=0 expects the caller to still supply exp(g): a Horner exp belongs in
fp32_exp_pipe, not inlined here". **`g_in` is `exp(g)` on both legs.** The stale
wording sent this layer's first build in with the raw log-decay, which multiplies
the state by `g` instead of `exp(g)` — 374/968 failures. Corrected in
`src/kda_recur.v` (comments only, netlist-neutral; `make kda` unchanged at
5184+5184), with a pointer telling callers to pass `kda_gate_step`'s `ge_out`.

*`INV_SQRT_DK` does not track `DK`.* `kda_recur` defaults it to `1/√8`; this slice
is `DK = 4`, which needs `1/√4`. It is a parameter rather than a computation
because `1/√128` is not exactly representable and deriving it from the approximate
`fp32_rsqrt` would put an approximation inside the bit-exact leg (§4.3c). **The
failure mode is worth remembering: `q` feeds `out` but not the state update, so
the recurrence's STATE still matched the golden while its OUTPUT was off by
exactly `√(8/DK)` = 1.414.** A checker that only compared the state would have
passed a wrong layer.

**Composition pinned from the reference, not inferred.** `causal_conv_step` is
ONE conv over `C = 3·qkv_dim`; `decay = exp(A_log)` from `ssm_a[64]` is a
per-layer constant, so precomputing it is faithful and `kda_gate_step` already
expects it that way; `g = f_b(f_a(h)) + dt_bias` is `[H, DK]`; `beta` is `[H]`.
The recurrence's output is **bf16-valued** at o_norm entry in the reference, which
is why `kda_onorm_step` takes a bf16 `x` port — the generator rounds there too,
and omitting that made the composed golden disagree on `y` alone while state and
history matched.

**Bounds are composed from the units' own published numbers**, not read off this
DUT: rel 6 % because §4.3e measures a KDA layer at 1–3 %, abs 0.03 because
`kda_onorm_step` is gated at abs 0.004 and `o_proj` sums `H·DV` of them. The first
attempt used abs 0.01 and passed with 0.6 % margin — that is luck, not a bound.

## 4.4 The executable specification (what `make glm53f-ref` pins)

Writing RTL for KDA / mHC / clamped SwiGLU from `config.json` alone would be
guessing: the config publishes `hc_sinkhorn_iters` and `swiglu_limit`, not the
**order of operations**. `tools/glm53_flash_ref.py` is that order, transcribed
from the reference implementation that `config.json`'s `transformers_version`
pins, with each trap named where a plausible guess diverges:

| trap | the wrong-but-plausible version |
|---|---|
| SwiGLU clamp is **asymmetric** | clamping both tensors symmetrically |
| FLA `l2norm` puts eps **inside** the sqrt | `x / max(norm, eps)`, or `F.normalize` |
| only `q` gets the `1/sqrt(Dk)` scale, **after** the l2norm | scaling `k` too, or scaling before normalising |
| KDA decays the state **before** reading `kv` | reading `kv` from the undecayed state |
| the delta rule writes `(v − kv)·beta`, not `v` | a plain outer-product write |
| forget gate uses the `lower_bound·sigmoid` branch | implementing the softplus branch, which is dead code at `lower_bound = −5.0` |
| mHC Sinkhorn is one column pass **then** `iters−1` (row, col) pairs | `iters` symmetric passes |
| mHC `post` is `2·sigmoid` (range `[0,2]`) | a plain sigmoid, halving the sublayer |
| `forget_gate` can emit **−0.0** (fp32 sigmoid saturates to 0.0) | emitting `+0.0`, which a bitwise gate catches |

**What this is not.** It is a faithful transcription whose self-test checks
internal consistency and the invariants the math must satisfy (delta rule
returns `v` at `beta=1`, decay shrinks the state, `comb` comes out doubly
stochastic, `post ∈ [0,2]`). **It has not been run against the real
checkpoint's activations**, so it is a specification, not a proof — the same
status the Laguna port's attention machine has.

## 5. Why the memory profile is not GLM-5.2's

Worth stating because it is the most likely thing to be quietly assumed wrong:
**only 11 of 45 blocks hold a growing KV cache.** The 34 KDA blocks carry a
fixed-size recurrent state instead — it does not grow with context length. A
1M-context memory budget derived from GLM-5.2's "every layer pages KV" model
would be badly wrong for this model, in the favourable direction.

That also softens the S_MAX / SWIN caveat (task B7): the attention-scratch
constraint now binds on 11 blocks, not 78.

**The budget is now quantified** (`tools/glm53_flash_memory_budget.py`, which
parses its model constants out of `configs/full_glm53_flash.vh` so it cannot
drift from the locked config):

| | GLM-5.2 | GLM-5.3-Flash |
|---|---|---|
| weights | 467 GB | 199.7 GB `[measured]` |
| KV @ 1M context | ~94 GB | **11.8 GB** `[derived]` |
| DSA indexer keys @ 1M | — | 0.37 GB `[derived]` |
| KDA recurrent state | — | 0.148 GB, **constant in context** |
| total resident @ 1M | ~561 GB | **212 GB** |

Per token the cached latent is `11 x 512 x 2 B = 11 KiB`, against GLM-5.2's
`78 x 576 x 2 B = 87.8 KiB` -- 11 of 45 layers rather than 78 of 78, and no
rotary tail because of NoPE (`attention.key_length` equals `kv_lora_rank`
exactly). A 1M-context budget carried over from GLM-5.2 is wrong for this model
by ~2.6x, in the favourable direction -- which is worth saying that way round,
because an error in the favourable direction still mis-sizes a board.

What that does to the hardware ladder -- including the trap that capacity fell
but the package/stack count must not, the ~8x-oversized rung-4 HBM tier, and the
all-HBM residency this newly makes reachable -- is in
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) §"GLM-5.3-Flash re-sizing".

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

The config header itself is gated, and so is every unit built on this branch:

```sh
make glm53f-config-guard      # 8/8:  4 cases x 2 tools
make glm53f-ref               # 15/15: the executable spec (4.4)
make fp-ieee                  # 10000: fp32_add's pinned 1-ULP non-conformance
make fp-sigmoid               # 3000 + 1087: the fp32 sigmoid and its exp ceiling
make kda kda-conv kda-gate kda-onorm     # the four non-GEMV KDA units
make mhc-sinkhorn             # 405 + 1088: the 39-pass projection (4.3k)
make mhc-map                  # 364 + 800:  pre / post / softmax + Sinkhorn (4.3k)
make mhc-gemv                 # 51 + 300:   fp32 acts x Q8_0 fn, RMS folded (4.3l)
make mhc-ops                  # 244 + 15392: collapse and mix, D-wide (4.3l)
make mhc-site                 # 39 + 2568:  one whole site, streams carried (4.3l)
make hc-block                 # 20 + 2310:  TWO sites per block, wiring (4.3m)
make kda-layer                # 84 + 968:   one whole KDA decode step (4.3n)
```

Each of those prints its own worst-case error, so a regression moves a number
rather than flipping a boolean. Between them they carry **29 must-fail
injections** — `INJ_KDA_NODECAY`, `INJ_CONV_FLIP`, `INJ_GATE_DECAY_AFTER`,
`INJ_ONORM_GATE_FIRST`, `INJ_SINK_{SYMM,ROWFIRST,NOEPS}`,
`INJ_MAP_{POST_NO2,PRE_NOEPS,COMB_NOEPS,SOFTMAX_NOMAX}`,
`INJ_OPS_{MIX_TRANSPOSE,MIX_NOPOST,COLLAPSE_NOPRE,POST_FIRST}`,
`INJ_GEMV_{Q8_NOSCALE,MEAN_SUM,NO_EPS}`,
`INJ_SITE_{IGNORE_SUB,NO_UPDATE,PRE_FOR_POST}`,
`INJ_HCB_{SAME_WEIGHTS,NORM_SWAP,SKIP_NORM,STALE_STREAMS}`,
`INJ_KDAL_{NO_STATE,CONV_NOHIST,QK_SWAP,GATE_ORDER}` — plus
`INJ_Q5K_NOMIN` in `make mixedtype` and the config guard's own 4 poisoned cases.
`INJ_SINK_PAIRWISE` exists but is deliberately **not** one of them (§4.3k explains
why a gate that cannot fail is worse than no gate).

**Cost of the full ladder, measured.** `make release-gate` on this machine took
**7 h 17 min** (2026-09-03; 54 targets, 102 pinned gates), strictly serial. The
long poles are pre-existing: the `PE_M=2` batched model sim alone ~90 min, then
`synth-glm` (whole-chip yosys), the netlist-equivalence checks and the SBY formal
targets. Everything this branch added (`glm53f-*`, `fp-ieee`, `kda*`) runs in
seconds. A Makefile audit for `make -j` safety found it is **not yet safe**: one
sim binary (`build/spec_depth_adapt_sim`) is written by two targets
(`spec-adapt`, `unittests`), and five vector generators are invoked by several
targets with fixed output paths (`q4k_matmul_gen.py` ×4, `glm_model_q4k_tb_gen.py`
×3, `swiglu_q4k_gen.py`, `route_trace.py`, `l3_image_pack.py` ×2 each) — run
concurrently, one target reads vectors generated for another: a false fail, or a
false pass. The fix is per-target output paths (Makefile only; the manifest
checker is already order-independent). Recorded as a finding; not applied, since
it changes how the headline gate executes and needs its own verification run.

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
