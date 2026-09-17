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

**Why this unit alone did not justify the define** (§4.3o then closed it). The
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

### 4.3o KDA weight streaming — the second guard condition closes

`src/glm53f_kda_gemv.v` + `src/glm53f_kda_attn.v` — `make kda-attn`, 726 + 228
checks, 3 must-fail injections. The nine Q8_0 projections are now **streamed off
`glm_matmul_q4k`** instead of handed to the layer, which is the difference
between a unit and a sublayer. `glm53f_kda_attn` is the module that goes where
`mla_attn_q4k` goes, for 34 of the 45 blocks.

**Q8_0 needed nothing new from the engine**, which is the concrete refutation of
§4.3g's "gap": `w_type = 2`, the code on `w_hp[7:0]` and the fp16 block scale on
`w_q8_d` were already `glm_matmul_q4k` inputs, and `weight_loader_q4k` already
emits them. The protocol was copied from `swiglu_expert_q4k` (403 lines) rather
than invented.

**The result that matters: the bound did not move.** `make kda-attn` runs the
*same* composed bounds as `make kda-layer` — rel 6 %, abs 0.03 — and the measured
worst is **identical at 7.8e-3**. Streaming the weights off real Q8_0 did not cost
accuracy, because the golden runs on the *dequantised* weights: the Q8_0 round
trip (worst 4.4e-3 on the weights) is part of the **input**, not of the error. Two
gates, two claims: `kda-layer` says the datapath is right, `kda-attn` says it is
still right when the weights arrive the way the system will actually deliver them.

**A contract change that made both gates honest.** The layer's `proj_out` port is
fp32-*typed* but now carries bf16-*valued* data, because `glm_matmul_q4k`'s
`c_out` is bf16 and so are the model's own linear layers. The generator and the
`kda-layer` stub were both changed to round there. Before that they disagreed —
the golden assumed bf16 while the stub returned fp32 — and the layer's worst error
read 1.19e-2 instead of 7.8e-3. Making the contract match reality *improved* the
number.

**An injection that had to be redesigned.** `INJ_KGV_GRP_STUCK` froze the output
group, which hangs the FSM: it does fail, but only by TB timeout, and it would
have cost the release gate ~3M simulated cycles on every run. Replaced with
`INJ_KGV_GRP_ALIAS`, which writes every group to slot 0 — wrong data, same
runtime. A must-fail injection should fail on the *answer*, not on the clock.

**`GLM53F_KDA_RTL_PRESENT` is now defined — the second of three.** The guard's
pinning case got stronger with it: the whole-model top must elaborate with only
`Q5K` on the command line, which is true only if **both** `HC` and `KDA` really
come from the header. Only the Q5_K loader path still poisons the top.

**What neither define claims.** Neither module instantiates a decoder layer.
Composing `glm53f_kda_attn` and the MLA/MoE sublayers inside `glm53f_hc_block` is
assembly, and the 4.19 MB/layer recurrent state still lives in registers rather
than BRAM/DDR — a residency decision, not plumbing.

### 4.3p Q5_K read path — the last guard condition closes, and the blocker was again a premise

`make q5k-loader`, 1610 checks, 1 must-fail injection. The packer now emits Q5_K
tiles (`ckpt_pack_q4k.pack_q5k_weight`) and the **real** `weight_loader_q4k`
streams them.

**The `$fatal` was guarding a premise that did not hold.** The loader refused a
Q5_K descriptor on the grounds that it "has no Q5_K tile geometry" and would
stream Q4_K geometry — "same widths, wrong bytes, no error". Reading both sides
showed a Q5_K tile needs **nothing new from the loader**:

* its header fields **are** Q4_K's (`d`, `dmin`, 12-byte packed scales), so the
  Q4_K *default* decode branch is already exactly right and `ns = nsblk·PE_N` is
  unchanged — the extra 32 bytes of a 176 B super-block are `qh`, and `qh` never
  reaches the RTL;
* its 5-bit code rides `mm_w_hp`, the same 16-bit-per-column lane Q6_K and Q8_0
  already use, **pre-assembled** by the packer as `nibble | 16·qh_bit`, which is
  what `glm_matmul_q4k`'s Q5_K arm reads.

So the missing half was the **packer**, not the loader — the third time on this
branch that something recorded as a blocker turned out to be a framing error
(§4.3g was the other). The `$fatal` is replaced by the reasoning, not just
deleted.

**Three independent checks, because "it round-trips" is not enough.**

| check | what it rules out |
|---|---|
| packer round-trip, Q5_K at **nb=2**, bit-exact | the tile is lossy — including the fifth bit, since `q5k_codes_to_qs_qh` rebuilds **both** `qs` and `qh` |
| dequant cross-check vs `q4k_ref.dequantize_block_q5_K` | the bytes survive but no longer *mean* the same thing |
| `q5k_loader_tb` at **nb=3** | the packer and the RTL disagree on layout — the file the packer writes is the file the loader reads, with expectations from the pre-pack source |

The injection packs the codes with the fifth bit **dropped** while leaving the
header and the expectations intact: a byte-plausible Q5_K image that is really
Q4_K data. It fails on the first beat (`0x1d` → `0x0d`). The corpus is checked to
use the fifth bit on ≥25 % of codes (measured 48.9 %) — otherwise the gate would
barely distinguish the two formats.

**Two bugs found on the way.**
* I overwrote `blk.0.ffn_down.weight` — Q4_K at **[8, 512], nb=2** — with the new
  Q5_K tensor, silently deleting the packer selftest's only multi-super-block
  Q4_K case. Caught by diffing the synthetic tensor list against `HEAD`.
* `pack_gguf` relocated per-tile bases under `if desc["type"] == "Q4_K"`, so Q5_K
  tiles pointed at the image's start instead of the tensor's. **It passed while
  the Q5_K tensor happened to be first in the image** (`off == 0`) and only broke
  once it was not — a hardcoded type check where a category was meant. Now
  `TILED_TYPES`.

**`GLM53F_Q5K_RTL_PRESENT` is defined. All three conditions are closed**, so the
whole-model top elaborates by default — which broke the guard's own must-fail
cases, since "no defines" is now a must-pass. Rather than let a gate that can no
longer fail stand, each machine gained a `GLM53F_NO_*` test escape and the guard
now forces them absent **one at a time**, requiring the top to go back to
un-elaboratable. Same 10 checks, and each condition is now individually
demonstrated to be load-bearing.

**What this does not mean.** The three conditions were always about whether the
three missing *machines* exist. They do. **The model is not assembled**: nothing
instantiates `glm53f_kda_attn` or `glm53f_hc_block` into a decoder layer, and
`glm_q4k_system` still drives the loader with a hardcoded single-tile descriptor
and no `desc_wtype` at all — a pre-existing gap that applies to Q4_K equally.
A passing elaboration is not a working model, and §4.2 tracks the difference.

### 4.3q Decoder block — a real sublayer inside the hyper-connection

`src/glm53f_decoder_block.v` — `make dec-block`, 676 + 20 checks, 2 must-fail
injections. The first module where a machine lives **inside** another: KDA
attention wired into the attention site of the two-site mHC block, with the
recurrence and the conv history advancing across it.

```
streams [4,D]
  -> attn site: collapse -> attn_norm -> glm53f_kda_attn -> mix
  -> FFN  site: collapse -> ffn_norm  -> <handshake>     -> mix
  -> streams'
```

**Why this was wiring rather than redesign.** The mHC sublayer contract never
changed: `attn_norm`/`ffn_norm` sit between `collapsed` and the sublayer on all 46
blocks [scan], so the sublayer still sees `[D]` bf16 in and `[D]` bf16 out — the
same shape `mla_attn_q4k` and the FFN already present. §4.3g's claim that
hyper-connections and the KDA wrapper were "one piece of work" came from the
premise that turned out to be wrong; they compose cleanly.

**The FFN site is still a handshake, and that is a finding rather than laziness.**
The census says GLM-5.3-Flash's dense FFN is **Q8_0** (`blk.N.ffn_{gate,up,down}`
[12288, 4096] ×3) and its MoE experts are a **mix** — `ffn_gate_exps` Q4_K×42 +
Q5_K×1, `ffn_down_exps` Q5_K×40 + Q6_K×3, shared expert Q8_0, router F32.
`swiglu_expert_q4k`'s `w_q` port is **4 bits per lane**, so it can carry none of
them. Hanging it off this block to make it look finished is exactly the
silent-wrong-weights failure the repo builds must-fail pairs against. A Q8_0
clamped SwiGLU is the next sibling.

**Scope, stated exactly.** This is blocks 0–2's attention half: those three are
the dense front (`N_DENSE = 3`) and they are KDA, since the first MLA block is 3.
For the other 31 KDA blocks the FFN is MoE; for the 11 MLA blocks the attention
site takes `mla_attn_q4k`, which is why `ATTN_KIND` exists as a parameter even
though only the KDA arm is built — and `ATTN_KIND = 1` `$fatal`s rather than
elaborating a block whose attention site is empty.

**Both injections target the composition, not the arithmetic** (the two machines
have their own gates): `INJ_DBLK_SITE_SWAP` exchanges which sublayer serves which
site, and `INJ_DBLK_NO_KDA` routes *both* sites to the FFN handshake so the
attention sublayer never runs. The second is the one worth having — the block
still completes and the streams still move, so nothing about the handshake would
reveal it.

### 4.3r Dense FFN, and a complete decoder layer for blocks 0–2

`src/glm53f_swiglu_q8.v` — `make swiglu-q8`, 204 + 15 checks — then wired into the
decoder block's FFN site, so `make dec-block` is now a **complete GLM-5.3-Flash
decoder layer** for blocks 0–2: KDA attention and a clamped SwiGLU, both inside
the two-site mHC block.

**Why a sibling and not `swiglu_expert_q4k`.** Its `w_q` port is **four bits per
lane** and the census says this FFN is Q8_0 (`blk.N.ffn_{gate,up,down}`
[12288, 4096] ×3). It cannot carry these weights — not "less accurately", at all.

**The finding: a composed block is not as tight as the sum of its parts.** Three
successive versions of the stream bound failed, each on the same element, and each
time the fix I reached for was wrong:

1. a flat `rel + abs` — failed where the DOWN reduction is large;
2. `+ post·(the FFN's own per-element tolerance)` — 24 → 2 failures;
3. `+ post·(the KDA sublayer's own bound)` — 2 → 1.

The remaining failure was not a missing term but a wrong *model*. Probing the DUT
showed the FFN's **input** already differed by 1.2 % (3 bf16 ULP on the normed
vector, from KDA's error upstream) while its **output** differed by 0.35 — a gain
of **~29×**, from silu and a 32-term DOWN reduction. Measured directly: perturbing
the attention sublayer's output by its own gated 0.03 moves a final stream by
**0.885**.

So the bound is an **envelope**: perturb the mHC map outputs by their gated ULP
bounds *and* the attention output by its gated abs bound, then re-run the whole
block. That is the same technique §4.3l used, and it is necessary here rather than
fastidious — adding the parts' bounds under-counts by the gain. Both injections
still fire hard (379 and 648 failing checks), so the looser bound did not cost the
gate anything.

**A symclamp injection that is deliberately not gated.** `INJ_SWQ8_SYMCLAMP`
exists but is **not** a must-fail: measured, the symmetric-gate reading moves the
result 0.25 against a 2.9 tolerance, because that tolerance is forced by
`glm_act`'s polynomial silu and no exact Python model of that polynomial exists. I
predicted it would not fire, then confirmed it. The asymmetry *is* gated on a
tighter slice by `make swiglu`'s own `INJ_SWIGLU_SYMCLAMP`. What this gate does
check is that the clamp is there at all — `INJ_SWQ8_NOCLAMP` fires by a wide
margin, because unclamped gates reach ±30 where silu is ~30 rather than ~10 — and
the generator asserts the corpus drives **both** bounds (98 upper, 59 lower).

**Still parameterised but not built**, and both `$fatal` rather than elaborating a
block with an empty site: `ATTN_KIND = 1` (MLA, 11 blocks) and `FFN_KIND = 1`
(MoE — `ffn_*_exps` is a Q4_K/Q5_K/Q6_K mix plus a router and a shared expert,
for the other 31 KDA blocks).

### 4.3s The weight type becomes a runtime input — `make swiglu-mt`

The MoE experts do not have *a* weight type. The census says
`ffn_{gate,up}_exps` is Q4_K ×42 and Q5_K ×1, and `ffn_down_exps` is Q5_K ×40 and
Q6_K ×3 — so the type varies per tensor **and** per block. `glm53f_swiglu_q8.v`
was therefore renamed `glm53f_swiglu_mt.v` and its type made a runtime input
(`wt_gate` / `wt_up` / `wt_down`) carrying the full loader bundle: Q4_K's code on
`w_q`, every other type's on `w_hp`, headers on `w_d`/`w_dmin`/`w_scales`,
`w_q6_sc` and `w_q8_d`. `make swiglu-q8` (204) and `make dec-block` (676) both
still pass unchanged, which is what says the generalisation is behaviour-preserving.

`make swiglu-mt` — **514 + 8 checks** — is a *separate* gate rather than more
cases in `swiglu-q8`, because Q4_K/Q5_K/Q6_K are 256-weight super-blocks: K must
be 256 here, while the MoE loop wants a small slice. One compromise slice would
have tested neither well. The two combos are the checkpoint's, not invented:
`(Q4_K, Q4_K, Q5_K)` and `(Q5_K, Q5_K, Q6_K)`.

**It found a latent defect on the real model.** The activation lanes were counted
with `reg [7:0] ai, ao` against `INTER[7:0]`. At the gated slice INTER = 32 that
is 32 and everything passes; at INTER = 256 it is **0**, `ai < 0` is never true,
the activation stage never starts, and the FSM hangs. The actual dense FFN is
INTER = 12288, which truncates to 0 as well — so this would have shipped, and
neither `swiglu-q8` (INTER = 32) nor `dec-block` could see it. Counters are now
sized `localparam AW = $clog2(INTER+1)`. Grepping the *class* rather than the
instance found one more (`TOPK[7:0]` in `glm53f_moe_router.v`), live only if
top-k could reach 256; it is 8, so that one is left alone rather than churning a
pinned gate.

**A second finding, in the generator.** Q6_K's 16 int8 scales were emitted as one
128-bit hex word. Hex is MSB-first, so scale 0 landed at bit offset 8×15 instead
of 8×0 — the order reversed, every Q6_K column decoded with the wrong per-16
scale, and the *sign* of the result flipped. `tools/q4k_mixed_gen.py` already
emits them as 16 separate bytes and `glm_matmul_mixed_tb.v` already reads them at
`8*i`; the fix was to match that existing convention in both places. The
generator's own self-test could not have caught it — it checks the reference
against the packer, not the serialisation order.

**The tolerance is measured, and it is one bf16 ULP.** This generator inherited
`max(0.06·|y|, 0.02·Σ|act·w|, 0.03)` from `swiglu_q4k_gen`. The Σ-of-absolute term
dominates because the DOWN dot product cancels, which made the tolerance **≈33 %
of |y|** — a check that could not fail short of a sign flip. Swept over seeds 0–7
(4096 outputs) the DUT is **bit-exact** with the reference: worst 0.000 ULP. It is
still a tolerance and not a bitwise check, because `silu` here uses a true
`np.exp` while `glm_act` uses a polynomial — bf16's 8-bit mantissa swallows that
difference on 4096 samples, which is not the same as never. So: one rounding
boundary of headroom, nothing more.

Three must-fail legs, and the narrow one is the point. `INJ_SWQ8_Q4K_TYPE` forces
Q4_K (also what an *undriven* `w_type` reads as). `INJ_SWMT_PASS_TYPE` keeps the
type a runtime input and only collapses the **per-pass select** to the gate's type
— invisible on two of three passes for the first combo, which is exactly why both
combos carry a down type that differs from their gate type. `INJ_SWQ8_NOCLAMP`
re-proves the clamp survived the generalisation.

### 4.3t MoE router — and an assumption I could not resolve, made testable

`src/glm53f_moe_router.v` — `make moe-router`, 80 + 123 checks, 2 must-fail
injections. Plus `moe_route` in the executable spec (`make glm53f-ref` 21 → **28**).

**`moe_router_q4k`'s math was already right**: sigmoid → top-k → renormalise →
scale, matching `expert_gating_func = 2`, `expert_weights_norm = True` and
`expert_weights_scale = 2.5`. Two things differ, and both are in the checkpoint:
its gate weights are Q4_K (`w_q` is 4 bits per lane) while `ffn_gate_inp` is
**F32** [4096, 288]; and it has **no `exp_probs_b` input** while GLM-5.3-Flash
carries one — `blk.N.exp_probs_b.bias [288] F32` — on all 43 MoE blocks. That is
the fifth time the census has redirected a design on this branch.

**fp32 sigmoid, not `glm_act`, and the reason is discreteness.** Everywhere else
a 1.2 % bf16 sigmoid error is a tolerance question. Here it is not: these scores
feed a **top-k**, so near a tie that error changes *which expert runs* — an
unbounded output change. This is the case §4.3j's fp32 sigmoid was built for, and
the first place on this branch where it is load-bearing rather than an
improvement.

**The assumption, stated rather than buried.** `exp_probs_b` is added when
**choosing** experts, and the weights come from the **unbiased** sigmoid scores.
That is the DeepSeek-v3 convention llama.cpp follows — it is **not** transcribed
from a GLM-5.3-Flash source, because none is checked out on this branch. The two
readings pick the **same experts** and differ only in the weights, so the
difference is real and quiet. Rather than pick silently:

* `glm53_flash_ref.moe_route` implements **both** (`bias_selects_only`), and
  `make glm53f-ref` asserts they **disagree** — if they ever agreed, the
  assumption would be untestable and this note would be pointless;
* `INJ_MOER_BIAS_WEIGHTS` is the other reading and must fail — it does, on the
  *weight* while the *same expert* is selected, which is precisely the shape of
  the mistake;
* `INJ_MOER_NO_BIAS` drops the bias from selection entirely and must fail too.

Anyone with the modeling source can settle this by reading one line. Until then
the RTL implements `bias_selects_only = True` and this section is where it is
recorded.

**A corpus filter that is a real limitation, not a convenience.** Draws whose
top-k margin (K-th minus (K+1)-th biased score) is under **1e-3** are rejected and
counted. At that margin the DUT's fp32 sigmoid (790 ULP) and the reference's
float64 one can legitimately disagree about the ordering, and the selection is
simply not determined by the reference. Measured, 0 of 16 draws needed rejecting
at this slice, and `exp_probs_b` changes the selection in **15 of 16** — so the
bias is doing real work in the corpus, not sitting inert.

**Two TB conventions I got wrong first**, both fixed by making the check match the
semantics rather than the implementation:
* the reference returns the selected indices **ascending** while `topk_select`
  returns them **score-descending**. Neither is more correct — the model sums over
  the selected experts, so emission order is a convention. The check is now
  order-independent: compare the selected **set**, then match each weight to its
  own index.
* a flat 0.01 weight tolerance is **below one bf16 ULP** at these magnitudes
  (0.0156 near 2.0) — a bound no correct DUT could meet. Now relative + absolute,
  and the measured worst is exactly one ULP.

**What the MoE still needs:** the expert FFN itself — 288 experts at inter 2048,
`ffn_{gate,up}_exps` Q4_K×42 + Q5_K×1 and `ffn_down_exps` Q5_K×40 + Q6_K×3, plus
the always-on shared expert (Q8_0, inter 2048). `FFN_KIND = 1` on the decoder
block still `$fatal`s.

### 4.3u The MoE FFN, and a complete decoder layer for blocks 3–44

`src/glm53f_moe_ffn.v` — `make moe-ffn`, **111 + 9 checks** — then wired into the
decoder block's FFN site behind `FFN_KIND = 1`, so `make dec-block`'s block is now
a complete decoder layer for **either** FFN kind:

    idx, w = router(x)
    y      = Σ_i w[i] · expert_idx[i](x)  +  shared(x)

One `glm53f_swiglu_mt` instance is reused serially across all TOPK+1 evaluations,
which is what `glm_decoder_block_q4k` already does for GLM-5.2: one expert's
weights are fetched per evaluation.

**The shared expert is added at weight 1**, and that is a transcription rather
than a guess — `expert_weights_scale` = 2.5 scales the *routed* weights only, and
`glm_decoder_block_q4k`'s own header already states `combine = Σ_e gate_e·y_e +
y_shared`. `glm53_flash_ref`'s self-test pins it by zeroing every routed expert
and requiring `y == shared(x)` exactly. It also carries its **own weight types**:
[scan] says routed experts are Q4_K/Q5_K/Q6_K while the shared one is Q8_0, so
`wt_sh_*` is a separate input triple, and the gate drives two *different* types
across that boundary for exactly that reason.

**The finding: the accumulation order is not observable, so it is not gated.** The
router emits its top-k score-descending; the reference accumulates ascending. fp
add does not associate, so in principle the order is visible — and the obvious
move is a must-fail leg that walks the router's order instead. Measured first:
ascending and score-descending are **bitwise identical** at the bf16 output on
**16/16 draws** across E = 8/16/32, TOPK = 3/8, INTER = 64/128, worst relative
difference exactly 0. The final fp32→bf16 rounding swallows the reordering. So
that leg would have passed and proved nothing. The order is still *pinned*, for
determinism — the FSM scans `ecur = 0..E-1` and evaluates when `ecur` is in the
selected set, which is ascending by construction and costs E cycles against
TOPK+1 expert evaluations — and the generator's self-test carries a **tripwire**
asserting the two orders still agree, so if that ever changes someone is told.

The four legs that *are* gated were each measured against the golden in Python
before being written, not written and hoped for: `INJ_MOE_WEIGHT_ROTATE` (right
experts, right weights, wrong pairing) 96/96 outputs differ, `INJ_MOE_SHARED_TYPE`
96/96, `INJ_MOE_WRONG_SELECT` 96/96, `INJ_MOE_SHARED_WEIGHTED` 94/96.

**A gate that reported PASS while reading garbage.** The first run of this TB
printed all 96 output checks passing with the vector stream **completely
misaligned**. `emit_cols` sizes its header lines from the *per-pass* K
(`NSB = K//256`, which is **0** at K = 32) while the TB strided by the
compile-time KMAX; and a `$fscanf` that matches nothing leaves its target
untouched, so every stale comparison happened to hold. The fix is not just the
stride: the vector stream now carries a **per-test sentinel** that the TB reads
back and aborts on, so a misalignment of even one token is a hard failure instead
of a green run. It earned its place immediately — it caught a second, different
misalignment (I had inserted the sentinel *before* the tolerance line) on its
first use.

The output is **bit-exact** with the golden (worst abs 0.0 over 96 outputs). The
only tolerance is on the router weight readback, and it deliberately uses
`make moe-router`'s own published bound (rel 0.02 + abs 0.01) rather than a
tighter one derived here: that quantity's accuracy is owned by that gate, and two
gates disagreeing about the same number is worse than a loose check. Measured, the
RTL differs from the golden by exactly one bf16 ULP on 6 of 9 weights — inside
that bound, and attributable to `fp32_add` (`make fp-ieee`).

**A third leg, because `dec-block` only runs `FFN_KIND = 0`.** Wiring the MoE arm
into the block would otherwise have left it never *elaborated* by any gate — a
port-width slip there surfaces only in a whole-model build, or not at all. So
`make moe-ffn` also builds `test/glm53f_moe_block_elab_tb.v`: two blocks side by
side, one of each kind, both **run** rather than merely reset. Running them is the
point — checking the ports while idle would be vacuous, since an idle MoE block
holds `moe_rw_req` at 0 too, and "it stayed 0" would then pass for either arm. Run,
the arms diverge: the MoE block's router asks for its first `ffn_gate_inp` row at
cycle **2744** (measured, and printed so a change is visible rather than absorbed
— the loop bound is 20000; the 200000 I first wrote cost 526 s of gate time for the
same claim), while the dense block holds that port tied off across the same
window. It claims elaboration and tie-off, not function: function is the 111
checks above.

**Still not built:** `ATTN_KIND = 1`, the 11 MLA blocks. That is now the only
`$fatal` left in the decoder block.

### 4.3v The MLA arm starts: NoPE, and the two absorptions

`tools/glm53f_mla_ref.py` (`make mla-ref`, **18 checks**) and
`src/glm53f_mla_score.v` (`make mla-score`, **102 + 8 checks, BITWISE**).

**It is a sibling, and the reason is structural rather than dimensional.** The repo
already has a bit-exact MLA model — `mla_attn` in `tools/glm_model_q4k_ref.py`,
mirroring `src/mla_attn_q4k.v` step for step. GLM-5.3-Flash is not that model with
different numbers in it:

- **NoPE.** `rope.dimension_count = 0`, `qk_rope_head_dim = 0`,
  `mla_use_nope = true`. No `W_kr`, no `k_rope` cache, no rotation of q — steps 3
  and part of 5 of the GLM-5.2 reference simply do not exist. A zero-width rotary
  tail is not expressible in Verilog, so this cannot be a re-parameterisation. The
  consequence the config guard already asserts: `attention.key_length ==
  kv_lora_rank` (both 512), where GLM-5.2's key was the latent *plus* a 64-wide
  rotary tail.
- **The projections are Q8_0**, not Q4_K — `mla_attn_q4k`'s `w_q` is four bits per
  lane, the same wall the FFN hit.
- qk_nope 192 → **256**, q_lora 2048 → **1536**, kv_lora **512** (now published,
  no longer the DeepSeek assumption), v_head 256, 64 heads.

**The unit built first is the inner loop, because that is the whole decode cost.**
`glm53f_mla_score` runs once per *cached token*:

    score[h][j] = ( qa[h] · rmsnorm(c_kv[j]) ) · 1/√qk_nope
    p[h][·]     = softmax over SMAX slots, slots ≥ s_len pinned to −inf
    ctx_lat[h]  = Σ_j p[h][j] · rmsnorm(c_kv[j])          ← still in the LATENT basis

`qa` arrives **already folded through W_uk**. That is the point of the latent form:
expanding a key costs `H·QK` MACs over `KV_LORA` for every cached token
(64·256·512 = 8.4 M at the real shapes) against `H·KV_LORA` = 32 K for a dot
against the latent — a factor of **256**. The output stays in the latent basis for
the same reason on the value side: `W_uv` is linear, so
`Σ_j p_j (W_uv·ckv_j) = W_uv·(Σ_j p_j ckv_j)`, and the caller expands once.

**Neither absorption is free, and that is the finding.** Both are exact in real
arithmetic; in fp they are different reduction orders. Measured:

| | differing outputs | worst rel |
|---|---|---|
| W_uk absorbed, qk = 8 | **0 / 48** | 0 |
| W_uk absorbed, qk = 32 | 69 / 192 | 1.3e-05 |
| W_uk absorbed, **qk = 256** | 517 / 768 | 3.4e-05 |
| both absorbed, kv_lora = 512 | 693 / 768 | 8.2e-05 |

So at the checkpoint's real width the form is plainly observable, and **the RTL's
choice of form is a numerical decision the golden has to match** — not a free
optimisation. At a toy qk = 8 the two forms are bitwise identical, which would have
made the check vacuous; the reference's self-test caught that and now runs a slice
wide enough to see it. This is the same shape of finding as the MoE accumulation
order (§4.3u), reached the same way — measure, then decide what may be claimed.

**The gate is bitwise, not a tolerance,** because every piece the unit uses already
has a bit-exact Python twin here: `rmsnorm_unit` at LANES=1, `glm_softmax`, and
`glm_fp.vh`'s bf16/fp32 semantics. The unit adds a new *dataflow* and no new
numerics, and the gate says exactly that. If it ever stops being bitwise the right
response is to find out what changed in the arithmetic, not to widen a bound.

**The corpus has to contain short windows.** A unit that never wrote the −inf pad
passes every test where `s_len == SMAX`, so the generator asserts both a padded and
a full case are present. Each must-fail leg was measured against the golden before
being written: `INJ_MLAS_NOPAD` moves 48 of 96 ctx elements — *all* of them in the
padded windows — `INJ_MLAS_NORESCALE` 80, `INJ_MLAS_HEAD0_PROBS` 40.

**The projection front followed** — `src/glm53f_mla_proj.v`, `make mla-proj`,
**100 + 9 checks, also bitwise**. It turns one token into exactly the two things
the score unit consumes:

    c_q   = W_dq  @ x                 →  rmsnorm  →  q = W_uq @ ·
    qa[h] = q[h]  @ W_uk[h]              the fold, one GEMV pass per head
    c_kv  = W_dkv @ x                    RAW

All four stream Q8_0 off the shared `glm_matmul_q4k`, which needed **nothing new**
— `w_type = 2`, the code on `w_hp[7:0]`, the fp16 block scale on `w_q8_d` were
already inputs, exactly as `glm53f_kda_gemv` found for KDA. The golden runs on the
**dequantised** weights, so the Q8_0 round trip is part of the *input* rather than
of the error; that is `make kda-attn`'s contract, reused.

**The fold's axis is the thing that needed a gate.** `qa[h][k]` reduces *down*
`W_uk`'s rows (over d), so the store is `[H][KV_LORA][QK]`. Reducing over the other
axis has the **same shape** and would ship silently, so the generator pins that a
transpose differs — and that perturbing one head's q leaves the other head's `qa`
bit-identical.

**A correctness bug in my own spec, found while designing this.** The reference
normalised `c_kv` on **write** *and* on read. GLM-5.2's model caches the latent raw
and normalises once, at use; mine applied rmsnorm twice. Every shape and identity
check still passed, because rmsnorm is nearly idempotent — the second pass only
divides by √(1+eps) plus rounding, **measured 2.8e-06 relative**. The fix is one
line; the durable part is the check that now stands next to it: a normalised vector
has RMS 1, so the self-test asserts the cached latent's RMS is **not** 1, and
`INJ_MLAP_NORM_CKV` is a must-fail leg at the port where the same mistake is loud.

### 4.3w The sublayer closes, and with it the last `$fatal`

`src/glm53f_mla_attn.v` — `make glm53f-mla-attn`, **125 + 8 checks, bitwise end to
end** — composes the projection front, the absorbed-latent score and a new output
stage (`src/glm53f_mla_out.v`: `W_uv` per head, then `W_o`). It is the module that
goes where `mla_attn_q4k` goes, and wiring it at `ATTN_KIND = 1` means **every one
of the 45 blocks now has a complete decoder layer**, in every attention × FFN
combination the checkpoint uses. Nothing in the block is parameterised-but-unbuilt
any more; the two `$fatal`s left are range guards, not absences.

**The golden composes rather than re-derives.** It calls the generators that
already gate the parts — `mla-proj`'s `project`, `mla-score`'s `ref_score` — plus
`matmul_q4k_col` for the two output GEMVs. That is not tidiness: it makes a
disagreement *mean* something specific, and it immediately did.

**What only a composed gate could see.** The first run had `ckv` wrong while
`make mla-proj` stayed bitwise — so the arithmetic was right and the *composition*
was not. `glm_matmul_q4k` **latches its header buses on `start`**, and a stage
pulses `mm_start` in `S_PREP`, one cycle *before* it raises `w_req`. Muxing the
shared weight channel on `w_req` therefore published the *other* stage's address
at exactly the moment the engine sampled `w_q8_d`, so every group after the first
dequantised against the wrong block scale. Selecting on the wrapper's own state
fixes it, and `INJ_MLAA_MUX_ON_WREQ` pins the fix rather than a hypothetical —
it is the bug, kept.

**The KV cache is pulled, not held, and that is a decision rather than an
omission.** The sublayer publishes its new latent on `ckv_wr` and asks for old
ones by index; those ports come straight out through the decoder block to whoever
owns memory. At the real shapes a latent is 512 values, so a 1 M-token context is
**~1 GB per MLA block** — a residency question (BRAM? DDR? paged?) that belongs
with the model, and the same open item KDA's 4.19 MB/layer recurrent state already
has. The consequence is stated in the module header rather than left implicit:
*this is not a complete attention layer on its own; it is complete given a cache.*
The gate supplies one from the testbench, which is exactly what the system will
have to do.

The corpus is required to contain `s_len == 1` (first token — the only key is the
one written this step), a padded window and a full one, and the generator asserts
all three: a unit ignoring `ckv_wr` would pass an all-first-token corpus, and one
never writing the −inf pad would pass an all-full-window one. `INJ_MLAA_NO_CKVWR`
and `INJ_MLAA_SKIP_OUT` move 80 of 125 checks each; the mux leg moves 118.

**`make moe-ffn`'s elaboration leg grew to cover this too** (3 → 5 checks): three
blocks side by side now, and on the MLA one the attention weight port goes live
while the KDA arm it replaced is held quiet in the same block.

### 4.3x The arms become runtime-selectable — because 45 layers are a mix

`ATTN_KIND` and `FFN_KIND` chose an arm at **elaboration**, which is right for a
single-kind build and impossible for a model. The repo's GLM-5.2 top runs **one**
decoder block L times and annotates each weight pull with the layer index;
GLM-5.3-Flash cannot do that with an elaboration-time arm, because its 45 layers
are a *mixture* — 34 KDA and 11 MLA, 3 dense and 42 MoE. Both machines have to be
in silicon anyway, so what was missing was being able to *choose per layer*.

So the parameters now mean **presence**, not selection: `0` = only the first arm,
`1` = only the second, **`2` = both, chosen at runtime** by the new `attn_sel` /
`ffn_sel` inputs. `0` and `1` are byte-for-byte what they were; with one arm
present the selector is a constant.

**The gate is an equivalence, and it is the strongest cheap claim available.** The
same testbench, the same vectors and the same golden, rebuilt with **both** arms
in silicon and the selectors pointing at the arms the golden describes, must give
the **identical** result — not "the new path elaborates" but *"the new path,
selected, is the old path"*. `make dec-block` now runs both builds: 676 and 676.

**It failed on its first run, 236 of 676.** At `FFN_KIND = 2` both FFN arms were
driving the block's `ffn_w_*` weight pull — **two drivers, X** — because each arm
had been wired straight to the block's port back when only one could exist. The
attention site had no such conflict, since its arms own separate port groups
(`kda_w_*` and `mla_w_*`). No single-kind build could have shown this: it needs
both arms present at once, which is exactly the configuration the equivalence gate
introduced.

**And a leg that keeps the equivalence honest.** Pointing the selectors at the
*other* arms must **not** reproduce the golden — 646 of 676 checks move. Without
it, a pair of inert selectors would pass the equivalence check and prove nothing.

### 4.3y State residency, decided from traffic rather than capacity

Two open items — KDA's recurrent state and MLA's KV cache — were both filed as "a
memory decision". They are one decision, and the residency table everyone reaches
for is the wrong input to it.

`tools/glm53_flash_memory_budget.py` already reported bytes read per token; what it
did not report is the **access pattern**, which is what actually decides placement.
At 1 M context:

| piece | capacity | per token | pattern |
|---|---|---|---|
| weights | 199.7 GB | 14.118 GB | stream, once, in order |
| KDA recurrent state | 148 MB | 285 MB (read **and** written) | **full read-modify-write**, 34 layers, never partial, never grows |
| MLA latent cache | **11.81 GB** | 23 MB | **sparse gather** of top-2048 out of the whole context |
| DSA index keys | 0.37 GB | 369 MB | **full scan**, and the only piece whose traffic grows |

**Capacity and traffic point at different pieces, in opposite directions.** The KV
cache is the big one to *store* (11.81 GB, 98 % of all non-weight state) and nearly
free to *read* — DSA gathers 2048 latents however long the context is. The index is
the reverse: 3 % of the capacity and **16×** the KV's traffic, because selecting a
top-k means scoring every pooled position. A placement argued from the residency
table alone gets both backwards.

So: **none of it belongs on-die, and the port shapes were already right.** The KDA
state is too large for a plausible SRAM (148 MB) but is sequential and pinned; the
KV is a gather; the index is a stream. All three are DDR-resident alongside the
weights, and all three are already *ports* on `glm53f_decoder_block` rather than
storage inside it — so what the model top has to add is per-layer **addressing**,
not a memory. The decision that was deferred turns out to be the one the interfaces
already encode; what was missing was the argument, not the RTL.

One tightening came with it: `TOPK_ATTN` (`[gguf] attention.indexer.top_k`) was read
with a default of 2048. It is now required, and `load_cfg` refuses a header without
it — a silent default there would have quietly decided the sparsity of the gather,
which is the whole claim the KV row rests on.

### 4.3z The per-tensor weight descriptor — the oldest open item

`src/glm53f_wdesc.v` — `make wdesc`, **384 + 22 checks**.

    (kind, layer, expert)  →  (base, klen, nsblk, wtype)

`glm_q4k_system` drives `weight_loader_q4k` with a **hardcoded single tile** —
`desc_base = 0`, `desc_nsblk = 1` — and leaves `desc_wtype` **undriven**, which the
loader reads as Q4_K. That was survivable with one model, one type and one tile in
play. It is not survivable here: [scan] says `ffn_{gate,up}_exps` is Q4_K ×42 +
Q5_K ×1 and `ffn_down_exps` is Q5_K ×40 + Q6_K ×3, so the **type varies per tensor
*and* per layer**, and a top that walks 45 layers must address a different tile for
every `(layer, expert)`.

**The shape is the checkpoint's own.** A flat descriptor would be 1412 entries; it
does not need to be, because within a kind the layers are a fixed stride apart and
the experts are uniform. So a **kind** carries `(base, layer stride, expert stride,
klen, nsblk, default type)` — about twenty rows — and what is left over is exactly
the UD quantisation bumps, as a short list of `(kind, layer) → type` **exceptions**.
That is not a compression trick: *"UD bump on `blk.{11,12,44}.ffn_down_exps`"* is
how the census describes the mix, and an exception list is what that sentence **is**.
Keeping it that way leaves the regular part checkable by arithmetic and the
irregular part short enough to read.

**Both ways of getting it wrong are silent, so both get a leg.** `INJ_WDESC_NO_EXC`
ignores the bumps, and `blk.{11,12,44}.ffn_down_exps` then streams Q5_K geometry
over Q6_K bytes — same widths, wrong decode, no error anywhere. `INJ_WDESC_NO_ESTR`
drops the expert stride, and every expert reads expert 0's bytes: a perfectly
well-formed model that has **one expert 288 times**. The generator asserts the
corpus can see them — a kind bumped on *some* layers and not others (so the default
and the exception are both observed), a non-zero expert stride, and more than one
expert and layer — because without that the legs would be decorative.

**The table reaches the testbench as a generated include**, not hand-copied
literals, so the RTL's parameters and the golden cannot drift apart silently.

**What is gated is the machine, not the values.** Turning the census into real byte
offsets needs the GGUF tensor map — the 199.7 GB checkpoint or at least its headers
— which this branch does not have. The table fed here is synthetic and *shaped*
like the real one; producing the real one is a data step, not an RTL step.

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
make kda-attn                 # 228 + 726:  the same, fetching its own Q8_0 weights (4.3o)
make q5k-loader               # 1610: packer-built Q5_K tile through the real loader (4.3p)
make dec-block                # 20 + 676:  a complete decoder layer, blocks 0-2 (4.3q, 4.3r)
make swiglu-q8                # 15 + 204:  the dense FFN, clamped SwiGLU over Q8_0 (4.3r)
make swiglu-mt                #  8 + 514:  the SAME unit on the MoE experts' mixed types (4.3s)
make moe-router               # 123 + 80:  sigmoid gating + exp_probs_b (4.3t)
make moe-ffn                  #   9 + 111: the expert loop + shared expert + combine (4.3u)
make mla-ref                  #        18: the NoPE MLA spec -- both absorptions measured (4.3v)
make mla-score                #   8 + 102: the absorbed-latent MLA inner loop, BITWISE (4.3v)
make mla-proj                 #   9 + 100: the Q8_0 projection front + the W_uk fold (4.3v)
make glm53f-mla-attn          #   8 + 125: the WHOLE NoPE MLA sublayer; ATTN_KIND=1 (4.3w)
make wdesc                    #  22 + 384: per-tensor weight descriptors (4.3z)
make dsa-indexer-ref          #       424: the DSA indexer spec, now in the ladder
```

Each of those prints its own worst-case error, so a regression moves a number
rather than flipping a boolean. Between them they carry **39 must-fail
injections** — `INJ_KDA_NODECAY`, `INJ_CONV_FLIP`, `INJ_GATE_DECAY_AFTER`,
`INJ_ONORM_GATE_FIRST`, `INJ_SINK_{SYMM,ROWFIRST,NOEPS}`,
`INJ_MAP_{POST_NO2,PRE_NOEPS,COMB_NOEPS,SOFTMAX_NOMAX}`,
`INJ_OPS_{MIX_TRANSPOSE,MIX_NOPOST,COLLAPSE_NOPRE,POST_FIRST}`,
`INJ_GEMV_{Q8_NOSCALE,MEAN_SUM,NO_EPS}`,
`INJ_SITE_{IGNORE_SUB,NO_UPDATE,PRE_FOR_POST}`,
`INJ_HCB_{SAME_WEIGHTS,NORM_SWAP,SKIP_NORM,STALE_STREAMS}`,
`INJ_KDAL_{NO_STATE,CONV_NOHIST,QK_SWAP,GATE_ORDER}`,
`INJ_KGV_{Q4K_TYPE,NO_SCALE,GRP_ALIAS}`, `INJ_DBLK_{SITE_SWAP,NO_KDA}`, `INJ_SWQ8_{NOCLAMP,Q4K_TYPE}`, `INJ_MOER_{NO_BIAS,BIAS_WEIGHTS}`,
the packer's qh-dropped image — plus
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
