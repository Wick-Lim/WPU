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
| 2 | **KDA linear-attention block** — recurrence core, conv step and gate step DONE; layer wrapper open | 34 / 45 layers | `src/kda_recur.v` (state update, `make kda`, §4.3c), `src/kda_conv_step.v` (k=4 causal conv + SiLU, `make kda-conv`, §4.3d) and `src/kda_gate_step.v` (forget gate → `exp(g)` and beta, `make kda-gate`, §4.3e). Still open: composing the layer — q/k/v/beta/f/g projections and `o_proj` are ordinary GEMVs and `o_norm` is `rmsnorm_unit` + a sigmoid gate, none yet wired around the three new units; and the state has to live in BRAM/DDR at the real shape (4.19 MB/layer). **And a finding from the gate step (§4.3e): the KDA gate path may need an fp32 sigmoid — the repo's only activation unit is bf16.** |
| 3 | **Hyper-connections (mHC)** — spec DONE, RTL open | every block's residual path | `tools/glm53_flash_ref.py: hyper_connection`, gated. Bigger than first scoped: the block carries **4 parallel residual streams**, so this changes the block interface, not just the adder. Still needs a fixed-point study before RTL |
| 4 | ~~**Clamped SwiGLU**~~ — **DONE** (RTL + 4-leg gate) | every FFN | `SWIGLU_CLAMP` in `swiglu_expert_q4k`, default 0 so the GLM-5.2 path stays byte-identical. Gated by `make q4k`: the feature leg, a **vacuity leg** (the clamp golden must FAIL against an unclamped DUT), and a **must-fail injection** that clamps the gate symmetrically |
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
