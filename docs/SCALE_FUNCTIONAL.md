# Functional scaling — Q4_K status, and the prior-FP8 real-dims operator sweep

> **Read this first — two tracks, kept strictly separate.** This doc tracks *functional scale
> confidence*: how far past the tiny committed slice the datapath is actually checked.
>
> - **Current Q4_K track (`main`).** The fast per-unit gates run at the small-but-faithful slice,
>   plus **structural elaboration at the true 753B UD-Q4_K_XL shape**. The **assembled-model
>   end-to-end numeric golden EXISTS** (`make model-q4k`, **1155/1155** bit-exact vs the assembled
>   numpy reference — item 1, since closed); the **real-dims operator sweep is since RE-RUN on
>   Q4_K** (`make scale-ops` — item 2, since closed) and the **batched PE_M>1 assembled-model
>   golden EXISTS** (`make batched-q4k` — item 3, since closed at its stated scope).
> - **Prior FP8 track (tag `fp8-verified-baseline`).** The real-dims operator sweep
>   (GEMM at K=6144, router over 256 experts top-8, …) and the batched multi-seq / decode-loop
>   bit-exact checks were achieved **on the FP8 track**. Those TBs (`*_fp8`,
>   `glm_model_fp8_multiseq_tb`, `glm_fp8_soc_ms_loop_tb`, …) are **deleted from `main`** and live only
>   on tag `fp8-verified-baseline`. Their numbers are reproduced below **as prior-FP8 measurements** — they have
>   **not** been re-run on Q4_K, and none of them is a claim about current `main`. Names of the form
>   `*_fp8` map to `*_q4k` equivalents on main (see the module map in the project briefing); those
>   Q4_K modules exist but were **not** exercised at the FP8 sweep's dims.

Consistent with the honest ledger in [`../README.md`](../README.md): the bit-exact datapath results on
Q4_K are the **GEMM core** (`glm_matmul_q4k`, bit-exact to the independent ggml-Q4_K reference
`tools/q4k_ref.py`) and, since closed, the **assembled full forward** (`make model-q4k`, 1155/1155
bit-exact vs the assembled numpy reference `tools/glm_model_q4k_ref.py`). The reference itself is now
**proven bitwise-equal to real GGUF bytes at the dequant layer** (376.6M weights, Q4_K/Q6_K/Q8_0 vs
llama.cpp's own dequant — [`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)); llama.cpp **whole-runtime**
equality stays out-of-contract. Everything else is scoped below.

---

## Current Q4_K functional scale (`main`)

**Proven today** (gated sim, per the README ledger — every "bit-exact" here means *bit-exact to
`tools/q4k_ref.py`*, never the real UD-Q4_K_XL GGUF file):

| Q4_K unit | dims exercised (`make q4k`) | status |
|---|---|---|
| `q4k_prim` (`q4k.vh`: fp16→fp32 + `get_scale_min_k4`) | primitive vectors | **18/18 — bit-exact vs ggml** |
| `glm_matmul_q4k` (block-dequant → fp32 MAC → bf16) | PE_M=2, PE_N=2, **KMAX=1024 → NSB=4 super-blocks** (a faithful slice; real K=6144 re-run in `make scale-ops`, item 2) | **160/160 — bit-exact vs ggml Q4_K** *(the one true bit-exact datapath result)* |
| `swiglu_expert_q4k` (gate/up/down + silu) | HIDDEN=8, INTER=8, KMAX=256 (NSB=1; INTER=2048 re-run in `make scale-ops`) | **240/240 — functional** (self-labeled; **not** bit-exact) |
| `moe_router_q4k` (gate GEMV → sigmoid → top-K → renorm) | HIDDEN=8, **N_EXPERT=8, TOPK=2** (real 256/top-8 re-run in `make scale-ops`) | **40/40 — renorm invariants** (**not** a numeric golden) |
| Assembled `glm_model_q4k` forward pass | committed-slice full forward (`make model-q4k`) + the spec loops | **`make model-q4k` 1155/1155 — bit-exact vs the assembled numpy golden** `tools/glm_model_q4k_ref.py` (logits+argmax+h_state; + `model-q4k-acthw` 1155/1155, ACT_HW result-invariant); **`spec_decode_top` 18/18 — spec==greedy** *(DUT-vs-DUT self-consistency)*; larger `spec_batched_top` / `spec_chain_top` via `make spec-slow` |

The Q4_K GEMM proves the **multi-super-block block-scaled accumulate** mechanism (NSB=4 super-blocks
of 256 weights each) bit-exact in the fast `make q4k` gate, and — since item 2 closed — **also at the
real projection K=6144 (NSB=24)** in the standalone `make scale-ops` sweep (table in item 2 below).

**Elaborated (structure only, not a sim, no golden):** `glm_model_q4k` elaborates clean at the **true
753B UD-Q4_K_XL shape** (DIM 6144 / L=78 / 256-expert / VOCAB 154880) — `test/full_config_elab_wrap.v`,
type/width check only, *"no stimulus, no golden, no run"* ([`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md)).
The whole-chip Q4_K top `glm_q4k_system_cdc` passes the yosys structural gate (`make synth-glm`, exit 0).

**OPEN on Q4_K — stated, not implied done:**

1. **Assembled-model end-to-end numeric golden — DONE (since closed).** `make model-q4k` runs the
   *assembled* `glm_model_q4k` full forward at the committed slice **bit-exact vs the assembled
   numpy golden** (`tools/glm_model_q4k_ref.py`, which imports the same `q4k_ref.py` dequant) —
   **1155/1155** on logits + argmax + h_state, plus `make model-q4k-acthw` (1155/1155, the ACT_HW
   resource knob result-invariant). The reimplementation's dequant layer is since **proven on real
   GGUF bytes** ([`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md)); **llama.cpp whole-runtime equality
   remains out-of-contract.** (The per-unit *numeric* TBs `glm_model_tb` /
   `mla_attn_tb` / `glm_decoder_block_tb` / `mtp_head_tb` still build against the generic bf16/fp32
   twins, but the assembled Q4_K path now has its own gate.)
2. **Real-dims operator sweep on Q4_K — DONE (since closed; gate `make scale-ops`).** The EXISTING
   operator TBs re-run — same goldens, same check contracts, TB `define overrides only (the vector
   generators grew parameters, they were not forked; every slice default stays byte-identical) — at
   the real GLM-5.2 operator magnitudes:

   | operator TB (`make scale-ops`) | real dims exercised | result |
   |---|---|---|
   | `glm_matmul_q4k` | **K ∈ {512, 2048, 6144}** (KMAX=6144 → **NSB=24** Q4_K super-blocks — the real per-projection K; 16/30 tiles at K=6144) | **ALL 120 TESTS PASSED — bit-exact vs ggml Q4_K** (`tools/q4k_ref.py`) |
   | `moe_router_q4k` | **N_EXPERT=256, TOPK=8** (the real expert count / top-K), HIDDEN=128 | **ALL 40 TESTS PASSED** (renorm Sum(w)=2.5, valid distinct top-8 — the slice TB's invariant, not a numeric golden) |
   | `swiglu_expert_q4k` | **INTER=2048** (real INTER_MOE; down proj = **8 Q4_K super-blocks**/column), HIDDEN=64 | **ALL 256 TESTS PASSED** (tolerance golden, as at the slice) |
   | `glm_softmax` | **LEN=2048** (the real DSA attention window `index_topk`) | **ALL 32896 TESTS PASSED** (worst rel 0.0051, fp64 golden + slice tolerances) |
   | `rmsnorm_unit` | **LEN=6144** (the real MODEL_DIM — the committed TB already sweeps it) | **ALL 55 TESTS PASSED** (worst rel ≤ 2⁻⁶ tol, fp64 golden) |
   | `rope_interleave_unit` | **ROT_DIM=64** (the real qk_rope_head_dim), positions to ~1M | **ALL 2512 TESTS PASSED** (fp64 golden) |

   **Scoped honestly:** the router/SwiGLU GEMV reductions stay below the real K=6144 (that reduction
   mechanism is exactly what the K=6144 GEMM leg proves bit-exact at NSB=24); there is no standalone
   `mla_attn_q4k` real-geometry TB on `main` (the MLA datapath is gated at the slice by
   `make model-q4k`; the fp8 real-geometry MLA numbers below remain FP8-only); and the softmax
   directed ramps hold the committed logit envelope (shifted args ≥ −63) at LEN=2048 — shifted-arg
   magnitudes beyond ~−180 are outside the exp pipe's documented softmax range and are not claimed.
3. **Batched PE_M>1 numeric golden on Q4_K — DONE at its stated scope (gate `make batched-q4k`);
   per-row-KV multi-seq / decode-loop TBs remain FP8-only.** `test/glm_model_q4k_pem_tb.v` runs the
   assembled `glm_model_q4k` at **PE_M=2** against standalone **PE_M=1** runs per row — the same
   committed-slice weight set as `make model-q4k` — and asserts per-row **logits + argmax + h_state
   BIT-EXACT** (X-aware bf16 pattern compare), with **row 0 additionally anchored bit-exact to the
   assembled numpy golden** (not just DUT-vs-DUT): **ALL 4233 TESTS PASSED** over 3 (pos, s_len)
   scenarios with per-row-divergent tokens/MoE routing. Scope: PE_M widening for a **shared**
   sequence (PER_ROW_*=0). The per-row-KV multi-sequence (`kc_seq`-keyed windows) and
   continuous-batching decode-loop TBs (`glm_model_fp8_multiseq_tb`, `glm_fp8_soc_ms_loop_tb`)
   remain **FP8-only** (tag `fp8-verified-baseline`) — `glm_q4k_soc_ms` still has no Q4_K equivalent of those.
   Batching / multi-seq serving is a **capability of the silicon**, not the B=1 personal box's
   operating mode (see the scope note below).

Real GLM-5.2 operator dims (config.json / [`ACCEL_GLM52.md`](ACCEL_GLM52.md), which records
`q_lora_rank = 2048` confirmed vs the real safetensors **on the prior FP8 track** — `q_a_proj.weight
[2048,6144]`; `kv_lora 512` is **[PENDING safetensors]**): NOPE=192, ROPE=64, V_DIM=256, KV_LORA=512,
**Q_LORA=2048**; INTER_MOE=2048, INTER_DENSE=12288; N_EXPERT=256, TOPK=8; super-block=256 weights;
per-projection K up to ~6144. These are **model-architecture** facts, independent of quant format.

---

## Prior FP8 track — real-dims operator sweep (tag `fp8-verified-baseline`; Q4_K re-run since DONE, `make scale-ops`)

> **Prior-FP8 measurements.** Everything in this section was measured on the **FP8** datapath, whose
> TBs (`glm_matmul_fp8`, `moe_router_fp8`, `swiglu_expert_fp8`, `mla_attn_fp8`, `glm_model_fp8`) are
> **deleted from `main`** and live on tag `fp8-verified-baseline`. The numbers are **not** Q4_K results. They are
> kept here to document what the real-dims sweep proved on the prior track; the equivalent Q4_K
> sweep is **since re-established** (item 2 above, `make scale-ops`) — except the MLA real-geometry
> leg, which has no standalone Q4_K TB on `main`.

On the FP8 track, the operator TBs were run at real per-head / per-block / per-expert sizes against
**parametric goldens** (independent fp64 models of the *same* block-scaled E4M3 math that regenerate
from the TB `localparam`s), emitting `ALL N TESTS PASSED` and `$fatal` on any mismatch/X. Verilator
`--binary` gave fast sims; slower ones fell back to iverilog.

| operator TB (tag `fp8-verified-baseline`) | real dims exercised | result | tool |
|---|---|---|---|
| `glm_matmul_fp8` → `glm_matmul_q4k` | **K=6144** (NB=**48** [128]-blocks — real projection K), BLK=128 | **ALL 224 PASSED** (worst err/tol 0.49) | verilator (0.64 s) |
| `moe_router_fp8` → `moe_router_q4k` | **N_EXPERT=256, TOPK=8** (the real expert count) | **ALL 230 PASSED** (top-K indices exact) | iverilog (7.7 min) |
| `swiglu_expert_fp8` → `swiglu_expert_q4k` | **INTER_MOE=2048, INTER_DENSE=12288** (down = 96 [128]-blocks) | **ALL 512 PASSED** (worst err/tol 0.039) | verilator |
| `mla_attn_fp8` → `mla_attn_q4k` | NOPE=192, ROPE=64, V_DIM=256, KV_LORA=512, H_HEADS=8, MODEL_DIM=256 | **ALL 3 PASSED** (worst rel **2.5e-4**) | verilator (73 s) |

On the FP8 track this covered two operators at the *full* real magnitudes (GEMM at real **K=6144** / 48
block-scale accumulations; router over the real **256** experts top-8, weights summing to 2.5), SwiGLU
at real **INTER_MOE=2048**, and MLA at real per-head / LoRA geometry (192/64/256/512). The Q4_K
equivalents (`*_q4k`) are **since re-run at real dims** by `make scale-ops` (item 2's table: GEMM
K=6144 bit-exact, router 256/top-8, SwiGLU INTER_MOE=2048) — with two honest deltas vs this FP8
table: no Q4_K re-run at **INTER_DENSE=12288**, and no standalone `mla_attn_q4k` real-geometry TB.

**Intermediate full-model (tag `fp8-verified-baseline`):** `glm_model_fp8` at **MODEL_DIM=256, VOCAB=512, L=8,
N_EXPERT=16, TOPK=4, H_HEADS=8** — 2×+ the committed slice on every axis — **ALL 3 PASSED** (gworst_rel
0.0068, 77 s). The whole embed → 8 FP8 layers → norm → LM head → argmax pass composed correctly at
larger scale **on the FP8 datapath**. The Q4_K assembled path now has its own numeric golden at the
*committed slice* (`make model-q4k`, 1155/1155 — item 1, since closed), but there is **no Q4_K
equivalent of this enlarged 2×-slice run**.

### Batched multi-sequence + decode loop (tag `fp8-verified-baseline`; Q4_K PE_M>1 golden since DONE — the per-row-KV multi-seq / decode-loop TBs remain FP8-only)

On the FP8 track the full `glm_model_fp8` also composed correctly on the *batch / sequence* axis. With
`PER_ROW_SEQ=1` each `PE_M` row was a **different** sequence attending its own KV window, while the
query-side weight/projection fetch was **shared** across sequences (the batching bandwidth win) —
`glm_model_fp8_multiseq_tb` (2 seqs) and `glm_model_fp8_multiseq4_tb` (B=4) proved per-row
argmax/logits **BIT-EXACT** vs the per-seq `PE_M=1` goldens across dense + sparse cases (~41% / ~52%
fewer attn-weight beats than B separate runs), byte-identical at `PER_ROW_SEQ=0`. A batched multi-seq
top `glm_fp8_soc_ms` (→ `glm_q4k_soc_ms` on main) wrapped this with a real `NSEQ`-window
`kv_cache_pager`, `expert_cache_pf`, and a REAL per-layer KV store (`kv_mem`); its **multi-step
continuous-batching decode loop** (`glm_fp8_soc_ms_loop_tb`) decoded N tokens/seq in one start with each
row's step-k token **BIT-EXACT** vs a standalone `PE_M=1` model decoding that sequence alone.

**These TBs are deleted from `main`** (they were FP8) — the numbers above are **prior-FP8**, not current.
On `main` the Q4_K batch axis now has its own **direct numeric golden**: `make batched-q4k` proves
`glm_model_q4k` at **PE_M=2 == per-row PE_M=1** bit-exact on logits+argmax+h_state, row 0 anchored to
the assembled numpy golden (item 3 above — shared-sequence scope). The **per-row-KV multi-seq**
(`kc_seq`-keyed windows) and **decode-loop** TBs have **not** been rebuilt on Q4_K; `glm_q4k_soc_ms`
still has no equivalent of `glm_fp8_soc_ms_loop_tb`.

**Scope (format-agnostic).** This batch / multi-seq axis is a **capability of the silicon**, not the
product's operating mode — the product is a **local, single-user box** that runs **B=1** (one user, the
full GLM-5.2 753B model in Q4_K, at the rung-dependent tok/s below). Verifying the batch axis proves the
*same* silicon *could* run batched — a **non-target, datacenter deployment** where many *different* users
share the weight/projection fetch (the "batching bandwidth win") — which the personal box does not do.

### The `mla_attn` multi-block block-scale fix — a TB-golden/stimulus bug, not an RTL bug (tag `fp8-verified-baseline`)

> Prior-FP8 finding, documented for methodology. The same multi-super-block block-scale reasoning
> applies to the Q4_K datapath, but the numbers below (64×, 2.5e-4) are FP8-track measurements.

At real MLA dims every weight projection has K > 128 (so NB > 1 [128]-K-blocks), but the committed FP8
MLA TB drove the block-dequant scale on **only K-block 0** of `w_scale`, leaving blocks ≥ 1 at scale 0 —
so the block-scaled DUT correctly dequantized block 0 and dropped every K beat past the first 128, a
`(K/BLK)×` under-sum that compounded through W_uk · W_uv (K=512 → 4×) and W_o (K=2048 → 16×) to the
observed **64×**. The fix replicated each matrix's uniform block scale across **all NB K-blocks**
(matching what the golden assumed). Byte-identical at the committed slice (KMAX ≤ BLK → NB=1); at real
dims the DUT and golden then agreed to **2.5e-4**. **No RTL change** — the multi-block block-scaled
datapath was already correct; the TB stimulus was single-block. Two independent agents found and fixed
the same bug; the committed slice stayed **7/7** byte-identical.

---

## Product speed is rung-dependent (format-agnostic)

Where product speed is quoted, it is **rung-dependent** (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)),
and the workload is **memory-bandwidth-bound**, so the *same* Q4_K RTL runs on every rung — only the
bandwidth the silicon can feed it changes:

- **~5–8 tok/s [EST]** on the near-term prove-it FPGA (rung ①) — slow but real + Q4_K bit-exact to the
  ggml reference.
- **~15–40 tok/s [EST]** on the funded custom board (rung ②) — the interactive product.
- **≈80 tok/s [measured-inputs EST]** at volume (pre-measurement band ~76–95; rung ③ SoC/ASIC — updated 2026-07: the primary rung-③ design
  point is now **512 GB LPDDR5X full residency**, see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)).

The full model is the ~467 GB `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` checkpoint (753B params, ~40B
active/token). All product tok/s figures are **[EST]** — roofline-modeled. The **FPGA fit is since
MEASURED** (Vivado ML 2026.1 real synth + full place&route of `glm_q4k_system_cdc` on **XCKU3P**:
142.3K LUT / 87.5%, ~100K FF, 421 DSP, hold met; routed-Fmax campaign closed at **46.5 MHz**, every
round re-proven bit-exact on the 1155-test assembled golden — see [`../fpga/README.md`](../fpga/README.md);
the old Gowin/nextpnr scaffold is removed), but **board bring-up is not done**, so absolute product
tok/s stays [EST].

> **UPDATE — measured-proxy design points ([`H_MEASUREMENT.md`](H_MEASUREMENT.md), OLMoE proxy
> first pass):** with measured h/U inputs the roofline menu reads NVMe 1–2 (no multipliers)
> ~0.5–1 tok/s; 90 GB DRAM + 100 GB/s → 13–24; 90 GB + 200 GB/s (ONFI 64ch) → 25–47;
> 225 GB + 200 GB/s → 54–127 (all still **[EST]**, MEASURED-PROXY inputs). The spec-decode
> multiplier in the roofline must be read as **A/U(K) ≈ 1.1–1.3× at K=4** (measured union factor
> U(4)=2.25–2.64), **not** a full ×K. See also [`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md).
>
> *(Updated 2026-07: U(K) is since **GLM-family MEASURED** — GLM-4.5-Air traced via MoE-gate hooks
> on an H100, U(4)=2.60–2.71 — superseding the OLMoE-proxy U above (OLMoE stays as the first-pass
> history), and the primary rung-③ design point pivoted to **512 GB LPDDR5X full residency,
> design point ≈80 tok/s [measured-inputs EST]** ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)); the streaming rows above
> (ONFI 64ch / the 225 GB-cache 54–127 band) now apply to rung ① / the hybrid-upside SKU (h ≥ 0.75)
> / >512 GB checkpoints, not the primary SKU.)*

---

## Coverage statement (honest — Q4_K)

Q4_K scale confidence today rests on three legs, scoped to what is actually checked:

1. **Structural** — `glm_model_q4k` elaborates clean at the true 753B UD-Q4_K_XL shape
   ([`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md)); the whole-chip Q4_K top passes the yosys gate
   (`make synth-glm`). *Structure/width only — no stimulus, no golden.*
2. **Operator-functional at the slice** — the Q4_K GEMM core is **bit-exact to the ggml-Q4_K reference**
   (`glm_matmul_q4k` 160/160, proving multi-super-block block-scale accumulate at NSB=4), with
   `swiglu_expert_q4k` functional (240/240) and `moe_router_q4k` renorm-invariant (40/40), all at slice
   dims — **plus the real-dims Q4_K sweep** (`make scale-ops`, item 2, since closed: GEMM bit-exact at
   K=6144/NSB=24 120/120; router 256/top-8 40/40; SwiGLU INTER_MOE=2048 256/256; softmax LEN=2048
   32896/32896; rmsnorm LEN=6144; rope ROT_DIM=64).
3. **Assembled forward pass** — **bit-exact vs the assembled numpy golden** at the committed slice
   (`make model-q4k` 1155/1155 + `model-q4k-acthw` 1155/1155 — item 1, since closed), plus the
   **batched PE_M>1 golden** (`make batched-q4k`, item 3: PE_M=2 rows == per-row PE_M=1 runs
   BIT-EXACT, row 0 anchored to the numpy golden), plus **spec==greedy** self-consistency
   (`spec_decode_top` 18/18; larger loops via `make spec-slow`).
   The golden's dequant layer is proven on real GGUF bytes; llama.cpp whole-runtime stays out-of-contract.

**Honestly capped.** Full-config *functional* sim remains infeasible (LM head ~2.38e8 K-beats/token);
it is covered structurally by leg 1, not by a run. The prior FP8 track additionally had a real-dims
operator sweep — since re-established on Q4_K (`make scale-ops`, item 2, minus the MLA real-geometry
leg) — and a real-tensor checkpoint validation, which remains branch-`fp8`-only.
The assembled-model end-to-end numeric golden is since **closed** (`make model-q4k`, item 1) and
the golden's **dequant layer is since proven on real GGUF bytes** ([`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md));
the **real-dims Q4_K operator sweep** (item 2, `make scale-ops`) and the **batched PE_M>1 Q4_K
golden** (item 3, `make batched-q4k`) are **since closed at their stated scopes**. The largest remaining
opens are the **per-row-KV multi-seq / decode-loop Q4_K TBs** (FP8-only, item 3 scope note), a
standalone **`mla_attn_q4k` real-geometry TB**, and the enlarged 2×-slice assembled run — llama.cpp
whole-runtime equality stays out-of-contract by design.
