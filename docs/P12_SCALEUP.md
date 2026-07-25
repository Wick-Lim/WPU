# P1.2 — full-config parameter scale-up (elaboration study)

> **Retargeted FP8 → Q4_K.** This study was first executed on the FP8 sibling hierarchy
> (now the *prior* track, preserved on tag `fp8-verified-baseline`). The
> module names below are the **current Q4_K product** units (`glm_model_q4k` /
> `glm_decoder_block_q4k` / `mla_attn_q4k` / `moe_router_q4k` / `swiglu_expert_q4k` /
> `glm_matmul_q4k`) — the drop-in Q4_K siblings share the *identical* parameterization,
> FSM, dataflow and latency of the FP8 originals, so the elaboration contract (width
> threading, `$clog2`, part-selects, connectivity) carries over unchanged. Where a
> **specific measured number** below (verilator warning counts, batched attn-weight-beat
> savings) was produced by the original FP8-track run, it is flagged as such with a Q4_K
> re-run marked **PENDING** — it is never relabelled as a Q4_K measurement. Product context:
> [`README.md`](../README.md), [`Q4K_RETARGET.md`](Q4K_RETARGET.md),
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md).

Task **B4** of the product next-steps plan: establish the *structural* contract for
scaling the committed RTL slice up to the real GLM-5.2 shape, without attempting a
full-config functional simulation (which is infeasible — see §6).

The single source of truth for the real shape is [`configs/full_glm52.vh`](../configs/full_glm52.vh)
(every value cited to `config.json` / [`docs/ACCEL_GLM52.md`](ACCEL_GLM52.md)).

## 1. Parameter map (slice → full)

| Parameter | Slice (committed TBs) | Full GLM-5.2 | Source |
|---|---|---|---|
| MODEL_DIM | 128 | 6144 | `hidden_size` |
| L / N_DENSE | 6 / 3 | 78 / 3 | `num_hidden_layers` / `first_k_dense_replace` |
| VOCAB | 256 | 154880 | `vocab_size` |
| H_HEADS | 4 | 64 | `num_attention_heads` |
| NOPE / ROPE | 16 / 16 | 192 / 64 | `qk_nope_head_dim` / `qk_rope_head_dim` |
| V_DIM | 32 | 256 | `v_head_dim` |
| Q_LORA / KV_LORA | 64 / 32 | 2048 / 512 † | `q_a_proj`/`kv_a_proj` low-rank sizes |
| N_EXPERT / TOPK | 8 / 2 | 256 / 8 | `n_routed_experts` / `num_experts_per_tok` |
| INTER_MOE / INTER_DENSE | 64 / 256 | 2048 / 12288 | `moe_intermediate_size` / `intermediate_size` |
| TOPK_ATTN | 8 | 2048 | `index_topk` |
| POSW | 20 | 20 | 2^20 = 1,048,576 ≥ 1M context |
| S_MAX | 8 | **keep small** | latent-ring depth — see §3 |

† `q_lora` = **2048, CONFIRMED** against the real checkpoint safetensors
(`q_a_proj.weight` shape `[2048,6144]`) — an earlier 1536 guess was corrected. `kv_lora`
= **512 remains the standard DeepSeek-MLA assumption, [PENDING]** safetensors confirmation.
Both are consistent with [`configs/full_glm52.vh`](../configs/full_glm52.vh) and the
companion [`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md).

## 2. Elaboration result — the parameterization threads structurally at real dims

**The authoritative full-config elaboration is now
[`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md).** It instantiates the compute-die top
`glm_model_q4k` at the **TRUE** full config (MODEL_DIM=6144, VOCAB=154880, N_EXPERT=256,
Q_LORA=2048) via the dangling-instance wrapper [`test/full_config_elab_wrap.v`](../test/full_config_elab_wrap.v)
and elaborates it with **`iverilog -tnull`** (type/width front-end only) — **zero errors**
across the full `glm_model_q4k → glm_decoder_block_q4k → {mla_attn_q4k, moe_router_q4k,
swiglu_expert_q4k, glm_matmul_q4k}` hierarchy. This is an **elaboration study, not a
simulation** (no stimulus, no golden — a full-config functional run is intractable, §6).

The earlier B4 study (this task) reached the same structural conclusion by an *independent*
parser (**verilator `--lint-only`**) at *reduced* MODEL_DIM/VOCAB (768/1024) with N_EXPERT=16,
Q_LORA=1536 — pure data-width scalings with no structural effect. Confirmed clean at
**NOPE=192, ROPE=64, V_DIM=256, KV_LORA=512, Q_LORA=1536, INTER_DENSE=12288,
INTER_MOE=2048, N_EXPERT=16, TOPK=8**:

- **Zero structural errors** — no negative replication, no unresolved parameter, no
  unknown module, no zero/negative-width range across the hierarchy.
- The residual verilator output was **512 SELRANGE + 56 PINMISSING** warnings — both
  artifacts of the *dangling* wrapper instance (no ports connected → open inputs make
  bit-selects nominally out-of-range); they are **not** RTL defects and do not appear
  when the module is driven by a real TB. (These specific warning counts, and the
  subsequent SELRANGE-family clear noted in [`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md),
  were **measured on the original FP8-track run**; the drop-in Q4_K siblings share the
  identical parameterization so the contract carries, and the full-config `iverilog -tnull`
  path above re-confirms 0 errors on the Q4_K tree. A verilator-side re-count of those exact
  warning totals on the Q4_K tree is **[PENDING]**.)

> yosys note: `hierarchy` re-elaboration of a parameter-overridden `glm_model_q4k`
> trips a spurious "Static cast is only supported in SystemVerilog mode" on the
> `EVW'(NEVAL-1)` cast in `glm_decoder_block_q4k.v` (a yosys `-sv`-flag-loss quirk on
> derived modules) — **not** an RTL problem (the line is valid SV; the whole-chip
> `make synth-glm` gate elaborates `glm_q4k_system_cdc` fine at slice params). The
> `iverilog -tnull` / verilator paths above are the authoritative elaboration check for
> full-config dims.

## 3. Latent finding — a config-validity constraint (dense FFN ≥ MoE FFN)

The FFN weight-pull MUX in `glm_decoder_block_q4k.v` zero-extends the MoE group/k fields
up to the dense fields, via clamped extension widths (`glm_decoder_block_q4k.v:453-456`):

```verilog
localparam integer FF_GXT = (FF_GWD > FF_GWM) ? (FF_GWD - FF_GWM) : 0;
localparam integer FF_KXT = (FF_KWD > FF_KWM) ? (FF_KWD - FF_KWM) : 0;
assign fw_grp = mode_q ? {{FF_GXT{1'b0}}, em_wgrp} : ed_wgrp;
assign fw_k   = mode_q ? {{FF_KXT{1'b0}}, em_wk}   : ed_wk;
```

with `FF_GWD = $clog2(max(INTER_DENSE,MODEL_DIM)/TN + 1)` and
`FF_GWM = $clog2(max(INTER_MOE,MODEL_DIM)/TN + 1)` (`glm_decoder_block_q4k.v:159-160`).
This **assumes `FF_GWD ≥ FF_GWM`**, i.e. the dense intermediate is at least as wide as the
MoE intermediate. If a config set `INTER_MOE > INTER_DENSE`, the raw `FF_GWD-FF_GWM` count
would go negative (an earlier elaboration surfaced `32'hfffffffd` = −3) and the netlist
would be malformed.

- **At the real GLM-5.2 config this holds**: INTER_DENSE=12288 ≫ INTER_MOE=2048, so the
  constraint is satisfied and there is no bug for the actual target.
- It is nonetheless a **latent, silent** structural constraint — now **guarded** (the
  follow-up is DONE): an elaboration-time guard (`glm_decoder_block_q4k.v:441-444`)
  `$fatal`s if `FF_GWM > FF_GWD` or `FF_KWM > FF_KWD`, and the `FF_GXT`/`FF_KXT` extension
  widths above are `max(0,…)`-clamped, so a future `INTER_MOE > INTER_DENSE`
  misconfiguration fails loudly instead of silently producing a bad netlist (byte-identical
  for every valid config).

## 4. Block-scale bookkeeping at non-block-multiple dims (B5)

The real config has projection/FFN dims that are **not** multiples of the Q4_K super-block
(e.g. per-head NOPE=192, ROPE=64; `W_kr` out=64; and any K that is not a multiple of 256).
Q4_K packs weights in **256-weight super-blocks** (fp16 `d`/`dmin` + 8 × 6-bit sub-block
scale/min over 32-weight sub-blocks + 128 B of 4-bit codes; see [`tools/q4k_ref.py`](../tools/q4k_ref.py)).
The Q4_K GEMM must therefore handle a **ragged final super-block** (and, within it, ragged
32-weight sub-blocks).

This is already proven by the committed `test/glm_matmul_q4k_tb.v` (`make q4k` →
`glm_matmul_q4k` **160/160 checks, bit-exact vs the ggml Q4_K reference** `tools/q4k_ref.py` —
whose dequant layer is now proven bitwise-equal to real GGUF bytes,
[`GGUF_CROSSCHECK.md`](GGUF_CROSSCHECK.md); llama.cpp whole-runtime stays out-of-contract):

- The vectors (`tools/q4k_matmul_gen.py` → `build/q4k_vec.txt`) sweep
  **K ∈ {32, 64, 128, 200, 256, 288, 512, 600, 768}** with *distinct* per-(column,
  super-block) `d`/`dmin`/6-bit-scale params. **K=200** exercises a *partial single*
  256-super-block; **K=288 and K=600** straddle super-block boundaries with a **ragged
  tail**. Every output element is checked against the ggml-exact dequant + fp32-MAC golden.
  The DUT streams `NSB = ceil(KMAX/256)` super-blocks and the golden mirrors it per
  super-block (`sb = k//256`, sub-block `(k%256)//32`).
- The N (output-column) direction is **caller-tiled**: `glm_matmul_q4k` processes `PE_N`
  columns per tile with a per-(col, super-block) scale, so a non-block-multiple output
  width is simply a different tile count at the caller (`mla_attn_q4k` /
  `glm_decoder_block_q4k`), not a module-level concern. B4's real-dim elaboration (§2)
  confirms those callers thread the real per-head widths (NOPE=192, ROPE=64, V_DIM=256)
  structurally.

So the ragged-super-block scale contract for full-config dims is **established** (the
q4k matmul TB + §2 elaboration). What remained **OPEN** for B5 at the time (since
**CLOSED** — see the update note after the quote):

> **The assembled `glm_model_q4k` end-to-end functional golden DID NOT EXIST at the time
> of this study** (since closed — see the update note below). The
> Q4_K arithmetic is verified only at the **per-op** level (`q4k_prim` 18/18,
> `glm_matmul_q4k` 160/160, `swiglu_expert_q4k` 240/240, `moe_router_q4k` 40/40 — see
> `make q4k` and [`Q4K_RETARGET.md`](Q4K_RETARGET.md)). The **model-level** TB
> (`test/glm_model_tb.v`) runs against the generic **bf16 twin** `src/glm_model.v` (which
> shares the FSM / dataflow but has **zero Q4_K** datapath) — so it re-proves the assembled
> *control/dataflow*, **not** the assembled Q4_K numeric path against any golden. A full
> *intermediate-size* end-to-end Q4_K functional run is deferred (a full-config functional
> run is infeasible per §6); closing the assembled-Q4_K-vs-golden gap is tracked in
> [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) / [`SCALE_FUNCTIONAL.md`](SCALE_FUNCTIONAL.md).

> **UPDATE — this gap is CLOSED.** The assembled end-to-end golden now **EXISTS and is
> gated**: `make model-q4k` runs the assembled `glm_model_q4k` full forward **bit-exact vs
> the assembled numpy reference** `tools/glm_model_q4k_ref.py` — **1155/1155** on
> logits + argmax + h_state (+ `make model-q4k-acthw` 1155/1155, proving the ACT_HW resource
> knob result-invariant). Still the team's own reimplementation — bit-exactness to the real
> GGUF bytes / llama.cpp remains OPEN (see [`../README.md`](../README.md)).

## 5. Batch-dimension scale-up — multi-sequence batched attention (B=4)

> **Scope — this is the NON-TARGET datacenter/aggregate regime, not the product.**
> The product is a **local, single-user box** that runs **B=1** (one user, one sequence);
> single-user interactive throughput (rung-dependent per [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md):
> **~5–8 tok/s [EST]** on the near-term prove-it FPGA today, **~15–40 tok/s [EST]** on the funded
> custom board, **≈80 [measured-inputs EST]** at volume (updated 2026-07: the rung-③ primary design point is now
> **512 GB full residency** — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)) — all
> bandwidth-roofline projections **[EST]**; the **Vivado fit is
> since MEASURED** — full place&route of `glm_q4k_system_cdc` on XCKU3P, routed Fmax 46.5 MHz,
> bit-exact on the 1155-test assembled golden, see [`../fpga/README.md`](../fpga/README.md) —
> while **board bring-up is still open**, and the measured-proxy h/U design points refine the rung
> numbers ([`H_MEASUREMENT.md`](H_MEASUREMENT.md): 90 GB DRAM + 100 GB/s → 13–24 tok/s; the
> earlier 225 GB + 200 GB/s → 54–127 tok/s band survives only as the **hybrid-upside-SKU-if-h≥0.75**
> case — the primary rung-③ point is full residency ≈80 [measured-inputs EST])) is the only metric that
> matters for it. Batching B *different*
> sequences is the *aggregate-serving* (datacenter) use of the **same** silicon — a legitimate
> analysis of what the RTL *could* do batched, kept here as a secondary result, but **never**
> the product's headline speed. The "batching bandwidth win" below applies only when many
> different users are co-batched, which the personal box does not do.

§1–§4 scale the *parameter* (model-dimension) axis. The **batch** axis scales
independently: `glm_model_q4k` runs `PE_M=B` **different** sequences in ONE forward
(`PER_ROW_SEQ=1`), each row attending its OWN sequence's KV window (per-row-slot union +
`kc_seq` routing) while the query-side weight/projection stream is **shared** across the B
sequences — the batching bandwidth win. This is P1.3 work; the datapath (`PER_ROW_SEQ`,
per-row-slot union, `kc_seq`) is present in the current Q4_K tree (`glm_model_q4k` /
`mla_attn_q4k`, batched top `glm_q4k_soc_ms`).

**Prior FP8-track measurement (tag `fp8-verified-baseline`); Q4_K standalone re-run [PENDING].** The
following B=2 / B=4 figures were produced by the FP8 multi-seq TBs
(`glm_model_fp8_multiseq_tb.v` / `glm_model_fp8_multiseq4_tb.v`, now on tag `fp8-verified-baseline`). The
union logic is retargeted unchanged into `mla_attn_q4k` / `glm_model_q4k`, so the beat-count
ratios are expected to reproduce, but a standalone Q4_K re-measurement is pending:

- **B=2:** 2 sequences batched, per-row argmax/logits **bit-identical vs a standalone
  `PE_M=1` run** (DUT-vs-DUT self-consistency — **not** a numeric golden), ~**41%** fewer
  attn-weight beats than two separate runs; dense + sparse.
- **B=4:** 4 different sequences batched in one forward, all 4 rows per-row **bit-identical
  vs the `PE_M=1` run**, dense AND sparse (`S > TOPK_ATTN`), with ~**52%** fewer attn-weight
  beats than 4 separate decodes.
- **`PER_ROW_SEQ=0` is byte-identical** to the shared-sequence datapath throughout.

Note the "golden" here is *itself* a `PE_M=1 glm_model_q4k`, so this is **DUT-vs-DUT
self-consistency**, not validation against an external numeric golden (the assembled Q4_K
model's numeric golden — `make model-q4k`, since closed per the §4 update — gates the B=1
forward, not these batched runs).

The **current** Q4_K coverage of the batched/self-consistency invariants runs through the
spec-decode tops (`make spec-slow`: `spec_batched_top` binds *committed == greedy* and the
÷K speedup against an independent `PE_M=1 glm_model_q4k` reference — a cycle-level TB
invariant; the product-level spec-decode *bandwidth* amortization is measured at
**A/U(K) ≈ 1.1–1.3× at K=4**, not ×K ([`H_MEASUREMENT.md`](H_MEASUREMENT.md) — U(K) since
GLM-family **measured** on GLM-4.5-Air, U(4)=2.60–2.71, superseding the first-pass OLMoE proxy);
`spec_chain_top` the MTP-chain draft). See [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) P1.3 for the rest of the
multi-seq stack (the `glm_q4k_soc_ms` batched top + host FSM, its `N_STEPS>1`
continuous-batching decode loop, the real `kv_mem` KV store, `DSA_REAL_IDX=1` under
multi-seq, `kv_cache_pager` `NSEQ` windows, and the expert-union-skip MoE batching now
**folded inline into `glm_decoder_block_q4k`** — the standalone `batched_moe.v` module was
removed).

## 6. What is explicitly OUT of scope for P1.2

- **Full-config functional simulation.** The LM-head GEMV alone streams
  MODEL_DIM(6144) × VOCAB(154880) ≈ 2.38e8 K-beats **per token**, and a 256-expert MoE
  layer runs into the billions of cycles — a single full-config token would take an
  impractical wall-clock time in iverilog. P1.2 is therefore a **structural +
  intermediate-size** contract (this doc + `configs/full_glm52.vh` + §2 elaboration),
  *not* "set the params and run the TB." Q4_K functional fidelity is proven **per-op** by
  the committed Q4_K TBs (`make q4k`); the assembled `glm_model_q4k` end-to-end numeric
  path against a golden is since **CLOSED** at the committed slice (`make model-q4k`
  1155/1155 — see the §4 update note).
- **Attention scratch at 1M context (S_MAX).** `mla_attn_q4k` sizes its
  `scores`/`probs`/`vstore` scratch by `S_MAX`, so a full-context `S_MAX=2^20` would make
  the scratch (and elaboration) explode. Decoupling the attention window from the 1M
  position field is task **B7** (SWIN vs S_MAX). Full-config integration keeps `S_MAX`
  small (the latent-ring depth), independent of the 1M `POSW`.
