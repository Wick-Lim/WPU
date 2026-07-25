# Code Coverage — Verilator structural (line / toggle / branch)

> **⚠️ TRACK NOTE (2026-07-08). The current / `main` track is Q4_K-native** (GGML Q4_K,
> targeting `unsloth/GLM-5.2-GGUF`). **FP8 is the PRIOR / PRESERVED track** on tag **`fp8-verified-baseline`**
> (tag `fp8-verified-baseline`), removed from `main` in commit `cbef69d`. Coverage here is
> **Verilator structural (line/toggle/branch) only — explicitly NOT a substitute** for the
> functional fidelity suite, and it measures **format-agnostic leaf/unit modules**, not the
> Q4_K numeric core or the assembled model. The large-integration `*_fp8` modules named in the
> out-of-scope list below are the prior track; the Q4_K equivalents are `glm_model_q4k`,
> `glm_decoder_block_q4k`, `mla_attn_q4k`, `glm_q4k_system*`, `glm_q4k_soc_ms`.

Structural code-coverage measurement of the GLM-5.2 **Q4_K** accelerator RTL, using
**Verilator 5.048 `--coverage-line --coverage-toggle`** (line + toggle + the
implied branch metric) driven by the project's existing behavioral testbenches.
Run with **`make coverage`**.

The primary simulator for this project is **iverilog 13.0** (every TB is an
iverilog-style `$display`/`initial` behavioral TB, and the fidelity gate is the
iverilog suite — `make unittests` + `make q4k` + `make formal`, all in `make all`
= 42 tests, 0 fail). iverilog has no built-in coverage instrumentation, so
coverage is measured with Verilator's `--binary` flow, which compiles the same
SystemVerilog testbench + RTL into a native sim that emits a `coverage.dat`
database. Only the subset of TBs that verilate **and** run cleanly under Verilator
is measured; the rest are documented as out-of-scope below (with the exact reason).

> **What this coverage does and does NOT reach.** The measured set is the fast
> **format-agnostic leaf/unit modules** (activation, softmax, RoPE, sampler,
> clock-gating, ECC, MBIST, reset-sync) that are shared by both tracks. It does
> **not** include: (a) the Q4_K numeric core (`glm_matmul_q4k`, `swiglu_expert_q4k`,
> `moe_router_q4k`, `q4k.vh` primitives) — those have **functional** gated TBs under
> `make q4k`, but a Verilator **structural** run of them is **PENDING** (the driver
> still lists the deleted FP8 matmul entry; see below); (b) the assembled model or
> whole chip — those are **elaboration-only** (see Scope). So this report is a
> structural-exercise complement to the fidelity suite, scoped to the leaves.

## Method

For each measured module, `make coverage` (driver: `tools/cov_run.sh`) does:

1. **Verilate** the TB + its RTL into a runnable binary:
   ```
   verilator --binary --coverage-line --coverage-toggle -Isrc --timing \
             --top-module <tb> <tb.v> <rtl srcs...>
   ```
2. **Run** the resulting `V<tb>` from the repo root, directing its coverage
   database to a per-module file with the runtime plusarg
   `+verilator+coverage+file+build/cov/<mod>/coverage.dat`. The TB must still
   print its usual `ALL <N> TESTS PASSED` (coverage is only counted for a
   passing functional run).
3. **Score the module's own source**: the `coverage.dat` records every point
   with its source filename, so the database is filtered to the module's
   *primary* source file (e.g. `src/glm_softmax.v`) and read with
   `verilator_coverage`, which prints `line` / `toggle` / `branch` covered/total.
4. **Merge** every per-module database into `build/cov/merged.dat`
   (`verilator_coverage --write`); the merged design-source total (all `src/`
   points, testbench points excluded) is reported at the bottom.

All build/run artifacts land in `build/cov/` (gitignored). The per-module and
merged summaries are printed to the terminal and saved to `build/cov/summary.txt`.

> **What "line" means here.** Verilator counts coverage *points* (basic-block
> statement groups and branch arms), not raw source lines — continuous-assign
> combinational lines are largely folded — so a small combinational module can
> have very few line points (e.g. 2–5) at 100 %. "toggle" is 0→1 and 1→0
> transitions on nets/registers of the module; a pure-function `.vh` header (only
> functions/localparams, e.g. `q4k.vh`) has no persistent nets, hence `0/0`
> toggle (`n/a`).

## Per-module coverage (last `make coverage` snapshot)

Coverage of each module's **own** primary source file, under its unit TB. All
modules below are **format-agnostic** (shared by both the Q4_K and prior FP8
tracks) and still present in the tree; these numbers are reproducible via
`make coverage` once the driver is repointed off the three removed entries noted
after the table.

| Module | Line % | Toggle % | Branch % | TB |
|---|---|---|---|---|
| `glm_act`              | 96.9 (63/65)    | 82.1 (7856/9572)  | 85.2 (46/54)  | `glm_act_tb` |
| `glm_softmax`          | 94.4 (34/36)    | 86.0 (3001/3490)  | 94.2 (98/104) | `glm_softmax_tb` |
| `rope_interleave_unit` | 97.4 (150/154)  | 95.9 (4148/4326)  | 90.6 (58/64)  | `rope_interleave_unit_tb` |
| `sampler`              | 76.0 (19/25)    | 41.0 (618/1508)   | 95.0 (38/40)  | `sampler_tb` |
| `clk_en_ctrl`          | 100.0 (5/5)     | 41.9 (250/596)    | 100.0 (4/4)   | `clk_en_ctrl_tb` |
| `clk_throttle`         | 100.0 (5/5)     | 42.9 (18/42)      | n/a (0/0)     | `clk_throttle_tb` |
| `icg_cell`             | 100.0 (2/2)     | 92.9 (26/28)      | 75.0 (6/8)    | `icg_cell_tb` |
| `ecc_secded`           | 84.6 (11/13)    | 99.9 (991/992)    | 93.8 (15/16)  | `ecc_secded_tb` (exhaustive SECDED) |
| `reset_sync`           | 100.0 (2/2)     | 100.0 (22/22)     | 100.0 (4/4)   | `reset_sync_tb` |
| `mbist_ctrl`           | 100.0 (6/6)     | 92.1 (116/126)    | 100.0 (14/14) | `mbist_ctrl_tb` |
| `ecc_mem_wrap`         | 42.9 (3/7)      | 71.5 (1029/1440)  | 87.5 (14/16)  | `ecc_mem_wrap_tb` |
| `kv_ecc_ring`          | 60.0 (6/10)     | 65.8 (1597/2426)  | 83.3 (10/12)  | `kv_ecc_ring_tb` |

> **`ecc_mem_wrap` re-measure PENDING.** `src/ecc_mem_wrap.v` and its TB were
> modified on the current branch after this snapshot; the row above is the last
> measured value and may shift on the next `make coverage`.

### Removed from the measured set (prior FP8 track — do NOT reproduce as-is)

Three modules that were in the earlier 15-module run are **gone from `main`** and
their coverage no longer reproduces. The `tools/cov_run.sh` working set still
lists the first two and must be repointed to the Q4_K core (that re-measurement is
**PENDING** — no fabricated Q4_K number is substituted here):

- **`fp8_e4m3` (`.vh`)** — the FP8 E4M3 codec. Deleted with the FP8 track
  (`src/fp8_e4m3.vh` + `test/fp8_e4m3_tb.v` removed). The Q4_K path uses `q4k.vh`
  instead; its primitives are checked **functionally** by `q4k_prim` (18/18, bit-exact
  to the ggml Q4_K reference) under `make q4k`, **not** by a Verilator structural run.
- **`glm_matmul_fp8`** — deleted; the Q4_K equivalent is **`glm_matmul_q4k`**
  (`src/glm_matmul_q4k.v`, checked bit-exact to the ggml `dequantize_row_q4_K`
  reference, 160/160, under `make q4k`). A Verilator **structural** coverage run of
  `glm_matmul_q4k` has **not** been done — `cov_run.sh` still references the deleted
  `glm_matmul_fp8` source and needs repointing. **[PENDING]**
- **`weight_decomp`** — its TB (`test/weight_decomp_tb.v`) and the Python FP8
  vector generator it consumed (`tools/fp8_gen.py`) were both removed; the module is
  no longer wired into any Q4_K build. Prior-track; not re-measured.

> **No merged total is reported for the current tree.** The earlier
> "554/631 line (87.8 %), 44 906/56 078 toggle (80.1 %), 642/722 branch (88.9 %)"
> figure was a **15-module** merge that **included** the three removed entries above
> (and merged their shared sources), so it does not describe the current Q4_K tree
> and is not reproducible. A clean merged total over the surviving 12 leaves (plus
> the Q4_K core, once added to the driver) is **PENDING** and is intentionally **not**
> recomputed by hand here (Verilator's cross-module shared-source de-duplication
> makes a hand-summed total wrong).

### Notes on the lower numbers (honest reading)

These are real, not hidden:

- **`ecc_mem_wrap` 42.9 % line / `kv_ecc_ring` 60 % line** — the uncovered
  points are the error-injection / uncorrectable-double-bit and init/scrub
  branches of the ECC RAM/ring that *this* unit TB does not drive on the primary
  wrapper source (the SECDED codec `ecc_secded` beneath them is exercised hard,
  99.9 % toggle). Exhaustive single-correct/double-detect fault coverage of these
  wrappers is proven functionally by their iverilog TBs and formally by
  `make formal` (`kv_cache_pager(ECC)` BMC).
- **`sampler` 41 % toggle / 76 % line** — the TB exercises one temperature +
  top-k/top-p configuration slice; the wide unused sampler datapath (full logit
  width, alternate sampling modes) never toggles.
- **`clk_throttle` / `clk_en_ctrl` 42–43 % toggle** — small control blocks whose
  wide counter/config buses are only partially swept by the directed TB.
- **`clk_throttle` branch `n/a`** — Verilator folds the prescaler's counter
  compare to no branch points.

## Reproduce

```
make coverage
```

Prerequisites: `verilator` (5.x) + `verilator_coverage` on `PATH`. Output:

- `build/cov/summary.txt` — the per-module + merged table.
- `build/cov/<module>/coverage.dat` — per-module raw database.
- `build/cov/merged.dat` — merged database of all measured modules.

> **Driver caveat.** `tools/cov_run.sh` currently still lists the three removed
> entries (`fp8_e4m3`, `glm_matmul_fp8`, `weight_decomp`); on the current tree
> those will report `BUILD-FAIL` (deleted sources/TBs) until the working set is
> repointed to the Q4_K core (`glm_matmul_q4k`, and optionally the `q4k.vh`
> primitives). The 12 surviving leaves above measure cleanly.

Annotated source (every line/branch tagged with its hit count; `%000000` marks
uncovered points) — run from the repo root so the source files resolve:

```
verilator_coverage --annotate build/cov/annotated --annotate-min 1 build/cov/merged.dat
```

## Scope — what is and is NOT covered (honest bounds)

**This is structural (line/toggle/branch) coverage of the committed *slice*
configuration** — the small parameterization the unit TBs instantiate. It is
**not**:

- **Functional coverage** — it does not assert that meaningful *scenarios* (all
  rounding modes, all corner FP values, every routing pattern) were hit; it only
  measures which RTL lines/toggles/branches were structurally exercised. The
  *functional* fidelity gate is the separate iverilog suite: `make unittests`,
  `make q4k` (the Q4_K unit gate — `q4k_prim` / `glm_matmul_q4k` bit-exact to the
  **ggml Q4_K reference** `tools/q4k_ref.py`, with `swiglu_expert_q4k` / `moe_router_q4k`
  checked functionally, not against a numeric golden), and `make formal`. That
  fidelity spans the **leaf/unit** level and — since the `make model-q4k` gate closed
  the gap — the **assembled** forward pass (`glm_model_q4k` **bit-exact vs the assembled
  numpy golden** `tools/glm_model_q4k_ref.py`, **1155/1155**; see the bf16-twin caveat
  update below).
- **Q4_K numeric-core coverage** — `glm_matmul_q4k`, `swiglu_expert_q4k`,
  `moe_router_q4k`, and the `q4k.vh` primitives have gated **functional** TBs
  (`make q4k`) but are **not yet** in this Verilator structural set. **[PENDING]**
- **Full-config coverage** — the production top runs at larger widths/depths
  (`PE_N`, `DDR_NCH`, `KV_RESIDENT`, layer count, `S_MAX`, expert count) than the
  measured slice. The slice→full-config relationship is checked by **elaboration
  only (no full-config sim)**: `make synth-glm` (yosys `hierarchy -check` +
  `check -assert` on the whole product top `glm_q4k_system_cdc`) and the
  `iverilog -tnull` / `verilator --lint-only` study in
  [`docs/FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md). There is **no** byte-identical
  token or functional sim at full config.
- **Whole-datapath / capstone coverage** — the large integration modules
  (`glm_model_q4k`, `glm_decoder_block_q4k`, `mla_attn_q4k`, `glm_q4k_system*`,
  `glm_q4k_soc_ms`, `spec_*`) are intentionally out of this run (minutes-long). The
  assembled Q4_K path now **does** have a gated functional sim (`make model-q4k`,
  1155/1155 vs the assembled numpy golden), but it is iverilog-gated and not
  instrumented in this Verilator structural set (the prior FP8 `batched_moe` `bcov`
  lives only on tag `fp8-verified-baseline`; `batched_moe` was folded inline into
  `glm_decoder_block_q4k` on `main`). Coverage here targets the fast leaf/unit
  modules.

> **bf16-twin caveat (updated — the functional half of the gap is since closed).** The
> model/decoder/attention/MTP-level testbenches (`glm_model_tb`,
> `glm_decoder_block_tb`, `mla_attn_tb`, `mtp_head_tb`) compile the **generic bf16
> twin** RTL — `src/glm_model.v`, `src/glm_decoder_block.v`, `src/mla_attn.v`,
> `src/swiglu_expert.v`, `src/moe_router.v`, `src/mtp_head.v` (**zero Q4_K**) — **not**
> the `_q4k` product modules, so **no structural coverage** exists here for the
> assembled Q4_K numeric path. The **functional golden**, however, now exists: the
> assembled `glm_model_q4k` full forward is gated **bit-exact vs the assembled numpy
> golden** (`make model-q4k` **1155/1155** + `model-q4k-acthw` 1155/1155,
> `tools/glm_model_q4k_ref.py`). The bit-exactness proven is Q4_K vs the team's own
> **ggml reimplementation** (`tools/q4k_ref.py` / `glm_model_q4k_ref.py`) at the
> **leaf** level (`make q4k`) and the **assembled** level (`make model-q4k`) — never
> the real published GGUF file / `llama.cpp`, which remains the open gap.

### Testbenches that do NOT verilate/run (skipped, with reason)

Not hacked or modified — skipped and recorded. These all pass under the primary
iverilog sim; the issue is Verilator-specific. All are format-agnostic support
modules (FP pipeline / matmul-pipe / top-k / indexer), unchanged by the FP8→Q4_K
migration.

| Module TB | Why skipped |
|---|---|
| `glm_fp_pipe_tb`     | The TB's bit-exact self-check builds its `exp` golden with the Verilog **`$exp`** real-number task; Verilator evaluates `$exp`/`real` differently than iverilog at underflow/large-magnitude inputs, so the golden diverges (measured exp rel-err up to 1.0) and the TB `$fatal`s under Verilator. Not a DUT bug — passes under iverilog. |
| `rmsnorm_unit_tb`    | Same class: the golden uses **`$sqrt`** (real) for the RMS reciprocal-norm reference; Verilator's `$sqrt`/`real` semantics diverge → 99 263 element mismatches and a `$fatal` under Verilator (byte-exact under iverilog). |
| `glm_matmul_pipe_tb` | Bit-exact FP self-check fails under Verilator (2595 tile mismatches, worst **relerr = 0** — i.e. numerically equal but differing bit patterns: `-0`/NaN/round-tie canonicalization Verilator resolves differently). Passes under iverilog. |
| `topk_select_tb`     | Uses **`process::self()`** + event controls that *require* `--timing`; Verilator 5.048's `--timing` code generator then emits broken C++ for it (`error: use of undeclared identifier '__Vfork_10__sync'`). Build fails — a Verilator codegen defect, not fixable without editing the TB. |
| `dsa_indexer_tb`     | Same `--timing` `__Vfork` codegen defect (`'__Vfork_4__sync'` undeclared); the TB uses the same process/event constructs. Build fails. |

**Bottom line.** Verilator gives real, reproducible **structural** coverage for
**12 format-agnostic leaf/unit modules**. Three earlier entries (`fp8_e4m3`,
`glm_matmul_fp8`, `weight_decomp`) were removed with the FP8 track and are **not**
re-measured; a structural run of the Q4_K numeric core (`glm_matmul_q4k` et al.)
and a fresh merged total are **PENDING**. The two skipped *build* failures are a
Verilator `--timing` codegen defect; the three skipped *run* failures are
Verilator `real`/`$exp`/`$sqrt`/bit-canonicalization disagreements with the
reference iverilog goldens. Correctness/fidelity is owned by the iverilog suite
(`make q4k` at the leaf level, `make model-q4k` at the assembled level — 1155/1155
vs the assembled numpy golden) and the formal proofs; this coverage report is a
structural-exercise complement to them. The assembled Q4_K path stays outside this
*structural* set, and bit-exactness to the real GGUF / llama.cpp stays open.
