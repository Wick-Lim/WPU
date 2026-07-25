# Q4_K system retarget — the NON-trivial work plan

*Scope: the retarget work that is **not** simple leaf-rewiring. The leaf operators
(`swiglu_expert_q4k`, `moe_router_q4k`, `mla_attn_q4k`) and the orchestration layer
(`glm_model_q4k`, `glm_decoder_block_q4k`, `mtp_head_q4k`) already exist and pass. What
remains is: (1) the four **system tops**, (2) the **weight path** (loader / packer /
cache / fabric sizing + per-tensor type map), (3) **FP8 removal** from `main`, and
(4) the **Makefile** Q4 gate. This document enumerates each precisely.*

*STATUS (2026-07): all four have since **landed on `main`** — the `glm_q4k_*` tops exist
(`make synth-glm` signs off `-top glm_q4k_system_cdc`), the Q4_K weight path **including
the mixed-type Q6_K/Q8_0/F16 consumers** is in and gated (`make q4k`, `make mixedtype`),
FP8 is **removed from `main`** (§3 executed), and the Makefile gate is wired — plus the
assembled full-forward numeric goldens `make model-q4k` / `make model-q4k-acthw`
(ALL 1155 TESTS bit-exact vs `tools/glm_model_q4k_ref.py`). Still OPEN: bit-exactness vs
the real GGUF bytes / llama.cpp runtime. The sections below stand as the executed-plan
record, with per-section (DONE) notes.*

Companion: `docs/Q4K_RETARGET.md` (numerics + leaf/operator status, already ✅). The FP8
baseline is preserved on tag `fp8-verified-baseline`; the §3 deletion
has since been executed on `main`.

---

## 0. The one contract change that drives everything

Every retarget below is a consequence of a **single weight-bus swap** at the compute-die
boundary. `glm_model_fp8` pulls FP8; `glm_model_q4k` pulls Q4_K. The die ports already exist
and differ exactly as:

| family | FP8 (`glm_model_fp8`) | Q4_K (`glm_model_q4k`) |
|---|---|---|
| attn | `aw_col [PE_N*8]`, `aw_scale [16*PE_N*A_NB]` | `aw_q [PE_N*4]`, `aw_d [16*PE_N*A_NSB]`, `aw_dmin [16*PE_N*A_NSB]`, `aw_scales [96*PE_N*A_NSB]` |
| router | `rw_col [8*N_EXPERT]`, `rw_scale [16*N_EXPERT*R_NB]` | `rw_q [4*N_EXPERT]`, `rw_d`, `rw_dmin`, `rw_scales [96*N_EXPERT*R_NSB]` |
| FFN | `fw_col [8*TN]`, `fw_col_up [8*TN]`, `fw_scale_g`, `fw_scale_u` | `fw_q [4*TN]`, `fw_q_up [4*TN]`, `fw_d_g/fw_dmin_g/fw_scales_g`, `fw_d_u/fw_dmin_u/fw_scales_u` |
| bf16 tail (embed `em_*`, norms `gn_*`/`fn_*`, KV `kc_*`, LM head `lw_*`) | **unchanged** | **unchanged** |

Two structural facts fall out of this table:

- **`A_NB → A_NSB`**: FP8 counts 128-wide K-*blocks* (`A_NB = ceil(K/128)`); Q4_K counts
  256-wide *super-blocks* (`A_NSB = ceil(K/256)`). Same for `R_NB→R_NSB`, `FF_NB_D→FF_NSB_D`.
  These derived params must be recomputed in every top that forwards the bus.
- **The scale bus triples**: one bf16 block-scale word per column becomes three fields
  (`d` fp16, `dmin` fp16, `scales` 96-bit) per column per super-block.

Everything else in the memory system (`expert_cache_pf`, `kv_cache_pager`, `ddr5_xbar`,
the FIFO/arbiter/CDC logic) is **byte-agnostic** — it moves addresses/slots/IDs, never
weight bytes — so it retargets by **parameter/doc only**, not logic.

---

## 1. System tops — what each must move to Q4_K

Four tops, in dependency order. Each is a **copy → swap the die instance → swap the
forwarded weight ports → recompute the `_NB→_NSB` derived params**. No datapath math is
touched. Recommended new names: `glm_q4k_soc.v`, `glm_q4k_soc_ms.v`, `glm_q4k_system.v`,
`glm_q4k_system_cdc.v` (siblings, so the FP8 tops stay buildable until §3 deletes them).
**(DONE — all four `glm_q4k_*` tops landed under these names; §3 has since deleted the
FP8 siblings. Gated by `synth-glm` elaboration + Verilator lint; dedicated top TBs are
still open.)**

### 1.1 `glm_fp8_soc.v` → `glm_q4k_soc.v` (548 lines)
The minimal SoC: die + `expert_cache_pf` + `kv_cache_pager` + single Flash arbiter.

- **Instance to swap:** `u_model : glm_model_fp8` → `glm_model_q4k` (§1, line ~280).
- **Weight buses to re-port** (top-level inputs + the `.aw_col(...)` etc. connections at
  lines ~296–299 and the port list at lines 191–208):
  - `aw_col/aw_scale` → `aw_q/aw_d/aw_dmin/aw_scales`
  - `rw_col/rw_scale` → `rw_q/rw_d/rw_dmin/rw_scales`
  - `fw_col/fw_col_up/fw_scale_g/fw_scale_u` → `fw_q/fw_q_up/fw_d_g/fw_dmin_g/fw_scales_g/fw_d_u/fw_dmin_u/fw_scales_u`
- **Derived params:** `A_NB/R_NB/FF_NB_D` (block counts) → `A_NSB/R_NSB/FF_NSB_D`
  (super-block counts, `ceil(K/256)`), mirroring `glm_model_q4k`.
- **Stays bf16 / unchanged:** the HOST FSM, `em_*/gn_*/fn_*/lw_*` pulls, `kc_*` KV read
  (KV latent rows are bf16, **not** quantized — `ROW_BITS=(KV_LORA+ROPE)*16` is invariant),
  the routed-expert episode-detect→FIFO→`expert_cache_pf`, `kv_cache_pager`, the Flash arbiter.
- **`expert_cache_pf` / `kv_cache_pager`:** **no logic change** — both are tag/slot/ID
  controllers. Only the doc comment ("GDDR6 expert-cache slot holds one expert's Q4_K
  weights, ~44% fewer bytes") updates.

### 1.2 `glm_fp8_soc_ms.v` → `glm_q4k_soc_ms.v` (523 lines)
Batched multi-sequence SoC: `glm_model_fp8` at `PE_M=B`, per-row KV via `kv_cache_pager`
`NSEQ=B`, plus `expert_cache_pf`.

- **Identical swap** to §1.1 (`u_model` die + the same three weight-bus families + `_NB→_NSB`).
- The PE_M batching, per-row `pos_vec/s_len_vec/seq_vec`, and NSEQ pager wiring are **weight-
  format-independent** — carried through unchanged (they already exist on `glm_model_q4k`).

### 1.3 `glm_fp8_system.v` → `glm_q4k_system.v` (1056 lines) — the largest
The production single-module system: die + `expert_cache_pf` + `kv_cache_pager` +
**`ddr5_xbar`** (multichannel fast tier) + **`weight_loader`** (weight-pull DMA) +
optional `weight_decomp`.

- **Die swap + weight-bus re-port + `_NB→_NSB`:** as §1.1.
- **`weight_loader` instance (§7, `u_loader`, lines ~668–681):** this is the one place the
  fabric touches weight *bytes*, so it changes with the loader rewrite in §2.1. The
  descriptor (`desc_base/desc_klen/desc_nblk`) and the `mm_*` drive change from the FP8
  `[128,128]`-block form to the Q4_K super-block form (`mm_w_scale` → `mm_w_d/mm_w_dmin/
  mm_w_scales`, and the weight-row lanes go 8-bit → 4-bit).
- **`ddr5_xbar` (§8, `u_xbar`):** **byte-agnostic.** `DATA_W=256` beat, the LOAD/SLOT/HOT
  priority issuer, `bank_rot` striping, tags — all unchanged. The addresses it stripes are
  now Q4_K byte-offsets (smaller image), a **doc-only** change.
- **C8 LOOPBACK generate (§9, `LOOPBACK==1`)**: this feeds `xbar_resp_data[PE_N*8-1:0]` back
  into the die's `aw_col`. For Q4_K it must feed `aw_q` instead — the returned lanes go
  **8-bit → 4-bit** (`die_aw_q <= xbar_resp_data[PE_N*4-1:0]`), and the loopback address
  encoding (`cur_addr[3:0]=aw_sel`, `aw_k`, `aw_grp`, `db_layer`) is unchanged. This is the
  **only non-mechanical edit** in the file (a width change on the staged lane register
  `lb_col_q`, and the `die_aw_*` net). LOOPBACK==0 (default) stays byte-identical.
- **`weight_decomp` path (DECOMP≥1, §7b):** the Huffman byte-stream reconstructor is
  **format-agnostic** (it decodes bytes into `WL_DATA_W` words). It works on the Q4_K image
  as-is; the compressed backing image is just the Q4_K byte stream. No logic change; the
  Huffman tables are re-fit to the Q4_K byte statistics (a `tools/` regen, not RTL).
- **Stays bf16 / unchanged:** everything in §1.1's "stays" list, plus the DDR5 fabric,
  loopback FSM control, and the loader FSM skeleton (only its byte layout moves — §2.1).

### 1.4 `glm_fp8_system_cdc.v` → `glm_q4k_system_cdc.v` (571 lines) — the synth top
Two-clock CDC wrapper. `make synth-glm` elaborates **this** as the whole-chip sign-off top
(was `hierarchy -top glm_fp8_system_cdc`; DONE — now `-top glm_q4k_system_cdc`).

- **Instance to swap:** `u_core : glm_fp8_system` → `glm_q4k_system` (line ~399).
- **Weight-bus re-port + `_NB→_NSB`:** as §1.1 — the CDC wrapper just forwards the same
  buses across the async-FIFO boundary; the FIFO widths that carry weight lanes shrink
  (8→4-bit code fields) and the scale-carrying FIFOs change shape (`scale` → `d/dmin/scales`).
- **Everything CDC (the async FIFOs, the two-clock handshake, reset sync) is format-
  agnostic** — retargets by width parameter only.
- **`synth-glm` (Makefile) retargets its `-top` to `glm_q4k_system_cdc`** and its
  `$(GLM_CDC_SRCS)` file list to the `*_q4k.v` sources (see §4).

---

## 2. Weight path — FP8 `[128,128]` block layout → GGUF Q4_K super-block layout

The whole point of the local-device target: the published `unsloth/GLM-5.2-GGUF :
UD-Q4_K_XL` (**467 GB, ~44.6% fewer bytes/weight**, ~38% smaller overall than the 753 GB
FP8 checkpoint) runs with **no re-quantization**.

**Footprint math (measured):** FP8 = 1 byte/weight + one bf16 (2 B) scale per 128 weights =
**1.0156 B/weight**. Q4_K = one 144-byte super-block per 256 weights = **0.5625 B/weight**
(`d` 2 + `dmin` 2 + `scales` 12 + `qs` 128 = 144). Ratio **0.554 → 44.6% smaller per Q4_K
weight**; blended with the Q6_K/Q8_0/F16 sensitive-tensor tail, the whole checkpoint lands
at 467/753 = **62% → ~38% smaller**.

### 2.1 `weight_loader.v` — the byte-layout rewrite (the real work) — DONE (`src/weight_loader_q4k.v` + `test/weight_loader_q4k_tb.v`)
Current loader (265 lines) reads a **word-addressed FP8 tile**: a SCALE region
(`nblk*PE_N` bf16 words, one per (K-block, column)) then a CODE region (`k_len` words, each
`word[8*pj+:8]=W[k][pj]`, one FP8 byte/column). It drives `glm_matmul_fp8`'s pull
(`mm_start`, `mm_w_scale`, `mm_w_row`, `mm_in_valid`).

Retarget → `weight_loader_q4k.v` driving `glm_matmul_q4k`'s pull:

- **Descriptor:** `{base, k_len, nblk}` → `{base, k_len, n_sblk}` where `n_sblk =
  ceil(K/256)` super-blocks (was 128-blocks).
- **Per-column super-block header** (read once at `start`, replaces the bf16 SCALE region):
  for each column `pj` and super-block `sb`, read `d` (fp16), `dmin` (fp16), and `scales`
  (96 bits = 12 bytes). Assemble into the `mm_w_d [16*PE_N*NSB]`, `mm_w_dmin [16*PE_N*NSB]`,
  `mm_w_scales [96*PE_N*NSB]` buses `glm_matmul_q4k` latches at `start`.
- **CODE region:** each K-beat carries a **4-bit** code per column (`mm_w_q [4*PE_N]`),
  packed 2 codes/byte, vs the FP8 8-bit `mm_w_row`. The stream FSM (`S_SCALE→S_START→
  S_STREAM→S_DONE`, latency-1 capture pipeline) is **kept verbatim**; only the field widths
  and the header parse change. Note the GGUF on-disk order is `qs[128]` = 4-bit nibbles for
  all 256 weights of a super-block — the loader unpacks nibble `k` of column `pj` per beat.
- **`DATA_W`:** stays `256` (a Q4_K super-block is 144 B = fits comfortably; wide enough to
  stream the header + a nibble beat). This aligns with the `ddr5_xbar` beat width.
- **Verification:** a `weight_loader_q4k_tb` mirroring the existing `weight_loader_tb`,
  driving `glm_matmul_q4k` and checking the streamed result bit-exactly against
  `tools/q4k_ref.py` (the same golden the leaf TBs already use).

### 2.2 `tools/ckpt_pack.py` — the memory-image / GGUF packer (rewrite) — landed as `tools/ckpt_pack_q4k.py`
Current packer (419 lines) parses HF `zai-org/GLM-5.2-FP8` (safetensors: `.weight`
F8_E4M3 + `.weight_scale_inv` F32/BF16 + bf16 tail) and emits the FP8 `weight_mem.hex`
that `weight_loader.v` reads. It imports `glm_fp8_ref` for `fp32_to_bf16`.

Retarget → `tools/ckpt_pack_q4k.py` (or a `--format q4k` mode):

- **Input:** parse the **GGUF** container (`unsloth/GLM-5.2-GGUF`) instead of HF
  safetensors. GGUF = magic + KV-metadata header + tensor-info table (name, dims, **ggml
  type enum**, offset) + aligned tensor blob. The per-tensor **type enum is the dynamic
  type map** (§2.5): `GGML_TYPE_Q4_K=12`, `Q6_K=14`, `Q8_0=8`, `F16=1`.
- **Per-type block emit** (bit-exact to `tools/q4k_ref.py`, which already dequants all three):
  - **Q4_K** — 144 B/256: emit `d`, `dmin`, `scales[12]`, `qs[128]` into the loader's
    super-block header + nibble-code layout (§2.1).
  - **Q6_K** — 210 B/256: `d` fp16 + `ql`/`qh` (6-bit) + `int8 scales[16]`.
  - **Q8_0** — 34 B/32: `d` fp16 + 32 `int8`.
  - **F16** — passthrough.
- **Image:** replace `fp8_bytes_to_codes` / `scale_tensor_to_bf16` / `pack_weight`
  (all FP8-block-specific) with per-super-block packers. `import glm_fp8_ref` → `import
  q4k_ref` for the fp16/dequant mirrors.
- **Round-trip self-test:** the existing `pack → unpack == original` harness is kept, now
  asserting Q4_K codes + fp16 `d/dmin` + 6-bit `scales` survive the image round-trip.
- **Footprint assertion:** the packer prints total bytes; assert the blended image is
  ~38% smaller than the FP8 image for the same tensor set (a regression guard on the moat).

### 2.3 `expert_cache_pf.v` — parameter/doc only (no logic)
The cache tracks **which expert id is resident in which slot** (valid/tag/rank/freq arrays,
LFU+LRU eviction). It **never touches weight bytes** — the bytes are served by the
Flash/DDR stub. Change: doc comment (a resident slot now holds one expert's **Q4_K**
weights, ~44% fewer bytes → **more experts fit per GDDR6 GB** → the `SLOTS` budget for a
fixed cache size roughly doubles). Optionally bump the default `SLOTS` to reflect the
Q4_K density. **No RTL logic edit.**

### 2.4 `ddr5_xbar.v` — parameter/doc only (no logic)
Byte-agnostic banked read fabric: presents a block address, banks it `N_CH` ways, returns
`DATA_W=256` beats. **No logic change.** The addresses are Q4_K byte-offsets (smaller
image → fewer beats per tile/expert → the same aggregate BW moves more tokens/s). If a
Q4_K super-block (144 B) is chosen as the native burst, `DATA_W` can stay 256 (two beats)
or widen — a tuning param, not a correctness change. Update the header BW commentary.

### 2.5 Per-tensor dynamic type map (Q4_K / Q6_K / Q8_0 / F16)
UD-Q4_K_XL is a **dynamic mix**: most tensors Q4_K, sensitive ones (some attention/output
projections, embeddings) kept higher-precision. The numerics for all four types are
**already proven** (`q4k.vh` + `q4k_ref.py`, see `docs/Q4K_RETARGET.md` table). What's new
at the system level is **routing the type per tensor**:

- **Source of truth:** the GGUF tensor-info table's ggml-type enum per tensor (read by the
  §2.2 packer). The packer emits, alongside `weight_mem.hex`, a **per-tensor type manifest**
  (`name → {type, base, k_len, n_sblk}`).
- **RTL selection (DONE — option (b) landed):** `glm_matmul_q4k` originally assumed Q4_K.
  For the mixed model either
  (a) keep a single Q4_K core and pre-dequant Q6_K/Q8_0/F16 tensors to the Q4_K bus in the
  loader, or (b) add a small per-tile `w_type` input that switches the dequant function
  (all four dequants are combinational `function automatic` in `q4k.vh`). Option (b)
  preserves each sensitive tensor at its native GGUF type (no lossy re-quant to Q4_K),
  keeping the loaded weight image byte-faithful to the checkpoint — note this is
  **weight-image fidelity, not inference bit-exactness vs llama.cpp** (which remains OPEN:
  the RTL uses bf16 acts + fp32 accumulate, a different arithmetic contract). The
  mixed-type consumer is **DONE**: `src/q4k_mixed.vh` dequant primitives, per-column
  `w_type` routing in `src/glm_matmul_q4k.v`, `desc_wtype` in `src/weight_loader_q4k.v` —
  gated by `make mixedtype` (`q6k_prim`, `q8_0_prim`, `glm_matmul_mixed` 32/32,
  `weight_loader_q4k_mixed` 192/192 incl. a 24-tile mixed sequence), bit-exact to the same
  `tools/q4k_ref.py` reimpl golden (still not bit-verified vs the real GGUF bytes). The
  type comes from the §2.2 manifest, latched per tile at `start`.

---

## 3. FP8 removal from `main` (DONE — executed; preserved on tag `fp8-verified-baseline`)

Once the Q4_K datapath is complete (§1–§2 landed + gated green), delete the FP8 track from
`main` in one commit. **(DONE — this removal has been executed on `main`; the list below
is the record of what was removed.)** **All of this is preserved on tag
`fp8-verified-baseline`.** Exact list:

**`src/` (FP8 datapath — 9 files):**
```
src/fp8_e4m3.vh
src/glm_matmul_fp8.v
src/swiglu_expert_fp8.v
src/moe_router_fp8.v
src/mla_attn_fp8.v
src/glm_decoder_block_fp8.v
src/glm_model_fp8.v
src/mtp_head_fp8.v
src/glm_fp8_soc.v  src/glm_fp8_soc_ms.v  src/glm_fp8_system.v  src/glm_fp8_system_cdc.v   (→ replaced by *_q4k siblings)
```

**`test/` (FP8 TBs — 33 files):**
```
fp8_e4m3_tb.v  glm_matmul_fp8_tb.v
swiglu_expert_fp8_tb.v  swiglu_expert_fp8_pem_tb.v
moe_router_fp8_tb.v  moe_router_fp8_pem_tb.v
mla_attn_fp8_tb.v  mla_attn_fp8_pem_tb.v  mla_attn_fp8_ppos_tb.v  mla_attn_fp8_pslen_tb.v
  mla_attn_fp8_perrow_pos_tb.v  mla_attn_fp8_multiseq_tb.v  mla_attn_fp8_multiseq_dsareal_tb.v
  mla_attn_fp8_swin_decouple_tb.v  mla_attn_fp8_sparse_perrow_tb.v
glm_decoder_block_fp8_tb.v  glm_decoder_block_fp8_union_tb.v
glm_model_fp8_tb.v  glm_model_fp8_2x_tb.v  glm_model_fp8_dump_tb.v
  glm_model_fp8_pem_tb.v  glm_model_fp8_multiseq_tb.v  glm_model_fp8_multiseq4_tb.v
mtp_head_fp8_tb.v  mtp_head_fp8_pem_tb.v
glm_fp8_soc_tb.v  glm_fp8_soc_ms_tb.v  glm_fp8_soc_ms_loop_tb.v
glm_fp8_system_tb.v  glm_fp8_system_cdc_tb.v  glm_fp8_system_perf_tb.v
  glm_fp8_system_loopback_tb.v  glm_fp8_system_decomp_tb.v
```

**`tools/` (FP8-only — 5 files):**
```
tools/fp8_gen.py  tools/fp8_ctxpack.py  tools/fp8_huff.py
tools/glm_fp8_ref.py  tools/glm_fp8_contract.py
```
*Caveat:* `tools/ckpt_pack.py` currently `import glm_fp8_ref` — its replacement
`ckpt_pack_q4k.py` (§2.2) must be landed **first** (imports `q4k_ref`) so removing
`glm_fp8_ref.py` doesn't break the packer. `test/full_config_elab_wrap.v` instances
`glm_model_fp8` — retarget it to `glm_model_q4k` before deletion.

**Makefile targets to delete** (§4 lists the survivors): every `*_fp8*` recipe in
`unittests`, plus `spec-slow`, `cache-study`, `bcov`, `bitacc`, the FP8 `synth-glm`
source list, and the `cdc` sim target's FP8 wiring. Retire `full_config_elab_wrap` FP8 refs.

**Keep (shared, format-agnostic — NOT FP8):** `glm_fp.vh`, `glm_fp_pipe.v`,
`glm_matmul.v/_pipe.v`, `glm_act.v`, `rmsnorm_unit.v`, `rope_interleave_unit.v`,
`glm_softmax.v`, `topk_select.v`, `dsa_indexer.v`, `sampler.v`, all memory-system RTL
(`ddr5_xbar`, `expert_cache_*`, `kv_cache_pager`, `weight_loader` → its q4k form, `flash_xbar`,
`cdc_async_fifo`, `reset_sync`, `ecc_*`, `mbist_ctrl`, clocking) and their TBs.

---

## 4. Makefile — the Q4 gate

**At planning time:** the Makefile had **zero** `q4k` targets; the four Q4_K unit TBs
(`q4k_prim`, `glm_matmul_q4k`, `swiglu_expert_q4k`, `moe_router_q4k`) existed and passed but
were **not wired into any gate**. `all: unittests synth-glm formal`, and `synth-glm`
elaborated `-top glm_fp8_system_cdc`. **(DONE — `q4k` is now wired into `unittests`, joined
by the `mixedtype` sub-gate and the assembled `model-q4k` / `model-q4k-acthw` full-forward
goldens, and `synth-glm` elaborates `-top glm_q4k_system_cdc` — the "After §3" shape in
§4.2 is the live state.)**

### 4.1 Add a `q4k` sub-gate (verified recipes — all four build & pass green today)
Add a `.PHONY: q4k` target, invoked by `unittests` (or run standalone). The exact recipes,
**confirmed working in this environment**:

```make
q4k:
	@mkdir -p $(BUILD_DIR)
	@# q4k_prim: fp16->fp32 + get_scale_min_k4 primitives (q4k.vh) vs ggml golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/q4k_prim_sim test/q4k_prim_tb.v
	@printf '[%s] ' "q4k_prim"; $(VVP) $(BUILD_DIR)/q4k_prim_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: q4k_prim"; exit 1; }
	@# glm_matmul_q4k: Q4_K GEMM core, bit-exact to ggml dequantize_row_q4_K.
	@python3 tools/q4k_matmul_gen.py >/dev/null            # -> build/q4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_q4k_sim test/glm_matmul_q4k_tb.v src/glm_matmul_q4k.v
	@printf '[%s] ' "glm_matmul_q4k"; $(VVP) $(BUILD_DIR)/glm_matmul_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_q4k"; exit 1; }
	@# swiglu_expert_q4k: MoE expert (gate/up/down + silu) on the Q4_K core.
	@python3 tools/swiglu_q4k_gen.py >/dev/null            # -> build/swiglu_q4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_q4k_sim test/swiglu_expert_q4k_tb.v \
	    src/swiglu_expert_q4k.v src/glm_matmul_q4k.v src/glm_act.v
	@printf '[%s] ' "swiglu_expert_q4k"; $(VVP) $(BUILD_DIR)/swiglu_expert_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_q4k"; exit 1; }
	@# moe_router_q4k: gating GEMV -> sigmoid -> top-K -> renorm on the Q4_K core.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_q4k_sim test/moe_router_q4k_tb.v \
	    src/moe_router_q4k.v src/glm_matmul_q4k.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_q4k"; $(VVP) $(BUILD_DIR)/moe_router_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_q4k"; exit 1; }
```

Expected green line (verified now): `q4k_prim` 18, `glm_matmul_q4k` 160, `swiglu_expert_q4k`
240, `moe_router_q4k` 40 TESTS PASSED. **Build-dep notes discovered while verifying:**
`moe_router_q4k` needs **`src/glm_fp_pipe.v`** (uses `fp32_add_pipe`/`fp32_rsqrt_pipe`);
`glm_matmul_q4k`/`swiglu_expert_q4k` do **not** (they use only the combinational `glm_fp.vh`
functions). `glm_matmul_q4k_tb` requires `build/q4k_vec.txt` (default gen dims PE_M=2/PE_N=2
match the TB); `swiglu_expert_q4k_tb` requires `build/swiglu_q4k_vec.txt`.

### 4.2 How `make all` looks for the Q4 track
Phased, so the FP8 gate keeps passing until §3:

- **During bring-up (both tracks live):** `unittests` runs the shared bf16 units + the FP8
  units + the new `q4k` sub-gate. Add a `q4k-system` synth target that elaborates
  `-top glm_q4k_system_cdc` (mirroring `synth-glm`) as each §1 top lands. `all` unchanged.
- **After §3 (FP8 deleted — the Q4 track *is* the product):**
  ```make
  all: unittests synth-glm formal          # unchanged phony names, Q4 contents
  unittests: <shared bf16 units> q4k <q4k orchestration/system TBs>
  synth-glm:   # -top glm_q4k_system_cdc, GLM_Q4K_CDC_SRCS = the *_q4k.v hierarchy
  ```
  i.e. `synth-glm` swaps `-top glm_fp8_system_cdc → glm_q4k_system_cdc` and its
  `$(GLM_CDC_SRCS)` list to the Q4_K sources; `formal` (memory-controller BMC) is
  format-agnostic and unchanged. Add the §1 top TBs (`glm_q4k_soc_tb`,
  `glm_q4k_system_cdc_tb`, …) and `weight_loader_q4k_tb` (§2.1) as they are written.

---

## 5. Ordered execution checklist

*(Status: steps 1, 5, 6 and 8 are DONE; steps 2–3 landed the four tops (their dedicated
top TBs remain open — the tops are gated by `synth-glm` elaboration + lint); step 4 landed
as `tools/ckpt_pack_q4k.py` with its synthetic-GGUF round-trip self-test. Still OPEN:
bit-exactness vs the real GGUF bytes / llama.cpp.)*

1. **`weight_loader_q4k.v` + `weight_loader_q4k_tb`** (§2.1) — the byte-layout keystone;
   verify bit-exact to `q4k_ref` driving `glm_matmul_q4k`.
2. **`glm_q4k_soc.v`** (§1.1) + TB — smallest top, proves the die-swap + bus re-port pattern.
3. **`glm_q4k_soc_ms.v`** (§1.2), **`glm_q4k_system.v`** (§1.3, incl. the LOOPBACK 8→4-bit
   edit), **`glm_q4k_system_cdc.v`** (§1.4).
4. **`tools/ckpt_pack_q4k.py`** (§2.2) — GGUF parse + per-type block emit + type manifest;
   round-trip + footprint self-test.
5. **Per-tensor type routing** (§2.5) — manifest → loader `w_type` mux (Q4_K/Q6_K/Q8_0/F16).
6. **Makefile `q4k` sub-gate** (§4.1) — wire in immediately (green today); retarget
   `synth-glm -top` as the tops land (§4.2).
7. **`expert_cache_pf` / `ddr5_xbar`** doc/param refresh (§2.3–2.4).
8. **FP8 removal** (§3) — only after 1–7 are gated green; preserved on tag `fp8-verified-baseline`.
9. **Docs/site** — footprint/BOM/tok/s + the moat row to the GGUF UD-Q4_K_XL basis
   (`docs/Q4K_RETARGET.md`, `ACCEL_GLM52.md`, `P2_MEMORY_MAP.md`).

**Design invariant throughout:** no arithmetic is reimplemented. Every system-top and
weight-path change is a **width/format/wiring** move against the already-proven
`glm_matmul_q4k` + `q4k.vh` numerics (bit-exact to ggml). The memory-system controllers
(`expert_cache_pf`, `kv_cache_pager`, `ddr5_xbar`, CDC FIFOs) are byte-agnostic and change
by parameter/doc only — the real code is `weight_loader_q4k.v` (§2.1) and
`ckpt_pack_q4k.py` (§2.2); everything else is mechanical.
