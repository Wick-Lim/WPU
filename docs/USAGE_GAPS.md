# Usage-Gap Register

**The gap between a verified accelerator and a usable device.**

> One-line truth: the RTL is verified; the software a user actually touches
> (host RAG / GUI / visualization / tuning / persistence, the USB-C transport,
> multi-context routing, and real provisioning) is largely **unbuilt**.
> *(Since the review: the prefix/KV cache and a v0.1 management console have
> landed — see the closed rows in §2/§3/§6.)*
> Several lifecycle and safety decisions are architecture — cheap to decide now,
> expensive to retrofit after board/boot-loader/protocol freeze.

This register turns the findings of a structured usage review into a durable
planning artifact. The review ran **63 agents** across the seven usage
dimensions a real owner passes through — provisioning, host software,
interactive session quality, multi-context, power/thermal/physical, the
flagship air-gapped RAG/GUI/viz workflow, and reliability/failure modes — and
**confirmed 49 problems** (37 high, 11 medium, 1 low) with reproducible
evidence.

This is **product-stage reality, not a regression.** The datapath, KV pager,
boot DMA engine, ECC/reset/MBIST building blocks, and formal properties are
strong and stay strong. What the review found is that the *experience layer*
wrapped around that silicon — the part a buyer plugs in and touches — has not
been built yet, and that a handful of decisions underneath it need to be locked
before hardware and protocol freeze. The public investor page already frames
this honestly; this document is the engineering-facing companion to that framing.

Scope note: many findings are the *same underlying gap* seen from different
usage angles (the ~70 s cold boot and the absent RAG/GUI each surfaced in four
or five dimensions). Those are **stated once and cross-referenced**, so the 49
raw findings collapse into a smaller set of distinct gaps below.

---

## LOCK IN NOW — architecture decisions, expensive later

Three decisions are not software backlog. They shape the board, the
boot-loader, and the USB wire format. If they are deferred past freeze, fixing
them means a hardware or protocol respin. Decide them before that freeze.

> **Progress (2026-07): all three now have a verified RTL/tool foundation**
> (parameter-gated, default-off, the default netlist proven byte-identical by
> yosys sequential equivalence — so nothing verified was disturbed):
> **§A** — `boot_loader` gained an `INTEGRITY` mode (magic/version/CRC manifest
> gate; fail-closed on truncated/wrong-version/bad-CRC/bad-magic — `done` never
> releases a bad model): `make boot-integrity` (3712 tests + equivalence). A real
> streaming provisioner, `tools/provision_image.py`, now packs a real GGUF into a
> binary block image + signed manifest (per-tensor sha256 + resident-hot/expert
> segment list a boot-loader can consume), proven on real GGUFs: `make
> provision-selftest`. *(Still open: the A/B dual-slot + atomic activate-pointer
> policy is a board/firmware decision on top of this foundation.)*
> **§B** — `weight_loader_q4k` gained a `WEIGHT_ECC` SECDED mode (single-bit
> corrected, double-bit flagged, corrected-error counter for scrub): `make
> weight-ecc` (+ equivalence). *(Still open: wiring the scrub loop + check-bit
> storage into the physical LPDDR array.)*
> **§C** — `glm_q4k_system_cdc` gained a `PROTO_CTX` mode carrying a
> context/sequence id end-to-end through the CDC FIFOs + a telemetry-counter
> readback: `make cdc-protocol` (+ equivalence). **Hardened 2026-07-22:** a
> mid-run `OP_TELEM` pop no longer retags/corrupts the in-flight run (`cur_ctx`
> follows the RUN, not the pop), and a launch arriving mid-run is **held, not
> dropped** (`launch_pend` parks exactly one; the FIFO buffers the rest) —
> proven by `make cdc-protocol`'s mid-run-telemetry + queued-launch tests
> (`test/cdc_protocol_ctx_tb.v` `midrun_telem`/`queued_launch`) plus a
> `-DINJECT_CTXTAG` must-fail build that re-introduces the pre-fix tagging and
> must be caught. *(Still open: full N-context
> scheduling in the core and the host-side multiplexer.)*

### A. Provisioning A/B dual-slot + boot-time integrity/version check — *brick prevention*

**Decide:** two model slots on NVMe (active / staged), an atomic
active-pointer flip, rollback to the last-good slot, and a boot-time
verify (hash/CRC + version + resident-set descriptor match) that gates
inference on "model present AND valid AND correct version."

**Why now:** today a model update is an undefined, all-or-nothing **467 GB
single-copy rewrite** — a power blip mid-write permanently bricks the box, with
no rollback and no version visibility (`docs/USBC_PRODUCT_PLAN.md:218,300,312`;
`docs/OPERATION_FLOW.md:298-299`). Worse, boot is a **raw block move with no
verification** — `src/boot_loader.v:14-18` copies NVMe→DRAM and
`:270-275` raises `done` purely on a word count, so a partial or wrong image
boots "ready" and returns confidently wrong output forever
(`docs/OPERATION_FLOW.md:89-90`; `host/wpu_device.py:9-14` mirrors only
start/busy/done/tok — no model-validity state). Slot layout, the spare NVMe
capacity for a second slot, and the boot-loader's verify/rollback FSM are all
**physical/boot-ROM commitments**. They cannot be added by a host update later.

### B. ECC + scrub on the resident ~467 GB weights

**Decide:** SECDED (or better) plus a background scrubber on the always-on
weight array, and the DRAM-retention / fast-resume story that goes with it.

**Why now:** the production top instantiates the KV pager with the **ECC param
omitted → ECC=0** (`src/glm_q4k_system.v:676-688`), and `grep 'ecc|secded|scrub'
src/ddr5_xbar.v` is empty. A 467 GB array that stays powered for the life of the
device has **zero bit-flip protection** — silent, undetectable weight rot that
degrades answers with no error and no logged event
(`docs/PRODUCT_ROADMAP.md:99`, P2.1 "Remains: DDR5/NVMe payload-byte ECC"). ECC
changes the memory width, the controller, and the scrub scheduling — it is a
**silicon/board decision**, not a firmware patch. The verified ECC RAM /
reset-sync / MBIST blocks already in-tree (commit `a3e0d3c`) are the raw
material; the decision is to *wire them into the resident-weight path*.

### C. USB/host protocol context/sequence-id + a power/telemetry channel

**Decide:** the on-wire frame carries a **context/sequence id** and a **sequence
number**, and reserves a **telemetry/control channel** (power/thermal readback,
eco-mode knob, model-update status, spec-chain accept rate).

**Why now:** the shipped host port carries only `{prompt_tok, start_pos, s_len}`
→ `{busy, tok_valid, next_tok}` (`src/glm_q4k_system_cdc.v:198-205,378`;
`host/wpu_device.py:12,154`). With no context id, the host **physically cannot
route tokens to N contexts** — multi-context is impossible end-to-end no matter
what the RTL can do, and the pager's `append_seq`/`gather_seq` ports
(`src/kv_cache_pager.v:112-117`) are never driven by the CDC top. With no
telemetry field, the promised **GUI power tuning has no wire** to `clk_throttle`
(the eco knob is RTL-built but unreachable — `docs/OPERATION_FLOW.md:105-108`,
`src/clk_throttle.v:38`), and throttle/thermal collapse is invisible to the
user. A wire format is frozen once host, firmware, and CDC RTL agree on it;
retrofitting fields after freeze breaks every deployed unit. **Add the fields
now, even if the endpoints that consume them ship later.**

---

## Findings by theme

Tags: **[LOCK-IN-NOW]** architecture, decide before freeze · **[SOFTWARE-TRACK]**
host/embedded software to build · **[DESIGN]** RTL/system design point to resolve ·
**[DOC-FIX]** inconsistency or under-communication to correct.

### 1 · Provisioning, boot, and updates

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **No real provisioning pipeline.** The packer has only ever processed synthetic data and emits a simulation `.hex` (one word/line), not an NVMe image (`tools/ckpt_pack_q4k.py:15-18,409-412`; real-GGUF end-to-end OPEN per `docs/OPERATION_FLOW.md:339-341`). | No delivered means to build a shippable unit; the "plug in and use it" first-run has no working step behind it. | high | [SOFTWARE-TRACK] |
| **Boot resident-set descriptor is never generated** — a gap between the packer's word-offset manifest and the physical segment table `boot_loader` consumes (`src/boot_loader.v:84-88,68-73`; `tools/ckpt_pack_q4k.py:391-397`; no descriptor generator in `tools/`). | The verified boot-loader cannot actually be driven for the real model. | high | [LOCK-IN-NOW] → §A |
| **Update = undefined all-or-nothing 467 GB rewrite, brick risk, no A/B/rollback/versioning.** | A routine update is a multi-hour single-copy rewrite that can permanently brick the box on a power blip. | high | [LOCK-IN-NOW] → §A |
| **No runtime model present/valid/version gate** — a partial or wrong image is DMA'd and inference released with no signal (`docs/OPERATION_FLOW.md:89-90`; `host/wpu_device.py:9-14`). | A mis-provisioned box appears "ready" and returns confidently wrong output. | high | [LOCK-IN-NOW] → §A |
| **Boot copies ~467 GB with no integrity check** — `done` on word count only, no CRC/hash/error path (`src/boot_loader.v:14-18,270-275,90-106`). | The box can boot "ready" on a corrupted model and produce wrong tokens indefinitely with no error. | high | [LOCK-IN-NOW] → §A |
| **`flash_layout.py` is miscredited as the provisioning tool** — it only prints channel-balance tables (`docs/USBC_PRODUCT_PLAN.md:116-117` vs `tools/flash_layout.py:1-29,265-272`). | Anyone provisioning by the docs hits a dead end. | med | [DOC-FIX] |
| **Air-gapped ingress path is undefined** — how the 467 GB file physically reaches the box on first setup or update is never specified (`docs/OPERATION_FLOW.md:104-108,298-299`; USB-C carries only tokens, `:126-128`). | The air-gapped customer has no supported way to load or refresh the model — the flagship use case has an undefined first step. | med | [DESIGN] |

> **Cross-ref — the ~70 s cold boot** appears here and in §2, §3, §5, §7. Stated
> once: the flagship residency box loads 467 GB from volatile LPDDR5X on **every**
> power-on in **~70 s** (`docs/R3_APPLIANCE_SPEC.md:18,114`), yet
> `docs/OPERATION_FLOW.md:96-99` advertises ~1–2 s (that is the 17 GB
> streaming-rung number), and the host models it as sub-second
> (`host/wpu_device.py:119,215`; `--boot-seconds` default 0.4). No in-flight
> readiness UX is built. **[DOC-FIX + SOFTWARE-TRACK]** — reconcile the number and
> build a progress signal; client timeouts otherwise fire before the box is ready.

### 2 · Host software the user touches

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **No USB-C transport or driver of any kind** — `USBBackend` is a to-build note (`host/README.md:154-156`); no `libusb`/`bulk_out` in repo. | Even the minimal "prompt in, token out over the cable" path cannot run; everything demonstrable is loopback-to-mock on one PC. | high | [SOFTWARE-TRACK] |
| **Multi-context host layer corrupts itself** — `ThreadingHTTPServer` over one shared mutable device, no lock/queue anywhere (`host/wpu_server.py:301-303`; `host/wpu_device.py:193,233-237`). | Two concurrent conversations interleave/garble or race; the "N agents" promise actively breaks under concurrency. | high | [SOFTWARE-TRACK] |
| ~~**Prefix/prompt caching not implemented and actively defeated**~~ **CLOSED (2026-07).** `WPUDevice` now keeps the session's KV-resident token ids (`_kv_ids`) and `generate()` feeds only the un-cached suffix (`host/wpu_device.py` `_prefix_reuse()`; `prefix_stats` counters surfaced at `/api/status`); measured **5.8× total prefill work** and **82.7% hit** on the real GLM template path (same measurement §3's long-prompt row cites). | Turn N no longer re-streams the whole history's weights; the quadratic-chat failure mode is gone. The first turn / a cold long paste still pays full serial prefill — that residue is §3's row. | ~~high~~ **closed** | [SOFTWARE-TRACK] |
| ~~**No telemetry/management endpoints**~~ **PARTLY CLOSED (2026-07):** `host/wpu_server.py` now serves `/api/status` (device state + telemetry dict + prefix-cache hit stats), `/api/provisioning`, `/api/settings`, and the `/console` web UI (`host/wpu_server.py:270-290`; `host/console.html`). **Still open:** power/thermal readback — the device-side telemetry counters are tokens/runs/done/stall (`PROTO_CTX`, default-off); no temp/power/throttle field exists anywhere end-to-end (§5's row). | Dashboards now have a backend and a v0.1 console UI; the power/thermal half of "observe via GUI" still has no data source. | high | [SOFTWARE-TRACK] |
| **No conversation-history persistence** — server is fully stateless (`host/wpu_server.py`, no storage code). | None of the user's history is retained; the "appliance that remembers you" value is absent. | high | [SOFTWARE-TRACK] |
| ~~**Only real-token backend is hardcoded to a deleted build**~~ **CLOSED (2026-07).** The sim backend was retargeted to the on-main `glm_model_q4k` product top — its header marks the fp8-era `build/glm_model_fp8_sim` path as History (`host/wpu_sim_backend.py:28-31`) — and two further real-token backends landed: `host/wpu_llama_backend.py` and `host/wpu_modal_backend.py`. *(This row's original cite `README.md:331` no longer exists — the README has since been rewritten shorter.)* | Evaluating the software on main no longer means a mock echo. | ~~med~~ **closed** | [SOFTWARE-TRACK] |
| **Readiness model understates cold start ~35–175×** (0.4–2 s modeled vs ~70 s real) (`host/wpu_device.py:119,215`; `host/wpu_server.py:278`). | Any UX built on it misrepresents the wait; calls during boot fail or hang on a misleading ready signal. | med | [DOC-FIX] |
| **Sampling/tuning knobs have no observable effect** — `configure_sampling` records but MockDevice ignores; penalties "accepted and ignored" (`host/wpu_device.py:162-172`; `host/README.md:90-94`). | A tuning GUI would show sliders that change nothing. | low | [SOFTWARE-TRACK] |

### 3 · Interactive session quality

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **Long-prompt prefill is serial and pays full per-token weight streaming** — catastrophic first-token latency (`host/wpu_device.py:174-183`; `src/glm_q4k_system.v:507-574`; on-die prefill write-back not implemented per `src/glm_q4k_soc_ms.v:218-220`). | **Rate corrected (2026-07): prefill runs at ~43 tok/s, not the ~80 previously stated here.** ~80 is the DECODE rate, and R3 §2's decode constant divides by the spec chain's acceptance — an amortisation prefill cannot use, because the prompt's tokens are already known, so there is nothing to speculate. Decode is `14·U/A + 11/A + 0.5` at the measured K=1 point (A_eff=1.87, U(1)=1.00) = **13.87 GB/tok** → 1.1 TB/s ÷ 13.87 ≈ 79 ≈ the stated 80. Prefill is the same formula at A=1/U=1: the full `14 + 11 + 0.5 = 25.5 GB/tok` → 1.1 TB/s ÷ 25.5 ≈ **43 tok/s** (v3-proto's 1.54 TB/s ≈ 60) — **1.84× the decode cost per token**. *(This row first cited R3's `≈15.4 GB/tok` as the decode constant; 15.4 is the retired K=4 proxy vintage and reproduces none of the project's live headlines — see R3 §2's re-derivation. The prefill arithmetic was unaffected: 25.5 does not depend on it.)* So a 4 K doc ≈ **95 s** to first token (not ~50 s) and 16 K ≈ **6.3 min** (not 3+) — a **1.7× understatement**. **Cross-turn recurrence is FIXED** (prefix/KV cache, `host/wpu_device.py`; measured 5.8× total prefill work and 82.7% hit on the real GLM template path), so this now bites only the FIRST turn / a cold long paste. Batched prefill would be the remaining lever — the KV-egress wall that blocked it is since proven closed, but the prefill *mode* itself is unbuilt and its ceiling modest — see the row below. | high | [DESIGN] |
| ~~**Batched prefill is unreachable: the die has no KV egress path**~~ **PREMISE CLOSED (2026-07): the KV egress path exists and is proven.** `mla_attn_q4k` now exposes the committed latent — `kv_lat_row`/`kv_lat_valid` plus the PE_M-wide `kv_lat_row_all` (`src/mla_attn_q4k.v:319-325`) — and the die-internal KV write-back (`SELF_KV`) is **BUILT + VERIFIED**: the die attends its own written per-(layer,pos) KV, bit-exact vs an independent (layer,pos) reference and byte-identical when off (`make self-kv-roundtrip` / `self-kv-equiv`, both in `release-gate`; `docs/KV_WRITEBACK_DESIGN.md` "Status: BUILT + VERIFIED"). Intra-batch causal MLA (`INTRA_CAUSAL`) additionally lets a PE_M row at position p+i attend row p+i−1's in-flight key — batched verify == serial, `make intra-batch-verify` (in `release-gate`). | **Still open: a batched-prefill *mode* is not built** — no top drives PE_M-wide prompt rows through the write-back for prefill (spec-decode's PE_M=K+1 verify batch is the only consumer). The ceiling stays modest even when built: the repo's own union formula caps it at **~1.5–2× at feasible B** (the expert union barely shrinks below B≈32; the 11× needs B≈256 ≈ 770 TFLOP/s, infeasible). Serves one case the prefix cache cannot: a cold long paste. | med | [DESIGN] |
| ~~**Raising the context window would SILENTLY FREEZE attention to the first TOPK tokens**~~ — **THREADED (2026-07).** `DSA_REAL_IDX` was a parameter of `src/mla_attn_q4k.v:170` **and of no other file in the repo**: `glm_decoder_block_q4k` instantiated the attention passing `PER_ROW_POS/SLEN/SEQ` but **not** it, so the production hierarchy hard-wired it to 0 and **=1 was unreachable from any top**. Now threaded `glm_q4k_system(_cdc)` → `glm_model_q4k` → `glm_decoder_block_q4k` → `mla_attn_q4k`, default 0. **Verified:** `make dsa-sparse-correct` runs the whole `glm_q4k_system` at **both** values against the standalone reference and both agree — `DSA=0 → tokens 12,14,14,14`, `DSA=1 → tokens 12,14,2,2`. The rows diverge at exactly `s_len>TOPK_ATTN=2`, where selection starts, **so the value provably survives `system → model → decoder → mla_attn`**: if any link dropped it, `=1` would emit `=0`'s tokens. The `=0` row is the "threading changed nothing at the default" claim, checked against the reference simulator. 1045 s, both green, in `release-gate` (slot #10). **NOT verified:** *byte-identical netlist* at `=0`. `make dsa-thread-equiv` is written but has **never completed** — measured three ways on this machine, none finishing: `equiv_induct` with real `mla_attn` (30m45s / 5.29 GB / still growing), an earlier attempt (>13 min, killed), and an RTLIL diff with no `memory` pass (>9 min per side, still in `proc`). `prep`/`proc` blow up on a decoder-sized module here, so the SAT layer never gets a fair run — a property of the tool, not the design. It is now **opt-in, deliberately out of `release-gate`**: an unrunnable gate does not raise the bar, it makes the whole suite unrunnable, which is what it did from `43de204` until this commit. **No claim rests on the residue** — cost of `=1` comes from yosys `stat` (+0.2%), not from equivalence — but until it goes green, "byte-identical" stays an argument. | **What this fixed: reachability, not the window.** At 0 the indexer is fed zero key-index vectors, so "every key scores 0 and top-K keeps keys 0..min(S,TOPK)-1 by lower-index tie-break — **Q-INDEPENDENT**" (`mla_attn_q4k.v:155-158`). At the committed S_MAX=8/TOPK_ATTN=8 that is a **no-op for any value** — the dense path never pulls keys at all (`:165-169`). Raise S_MAX past TOPK_ATTN with the old unthreaded default and every query at every position would attend to **the first 8 tokens, forever**: fluent output, frozen prefix, green tests (nothing asserted **which** keys were selected). Raising the window is now a **decision** (`DSA_REAL_IDX=1`, proven bit-exact at the leaf by `make mla-sparse` at PE_M=3) rather than an accident of defaults. **MEASURED 2026-07: turning it on costs +0.2%.** Forcing the sparse regime (TOPK_ATTN=2 < S_MAX=8) on the ratio-faithful perf config: DSA=0 -> 21,617 cyc at s_len=4, DSA=1 -> 21,665 (+0.2%), and S_DSAPF goes 0 -> 84, i.e. the query-dependent prefetch actually runs. **So the thing that makes attention correct is essentially free, and it is off.** Also measured: cost saturates once s_len > TOPK_ATTN (increments 2,275 -> 953 -> 9), so attention is O(min(s_len, TOPK_ATTN)), NOT O(n) -- the window can be raised without a throughput explosion. The exposure is therefore purely CORRECTNESS: at DSA=0 a raised window is cheap, fast, and silently reads only the first k keys. **The context window itself is still NOT raised** — see the rows below; this removed the trap that made raising it unsafe. | ~~high~~ **closed** | [DESIGN] |
| **The roofline that sizes the array cannot express what actually limits it** — `cycles/token` is AFFINE in 1/lane, not inversely proportional: measured `cycles = 7,358 + 14,176/TN` fits four TN points to **0.00%** (`make lane-scaling`). Sweeping EVERY lane knob 4x (TN 4→16, PE_N 2→16, LM_TN 4→16) buys **1.90x total** and leaves 53% of cycles/token untouched. An FSM state histogram at max lanes: **T_ATTN 66.7%** (only −38% for 8x PE_N), T_EXPW 10.0%, lane-invariant rn/route/acc/radd 15%, **T_ACC 1.7%**, T_ESCAN 0 (dead at PE_M=1). | **Every tok/s in the docs comes from `bandwidth ÷ bytes-per-token`, a model with no term for this.** R3 §3's array spec (12,732 lanes → 110 tok/s) implicitly sets the lane-invariant residue to zero; the RTL says it is half the cycles at the measured config. So the array size is being chosen against a model the RTL contradicts, and a throughput project aimed at lanes (or at the expert path, 10% at max lanes) would buy far less than the roofline promises — **attention, which no doc identifies as the bottleneck, is what is left**. Also refutes the design study's headline that T_ACC / fp32-add order at TOPK=8 is the cost centre: measured 1.7%. **SCOPE**: the perf TB is MODEL_DIM=16 / INTER_MOE=16 / TOPK=2 / L=4; the real config's ratios differ (6144 / 2048 / 8 / 78 — MODEL_DIM/INTER_MOE is 1.0 here vs 3.0 there, and a first estimate puts T_ACC at ~22% at real shape, so it may return). What transfers: the residue EXISTS and is large, the affine model is exact, T_ACC is not it here, T_ESCAN never runs. Real-shape ratio = **[측정필요]**. | high | [DESIGN] |
| **Committed context window is tiny** — S_MAX=8 attention window, KV_CTX=1024 (`src/glm_q4k_system.v:128,145-146`; `src/mla_attn.v:210-212`); 1 M context is elaboration-only (`docs/FULL_CONFIG_ELAB.md:56-61`) while the UI implies ~500 K (`docs/USBC_PRODUCT_PLAN.md:96`). | Near the committed config, anything beyond a few tokens is silently outside the window; scaled up, the long-context path has never run end-to-end. **MEASURED 2026-07: the compute cost of a bigger window is ~nil.** With DSA_REAL_IDX=1 and TOPK_ATTN capping the key count, S_MAX 8->64 costs **+0.5%** (20,586 -> 20,698 cyc/tok) and the KV/score cycles do not move AT ALL -- attention is O(min(s_len, TOPK_ATTN)), so window SIZE does not enter its cost. System stayed consistent with the standalone reference at every point (`make dsa-sparse-correct` proves DSA=1 at the system level; `make lane-scaling-sparse` has the cost). **So the blocker was never throughput -- it is (a) turning DSA on, which costs +0.2%, and (b) KV capacity at 87.8 KB/token (R3 §5c: ~65K tokens on v3-proto).** Raising S_MAX without DSA=1 is the one thing that must not happen: cheap, fast, and silently reading only the first k keys. | high | [DESIGN] |
| **No context-overflow policy or length guard host-side** — ~~messages forwarded with no bound~~ **PARTLY CLOSED (2026-07): the host now refuses.** `WPUDevice.context_capacity` + `_check_context_fits()` bound the turn against the LAST position it would touch (prompt **and** the tokens it may generate — the aliasing happens on write, so a prompt that fits with a reply that doesn't would clobber live rows mid-stream) and raise `ContextOverflow` instead. Mutation-tested (5/5). **Still open device-side**: `src/kv_cache_pager.v:73-74,193` still aliases if anything drives it past the ring, and `context_capacity` is None until a backend sets it — the guard is a host contract, not an RTL interlock. | A long paste yields silently wrong output (aliased KV) instead of a graceful truncate or clear error — looks like hallucination to defense/finance/health users. | high | [DESIGN] |
| ~~**Prefix-cache is a doc bullet only**~~ **CLOSED (2026-07)** — dedup of the §2 caching row, closed with it: `host/wpu_device.py` `_prefix_reuse()` keeps the session's KV resident and re-feeds only the new suffix (measured 5.8× / 82.7% hit — see the long-prompt row above). | The "interactive, cache-always-on" feel is now real from turn 2 onward; the first turn still pays full serial prefill. | ~~high~~ **closed** | [SOFTWARE-TRACK] |
| *Cross-ref:* **~70 s cold start with no readiness UX** (see §1 box) — jarring instant-on failure for a phone-tethered user. | | high | [DOC-FIX] |
| *Cross-ref:* **Multi-context aggregate-throughput promise contradicts B=1 scope** (see §4). | | high | [DESIGN] |

### 4 · Multi-context

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **Shipped USB-C protocol has no context/sequence id** — host cannot route tokens to N contexts; pager's per-seq ports never driven (`src/glm_q4k_system_cdc.v:198-205,378`; `src/kv_cache_pager.v:112-117`). | Multi-context is impossible end-to-end regardless of RTL. | high | [LOCK-IN-NOW] → §C |
| **단일 컨텍스트 = 사양 (2026-07 결정). 갭 아님.** ~~**Production top instantiates the pager at NSEQ=1** — multi-context RTL lives only in the unshipped `glm_q4k_soc_ms` side module, Q4_K re-run PENDING (`src/glm_q4k_system.v:676-687`; `src/glm_q4k_system_cdc.v:6-7,33`; `src/glm_q4k_soc_ms.v:101`; `docs/OPERATION_FLOW.md:321`).~~ | KV 산수가 이 결정을 강제한다: v3-proto의 KV 예산은 ~13GB(480−467)이고 GLM-5.2 실형상의 MLA KV는 87.8 KB/token이라, 컨텍스트 1개면 ~65K 토큰(2ⁿ 링 내림 후)이지만 4개로 쪼개면 각 ~16K — 프론티어 모델에 쓸모없는 창이다. 프로덕션 탑은 이미 그렇게 돼 있다: `glm_q4k_system`에 **NSEQ가 0번 등장**하고(페이저 기본값 1), 멀티컨텍스트는 게이트도 TB도 없는 미출하 `glm_q4k_soc_ms`에만 있다. §1의 '단일 사용자 오프라인 박스'와도 정합하고, **성능도 안 잃는다** — PE_M은 배치가 아니라 스펙체인 검증 폭이라 투기 디코딩은 그대로다. 대화 전환은 멀티-상주가 아니라 KV의 NVMe 스왑으로 푼다(2K 대화 KV=0.2GB ≈ 30ms vs 재프리필 37초). **그 스왑의 다이 쪽 벽은 사라졌다 (2026-07): `kv_lat_row` 출구 + SELF_KV 왕복이 증명됐다**(`make self-kv-roundtrip`/`self-kv-equiv`, §3의 KV 출구 행 참조) — **남은 것은 KV를 NVMe로 실제로 내보내고 되돌리는 페이저/호스트 배관이다.** | ~~high~~ **비-목표** | [DESIGN] |
| **No multi-context software path** — host stack is hard single-session, keyed by a global cursor (`host/wpu_device.py:102-113,193,227-254`; `host/wpu_server.py:68-75,303`). | Two contexts produce cross-talk or a mid-generation reset; the box is one-conversation-at-a-time. | high | [SOFTWARE-TRACK] |
| **비-목표 (2026-07): 단일 컨텍스트 결정으로 소멸.** ~~**Batching is fixed lockstep, not elastic** — host FSM prefills all seqs then steps in lockstep, no per-context arrival or EOS (`src/glm_q4k_soc_ms.v:321-445,397,430`).~~ | 합류할 배치가 없다. | ~~high~~ **비-목표** | [DESIGN] |
| **No KV-sharing / prefix dedup across contexts** — NSEQ independent windows (`src/kv_cache_pager.v:23-32,90-93`); the owner's "better-than-linear via KV-sharing" premise isn't what the design does. | The mechanism the aggregate-throughput number rests on is absent, and the number is unmeasured and contradicted by the docs' own notes. | high | [DESIGN] |
| **비-목표 (2026-07): 단일 컨텍스트 결정으로 소멸.** ~~**Context count conflated with KV byte budget; no admission control** beyond a fixed lane count (`src/glm_q4k_soc_ms.v:86,101`; no scheduler in `host/`).~~ | 컨텍스트 수가 1로 고정이라 admission control의 대상이 없다. 대신 **길이** 상한이 실재하며, 그건 위 오버플로 가드가 강제한다. | ~~high~~ **비-목표** | [DESIGN] |
| **비-목표 (2026-07): 격리할 컨텍스트가 하나뿐.** ~~**Per-context KV isolation is addressing-only** — no bounds/permission enforcement, no adversarial cross-context test (`src/kv_cache_pager.v:189-198,182-185`; `src/glm_q4k_soc_ms.v:491-492`).~~ | 재개 조건: 멀티컨텍스트를 실제로 추구할 때. | ~~med~~ **비-목표** | [DESIGN] |

### 5 · Power, thermal, physical

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **"Plug USB-C into my laptop and go" is physically impossible** — the box is dead until a second power source is attached; a laptop port sources only 7.5–15 W vs the box's floor of ≥50–78 W [EST] (v3-volume) / ≥64–99 W [EST] (v3-proto) (`docs/R3_APPLIANCE_SPEC.md` §4 — the old ~40–60/~50–80 W figures are retired there), and what ships (own DC / USB-PD EPR / bundled brick) is unresolved (`docs/R3_APPLIANCE_SPEC.md:81-83`; `docs/USBC_PRODUCT_PLAN.md:310,328-330`; no "no-power" device state in `host/wpu_device.py`). | The single most-stated usage promise fails at the very first action; the user thinks the device is broken. | high | [DESIGN] |
| **Volatile 512 GB LPDDR5X: any power blip wipes the model AND all resident cache** — contradicting both "plug in, ready" and "unconditional caching" (`docs/R3_APPLIANCE_SPEC.md:18,114`; no retention/fast-resume in `src/`). | Every cold start is a ~70 s wait; a bumped cable means 70 s + total loss of the promised always-reused KV. | high | [LOCK-IN-NOW] → §B |
| **No power/thermal telemetry surfaced** — DeviceState is only BOOTING/READY/BUSY; no temp/power/throttle attribute (`host/wpu_device.py:96-99`; `docs/USBC_PRODUCT_PLAN.md:281`). | An 80→8 tok/s throttle drop with no on-screen reason reads as "broken"; the user can't tell it's thermal/power or act on it. | high | [LOCK-IN-NOW] → §C |
| **No closed-loop thermal management in RTL** — the throttle knob is a static input tied to 0, no sensor drives it (`src/clk_throttle.v:38`; `src/clk_gate_cluster.v:76`). | During the long sessions the box is pitched for, it can run hot and loud with no automatic protection; "≤35 dBA quiet" has no mechanism behind it. | med | [DESIGN] |
| **Eco/power knob is unreachable by the user** — the protocol carries only token IDs, so the planned GUI power tuning has no wire to `clk_throttle` (`docs/OPERATION_FLOW.md:105-108`; `host/wpu_device.py:12,154`). | The eco/quiet mode is marketed and RTL-built but is a dead control. | med | [LOCK-IN-NOW] → §C |
| **Power number is inconsistent across docs** — 40–60 W vs 80–110 W vs 30 W throttled (`docs/R3_APPLIANCE_SPEC.md:79,173-175`; `docs/USBC_PRODUCT_PLAN.md:50,211,320`). | The buyer can't size an adapter, predict desk heat, or set noise expectations; two docs give two pictures. | med | [DOC-FIX] |
| **Idle/standby power for the 512 GB box is unanalyzed** — the ≤10 W target is inherited from the smaller streaming box, yet the 70 s boot pushes 24/7-on (`docs/USBC_PRODUCT_PLAN.md:212`; `docs/LOW_POWER.md:325`). | The user pays continuous standby watts and 24/7 fan noise without being told the number; ≤10 W likely doesn't hold. | med | [DOC-FIX] |

### 6 · The flagship air-gapped RAG / GUI / visualization workflow

> This is the single most-cited reason to buy the box, and its RAG / visualization /
> chat-GUI surfaces are **unbuilt** — the sole landed piece is the v0.1 management
> console (`host/console.html`, web-UI row below), which is device management, not
> the flagship workflow.
> Stated once here; it also surfaced under host software (§2) and reliability (§7).

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **The entire RAG stack does not exist** — embedder residency, RAG store, graph view are labeled NEW work items; `grep embed/vector/retriev` over host+src = 0 hits (`docs/USBC_PRODUCT_PLAN.md:90-91,108-109`; `host/wpu_server.py:196-210`). | When a design partner says "show me the RAG," there is nothing to run — not a prototype, not a stub. | high | [SOFTWARE-TRACK] |
| **Embedded web UI / GUI — PARTLY CLOSED (2026-07): `host/console.html` exists.** ~~`find *.html/*.js/*.css` = 0 files~~ — that evidence is stale: `host/console.html` (the tracked v0.1 WPU 관리 콘솔, a deliberate deliverable per `docs/PRODUCT_SPEC.md:73`) is served by `host/wpu_server.py` at `/console`, fed by `/api/status` (telemetry + prefix-cache stats) / `/api/provisioning` / `/api/settings`. **Still open:** it is a *management console served by the host-side Python server on the attached PC* — the on-device "plug in and a UI appears at `wpu.local`" experience (server embedded in the box) and the §6 chat / RAG / visualization GUI surfaces remain unbuilt (`docs/USBC_PRODUCT_PLAN.md:84-94`). | The "no software to install" demo still cannot happen (the console needs the host Python server), but "a technical user curling a Python endpoint" is no longer the best case — a browser console with live telemetry exists. | high | [SOFTWARE-TRACK] |
| **Visualization graphs unbuilt with no data plumbing** — knowledge graph / timeline / spec-chain telemetry have no collection path; USB-C carries only token IDs + position (`docs/USBC_PRODUCT_PLAN.md:92-94`; `docs/OPERATION_FLOW.md:105-107`). | The "see visualization graphs" half of the flagship triad is nonexistent and can't be quickly prototyped. | high | [SOFTWARE-TRACK] |
| **GUI tuning surface has nothing to tune and no GUI** — sampling knobs are API-only and partly inert (`docs/USBC_PRODUCT_PLAN.md:93`; `host/README.md` sampling table). | The "tune it via GUI" third pillar is unbuilt and, where wired, non-functional against the only runnable backend. | high | [SOFTWARE-TRACK] |

### 7 · Reliability and failure modes

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **No ECC on the resident model weights** — 467 GB always-on DRAM with zero bit-flip protection or scrub (`src/glm_q4k_system.v:676-688`; empty `grep ecc\|secded\|scrub`; `docs/PRODUCT_ROADMAP.md:99`, P2.1). | Silent, undetectable weight rot; answers subtly degrade with no error — intolerable for a defense/finance/health buyer trusting local answers. | high | [LOCK-IN-NOW] → §B |
| **Power loss = full ~70 s cold reload + total session loss, no hold-up/resume** — `start` always restarts from `rseg=0` (`src/boot_loader.v:207-221`; `docs/R3_APPLIANCE_SPEC.md:16-21`). | Every power blip is a multi-minute total outage plus loss of the entire conversation/RAG working state — the opposite of "always-on personal infra." | high | [DESIGN] |
| **Boot has no integrity check** — a bad/partial provisioning or NVMe read error yields a silently wrong model (dedup of §1; `src/boot_loader.v:14-18,270-275`). | The box can run a corrupted model indefinitely with no error signalled. | high | [LOCK-IN-NOW] → §A |
| **Context/KV overflow wraps silently** — unbounded position counter aliases past KV_CTX, no "context full" signal (dedup of §3; `src/kv_cache_pager.v:177,306-309`; `src/glm_q4k_system.v:145`). | A long session quietly starts returning garbage attention instead of cleanly reporting "context full." | high | [DESIGN] |
| **Encryption-at-rest for NVMe KV/history is unbuilt** — promised to the security buyer; `grep encrypt\|aes\|cipher` over src+host = empty (`docs/USBC_PRODUCT_PLAN.md:99-100`). | For the buyer who chose the box because "data never leaves," a removed drive exposes plaintext history and KV — the security promise fails at physical-theft, the exact boundary it was written to cover. | high | [SOFTWARE-TRACK] |
| **Unconditional caching writes continuously to the single M.2 that also holds the 467 GB model** — no wear-leveling/quota (`docs/USBC_PRODUCT_PLAN.md:95-104`; `docs/R3_APPLIANCE_SPEC.md:115`; `docs/USBC_PRODUCT_PLAN.md:294`). | On a device meant to run for years, steady cache writes wear the drive that holds the model; when it degrades, the user loses history and model store at once. | med | [DESIGN] |
| *Cross-ref:* **RAG/viz 0% built, GUI = v0.1 management console only** (§6) and **multi-context contradicts B=1 scope** (§4) also surfaced as reliability failures. | | high | [SOFTWARE-TRACK] / [DESIGN] |

---

## Severity summary

Counts are over the **49 confirmed findings** as reviewed (before the dedup
above, so the ~70 s boot and the RAG/GUI gap are each counted once per dimension
they broke in):

| Severity | Count |
|---|---:|
| High | 37 |
| Medium | 11 |
| Low | 1 |
| **Total** | **49** |

By classification (post-dedup, distinct gaps):

| Tag | Meaning | Distinct gaps |
|---|---|---:|
| **[LOCK-IN-NOW]** | Architecture — decide before board/boot-loader/protocol freeze | 3 decisions (§A ECC-adjacent boot/provisioning, §B ECC+scrub, §C protocol) covering ~8 findings |
| **[SOFTWARE-TRACK]** | Host/embedded software to build (transport, RAG, GUI, viz, persistence, caching, encryption) | ~17 |
| **[DESIGN]** | RTL/system design point to resolve (context window, prefill, multi-context batching, power delivery, thermal, wear) | ~15 |
| **[DOC-FIX]** | Inconsistency or under-communication to correct now (boot timing, power band, idle power) | ~5 |

---

## Closing — honest framing

**This is product-stage reality, not a regression.** Nothing in this register
says the accelerator is broken. The datapath is bit-exact against a numpy
golden, the KV pager and boot DMA engine are verified, the ECC / reset-sync /
MBIST / clock-gating building blocks exist and pass, and the formal properties
hold. That work is real and it is strong.

What the review makes unambiguous is that a **verified accelerator is not yet a
usable device.** The software a buyer touches — the transport, the RAG store,
the embedded UI, the visualizations, the tuning surface, history persistence,
multi-context routing, and real provisioning — is largely unbuilt, and a small
number of lifecycle/safety decisions (A/B provisioning + boot integrity, ECC +
scrub on resident weights, a context-id + telemetry protocol) need to be locked
before hardware and protocol freeze because they are cheap now and a respin
later.

The right read of this document is not "the project regressed." It is: **the
hard, de-riskable silicon problem is largely solved; the remaining work is the
experience and lifecycle layer, and three of those items must be decided before
freeze.** Sequencing the three LOCK-IN-NOW decisions ahead of the
software-track build is the single highest-leverage planning move available.
