# ULTRA_PERF_MODULES — bottom-up module-level ultra-high-performance plan

> **Companion to [`ULTRA_PERF.md`](ULTRA_PERF.md).** That report is the *top-down system-lever* list.
> This one is the **bottom-up microarchitecture pass** the modules were read for — critical-path /
> Fmax limiters, per-cycle throughput, internal serialization, and lane/fabric-width ceilings — each
> grounded in the actual RTL (file:line) and tagged **bit-exact-preserving** vs **output-changing**,
> with every impact number **[EST]** unless it cites a measured result. Governing model: bandwidth-
> bound `tok/s ≈ BW ÷ 13.87 GB/token`; the die is ~20–25% utilized but routed Fmax is **46.5 MHz**
> (route-dominated, worst path `u_moe/y_out → hbuf`), so ultra-perf = make the die **faster + wider**
> to consume bandwidth, **de-serialize** sequential phases, and **cut per-token bytes/stalls**.

## Executive summary

The bottom-up module review confirms the roofline: on the single-user product tok/s is delivered-bandwidth-bound, and the delivered bandwidth is throttled not by channel count but by Little's-law depth and single-lane seams that the top-down docs treat as one line ('stripe across N channels'). The highest-tok/s levers are therefore a fetch-path cluster the modules expose concretely -- a non-blocking MSHR expert cache (single outstanding miss caps demand BW at 1/FLASH_LAT and starves every downstream widening), a multi-DRAIN fabric (N_CH today buys only latency-hiding because the response arbiter grants one beat/cycle), a multi-port hot issuer, and in-RTL deterministic prefetch of the already-known top-k union during the norm/accumulate windows. Second is the Fmax cluster: the repo's ONE measured routed-Fmax limiter is a ~98 Kbit combinational expert-output bus into a far mux (u_moe/y_out -> hbuf, 21.2 ns, 59% wire), fixable by a narrow registered read port plus registering the identical leaf shapes in glm_matmul_q4k.c_out and the SwiGLU merge -- ~0 on today's bandwidth-bound B=1 but the precondition for the die to ever consume 100 GB/s (at 46.5 MHz that needs ~4,300 lanes). Third is de-serialization and lane-widening of the compute tail (single-lane bf16 SHN tail, gate||up, router||shared-expert, running argmax) that cut the measured cyc_per_tok~10,896 and raise duty cycle; the two output-changing levers (L-way accumulate, single-Newton reciprocal) are ranked last and gated behind golden rebaseline because they break the bit-exact contract.

## MEASURED cycle attribution — the basis for re-ranking everything below

Everything in this section is **measured** (`make rank3sys`, `RESIDENT=1 N_EXPERT=8 DDR_NCH=4`,
4-token decode) and every bucket was closed to the exact cycle against the RTL. It supersedes the
`[EST]` reasoning that produced the original ranking, because that ranking assumed the die was
fetch-bound. **It is not: the die was measured at 0.37 memory-requests/cycle, and widening the
request port 1→4 changed decode cycles by ZERO (rank 3).** At this configuration the die is
compute-bound, so the compute buckets below — not the fetch levers — are where cycles live.

**First, a correction to the instrumentation itself.** `PERF_STATE` counts every post-reset posedge
of the *TB* clock, not the per-token measurement window. Post-reset posedges are exactly
`(453325 − 45)/10 + 1 = 45,329`, which is exactly the printed bucket sum; the decode window is
43,724. The 1,605-cycle excess decomposes with no free parameters: 2 (post-reset lead-in) +
3 × 401 (inter-token `settle(400)` gaps) + 400 (tail to `$finish`). All of it lands in `idle`,
because `done=16` equals exactly 16 layer-invocations, so `T_DONE` is a strict 1-cycle state and the
block can only park in `T_IDLE`. Therefore **in-window `idle` = 2,686 − 1,605 = 1,081**, and the
remaining budget closes exactly: `1081+816+24824+256+807+5560+744+8852+384+128+256+16+0 = 43,724`.

> Any percentage taken against 45,329 is wrong: non-idle buckets are understated by ×0.9646 and
> `idle` is overstated 2.5× (6.1% printed vs 2.5% real). The denominator is **43,724**.

| bucket | cycles | % decode | what it actually is |
|---|---|---|---|
| `attn` | 24,824 | **56.8%** | `T_ATTN` → `mla_attn_q4k` |
| `expw` | 8,852 | **20.2%** | **expert FFN compute, not a stall** — 8,712 compute + 140 refill stall (98.4% compute) |
| `ffnd` | 5,560 | 12.7% | dense SwiGLU, 8 invocations × 695 |
| `idle` | 1,081 | 2.5% | model-level work this histogram structurally cannot see (embedding, final norm, LM head) |
| `rn1` | 816 | 1.87% | 16 × (2·LEN+19) at MODEL_DIM=16 |
| `rn2` | 807 | 1.85% | **not** 816: one invocation took the rsqrt early-exit (`rmsnorm_unit.v:275-285`), 816−9 |
| `route` | 744 | 1.70% | 8 × 93 |
| `acc`/`radd1`/`radd2`/`fcomb` | 384/256/256/128 | <1% each | |

Inside `attn` (this sub-histogram is gated on `T_ATTN`, so its sum matching `attn` is tautological,
not independent evidence):

| | cycles | what it actually is |
|---|---|---|
| `key` | **9,560** | **not a fetch bucket.** 40 key-visits × **239 cycles exactly**. It is the MLA *up-projection*: re-deriving K and V from every key's compressed latent, every head-group, every layer, every token, with no caching of the projected K/V. The actual KV read is ~2 of those 239 cycles (`SELF_KV=0` makes `kc_valid` a 1-cycle delay of `kc_req`). |
| `soft` | 5,376 | 32 rows × **168 cycles exactly**. `LEN = SWIN = min(S_MAX,TOPK_ATTN)`, the DSA top-K window — **not** sequence length. So this is a top-K-window lever, not a long-context lever. |
| `quq`/`out` | 1,952 ea | |
| `qdq`/`kvdkv` | 1,504 ea | |
| `dsa` | **64** | 0.15% of decode — rank 15 is correctly labelled long-context-only and is nothing here |

**The finding that reframes the compute work:** of `key`'s 239 cycles per visit, ~124 are the two
Q4_K GEMVs (lane count can shrink those) but **~68 are pipeline *drain*, not math** —
`glm_matmul_pipe.v:128-131` gives drain = `FP_MAC_LAT(7) + TREE_LAT(3×FP_ADD_LAT=15) + 1 = 23`
cycles to finish an 8-element dot product. That is ≈2,720 cycles, **6.2% of the entire decode, spent
draining an adder tree**. No amount of lane-widening touches it; only shortening or overlapping the
drain does.

### Measured ceilings of the remaining ranks (replaces their `[EST]` impact lines)

| rank | measured ceiling | verdict |
|---|---|---|
| 9 SwiGLU gate\|\|up | **6,368 cyc (14.6%)** implementable of 9,056 absolute | biggest remaining lever |
| 16 MLA key path | 1,824 cyc (4.17%) | worth doing, plan needs rework |
| 14 softmax pipeline | 864 cyc (1.98%) | **only plan that survived adversarial review** |
| 8 rmsnorm/bf16 tail | ~2,183 cyc gross | plan's safety apparatus was refuted |
| 13 topk_select pipeline | **ZERO cycles** | `topk_select` is 96 of `route`=744; the `S_EXTR` tournament it pipelines is **16 cycles = 0.037% of decode**, and `topk_select.v:360-378` already retires one pass/cycle with `:372` making passes strictly serial — pipelining can only ADD cycles. **Re-scope to an Fmax lever or drop.** |
| 10 router \|\| shared expert | ≤744 cyc (1.70%) | NOT_WORTH_IT; the drafted RTL also hangs at some geometries |

**Incidental finding:** `src/sampler.v` is in `GLM_Q4K_SYS_SRCS` but is **instantiated nowhere** in
`src/` — it is compiled into every system build and used by nothing.

## UNVERIFIED: timing and area of every landed lever

Every perf item landed so far is proven on **cycles and bits**, and on **nothing else**. That is a
real gap, not a formality, and it is stated here so no reader mistakes a cycle number for a tok/s
number.

The repo has exactly ONE measured routed-Fmax limiter: `u_moe/y_out → hbuf`, **21.2 ns, 59% wire,
46.5 MHz**. At that frequency a 1.7% Fmax loss cancels a 1.7% cycle win outright — so for any lever
whose new hardware lands near that path, the sign of the net effect is currently **unknown**.

| lever | what it adds | why timing/area is open |
|---|---|---|
| **9 `GU_CONC`** | a **second `glm_matmul_q4k`** per `swiglu_expert_q4k` (×2 instances in the decoder) | Its `c_out` feeds `up_hold` → the silu·up merge → `hbuf`. That is *exactly* the measured critical-path region. Biggest cycle win (8.42%) **and** the biggest timing risk. |
| **7 `REG_COUT`** | a register stage on `glm_matmul_q4k.c_out` | Intended to *help* Fmax — that is the whole point of the item — but the benefit is `[EST]` until a re-fit measures it. |
| **14 `SM_PIPE`** | no new hardware (deletes two interlock terms) | Cannot add logic depth, but it raises fp-pipe utilisation; area is flat, timing is presumably neutral — **presumably**, not measured. |
| **2/3 `RESP_LANES`/`REQ_LANES`** | per-lane muxes + a wider drain in `ddr5_xbar` | Widening the fabric is where the doc's own header predicts congestion risk lands. |

**What a re-fit must report, per lever, at `GLM52_SLICE_*` and at the shipping shape:**
WNS/TNS on the `u_moe/y_out → hbuf` path, achieved Fmax, LUT/FF/DSP/BRAM deltas vs the default
build, and post-route congestion. The decision rule is one line:
`net tok/s = (cycle win) × (Fmax_new / Fmax_base)` — a lever is only real if that product exceeds 1.

**Until then**: every landed lever ships **default-off**, every claim in this document is a *cycle*
claim, and no tok/s claim is made for any of them. `make perf-gates` measures cycles; nothing in
this repo measures timing.

## Ranked plan (highest tok/s impact first)

### 1. Non-blocking MSHR expert-cache refill (overlap the 8 top-k miss latencies)

> **STATUS: BUILT + MEASURED (standalone).** `src/expert_cache_mshr.v` + `make mshr`.
> **Measured: 8 cold misses complete in 43 cycles vs the blocking baseline's 160**
> (`FLASH_LAT=20`, high-water 8 concurrent) — a **3.7× overlap of the refill
> latencies**, timed by the gate rather than asserted. Correctness under concurrency
> is proven with two must-FAIL injections (victim collision, duplicate fetch) plus
> out-of-order tagged completion. **Not yet wired into the product top** — the
> verified datapath is untouched. Integration additionally requires widening the
> *other two* single-outstanding seams the review found around it: the `awaiting`
> request issuer (`glm_q4k_system.v` ~:857, exactly one outstanding request to the
> cache) and the single untagged flash arbiter (~:1000, `fl_busy`, shared with the
> KV pager). **The end-to-end tok/s effect stays `[EST]` until those land and it is
> measured on the integrated top.**
>
> Building it surfaced a hazard inspection alone missed, and the gate caught it: a
> slot must stay protected from re-victimisation **until every MSHR entry bound to
> it has responded**, not merely until its data installs — otherwise a merged
> entry's late response names a slot that has already been reused.
- **Modules:** `src/expert_cache_ctrl.v (S_FETCH blocking FSM :184-217), src/expert_cache_pf.v (:389-429)`
- **Change:** Replace the single-outstanding blocking S_FETCH (issue one flash_req, freeze until flash_done) with an M-entry MSHR array: on a miss allocate an MSHR, issue flash_req, keep accepting/issuing the next miss; install each expert when its tagged flash_done returns (match by tag). Issue all top-8 union misses concurrently so they stripe across the N_CH fabric. Keep the exact-LRU directory; only the refill FSM becomes non-blocking.
- **Why (perf model):** This is the Little's-law numerator on the memory wall: tok/s = BW/[(1-h)*footprint], and delivered BW = outstanding_requests/latency. A single outstanding miss hard-caps demand BW at 1/FLASH_LAT no matter how many channels exist, so the fabric depth from ranks 2-5 is UNUSED until this lands. The measured EXPERT_STALL sweep shows exposed stall = 3*FLASH_LAT+9 per miss growing linearly (11 cyc @FLASH_LAT=8 -> 2,567 @1024); with real-NVMe FLASH_LAT in the thousands, serializing the 8 per-token experts is the dominant cyc_per_tok term. Overlapping them is the single biggest B=1 tok/s lever that is a genuinely module-level (not top-down) finding.
- **Impact `[EST]`:** [EST] up to ~8x on the exposed expert-fetch stall (top-8 overlap) plus ~QDEPTH x latency-hiding; on the bandwidth-bound single-user product this is a direct multiplier on the wall term. Needs a cycle re-measure (make perf-q4k with EXPERT_STALL and an MSHR model) to confirm the realized multiplier.
- **Contract:** preserving · **Effort:** large · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 2. Multi-lane read fabric (turn N_CH into sustained N beats/cycle, not just latency-hiding)

> **STATUS: BUILT + MEASURED for `ddr5_xbar` (`make rank2`), default-off.** The claim that the
> response arbiter — not the channel count — is the throttle is now **measured, not argued**: with
> every channel kept supplied, `N_CH=8` sustains **exactly 1.000 beats/cycle** at the committed
> single-grant drain. `RESP_LANES=4` sustains **4.000 beats/cycle** on the same rig — a clean 4×.
> Per-channel order is preserved, no beat is lost or duplicated, and the specific hazard multi-drain
> introduces (two lanes popping the *same* channel in one cycle) is checked explicitly.
> `RESP_LANES=1` is **netlist-identical** to the pre-change fabric (197 cells, `make synth-equiv
> MOD=ddr5_xbar`) — the 1-lane path is the committed code verbatim, selected by generate.
> **Still to do for this rank:** the same treatment for `flash_xbar`, the N_REQ request-side
> crossbar, and wiring the wide response bus into the die (which is where the Vivado congestion
> risk actually lands, and rank 12's sizing rule applies at the CDC).
- **Modules:** `src/ddr5_xbar.v (req banking :154-166, single-grant RR drain :212-238), src/flash_xbar.v (:151-167, :274-276)`
- **Change:** Widen both fabrics from a single requester port + one-grant-per-cycle drain to: (a) an N_REQ front-end crossbar accepting up to one request per distinct target channel each cycle, and (b) a multi-drain arbiter that pops up to N_CH FIFO heads/cycle onto an N_CH*DATA_W (or N_LANE-wide) response bus, plus a small completion/reorder buffer (tags already return out-of-order). The fabric stays data-agnostic (opaque beats).
- **Why (perf model):** Roofline: to consume the ~100 GB/s the die needs, delivered beats/cycle must rise. Today N_CH raises only outstanding depth (a channel can be busy) but the single-grant drain still delivers 1 beat/cycle = 1x a single channel (~32 GB/s @1GHz); the other N_CH-1 completed reads just back up in per-channel FIFOs. This is the literal single-lane ceiling the ULTRA_PERF banner and P1.1 flag but leave at 'stripe across N channels'; the module-level refinement is that the RESPONSE ARBITER, not just the channel count, is the throttle.
- **Impact `[EST]`:** [EST] up to ~N_CH x sustained fabric BW (8-12x at DDR_NCH=8-12): converts the committed single-lane ~32 GB/s into the ~400-600 GB/s the header targets. Realizes rank 1's outstanding requests as actual bytes. Needs a Vivado re-fit (the wide resp bus is itself a route/congestion risk).
- **Contract:** preserving · **Effort:** large · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 3. Multi-port hot-weight issuer (stop serializing 5-7 weight families onto one requester port)

> **STATUS: fabric half BUILT + MEASURED (`make rank3`), default-off.** `ddr5_xbar` gained
> `REQ_LANES`: lanes aimed at **distinct channels are all accepted in the same cycle**, and when two
> lanes bank to the same channel the lower lane wins while the other's `req_ready` simply stays low —
> nothing is dropped or reordered. **Measured: 1.000 → 4.000 requests/cycle** at `REQ_LANES=4`
> (alongside 4.000 beats/cycle drained). `REQ_LANES=1` is **netlist-identical** to the committed
> fabric (197 cells) — the 1-lane banking is the committed code verbatim.
>
> **System half: BUILT + MEASURED — and the measurement says it buys NOTHING here (`make rank3sys`).**
> `glm_q4k_system` gained `SYS_REQ_LANES`: the L highest-priority pending families are handed to L
> independent fabric ports (lane l banks on `bank_rot+l`, because ef/load/slot/hot all bank on
> `bank_rot` and same-`bank_rot` lanes would collide on one channel and cancel the whole gain).
> `SYS_REQ_LANES=1` is **netlist-identical** to the committed system (`make resident-equiv`).
>
> | | requests | decode cycles | issue rate |
> |---|---|---|---|
> | `SYS_REQ_LANES=1` | 16,068 | 43,724 | **0.367** reqs/cycle |
> | `SYS_REQ_LANES=4` | 16,322 | **43,724** | **0.373** reqs/cycle |
>
> Identical decode cycles — per token too (9473 / 10420 / 11437 / 12394 both ways) — with every
> committed token still `==` the independent `glm_model_q4k` reference. **Four ports bought zero
> cycles.** The die only offers ~0.37 requests/cycle at this configuration, so the single-port
> issuer was never the limit; it is compute-bound, not request-port-bound. The 254 extra requests
> are parallel `TAG_HOT` traffic (a pure traffic generator — `TAG_HOT` is issued but matched by no
> response consumer), not work pulled earlier.
>
> **What this does NOT establish:** the measurement is one small slice (TN=4/PE_N=2-class,
> `N_EXPERT=8`, `DDR_NCH=4`, `RESIDENT=1`, 4 tokens). It does not show the lever is useless at the
> 753B geometry, where the compute/fetch ratio is different. It shows the *premise under which rank 3
> was written* — "the die can express at most 1 beat/cycle of weight demand, so the fabric's width is
> unreachable" — is **false at the configuration we can actually measure**. The parameter is kept,
> default-off and netlist-proven, so the claim can be re-tested rather than re-argued.
>
> **Found along the way (a real bug, unrelated to the perf question):** the perf TB's DDR model
> computed its free in-flight slot ONCE outside the per-channel loop, so two channels accepting in
> the same cycle both wrote the SAME slot — one read silently lost, never answered, die waits
> forever. Invisible while the fabric had one request port; an instant 2.5-hour hang the moment it
> had more. Fixed (distinct slot per accepting channel) and `mem_req_ready` is now capacity-aware, so
> exhaustion FAILs loudly instead of hanging. Same shape as the Flash backend's
> one-completion-per-cycle bug fixed earlier.
- **Modules:** `src/glm_q4k_system.v (priority issuer :1220,:1241-1250,:1289-1344; coalesced p_hot :1322)`
- **Change:** Replace the OR-reduced single p_hot / one xreq_valid handshake with per-family request queues (or a small outstanding-request table) feeding K parallel xbar requester ports, so multiple simultaneous die pulls (em/gn/aw/rw/fw/fn/lw, plus LOAD/SLOT/HOT/EFILL) map to distinct channel reads in the same cycle instead of one coalesced beat. At minimum issue LOAD+SLOT+HOT to distinct channels each cycle.
- **Why (perf model) — MEASURED FALSE at the tested configuration, see STATUS above:** Even with ranks 1-2 landed, the integrated top presents at most ONE banked read/cycle (mutually-exclusive one-hot selects; distinct hot addresses discarded, xreq_addr for TAG_HOT is just bank_rot placeholder), so the die can express at most 1 beat/cycle of weight demand regardless of DDR_NCH. This is the 'die must be WIDER to consume bandwidth' lever at the fabric seam: without it, the multi-lane fabric is bottlenecked here.
- **Impact — MEASURED 0 at the tested configuration** (was `[EST]` ~Kx expressible weight-fetch bandwidth; that estimate assumed the die was request-port-limited, and it is not here). Today observation-only (die pulls combinationally from a TB stub), so the realized number is gated on real-PHY bring-up.
- **Contract:** preserving · **Effort:** large · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 4. In-RTL deterministic prefetch during the norm/accumulate compute windows

> **STATUS: BUILT + MEASURED (`make rank4`), default-off.** The decoder now publishes the top-k
> union the cycle routing completes (`route_valid`/`route_set`, pure observation), and the system
> gained a `PF_EXACT` issuer that offers the rest of that union to the expert cache's prefetch port
> while expert 0 is still being evaluated. **Measured on a thrashing config (8 experts / 2 slots /
> `FLASH_LAT=256`): exposed stall 2,590 → 2,070 cycles (−20%)**, with token 1 going 259 → **0**
> (48/0 hits vs 47/1) and cyc/token 10,665 → 10,406. **Every committed token still matches the
> standalone reference** — this changes *when* bytes are fetched, never which bytes or their order.
> `PF_EXACT=0` (default) keeps the demand-only behaviour **and is netlist-identical to the committed
> system** — the first attempt was not: the decoder registered `route_set` unconditionally and the
> prefetch mux did not fold, which `make resident-equiv` caught. The exposure is now gated by
> `ROUTE_EXPOSE` (threaded from `PF_EXACT`), so at the default no registers and no mux are inferred. Composes multiplicatively with rank 1:
> the MSHR supplies the outstanding slots this prefetch would fill.
- **Modules:** `src/glm_decoder_block_q4k.v (sel_e captured whole at T_ROUTE :807-811; experts fetched serially T_ESCAN/T_EXPW/T_ACC :837-912), src/glm_q4k_system.v (episode detector on DEMAND cur_routed :826; expert_cache_pf.pf_valid/pf_expert_id are external inputs only :522-523)`
- **Change:** Drive expert_cache_pf.pf_valid/pf_expert_id INTERNALLY from the captured sel_e union the moment routing completes, so experts 1..NEVAL-1 stream from NVMe under expert-0's T_ACC; and issue the layer's aw_*/rw_* reads during the preceding ~5*MODEL_DIM cycles of pre-attn/pre-FFN RMSNorm + residual-add. Reorders fetch timing only; the die consumes the same bytes in the same order.
- **Why (perf model):** The die is strictly demand-pull: the episode detector fires the cycle the die asks, so every miss is fully exposed (EXPERT_STALL=1 freezes 3*FLASH_LAT+9). But the upcoming weight set is DETERMINISTIC and known early (top-k union at T_ROUTE; layer hot-weights from db_layer). This is the exact-router prefetch of ULTRA_PERF #8 / P3.1, made concrete in-RTL, and unlike the predictor (measured no-op) it is entropy-free — the set is known, not guessed. Composes multiplicatively with rank 1 (MSHR provides the outstanding slots the prefetch fills).
- **Impact `[EST]`:** [EST] hides (NEVAL-1) of NEVAL per-expert miss latencies behind ~NEVAL*MODEL_DIM compute + the attention/router miss behind ~5*MODEL_DIM of norms; cuts exposed EXPERT_STALL substantially at real FLASH_LAT. The 'keep NVMe busy / raise duty cycle' lever. Confirm with the EXPERT_STALL cycle sweep.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 5. Deepen the weight-loader / header read pipeline to Q outstanding reads
- **Modules:** `src/weight_loader_q4k.v (S_STREAM 1 beat/cycle, single code_pending marker :212,:496-511; S_SCALE single rd_v :183-184)`
- **Change:** Replace the latency-1 single-in-flight read (code_pending / rd_v) with a Q-deep tag/skid queue: issue reads ahead and consume returns in order, Q ~ round-trip latency. Pairs with ranks 2-3 so the loader actually drives N_CH channels.
- **Why (perf model):** Prefetch depth is exactly 1 today (assumes mem_data at t+1, a TB stub). Against real multi-hundred-cycle DDR/flash latency the loader stalls one full latency per beat and can never saturate a deep fabric (BW = outstanding/latency again). This is P1.3 made specific to the loader's read FSM.
- **Impact `[EST]`:** [EST] recovers the latency-bound gap: sustains ~1 beat/cycle (or N_LANE/cycle) instead of ~1/latency under real memory latency. Enabler for ranks 2-3's bandwidth to reach the GEMM.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 6. Register the MEASURED worst path: narrow registered expert-output read port
- **Modules:** `src/swiglu_expert_q4k.v (y_out wide reg :55; combinational egress), src/glm_decoder_block_q4k.v (em_y :441; far HIDDEN:1 mux sh_emy_f :616)`
- **Change:** Convert swiglu_expert_q4k's ~98 Kbit combinational y_out port into a narrow registered read port: add a y_rd_idx input and a registered [16*PE_M-1:0] y_rd_val output so the HIDDEN:1 mux lives INSIDE the expert next to its y_out storage and only 16*PE_M bits cross the module boundary. T_ACC already streams one comb_i/cycle; present comb_i at t, accumulate the registered read at t+1 (1-cycle fill, throughput unchanged).
- **Why (perf model):** This is the repo's single named routed-Fmax limiter: u_moe/y_out -> hbuf wide bus, 21.2 ns, 59% wire, pinning routed Fmax at 46.5 MHz (FPGA_DEMO_PLAN.md:53-54). It is route-dominated (physical), not arithmetic. Fmax is the lever the task names for ultra-perf: at 46.5 MHz consuming 100 GB/s needs ~4,300 dequant lanes (infeasible on KU3P); at 200 MHz-class ~1,000. Removing ~98 Kbit of wire + the remote mux from the cone is a plausible multi-ns cut on the whole clock.
- **Impact `[EST]`:** [EST] directly attacks the 46.5 MHz worst path; ~0 on today's bandwidth-bound B=1 tok/s (compute hidden under the fetch) but real for slice-demo wall-clock (tok/s_slice = Fmax/cyc_per_tok), prefill/TTFT, and the compute-bound rung-3 regime. Requires a Vivado re-fit to confirm the ns cut and that the next path is arithmetic, not another route.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 7. Register the leaf-level shapes of the same worst path (matmul c_out + silu*up merge)

> **STATUS: (a) BUILT — `glm_matmul_q4k` gained `REG_COUT` (`make rank7`).** Default `REG_COUT=0`
> is the committed behaviour and its synthesized netlist is proven **identical to the pre-change
> module** (2,555 cells, cell-for-cell). At `REG_COUT=1` the `PE_M*PE_N` rounders and the wide C bus
> are registered and `out_valid` moves with the data, so the 160-test ggml-Q4_K golden passes
> **bit-exact in BOTH settings** — the extra cycle is invisible to any consumer that waits on
> `out_valid`, which is all of them. **The Fmax gain remains `[EST]` until a Vivado re-fit measures
> it**, and enabling it in the product is gated on that measurement. (b) the SwiGLU `silu*up` merge
> register is not done.
- **Modules:** `src/glm_matmul_q4k.v (combinational c_out fp32_to_bf16 fanout :284-287), src/swiglu_expert_q4k.v (combinational bf16_mul silu*up feeding hbuf :205-216)`
- **Change:** (a) Register c_out at the glm_matmul_q4k leaf boundary (one flop bank of PE_M*PE_N fp32_to_bf16 results) and delay out_valid by 1 cycle so consumers capture on the same edge; the wide bus becomes a reg->reg segment P&R can place. (b) Register the bf16_mul(act_y, up_hold) result one beat before the hbuf write (the FSM already idles in the merge cycle, so +1 latency is free).
- **Why (perf model):** Both are the same combinational-wide-fanout-into-hbuf shape as rank 6, one module deeper: the matmul c_out rounders and the SwiGLU merge multiply both fan out combinationally onto the inter-module wire that becomes the 46.5 MHz path. Registering them moves the arithmetic off the route and lets the repipeline campaign (already 4.6x) continue. Small effort, bit-exact, composes with rank 6.
- **Impact `[EST]`:** [EST] shortens the same named worst path; Fmax gain compounds with rank 6 and the closed 4.6x campaign. +1 cycle latency each, throughput unchanged. Confirm via Vivado re-fit.
- **Contract:** preserving · **Effort:** small · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 8. Widen the single-lane scalar bf16 tail + rmsnorm to W lanes
- **Modules:** `src/glm_decoder_block_q4k.v (SHN datapath T_RADD1/T_RADD2/T_ACC/T_FCOMB, 1 elem/cycle :611-618,:750-934; LANES=1 rmsnorm_units :323), src/rmsnorm_unit.v (LANES knob)`
- **Change:** Widen the shared fp32-add/narrow SHN datapath and both rmsnorm_units to W lanes: process W independent elements/cycle in the residual adds, expert scale+accumulate, finalize, and the norm reduce/normalize passes. Elements are cross-independent so W parallel adders/narrows cut each phase to MODEL_DIM/W cycles with no reordering.
- **Why (perf model):** The elementwise tail is ~12-14*MODEL_DIM single-lane cycles/MoE layer (~80K cyc at MODEL_DIM=6144) and is a big share of the measured slice cyc_per_tok~10,896. Lowering cyc_per_tok raises tok/s_slice = Fmax/cyc_per_tok directly AND raises the die's duty cycle so it can consume more fetched bytes/sec (the 'make the die wider' lever). Unlike GEMM widening, this is pure elementwise so it is trivially bit-exact.
- **Impact `[EST]`:** [EST] ~Wx fewer tail cycles/layer (W=8 -> ~80K->~10K). Big win for slice-demo tok/s, prefill/TTFT, and compute-bound rung-3; ~0 on B=1 decode ONLY once the tail is already hidden under fetch, but today the tail is a real cyc_per_tok term so it helps the slice number now.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** new

### 9. SwiGLU gate||up concurrency + inter-pass double-buffering

> **STATUS: BUILT + MEASURED (`make rank9`, `make rank9-measure`), default-off. The predicted
> ceiling was realised EXACTLY.** `swiglu_expert_q4k` gained `GU_CONC`; at 1 a second
> `glm_matmul_q4k` runs the up GEMV concurrently with the gate GEMV off the same `mm_start` /
> `mm_in_valid` / `a_col`, so `S_UPP`/`S_UP`/`S_UPW` are never entered and `up_pass` never rises
> (which *removes* the four operand muxes at `:97-100` rather than adding any).
>
> | | `ffnd` | `expw` | decode cycles |
> |---|---|---|---|
> | `GU_CONC=0` | 5,560 | 8,852 | 43,724 |
> | `GU_CONC=1` | **4,088** | **6,644** | **40,044** |
>
> **−3,680 cycles = 8.42% of the decode window.** Three things agree independently:
> (a) the predicted ceiling was `23 cyc × 160 groups = 3,680` and the measurement is **exactly
> 3,680**; (b) the decode window fell by **exactly the same 3,680**, so nothing downstream absorbed
> any of it; (c) the saving decomposes cleanly — dense `1,472/8 = 184 = 8 groups × 23`, MoE
> `2,208/24 = 92 = 4 groups × 23`, total `8×8 + 24×4 = 160` groups.
>
> **Contrast with rank 3, because the contrast is the lesson.** Rank 3's arithmetic was also
> correct and its measured effect was still ZERO cycles: the die simply did not have the requests
> to fill the widened port. Here the serialisation removed *was* on the critical path. Being able
> to compute a ceiling says nothing about whether removing that thing helps; only the measurement does.
>
> **Proof, and why the usual one could not be used.** The perf TB's binding check compares
> `glm_q4k_system` against a standalone `glm_model_q4k` and **both instantiate
> `swiglu_expert_q4k`** — a leaf parameter moves both sides identically, so that check is BLIND
> here. It was valid for ranks 3 and 4 only because those changed `glm_q4k_system`, which the
> reference does not contain. Bit-exactness is therefore proven against the **numpy golden**
> (`tools/glm_model_q4k_ref.py`), which contains no RTL at all: **1155/1155 at both `GU_CONC=0` and
> `GU_CONC=1`**. Netlist: `GU_CONC=0` is **IDENTICAL** to the pinned base (13,688 cells) and
> `GU_CONC=1` **DIFFERS** (liveness — without that second half the first proves nothing);
> `glm_q4k_system.v` stays PROVEN against `05639bf` after the parameter threading.
> Injection `-DINJ_GU_WRONGUP` captures the gate result as `up` (giving `silu(gate)·gate`) and must
> fail the golden — otherwise the golden is not constraining the concurrent engine's *value*.
>
> Bit-exactness is structural: during `S_UP` the committed FSM re-issues the **identical** request
> key (`:166` changes only `up_pass`; `pass_sel`, `w_grp = grp` and `w_k` restarting at 0 are all
> unchanged), so the up pass is a byte-for-byte replay of the gate pass. Merging them issues a
> strict **subset** of today's weight requests — each `(sel,grp,k)` once instead of twice — and
> never a different one.
>
> **Found while building this:** a per-cycle `$fatal` is not synthesizable (yosys: "Can't resolve
> task name `$fatal`"); every other `$fatal` in `src/` sits in an `initial` block, which yosys
> tolerates. It is guarded with `` `ifndef YOSYS ``, **not** the repo's `` `ifndef SYNTHESIS `` —
> nothing in this repo ever defines `SYNTHESIS` (zero hits across the Makefile and `tools/*.sh`),
> so that guard does not actually exclude anything from synthesis.
>
> **Stage 2 (`GU_PIPE`) was NOT built** — `glm_act`'s `HW_LANES` wrapper drops `in_valid` while
> running, so it would require `ACT_HW==0` and be silently wrong in the configuration the FPGA fit
> was measured in.


- **Modules:** `src/swiglu_expert_q4k.v (serial S_GATE then S_UP on one shared matmul :162-173; drain-stall S_GATEW/S_UPW/S_DNW :164-195)`
- **Change:** (a) Add a second glm_matmul_q4k (or widen PE_N to consume w_q|w_q_up in one pass) so the gate and up GEMVs — structurally independent over the same activation, with both weight codes already arriving every beat — run concurrently instead of time-multiplexed. (b) Double-buffer the group accumulator/weight-scale latch so the next group's K-stream launches while the current group drains and its silu*up merge completes.
- **Why (perf model):** Gate+up are 2*HIDDEN cycles/group today where HIDDEN would suffice, and each pass eats a full FP-pipe drain bubble between groups (repeated NG_GU+NG_D times/expert). Halving the dominant FFN compute term raises the compute ceiling so more HBM bandwidth can be consumed at PE_M>1 — a prefill/TTFT and duty-cycle lever. Each GEMV keeps its own fp32 K-order accumulation, so bit-exact.
- **Impact `[EST]`:** [EST] ~2x fewer cycles on the SwiGLU gate/up phase (the dominant FFN term) + removal of (NG_GU+NG_D)*L_drain bubbles/expert. Costs one extra matmul's LUTs/dequant (area on the die, which raises congestion — watch the rank 6/7 Fmax path).
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** new

### 10. Run the router GEMV concurrently with the always-on shared expert
- **Modules:** `src/moe_router_q4k.v (serial GEMV->sigmoid->topk->renorm->recip->mul :169-178,:408-482), src/glm_decoder_block_q4k.v (shared expert after T_ROUTE :509)`
- **Change:** Launch the shared-expert swiglu on nrm=RMSNorm(h) in PARALLEL with the router gate GEMV (the instances already exist: u_dense/u_moe vs u_router). Both consume the same nrm and are independent, so the router's entire HIDDEN-cycle GEMV + sigmoid/topk/renorm tail hides under the much longer shared-expert eval.
- **Why (perf model):** The shared expert needs no routing decision yet is serialized behind the router today. Overlapping two already-instantiated engines is exactly the sequential-phase de-serialization the task calls for ('overlap so lanes scale linearly not ~2.40x'). Results independent -> bit-exact.
- **Impact `[EST]`:** [EST] hides ~HIDDEN + N_EXPERT + tail cycles per MoE layer under shared-expert compute; a per-layer duty-cycle/TTFT lever, model-faithful. Multiplies across 75 MoE layers.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** new

### 11. Fold running argmax + overlap LM-head tile drain (kill the per-token VOCAB serial tails)
- **Modules:** `src/mtp_head_q4k.v (S_LMWAIT then disjoint S_ARGMAX scan :679-711), src/glm_model_q4k.v (M_LMTILE/M_LMWAIT drain ping-pong :582-629; M_ARGMAX VOCAB scan :634-651; serial M_EMBED :504-526)`
- **Change:** (a) Fold a RUNNING argmax into S_LMWAIT/M_LMTILE: as each LM_TN-wide logit tile lands, compare against the per-row running best (bf16_gt, lower-index tie-break preserved by tile order) so the separate VOCAB-cycle scan disappears. (b) Overlap the next vtile's K-stream with the current tile's matmul drain (glm_matmul_pipe accepts a new start while draining). (c) Widen the embedding load to multiple elements/cycle.
- **Why (perf model):** The argmax is a disjoint ~VOCAB-cycle (154,880 at real vocab) serial phase after the full LM-head GEMV, and each of NVTILE tiles pays a full pipe drain with no double-buffering. This is once-per-token, so it directly amortizes the speculative K-token verify cost (each drafted/verified token pays it). bf16 compare, tile-ordered -> bit-exact.
- **Impact `[EST]`:** [EST] removes the ~155K-cycle argmax + ~NVTILE drain bubbles from every MTP/verify and every decode token; material for slice cyc_per_tok, prefill, and spec-decode verify latency. Secondary to the per-layer loop but real per token.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** new

### 12. CDC crossing at the wide beat (don't re-serialize the widened fabric)

> **STATUS: VERIFIED + MEASURED (`make rank12`).** `cdc_async_fifo` is already
> `DATA_W`-parameterized, so the wide crossing needed confirming, not building — and the
> confirmation found the part that is *not* automatic. **Width:** a 256-bit (8×32b) lane-tagged
> beat crosses **atomically and in order**, checked per lane. **Depth:** the crossing sustains
> **1.003 beats/cycle at depth 16** — but the same test at **depth 4 falls to 0.617 beats/cycle
> with zero data loss**, which is exactly the silent re-serialization this rank warns about: four
> entries cannot cover the two 2-FF synchronizer hops, so `full` throttles the writer while every
> functional check still passes. So the actionable output for ranks 2–3 is a **sizing rule, not a
> code change**: depth must cover the cross-domain round trip. (Also found: `ADDR_W ≥ 2` is a
> structural floor — `ADDR_W=1` does not elaborate.)
- **Modules:** `src/cdc_async_fifo.v (1 word/rclk edge :72-88,:190-198)`
- **Change:** Instantiate the core<->die clock crossing at the WIDE beat (N_LANE*DATA_W) or as N_LANE parallel FIFOs, and size ADDR_W for the cross-domain round-trip so it never back-pressures.
- **Why (perf model):** A single narrow CDC FIFO between the widened fabric (ranks 2-3) and the die would re-choke the crossing to 1 word/cycle, undoing the N_CH win. Pure enabler for the fabric-width levers.
- **Impact `[EST]`:** [EST] prevents the CDC boundary from capping a widened fabric at 1 beat/cycle. Enabler, not a standalone speedup.
- **Contract:** preserving · **Effort:** small · **vs ULTRA_PERF:** new

### 13. Pipeline the topk_select tournament (implement the advertised TREE_PIPE registers)
- **Modules:** `src/topk_select.v (single-cycle NLEV-deep fp32_gt cone :255-289; S_EXTR :360-378)`
- **Change:** Actually insert the TREE_PIPE register banks the header claims: pipeline the argmax tournament every TREE_PIPE levels so each extraction pass costs ceil(depth/TREE_PIPE) short cycles instead of one O(log N)-deep combinational cone. Left-child tie-break is preserved by register insertion. Give the FSM the extra cycles.
- **Why (perf model):** The shared selector is a single-cycle 8-deep (router N=256) / 11-deep (indexer N=2048) fp32_gt cone that retiming cannot fix (no registers exist, 1-cycle FSM budget). It is a real Fmax limiter on both the router top-8 and the DSA indexer. Bit-exact (register insertion only).
- **Impact `[EST]`:** [EST] removes an 8-11-deep comparator cone from the critical path; adds K*depth latency (negligible for router K=8). Fmax help alongside ranks 6-7.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 14. Pipeline the softmax shift/normalize passes to 1 issue/cycle

> **STATUS: BUILT + MEASURED (`make rank14`), default-off. The only remaining plan that survived
> adversarial review.** `glm_softmax` gained `SM_PIPE`; `SM_PIPE=1` deletes exactly two tokens —
> `!sh_busy && ` at `:294` and `!nm_busy && ` at `:418` — so both passes issue one op per cycle.
> **Measured: 2,892 → 2,136 cycles over a 12-row campaign (1.354×)**, and every output word is
> **identical as a raw 16-bit pattern** to the `SM_PIPE=0` DUT. `SM_PIPE=0` is **netlist-identical**
> to the committed module (2,400 cells, `tools/synth_equiv.sh`) — the committed block is kept
> byte-for-byte inside the generate branch rather than folding a `SM_PIPE ?` term into the two
> conditions, because that fold is the shape that has broken default netlist identity three times here.
>
> Why it is bit-exact, structurally (not by testing luck): `fp32_add_pipe`/`fp32_mul_pipe`
> (`glm_fp_pipe.v:516-563`, `:192-207`) are flat feed-forward register chains with the valid bit
> shifted alongside the data — no enable, no per-op shared state — so back-to-back ops give the same
> per-op results as isolated ops, in issue order. The capture already indexes by the **in-order**
> counters (`sh_out_cnt`/`nm_out_cnt`). No RAW hazard on the in-place `xbuf` reuse: index *i* is read
> at `P0+i` and written back at `P0+FP_ADD_LAT+1+i = P0+6+i`, so the read leads the write by 6 for
> every *i* and every LEN. The same 1-issue/cycle idiom is already committed and bit-exact in
> `S_EXP` (`:330-339`).
>
> The test is **DUT-vs-DUT with `===`, not the committed golden**: `test/glm_softmax_tb.v` compares
> against a numeric golden with several bf16 ULP of tolerance and therefore *cannot* prove
> bit-exactness. Paired injection `-DINJ_SM_PIPE_OOO` captures at the issue index instead of the
> in-order index (the issue counter runs 6 ahead once the interlock is gone) and **correctly fails
> with 83 mismatches** — so the identity check is load-bearing.
>
> **Two of this section's original claims below are WRONG, and are left visible rather than edited
> away:** (a) "Matters at long context (softmax per head over sequence length S)" — `glm_softmax`'s
> `LEN` is `SWIN = min(S_MAX, TOPK_ATTN)` (`mla_attn_q4k.v:123`), the DSA top-K window, **never** the
> sequence length. This is a top-K-window lever, not a long-context one. (b) the "~1/5 and ~1/2 of
> peak, net latency ~13·LEN" rates are wrong; the true per-row cost is `21·LEN + 84`.
>
> **Scope of the win, stated honestly:** `soft` = 5,376 cycles = 32 rows × 168 cycles exactly at the
> measured slice, so the ceiling in the decode window is **864 cycles ≈ 1.98%** — real, verified,
> and small. `SM_PIPE` is not yet threaded to `glm_q4k_system`, so no decode-window number is
> claimed here; the 1.354× is a module-level measurement.

- **Modules:** `src/glm_softmax.v (S_SHIFT drains 1 op at a time :291-318; S_EXP serial sum :344-364; S_NORM :416-434)`
- **Change:** Pipeline the shift pass and the normalize pass at 1 issue/cycle with in-order capture counters (elements are independent -> bit-exact). Keep the sum reduction in sequential K-order but let it fold as exp results stream, or accept an output-changing interleaved-partial sum for full 1/cycle.
- **Why (perf model):** The passes drain the pipelined fp32_add_pipe/fp32_mul_pipe one op at a time, running them at ~1/5 and ~1/2 of peak; net latency ~13*LEN where independent elements could stream. Matters at long context (softmax per head over sequence length S), a long-context latency lever, not B=1 HBM decode.
- **Impact `[EST]`:** [EST] ~2.5x softmax latency cut (shift+norm, bit-exact); up to ~5x if the sum is also parallelized (that part is output-changing).
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 15. Parallelize + de-serialize the DSA indexer (long-context wall only)
- **Modules:** `src/dsa_indexer.v (single shared fp32_mac_pipe :195-199; serial S_FETCH/S_SCORE/S_TKL :389-497)`
- **Change:** (a) Instantiate P independent fp32_mac_pipe units, partition the LANES key-set into P disjoint subsets (each key walks its OWN in-order term chain feeding its OWN pipe), merge the per-pipe tag-FIFOs -> ~P term/cycle, each key's fp32 FMA order untouched -> bit-exact. (b) Double-buffer the lane key-vectors so group g+1 fetches while group g scores, and stream finished scores into topk_select's load port as grp_all_done fires so top-K overlaps scoring.
- **Why (perf model):** Throughput is capped at 1 MAC term/cycle by the single pipe (LANES only hides latency); scoring is ~S*IDX_DIM cycles, the O(S) long-context attention wall (in-order indexer ~0.05 tok/s at 1M ctx per ULTRA_PERF #11). This is the bit-exact P-pipe variant of #11 plus the fetch/topk overlap bubbles. No effect on B=1 bandwidth-bound decode — the real lever only where the indexer, not HBM, is the wall.
- **Impact `[EST]`:** [EST] ~Px cut in the O(S)*IDX_DIM scoring wall at 1M context (P=4 -> ~4x) + removal of the fetch/topk-drain bubbles (~10-25%). Long-context regime only.
- **Contract:** preserving · **Effort:** medium · **vs ULTRA_PERF:** refines-ULTRA_PERF

### 16. MLA per-key prefetch + concurrent W_uk||W_uv + two-engine overlap
- **Modules:** `src/mla_attn_q4k.v (serial per-key K_RDREQ..K_SCORE loop :908-917; mutually-exclusive u_mm_fp8/u_mm_bf16 by gv_score :727-744; 13 serial phases :870-885)`
- **Change:** (a) Prefetch the next union key's c_kv/k_rope during the current key's UK/UV/score (the kc_* port is idle then). (b) Run W_uk||W_uv concurrently (second engine or fused PE_N) since both read the shared ckv_n. (c) Overlap the bf16 score engine of already-projected keys with the Q4_K W_o output projection of the accumulating context. Keep each accumulation's operand order to stay bit-exact.
- **Why (perf model):** The S_KEY loop serializes cache-read latency of key s+1 behind projection/score of key s, and W_uk/W_uv run sequentially though both consume ckv_n; only one of the two GEMV engines is ever active. At real TOPK_ATTN=2048 this loop is the sparse-attention cost driver and the module-local face of sublinear lane scaling. The cross-phase reorderings that preserve operand order are bit-exact; the two-engine overlap must be checked against the golden to confirm no accumulation-order change (mark output-changing if it reorders any sum).
- **Impact `[EST]`:** [EST] hides per-key cache latency across u_cnt keys and ~halves per-key projection; raises die occupancy so lanes scale nearer linear at PE_M batch. Long-context / sparse-attention wall, not B=1 HBM decode.
- **Contract:** preserving · **Effort:** large · **vs ULTRA_PERF:** new

### 17. [OUTPUT-CHANGING] L-way interleaved matmul accumulate (remove the loop-carried fp32_add)
- **Modules:** `src/glm_matmul_q4k.v (loop-carried single-cycle fp32_add on vp[3] :268-270), tools/q4k_ref.py (golden rebaseline)`
- **Change:** Adopt the L-way interleaved partial-sum accumulate already proven in the bf16 twin glm_matmul_pipe: L sub-accumulators each updated once every L cycles through the pipelined adder, then a log2(L) reduce tree. Removes the recurrence. REQUIRES rebaselining tools/q4k_ref.py to the L-way fp32 grouping and re-running the 1155-test golden.
- **Why (perf model):** A whole combinational fp32_add (variable-align + 28b add + LZ-normalize + RNE round) sits in the loop-carried critical path, one add/beat. It is the core's arithmetic Fmax ceiling (~37 MHz-class add cone per glm_matmul_pipe's own history) that becomes the wall ONCE the routing wall (ranks 6-7) is cleared. It is bit-exactness-locked to q4k_ref.py's sequential K-order, so it cannot be pipelined in place without changing outputs.
- **Impact `[EST]`:** [EST] ~3-5x the matmul core's arithmetic Fmax ceiling; unifies the Q4_K and bf16 datapaths. Breaks the bit-exact contract (fp32 add is non-associative) -> must be re-verified against a rebaselined golden and re-checked vs GGUF crosscheck tolerance. Only pursue after the route wall is the confirmed limiter.
- **Contract:** output-changing · **Effort:** medium · **vs ULTRA_PERF:** new

### 18. [OUTPUT-CHANGING] Single-Newton reciprocal in glm_act (shrink the die's largest block)
- **Modules:** `src/glm_act.v (2-iteration Quake rsqrt, R1..R10, 20-stage LAT=20 :431-451)`
- **Change:** Drop the sigmoid/silu 1/(1+exp) reciprocal from two Newton iterations to one (or a direct 1/d Newton skipping the final square), since d=1+exp(z)>=1 and the bf16 output needs only ~2^-8 while the current path provisions <2^-22.
- **Why (perf model):** The router instantiates LANES=N_EXPERT*PE_M of this core (~48K LUT, the single largest block; ACT_HW=1 lane-serializes it). Routed util is 87.5% and the worst path is 59% wire -> congestion-driven, so halving the reciprocal datapath area on the biggest unit directly relieves the route-dominated worst path (indirect Fmax help). Costs a few ULPs -> OUTPUT-CHANGING, must be validated against the accuracy contract.
- **Impact `[EST]`:** [EST] large area cut on the largest block -> lower congestion -> indirect Fmax help via ranks 6-7; a quality knob (few ULPs on the activation). Not a standalone tok/s number; measure the congestion/Fmax delta in Vivado and the accuracy delta vs golden.
- **Contract:** output-changing · **Effort:** medium · **vs ULTRA_PERF:** new

## Clusters (how to sequence the work)

The ranked items group into three engineering clusters, best executed in order:

1. **Fetch-path (ranks 1–5) — the tok/s ceiling for the single-user product.** Non-blocking MSHR
   expert cache, multi-drain fabric, multi-port hot issuer, in-RTL deterministic prefetch of the known
   top-k union, deep loader read queue. All **bit-exact-preserving** (reorder/parallelize fetches, not
   arithmetic). Little's-law depth + single-lane seams — not channel count — throttle delivered BW.
2. **Fmax (ranks 6–7, 13) — the precondition to ever consume 100 GB/s.** Register the one *measured*
   routed limiter (the ~98 Kbit combinational expert-output bus into a far mux) and the identical leaf
   shapes (`glm_matmul_q4k.c_out`, the SwiGLU merge), pipeline the `topk_select` tournament. **~0 on
   today's B=1 bandwidth-bound decode**, but at 46.5 MHz consuming 100 GB/s needs ~4,300 dequant lanes
   (infeasible); each of these is bit-exact and needs a Vivado re-fit to confirm the ns cut.
3. **De-serialize + widen the compute tail (ranks 8–12, 14–16).** Single-lane bf16 tail → W lanes,
   gate‖up concurrency, router‖shared-expert overlap, running argmax, CDC at the wide beat, softmax/
   indexer/MLA pipelining. Cuts the measured `cyc_per_tok ≈ 10,896` and raises duty cycle — biggest for
   prefill/TTFT, slice-demo wall-clock, and the compute-bound rung-③ regime.

The two **output-changing** levers (rank 17 L-way interleaved fp32 accumulate, rank 18 single-Newton
reciprocal) are ranked last and **gated behind a golden rebaseline + GGUF-crosscheck re-check** — they
break the bit-exact contract and only pay off once the routing wall (cluster 2) is the confirmed limiter.

## Dropped / folded (not re-proposed)

- weight_loader_q4k decouple Q4_K(4b/col) from mixed(16b/col) bus width to reach PE_N=64 at DATA_W=256 -- DROPPED as likely already resolved: docs/IMPROVEMENT_PLAN.md P1.1 states the weight-loader PE_N<=16 lane cap is 'root-caused and fixed (loader->GEMM bit-exact through PE_N=64; make weight-loader-lanes gates PE_N=32)'. The mixed-type 16*PE_N<=DATA_W guard is very likely the same cap; needs a one-line confirm against the current elaboration guard before re-proposing.
- axi_master_dma.v full-AXI4 bursts + multiple outstanding IDs + 256b bus -- DROPPED as ~0 tok/s: weight and KV traffic flow through ddr5_xbar/flash_xbar/weight_loader, not this AXI4-Lite port, so it is control-plane only. The finding itself concludes 'if it stays control-plane only, document that and keep weight/KV traffic off it' -- so the correct action is a comment, not a datapath change; no roofline impact.
- dsa_indexer double-buffer fetch + overlap topk drain -- FOLDED into rank 15(b) rather than listed separately (same module, same long-context regime, complementary to the P-pipe parallelization).
- mla_attn two-engine cross-phase overlap (was tagged bitexact:unclear) -- FOLDED into rank 16(c) with an explicit honesty caveat that any reordering of an accumulation's operand order flips it to output-changing and must be golden-checked.
- swiglu_expert inter-pass double-buffering -- FOLDED into rank 9(b) (same module, same prefill lever as gate||up).
- glm_q4k_system.v §8 multi-wide DDR issuer -- MERGED with the DDR fast-tier issuer finding into rank 3 (identical single-requester-port serializer, same fix).

---

*Generated from a bottom-up 4-tier RTL microarchitecture review (arithmetic leaves → memory fabric →
attention/MoE/spec → system top) with adversarial synthesis. Every item cites the RTL it came from;
confirm each Fmax/bandwidth `[EST]` with a Vivado re-fit or the `EXPERT_STALL`/`perf-q4k` cycle sweep
before quoting it as measured.*
