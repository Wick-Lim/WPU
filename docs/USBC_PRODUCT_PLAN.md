# USB-C Personal AI Appliance — Productization Plan

**What this is.** The concrete plan to turn the verified accelerator into a **shippable USB-C
external device**: a small, self-powered, active-cooled box that runs the full GLM-5.2 (753B-param
MoE, ~40B active/token) from its published **Q4_K GGUF** (`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`,
~467 GB) locally — **fully offline / air-gapped, no internet ever** — and streams tokens to a host
computer over a single USB-C cable.

**Relationship to the other roadmaps (read together):**
- [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) — the **performance ladder** (the anchor for every tok/s
  headline here). Throughput is set by **memory bandwidth**, which is set by the FPGA/silicon's IO +
  PHY, which is set by budget — so speed is **staged across rungs**: ① prove-it FPGA ~5–8 · ② funded
  custom board ~15–40 · ③ SoC/ASIC at volume design point **≈80 tok/s [measured-inputs EST]** (updated 2026-07 — the rung-③
  primary design point pivoted to **full residency**: 512 GB LPDDR5X holds the whole ~467 GB
  checkpoint; see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)). Read every tok/s in this doc as
  **rung-dependent** per that ladder.
- [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) — the **RTL / silicon track** (P1–P4: real-model
  fidelity, full-scale, robustness, vendor-IP + FPGA physical). *That* makes the chip correct and
  real. **This** doc is the **device / appliance track** on top of it: form factor, power, thermal,
  host software, enclosure, manufacturing, go-to-market.
- [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md) — the single-package memory/tiering + cost
  model (DDR5 fast tier + 1–4 TB NVMe bulk store; USB-C carries only token IDs).
- [`LOW_POWER.md`](LOW_POWER.md) — the energy budget + the byte-identical low-power levers (the ~80 %
  storage-read-byte reality — NVMe/PCIe reads dominate per-token energy; spec ÷K, compression, DVFS).
- [`OPERATION_FLOW.md`](OPERATION_FLOW.md) — how one token flows end-to-end.

> **All power/cost/size figures here are [EST]** — modeled, not measured on silicon or a board.
> The **FPGA fit is now MEASURED** (D0.2 DONE — Vivado ML 2026.1 routed fit of `glm_q4k_system_cdc`
> on **XCKU3P**, compact config + ACT_HW=1: 142,320 LUT / 87.5 % (synth-stage; routed 141,298 LUT,
> `fpga/results/util_routed_ku3p_acthw1.rpt`), ~100K FF, 421 DSP, 0 BRAM, hold
> met, routed Fmax **46.5 MHz** after a closed bit-exact repipelining campaign — see
> [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); that pins the FPGA class (KU3P-class) and unblocks
> size, thermal, BOM and power. **Board bring-up (D1) is still open**; the remaining D0 de-risk is
> real-GGUF fidelity (D0.1).

---

## 1. Product definition

**One-liner:** *frontier AI that works with the ethernet unplugged* — plug a USB-C box into the
computer you already own and run the full 753 B model **fully offline / air-gapped**. Nothing
leaves because there's no path out (the audit is literally *"does it still work with the ethernet
unplugged?"* — yes), and no vendor can rate-limit, deprecate, or cut you off — so private and
subscription-free come as the *result*.

| | |
|---|---|
| **Form factor** | small active-cooled external box (external-SSD → mini-PC sized), self-powered, **USB-C data link** to host |
| **What it runs** | the full GLM-5.2 (753B MoE) from its published Q4_K GGUF (`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`, ~467 GB) — the format local inference (llama.cpp) actually runs. Q4_K GEMM core is **bit-exact to the ggml-Q4_K reference `tools/q4k_ref.py`**, and that reference is itself **proven bitwise-equal to the real downloaded GGUF bytes at the dequant layer** (376,586,240 weights vs llama.cpp's own kernels — `docs/GGUF_CROSSCHECK.md`, closed 2026-07-10/11); llama.cpp *whole-runtime* numeric equality stays out-of-contract (op orders differ); the mixed-type (Q6_K/Q8_0/F16) RTL consumers are now **DONE** (`make mixedtype` — see §2) |
| **Throughput** | **rung-dependent** [EST] — ~5–8 tok/s on the near-term **prove-it** FPGA, ~15–40 on the **funded custom board** (the old flat ~25–40 was this rung-② number); staged by memory bandwidth per [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md). *(Update — measured-proxy design points, [`H_MEASUREMENT.md`](H_MEASUREMENT.md): NVMe-only ~0.5–1; 90 GB DRAM + 100 GB/s ~13–24; 90 GB + 200 GB/s ~25–47; 225 GB + 200 GB/s ~54–127 tok/s [EST] — updated 2026-07: these **streaming** points now apply to rung ① / the hybrid upside SKU / >512 GB checkpoints; the rung-③ primary is **full residency, design point ≈80 tok/s [measured-inputs EST]** ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)); spec multiplier = A/U(K), U now GLM-4.5-Air measured — U(4)=2.60–2.71, superseding the OLMoE proxy)* |
| **Power** | **≥50–78 W** (v3-volume residency SKU, self-powered) · **≥64–99 W** v3-proto · eco/throttle ~30 W and 15 W travel mode **[UNVERIFIED — no static/self-refresh model for 480 GB exists]** — all **[EST, 재도출 2026-07]**, all **floors with an UNVERIFIED SoC term on top**, not budgets. See §7 / [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §4. *(Old published: ~40–60 / ~50–80 W — retired, never derived. The old ~80–110 W was the pre-pivot streaming SKU: the new floor overlaps it by coincidence of arithmetic, NOT because the pivot came undone — see §7.)* |
| **Interface** | USB-C (USB 3.2 Gen 2 is ample — only token IDs cross; heavy traffic stays internal) |
| **Host** | thin driver + a local OpenAI-compatible endpoint → existing chat UIs / editors point at it |
| **Target user** | air-gap / offline-mandated users (SCIF, defense-forward, isolated OT/critical-infra, field/edge), privacy-critical individuals, local-AI enthusiasts, cost-heavy power users |
| **What it is NOT** | not a bus-powered dongle, not a multi-user server, not a general computer, not multi-modal |

**Why the form factor fits the architecture.** The heavy traffic (NVMe↔DDR5↔die, ~25 GB/token —
40B active × ~0.6 B/param at Q4_K) is **entirely internal**; USB-C carries only the text token
stream (a few bytes/token) + control. So USB-C bandwidth is a **non-issue** and the box is genuinely
self-contained. The production top `glm_q4k_system_cdc` **already carries the 2-clock host/USB CDC**
(structural sign-off via `make synth-glm`, exit 0 — **elaboration, not a sim**; the CDC/reset
architecture — `cdc_async_fifo` gray-pointer + `reset_sync` — is formally sound) — the device
interface was designed in from the start.

**Positioning.** Unlike a Mac Studio (replace your computer) or a consumer GPU (can't hold/stream
467 GB at all), this is an **accessory that adds frontier-model capability to the machine you
already have** — closer in spirit to an eGPU, but sipping desktop-PC power and running a far bigger
model. It creates a new category: an **external frontier-LLM accelerator** (vs. edge-AI USB sticks
like Coral/Hailo, which only run tiny models). And unlike **any** cloud option — including
"secured cloud" (in-VPC/tenant deployment, zero-retention APIs, confidential-computing/TEE
enclaves) — it runs **with the ethernet unplugged**; every cloud variant still needs connectivity
and fails that test, which is what unlocks the air-gap / offline environments others can't serve
(SCIFs, defense-forward ops, isolated OT/critical-infra, field/edge). The moat is the
**combination — offline + full frontier (753 B) + appliance price** — since offline *alone* is
table-stakes (a 70 B laptop model is offline too): the 70 B laptop fails frontier quality, 8×H100
fails price/form-factor, and secured cloud fails the unplugged test.

---

## UX 비전 (2026-07 확정 방향) — "꽂으면 뜬다", 전부 박스 안에서

**타깃 사용자**: 인터넷이 차단된 고보안 조직([`ICP.md`](ICP.md) — 오프라인이
*필수*인 구매자). 핵심 설계 원칙: **호스트에 아무것도 설치하지 않는다.**

1. **USB-C = 네트워크 장치(CDC-NCM/RNDIS)** — 드라이버/설치 없이 잡히고, 박스가
   자체 웹서버로 `http://wpu.local`에 **에이전트 UI를 서빙**. 윈도우/맥/리눅스
   공통, 잠긴 환경의 소프트웨어 설치 승인 문제가 원천 소멸(보안 검토 대상이
   "USB 장치 1개"). 같은 링크로 OpenAI-호환 API 병행 노출(IDE/도구 연동,
   `host/` 스캐폴드가 씨앗).
2. **대화 이력 → NVMe** (모델 467GB 제외 ~530GB 여유) — 수년치 로그+임베딩.
3. **로컬 RAG**: 소형 임베딩 모델(~1–2GB)을 DRAM 여유(KV 예산 45GB 내)에 상주,
   조직 문서+자기 이력 인덱스는 NVMe. **데이터가 박스 밖으로 0바이트.**
4. **시각화**: 대화/문서 지식그래프·타임라인을 박스 UI가 렌더.
5. **GUI 튜닝**: 샘플링·시스템 프롬프트·RAG 범위 + **적응형 스펙체인 텔레메트리**
   (spec_decode_seq의 pass_acc/pass_dep 탭 → 수락률·깊이 실시간 그래프).
6. **캐싱은 무조건·투명 (설계 결정 2026-07)**: 서버가 아니라 개인 장치이므로
   캐시가 옵션/튜너블이 아니라 **상시 기본값** — 모든 컨텍스트 KV 상주(45GB
   ≈50만 토큰), 프리픽스 캐시 상시 on(시스템 프롬프트는 평생 1회 계산), RAG
   인덱스 DRAM 상주. 예산 초과 시에만 LRU→NVMe 투명 스필(관리 UI 없음, 사용량
   미터만). NVMe에 닿는 KV/이력은 박스 키로 저장 시 암호화(디스크 탈거 위협
   대응).
7. **멀티컨텍스트**: 배치 디코드(PE_M 행 + kc_seq 시퀀스별 KV 라우팅, RTL 검증
   완료)로 동시 에이전트 N개의 합산 처리량이 N과 함께 상승 (N=4 시 합산
   ~68 tok/s [EST], naive 1/N 대비 +58%) — "개인 에이전트 팜". KV 총예산
   ~45GB(≈50만 토큰 상주), 초과분은 NVMe 페이징.
8. **보안 완결성**: 폰홈/텔레메트리 없음, 서명된 오프라인 업데이트 패키지,
   단일 장치 감사 범위.

**스코프**: 전부 호스트/SoC-소프트웨어 트랙(RTL 무관) — 임베디드 웹서버+UI,
임베더 상주, RAG 스토어, 그래프 뷰가 신규 작업 항목.

## 1a. Power-on behavior (plug → ready → tokens)

The user experience is close to *"plug in, use it,"* with **a short boot** — not instant-on, and
with **one one-time setup**:

- **One-time provisioning (factory/first setup):** the ~467 GB Q4_K model is written to the internal
  **NVMe SSD** (`flash_layout.py` — committed tool name; the offline expert→channel placement pass). Done once; survives power cycles.
- **Every power-on — boot time is per-SKU [EST]:** power → clocks/PLL lock → resets → DDR5 PHY
  training + NVMe init → **`boot_loader` copies the resident set NVMe→DRAM** → its `done` **releases
  inference** → USB device enumerates on the host. **Inference is gated by `boot_loader.done`, not
  by power-on.** How much is copied — and therefore how long the boot is — depends on the SKU:
  - **Streaming SKU:** only the **~17 GB hot partition** (attention/dense/shared/embed/LM-head —
    canonical: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2) is loaded; the 256 routed experts
    stay on NVMe and demand-stream per token → **a few s** (~1–2 s multi-NVMe, a few s single drive).
  - **Primary full-residency SKU (R3):** the **whole ~467 GB** checkpoint is loaded NVMe→LPDDR5X (512 GB
    resident, volatile) → **~70 s cold boot on EVERY power-on**. This is the flagship box; plan the UX
    around the ~70 s figure, not the streaming SKU's few seconds.

  (Full condition list + RTL detail: [`OPERATION_FLOW.md`](OPERATION_FLOW.md) §1.)
- **Per request:** the host sends token IDs + position over USB-C; the box streams the demand
  experts from the **NVMe SSD**, runs the model, returns next tokens. **Session KV lives in the box's DDR5** —
  the host only exchanges tokens.

**What crosses USB-C:** only `start` / `prompt_tok` / `start_pos` / `s_len` in and
`next_tok` / `tok_valid` / `busy` / `done` out — token IDs + control. The heavy weight/KV traffic
never leaves the box.

**Two power inputs, one data cable.** Because the box is self-powered (§7), the physical picture is:
a **DC/PD power input** + a **USB-C data cable** to the host. It is not a bus-powered stick; think
"powered external box that appears to the host as a local AI endpoint after a boot" — a **~1–2 s boot
on the streaming SKU**, but **~70 s on every power-on for the primary full-residency SKU (R3)** that
reloads the whole ~467 GB into volatile LPDDR5X (see §1a per-SKU note above)."

**Device-readiness signaling (to build):** the host driver should surface the boot state (booting →
loading resident set → ready) so the app can show "warming up" with an accurate ETA instead of failing
calls before `done`. On the residency SKU the ETA is a genuine **~70 s** (percent-loaded of ~467 GB),
so this is real warming-up UX, not a spinner flash. (Phase D2.)

---

## 2. Current state (honest baseline)

**Done (the core IP — see [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) "Keep"):**
- Q4_K datapath (MLA + DSA + 256-expert MoE + MTP), assembled from Q4_K units whose **GEMM core
  (`glm_matmul_q4k`) is bit-exact to the ggml-Q4_K reference `tools/q4k_ref.py`** at a
  small-but-faithful slice (`glm_matmul_q4k` **160/160**; `q4k_prim` **18/18**;
  `swiglu_expert_q4k` **240/240** functional; `moe_router_q4k` **40/40** invariants — `make q4k`).
  **Honest scope:** bit-exact is vs `tools/q4k_ref.py`, which is itself sealed against the **real GGUF bytes at the dequant layer** (`docs/GGUF_CROSSCHECK.md`); llama.cpp *whole-runtime* equality stays out-of-contract;
  the **assembled** `glm_model_q4k` now has an **end-to-end golden** (`make model-q4k` **1155** +
  `model-q4k-acthw` **1155**), plus **spec==greedy** self-consistency (`spec_decode_top` **18/18**,
  DUT-vs-DUT).
- The memory system (ddr5_xbar, flash_xbar, kv_cache_pager, expert_cache_pf, weight_loader_q4k,
  boot_loader), BMC-proven (these blocks are **byte-agnostic** — they move addresses/slots/IDs, not
  weight bytes — so they carried over from the prior FP8 track unchanged in logic); the batching
  stack (PE_M decode-batching on the Q4_K wrappers, union-skip MoE now folded **inline** into
  `glm_decoder_block_q4k`); the **spec ÷K weight-load hardware** (`spec_batched_top` /
  `spec_chain_top`, spec==greedy at K>1 via `make spec-slow`).
  (`flash_xbar` is the **medium-agnostic storage-read fabric** — address→weight-bytes with
  latency hiding — that in the product **fronts an NVMe/PCIe host controller**; the RTL name is
  a committed identifier, and only the NAND-specific backend is swapped — the abstraction, the
  compute die, `weight_loader_q4k`, `expert_cache_pf` and `kv_cache_pager` are unchanged.)
- Production top `glm_q4k_system_cdc` with **host/USB + memory + compute CDC** — structural sign-off
  (`make synth-glm`, yosys `hierarchy -check` + `check -assert` exit 0, no unresolved
  hierarchy/comb-loop/multi-driver/inferred-latch; **elaboration, not a sim**), with the CDC/reset
  architecture (`cdc_async_fifo` gray-pointer, `reset_sync`) formally sound.
- **Format-agnostic power levers (measured on the RTL/trace harnesses, current):** clock-gating
  (`clk_en_ctrl` — 73.75 % of idle dynamic power gated, formally safe), DVFS/eco frequency prescaler
  (`clk_throttle` — run the die f/div in the ~4–5× slack, byte-identical, BMC-proven), spec ÷K
  (K=2 ≈ +23 %), flash striping (`flash_layout.py` — expert→channel placement, ~+40 %). Predictor
  prefetch is a **measured NO-OP** (popular experts already resident — kept honest).
- **Compute-side PPA levers are prior-FP8-track only (tag `fp8-verified-baseline`; Q4_K re-run PENDING):** the
  BFP fixed-point accumulator (−87.6 % cells), die-shrink, and `weight_decomp` compression (1.34×)
  were measured on the FP8 datapath and are **not** re-measured for Q4_K — presented here as prior-FP8
  numbers, not Q4_K claims (on an NVMe-bound die they cut area/power/timing but do not move tok/s).
- Energy + BOM **models** ([`LOW_POWER.md`](LOW_POWER.md), [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md)).
- **Host software scaffold** ([`host/`](../host/README.md)) — a local **OpenAI-compatible server**
  (`/v1/chat/completions`, streaming SSE), the exact RTL host protocol (`wpu_device.py`, mirrors
  `glm_q4k_system_cdc` + the boot-loader-done readiness gate), the **real GLM-5.2 BPE tokenizer** +
  a port of GLM's chat template, and host-side sampling — buildable/testable with **zero hardware**
  against a mock backend (the Phase D2 first deliverable). *(The simulator backend targets the
  on-`main` `glm_model_q4k` slice via `vvp` and returns real RTL argmax tokens — parse/protocol
  path covered by `make host-test` (32 tests), the full `vvp` run validated separately.)*

**Not done (the gaps to a product):**
- **Real-model full-scale fidelity** (PRODUCT_ROADMAP P1 — *the* gate: the real ~467 GB Q4_K weights
  must produce the real model's tokens end-to-end, checked against a numeric golden). The assembled
  `glm_model_q4k` end-to-end golden is now **DONE** (`make model-q4k` 1155 + `model-q4k-acthw` 1155);
  what remains **OPEN** is bit-exactness vs the **real downloaded GGUF /
  llama.cpp** (a different arithmetic contract — Q8-quantized activations + integer dot vs our
  bf16-activation / fp32-accumulate path). **Blocking for the product claim.**
- **Mixed-type (Q6_K / Q8_0 / F16) path** — **DONE** (`make mixedtype`): the RTL now has
  Q6_K/Q8_0/F16 consumers, so the published UD-Q4_K_XL dynamic mix (sensitive tensors at higher
  precision) can be consumed as-is. *(The remaining fidelity gap is the real-GGUF / llama.cpp
  bit-exactness above, not type coverage.)*
- **FPGA fit / bitstream** — fit **MEASURED/DONE** (Vivado routed fit + closed Fmax campaign on
  XCKU3P, 46.5 MHz — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); **no real board yet** (board
  bring-up is D1).
- **Host software — the real backend + driver** — the API/protocol/tokenizer layer is scaffolded
  (above); what remains is a signed cross-OS **USB-C driver** and a **real backend** (simulator- then
  hardware-backed) + production runtime/scheduler behind the swappable mock.
- **Physical device** — no PCB, no power architecture, no thermal design, no enclosure, no
  certification.

---

## 3. Target product spec (the requirements to hit)

| Spec | Target | Notes |
|---|---|---|
| Model | GLM-5.2 `UD-Q4_K_XL` (Q4_K GGUF, ~467 GB) | the differentiator is **full 753B frontier locally**, not a smaller local model; Q4_K GEMM core bit-exact to `tools/q4k_ref.py` (our ggml reimpl), **not** the real GGUF / llama.cpp |
| Throughput | **rung-②** funded board ≥ 25 tok/s (goal 30–40) single-user; **rung-①** prove-it FPGA ~5–8 [EST] | comfortable interactive on the product rung; staged per [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) |
| Power (peak) | **≥50–78 W [EST] v3-volume** (≥64–99 W [EST] v3-proto) — **floors, not budgets** (SoC term UNVERIFIED), self-powered; adapter ~100–140 W USB-PD/DC | floors per §7 / R3 §4 (old ~40–60 / ~50–80 W retired — never derived; the ≤110 W target was the pre-pivot streaming SKU) |
| Idle power | ≤ 10 W target [EST] | clock-gating + DVFS *(inherited from the smaller streaming box; the 512 GB residency box's idle is not yet separately analyzed — see §7)* |
| Size | ≤ ~1 L enclosure | external-SSD → small-mini-PC |
| Noise | quiet (≤ ~35 dBA) | desktop appliance |
| Interface | USB-C (USB 3.2 Gen 2 min) | + separate DC power in |
| Host OS | macOS + Windows + Linux | signed driver + local server |
| Retail price | ~$2.5–8 k [EST] | FPGA-class dependent (see §8) |
| Model update | field-updatable weights (rewrite the NVMe model store) | new GLM point releases |

---

## 4. The device-specific gap (beyond the RTL track)

The RTL track (PRODUCT_ROADMAP P1–P4) makes the accelerator correct & synthesizable. The **device**
adds these workstreams, which that roadmap only touches lightly:

1. **Power architecture** — even the v3-volume ≥50–78 W [EST] floor (≥64–99 W v3-proto; R3 §4) is above the 15 W
   USB-C default, so it can't be bus-powered; needs a self-powered design (own DC/PD input, ~100–140 W
   adapter). (§7) *(The ~80–110 W in older drafts was the pre-pivot streaming SKU.)*
2. **Thermal / acoustics** — dissipate the ≥50–78 W [EST] v3 floor quietly in ~1 L. (§7)
3. **Host software & UX** — driver + local OpenAI-compatible server + onboarding; the difference
   between "a board" and "a product people use." (Phase D2)
4. **Industrial design & enclosure** — the physical box, connectors, LEDs, mounting.
5. **Manufacturing & certification** — DFM, USB-IF, FCC/CE/EMC, safety, yield/binning.
6. **Support & lifecycle** — warranty, RMA, firmware/model updates, docs.

---

## 5. Phased plan (device track — "D" phases, gated)

Each phase has a **GATE**: a go/no-go you must pass before spending on the next.

### Phase D0 — De-risk & measure *(do FIRST; cheap, decisive)*
- **D0.1 Real-model fidelity** — execute PRODUCT_ROADMAP P1.1 (real ~467 GB Q4_K checkpoint → real
  tokens, against a numeric golden; and close the Q4_K-only / mixed-type gap so the published
  UD-Q4_K_XL mix can be consumed as-is). **Hard prerequisite** — no product until this passes.
  *(Partly DONE: the assembled end-to-end golden (`make model-q4k` 1155) and the mixed-type
  consumers (`make mixedtype`) are in; the real-checkpoint / llama.cpp bit-exactness remains open.)*
- **D0.2 FPGA fit** — **(DONE — Vivado ML 2026.1 routed fit of `glm_q4k_system_cdc` on XCKU3P,
  compact config + ACT_HW=1: 142,320 LUT / 87.5 % synth-stage (routed 141,298 LUT), 421 DSP, 0 BRAM, routed Fmax 46.5 MHz, campaign
  closed bit-exact on the 1155-test golden; the old Gowin/nextpnr scaffold is removed. FPGA class =
  KU3P-class.)* ~~take the design through a vendor flow to get real utilization → pick the FPGA
  class~~ — was the project's #1 unknown; now measured.
- **D0.3 Power point** — decide the operating point (§7): the v3-volume ≥50–78 W [EST] self-powered "fast"
  point vs the eco/throttle "quiet/small" point (down to ~30 W). Drives thermal + PSU + enclosure.
- **D0.4 Bring-up feasibility** — confirm the target FPGA dev-kit has the DDR5 + NVMe (M.2/PCIe) +
  USB-C IO the design needs. *(Update: the measured fit landed on **XCKU3P** via Vivado, so the
  bring-up board is now **KU3P-class**; the Gowin/nextpnr path — and with it the Tang Mega 138K Pro
  (GW5AT-138) dev-board plan below — is superseded.)* *Dev-board vs product split:* a dev board is a
  **bring-up / reduced-demo board only** — nowhere near the model store — so the **product needs a
  custom board** (KU3P-class + big DDR5 + NVMe via M.2/PCIe). The dev board proves the pipeline at
  reduced scale; the NVMe/DDR5 model store lives on the custom board (Phase D3).
- **GATE D0:** real tokens match the real model **and** the design fits a costable FPGA at a viable
  power point. *(The fit half is **DONE** — it fits XCKU3P, a low-end UltraScale+, so no >$8 k
  data-center card is forced; the real-GGUF fidelity half remains open.)*

### Phase D1 — FPGA prototype bring-up
- D1.1 Port the full-scale RTL to the chosen FPGA; DDR5 + NVMe (M.2/PCIe) + USB-C PHY integration
  (vendor IP, PRODUCT_ROADMAP P3.1).
- D1.2 Boot-load the real Q4_K model to the NVMe model store (productionize `flash_layout.py`, the
  offline GGUF→NVMe expert/channel packing); first end-to-end token over USB-C on real hardware.
- D1.3 Measure the **real** tok/s, watts, thermals — replace every [EST] with a number.
- **GATE D1:** the dev-kit emits the real model's tokens over USB-C at ≥ target tok/s within the
  power point. *This is the "it actually works as a device" proof.*

### Phase D2 — Host software stack *(parallel with D1 once tokens flow)*
- D2.1 USB-C device driver (macOS/Windows/Linux), signed; robust enumeration / hot-plug / recovery.
- D2.2 **Local OpenAI-compatible server** — so existing clients (chat UIs, VS Code, Claude Code,
  etc.) point at `localhost` and just work. Tokenizer + sampling params + streaming. *(API surface +
  protocol + real GLM BPE tokenizer + chat template + streaming already scaffolded in
  [`host/`](../host/README.md) against a mock backend; remaining is the real backend + driver.)*
- D2.3 Management app: model load/update, health, thermal/power telemetry, firmware update.
- **GATE D2:** a user installs the app, points their editor at the box, and chats — no CLI.

### Phase D3 — Custom board, power, thermal, enclosure
- D3.1 Multi-layer controlled-impedance PCB (FPGA + DDR5 + NVMe/M.2 + USB-C + power); PRODUCT_ROADMAP P4.1.
- D3.2 Power subsystem: PSU / USB-PD, power domains, DVFS hooks (PRODUCT_ROADMAP P2.4), protection.
- D3.3 Thermal: heatsink + quiet fan curve for the power point; acoustic target.
- D3.4 Industrial design: enclosure, connectors, status LEDs, EMI shielding.
- **GATE D3:** a custom unit hits target tok/s/W in the final thermal/acoustic envelope.

### Phase D4 — DFM, certification, pilot build
- D4.1 Design-for-manufacture, test fixtures, board bring-up automation, yield/binning.
- D4.2 Certification: USB-IF, FCC/CE, EMC, electrical safety.
- D4.3 Reliability qual: temp/voltage/aging, NVMe endurance (read-mostly → benign; writes only on model update), burn-in.
- D4.4 Small pilot production run.
- **GATE D4:** certified units pass reliability + a pilot cohort uses them without support fires.

### Phase D5 — Launch & lifecycle
- D5.1 Packaging, docs, onboarding, warranty/RMA.
- D5.2 Model-update pipeline (new GLM point releases → NVMe model-store rewrite tool).
- D5.3 Support + a channel for architecture-change models (new arch ⇒ RTL update ⇒ bitstream push).

---

## 6. Key decisions / forks (resolve early)

| Decision | Options | Recommendation |
|---|---|---|
| **Power point** | ≥50–78 W [EST] v3-volume "fast" floor (self-powered) · down to ~30 W eco/throttle (slower) | ship **self-powered at the v3-volume ≥50–78 W [EST] floor** (interactive tok/s); offer a quiet/eco mode — **the RTL knob exists** (`clk_throttle` runs the die f/div, byte-identical, [`LOW_POWER.md`](LOW_POWER.md) §4). *(Floor per R3 §4 / §7 — the old ~40–60 W is retired, never derived; the old ~90–100 W was the pre-pivot streaming SKU.)* |
| **Power delivery** | own PSU/barrel · USB-PD (~100–140 W per R3 §4) · bundled PD brick | **own DC input** (a ~100–140 W adapter covers v3-volume/proto with headroom; EPR host+cable support is still rare); USB-C = data |
| **FPGA class** | mid FPGA (~$0.5–2 k) · data-center card (~$3–8 k) | **decided — D0.2 measured**: fits **XCKU3P** (KU3P-class, low-end UltraScale+) at 87.5 % LUT |
| **Model updates** | NVMe model-store rewrite tool · sealed | **field-updatable** (GLM ships point releases) |
| **Openness** | closed appliance · open driver/API | **open local API** (drives adoption via existing clients) |
| **NVMe drive grade** | TLC (balanced) · SLC (low-power, $$) · QLC (cheap, slow) | **TLC** NVMe bulk; SLC only if power ≫ cost ([`LOW_POWER.md`](LOW_POWER.md)) |

---

## 7. Power & thermal (the hard constraint)

**Power envelope — NOT canonical, and NOT a budget (config-labeled; re-derived 2026-07):** these are
**floors with an open term on top**. The DRAM rail is defensible from [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)
§4's own coefficient; the **SoC rail is UNVERIFIED** — the repo contains no pJ/op for the lane, no
gate-density→W model, and no answer to whether the 4–6 pJ/bit already includes the SoC-side PHY (if it
does not, a large term is missing; if it does, part is double-counted). The old ~15–25 W SoC figure was
written when §3 said 3,072 lanes; §3 now says ~12,732 (measured 0.62 B/param, dedicated per-phase engines). **Do not scale it by 1.4× or 1.8× to make a total
appear — neither factor has a basis.** This heading previously read "Canonical" while citing a source
that now disowns the number (R3 §3 미해결); that is fixed here.

| config | draw [EST] | what it is |
|---|---|---|
| **v3-volume** (primary residency SKU) | **≥50–78 W [EST, 재도출]** | LPDDR5X rail **35–53 W** (= 1.1 TB/s × 8 × 4–6 pJ/bit, R3 §4's own coefficient and own method) + SoC **UNVERIFIED**. The old ~25–40 W rail figure is **retired: it reproduces at no design point this project has ever held** (it back-solves to ~780–830 GB/s, which never existed) — R3 §4. **Plan an adapter around the ≥ sign, not the number.** |
| **v3-proto** (1.54 TB/s higher-BW variant) | **≥64–99 W [EST, 재도출]** | DRAM rail **49–74 W** (= 1.54 TB/s × 8 × 4–6 pJ/bit) + SoC **UNVERIFIED**. The old ~35–55 W rail was the retired ~25–40 W scaled ×1.4 — its span ratio (1.571) still carries that base's 40/25 = 1.600 signature, where a true 4–6 pJ/bit sweep must give exactly 6/4 = 1.500. Right scaling, wrong base. Cooling **method** (vapor chamber + fan) carries over; **"same cooling class" and the 저소음 qualifier do not** — ~64–99 W in ~1 L is ~1.6× the heat R3 §7 specced. (§5c — the earlier "(§5a)" cross-ref was wrong.) |
| **eco / throttle** | **down to ~30 W**, or **15 W bus-powered "travel mode"** | `clk_throttle` runs the die f/div (byte-identical, BMC-proven, [`LOW_POWER.md`](LOW_POWER.md) §4) → single-digit tok/s |

> **Note on the older ~80–110 W figure.** Earlier drafts of this doc cited **~80–110 W** at
> interactive. That is the **pre-residency-pivot (v2-era) streaming-SKU** number, when NAND streaming
> dominated the thermal budget (~40–90 W on the NAND read — R3 §4 v2 = ~70–120 W). The residency pivot
> **deletes the NAND-stream term**, so that particular ~80–110 W band is history and is **not** what a
> residency box draws for that reason.
>
> **But read this before concluding the pivot is undone (2026-07 재도출).** The honest v3-proto number
> — **≥64–99 W** — lands *inside* the very band this note disowns. **Same number, different physics,
> and the coincidence is meaningless.** The v2 ~80–110 W was **NAND read energy** (~40–90 W of it), a
> term the pivot genuinely deleted and which has not returned. The new ≥64–99 W is **LPDDR5X rail
> energy at 1.54 TB/s** — a term that grew because the *design point's bandwidth* grew (650 GB/s →
> 1.1 → 1.54), computed with R3 §4's own 4–6 pJ/bit by R3 §4's own method. The pivot's thermal win is
> real and intact; the box is nonetheless hotter than published, because **the published number was
> never derived** (§4's ~25–40 W rail reproduces at no rate this project has held). Two independent
> facts, not one walked back.

All configs are **above the 15 W USB-C default** (except the deliberate travel-mode throttle), so pick
an adapter for the v3 draw:

| USB-PD tier | budget | verdict for the v3 residency box |
|---|---|---|
| USB-C default | 15 W | ✗ full speed — but = the **15 W bus-powered "travel/eco" throttle** (single-digit tok/s) |
| PD SPR | 60 / 100 W | **100 W: v3-volume only, and only if the re-derived 50–78 W [EST] band holds** — the SoC term in it is UNVERIFIED, so this is a floor, not a budget. **Does NOT cover v3-proto** (≥64–99 W [EST]): at the top of that band a 100 W brick has ~1 W of margin *before* the SoC term is counted. 60 W △ (throttled) |
| PD 3.1 EPR | 140–240 W | **v3-proto requires EPR** on the re-derived numbers — and this row's own warning applies: it needs a compatible host **and** cable (**rare**), so this is a plugability constraint on the product, not spare headroom. NOTE: **do not claim 140 W is *sufficient* either** — that needs the UNVERIFIED SoC term. (R3's DC 19 V / USB-PD 100–140 W recommendation is unchanged; what changed is the claim that 100 W suffices.) |

**Conclusion: a self-powered box (own DC/PD input, ~100–140 W adapter per R3 §4) with USB-C as the data
link** — the eGPU / NAS model. A laptop port cannot both source the box's draw **and** have the host
consume data, which is the other reason power is separated from the USB-C data cable. Thermal: the
≥50–78 W [EST] v3 floor dissipated quietly in ~1 L needs a heatsink + a tuned fan; the eco/throttle mode
(down to ~30 W) enables smaller/quieter builds at lower tok/s. Power breakdown is **memory-dominated**
(LPDDR5X the dominant rail on v3); the compute die is ~20–30 % and mostly gated.

---

## 8. BOM & pricing [EST]

*Canonical per-rung BOM + per-seat economics: [`BOM.md`](BOM.md). Summary below.*

| Component | Rung ① prove-it (budget FPGA) | Rung ② product (custom board) |
|---|---|---|
| FPGA (largest uncertainty) | ~$0.5–1.5 k (low/mid, e.g. KU3P / GW5AT-class) | ~$3–8 k (Versal / Agilex / HBM-class) |
| Fast DDR tier | ~$150–300 (**DDR4** ~4 ch, near-term) | ~$300 (**DDR5** multi-ch / HBM, 64–128 GB) |
| NVMe SSD ~1–2 TB (M.2 / PCIe Gen3–4 x4) | ~$100–250 (1 drive) | ~$300–500 (2–4 TB or multi-drive) |
| PCB / PSU / controllers / enclosure | ~$150–300 | ~$300 |
| **BOM total** | **~$1.0–2.4 k** | **~$4–9 k** |
| **Indicative retail (≈2× BOM)** | **~$2.5–5 k** | **~$8–16 k** |

**Rung mapping ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)):** the two paths are the ladder's first two
rungs — **rung ①** the ~$1–2 k prove-it demo box (budget FPGA + DDR4, the near-term build) and **rung ②**
the funded product (custom board, DDR5/HBM) — the ladder's ~$3–6 k box, up to ~$4–9 k for the
HBM / data-center-card variant. **Rung ③ — a SoC/ASIC at manufacturing volume** — is the cost endgame:
many-channel PHY + near-memory Q4_K (low-precision) compute at ~TB/s, with **lower $/seat and lower power**
once the multi-million NRE amortizes over volume. Not now (no volume, no capital); sequenced *after* the
FPGA rungs prove product-market fit — the same verified RTL on every rung. *(Updated 2026-07: the
rung-③ **primary** design point is now **full residency** — 512 GB LPDDR5X (16×32 GB, 1024-bit
on-package substrate, ~1.1 TB/s) holding the whole ~467 GB checkpoint, cold store = one commodity
M.2 NVMe (boot-load ~70 s), box ≥50–78 W [EST] floor (R3 §4 — old ~40–60 W retired), board 120×80 mm, BOM ~$1.8–2.4 k, design point **≈80 tok/s [measured-inputs EST]**;
the ONFI streaming tier is deleted from the primary SKU (pads stay on-die for the hybrid upside
SKU); HBM stays the long-range ceiling — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)*

**Cost insight:** power is memory-dominated but **cost is FPGA-dominated** — the FPGA class (set by
D0.2) is the pivotal BOM lever. Algorithmic levers (spec ÷K, compression, DVFS) improve
power/throughput **without** touching BOM.

**vs the alternative:** an 8×H200 cloud/cluster is ~$250–300 k capital + ~6–10 kW; this box is
~50–100× cheaper capital and ~50–70× lower power — the trade is single-user throughput. And for an
air-gap / offline buyer the cloud isn't on the menu at any price: even a "secured cloud" (VPC /
zero-retention / TEE) needs connectivity and fails the unplugged test.

---

## 9. Risks & mitigations (ranked)

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **FPGA fit** — **CLOSED**: D0.2 measured — fits **XCKU3P** (142,320 LUT / 87.5 % synth-stage, routed 141,298 LUT, 421 DSP, routed 46.5 MHz), no data-center card needed | closed | headroom is tight (87.5 % LUT) — die-shrink levers ([`MINIATURIZATION.md`](MINIATURIZATION.md)) + compact config stay relevant |
| 2 | **Real-model fidelity not fully closed** — assembled-model golden now **DONE** (`make model-q4k` 1155) and the mixed-type consumers landed (`make mixedtype`); still **not bit-verified vs the real GGUF / llama.cpp** | high | D0.1 (PRODUCT_ROADMAP P1.1) — the remaining piece is the real-GGUF / llama.cpp bit-exactness |
| 3 | **Power > USB-PD** | med | self-powered design (§7); accept it's not a bus stick |
| 4 | **Host software effort** — driver + server + cross-OS is real work | med | OpenAI-compatible API to reuse the existing client ecosystem |
| 5 | **Thermal/acoustics** in a small box | med | power-point choice; eco mode; proven eGPU-class cooling |
| 6 | **Model architecture churn** — a new model arch needs RTL changes | med | field weight-updates for point releases; RTL update path for arch changes |
| 7 | **Certification / manufacturing** cost & time | med | budget D4; pilot before scale |
| 8 | **Real watts/tok-s differ from [EST]** | low-med | D1.3 replaces every estimate with a measurement |

---

## 10. Rough timeline [EST] (small team; parallelizable)

| Phase | Scope | Rough duration |
|---|---|---|
| D0 | fidelity + FPGA fit + power point | 1–3 months |
| D1 | FPGA prototype, first real tokens over USB-C | 3–6 months |
| D2 | host software (parallel with D1/D3) | 3–6 months |
| D3 | custom board + power + thermal + enclosure | 6–9 months |
| D4 | DFM + certification + pilot | 6–12 months |
| D5 | launch + lifecycle | ongoing |

**Verified RTL → shipping consumer device: ~18–36 months**, dominated by board/cert/mfg, not the
RTL. D0 is cheap and decisive — **do it before committing to the rest.**

---

## 11. Immediate next steps (this quarter)

1. **D0.1** — run the real-checkpoint full-model fidelity check on a GPU host (PRODUCT_ROADMAP P1.1).
2. **D0.2** — **(DONE — Vivado routed fit + closed Fmax campaign on XCKU3P; see §5 D0.2.)**
3. **D0.3** — pick the power point (v3-volume ≥50–78 W [EST] floor, self-powered, recommended; §7) and draft the thermal envelope.
4. Wire the **host-side local OpenAI-compatible server** ([`host/`](../host/README.md), already
   scaffolded against a mock backend) to a simulator-backed backend now (no hardware needed) so the
   software is ready when D1 tokens flow.

---

## Status
- **This plan: drafted** (device track, complements the RTL track in PRODUCT_ROADMAP.md).
- **Gating unknowns:** real-GGUF / llama.cpp fidelity (D0.1, open). **FPGA fit (D0.2) is CLOSED —
  MEASURED** (Vivado routed fit on XCKU3P, 46.5 MHz, campaign closed).
- **Everything downstream** (size, thermal, BOM, price, timeline) is bounded by the FPGA class D0.2
  returned: **KU3P-class**.
