# 홍보 킷 — 복붙-레디 초안 3종 (게시는 본인 계정으로)

공통 원칙: proven/measured만 헤드라인, tok/s는 반드시 [EST]/estimated, "no silicon,
no board" 선제 명시. 이 커뮤니티들에선 정직한 스코핑이 곧 신뢰 = 최고의 마케팅.

---

## 1. r/LocalLLaMA (최우선)

**Title:**
```
I wrote a Verilog accelerator that is bit-exact to llama.cpp's GGUF dequant — verified on 376M real checkpoint weights (targeting a 753B MoE, fully offline)
```

**Body:**
```
Over the past months I built an open-source RTL (Verilog) accelerator around one
idea: instead of inventing a new quantization, make silicon that is BITWISE
faithful to the GGUF k-quant format you already use — so the hardware inherits
the whole GGUF checkpoint ecosystem by construction.

What's actually proven (not marketing):
- The RTL dequant of Q4_K/Q6_K/Q8_0 is bit-exact to an independent reference,
  and that reference is bitwise-equal to llama.cpp's own dequantize_row_* on
  376,586,240 weights extracted from two real published GGUF files (compared as
  raw uint32, no tolerance).
- The full model datapath (MLA latent attention + DSA sparse attention +
  fine-grained MoE, targeting GLM's 753B UD-Q4_K_XL) is assembled and bit-exact
  to a numpy golden at a faithful slice, and elaborates clean at the true 753B
  shape.
- Speculative decoding is composed into the memory system and PROVEN in RTL:
  the committed stream is a bit-exact prefix of greedy decoding, and the
  weight-load amortization is measured from a hardware counter.
- Memory controllers are formally verified (BMC + k-induction). Every
  load-bearing test gate is paired with a fault-injection build that must fail
  (a test that can't fail proves nothing).
- The compute top places & routes on a Kintex UltraScale+ (hold met, 46.5 MHz).

What's honestly NOT done: no silicon, no running board, the 467GB checkpoint
has not been run end-to-end, and every tok/s figure is a roofline estimate
(design point ≈80 tok/s [EST] for a 512GB LPDDR5X residency box at 1.1 TB/s).
The README's verification ledger tags every claim as proven / measured /
elaborated / estimated.

Repo (Apache-2.0, every claim reproducible via make gates):
https://github.com/Wick-Lim/WPU
Landing: https://wick-lim.github.io/WPU/
Preprint draft: paper/wpu.pdf in the repo.

Also: I'm looking for an arXiv cs.AR endorsement (first-time author) — link in
the README if you're qualified and find the work credible.

Happy to answer anything — especially skeptical questions; the whole project is
built around being checkable.
```

---

## 2. Hacker News (Show HN)

**Title (80자 내):**
```
Show HN: A Verilog LLM accelerator bit-exact to llama.cpp's GGUF dequant
```

**URL:** `https://github.com/Wick-Lim/WPU`

**First comment (본인이 바로 다는 설명 코멘트):**
```
Author here. The core idea: accelerators usually get validated against the
author's own reimplementation of the reference numerics — a self-referential
gap. This project makes the contract explicit instead: the RTL's k-quant
dequantization is proven bitwise-equal (raw uint32, no tolerance) to
llama.cpp's own kernels on 376,586,240 weights from two real published GGUF
files, so the silicon inherits the GGUF checkpoint ecosystem by construction.

Two things I think HN might find interesting beyond the headline:

1. Speculative decoding proven in RTL: the committed token stream is proven to
   be a bit-exact prefix of greedy decoding (any accept schedule), and the
   bandwidth-amortization factor is measured from a hardware weight-load
   counter — an all-accept schedule commits exactly K+1 tokens per weight load.

2. A verification discipline where every load-bearing gate is paired with a
   fault-injection build that MUST fail, and CI pins the exact per-gate test
   counts (a testbench silently running fewer tests is caught). Along the way
   we found that output-insensitive inputs (MoE router codes absorbed by top-k
   selection) defeat end-to-end token-equality checks entirely — you need
   direct per-beat bindings on the die's inputs to prove those bytes are even
   consumed.

Honest scope, stated up front: pre-silicon. No board, no chip, no end-to-end
467GB run; all throughput numbers are roofline estimates tagged [EST]. The
README's ledger separates proven / measured / elaborated / estimated for every
claim, and each is reproducible from the repo's make gates.
```

---

## 3. X(트위터) 스레드 (6트윗)

```
1/ I built a Verilog accelerator that is BIT-EXACT to llama.cpp's GGUF dequant
— verified on 376,586,240 real checkpoint weights, raw uint32 equality, no
tolerance. Target: the 753B GLM MoE, fully offline. 🧵

2/ The idea: don't invent a new quantization. Make silicon bitwise-faithful to
the k-quant format the local-LLM world already uses — and the hardware inherits
the entire GGUF checkpoint zoo by construction.

3/ Speculative decoding, proven in RTL: the committed stream is a bit-exact
prefix of greedy decoding (spec==greedy invariant), and the weight-load
amortization is measured from a hardware counter — all-accept hits exactly K+1
tokens per load.

4/ Favorite verification find: MoE router weight codes are OUTPUT-INSENSITIVE —
corrupt them and top-k selection absorbs it, so end-to-end token checks pass on
garbage. You need per-beat bindings on the die's inputs. Every gate here is
paired with a fault injection that must fail.

5/ Honest scope: pre-silicon. FPGA place&route is measured (Kintex US+, hold
met); tok/s is a roofline ESTIMATE (≈80 [EST] on a 512GB LPDDR5X residency
box). The README ledger tags every claim proven/measured/elaborated/estimated.

6/ Apache-2.0, every claim reproducible via make gates:
https://github.com/Wick-Lim/WPU
Preprint draft in paper/. Also seeking an arXiv cs.AR endorsement (first-time
author) — link in the README. 🙏
```

---

## 4. llama.cpp GitHub Discussion (Show and tell)

**Title:** `Hardware bit-exact to ggml's k-quant dequant — an open Verilog accelerator`

**Body 요지 (짧게):** ggml의 dequantize_row_* 대비 376M 실가중치 bitwise 검증
사실 + 리포 링크 + "ggml을 하드웨어 계약의 golden으로 썼다, 감사하다" 톤.
프로젝트 홍보보다 기여/감사 프레이밍이 이 커뮤니티에 맞음.

---

## 타이밍 권고

1. 지금: r/LocalLLaMA + llama.cpp Discussion (endorsement에도 직결)
2. 반응 보고 1–2일 내: Show HN (HN은 재게시 어려우니 README가 다듬어진 상태에서)
3. arXiv 게재 확정 후: X 스레드에 arXiv 링크 추가하여 2차 게시
```
