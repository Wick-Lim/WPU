# arXiv 제출 킷 — 네 기기 브라우저에서 ~5분

파일: 이 리포의 `paper/wpu.tex` 하나만 업로드하면 됨 (단일 파일 제출, 그림 없음).
로컬에 리포가 있으면 `git pull` 후 그 파일 사용; 없으면 GitHub 웹에서 raw 다운로드.

---

## 0. 제출 전 필수 결정 — 리포 공개

**논문이 "reproducible from the public repository at github.com/Wick-Lim/WPU"라고
주장하는데 리포는 현재 PRIVATE.** 제출 전에 둘 중 하나:
- (권장) GitHub → Settings → General → Danger Zone → **Change visibility → Public**
- 또는 논문에서 public/reproducible 문구를 수정 (비권장 — 재현성이 핵심 강점)

## 1. 로그인 & 시작

1. arxiv.org 로그인 (계정: wicklim90@gmail.com)
2. 우상단 **Submit** → **Start New Submission**
3. cs.AR 첫 제출이면 **endorsement** 요구가 뜰 수 있음 — 뜨면 안내 절차대로
   (기존 arXiv 저자에게 endorsement 코드 요청). 안 뜨면 그대로 진행.

## 2. License

- 권장: **arXiv.org perpetual, non-exclusive license** (기본, 가장 보수적)
- 코드처럼 자유 재배포를 원하면 CC BY 4.0 — 저자 선택 사항

## 3. Category

- Primary: **cs.AR** (Hardware Architecture)
- (선택) cross-list: cs.LG

## 4. 파일 업로드

- `wpu.tex` 단일 파일 업로드 → arXiv가 자동 컴파일 (tectonic으로 에러 0 확인됨)
- 컴파일 프리뷰에서 PDF 확인 (참고본: `paper/wpu.pdf`)

## 5. 메타데이터 — 아래 그대로 복붙

**Title:**
```
Bit-Exact by Construction: A Verification-First RTL Accelerator that Inherits the GGUF k-Quant Checkpoint Ecosystem
```

**Authors:**
```
Wick-Lim
```

**Abstract (plain text):**
```
We present a synthesizable RTL accelerator for running a frontier open-weight
mixture-of-experts model -- the 753B-parameter GLM-5.2, in its published ~467 GB
GGUF k-quant form -- as a single-user, fully offline appliance. The work is
deliberately verification-first and pre-silicon: no fabricated chip, no measured
end-to-end token rate. The central claim is a bit-exactness contract: the
dequantization of GGUF k-quants (Q4_K/Q6_K/Q8_0) is proven, via an independently
cross-checked reference, bitwise-equal to llama.cpp's own ggml kernels on
376,586,240 weights from two real published GGUF files, so the silicon inherits
the community's checkpoint ecosystem by construction. This closes the
"self-referential golden" gap endemic to reimplementation-based accelerators.
Around the contract we contribute: (i) speculative decoding proven in RTL -- the
committed stream is a bit-exact prefix of greedy decoding (spec==greedy), enabled
by an intra-batch causal attention mode and a die-internal KV write-back, with
the amortization A_eff measured from a hardware weight-load counter; (ii) an
adversarial verification discipline -- load-bearing gates paired with must-fail
fault injections, exact per-gate test counts pinned by the release gate, and the
finding that output-insensitive inputs (MoE router codes absorbed by top-k;
normalization gains) defeat end-to-end equality checks and require direct
per-beat die-input bindings; and (iii) a bandwidth-bound residency design whose
roofline uses measured expert-union growth and accept rates, with its stall
mechanism validated on real RTL cycles (~73x under residency). The datapath
places and routes on a commodity FPGA. All projected token rates are tagged as
estimates; the remaining physical gates -- vendor PHY IP and tapeout -- are
named, not hidden.
```

**Comments (선택):**
```
Pre-silicon; every proven/measured claim is reproducible from the repository at
https://github.com/Wick-Lim/WPU
```

## 6. 프리뷰 & 확정

- Preview에서 PDF·메타데이터 최종 확인 → **Submit article**
- 제출 후 보통 1–2 영업일 내 공개(announcement) 큐 배정

## 참고

- 초록은 arXiv 1,920자 한도 내(~1,821자)로 맞춰져 있음
- 본문은 2회 적대적 검증(72건 해소) + tectonic 컴파일 통과본
- 문의/수정 필요 시 이 세션에서 계속 지원 가능
