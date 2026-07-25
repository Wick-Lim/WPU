# Paper — arXiv preprint (draft)

`wpu.tex` is a self-contained, dependency-light LaTeX preprint:

> **Bit-Exact by Construction: A Verification-First RTL Accelerator that
> Inherits the GGUF k-Quant Checkpoint Ecosystem**

The headline claim is the bit-exactness *contract* (RTL dequant ≡ independent
reference ≡ llama.cpp's own `ggml` kernels on 376,586,240 real GGUF weights),
with three supporting contributions: speculative decoding proven correct in RTL
(the `spec==greedy` invariant + hardware-counter-measured A_eff), the
adversarial non-vacuity verification discipline (injection pairing, exact-count
manifests, the output-insensitivity finding and direct per-beat bindings,
PHY-closure loopback), and the measured-inputs bandwidth roofline.

**Honest scope:** deliberately *pre-silicon*. It reports design, bit-exact and
formal verification, an FPGA place-and-route (hold met, routed Fmax 46.5 MHz),
and cycle-accurate stall measurement; every token rate is a roofline projection
tagged `[EST]`. No fabricated chip, no running board, no end-to-end 467 GB run.
The verification ledger (Table 1) is reproducible from this repository.

**Fact-check status:** the draft was adversarially fact-checked against the
repository (5 independent passes: verification numbers, architecture/measured
numbers, overclaim audit, citation web-verification, LaTeX sanity — 33 findings,
all resolved). Two claims were made TRUE rather than reworded: the phantom-KV
injection (`spec-greedy` step 3, `-DINJECT_PHANTOM_KV`) and the router-code
injection (`loopback-rest` step 3, `-DLBRESTINJECT_RW`) are now committed,
asserted builds. All arXiv IDs in the bibliography were resolved against
arxiv.org and corrected where the scouting metadata was wrong (notably
`cascade` → Saxena et al., *Utility-driven speculative decoding for
mixture-of-experts*).

## Build

No exotic packages (only `geometry`, `amsmath`, `booktabs`, `graphicx`,
`xcolor`, `enumitem`, `url`, `hyperref`). Any of:

```sh
pdflatex wpu.tex && pdflatex wpu.tex     # twice, for refs + hyperlinks
# or
latexmk -pdf wpu.tex
```

Or upload `wpu.tex` directly to Overleaf or arXiv (single-file submission,
`cs.AR` primary).

## Before posting to arXiv — checklist

- Compile twice and eyeball the PDF (no compiler exists on this dev machine;
  the file passed structural lint only).
- Re-read *Limitations and Honest Scope* against the current repository state
  so no claim drifts ahead of the evidence.
- Skim the `et al.` bibliography entries and expand author lists where arXiv
  metadata provides them.
- Author/affiliation on the title page: `Wick-Lim, Independent Researcher`.

## Status

Draft v2 — fully synchronized with the repository's verification ledger
(`README.md`), the speculative-composition and KV-write-back docs
(`docs/SPEC_COMPOSITION_DESIGN.md`, `docs/KV_WRITEBACK_DESIGN.md`), the GGUF
cross-check (`docs/GGUF_CROSSCHECK.md`), the measured design-space docs
(`docs/R3_APPLIANCE_SPEC.md`, `docs/H_MEASUREMENT.md`,
`docs/CYCLE_EMULATION.md`), and the board study.
