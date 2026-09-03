#!/usr/bin/env python3
"""
kda_onorm_gen.py -- golden vectors for the KDA output norm step
(src/kda_onorm_step.v): Glm5NextTextRMSNormGated on the recurrence output.

Reference (tools/glm53_flash_ref.py rmsnorm_gated, transcribed):
    x    : the recurrence output core_attn_out -- ALREADY bf16-valued when it
           reaches o_norm (recurrent_kimi_delta_attention returns .to(bf16)),
           then upcast to fp32 inside the norm.
    y    = weight[i] * ( x[i] * rsqrt( mean(x^2) + eps ) ) * sigmoid(gate[i])
    out  = y.to(bf16)                      one rounding, at the end
    eps  = rms_norm_eps = 1e-5 ; per head over head_dim = DV = 128 ;
    weight = ssm_norm.weight [128] F32 per KDA block ; gate = g_b(g_a(h)), [H, DV].

HOW THE RTL COMPOSES IT, and why the golden is shaped this way.  The repo's
rmsnorm_unit is bf16-in / fp32-reduce / bf16-out and applies gamma INSIDE its
normalize pass.  Multiplying sigmoid(gate) onto its bf16 OUTPUT would round twice
where the reference rounds once, so the RTL instead folds the gate into gamma:
    gamma_eff[i] = bf16( weight[i] * sigmoid(gate[i]) )
and feeds the proven unit unmodified.  That keeps ONE final rounding (faithful)
at the cost of rounding gamma_eff to bf16 (the reference keeps weight*sigmoid in
fp32) -- a divergence this emitter MEASURES rather than assumes: it reports the
worst error the gamma_eff rounding alone induces on `out`.

ACCURACY: three approximation sources -- Quake fp32_rsqrt, glm_act's bf16
polynomial sigmoid, and the bf16 gamma_eff -- so this is a TOLERANCE leg only.
The bf16 rounding of x itself is NOT a divergence: the reference's x is bf16.

Vector format:
  NTEST H DV
  per test:
    weight : DV      4hex bf16   (F32 in the GGUF; rounded to the unit's bf16 gamma lane
                                  -- the reference weight is bf16-valued in a bf16 model)
    gate   : H*DV    8hex fp32   g_b(g_a(h)) output, pre-sigmoid
    x      : H*DV    4hex bf16   recurrence output, bf16-valued
    out    : H*DV    4hex bf16   golden
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
import q4k_ref  # noqa: E402

F32 = np.float32


def f32b(x):
    return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])


def bf16b(x):
    v = q4k_ref.bf16_round(np.float32(x))
    return (int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16) & 0xFFFF


def bf16v(x):
    """fp32 -> bf16-valued fp32 (RNE)."""
    return q4k_ref.bf16_round(np.float32(x))


def gen(ntest, H, DV, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {DV}"]
    worst_geff = 0.0
    for _ in range(ntest):
        w = np.array([bf16v(v) for v in (1.0 + 0.3 * rng.normal(size=DV))], F32)
        gate = (rng.normal(size=(H, DV)) * 2.5).astype(F32)
        x = np.array([[bf16v(v) for v in row] for row in rng.normal(size=(H, DV)) * 0.8], F32)

        # golden: the reference, fp32 throughout, one bf16 rounding at the end
        y = np.stack([ref.rmsnorm_gated(x[h], w, gate[h], eps=1e-5) for h in range(H)])
        y_bf = np.vectorize(bf16b)(y)

        # the composition's own extra rounding, measured: gamma_eff to bf16
        sig = ref.sigmoid(gate)
        geff_bf = np.vectorize(bf16v, otypes=[F32])(w[None, :] * sig)
        y2 = np.stack([
            (x[h] / np.sqrt(np.mean(x[h].astype(np.float64) ** 2) + 1e-5)).astype(F32) * geff_bf[h]
            for h in range(H)]).astype(F32)
        rel = np.abs(y2 - y) / np.maximum(np.abs(y), 1e-3)
        worst_geff = max(worst_geff, float(rel.max()))

        out.append(" ".join(f"{bf16b(v):04x}" for v in w))
        out.append(" ".join(f"{f32b(v):08x}" for v in gate.reshape(-1)))
        out.append(" ".join(f"{bf16b(v):04x}" for v in x.reshape(-1)))
        out.append(" ".join(f"{b:04x}" for b in y_bf.reshape(-1)))
    if report:
        print(f"worst rel err on out from rounding gamma_eff=weight*sigmoid(gate) to bf16 alone "
              f"(|out|>=1e-3): {worst_geff:.3e}", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    """The gate-folding identity must hold in exact arithmetic: applying sigmoid to
    gamma before the norm equals applying it to the normalized output after --
    that is what lets the RTL reuse rmsnorm_unit unmodified.  And the plausible
    misreading (gating x BEFORE the norm, which changes the variance) must NOT."""
    rng = np.random.default_rng(5)
    H, DV = 3, 16
    n = 0; bad = 0
    for _ in range(50):
        w = (1.0 + 0.3 * rng.normal(size=DV)).astype(F32)
        gate = (rng.normal(size=(H, DV)) * 2.5).astype(F32)
        x = (rng.normal(size=(H, DV)) * 0.8).astype(F32)
        for h in range(H):
            y_ref = ref.rmsnorm_gated(x[h], w, gate[h], eps=1e-5).astype(np.float64)
            inv = 1.0 / np.sqrt(np.mean(x[h].astype(np.float64) ** 2) + 1e-5)
            y_fold = x[h].astype(np.float64) * inv * (w.astype(np.float64) * ref.sigmoid(gate[h]).astype(np.float64))
            n += 1
            if not np.allclose(y_ref, y_fold, rtol=1e-5, atol=1e-6):
                bad += 1
            # misreading: gate x first, then norm -- variance changes, must differ
            xg = x[h].astype(np.float64) * ref.sigmoid(gate[h]).astype(np.float64)
            y_wrong = xg / np.sqrt(np.mean(xg ** 2) + 1e-5) * w.astype(np.float64)
            n += 1
            if np.allclose(y_ref, y_wrong, rtol=1e-3, atol=1e-3):
                bad += 1
    if bad:
        print(f"kda_onorm_gen self-test: {bad}/{n} FAILED"); return 1
    print(f"ALL {n} TESTS PASSED (gate folds into gamma exactly; gating x before the norm never matches, so INJ_ONORM_GATE_FIRST is live)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 24
    H = int(a[1]) if len(a) > 1 else 3
    DV = int(a[2]) if len(a) > 2 else 16
    outp = a[3] if len(a) > 3 else "build/kda_onorm_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w").write(gen(ntest, H, DV))
    print(f"wrote {outp}: {ntest} tests H={H} DV={DV}")
