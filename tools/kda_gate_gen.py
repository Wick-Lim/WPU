#!/usr/bin/env python3
"""
kda_gate_gen.py -- golden vectors for the KDA forget-gate + beta step
(src/kda_gate_step.v), the elementwise stage between the projections and the
recurrence.

Reference (tools/glm53_flash_ref.py forget_gate, transcribed from
Glm5NextTextForgetGate.forward, and Glm5NextTextLinearAttention.forward):
    t[h,d]  = decay[h] * (f[h,d] + dt_bias[h,d])      decay = exp(A_log[h])
    g[h,d]  = lower_bound * sigmoid(t[h,d])            lower_bound = -5.0
    ge[h,d] = exp(g[h,d])                              what kda_recur consumes
    beta[h] = sigmoid(b[h])

DESIGN DECISIONS THE GOLDEN ENCODES
  * decay[h] = exp(A_log[h]) is a function of a STATIC weight (ssm_a [64] F32),
    so it is host-precomputed once per layer and enters as an fp32 input; the
    datapath spends no exp unit on it.  The golden therefore takes decay, not
    A_log, and the two are related exactly as the reference relates them.
  * lower_bound = -5.0 selects the `lower_bound * sigmoid` branch; the softplus
    branch in the reference is DEAD CODE for this checkpoint and is not modelled.

ACCURACY CONTRACT -- the honest part.
  The only sigmoid in this repo (glm_act, MODE 0) is bf16-in / bf16-out with a
  polynomial exp.  The reference computes everything here in fp32.  So the RTL's
  sigmoid argument is ROUNDED TO bf16 before the sigmoid, and the sigmoid itself
  is approximate.  Both ge and beta are therefore TOLERANCE legs, and the
  tolerance is MEASURED, not chosen: this emitter reports the worst relative
  error a bf16-rounded argument alone induces on ge over the corpus, so the TB's
  bound can be traced to a cause rather than tuned until green.  If that error is
  too large for the recurrence, the finding is "the KDA gate path needs an fp32
  sigmoid" -- stated, not hidden.

  What IS exact and checked BITWISE: the -5.0 multiply's sign handling.  fp32
  sigmoid saturates to exactly 0.0 for large negative t, and -5.0 * +0.0 must be
  **-0.0** (sign bit set).  The reference pins that (see forget_gate's self-test);
  the RTL exposes g so the TB can check the sign bit on every saturated element.

Vector format:
  NTEST H DK
  per test:
    decay   : H       8hex fp32   exp(A_log), host-precomputed
    b       : H       8hex fp32   b_proj output (pre-sigmoid)
    f       : H*DK    8hex fp32   f_b_proj(f_a_proj(h)) output
    dt_bias : H*DK    8hex fp32
    g       : H*DK    8hex fp32   golden -5*sigmoid(t)   (sign-of-zero checked bitwise)
    ge      : H*DK    8hex fp32   golden exp(g)          (tolerance)
    beta    : H       8hex fp32   golden sigmoid(b)      (tolerance)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32
LB = F32(-5.0)


def f32b(x):
    return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])


def bf16_round_val(x):
    """fp32 -> bf16 (RNE) -> back to fp32, to model glm_act's input rounding."""
    u = f32b(x)
    lsb = (u >> 16) & 1
    u2 = (u + 0x7FFF + lsb) & 0xFFFF0000
    return np.frombuffer(np.uint32(u2).tobytes(), F32)[0]


def gen(ntest, H, DK, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {DK}"]
    worst_rel_ge = 0.0
    n_sat0 = n_sat5 = 0
    for ti in range(ntest):
        # A_log in the reference is a learned per-head scalar; decay = exp(A_log).
        A_log = rng.normal(size=H).astype(F32) * F32(0.5)
        decay = np.exp(A_log.astype(np.float64)).astype(F32)
        # f + dt_bias spans a wide range so BOTH saturation ends of the sigmoid are
        # exercised: t << 0 -> sigmoid 0 -> g = -0.0 ; t >> 0 -> sigmoid 1 -> g = -5.0
        # Saturation is DELIBERATE, not left to chance: the reference sigmoid
        # (float64 -> F32) is exactly 0.0 only for t < ~-104 and exactly 1.0 only
        # for t > ~17.3, so every 4th test scales f to push t far past both ends.
        fscale = 60.0 if ti % 4 == 0 else (6.0 if ti % 4 == 1 else 1.5)
        f = (rng.normal(size=(H, DK)) * fscale).astype(F32)
        dt = rng.normal(size=(H, DK)).astype(F32)
        b = (rng.normal(size=H) * 2.0).astype(F32)

        g = ref.forget_gate(f, dt, A_log[:, None], -5.0)          # [H, DK], fp32
        ge = np.exp(g.astype(np.float64)).astype(F32)
        beta = ref.sigmoid(b)

        n_sat0 += int((g == 0.0).sum())
        n_sat5 += int((g == -5.0).sum())

        # error attributable to glm_act's bf16 input rounding ALONE (its polynomial
        # adds more): recompute ge with t rounded to bf16 before the sigmoid.
        t = (decay[:, None] * (f + dt)).astype(F32)
        t_bf = np.vectorize(bf16_round_val, otypes=[F32])(t)
        g_bf = (LB * ref.sigmoid(t_bf)).astype(F32)
        ge_bf = np.exp(g_bf.astype(np.float64)).astype(F32)
        rel = np.abs(ge_bf - ge) / np.maximum(np.abs(ge), 1e-30)
        worst_rel_ge = max(worst_rel_ge, float(rel.max()))

        out += [" ".join(f"{f32b(v):08x}" for v in decay),
                " ".join(f"{f32b(v):08x}" for v in b),
                " ".join(f"{f32b(v):08x}" for v in f.reshape(-1)),
                " ".join(f"{f32b(v):08x}" for v in dt.reshape(-1)),
                " ".join(f"{f32b(v):08x}" for v in g.reshape(-1)),
                " ".join(f"{f32b(v):08x}" for v in ge.reshape(-1)),
                " ".join(f"{f32b(v):08x}" for v in beta)]
    if report:
        # A gate over a corpus that never saturates would leave the -0.0 contract
        # untested; insist both ends were hit.
        assert n_sat0 > 0 and n_sat5 > 0, (
            f"gate corpus never saturated (g==-0.0: {n_sat0}, g==-5.0: {n_sat5}); "
            f"widen the f scale or the sign-of-zero leg is vacuous")
        print(f"corpus: g saturated to -0.0 {n_sat0}x and to -5.0 {n_sat5}x (both nonzero -> "
              f"sign-of-zero leg is live); worst rel err on ge from bf16-rounding the sigmoid "
              f"argument alone = {worst_rel_ge:.3e}", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    """The signed-zero contract must hold in the golden itself, and the golden's g
    must lie in [-5, 0] with both endpoints attained."""
    rng = np.random.default_rng(11)
    H, DK = 4, 16
    A_log = rng.normal(size=H).astype(F32)
    f = (rng.normal(size=(H, DK)) * 60.0).astype(F32)     # deliberately saturating
    dt = rng.normal(size=(H, DK)).astype(F32)
    g = ref.forget_gate(f, dt, A_log[:, None], -5.0)
    conds = {
        "g in [-5, 0]":              bool(((g >= -5.0) & (g <= 0.0)).all()),
        "attains -5.0":              bool((g == -5.0).any()),
        "attains 0.0":               bool((g == 0.0).any()),
        "every zero is -0.0":        bool(np.signbit(g[g == 0.0]).all()),
        "exp(g) finite":             bool(np.isfinite(np.exp(g.astype(np.float64))).all()),
    }
    n = len(conds)
    if not all(conds.values()):
        for k, v in conds.items():
            print(f"  {'ok  ' if v else 'FAIL'} {k}")
        print(f"kda_gate_gen self-test: {sum(not v for v in conds.values())}/{n} FAILED"); return 1
    print(f"ALL {n} TESTS PASSED (g in [-5,0], both endpoints attained, every saturated "
          f"zero is -0.0, exp(g) finite -- on a deliberately saturating corpus)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 24
    H = int(a[1]) if len(a) > 1 else 3
    DK = int(a[2]) if len(a) > 2 else 8
    outp = a[3] if len(a) > 3 else "build/kda_gate_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w").write(gen(ntest, H, DK))
    print(f"wrote {outp}: {ntest} tests H={H} DK={DK}")
