#!/usr/bin/env python3
"""
glm53f_swiglu_q8_gen.py -- vectors for test/glm53f_swiglu_q8_tb.v
(src/glm53f_swiglu_q8.v: GLM-5.3-Flash's dense FFN, clamped SwiGLU over Q8_0).

The activation model is swiglu_q4k_gen's -- glm_act SILU semantics, sigmoid on a
+/-16-railed input then x*sigmoid, bf16 out -- imported rather than rewritten, so
the two SwiGLU gates cannot drift apart. The per-element tolerance follows the
same shape for the same reason: glm_act's polynomial silu is an approximation, so
the DOWN reduction is a functional check, and the bit-exact claim belongs to
glm_matmul_q4k's own gate.

THE CLAMP IS THE POINT AND IT IS ASYMMETRIC (swiglu_limit = 10.0 on every block):
    gate.clamp(min=None, max=+limit)     upper bound ONLY
    up  .clamp(min=-limit, max=+limit)   both bounds
Clamping the gate symmetrically only moves large-NEGATIVE gates, where silu is
already near zero. MEASURED at this slice, that difference reaches 0.25 against a
tolerance of 2.9 -- so the symmetric reading is NOT gated here, and saying so is
the point: the tolerance is forced by glm_act's polynomial silu (the 0.02*mag
term), no exact Python model of that polynomial exists, and a must-fail entry that
cannot fail is worse than none. The asymmetry IS gated, on a tighter slice, by
`make swiglu`'s own INJ_SWIGLU_SYMCLAMP leg on the Q4_K unit.

What this gate does check is that the clamp is THERE at all: the corpus is
verified to drive both bounds, and an unclamped DUT must fail the clamped golden
(the vacuity leg). That difference is large -- unclamped gates reach +/-30, where
silu is ~30 rather than ~10 -- so it is not near any tolerance.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swiglu_q4k_gen import bf16_round, bf16bits, bf16_mul, silu  # noqa: E402
from glm53f_kda_attn_gen import q80_rows  # noqa: E402

F32 = np.float32
LIMIT = 10.0


def fp16b(v):
    return int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0])


def seq_gemv_bf16(W, x):
    """fp32-accumulate, bf16 out -- glm_matmul_q4k's c_out shape."""
    out = np.empty(W.shape[0], F32)
    for r in range(W.shape[0]):
        a = F32(0.0)
        for c in range(W.shape[1]):
            a = F32(a + F32(F32(W[r, c]) * F32(x[c])))
        out[r] = bf16_round(a)
    return out


def ffn(x, Wg, Wu, Wd, clamp=True, sym_gate=False):
    gate = seq_gemv_bf16(Wg, x)
    up = seq_gemv_bf16(Wu, x)
    act = np.empty(gate.shape[0], F32)
    n_hi = n_lo = 0
    for i in range(gate.shape[0]):
        g, u = float(gate[i]), float(up[i])
        if clamp:
            if sym_gate:
                if abs(g) > LIMIT:
                    g = LIMIT if g > 0 else -LIMIT
            else:
                if g > LIMIT:
                    g = LIMIT; n_hi += 1
            if u > LIMIT:
                u = LIMIT; n_hi += 1
            elif u < -LIMIT:
                u = -LIMIT; n_lo += 1
        act[i] = bf16_mul(silu(F32(g)), F32(u))
    return seq_gemv_bf16(Wd, act), act, (n_hi, n_lo)


def _draw(rng, HIDDEN, INTER):
    # scaled so the pre-activation reliably reaches BOTH clamp bounds
    Wg = (rng.normal(size=(INTER, HIDDEN)) * 1.1).astype(F32)
    Wu = (rng.normal(size=(INTER, HIDDEN)) * 1.1).astype(F32)
    Wd = (rng.normal(size=(HIDDEN, INTER)) * 0.3).astype(F32)
    x = np.array([bf16_round(v) for v in rng.normal(size=HIDDEN) * 2.0], F32)
    return x, Wg, Wu, Wd


def gen(ntest, HIDDEN=16, INTER=32, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {HIDDEN} {INTER}"]
    hi = lo = 0
    for _ in range(ntest):
        x, Wg, Wu, Wd = _draw(rng, HIDDEN, INTER)
        qg, dg, deg = q80_rows(Wg)
        qu, du, deu = q80_rows(Wu)
        qd, dd, ded = q80_rows(Wd)
        y, act, (a, b) = ffn(x, deg, deu, ded)
        hi += a; lo += b
        out.append(" ".join(f"{bf16bits(v):04x}" for v in x))
        for q, d in ((qg, dg), (qu, du), (qd, dd)):
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in q.reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(d).reshape(-1)))
        out.append(" ".join(f"{bf16bits(v):04x}" for v in y))
        tol = []
        for o in range(HIDDEN):
            mag = float(np.sum(np.abs(act) * np.abs(ded[o])))
            tol.append(max(0.06 * abs(float(y[o])), 0.02 * mag, 0.03))
        out.append(" ".join(f"{t:.6f}" for t in tol))
    if report:
        print(f"clamp actually exercised: {hi} upper-bound hits, {lo} lower-bound hits "
              f"over {ntest} tests", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0x5C)
    HIDDEN, INTER = 16, 32
    n = 0
    fails = []
    live = {"noclamp": 0}
    sym_worst = 0.0
    sym_tolworst = 0.0
    hi = lo = 0
    trials = 12
    for _ in range(trials):
        x, Wg, Wu, Wd = _draw(rng, HIDDEN, INTER)
        _, dg, deg = q80_rows(Wg)
        _, du, deu = q80_rows(Wu)
        _, dd, ded = q80_rows(Wd)
        y, _, (a, b) = ffn(x, deg, deu, ded)
        hi += a; lo += b
        n += 1
        if y.shape != (HIDDEN,):
            fails.append("y shape")
        if not np.array_equal(ffn(x, deg, deu, ded, clamp=False)[0], y):
            live["noclamp"] += 1
        # measure (do not gate) the symmetric-clamp reading against the tolerance
        # this datapath's own glm_act error forces
        ys, act, _ = ffn(x, deg, deu, ded, sym_gate=True)[0], None, None
        _, act, _ = ffn(x, deg, deu, ded)
        for o in range(HIDDEN):
            dd_ = abs(float(ys[o] - y[o]))
            tl = max(0.06 * abs(float(y[o])),
                     0.02 * float(np.sum(np.abs(act) * np.abs(ded[o]))), 0.03)
            if dd_ > sym_worst:
                sym_worst, sym_tolworst = dd_, tl
    n += 2
    if hi == 0:
        fails.append("the corpus never hits the UPPER clamp bound -- it would not "
                     "distinguish a clamped SwiGLU from an unclamped one")
    if lo == 0:
        fails.append("the corpus never hits the LOWER clamp bound -- the asymmetry "
                     "between gate and up would be untested")
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"glm53f_swiglu_q8_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (corpus hits both clamp bounds: {hi} upper, {lo} lower; "
          f"noclamp changes the answer {live['noclamp']}/{trials}; symmetric-gate "
          f"clamp measured at worst {sym_worst:.4f} vs its {sym_tolworst:.4f} tolerance "
          f"-- inside it, so deliberately NOT gated here, see the module docstring)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 12
    outp = a[1] if len(a) > 1 else "build/glm53f_swiglu_q8_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest))
    print(f"wrote {outp}: {ntest} tests")
