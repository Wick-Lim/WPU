#!/usr/bin/env python3
"""
mhc_sinkhorn_gen.py -- vectors for test/mhc_sinkhorn_tb.v (src/mhc_sinkhorn.v).

The Sinkhorn projection inside GLM-5.3-Flash's hyper-connections: a 4x4
row-stochastic matrix pushed toward doubly stochastic by 1 column normalise then
`iters-1` (row, column) pairs, every normalise dividing by (sum + eps).

WHAT THE GOLDEN IS, AND WHY IT IS NOT BITWISE.  The golden is the reference's
own arithmetic: fp32, TRUE division, sequential reduction order.  The DUT cannot
match it bit-for-bit for two stated reasons -- it has no divider (x*recip(y) via
glm_fp_recip.vh) and fp32_add is 1 ULP low on ~0.04% of pairs (`make fp-ieee`).
So this generator also EMULATES the DUT exactly (same Newton seed and steps, same
sequential order) and measures the gap, and the TB bound is set from that
measurement with headroom -- the same stance as `make fp-sigmoid`.

Reduction order: numpy's .sum() is pairwise for length >= 8 but SEQUENTIAL at
length 4 (measured 0/4000 differing on both axes), so a streaming datapath can
match it here.  `seq_sum` pins the order anyway rather than relying on that.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32
EPS = F32(1e-6)


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def seq_sum(a, axis):
    """fp32 reduction in explicit sequential index order (see module docstring)."""
    a = np.moveaxis(np.asarray(a, F32), axis, 0)
    s = a[0].astype(F32)
    for i in range(1, a.shape[0]):
        s = (s + a[i]).astype(F32)
    return s


def pair_sum(a, axis):
    """The WRONG order: pairwise. Only used to check INJ_SINK_PAIRWISE is live."""
    a = np.moveaxis(np.asarray(a, F32), axis, 0)
    assert a.shape[0] == 4
    return ((a[0] + a[1]).astype(F32) + (a[2] + a[3]).astype(F32)).astype(F32)


def newton_recip(y, iters=4):
    """Bit-for-bit glm_fp_recip.vh: seed 0x7EF311C3 - bits, then r*(2 - y*r)."""
    y = np.asarray(y, F32)
    yb = np.frombuffer(np.ascontiguousarray(y).tobytes(), np.uint32).astype(np.int64)
    rb = ((np.int64(0x7EF311C3) - yb) & 0xFFFFFFFF).astype(np.uint32)
    r = np.frombuffer(rb.tobytes(), F32).reshape(y.shape).copy()
    for _ in range(iters):
        t = (y * r).astype(F32)
        t = (F32(2.0) - t).astype(F32)
        r = (r * t).astype(F32)
    return r


def sinkhorn(c, iters=20, use_recip=False, no_eps=False,
             row_first=False, symm=False, pairwise=False):
    """The reference schedule, with each trap available as a flag."""
    c = np.asarray(c, F32).copy()
    if symm:
        axes = [-1, -2] * iters                     # INJ_SINK_SYMM
    elif row_first:
        axes = [-1] + [-2, -1] * (iters - 1)        # INJ_SINK_ROWFIRST
    else:
        axes = [-2] + [-1, -2] * (iters - 1)        # the reference: column first
    red = pair_sum if pairwise else seq_sum
    for ax in axes:
        s = red(c, ax)
        s = s if no_eps else (s + EPS).astype(F32)  # INJ_SINK_NOEPS drops the eps
        s = np.expand_dims(s, ax)
        c = (c * newton_recip(s)).astype(F32) if use_recip else (c / s).astype(F32)
    return c


def resid(c):
    return max(float(abs(c.sum(-1) - 1).max()), float(abs(c.sum(-2) - 1).max()))


def _corpus(ntest, H, rng):
    """Realistic softmax rows across the logit spreads 4.3i swept, plus adversarial."""
    out = []
    spreads = [0.35, 0.7, 1.42, 2.84, 5.68, 11.36]
    for i in range(ntest):
        kind = i % 8
        if kind == 6:                       # near-uniform: already doubly stochastic
            c = np.full((H, H), 1.0 / H, F32)
        elif kind == 7:                     # near one-hot: the hardest for Sinkhorn
            lg = (np.eye(H) * 30.0).astype(F32)
            c = (ref.softmax(lg, -1) + EPS).astype(F32)
        else:
            sc = spreads[i % len(spreads)]
            lg = (rng.normal(size=(H, H)) * sc).astype(F32)
            c = (ref.softmax(lg, -1) + EPS).astype(F32)
        out.append(c)
    return out


def gen(ntest, H=4, iters=20, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {iters}"]
    worst_ulp, worst_abs, worst_res_g, worst_res_d = 0, 0.0, 0.0, 0.0
    for c0 in _corpus(ntest, H, rng):
        g = sinkhorn(c0, iters)                       # golden: true division
        d = sinkhorn(c0, iters, use_recip=True)       # what the DUT computes
        gb = np.frombuffer(np.ascontiguousarray(g).tobytes(), np.int32).astype(np.int64)
        db = np.frombuffer(np.ascontiguousarray(d).tobytes(), np.int32).astype(np.int64)
        worst_ulp = max(worst_ulp, int(np.abs(gb - db).max()))
        worst_abs = max(worst_abs, float(np.abs(g - d).max()))
        worst_res_g = max(worst_res_g, resid(g))
        worst_res_d = max(worst_res_d, resid(d))
        out.append(" ".join(f"{f32b(v):08x}" for v in c0.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in g.reshape(-1)))
    if report:
        print(f"emulated DUT (x*recip) vs golden (true div) over {ntest} tests: "
              f"worst {worst_ulp} ULP, {worst_abs:.3e} abs", file=sys.stderr)
        print(f"double-stochasticity residual at {iters} iters: golden {worst_res_g:.3e}, "
              f"DUT {worst_res_d:.3e}  (entries ~{1.0/H:.2f})", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    """Prove each documented trap actually changes the answer -- otherwise the
    corresponding must-fail injection is decoration."""
    rng = np.random.default_rng(0x51)
    n = 0
    fails = []
    live = {k: 0 for k in ("symm", "row_first", "no_eps", "pairwise")}
    trials = 200
    for _ in range(trials):
        c0 = (ref.softmax((rng.normal(size=(4, 4)) * 1.42).astype(F32), -1) + EPS).astype(F32)
        g = sinkhorn(c0, 20)
        n += 1
        if not np.all(np.isfinite(g)):
            fails.append("golden not finite")
        # the reciprocal substitution must stay far below a bf16 map's 2.8e-3
        n += 1
        if float(np.abs(sinkhorn(c0, 20, use_recip=True) - g).max()) > 1e-5:
            fails.append("x*recip(y) moved comb by more than 1e-5")
        for k in live:
            if not np.array_equal(sinkhorn(c0, 20, **{k: True}), g):
                live[k] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    # 39 passes, not 40 and not 20: the off-by-one the reference docstring warns about
    n += 1
    c0 = (ref.softmax((rng.normal(size=(4, 4)) * 1.42).astype(F32), -1) + EPS).astype(F32)
    if np.array_equal(sinkhorn(c0, 20), sinkhorn(c0, 21)):
        fails.append("iters=20 and 21 agree -- the iteration count is not observable")
    if fails:
        print(f"mhc_sinkhorn_gen self-test: {len(fails)}/{n} FAILED")
        for f in fails[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (trap liveness over {trials} draws: " +
          ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 64
    H = int(a[1]) if len(a) > 1 else 4
    iters = int(a[2]) if len(a) > 2 else 20
    outp = a[3] if len(a) > 3 else "build/mhc_sinkhorn_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, H, iters))
    print(f"wrote {outp}: {ntest} tests H={H} iters={iters}")
