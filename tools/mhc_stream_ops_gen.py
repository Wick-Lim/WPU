#!/usr/bin/env python3
"""
mhc_stream_ops_gen.py -- vectors for test/mhc_stream_ops_tb.v (src/mhc_stream_ops.v).

The two D-wide datapaths of the mHC residual path:
    COLLAPSE  collapsed[d]   = sum_h pre[h] * streams[h][d]
    MIX       streams'[h][d] = sum_g comb[h][g]*streams[g][d] + post[h]*sub[d]

The golden is tools/glm53_flash_ref.py's hc_collapse / hc_mix, which pin the
reduction order SEQUENTIALLY because numpy's own matmul does not use it (measured
300/300 differing on [4,4]@[4,D]; the gap is one fp32 rounding, an FMA-vs-mul-then-add
difference, but it is not zero). Both are pure fp32 mul/add, so unlike the mHC map
this unit CAN be bitwise -- and is gated that way, with fp32_add's known 1-ULP
non-conformance (`make fp-ieee`) as the only slack.

The corpus uses REAL map outputs -- comb from an actual Sinkhorn projection, so it
is doubly stochastic and its four terms nearly cancel; pre = sigmoid+eps in
(0,1]; post = 2*sigmoid in [0,2]. Feeding uniform randoms instead would miss the
cancellation that makes the reduction order matter at all.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_map_gen import ref_map  # noqa: E402

F32 = np.float32
EPS = F32(1e-6)


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def _draw(rng, H, D):
    """A realistic (streams, pre, comb, post, sub): comb is a true Sinkhorn output."""
    MIX = (2 + H) * H
    mixed = (rng.normal(size=MIX) * rng.choice([0.5, 1.42, 4.0])).astype(F32)
    base = rng.normal(size=MIX).astype(F32)
    scale = np.array([1.0, 1.0, 1.0], F32)
    pre, post, comb = ref_map(mixed, base, scale, H, 20)
    streams = (rng.normal(size=(H, D)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
    sub = (rng.normal(size=D) * 0.5).astype(F32)
    return streams, pre, comb, post, sub


def gen(ntest, H=4, D=64, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {D}"]
    worst_cancel = 0.0
    for _ in range(ntest):
        streams, pre, comb, post, sub = _draw(rng, H, D)
        coll = ref.hc_collapse(streams, pre)
        mixed = ref.hc_mix(streams, comb, post, sub)
        # how much cancellation the corpus actually exercises, reported so the
        # "the order matters" claim is backed by this corpus, not by assertion
        terms = np.abs(comb[:, :, None] * streams[None, :, :]).sum(1)
        worst_cancel = max(worst_cancel,
                           float(np.max(terms / np.maximum(np.abs(mixed), 1e-12))))
        out.append(" ".join(f"{f32b(v):08x}" for v in streams.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in pre))
        out.append(" ".join(f"{f32b(v):08x}" for v in comb.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in post))
        out.append(" ".join(f"{f32b(v):08x}" for v in sub))
        out.append(" ".join(f"{f32b(v):08x}" for v in coll))
        out.append(" ".join(f"{f32b(v):08x}" for v in mixed.reshape(-1)))
    if report:
        print(f"corpus cancellation: worst sum|terms| / |result| = {worst_cancel:.1f}x "
              f"(1.0 = no cancellation)", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0xA1)
    H, D = 4, 64
    n = 0
    fails = []
    live = {k: 0 for k in ("mix_transpose", "mix_nopost", "collapse_nopre", "post_first")}
    trials = 120
    for _ in range(trials):
        streams, pre, comb, post, sub = _draw(rng, H, D)
        coll = ref.hc_collapse(streams, pre)
        mixed = ref.hc_mix(streams, comb, post, sub)
        n += 2
        if coll.shape != (D,):
            fails.append("collapse shape")
        if mixed.shape != (H, D):
            fails.append("mix shape")

        if not np.array_equal(ref.hc_mix(streams, comb.T.copy(), post, sub), mixed):
            live["mix_transpose"] += 1
        if not np.array_equal(ref.hc_mix(streams, comb, np.zeros(H, F32), sub), mixed):
            live["mix_nopost"] += 1
        if not np.array_equal(ref.hc_collapse(streams, np.ones(H, F32)), coll):
            live["collapse_nopre"] += 1
        # post added FIRST instead of last -- a pure rounding-order change
        pf = np.empty((H, D), F32)
        for h in range(H):
            a = (post[h] * sub).astype(F32)
            for g in range(H):
                a = (a + (comb[h, g] * streams[g]).astype(F32)).astype(F32)
            pf[h] = a
        if not np.array_equal(pf, mixed):
            live["post_first"] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"mhc_stream_ops_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (trap liveness over {trials} draws: " +
          ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 16
    H = int(a[1]) if len(a) > 1 else 4
    D = int(a[2]) if len(a) > 2 else 64
    outp = a[3] if len(a) > 3 else "build/mhc_stream_ops_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, H, D))
    print(f"wrote {outp}: {ntest} tests H={H} D={D}")
