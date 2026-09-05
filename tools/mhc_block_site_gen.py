#!/usr/bin/env python3
"""
mhc_block_site_gen.py -- vectors for test/mhc_block_site_tb.v (src/mhc_block_site.v).

ONE mHC site end to end: fn GEMV -> map -> collapse -> [sublayer] -> mix, with the
four residual streams held across the site.

THE GOLDEN COMPOSES THE SPEC, NOT THE UNITS.  `mixed` comes from the bit-exact
GEMV emulation (that unit IS bitwise), but pre/post/comb come from
tools/mhc_map_gen.ref_map -- the float64 reference -- and collapse/mix from
tools/glm53_flash_ref.py's pinned hc_collapse / hc_mix.  So this checks that the
composition reproduces the SPECIFICATION, not merely that three RTL units were
wired together.

THE BOUND COMES FROM THE UNITS' OWN PUBLISHED CONTRACTS.  mhc_map_step is gated at
1024 ULP on pre/post and 16384 on comb (its polynomial exp is the limit). This
generator perturbs pre/post/comb by exactly those amounts, in the worst-case
direction, and pushes them through collapse and mix -- so the site's bound is
implied by the map's gate rather than read off this DUT.

The stub sublayer is sub_out = 0.5 * collapsed: exact in fp32, and COUPLED to
collapsed, so a broken collapse cannot hide behind an independent sublayer value.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_map_gen import ref_map  # noqa: E402
from mhc_fn_gemv_gen import q80_pack, dut_emulate  # noqa: E402

F32 = np.float32
MAP_ULP = {"pre": 1024, "post": 1024, "comb": 16384}


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def nudge(x, n):
    x = np.asarray(x, F32)
    b = np.frombuffer(np.ascontiguousarray(x).tobytes(), np.int32).astype(np.int64)
    return np.frombuffer(((b + n) & 0xFFFFFFFF).astype(np.uint32).tobytes(),
                         F32).reshape(x.shape)


def site(streams, q, d, base, scale, H, iters=20):
    """The specification for one site, with the stub sublayer folded in."""
    mixed = dut_emulate(streams.reshape(-1), q, d)
    pre, post, comb = ref_map(mixed, base, scale, H, iters)
    coll = ref.hc_collapse(streams, pre)
    sub = (coll * F32(0.5)).astype(F32)
    return coll, ref.hc_mix(streams, comb, post, sub), pre, post, comb


def envelope(streams, pre, post, comb, coll, out, H):
    """Worst deviation implied by the MAP's own gated bounds.

    Reported as ABSOLUTE error, not ULP. `comb` is doubly stochastic, so the mix's
    four terms nearly cancel and some outputs land near zero; a ULP (or relative)
    bound on those is dominated by the cancellation and comes out ~1.4e6 ULP,
    which at stream magnitudes is ~17% -- a bound that gates nothing. Absolute
    error against O(1) streams is the metric that means something here.
    """
    w_c, w_s = 0.0, 0.0
    for sg in (+1, -1):
        p = nudge(pre, sg * MAP_ULP["pre"])
        q = nudge(post, sg * MAP_ULP["post"])
        c = nudge(comb, sg * MAP_ULP["comb"])
        c2 = ref.hc_collapse(streams, p)
        s2 = ref.hc_mix(streams, c, q, (c2 * F32(0.5)).astype(F32))
        w_c = max(w_c, float(np.abs(np.asarray(c2, F32) - np.asarray(coll, F32)).max()))
        w_s = max(w_s, float(np.abs(np.asarray(s2, F32) - np.asarray(out, F32)).max()))
    return w_c, w_s


def _ulp(a, b):
    ab = np.frombuffer(np.ascontiguousarray(np.asarray(a, F32)).tobytes(), np.int32).astype(np.int64)
    bb = np.frombuffer(np.ascontiguousarray(np.asarray(b, F32)).tobytes(), np.int32).astype(np.int64)
    return int(np.abs(ab - bb).max())


def gen(ntest, H=4, D=64, seed=0, report=True):
    rng = np.random.default_rng(seed)
    K, ROWS = H * D, (2 + H) * H
    out_lines = [f"{ntest} {H} {D}"]
    env_c = env_s = 0.0
    peak = 0.0
    for _ in range(ntest):
        fn = (rng.normal(size=(ROWS, K)) * (1.0 / np.sqrt(K))).astype(F32)
        q, d = q80_pack(fn)
        base = rng.normal(size=ROWS).astype(F32)
        scale = (F32(1.0) + rng.normal(size=3).astype(F32) * F32(0.3)).astype(F32)
        streams = (rng.normal(size=(H, D)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
        coll, new, pre, post, comb = site(streams, q, d, base, scale, H)
        a, b = envelope(streams, pre, post, comb, coll, new, H)
        env_c, env_s = max(env_c, a), max(env_s, b)
        peak = max(peak, float(np.abs(new).max()), float(np.abs(coll).max()))
        out_lines.append(" ".join(f"{f32b(v):08x}" for v in streams.reshape(-1)))
        out_lines.append(" ".join(f"{int(np.uint8(v)):02x}" for v in q.reshape(-1)))
        out_lines.append(" ".join(f"{int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0]):04x}"
                                  for v in np.asarray(d).reshape(-1)))
        out_lines.append(" ".join(f"{f32b(v):08x}" for v in base))
        out_lines.append(" ".join(f"{f32b(v):08x}" for v in scale))
        out_lines.append(" ".join(f"{f32b(v):08x}" for v in coll))
        out_lines.append(" ".join(f"{f32b(v):08x}" for v in new.reshape(-1)))
    if report:
        print(f"envelope implied by mhc_map's own gated bounds (pre/post 1024, comb 16384 "
              f"ULP), ABSOLUTE: collapsed {env_c:.3e}, streams {env_s:.3e} "
              f"(peak |value| in corpus {peak:.2f})", file=sys.stderr)
    return "\n".join(out_lines) + "\n"


def _selftest():
    rng = np.random.default_rng(0xD0)
    H, D = 4, 64
    K, ROWS = H * D, (2 + H) * H
    n = 0
    fails = []
    live = {k: 0 for k in ("ignore_sub", "no_update", "pre_for_post")}
    trials = 12
    for _ in range(trials):
        fn = (rng.normal(size=(ROWS, K)) * (1.0 / np.sqrt(K))).astype(F32)
        q, d = q80_pack(fn)
        base = rng.normal(size=ROWS).astype(F32)
        scale = np.array([1.0, 1.0, 1.0], F32)
        streams = (rng.normal(size=(H, D)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
        coll, new, pre, post, comb = site(streams, q, d, base, scale, H)

        n += 2
        if coll.shape != (D,):
            fails.append("collapsed shape")
        if new.shape != (H, D):
            fails.append("streams shape")
        # the site must actually MOVE the streams
        n += 1
        if np.array_equal(new, streams):
            fails.append("site left the streams unchanged")

        sub = (coll * F32(0.5)).astype(F32)
        if not np.array_equal(ref.hc_mix(streams, comb, post, np.zeros(D, F32)), new):
            live["ignore_sub"] += 1
        if not np.array_equal(streams, new):
            live["no_update"] += 1
        if not np.array_equal(ref.hc_mix(streams, comb, pre, sub), new):
            live["pre_for_post"] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"mhc_block_site_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (site moves the streams; trap liveness over {trials} draws: " +
          ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 8
    H = int(a[1]) if len(a) > 1 else 4
    D = int(a[2]) if len(a) > 2 else 64
    outp = a[3] if len(a) > 3 else "build/mhc_block_site_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, H, D))
    print(f"wrote {outp}: {ntest} tests H={H} D={D}")
