#!/usr/bin/env python3
"""
glm53f_hc_block_gen.py -- vectors for test/glm53f_hc_block_tb.v
(the block skeleton: TWO mHC sites wrapped around two sublayers).

WHAT THIS GATE IS FOR.  mhc-site already pins the mHC numerics of ONE site. What
is new at block level is the WIRING: two sites driven from one instance with the
weights muxed, each with its OWN learned norm between `collapsed` and its
sublayer, and the streams the first site writes threaded into the second. The
injections target exactly that, and the golden is a full two-site composition so a
routing mistake cannot hide.

rmsnorm_unit is modelled EXACTLY (LANES=1): bf16 in, sequential fp32 sum of
squares, mean = sumsq * 1/LEN, + eps, the same Quake rsqrt, then
bf16(x*inv*gamma). So the only non-bitwise term in the whole composition is the
one mhc_map_step already publishes -- its polynomial exp.

The stub sublayer is sub_out = 0.5 * normed, which is exact in bf16 (an exponent
step) and COUPLED to the norm output, so a skipped or mis-routed norm cannot hide.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_map_gen import ref_map  # noqa: E402
from mhc_fn_gemv_gen import q80_pack, dut_emulate, fp32_rsqrt  # noqa: E402
from mhc_precision_study import bf16  # noqa: E402

F32 = np.float32
RMS_EPS = F32(1e-5)
MAP_ULP = {"pre": 1024, "post": 1024, "comb": 16384}


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def bf16b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16


def nudge(x, n):
    x = np.asarray(x, F32)
    b = np.frombuffer(np.ascontiguousarray(x).tobytes(), np.int32).astype(np.int64)
    return np.frombuffer(((b + n) & 0xFFFFFFFF).astype(np.uint32).tobytes(),
                         F32).reshape(x.shape)


def rmsnorm_bf16(x_bf, gam_bf):
    """Bit-for-bit src/rmsnorm_unit.v at LANES=1."""
    x_bf = np.asarray(x_bf, F32)
    gam_bf = np.asarray(gam_bf, F32)
    n = x_bf.shape[0]
    sumsq = F32(0.0)
    for v in x_bf:
        sumsq = F32(sumsq + F32(F32(v) * F32(v)))
    mean = F32(sumsq * F32(1.0 / n))
    inv = fp32_rsqrt(F32(mean + RMS_EPS))
    out = np.empty(n, F32)
    for i in range(n):
        out[i] = bf16(F32(F32(F32(x_bf[i]) * inv) * F32(gam_bf[i])))
    return out


def one_site(streams, q, d, base, scale, gam_bf, H, pert=None):
    mixed = dut_emulate(streams.reshape(-1), q, d)
    pre, post, comb = ref_map(mixed, base, scale, H, 20)
    if pert is not None:
        pre = nudge(pre, pert * MAP_ULP["pre"])
        post = nudge(post, pert * MAP_ULP["post"])
        comb = nudge(comb, pert * MAP_ULP["comb"])
    coll = ref.hc_collapse(streams, pre)
    normed = rmsnorm_bf16(bf16(coll), gam_bf)
    sub = bf16((normed * F32(0.5)).astype(F32))
    return normed, ref.hc_mix(streams, comb, post, sub.astype(F32))


def block(streams, sites, H, pert=None):
    s = streams
    normeds = []
    for (q, d, base, scale, gam) in sites:
        n, s = one_site(s, q, d, base, scale, gam, H, pert)
        normeds.append(n)
    return normeds, s


def _draw(rng, H, D):
    K, ROWS = H * D, (2 + H) * H
    sites = []
    for _ in range(2):
        fn = (rng.normal(size=(ROWS, K)) * (1.0 / np.sqrt(K))).astype(F32)
        q, d = q80_pack(fn)
        base = rng.normal(size=ROWS).astype(F32)
        scale = (F32(1.0) + rng.normal(size=3).astype(F32) * F32(0.3)).astype(F32)
        gam = bf16((F32(1.0) + rng.normal(size=D).astype(F32) * F32(0.3)).astype(F32))
        sites.append((q, d, base, scale, gam))
    streams = (rng.normal(size=(H, D)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
    return streams, sites


def gen(ntest, H=4, D=64, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {D}"]
    env_n = env_s = 0.0
    for _ in range(ntest):
        streams, sites = _draw(rng, H, D)
        normeds, final = block(streams, sites, H)
        for sg in (+1, -1):
            n2, f2 = block(streams, sites, H, pert=sg)
            env_n = max(env_n, max(float(np.abs(a - b).max()) for a, b in zip(n2, normeds)))
            env_s = max(env_s, float(np.abs(f2 - final).max()))
        out.append(" ".join(f"{f32b(v):08x}" for v in streams.reshape(-1)))
        for (q, d, base, scale, gam) in sites:
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in q.reshape(-1)))
            out.append(" ".join(f"{int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0]):04x}"
                                for v in np.asarray(d).reshape(-1)))
            out.append(" ".join(f"{f32b(v):08x}" for v in base))
            out.append(" ".join(f"{f32b(v):08x}" for v in scale))
            out.append(" ".join(f"{bf16b(v):04x}" for v in gam))
        for n in normeds:
            out.append(" ".join(f"{bf16b(v):04x}" for v in n))
        out.append(" ".join(f"{f32b(v):08x}" for v in final.reshape(-1)))
    if report:
        print(f"envelope across BOTH sites, implied by mhc_map's gated bounds, ABSOLUTE: "
              f"normed {env_n:.3e}, final streams {env_s:.3e}", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0xB10)
    H, D = 4, 64
    n = 0
    fails = []
    live = {k: 0 for k in ("same_weights", "norm_swap", "skip_norm", "stale_streams")}
    trials = 8
    for _ in range(trials):
        streams, sites = _draw(rng, H, D)
        normeds, final = block(streams, sites, H)
        n += 2
        if len(normeds) != 2:
            fails.append("expected two sites")
        if np.array_equal(final, streams):
            fails.append("block left the streams unchanged")
        # the two sites must not be interchangeable
        if not np.array_equal(block(streams, [sites[0], sites[0]], H)[1], final):
            live["same_weights"] += 1
        swapped = [(q, d, b, s, sites[1 - i][4]) for i, (q, d, b, s, g) in enumerate(sites)]
        if not np.array_equal(block(streams, swapped, H)[1], final):
            live["norm_swap"] += 1
        # skipping the norm: feed bf16(collapsed) straight to the stub
        s2 = streams
        for (q, d, base, scale, gam) in sites:
            mixed = dut_emulate(s2.reshape(-1), q, d)
            pre, post, comb = ref_map(mixed, base, scale, H, 20)
            coll = ref.hc_collapse(s2, pre)
            sub = bf16((bf16(coll) * F32(0.5)).astype(F32))
            s2 = ref.hc_mix(s2, comb, post, sub.astype(F32))
        if not np.array_equal(s2, final):
            live["skip_norm"] += 1
        # stale streams: the FFN site re-reads the ORIGINAL streams
        _, s_a = one_site(streams, *sites[0], H)
        _, s_stale = one_site(streams, *sites[1], H)
        if not np.array_equal(s_stale, final):
            live["stale_streams"] += 1
        del s_a
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"glm53f_hc_block_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (two sites, streams threaded; trap liveness over "
          f"{trials} draws: " + ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 6
    H = int(a[1]) if len(a) > 1 else 4
    D = int(a[2]) if len(a) > 2 else 64
    outp = a[3] if len(a) > 3 else "build/glm53f_hc_block_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, H, D))
    print(f"wrote {outp}: {ntest} tests H={H} D={D}")
