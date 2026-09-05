#!/usr/bin/env python3
"""
mhc_fn_gemv_gen.py -- vectors for test/mhc_fn_gemv_tb.v (src/mhc_fn_gemv.v).

The mHC `fn` GEMV: raw residual streams -> the 24 mixed logits, with the RMS
normalisation folded past the GEMV (rms is a scalar, so one streaming pass does
both sum(x^2) and all 24 dot products).

TWO GOLDENS, ON PURPOSE.
  * The EMITTED golden emulates the DUT exactly -- Q8_0 dequant as
    fp16_scale * int8_code, strictly sequential fp32 accumulation from +0.0, mean
    = sum * 2^-log2(K), and glm_fp.vh's Quake rsqrt (0x5F3759DF seed + 2 Newton
    steps) reproduced bit for bit. So the gate is BITWISE, with fp32_add's
    measured 1-ULP non-conformance (`make fp-ieee`) the only slack.
  * The REPORTED number is the deviation from the SPEC path -- normalise first in
    float64, then GEMV -- which is what the fold and the approximate rsqrt
    actually cost against tools/glm53_flash_ref.py. A bitwise gate against an
    emulation only proves the RTL matches my model of it; that second number is
    what says the model is the right one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

F32 = np.float32
RMS_EPS = F32(1e-5)


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def fp32_rsqrt(x):
    """Bit-for-bit src/glm_fp.vh fp32_rsqrt: Quake seed + 2 Newton refinements."""
    x = F32(x)
    xhalf = F32(F32(0.5) * x)
    xb = int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])
    yb = (0x5F3759DF - (xb >> 1)) & 0xFFFFFFFF
    y = np.frombuffer(np.uint32(yb).tobytes(), F32)[0]
    for _ in range(2):
        yy = F32(y * y)
        xyy = F32(xhalf * yy)
        t = F32(F32(1.5) - xyy)
        y = F32(y * t)
    return F32(y)


def q80_pack(w, qk=32):
    """ggml Q8_0: per-`qk` block fp16 scale + int8 codes. Returns (codes, scales)."""
    w = np.asarray(w, F32)
    rows, k = w.shape
    b = w.reshape(rows, k // qk, qk)
    d = (np.abs(b).max(-1) / 127.0).astype(np.float16)
    dz = np.where(np.asarray(d, F32) == 0, F32(1), np.asarray(d, F32))
    q = np.clip(np.rint(b / dz[:, :, None]), -128, 127).astype(np.int8)
    return q.reshape(rows, k), d


def dut_emulate(x, q, d, qk=32):
    """Exactly what src/mhc_fn_gemv.v computes, in the same order."""
    rows, K = q.shape
    acc = np.zeros(rows, F32)
    sq = F32(0.0)
    dq = np.asarray(d, F32)
    for k in range(K):
        xk = F32(x[k])
        blk = k // qk
        for r in range(rows):
            w = F32(dq[r, blk] * F32(q[r, k]))
            acc[r] = F32(acc[r] + F32(w * xk))
        sq = F32(sq + F32(xk * xk))
    mean = F32(sq * F32(1.0 / K))
    rinv = fp32_rsqrt(F32(mean + RMS_EPS))
    return (acc * rinv).astype(F32)


def spec_path(x, q, d):
    """The reference shape: normalise in float64, then GEMV on the same weights."""
    rows, K = q.shape
    w = (np.asarray(d, F32)[:, :, None] * q.reshape(rows, K // 32, 32).astype(F32)
         ).reshape(rows, K).astype(F32)
    rms = np.sqrt((np.asarray(x, np.float64) ** 2).mean() + float(RMS_EPS))
    flat = (np.asarray(x, F32) / F32(rms)).astype(F32)
    return (w @ flat).astype(F32)


def gen(ntest, H=4, D=64, seed=0, report=True):
    rng = np.random.default_rng(seed)
    K = H * D
    ROWS = (2 + H) * H
    out = [f"{ntest} {H} {D}"]
    worst_spec = 0.0
    for _ in range(ntest):
        fn = (rng.normal(size=(ROWS, K)) * (1.0 / np.sqrt(K))).astype(F32)
        q, d = q80_pack(fn)
        x = (rng.normal(size=K) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
        g = dut_emulate(x, q, d)
        s = spec_path(x, q, d)
        worst_spec = max(worst_spec,
                         float(np.max(np.abs(g - s) / np.maximum(np.abs(s), 1e-6))))
        out.append(" ".join(f"{f32b(v):08x}" for v in x))
        out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in q.reshape(-1)))
        out.append(" ".join(f"{int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0]):04x}"
                            for v in np.asarray(d).reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in g))
    if report:
        print(f"fold + Quake rsqrt vs the spec path (normalise-then-GEMV, float64 rms): "
              f"worst {worst_spec:.3e} rel on `mixed` over {ntest} tests", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0xC3)
    H, D = 4, 64
    K, ROWS = H * D, (2 + H) * H
    n = 0
    fails = []
    live = {k: 0 for k in ("q8_noscale", "mean_sum", "no_eps")}
    trials = 24
    for _ in range(trials):
        fn = (rng.normal(size=(ROWS, K)) * (1.0 / np.sqrt(K))).astype(F32)
        q, d = q80_pack(fn)
        x = (rng.normal(size=K) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
        g = dut_emulate(x, q, d)

        # the emulation must track the spec path closely, or the model is wrong
        n += 1
        rel = float(np.max(np.abs(g - spec_path(x, q, d)) /
                           np.maximum(np.abs(spec_path(x, q, d)), 1e-6)))
        if rel > 5e-3:
            fails.append(f"emulation drifted from the spec path: {rel:.2e}")

        # Q8_0 round trip must actually be lossy-but-close, not silently exact
        n += 1
        wdq = (np.asarray(d, F32)[:, :, None] *
               q.reshape(ROWS, K // 32, 32).astype(F32)).reshape(ROWS, K)
        if not np.allclose(wdq, fn, atol=0.02):
            fails.append("Q8_0 pack/unpack is not close to the original weights")

        acc = np.zeros(ROWS, F32)
        sq = F32(0.0)
        dq = np.asarray(d, F32)
        for k in range(K):
            xk = F32(x[k])
            for r in range(ROWS):
                acc[r] = F32(acc[r] + F32(F32(dq[r, k // 32] * F32(q[r, k])) * xk))
            sq = F32(sq + F32(xk * xk))
        if not np.array_equal((acc * fp32_rsqrt(F32(F32(sq * F32(1.0 / K)) + RMS_EPS))
                               ).astype(F32),
                              (acc * fp32_rsqrt(F32(sq + RMS_EPS))).astype(F32)):
            live["mean_sum"] += 1
        if not np.array_equal((acc * fp32_rsqrt(F32(F32(sq * F32(1.0 / K)) + RMS_EPS))
                               ).astype(F32),
                              (acc * fp32_rsqrt(F32(sq * F32(1.0 / K)))).astype(F32)):
            live["no_eps"] += 1
        accn = np.zeros(ROWS, F32)
        for k in range(K):
            xk = F32(x[k])
            for r in range(ROWS):
                accn[r] = F32(accn[r] + F32(F32(q[r, k]) * xk))
        if not np.array_equal(accn, acc):
            live["q8_noscale"] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"mhc_fn_gemv_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (DUT emulation tracks the spec path; Q8_0 round trip "
          f"lossy-but-close; trap liveness over {trials} draws: " +
          ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 12
    H = int(a[1]) if len(a) > 1 else 4
    D = int(a[2]) if len(a) > 2 else 64
    outp = a[3] if len(a) > 3 else "build/mhc_fn_gemv_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, H, D))
    print(f"wrote {outp}: {ntest} tests H={H} D={D}")
