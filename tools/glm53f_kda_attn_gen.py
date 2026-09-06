#!/usr/bin/env python3
"""
glm53f_kda_attn_gen.py -- vectors for test/glm53f_kda_attn_tb.v
(src/glm53f_kda_attn.v: the KDA layer PLUS the engine that fetches its own
nine Q8_0 projections off glm_matmul_q4k).

`make kda-layer` checks the datapath with the projections HANDED to it. This
checks that the same datapath still lands when they are STREAMED off real Q8_0
weights -- which is the difference between a unit and a sublayer, and what
GLM53F_KDA_RTL_PRESENT is waiting on.

The golden reuses glm53f_kda_layer_gen.layer on the DEQUANTISED weights, so any
disagreement is the streaming path, not the quantiser: the Q8_0 round trip is
part of the input, not part of the error.

Q8_0 BLOCKING ON THE SLICE.  ggml Q8_0 groups 32 consecutive K values per fp16
scale. Every projection here has K <= 16, so there is exactly ONE block per row
and the engine's `blk = pj*NB8 + (k>>5)` indexes block 0 throughout. At the real
shape (K = 4096 / 8192 / 128) there are many blocks and the same indexing walks
them; nothing about the layout changes, only how many there are.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm53f_kda_layer_gen import _draw, layer, ORDER  # noqa: E402

F32 = np.float32


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def bf16b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16


def fp16b(v):
    return int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0])


def q80_rows(W):
    """One fp16 scale + int8 codes per ROW (K <= 32 on this slice: one block)."""
    W = np.asarray(W, F32)
    amax = np.abs(W).max(axis=1)
    d = (amax / 127.0).astype(np.float16)
    dz = np.where(np.asarray(d, F32) == 0, F32(1), np.asarray(d, F32))
    q = np.clip(np.rint(W / dz[:, None]), -128, 127).astype(np.int8)
    deq = (q.astype(F32) * np.asarray(d, F32)[:, None]).astype(F32)
    return q, d, deq


def gen(ntest, MD=16, H=2, DK=4, DV=4, RANK=4, CK=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {MD} {H} {DK} {DV} {RANK} {CK}"]
    worst_q = 0.0
    for _ in range(ntest):
        x, W, cw, dtb, a_log, onw, state, hist = _draw(rng, MD, H, DK, DV, RANK, CK)
        codes, scales, deq = {}, {}, {}
        for nm in ORDER:
            q, d, dq = q80_rows(W[nm])
            codes[nm], scales[nm], deq[nm] = q, d, dq
            worst_q = max(worst_q, float(np.abs(dq - W[nm]).max()))
        y_out, s2, hist2, _ = layer(x, deq, cw, dtb, a_log, onw, state, hist, H, DK, DV)

        out.append(" ".join(f"{bf16b(v):04x}" for v in x))
        for nm in ORDER:
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in codes[nm].reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(scales[nm]).reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in np.exp(a_log.astype(np.float64)).astype(F32)))
        out.append(" ".join(f"{f32b(v):08x}" for v in dtb))
        out.append(" ".join(f"{f32b(v):08x}" for v in cw.reshape(-1)))
        out.append(" ".join(f"{bf16b(v):04x}" for v in onw))
        out.append(" ".join(f"{f32b(v):08x}" for v in state.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in hist.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in y_out))
        out.append(" ".join(f"{f32b(v):08x}" for v in s2.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in hist2.reshape(-1)))
    if report:
        print(f"Q8_0 round trip on the projection weights: worst |deq - w| = {worst_q:.3e} "
              f"(part of the INPUT here, not of the error)", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0xA77)
    MD, H, DK, DV, RANK, CK = 16, 2, 4, 4, 4, 4
    n = 0
    fails = []
    trials = 12
    worst = 0.0
    for _ in range(trials):
        x, W, cw, dtb, a_log, onw, state, hist = _draw(rng, MD, H, DK, DV, RANK, CK)
        deq = {}
        for nm in ORDER:
            q, d, dq = q80_rows(W[nm])
            deq[nm] = dq
            n += 1
            # the round trip must be lossy-but-close, and must NOT be silently exact
            if not np.allclose(dq, W[nm], atol=0.01):
                fails.append(f"Q8_0 round trip too lossy on {nm}")
            n += 1
            if np.array_equal(dq, W[nm]):
                fails.append(f"Q8_0 round trip is bit-identical on {nm} -- not exercising quantisation")
        y_q, _, _, _ = layer(x, deq, cw, dtb, a_log, onw, state, hist, H, DK, DV)
        y_f, _, _, _ = layer(x, W, cw, dtb, a_log, onw, state, hist, H, DK, DV)
        worst = max(worst, float(np.abs(y_q - y_f).max()))
        n += 1
        if np.array_equal(y_q, y_f):
            fails.append("quantised and unquantised layers agree bitwise -- Q8_0 is not reaching the output")
    if fails:
        print(f"glm53f_kda_attn_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (Q8_0 round trip lossy-but-close on all 9 projections and "
          f"visible at the output: worst |y_q8 - y_fp32| = {worst:.3e})")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 6
    outp = a[1] if len(a) > 1 else "build/glm53f_kda_attn_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest))
    print(f"wrote {outp}: {ntest} tests")
