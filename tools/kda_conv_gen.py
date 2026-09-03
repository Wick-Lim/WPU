#!/usr/bin/env python3
"""
kda_conv_gen.py -- golden vectors for the KDA short causal conv step
(src/kda_conv_step.v): depthwise K-tap conv over a [C, K-1] history, then SiLU.

Reference: `causal_conv1d_update` (seq_len == 1) -- cat(state, x), keep the last
K-1 as the new state, depthwise F.conv1d with groups=C and NO bias, then the
activation. The GGUF carries ssm_conv1d_{q,k,v}.weight as [4, 1, 8192] F32
(K=4 taps, 8192 = 64 heads x 128), and `conv1d.bias = False`.

ACCURACY CONTRACT.  The reference runs the conv in the WEIGHT dtype (bf16) and
lets torch pick the accumulation order, which is implementation-defined and so
cannot be matched bitwise by any fixed datapath.  This repo therefore pins its
OWN order as the contract and states the divergence, exactly as it does for the
KDA reductions and for llama.cpp:
    conv[c] = bf16_RNE( ((w0*s0 + w1*s1) + w2*s2) + w3*x )     fp32 mul/add,
                                                               taps ASCENDING,
                                                               oldest -> newest
    y[c]    = silu(conv[c])
Two legs fall out of that, and the RTL exposes the pre-activation value so the
first can be checked BITWISE:
    conv_bf16 : bitwise  -- fp32 mul/add + one RNE round, all exact primitives
                            (modulo fp32_add's pinned 1-ULP gap, `make fp-ieee`)
    y_bf16    : tolerance -- glm_act's SiLU is a polynomial approximation

Orientation is a trap worth a must-fail: F.conv1d CORRELATES (no kernel flip),
so w[K-1] multiplies the NEWEST sample.  -DINJ_CONV_FLIP reverses the taps.

Vector format:
  NTEST C K
  per test:
    w         : C*K     8hex fp32   channel-major, taps oldest..newest
    state_in  : C*(K-1) 8hex fp32   channel-major, oldest..newest
    x         : C       8hex fp32
    conv_bf16 : C       4hex bf16   pre-activation golden (bitwise leg)
    y_bf16    : C       4hex bf16   post-SiLU golden      (tolerance leg)
    state_out : C*(K-1) 8hex fp32
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
    """fp32 -> bf16 bits, round-to-nearest-even (q4k_ref.bf16_round is the
    repo's proven RNE golden)."""
    v = q4k_ref.bf16_round(np.float32(x))
    return (int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16) & 0xFFFF


def conv_pinned(state_row, x, w_row):
    """The CONTRACT order: fp32 products, accumulated ascending oldest->newest."""
    win = list(np.asarray(state_row, F32)) + [F32(x)]
    acc = F32(0.0)
    for k in range(len(w_row)):
        acc = F32(acc + F32(F32(w_row[k]) * win[k]))
    return acc


def gen(ntest, C, K, seed=0):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {C} {K}"]
    for _ in range(ntest):
        w = rng.normal(size=(C, K)).astype(F32) * F32(0.5)
        st = rng.normal(size=(C, K - 1)).astype(F32)
        x = rng.normal(size=C).astype(F32)
        conv = np.array([conv_pinned(st[c], x[c], w[c]) for c in range(C)], F32)
        conv_bf = np.array([bf16b(v) for v in conv])
        # SiLU is applied to the bf16-ROUNDED conv value, since that is what the
        # RTL hands glm_act; golden silu in fp64 then rounded to bf16.
        conv_bf_val = np.array([np.frombuffer(np.uint32(b << 16).tobytes(), F32)[0]
                                for b in conv_bf], F32)
        y_bf = np.array([bf16b(ref.silu(v)) for v in conv_bf_val])
        st_new = np.concatenate([st[:, 1:], x[:, None]], axis=1).astype(F32)

        # cross-check the pinned order against the transcribed reference at the
        # level where they MUST agree: the window/shift semantics.
        y_ref, st_ref = ref.causal_conv_step(st, x, w)
        assert np.array_equal(np.frombuffer(st_new.tobytes(), np.uint32),
                              np.frombuffer(st_ref.tobytes(), np.uint32)), \
            "state shift disagrees with the reference"

        out.append(" ".join(f"{f32b(v):08x}" for v in w.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in st.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in x))
        out.append(" ".join(f"{b:04x}" for b in conv_bf))
        out.append(" ".join(f"{b:04x}" for b in y_bf))
        out.append(" ".join(f"{f32b(v):08x}" for v in st_new.reshape(-1)))
    return "\n".join(out) + "\n"


def _selftest():
    """The pinned ascending-tap order must equal the reference's dot up to
    fp32 reassociation -- and, critically, a FLIPPED tap order must NOT: that is
    what makes the INJ_CONV_FLIP leg mean something."""
    rng = np.random.default_rng(3)
    C, K = 8, 4
    same = flipped_differs = 0
    n = 200
    for _ in range(n):
        w = rng.normal(size=(C, K)).astype(F32)
        st = rng.normal(size=(C, K - 1)).astype(F32)
        x = rng.normal(size=C).astype(F32)
        y_ref, _ = ref.causal_conv_step(st, x, w)
        # the reference's silu input is the fp32 dot; compare pre-activation via
        # inverting is impossible, so compare the dots directly with a tolerance
        dot = np.array([conv_pinned(st[c], x[c], w[c]) for c in range(C)], F32)
        dot_ref = (np.concatenate([st, x[:, None]], 1) * w).sum(axis=1).astype(F32)
        if np.allclose(dot, dot_ref, rtol=1e-5, atol=1e-6):
            same += 1
        dot_flip = np.array([conv_pinned(st[c], x[c], w[c][::-1]) for c in range(C)], F32)
        if not np.allclose(dot_flip, dot_ref, rtol=1e-3, atol=1e-3):
            flipped_differs += 1
    if same != n or flipped_differs != n:
        print(f"kda_conv_gen self-test FAILED: pinned==ref {same}/{n}, flipped!=ref {flipped_differs}/{n}")
        return 1
    print(f"ALL {2 * n} TESTS PASSED (pinned ascending-tap dot matches the reference "
          f"to fp32 reassociation; a flipped-tap dot never does, so INJ_CONV_FLIP is live)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 32
    C = int(a[1]) if len(a) > 1 else 8
    K = int(a[2]) if len(a) > 2 else 4
    outp = a[3] if len(a) > 3 else "build/kda_conv_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w").write(gen(ntest, C, K))
    print(f"wrote {outp}: {ntest} tests C={C} K={K}")
