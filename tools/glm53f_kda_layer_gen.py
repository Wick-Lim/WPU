#!/usr/bin/env python3
"""
glm53f_kda_layer_gen.py -- vectors for test/glm53f_kda_layer_tb.v
(src/glm53f_kda_layer.v: ONE Kimi Delta Attention layer, 34 of 45 blocks).

The golden composes tools/glm53_flash_ref.py end to end -- projections, the ONE
depthwise conv over the CONCATENATION of q,k,v with SiLU, the forget gate, the
delta-rule recurrence and the gated output norm -- so passing means the layer
reproduces the SPECIFICATION, not that four already-gated units were wired up.

TOLERANCE, and where the bound comes from.  The four units are individually gated
at their own tolerances and the dominant term is known and already measured: the
gate path runs on glm_act's bf16 sigmoid, worth ~1.24 %/1.34 % on the two values
the recurrence consumes, and the compounding study puts a KDA layer's output error
at 1-3 % (docs/GLM53_FLASH_PORT.md 4.3e). The TB bound is 6 % relative + 0.01
absolute -- roughly 2x that measured envelope -- and the worst is printed, so a
regression moves a number. The bound is NOT read off this DUT.

WHAT THE STUB PROJECTION RESPONDER IS FOR.  The nine GEMVs are answered
behaviourally in the TB from the weights in this file, in fp32 and in the same
sequential order this generator uses. That isolates the LAYER's job -- sequencing,
the conv history, the recurrent state, the fp32/bf16 boundaries -- from the GEMV
engine, which q4k/mixedtype already gate. Driving glm_matmul_q4k with Q8_0 lanes
is the follow-on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_precision_study import bf16  # noqa: E402

F32 = np.float32


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def bf16b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16


def seq_gemv(W, x):
    """fp32-accumulate GEMV, sequential order, bf16 OUT.

    The bf16 output is not a shortcut: glm_matmul_q4k's c_out is bf16, the model's
    own linear layers are bf16-out, and every activation in this repo is bf16. The
    layer's proj_out port stays fp32-TYPED and carries bf16-VALUED data, so the
    weight-streaming step can drop the real engine in without changing the layer's
    contract. Accumulation stays fp32 -- that is the engine's behaviour too.
    """
    W = np.asarray(W, F32)
    x = np.asarray(x, F32)
    out = np.empty(W.shape[0], F32)
    for r in range(W.shape[0]):
        a = F32(0.0)
        for c in range(W.shape[1]):
            a = F32(a + F32(W[r, c] * x[c]))
        out[r] = bf16(a)
    return out


def layer(x_bf, W, cw, dtb, a_log, onw, state, hist, H, DK, DV):
    """The specification for one KDA decode step."""
    x = np.asarray(x_bf, F32)
    q = seq_gemv(W["q"], x)
    k = seq_gemv(W["k"], x)
    v = seq_gemv(W["v"], x)
    cat = np.concatenate([q, k, v]).astype(F32)
    y_conv, hist2 = ref.causal_conv_step(hist, cat, cw)
    n = H * DK
    qc = y_conv[:n].reshape(H, DK)
    kc = y_conv[n:2 * n].reshape(H, DK)
    vc = y_conv[2 * n:].reshape(H, DV)

    b_raw = seq_gemv(W["b"], x)
    f = seq_gemv(W["fb"], seq_gemv(W["fa"], x))
    ggate = seq_gemv(W["gb"], seq_gemv(W["ga"], x))

    g = ref.forget_gate(f.reshape(H, DK), dtb.reshape(H, DK), a_log.reshape(H, 1))
    beta = ref.sigmoid(b_raw)

    out, s2 = ref.kda_step(state, qc, kc, vc, g, beta)
    # The recurrence's output is bf16-VALUED at o_norm entry in the reference
    # (recurrent_kimi_delta_attention returns .to(bfloat16)), which is why
    # kda_onorm_step takes a bf16 x port. Rounding here is faithful to the model,
    # not a concession to the DUT -- leaving it out made the composed golden
    # disagree with the layer on `y` alone while state and hist matched.
    out_bf = bf16(out)
    y = np.stack([ref.rmsnorm_gated(out_bf[h], onw, ggate.reshape(H, DV)[h])
                  for h in range(H)])
    y_out = seq_gemv(W["o"], y.reshape(-1))
    return y_out, s2, hist2, out


def _draw(rng, MD, H, DK, DV, RANK, CK):
    n = H * DK
    W = {
        "q": (rng.normal(size=(n, MD)) * 0.3).astype(F32),
        "k": (rng.normal(size=(n, MD)) * 0.3).astype(F32),
        "v": (rng.normal(size=(H * DV, MD)) * 0.3).astype(F32),
        "b": (rng.normal(size=(H, MD)) * 0.3).astype(F32),
        "fa": (rng.normal(size=(RANK, MD)) * 0.3).astype(F32),
        "fb": (rng.normal(size=(n, RANK)) * 0.3).astype(F32),
        "ga": (rng.normal(size=(RANK, MD)) * 0.3).astype(F32),
        "gb": (rng.normal(size=(H * DV, RANK)) * 0.3).astype(F32),
        "o": (rng.normal(size=(MD, H * DV)) * 0.3).astype(F32),
    }
    x = bf16((rng.normal(size=MD) * 0.8).astype(F32))
    cw = (rng.normal(size=(3 * n, CK)) * 0.5).astype(F32)
    dtb = (rng.normal(size=n) * 0.5).astype(F32)
    a_log = (rng.normal(size=H) * 0.5).astype(F32)
    onw = bf16((F32(1.0) + rng.normal(size=DV).astype(F32) * F32(0.3)).astype(F32))
    state = (rng.normal(size=(H, DK, DV)) * 0.2).astype(F32)
    hist = (rng.normal(size=(3 * n, CK - 1)) * 0.4).astype(F32)
    return x, W, cw, dtb, a_log, onw, state, hist


ORDER = ["q", "k", "v", "b", "fa", "fb", "ga", "gb", "o"]


def gen(ntest, MD=16, H=2, DK=4, DV=4, RANK=4, CK=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {MD} {H} {DK} {DV} {RANK} {CK}"]
    worst_state = 0.0
    for _ in range(ntest):
        x, W, cw, dtb, a_log, onw, state, hist = _draw(rng, MD, H, DK, DV, RANK, CK)
        y_out, s2, hist2, rec = layer(x, W, cw, dtb, a_log, onw, state, hist, H, DK, DV)
        worst_state = max(worst_state, float(np.abs(s2 - state).max()))
        out.append(" ".join(f"{bf16b(v):04x}" for v in x))
        for nm in ORDER:
            out.append(" ".join(f"{f32b(v):08x}" for v in np.asarray(W[nm]).reshape(-1)))
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
        print(f"the recurrence actually moves the state: worst |s' - s| = {worst_state:.3e} "
              f"over {ntest} tests", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0x4DA)
    MD, H, DK, DV, RANK, CK = 16, 2, 4, 4, 4, 4
    n = 0
    fails = []
    live = {k: 0 for k in ("no_state", "conv_nohist", "qk_swap", "gate_order")}
    trials = 16
    for _ in range(trials):
        x, W, cw, dtb, a_log, onw, state, hist = _draw(rng, MD, H, DK, DV, RANK, CK)
        y, s2, h2, rec = layer(x, W, cw, dtb, a_log, onw, state, hist, H, DK, DV)
        n += 3
        if y.shape != (MD,):
            fails.append("y_out shape")
        if s2.shape != state.shape:
            fails.append("state shape")
        if h2.shape != hist.shape:
            fails.append("hist shape")
        # the layer must MOVE both pieces of state
        n += 2
        if np.array_equal(s2, state):
            fails.append("recurrence left the state unchanged")
        if np.array_equal(h2, hist):
            fails.append("conv history did not advance")

        if not np.array_equal(layer(x, W, cw, dtb, a_log, onw, s2, hist, H, DK, DV)[0], y):
            live["no_state"] += 1
        if not np.array_equal(layer(x, W, cw, dtb, a_log, onw, state, h2, H, DK, DV)[0], y):
            live["conv_nohist"] += 1
        Wsw = dict(W); Wsw["q"], Wsw["k"] = W["k"], W["q"]
        if not np.array_equal(layer(x, Wsw, cw, dtb, a_log, onw, state, hist, H, DK, DV)[0], y):
            live["qk_swap"] += 1
        Wgo = dict(W); Wgo["gb"], Wgo["ga"] = W["fb"], W["fa"]
        if not np.array_equal(layer(x, Wgo, cw, dtb, a_log, onw, state, hist, H, DK, DV)[0], y):
            live["gate_order"] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"glm53f_kda_layer_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (both state pieces advance; trap liveness over "
          f"{trials} draws: " + ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 8
    outp = a[1] if len(a) > 1 else "build/glm53f_kda_layer_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest))
    print(f"wrote {outp}: {ntest} tests")
