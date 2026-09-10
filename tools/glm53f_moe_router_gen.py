#!/usr/bin/env python3
"""
glm53f_moe_router_gen.py -- vectors for test/glm53f_moe_router_tb.v
(src/glm53f_moe_router.v: GLM-5.3-Flash's MoE router).

Golden is tools/glm53_flash_ref.py's moe_route, which is built from the
checkpoint's own metadata: sigmoid gating (expert_gating_func = 2), top-k over
scores + exp_probs_b, renormalise (expert_weights_norm = True), scale 2.5.

SELECTION IS DISCRETE, so the corpus is filtered.  The DUT's sigmoid is
fp32_sigmoid_pipe (measured 790 ULP, ~1e-4 relative), and the reference's is
float64. Near a tie between the K-th and (K+1)-th biased score that difference
decides WHICH EXPERT RUNS -- an unbounded output change, not a small one. Draws
whose top-k margin is inside a MARGIN of 1e-3 are therefore rejected and counted,
rather than left in to make the gate flaky. That exclusion is a real limitation of
the datapath, not a convenience: at a genuine tie this router's selection is not
determined by the reference, and the ledger says so (4.3s).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_precision_study import bf16  # noqa: E402

F32 = np.float32
SCALE = 2.5
MARGIN = 1e-3


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def bf16b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16


def seq_logits(Wg, x):
    """logits[e] = sum_k Wg[k][e]*x[k], fp32, sequential -- what the RTL does."""
    E = Wg.shape[1]
    out = np.zeros(E, F32)
    for k in range(Wg.shape[0]):
        for e in range(E):
            out[e] = F32(out[e] + F32(F32(Wg[k, e]) * F32(x[k])))
    return out


def margin_of(scores, bias, topk):
    ch = (scores + bias).astype(F32)
    srt = np.sort(ch)[::-1]
    return float(srt[topk - 1] - srt[topk])


def gen(ntest, HIDDEN=16, E=8, TOPK=2, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {HIDDEN} {E} {TOPK}"]
    rejected = 0
    flipped = 0
    made = 0
    while made < ntest:
        Wg = (rng.normal(size=(HIDDEN, E)) * 0.6).astype(F32)
        bias = (rng.normal(size=E) * 0.5).astype(F32)
        x = np.array([bf16(v) for v in rng.normal(size=HIDDEN) * 1.5], F32)
        lg = seq_logits(Wg, x)
        sc = ref.sigmoid(lg).astype(F32)
        if margin_of(sc, bias, TOPK) < MARGIN:
            rejected += 1
            continue
        idx, w = ref.moe_route(lg, bias, TOPK, SCALE)
        idx0, _ = ref.moe_route(lg, np.zeros(E, F32), TOPK, SCALE)
        if not np.array_equal(idx, idx0):
            flipped += 1
        made += 1
        out.append(" ".join(f"{bf16b(v):04x}" for v in x))
        for k in range(HIDDEN):
            out.append(" ".join(f"{f32b(v):08x}" for v in Wg[k]))
        out.append(" ".join(f"{f32b(v):08x}" for v in bias))
        out.append(" ".join(f"{int(v)}" for v in idx))
        out.append(" ".join(f"{bf16b(v):04x}" for v in w))
    if report:
        print(f"router corpus: {made} kept, {rejected} rejected for a top-k margin "
              f"under {MARGIN} (selection would be undetermined there); exp_probs_b "
              f"changes the selection in {flipped}/{made}", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0x30E)
    HIDDEN, E, TOPK = 16, 8, 2
    n = 0
    fails = []
    live = {"no_bias": 0, "bias_weights": 0}
    trials = 60
    kept = 0
    for _ in range(trials):
        Wg = (rng.normal(size=(HIDDEN, E)) * 0.6).astype(F32)
        bias = (rng.normal(size=E) * 0.5).astype(F32)
        x = np.array([bf16(v) for v in rng.normal(size=HIDDEN) * 1.5], F32)
        lg = seq_logits(Wg, x)
        sc = ref.sigmoid(lg).astype(F32)
        if margin_of(sc, bias, TOPK) < MARGIN:
            continue
        kept += 1
        idx, w = ref.moe_route(lg, bias, TOPK, SCALE)
        n += 2
        if len(set(idx.tolist())) != TOPK:
            fails.append("repeated expert index")
        if abs(float(w.sum()) - SCALE) > 2e-3:
            fails.append("normalised weights do not sum to the scale")
        idx0, _ = ref.moe_route(lg, np.zeros(E, F32), TOPK, SCALE)
        if not np.array_equal(idx0, idx):
            live["no_bias"] += 1
        _, w2 = ref.moe_route(lg, bias, TOPK, SCALE, bias_selects_only=False)
        if not np.array_equal(w2, w):
            live["bias_weights"] += 1
    n += 1
    if kept < trials // 2:
        fails.append(f"only {kept}/{trials} draws survived the margin filter -- "
                     "the corpus generator would be rejecting most of its work")
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"glm53f_moe_router_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED ({kept}/{trials} draws clear the {MARGIN} top-k margin; "
          f"trap liveness: " + ", ".join(f"{k} {v}/{kept}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 16
    outp = a[1] if len(a) > 1 else "build/glm53f_moe_router_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest))
    print(f"wrote {outp}: {ntest} tests")
