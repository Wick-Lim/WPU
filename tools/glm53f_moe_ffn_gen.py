#!/usr/bin/env python3
"""
glm53f_moe_ffn_gen.py -- vectors for test/glm53f_moe_ffn_tb.v
(src/glm53f_moe_ffn.v: GLM-5.3-Flash's MoE FFN -- route, run the chosen experts,
add the always-on shared expert).

The golden is tools/glm53_flash_ref.py's moe_ffn, called with
expert_out_bf16=True because the RTL reuses glm53f_swiglu_mt, whose y_out port is
bf16 -- the hardware really does round once per expert before the weighted sum.

WHAT THIS GATE CLAIMS, AND WHAT IT DOES NOT.  The claim here is the LOOP and the
COMBINE: that exactly the selected experts run, in ascending index order, each
scaled by its own router weight, with the shared expert added at weight 1 and the
accumulation carried in fp32. It deliberately does NOT re-claim the per-type
arithmetic -- Q4_K/Q5_K/Q6_K need K=256 super-blocks, and `make swiglu-mt` gates
them at that width. Running this loop at K=256 would make the corpus 64x larger
to re-test something already gated.

So the slice runs the routed experts as Q8_0 and the shared expert as F16. Those
are not the checkpoint's types, and the header says so -- but the SPLIT is
faithful: [scan] says the routed experts are Q4_K/Q5_K/Q6_K while the shared one
is Q8_0, so `wt_sh_*` is a separate input triple from `wt_*`. Driving two
DIFFERENT types across that boundary is what proves the two triples are plumbed
independently, which is the property the checkpoint actually depends on.

TOLERANCE. One bf16 ULP of the expected value, measured (see the sweep recorded
in docs/GLM53_FLASH_PORT.md). The slack that needs covering is glm_fp.vh's
fp32_add, which is 1 ULP low on ~0.04% of pairs (`make fp-ieee`); the fp32
accumulator here is 9 adds deep per output, and the bf16 output rounding masks
nearly all of it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref          # noqa: E402
import q4k_mixed_gen as mg             # noqa: E402
from glm53f_moe_router_gen import seq_logits, margin_of, f32b, bf16b  # noqa: E402
from glm53f_swiglu_mt_gen import emit_cols, build_pass, ffn           # noqa: E402

F32 = np.float32
SCALE = 2.5
MARGIN = 1e-3
WT_ROUTED = mg.WT_Q80       # see the header: the split is faithful, the types are not
WT_SHARED = mg.WT_F16


def make_expert(rng, HIDDEN, INTER, wt):
    return (build_pass(rng, INTER,  HIDDEN, wt),     # gate  [INTER x HIDDEN]
            build_pass(rng, INTER,  HIDDEN, wt),     # up    [INTER x HIDDEN]
            build_pass(rng, HIDDEN, INTER,  wt))     # down  [HIDDEN x INTER]


def dense(cols):
    """column set -> dense [rows, K] of the DEQUANTISED weights."""
    return np.array([c["wdeq"] for c in cols], F32)


def compose(x, experts, shared, idx, w):
    """acc = SUM_i bf16(w_i) * expert_i(x) + shared(x), fp32, ascending index.

    Each expert's own output is the bf16 that glm53f_swiglu_mt emits, and the
    weight is the bf16 the router emits -- both ports, not modelling choices.
    """
    HIDDEN = len(shared[2])
    acc = np.zeros(HIDDEN, F32)
    for j, ei in enumerate(idx):                      # ascending
        ye, _, _ = ffn(x, *experts[int(ei)])          # bf16 per element
        wb = ref._bf16(F32(w[j]))                     # the router's bf16 port
        for h in range(HIDDEN):
            acc[h] = F32(acc[h] + F32(wb * ye[h]))
    ys, _, _ = ffn(x, *shared)
    for h in range(HIDDEN):
        acc[h] = F32(acc[h] + ys[h])                  # weight 1, last
    return np.array([ref._bf16(v) for v in acc], F32)


def gen(ntest, HIDDEN=32, INTER=64, E=8, TOPK=3, TN=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {HIDDEN} {INTER} {E} {TOPK} {TN}"]
    made = rejected = flipped = 0
    while made < ntest:
        Wg   = (rng.normal(size=(HIDDEN, E)) * 0.6).astype(F32)
        bias = (rng.normal(size=E) * 0.5).astype(F32)
        x    = np.array([ref._bf16(v) for v in rng.normal(size=HIDDEN) * 1.2], F32)
        lg   = seq_logits(Wg, x)
        sc   = ref.sigmoid(lg).astype(F32)
        if margin_of(sc, bias, TOPK) < MARGIN:
            rejected += 1
            continue
        idx, w = ref.moe_route(lg, bias, TOPK, SCALE)
        idx0, _ = ref.moe_route(lg, np.zeros(E, F32), TOPK, SCALE)
        if not np.array_equal(idx, idx0):
            flipped += 1

        experts = [make_expert(rng, HIDDEN, INTER, WT_ROUTED) for _ in range(E)]
        shared  = make_expert(rng, HIDDEN, INTER, WT_SHARED)
        y = compose(x, experts, shared, idx, w)
        made += 1

        out.append(" ".join(f"{bf16b(v):04x}" for v in x))
        for k in range(HIDDEN):
            out.append(" ".join(f"{f32b(v):08x}" for v in Wg[k]))
        out.append(" ".join(f"{f32b(v):08x}" for v in bias))
        for (G, U, D) in experts:
            emit_cols(out, G, HIDDEN, TN)
            emit_cols(out, U, HIDDEN, TN)
            emit_cols(out, D, INTER,  TN)
        emit_cols(out, shared[0], HIDDEN, TN)
        emit_cols(out, shared[1], HIDDEN, TN)
        emit_cols(out, shared[2], INTER,  TN)
        out.append(" ".join(f"{int(v)}" for v in idx))
        out.append(" ".join(f"{bf16b(v):04x}" for v in w))
        out.append(" ".join(f"{bf16b(v):04x}" for v in y))
        tol = []
        for v in y:
            a = abs(float(v))
            tol.append(np.ldexp(1.0, int(np.floor(np.log2(a))) - 7) if a > 0 else 1e-30)
        out.append(" ".join(f"{v:.9g}" for v in tol))
        # SENTINEL, and it must be the LAST line of the test.  emit_cols sizes its
        # header lines from the PER-PASS K (NSB = K//256, NB8 = K//32), which
        # differs from the TB's compile-time KMAX strides on every pass where
        # K != KMAX.  Get that wrong and the TB reads garbage -- and the first
        # version of this gate reported all 96 output checks PASSING while
        # completely misaligned, because a $fscanf that matches nothing leaves its
        # target untouched and every stale compare happened to hold.  This token is
        # read back and verified per test, so a misalignment of even one token is a
        # hard failure instead of a green run.
        out.append(f"a5a5a5a5 {made - 1}")
    if report:
        print(f"moe-ffn: {made} tests kept, {rejected} rejected for a top-k margin "
              f"under {MARGIN}; exp_probs_b changes the selection in {flipped}/{made}; "
              f"routed experts {WT_ROUTED} / shared {WT_SHARED} (see header)",
              file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    n = 0
    fails = []

    def chk(c, m):
        nonlocal n
        n += 1
        if not c:
            fails.append(m)

    rng = np.random.default_rng(0xA11)
    # K MUST be a multiple of 32 on both passes: q4k_mixed_gen's Q8_0 builder
    # lays out K//32 blocks, so HIDDEN=8 silently yields ZERO blocks and an
    # empty column.  gate/up run K=HIDDEN and down runs K=INTER, so both.
    H, I, E, K = 32, 64, 6, 3
    Wg   = (rng.normal(size=(H, E)) * 0.6).astype(F32)
    bias = (rng.normal(size=E) * 0.5).astype(F32)
    x    = np.array([ref._bf16(v) for v in rng.normal(size=H) * 1.2], F32)
    lg   = seq_logits(Wg, x)
    idx, w = ref.moe_route(lg, bias, K, SCALE)
    experts = [make_expert(rng, H, I, WT_ROUTED) for _ in range(E)]
    shared  = make_expert(rng, H, I, WT_SHARED)

    chk(list(idx) == sorted(idx), "indices are not ascending -- the accumulate order is unpinned")
    chk(len(set(int(v) for v in idx)) == K, "top-k not distinct")

    y = compose(x, experts, shared, idx, w)
    chk(y.shape == (H,), "output shape wrong")
    chk(np.isfinite(y).all(), "non-finite output")

    # the two type triples really are different, or the gate proves nothing
    chk(WT_ROUTED != WT_SHARED,
        "routed and shared use the SAME type -- the separate wt_sh_* triple is untested")

    # TRIPWIRE, not a pin.  The accumulation order is fixed to ascending index for
    # determinism, but it is NOT gated, because at the bf16 output it is not
    # observable: ascending and score-descending were BITWISE IDENTICAL on 16/16
    # draws over E=8/16/32, TOPK=3/8, INTER=64/128 (worst relative difference
    # exactly 0 -- the final fp32->bf16 rounding swallows the reordering).  An
    # order injection would pass and prove nothing, so there is none.
    #   This check asserts the two orders still AGREE. If that ever stops being
    # true -- a wider slice, an fp32 output port, a deeper top-k -- this fires and
    # the decision above should be revisited, because then the order WOULD be
    # gateable and should be gated.
    order = sorted(range(E), key=lambda i: -float(ref.sigmoid(lg)[i] + bias[i]))
    desc  = [i for i in order[:K]]
    wmap  = {int(i): w[j] for j, i in enumerate(idx)}
    y_desc = compose(x, experts, shared, desc, [wmap[int(i)] for i in desc])
    chk(np.array_equal(y, y_desc),
        "ascending and score-descending accumulation now DIFFER -- the order has "
        "become observable, so it should get a must-fail leg (see the module header)")

    # the shared expert is added at weight 1: drop it and the answer must move by
    # exactly the shared expert's own output
    ys, _, _ = ffn(x, *shared)
    acc_routed = np.zeros(H, F32)
    for j, ei in enumerate(idx):
        ye, _, _ = ffn(x, *experts[int(ei)])
        wb = ref._bf16(F32(w[j]))
        for h in range(H):
            acc_routed[h] = F32(acc_routed[h] + F32(wb * ye[h]))
    recomposed = np.array([ref._bf16(F32(acc_routed[h] + ys[h])) for h in range(H)], F32)
    chk(np.array_equal(y, recomposed), "shared expert is not the final weight-1 add")

    # an UNSELECTED expert must not matter
    unsel = [e for e in range(E) if e not in set(int(v) for v in idx)]
    chk(len(unsel) > 0, "degenerate: every expert selected")
    e2 = list(experts)
    e2[unsel[0]] = make_expert(np.random.default_rng(0xBEEF), H, I, WT_ROUTED)
    chk(np.array_equal(y, compose(x, e2, shared, idx, w)),
        "replacing an UNSELECTED expert changed the answer")

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_moe_ffn_gen self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (ascending accumulate order, distinct top-k, "
          f"order not observable at bf16 so it is a tripwire not a pin, the "
          f"shared expert is the final weight-1 add, unselected experts inert, "
          f"and the routed/shared type triples differ)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 3
    outp = a[1] if len(a) > 1 else "build/glm53f_moe_ffn_vec.txt"
    seed = int(a[2]) if len(a) > 2 else 0
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, seed=seed))
    print(f"wrote {outp}: {ntest} tests")
