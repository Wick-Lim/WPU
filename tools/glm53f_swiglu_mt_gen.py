#!/usr/bin/env python3
"""
glm53f_swiglu_mt_gen.py -- vectors for test/glm53f_swiglu_mt_tb.v.

`make swiglu-q8` runs glm53f_swiglu_mt in its Q8_0 configuration, at the decoder
block's own slice. This runs it on the type combinations the MoE experts actually
use, and it is a DIFFERENT gate for a different claim: that the runtime weight
type selects the right BUS. The engine reads Q4_K's code off `w_q`, everything
else off `w_hp`, and the headers off three further buses -- so "the type is an
input" is a plumbing claim, and the per-type arithmetic is already gated by
`make mixedtype`.

WHY K = 256 HERE AND 16 THERE.  Q4_K/Q5_K/Q6_K are 256-weight super-blocks, so a
column builder needs K % 256 == 0. The MoE loop, by contrast, wants a SMALL slice
or its vectors run to megabytes. The two claims are therefore gated at the two
slices where each is testable, rather than one compromise slice that tests neither
well.

The combinations are the checkpoint's, not invented [scan]:
    (Q4_K, Q4_K, Q5_K)  ffn_{gate,up}_exps Q4_K x42, ffn_down_exps Q5_K x40
    (Q5_K, Q5_K, Q6_K)  the one Q5_K gate/up block, and the three Q6_K down blocks
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4k_mixed_gen as mg  # noqa: E402
from swiglu_q4k_gen import bf16_round, bf16bits, bf16_mul, silu  # noqa: E402

F32 = np.float32
LIMIT = 10.0
COMBOS = [(mg.WT_Q4K, mg.WT_Q4K, mg.WT_Q5K),
          (mg.WT_Q5K, mg.WT_Q5K, mg.WT_Q6K)]


def build_pass(rng, rows, K, wt):
    """`rows` columns of type `wt`, in the engine's own layout."""
    return [mg.BUILDERS[wt](rng, K) for _ in range(rows)]


def ffn(x, G, U, D):
    """gate/up/down as dequantised column sets -> y, with the asymmetric clamp."""
    inter = len(G)
    gate = np.array([bf16_round(float(np.dot(c["wdeq"].astype(np.float64),
                                             x.astype(np.float64)))) for c in G], F32)
    up = np.array([bf16_round(float(np.dot(c["wdeq"].astype(np.float64),
                                           x.astype(np.float64)))) for c in U], F32)
    act = np.empty(inter, F32)
    n_hi = n_lo = 0
    for i in range(inter):
        g, u = float(gate[i]), float(up[i])
        if g > LIMIT:
            g = LIMIT; n_hi += 1
        if u > LIMIT:
            u = LIMIT; n_hi += 1
        elif u < -LIMIT:
            u = -LIMIT; n_lo += 1
        act[i] = bf16_mul(silu(F32(g)), F32(u))
    y = np.array([bf16_round(float(np.dot(c["wdeq"].astype(np.float64),
                                          act.astype(np.float64)))) for c in D], F32)
    return y, act, (n_hi, n_lo)


def emit_cols(out, cols, K, TN):
    """Per-column header + code streams, grouped TN at a time the way the TB serves."""
    NSB, NB8 = K // 256, K // 32
    out.append(" ".join(f"{c['d_h'][sb] if c['type'] in (mg.WT_Q4K, mg.WT_Q6K, mg.WT_Q5K) else 0:04x}"
                        for c in cols for sb in range(NSB)))
    out.append(" ".join(f"{c['dm_h'][sb] if c['type'] in (mg.WT_Q4K, mg.WT_Q5K) else 0:04x}"
                        for c in cols for sb in range(NSB)))
    out.append(" ".join(f"{c['sc96'][sb] if c['type'] in (mg.WT_Q4K, mg.WT_Q5K) else 0:024x}"
                        for c in cols for sb in range(NSB)))
    # Q6_K's 16 int8 scales go out as 16 SEPARATE bytes, scale i first -- the
    # convention tools/q4k_mixed_gen.py already uses and glm_matmul_mixed_tb.v
    # already reads, placing scale i at bit offset 8*i.  Packing them into one
    # 128-bit hex word instead puts scale 0 in the MOST significant byte, i.e.
    # at offset 8*15: the order reverses, every Q6_K column decodes with the
    # wrong per-16 scale, and the sign of the result flips.  That is exactly the
    # bug this gate caught on its first run.
    q6 = []
    for c in cols:
        for sb in range(NSB):
            sc = c["sc16"][sb] if c["type"] == mg.WT_Q6K else [0] * 16
            q6.extend(f"{s & 0xFF:02x}" for s in sc)
    out.append(" ".join(q6))
    out.append(" ".join(f"{c['d_h'][b] if c['type'] == mg.WT_Q80 else 0:04x}"
                        for c in cols for b in range(NB8)))
    out.append(" ".join(f"{v:04x}" for c in cols for v in c["code"]))


def gen(ntest, HIDDEN=256, INTER=256, TN=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {HIDDEN} {INTER} {TN}"]
    hi = lo = 0
    for t in range(ntest):
        wg, wu, wd = COMBOS[t % len(COMBOS)]
        G = build_pass(rng, INTER, HIDDEN, wg)
        U = build_pass(rng, INTER, HIDDEN, wu)
        D = build_pass(rng, HIDDEN, INTER, wd)
        x = np.array([bf16_round(v) for v in rng.normal(size=HIDDEN) * 0.9], F32)
        y, act, (a, b) = ffn(x, G, U, D)
        hi += a; lo += b
        out.append(f"{wg} {wu} {wd}")
        out.append(" ".join(f"{bf16bits(v):04x}" for v in x))
        emit_cols(out, G, HIDDEN, TN)
        emit_cols(out, U, HIDDEN, TN)
        emit_cols(out, D, INTER, TN)
        out.append(" ".join(f"{bf16bits(v):04x}" for v in y))
        # TOLERANCE = ONE bf16 ULP of the expected value, and that is MEASURED,
        # not modelled.  The obvious bound here -- the one swiglu_q4k_gen uses and
        # this generator inherited -- is max(0.06*|y|, 0.02*sum|act*w|, 0.03).  The
        # sum-of-absolute-products term dominates because the dot product cancels,
        # which made the tolerance ~33 % of |y|: a check that could not fail for
        # any reason short of the sign flipping.
        #   Swept seeds 0..7 (2 tokens x 256 outputs each = 4096 outputs) against
        # the RTL: worst error 0.000 ULP, i.e. BIT-EXACT everywhere.  The reason is
        # that `silu` here uses a true np.exp while glm_act uses a polynomial, and
        # bf16's 8-bit mantissa swallows the difference.  "Swallows it on 4096
        # samples" is not "cannot ever differ", so this is a 1-ULP tolerance rather
        # than a bitwise check: one rounding boundary of headroom, nothing more.
        tol = []
        for o in range(HIDDEN):
            v = abs(float(y[o]))
            tol.append(np.ldexp(1.0, int(np.floor(np.log2(v))) - 7) if v > 0 else 1e-30)
        out.append(" ".join(f"{v:.9g}" for v in tol))
    if report:
        print(f"swiglu-mt: {ntest} tests over {len(COMBOS)} checkpoint type combos; "
              f"clamp exercised {hi} upper / {lo} lower", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0x77)
    HIDDEN = INTER = 256
    n = 0
    fails = []
    hi = lo = 0
    for wg, wu, wd in COMBOS:
        # INTER is forced to 256 by the DOWN pass's K (super-blocks are 256
        # wide), so gate/up must have 256 columns; only the number of DOWN rows
        # is free, and 4 is enough to self-test.
        G = build_pass(rng, INTER, HIDDEN, wg)
        U = build_pass(rng, INTER, HIDDEN, wu)
        D = build_pass(rng, 4, INTER, wd)
        x = np.array([bf16_round(v) for v in rng.normal(size=HIDDEN) * 0.9], F32)
        y, act, (a, b) = ffn(x, G, U, D)
        hi += a; lo += b
        n += 3
        if y.shape != (4,):
            fails.append("y shape")
        # each builder must produce the type it was asked for, and a code stream
        # in that type's range -- a silent fallback to Q4_K is the failure mode
        for nm, cols, wt in (("gate", G, wg), ("up", U, wu), ("down", D, wd)):
            if any(c["type"] != wt for c in cols):
                fails.append(f"{nm}: builder returned the wrong type")
        rng_max = {mg.WT_Q4K: 16, mg.WT_Q5K: 32, mg.WT_Q6K: 64}
        for nm, cols, wt in (("gate", G, wg), ("up", U, wu), ("down", D, wd)):
            if max(max(c["code"]) for c in cols) >= rng_max[wt]:
                fails.append(f"{nm}: code out of range for {wt}")
    n += 2
    if hi == 0 or lo == 0:
        fails.append("the corpus does not drive both clamp bounds")
    if fails:
        print(f"glm53f_swiglu_mt_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (both checkpoint type combos build in their own "
          f"code range; clamp driven {hi} upper / {lo} lower)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 2
    outp = a[1] if len(a) > 1 else "build/glm53f_swiglu_mt_vec.txt"
    seed = int(a[2]) if len(a) > 2 else 0     # the tolerance sweep varies this
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, seed=seed))
    print(f"wrote {outp}: {ntest} tests")
