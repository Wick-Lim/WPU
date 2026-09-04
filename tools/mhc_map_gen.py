#!/usr/bin/env python3
"""
mhc_map_gen.py -- vectors for test/mhc_map_step_tb.v (src/mhc_map_step.v).

The mHC map: 24 mixed logits -> pre[4], post[4], comb[4x4] (Sinkhorn included).

THE GOLDEN IS THE REFERENCE'S OWN OUTPUT.  `pre` is not returned by
ref.hyper_connection, so this generator recomputes it from the documented formula
and then CROSS-CHECKS it against something the reference does return:
    sum_h pre[h] * hidden_streams[h]  ==  collapsed
That is the self-test's job, and it is what makes `pre` a reference number here
rather than a restatement of my own reading.

WHERE THE TOLERANCE COMES FROM.  The reference evaluates sigmoid and softmax in
float64 and casts to fp32; the DUT is fp32 over fp32_exp_pipe, whose polynomial is
measured at 1899 ULP / 2.3e-4 (`make fp-exp-acc`), with fp32_sigmoid_pipe measured
at 790 ULP on top of it.  So instead of running the DUT and calling whatever it
prints the tolerance, this predicts an ENVELOPE: perturb every exp by +/-2.3e-4
and every sigmoid by +/-790 ULP in the worst-case direction, push that through the
softmax renormalise and all 39 Sinkhorn passes (with x*recip(y), no divider), and
report the resulting spread.  The TB bound is set from the envelope, so the test
constrains the implementation rather than describing it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_sinkhorn_gen import newton_recip, seq_sum, sinkhorn  # noqa: E402

F32 = np.float32
EPS = F32(1e-6)
EXP_REL = 2.3e-4        # fp32_exp_pipe, measured
SIG_ULP = 790           # fp32_sigmoid_pipe, measured


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def nudge_ulp(x, n):
    """x moved n ULP away from zero (n may be negative).

    NOT applied at the sigmoid's saturation rails.  fp32_sigmoid_pipe returns
    EXACTLY 1.0 and EXACTLY 0.0 there -- that is a gated property, not a hope
    (`make fp-sigmoid` pins 130 bitwise-1.0 and 34 bitwise-0.0 points, and it is
    the reason the unit exists: bf16 cannot represent 1 + 1e-6).  Perturbing those
    points would model an error the DUT provably does not make, and it does not
    merely inflate the bound -- nudging 0.0 downward wraps the sign bit and
    produces a meaningless 9e8 ULP 'envelope', which is how this surfaced.
    """
    x = np.asarray(x, F32)
    b = np.frombuffer(np.ascontiguousarray(x).tobytes(), np.int32).astype(np.int64)
    out = np.frombuffer(((b + n) & 0xFFFFFFFF).astype(np.uint32).tobytes(),
                        F32).reshape(x.shape)
    railed = (x == F32(0.0)) | (x == F32(1.0))
    return np.where(railed, x, out).astype(F32)


def ref_map(mixed, base, scale, H=4, iters=20):
    """The reference's map, float64 transcendentals cast to fp32, as it does."""
    mixed = np.asarray(mixed, F32)
    base = np.asarray(base, F32)
    s0, s1, s2 = [F32(v) for v in np.asarray(scale, F32)]
    pre = (ref.sigmoid(mixed[:H] * s0 + base[:H]) + EPS).astype(F32)
    post = (F32(2.0) * ref.sigmoid(mixed[H:2*H] * s1 + base[H:2*H])).astype(F32)
    cw = (mixed[2*H:].reshape(H, H) * s2 + base[2*H:].reshape(H, H)).astype(F32)
    c0 = (ref.softmax(cw, -1) + EPS).astype(F32)
    return pre, post, sinkhorn(c0, iters)


def envelope(mixed, base, scale, H=4, iters=20):
    """Worst-case fp32-datapath deviation from ref_map, predicted not measured."""
    mixed = np.asarray(mixed, F32); base = np.asarray(base, F32)
    s0, s1, s2 = [F32(v) for v in np.asarray(scale, F32)]
    g_pre, g_post, g_comb = ref_map(mixed, base, scale, H, iters)
    worst = {"pre": 0, "post": 0, "comb": 0}
    cw = (mixed[2*H:].reshape(H, H) * s2 + base[2*H:].reshape(H, H)).astype(F32)
    for sgn in (+1, -1):
        p = nudge_ulp(ref.sigmoid(mixed[:H] * s0 + base[:H]), sgn * SIG_ULP)
        worst["pre"] = max(worst["pre"], _ulp((p + EPS).astype(F32), g_pre))
        q = nudge_ulp(ref.sigmoid(mixed[H:2*H] * s1 + base[H:2*H]), sgn * SIG_ULP)
        worst["post"] = max(worst["post"], _ulp((F32(2.0) * q).astype(F32), g_post))
        # exp perturbed the worst way, then renormalised sequentially and Sinkhorned
        e = np.exp((cw - cw.max(-1, keepdims=True)).astype(np.float64)).astype(F32)
        # opposite signs on the diagonal vs the rest maximise the ratio's error
        pert = np.full((H, H), 1.0 - sgn * EXP_REL, F32)
        np.fill_diagonal(pert, 1.0 + sgn * EXP_REL)
        ep = (e * pert).astype(F32)
        c = (ep * newton_recip(np.expand_dims(seq_sum(ep, -1), -1))).astype(F32)
        c = (c + EPS).astype(F32)
        worst["comb"] = max(worst["comb"], _ulp(sinkhorn(c, iters, use_recip=True), g_comb))
    return worst


def _ulp(a, b):
    ab = np.frombuffer(np.ascontiguousarray(np.asarray(a, F32)).tobytes(), np.int32).astype(np.int64)
    bb = np.frombuffer(np.ascontiguousarray(np.asarray(b, F32)).tobytes(), np.int32).astype(np.int64)
    return int(np.abs(ab - bb).max())


def _draw(rng, H, D, wide=False):
    """A block's mHC parameters and streams, at the logit spreads 4.3i swept.

    `wide` is the SATURATION case, and it is in the corpus deliberately.  The
    softmax's max subtraction is invariant in exact arithmetic and is only a
    rounding difference at ordinary logit spreads -- so a corpus of ordinary
    spreads cannot tell a correct softmax from one that skips the shift, and
    INJ_MAP_SOFTMAX_NOMAX measurably does NOT fire on one (checked).  Past
    |logit| ~ 88, fp32_exp_pipe overflows to +inf and flushes to zero, and the
    shift stops being cosmetic.  That is the case the trap is about, so it has to
    be represented; it also drives the sigmoids into their exact-0/exact-1 rails.
    """
    sp = rng.choice([0.35, 1.42, 5.68, 11.36])
    fn = (rng.normal(size=((2 + H) * H, H * D)) * (sp / np.sqrt(H * D))).astype(F32)
    base = (rng.normal(size=(2 + H) * H) * (120.0 if wide else sp)).astype(F32)
    scale = (F32(1.0) + rng.normal(size=3).astype(F32) * F32(0.3)).astype(F32)
    hs = (rng.normal(size=(H, D)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
    flat = hs.reshape(-1).astype(F32)
    flat = (flat / np.sqrt((flat.astype(np.float64) ** 2).mean() + 1e-5)).astype(F32)
    mixed = (fn @ flat).astype(F32)
    return hs, fn, base, scale, mixed


def gen(ntest, H=4, iters=20, D=64, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {iters}"]
    env = {"pre": 0, "post": 0, "comb": 0}
    for ti in range(ntest):
        _, _, base, scale, mixed = _draw(rng, H, D, wide=(ti % 8 == 7))
        pre, post, comb = ref_map(mixed, base, scale, H, iters)
        e = envelope(mixed, base, scale, H, iters)
        for k in env:
            env[k] = max(env[k], e[k])
        out.append(" ".join(f"{f32b(v):08x}" for v in mixed))
        out.append(" ".join(f"{f32b(v):08x}" for v in base))
        out.append(" ".join(f"{f32b(v):08x}" for v in scale))
        out.append(" ".join(f"{f32b(v):08x}" for v in pre))
        out.append(" ".join(f"{f32b(v):08x}" for v in post))
        out.append(" ".join(f"{f32b(v):08x}" for v in comb.reshape(-1)))
    if report:
        print(f"predicted fp32-datapath envelope vs the float64 reference over {ntest} tests: "
              f"pre {env['pre']} ULP, post {env['post']} ULP, comb {env['comb']} ULP",
              file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0x11C)
    H, D = 4, 64
    n = 0
    fails = []
    live = {k: 0 for k in ("post_no2", "pre_noeps", "comb_noeps", "softmax_nomax")}
    trials = 120
    for ti in range(trials):
        hs, fn, base, scale, mixed = _draw(rng, H, D, wide=(ti % 8 == 7))
        pre, post, comb = ref_map(mixed, base, scale, H)

        # `pre` is validated against something the reference RETURNS:
        #     collapsed = sum_h pre[h] * hidden_streams[h]
        r_post, r_comb, r_coll = ref.hyper_connection(hs, fn, base, scale, H, 20, 1e-6, 1e-5)
        n += 1
        if not np.allclose((pre[:, None] * hs).sum(0), r_coll, rtol=1e-5, atol=1e-6):
            fails.append("pre disagrees with the reference's own `collapsed`")
        n += 1
        if not np.array_equal(post, r_post):
            fails.append("post is not bitwise the reference's post")
        n += 1
        if not np.array_equal(comb, r_comb):
            fails.append("comb is not bitwise the reference's comb")

        cw = (mixed[2*H:].reshape(H, H) * scale[2] + base[2*H:].reshape(H, H)).astype(F32)
        s0, s1 = F32(scale[0]), F32(scale[1])
        variants = {
            "post_no2":      ref.sigmoid(mixed[H:2*H] * s1 + base[H:2*H]).astype(F32),
            "pre_noeps":     ref.sigmoid(mixed[:H] * s0 + base[:H]).astype(F32),
            "comb_noeps":    sinkhorn(ref.softmax(cw, -1).astype(F32), 20),
            "softmax_nomax": None,
        }
        # No max subtraction: exp of the raw logits, fp32, then renormalise.  On a
        # `wide` draw this OVERFLOWS to +inf and the renormalise yields nan -- that
        # is not an accident in the test, it is the defect the shift prevents, so
        # the overflow is expected here and only here.
        with np.errstate(over="ignore", invalid="ignore"):
            e_raw = np.exp(cw.astype(np.float64)).astype(F32)
            variants["softmax_nomax"] = sinkhorn(
                ((e_raw * newton_recip(np.expand_dims(seq_sum(e_raw, -1), -1))).astype(F32)
                 + EPS).astype(F32), 20)
        golden = {"post_no2": post, "pre_noeps": pre,
                  "comb_noeps": comb, "softmax_nomax": comb}
        for k, v in variants.items():
            # array_equal is False for nan vs a number, which is the right verdict
            # here: a nan comb is a broken comb.
            if not np.array_equal(v, golden[k]):
                live[k] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"mhc_map_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (pre matches the reference's own collapsed; post/comb bitwise; "
          "trap liveness over " + str(trials) + " draws: " +
          ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 32
    H = int(a[1]) if len(a) > 1 else 4
    iters = int(a[2]) if len(a) > 2 else 20
    outp = a[3] if len(a) > 3 else "build/mhc_map_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, H, iters))
    print(f"wrote {outp}: {ntest} tests H={H} iters={iters}")
