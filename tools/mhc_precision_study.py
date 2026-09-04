#!/usr/bin/env python3
"""
mhc_precision_study.py -- what precision does the mHC hyper-connection need?

docs/GLM53_FLASH_PORT.md 4.2 item 3 says mHC "needs a numerically-careful
fixed-point study before RTL".  This is that study, on the transcribed reference
(tools/glm53_flash_ref.py hyper_connection), with NO RTL and random weights --
so it is decision evidence about precision budgets, not a claim about the model.

mHC runs on EVERY one of the 45 blocks, twice (attention site and FFN site):
  pre  = sigmoid(pre_w*s0 + b) + eps            collapses 4 streams -> 1 sublayer input
  post = 2*sigmoid(post_w*s1 + b)               places the sublayer output, range [0,2]
  comb = Sinkhorn_20( softmax(comb_w*s2 + b) )  4x4, re-mixes the 4 streams
  streams' = comb @ streams + post (x) sublayer_out      (per the paper's residual form)

Questions answered, each with a number:
  Q1  Sinkhorn: after the reference's 1 col pass + 19 (row,col) pairs, how far is
      `comb` from doubly stochastic (max |row sum - 1|, |col sum - 1|)?  Answered as a
      POPULATION, median/p90/worst, and swept over the comb-logit spread -- because
      the first version of this study answered it from one draw and mistook that
      draw's residual for a bound.
  Q2  bf16 comb: if the mHC map is evaluated in bf16 (as glm_act-style units would),
      how far does `comb` move from the fp32 comb (max abs element error)?
  Q3  compounding: apply the block recurrence streams' = comb @ streams + post*u for
      45 blocks with fp32 comb vs bf16 comb (same random sublayer outputs u), and
      report the stream divergence (RMS rel) at blocks 1, 11, 23, 45.
  Q4  the dynamic ranges the RTL must carry: pre in (eps, 1+eps), post in [0,2],
      comb in (0,1) -- and the smallest comb entry seen (an fp exponent question).
  Q5  Sinkhorn does 40 divisions.  The RTL has no divider -- src/glm_fp_recip.vh is a
      Newton reciprocal, so every normalise becomes x * recip(sum+eps).  Does that
      substitution, compounded over 20 iterations, move `comb`?
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32
HC, D, EPS, RMS_EPS = 4, 4096, 1e-6, 1e-5
REPS = 200          # Q1 is a population statistic; a single draw hides the tail
MIX = (2 + HC) * HC


def bf16(x):
    """fp32 -> bf16-valued fp32 (RNE)."""
    u = np.frombuffer(np.asarray(x, F32).tobytes(), np.uint32).copy()
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return np.frombuffer(u.tobytes(), F32).reshape(np.shape(x))


def sinkhorn_residual(c):
    return max(float(np.abs(c.sum(-1) - 1).max()), float(np.abs(c.sum(-2) - 1).max()))


def main():
    rng = np.random.default_rng(0)
    # a block's mHC parameters: fn is [mix, HC*D]; scale it like a trained small init
    fn = (rng.normal(size=(MIX, HC * D)) * (1.0 / np.sqrt(HC * D))).astype(F32)
    base = rng.normal(size=MIX).astype(F32)
    scale = np.array([1.0, 1.0, 1.0], F32)

    print("mHC precision study  (hc_mult=4, D=4096, sinkhorn_iters=20, eps=1e-6; random weights)\n")

    # ---- Q1: how close to doubly stochastic is comb, over a POPULATION ----
    # CORRECTED 2026-09-04.  The first version of this study measured ONE random
    # draw (seed 0) and reported its residual, 1.013e-06, as the fp32 answer --
    # concluding "the reference's 20 iterations are past the plateau, correct and
    # sufficient".  1.013e-06 is right for that draw and is the MEDIAN of the
    # population, but it is not a bound: the distribution has a heavy tail, and a
    # single sample cannot see it.  Measured over draws and over the logit scale
    # below, and the conclusion changes -- see the note printed at the end of Q1.
    print("Q1  Sinkhorn residual (max |row/col sum - 1|), fp32, over "
          f"{REPS} independent draws of (fn, base, hidden_streams):")
    print(f"      {'iters':>5} {'median':>10} {'p90':>10} {'worst':>10}")
    q1 = {}
    for it in (1, 2, 5, 10, 20, 40, 100):
        v = np.empty(REPS, np.float64)
        for k in range(REPS):
            r = np.random.default_rng(1000 + k)          # same draws for every `it`
            f = (r.normal(size=(MIX, HC * D)) * (1.0 / np.sqrt(HC * D))).astype(F32)
            b = r.normal(size=MIX).astype(F32)
            h = r.normal(size=(HC, D)).astype(F32)
            _, comb, _ = ref.hyper_connection(h, f, b, scale, HC, it, EPS, RMS_EPS)
            v[k] = sinkhorn_residual(comb)
        q1[it] = v
        print(f"      {it:>5} {np.median(v):>10.2e} {np.percentile(v, 90):>10.2e} {v.max():>10.2e}")

    # the residual at a FIXED 20 iterations is input-conditioned: how hard the
    # matrix is depends on the spread of the comb logits (fn @ flat) * s2 + comb_b.
    # The construction above gives logit std ~1.42; sweep it to bracket the model,
    # whose trained fn/base spread is not published.
    print(f"\n      residual @20 iters vs the SPREAD of the comb logits (worst of {REPS}):")
    print(f"      {'logit std':>10} {'@20':>10} {'@40':>10} {'@100':>10}")
    for sc in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        w = {20: 0.0, 40: 0.0, 100: 0.0}
        for k in range(REPS):
            r = np.random.default_rng(2000 + k)
            f = (r.normal(size=(MIX, HC * D)) * (sc / np.sqrt(HC * D))).astype(F32)
            b = (r.normal(size=MIX) * sc).astype(F32)
            h = r.normal(size=(HC, D)).astype(F32)
            for it in w:
                _, comb, _ = ref.hyper_connection(h, f, b, scale, HC, it, EPS, RMS_EPS)
                w[it] = max(w[it], sinkhorn_residual(comb))
        print(f"      {sc * 1.42:>10.2f} {w[20]:>10.2e} {w[40]:>10.2e} {w[100]:>10.2e}")

    print("      -> the eps=1e-6 floor is reached by the MEDIAN draw at 20 iterations, but")
    print("         the tail is not at the floor, and it grows with the logit spread.  So 20")
    print("         is NOT a convergence criterion that happens to be satisfied -- it is a")
    print("         PUBLISHED CONSTANT (hc_sinkhorn_iters=20) and the matrix the model uses is")
    print("         whatever 20 iterations produce, doubly stochastic or not.  Consequence for")
    print("         RTL: run exactly 20, fixed schedule, no convergence-based early exit --")
    print("         an early exit would be faster AND less faithful.\n")

    # ---- Q2: bf16 evaluation of the map ----
    worst = 0.0; worst_res_bf = 0.0; worst_res_32 = 0.0
    for _ in range(64):
        hs = rng.normal(size=(HC, D)).astype(F32)
        _, comb32, _ = ref.hyper_connection(hs, fn, base, scale, HC, 20, EPS, RMS_EPS)
        _, combbf, _ = ref.hyper_connection(bf16(hs), bf16(fn), bf16(base), scale, HC, 20, EPS, RMS_EPS)
        combbf = bf16(combbf)
        worst = max(worst, float(np.abs(combbf - comb32).max()))
        worst_res_bf = max(worst_res_bf, sinkhorn_residual(combbf))
        worst_res_32 = max(worst_res_32, sinkhorn_residual(comb32))
    print(f"Q2  comb evaluated with bf16 inputs/weights and bf16-rounded output vs fp32:")
    print(f"      max |comb_bf16 - comb_fp32| = {worst:.3e}   (comb entries are in (0,1); typical ~0.25)")
    print(f"      Sinkhorn residual of the bf16 comb = {worst_res_bf:.3e}  (fp32, SAME corpus: {worst_res_32:.3e})")
    print(f"      -> bf16 costs {worst_res_bf / max(worst_res_32, 1e-30):.0f}x the fp32 residual on the same inputs.  NOTE: an\n"
          f"         earlier version of this study compared this {worst_res_bf:.1e} against fp32's MEDIAN\n"
          f"         1.0e-6 and reported a ~3000x gap; paired, the gap is {worst_res_bf / max(worst_res_32, 1e-30):.0f}x.  bf16 is still the\n"
          f"         worse choice, and Q3 prices it, but the honest ratio is this one.\n")

    # ---- Q3: compounding across 45 blocks ----
    def run(prec):
        streams = rng.normal(size=(HC, D)).astype(F32)
        s32 = streams.copy(); sbf = streams.copy()
        out = {}
        for blk in range(1, 46):
            u = rng.normal(size=D).astype(F32) * F32(0.5)          # a sublayer output
            post32, comb32, _ = ref.hyper_connection(s32, fn, base, scale, HC, 20, EPS, RMS_EPS)
            postbf, combbf, _ = ref.hyper_connection(bf16(sbf), bf16(fn), bf16(base), scale, HC, 20, EPS, RMS_EPS)
            if prec == "bf16comb":
                combbf = bf16(combbf); postbf = bf16(postbf)
            s32 = (comb32 @ s32 + post32[:, None] * u[None, :]).astype(F32)
            sbf = (combbf @ sbf + postbf[:, None] * u[None, :]).astype(F32)
            if blk in (1, 11, 23, 45):
                out[blk] = float(np.sqrt(np.mean((sbf - s32) ** 2)) / (np.sqrt(np.mean(s32 ** 2)) + 1e-30))
        return out
    r = run("bf16comb")
    print("Q3  45-block residual-stream divergence, bf16 mHC map vs fp32 (RMS rel):")
    for b in (1, 11, 23, 45):
        print(f"      block {b:>2}: {r[b]:.3e}")
    print("      -> the mix is contractive (comb is stochastic), so this settles rather than explodes;")
    print("         the number above is the price of a bf16 mHC map at the END of the stack.\n")

    # ---- Q4: dynamic ranges ----
    mn_comb, mx_post, mn_pre, mx_pre = 1.0, 0.0, 1.0, 0.0
    for _ in range(64):
        hs = (rng.normal(size=(HC, D)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
        post, comb, _ = ref.hyper_connection(hs, fn, base, scale * F32(4.0), HC, 20, EPS, RMS_EPS)
        pre = ref.sigmoid((fn[:HC] @ (hs.reshape(-1) / np.sqrt(np.mean(hs.astype(np.float64)**2) + RMS_EPS)).astype(F32)) * F32(4.0) + base[:HC]) + F32(EPS)
        mn_comb = min(mn_comb, float(comb.min())); mx_post = max(mx_post, float(post.max()))
        mn_pre = min(mn_pre, float(pre.min())); mx_pre = max(mx_pre, float(pre.max()))
    print("Q4  dynamic ranges the RTL must carry (scale x4 to stress saturation):")
    print(f"      pre  in ({mn_pre:.3e}, {mx_pre:.6f})   post in [0, {mx_post:.6f}]   smallest comb entry {mn_comb:.3e}")
    print(f"      -> comb needs exponent range down to ~1e-6 (eps floor): fine in fp32/bf16 exponent, NOT in a naive fixed-point mantissa")

    # ---- Q5: the RTL has no divider -- x * recip(sum+eps) instead ----
    # Sinkhorn's 20 iterations are 40 normalises, i.e. 40 divisions, and this repo
    # has no fp32 divide: src/glm_fp_recip.vh is a 4-iteration Newton reciprocal,
    # measured to within 1 ULP.  Substituting x*recip(y) for x/y inside a 20-step
    # fixed-point-like iteration is the kind of thing that can drift, so measure it
    # rather than assume.  Emulated here at fp32 with a 1-ULP-perturbed reciprocal,
    # the worst case the RTL primitive is gated to.
    def recip_1ulp(y):
        """1/y then nudged 1 ULP the worst way -- an upper bound on glm_fp_recip."""
        r = (F32(1.0) / np.asarray(y, F32)).astype(F32)
        u = np.frombuffer(np.ascontiguousarray(r).tobytes(), np.uint32).copy()
        u += 1                                        # +1 ULP everywhere: worst-case bias
        return np.frombuffer(u.tobytes(), F32).reshape(np.shape(r))

    def sinkhorn_recip(c, iters):
        c = (c * recip_1ulp(c.sum(-2, keepdims=True) + F32(EPS))).astype(F32)
        for _ in range(iters - 1):
            c = (c * recip_1ulp(c.sum(-1, keepdims=True) + F32(EPS))).astype(F32)
            c = (c * recip_1ulp(c.sum(-2, keepdims=True) + F32(EPS))).astype(F32)
        return c

    worst_d, worst_r_div, worst_r_rec = 0.0, 0.0, 0.0
    for k in range(REPS):
        r = np.random.default_rng(3000 + k)
        f = (r.normal(size=(MIX, HC * D)) * (1.0 / np.sqrt(HC * D))).astype(F32)
        b = r.normal(size=MIX).astype(F32)
        h = r.normal(size=(HC, D)).astype(F32)
        _, comb_div, _ = ref.hyper_connection(h, f, b, scale, HC, 20, EPS, RMS_EPS)
        flat = h.reshape(-1).astype(F32)
        flat = (flat / np.sqrt((flat.astype(np.float64) ** 2).mean() + RMS_EPS)).astype(F32)
        cw = (f @ flat).astype(F32)[2 * HC:].reshape(HC, HC) + b[2 * HC:].reshape(HC, HC)
        c0 = (ref.softmax(cw, -1) + F32(EPS)).astype(F32)
        comb_rec = sinkhorn_recip(c0, 20)
        worst_d = max(worst_d, float(np.abs(comb_rec - comb_div).max()))
        worst_r_div = max(worst_r_div, sinkhorn_residual(comb_div))
        worst_r_rec = max(worst_r_rec, sinkhorn_residual(comb_rec))
    print("\nQ5  Sinkhorn with x*recip(y) instead of x/y (40 substituted divisions, +1 ULP recip):")
    print(f"      max |comb(recip) - comb(div)| = {worst_d:.3e}   (comb entries are ~0.25)")
    print(f"      double-stochasticity residual: div {worst_r_div:.3e}  vs  recip {worst_r_rec:.3e}")
    print("      -> the substitution is safe: it moves comb by ~1e-7, four orders below the")
    print("         3e-3 a bf16 map costs (Q2), and it does not degrade the residual.  So mHC")
    print("         RTL can be built on glm_fp_recip.vh; no divider is needed.")


if __name__ == "__main__":
    main()
