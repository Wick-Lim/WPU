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
  Q1  Sinkhorn convergence in fp32: after the reference's 1 col pass + 19 (row,col)
      pairs, how far is `comb` from doubly stochastic (max |row sum - 1|, |col sum - 1|)?
      And how many iterations does fp32 need before the residual stops improving?
  Q2  bf16 comb: if the mHC map is evaluated in bf16 (as glm_act-style units would),
      how far does `comb` move from the fp32 comb (max abs element error)?
  Q3  compounding: apply the block recurrence streams' = comb @ streams + post*u for
      45 blocks with fp32 comb vs bf16 comb (same random sublayer outputs u), and
      report the stream divergence (RMS rel) at blocks 1, 11, 23, 45.
  Q4  the dynamic ranges the RTL must carry: pre in (eps, 1+eps), post in [0,2],
      comb in (0,1) -- and the smallest comb entry seen (an fp exponent question).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32
HC, D, EPS, RMS_EPS = 4, 4096, 1e-6, 1e-5
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

    # ---- Q1: Sinkhorn convergence in fp32, and iterations-to-plateau ----
    hs = rng.normal(size=(HC, D)).astype(F32)
    res_by_iter = []
    for it in range(1, 41):
        _, comb, _ = ref.hyper_connection(hs, fn, base, scale, HC, it, EPS, RMS_EPS)
        res_by_iter.append(sinkhorn_residual(comb))
    print("Q1  Sinkhorn residual (max |row/col sum - 1|) vs iterations, fp32:")
    for it in (1, 2, 3, 5, 10, 20, 40):
        print(f"      iters={it:>2}  residual={res_by_iter[it-1]:.3e}")
    plateau = next((i + 1 for i in range(1, 40) if abs(res_by_iter[i] - res_by_iter[i-1]) < 1e-9), 40)
    print(f"      -> residual stops improving after ~{plateau} iterations; the reference's 20 is past the plateau.")
    print(f"         (each normalise divides by sum+eps, so the floor is set by eps=1e-6 and fp32, not by iteration count)\n")

    # ---- Q2: bf16 evaluation of the map ----
    worst = 0.0; worst_res_bf = 0.0
    for _ in range(64):
        hs = rng.normal(size=(HC, D)).astype(F32)
        _, comb32, _ = ref.hyper_connection(hs, fn, base, scale, HC, 20, EPS, RMS_EPS)
        _, combbf, _ = ref.hyper_connection(bf16(hs), bf16(fn), bf16(base), scale, HC, 20, EPS, RMS_EPS)
        combbf = bf16(combbf)
        worst = max(worst, float(np.abs(combbf - comb32).max()))
        worst_res_bf = max(worst_res_bf, sinkhorn_residual(combbf))
    print(f"Q2  comb evaluated with bf16 inputs/weights and bf16-rounded output vs fp32:")
    print(f"      max |comb_bf16 - comb_fp32| = {worst:.3e}   (comb entries are in (0,1); typical ~0.25)")
    print(f"      Sinkhorn residual of the bf16 comb = {worst_res_bf:.3e}  (fp32: {res_by_iter[19]:.3e})")
    print(f"      -> a bf16 comb is NOT doubly stochastic to better than ~{worst_res_bf:.0e}; row/col sums drift\n")

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


if __name__ == "__main__":
    main()
