#!/usr/bin/env python3
"""
glm53f_mla_score_gen.py -- vectors for test/glm53f_mla_score_tb.v
(src/glm53f_mla_score.v: the absorbed-latent inner loop of GLM-5.3-Flash MLA).

THE GOLDEN IS BIT-EXACT, not a tolerance model, because every piece the DUT uses
already has a bit-exact Python twin in this repo:
  * rmsnorm_unit at LANES=1  -> rmsnorm_bf16 (tools/glm53f_hc_block_gen.py)
  * glm_softmax              -> glm_softmax  (tools/glm_model_q4k_ref.py)
  * bf16 products / fp32 accumulation -> glm_fp.vh semantics, modelled inline
So the unit adds a new DATAFLOW, not new numerics, and the gate can say so
exactly rather than within a bound. tools/glm53f_mla_ref.py holds the *structural*
spec (NoPE, the latent cache, why both absorptions are valid and what they cost);
this file holds the hardware's arithmetic.

WHAT IT CHECKS, AND WHAT IT DOES NOT.  The claim is the inner loop:
    score[h][j] = (qa[h] . rmsnorm(c_kv[j])) * SCALE
    p           = softmax over SMAX slots, slots >= s_len pinned to -inf
    ctx_lat[h]  = SUM_j p[h][j] * rmsnorm(c_kv[j])
It does NOT claim the projections (W_dq/W_uq/W_dkv, and the fold of W_uk into q)
-- those are Q8_0 GEMVs and belong with a glm53f_kda_gemv-shaped fetch unit, gated
separately. `qa` arrives already folded, which is exactly the interface the RTL
has.

THE PADDING IS THE POINT OF THE CORPUS.  s_len is drawn BELOW SMAX on purpose so
the -inf slots are exercised: a unit that forgot them would still pass every test
where s_len == SMAX. The generator asserts the corpus contains at least one test
with s_len < SMAX and one with s_len == SMAX.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm53f_hc_block_gen import rmsnorm_bf16, bf16, bf16b  # noqa: E402
from glm_model_q4k_ref import glm_softmax                  # noqa: E402

F32 = np.float32
NEG_BIG = 0xFF80          # -inf bf16, the softmax pad the RTL writes


def b2f(u16):
    return np.frombuffer(np.uint32(u16 << 16).tobytes(), F32)[0]


def dot_bf16_f32(a, b, scale):
    """fp32 accumulation of bf16 products, in index order, then *scale, then bf16.
    Exactly what S_DOT does on pass 0."""
    acc = F32(0.0)
    for x, y in zip(a, b):
        acc = F32(acc + bf16(F32(F32(x) * F32(y))))
    return bf16(F32(acc * scale))


def ref_score(qa, cache, s_len, smax, scale):
    """The whole unit, bit-for-bit. Returns (ctx_lat[H,KVL], scores, probs)."""
    H, KVL = qa.shape
    one = np.ones(KVL, F32)
    ckvn = [rmsnorm_bf16(cache[j], one) for j in range(s_len)]

    scores = np.full((H, smax), b2f(NEG_BIG), F32)
    for h in range(H):
        for j in range(s_len):
            scores[h, j] = dot_bf16_f32(qa[h], ckvn[j], scale)

    probs = np.zeros((H, smax), F32)
    for h in range(H):
        p = glm_softmax([scores[h, s] for s in range(smax)])
        for s in range(s_len):                      # slots >= s_len forced to 0
            probs[h, s] = p[s]

    ctx = np.zeros((H, KVL), F32)
    for h in range(H):
        for k in range(KVL):
            acc = F32(0.0)
            for j in range(s_len):                  # index order, fp32
                acc = F32(acc + bf16(F32(F32(probs[h, j]) * F32(ckvn[j][k]))))
            ctx[h, k] = bf16(acc)
    return ctx, scores, probs


def gen(ntest, H=2, KVL=8, SMAX=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    scale = F32(1.0 / np.sqrt(KVL))       # the slice's stand-in for 1/sqrt(qk_nope)
    out = [f"{ntest} {H} {KVL} {SMAX}"]
    n_short = n_full = 0
    for t in range(ntest):
        s_len = int(rng.integers(1, SMAX + 1))
        if t == 0:
            s_len = SMAX                   # guarantee both ends appear
        elif t == 1:
            s_len = 1
        n_full += (s_len == SMAX)
        n_short += (s_len < SMAX)
        qa = np.array([[bf16(v) for v in rng.normal(size=KVL) * 0.8]
                       for _ in range(H)], F32)
        cache = np.array([[bf16(v) for v in rng.normal(size=KVL) * 1.1]
                          for _ in range(SMAX)], F32)
        ctx, _, _ = ref_score(qa, cache, s_len, SMAX, scale)
        out.append(f"{s_len}")
        out.append(" ".join(f"{bf16b(v):04x}" for row in qa for v in row))
        out.append(" ".join(f"{bf16b(v):04x}" for row in cache for v in row))
        out.append(" ".join(f"{bf16b(v):04x}" for row in ctx for v in row))
    assert n_short and n_full, ("the corpus must contain both a PADDED case and a "
                                "full-window case, or the -inf pad is untested")
    if report:
        print(f"mla-score: {ntest} tests, H={H} KVL={KVL} SMAX={SMAX}; "
              f"{n_short} with s_len < SMAX (the -inf pad is live), {n_full} full",
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

    rng = np.random.default_rng(0xA5C)
    H, KVL, SMAX = 3, 8, 5
    scale = F32(1.0 / np.sqrt(KVL))
    qa = np.array([[bf16(v) for v in rng.normal(size=KVL)] for _ in range(H)], F32)
    cache = np.array([[bf16(v) for v in rng.normal(size=KVL)] for _ in range(SMAX)], F32)

    ctx, sc, pr = ref_score(qa, cache, SMAX, SMAX, scale)
    chk(ctx.shape == (H, KVL), "ctx must be [H, KVL] -- the LATENT basis, not [H, V]")
    chk(np.isfinite(ctx).all(), "non-finite ctx")
    chk(all(abs(float(pr[h].sum()) - 1.0) < 2e-2 for h in range(H)),
        "softmax rows must sum to ~1")

    # PADDING: a shorter window must equal the same window with the extra keys
    # simply absent -- that is what the -inf pad has to buy.
    ctx_s, _, pr_s = ref_score(qa, cache, 2, SMAX, scale)
    ctx_c, _, _ = ref_score(qa, cache[:2], 2, 2, scale)
    chk(np.array_equal(ctx_s, ctx_c),
        "padding to SMAX changed the answer -- the -inf slots are contributing")
    chk(float(pr_s[:, 2:].max()) == 0.0, "a padded slot got non-zero probability")

    # the pad must MATTER: without it (pad = 0.0 instead of -inf) the answer moves
    scores_bad = sc.copy()
    scores_bad[:, 2:] = F32(0.0)
    p_bad = np.zeros((H, SMAX), F32)
    for h in range(H):
        p_bad[h] = glm_softmax([scores_bad[h, s] for s in range(SMAX)])
    chk(not np.allclose(p_bad[:, :2], pr_s[:, :2], atol=1e-6),
        "padding with 0.0 instead of -inf gives the same probabilities -- then the "
        "pad value is untested and the corpus proves nothing about it")

    # a single key: softmax is 1, so ctx is exactly that normalised latent
    ctx1, _, _ = ref_score(qa, cache, 1, SMAX, scale)
    one = np.ones(KVL, F32)
    ck0 = rmsnorm_bf16(cache[0], one)
    chk(all(abs(float(ctx1[h, k]) - float(ck0[k])) <= abs(float(ck0[k])) * 0.02
            for h in range(H) for k in range(KVL)),
        "with one key the context must be that key's normalised latent")

    # qa = 0 makes every score equal, so the context is the MEAN of the latents
    ctxu, _, pru = ref_score(np.zeros((H, KVL), F32), cache, SMAX, SMAX, scale)
    chk(float(pru[:, :SMAX].std(axis=1).max()) < 1e-3,
        "equal scores did not give a uniform distribution")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_mla_score_gen self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (ctx stays in the latent basis; softmax rows sum "
          f"to 1; the -inf pad makes a padded window identical to the shorter one "
          f"AND padding with 0.0 instead would change it, so the pad value is "
          f"actually tested; one key gives that key's latent; equal scores give a "
          f"uniform distribution)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 6
    outp = a[1] if len(a) > 1 else "build/glm53f_mla_score_vec.txt"
    seed = int(a[2]) if len(a) > 2 else 0
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, seed=seed))
    print(f"wrote {outp}: {ntest} tests")
