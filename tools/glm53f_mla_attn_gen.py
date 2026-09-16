#!/usr/bin/env python3
"""
glm53f_mla_attn_gen.py -- vectors for test/glm53f_mla_attn_tb.v
(src/glm53f_mla_attn.v: the whole GLM-5.3-Flash MLA sublayer).

    proj   x           -> qa (q folded through W_uk), RAW latent c_kv
    score  qa + cache  -> ctx in the LATENT basis
    out    ctx_lat     -> W_uv per head -> W_o -> y

BIT-EXACT END TO END, by COMPOSING the twins that already gate the parts:
    glm53f_mla_proj_gen.project      (make mla-proj)
    glm53f_mla_score_gen.ref_score   (make mla-score)
    matmul_q4k_col                   for W_uv and W_o
Nothing is re-derived here, which is the point: if this gate ever disagrees with
the unit gates, the composition is wrong rather than the arithmetic.

WHAT THE CORPUS HAS TO EXERCISE, and does:
  * s_len < SMAX as well as s_len == SMAX, so the score's -inf padding is live;
  * s_len == 1 (the first token of a sequence: the cache holds ONLY this token's
    own latent, written this step);
  * a PRE-EXISTING cache, so the new latent is appended rather than being the
    whole story -- a unit that ignored ckv_wr would still pass an s_len==1 corpus.

THE CACHE IS THE TESTBENCH'S. The DUT publishes the new latent and pulls old ones
by index; it stores nothing. That is the module's contract (a 1 M-token context is
~1 GB of latent per MLA block, a residency decision that belongs with the model),
so the generator emits the cache contents as data the TB owns.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4k_ref import matmul_q4k_col                                 # noqa: E402
from glm53f_hc_block_gen import bf16, bf16b                        # noqa: E402
from glm53f_kda_attn_gen import q80_rows, fp16b                    # noqa: E402
from glm53f_mla_proj_gen import project                            # noqa: E402
from glm53f_mla_score_gen import ref_score                         # noqa: E402

F32 = np.float32
WNAMES = ("Wdq", "Wuq", "Wuk", "Wdkv", "Wuv", "Wo")


def sublayer(x, cache_prev, s_len, deq, SMAX, QK):
    """The whole sublayer, bit-for-bit. Returns (y, ckv, ctx_lat)."""
    qa, ckv, _, _ = project(x, deq["Wdq"], deq["Wuq"], deq["Wuk"], deq["Wdkv"])
    cache = np.concatenate([np.asarray(cache_prev, F32).reshape(-1, ckv.shape[0]),
                            ckv[None, :]], axis=0)
    assert cache.shape[0] == s_len, "the new latent must be the s_len-th key"
    full = np.zeros((SMAX, ckv.shape[0]), F32)
    full[:s_len] = cache
    ctx_lat, _, _ = ref_score(qa, full, s_len, SMAX, F32(1.0 / np.sqrt(QK)))
    H, VD = deq["Wuv"].shape[0], deq["Wuv"].shape[1]
    ctx = np.zeros((H, VD), F32)
    for h in range(H):
        for d in range(VD):
            ctx[h, d] = matmul_q4k_col(ctx_lat[h], deq["Wuv"][h, d])
    flat = ctx.reshape(-1)
    y = np.array([matmul_q4k_col(flat, deq["Wo"][i])
                  for i in range(deq["Wo"].shape[0])], F32)
    return y, ckv, ctx_lat


def _draw(rng, MD, H, QK, QLORA, KVL, VD):
    x = np.array([bf16(v) for v in rng.normal(size=MD) * 0.9], F32)
    raw = dict(
        Wdq=(rng.normal(size=(QLORA, MD)) * 0.5).astype(F32),
        Wuq=(rng.normal(size=(H * QK, QLORA)) * 0.5).astype(F32),
        Wuk=(rng.normal(size=(H, KVL, QK)) * 0.5).astype(F32),
        Wdkv=(rng.normal(size=(KVL, MD)) * 0.5).astype(F32),
        Wuv=(rng.normal(size=(H, VD, KVL)) * 0.5).astype(F32),
        Wo=(rng.normal(size=(MD, H * VD)) * 0.5).astype(F32))
    codes, scales, deq = {}, {}, {}
    for nm in WNAMES:
        W = raw[nm]
        q, d, dq = q80_rows(W.reshape(-1, W.shape[-1]))
        codes[nm], scales[nm], deq[nm] = q, d, dq.reshape(W.shape)
    return x, codes, scales, deq


def gen(ntest, MD=16, H=2, QK=8, QLORA=8, KVL=8, VD=8, SMAX=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {MD} {H} {QK} {QLORA} {KVL} {VD} {SMAX}"]
    n_first = n_pad = n_full = 0
    for t in range(ntest):
        s_len = int(rng.integers(1, SMAX + 1))
        if t == 0:
            s_len = 1                      # first token: cache is only its own
        elif t == 1:
            s_len = SMAX                   # full window
        n_first += (s_len == 1)
        n_pad += (s_len < SMAX)
        n_full += (s_len == SMAX)
        x, codes, scales, deq = _draw(rng, MD, H, QK, QLORA, KVL, VD)
        prev = np.array([[bf16(v) for v in rng.normal(size=KVL) * 1.1]
                         for _ in range(s_len - 1)], F32).reshape(-1, KVL)
        y, ckv, _ = sublayer(x, prev, s_len, deq, SMAX, QK)

        out.append(f"{s_len}")
        out.append(" ".join(f"{bf16b(v):04x}" for v in x))
        for nm in WNAMES:
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in codes[nm].reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(scales[nm]).reshape(-1)))
        # the pre-existing cache, padded to SMAX-1 rows so the TB reads a fixed count
        pad = np.zeros((SMAX - 1, KVL), F32)
        if s_len > 1:
            pad[:s_len - 1] = prev
        out.append(" ".join(f"{bf16b(v):04x}" for row in pad for v in row))
        out.append(" ".join(f"{bf16b(v):04x}" for v in ckv))
        out.append(" ".join(f"{bf16b(v):04x}" for v in y))
    assert n_first and n_pad and n_full, (
        "the corpus must contain s_len == 1 (first token), a PADDED window and a "
        "FULL one, or the cache append and the -inf pad are not both exercised")
    if report:
        print(f"glm53f-mla-attn: {ntest} tokens, MD={MD} H={H} QK={QK} "
              f"q_lora={QLORA} kv_lora={KVL} v={VD} SMAX={SMAX}; "
              f"{n_first} first-token, {n_pad} padded, {n_full} full-window",
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

    rng = np.random.default_rng(0xC0FE)
    MD, H, QK, QLORA, KVL, VD, SMAX = 16, 2, 8, 8, 8, 8, 4
    x, codes, scales, deq = _draw(rng, MD, H, QK, QLORA, KVL, VD)
    prev = np.array([[bf16(v) for v in rng.normal(size=KVL)] for _ in range(2)], F32)

    y3, ckv3, cl3 = sublayer(x, prev, 3, deq, SMAX, QK)
    chk(y3.shape == (MD,), "the sublayer must return a [MODEL_DIM] vector")
    chk(np.isfinite(y3).all(), "non-finite output")
    chk(cl3.shape == (H, KVL), "the context must reach the output stage in the LATENT basis")

    # THE NEW LATENT MUST BE IN THE WINDOW. Running with the same prev but claiming
    # s_len = 2 (i.e. dropping this token's own latent) must change the answer.
    y2, _, _ = sublayer(x, prev[:1], 2, deq, SMAX, QK)
    chk(not np.array_equal(y3, y2),
        "dropping a cached key left the answer unchanged -- the cache is inert")

    # FIRST TOKEN: the cache is exactly this token's own latent, so the softmax is
    # over one key and the context is that latent.
    y1, ckv1, cl1 = sublayer(x, np.zeros((0, KVL), F32), 1, deq, SMAX, QK)
    from glm53f_hc_block_gen import rmsnorm_bf16
    ck1n = rmsnorm_bf16(ckv1, np.ones(KVL, F32))
    chk(all(abs(float(cl1[h, k]) - float(ck1n[k])) <= abs(float(ck1n[k])) * 0.02 + 1e-3
            for h in range(H) for k in range(KVL)),
        "with one key the latent context must be this token's own normalised latent")

    # THE CACHED LATENT IS RAW: a normalised one has RMS 1.
    rms = float(np.sqrt((ckv3.astype(np.float64) ** 2).mean()))
    chk(abs(rms - 1.0) > 1e-3,
        f"the published latent has RMS {rms:.6f} ~ 1 -- it is normalised on write")

    # THE OUTPUT STAGE IS LIVE ON EVERY HEAD: perturbing head 1's W_uv must move y.
    d2 = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in deq.items()}
    d2["Wuv"] = deq["Wuv"].copy()
    d2["Wuv"][1] = (d2["Wuv"][1] + F32(0.7)).astype(F32)
    yb, _, _ = sublayer(x, prev, 3, d2, SMAX, QK)
    chk(not np.array_equal(y3, yb), "head 1's W_uv does not reach the output")

    # COMPOSITION, not re-derivation: the pieces here are the gated ones.
    chk(project.__module__.endswith("glm53f_mla_proj_gen")
        and ref_score.__module__.endswith("glm53f_mla_score_gen"),
        "the golden must COMPOSE the unit generators, not re-implement them")

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_mla_attn_gen self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (the sublayer returns [MODEL_DIM]; the context "
          f"reaches the output stage in the latent basis; the cache is live -- "
          f"dropping a key changes the answer; the first token attends to its own "
          f"latent; the published latent is RAW (RMS != 1); every head's W_uv "
          f"reaches the output; and the golden COMPOSES the already-gated unit "
          f"generators rather than re-deriving them)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 5
    outp = a[1] if len(a) > 1 else "build/glm53f_mla_attn_vec.txt"
    seed = int(a[2]) if len(a) > 2 else 0
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, seed=seed))
    print(f"wrote {outp}: {ntest} tests")
