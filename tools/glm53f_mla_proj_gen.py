#!/usr/bin/env python3
"""
glm53f_mla_proj_gen.py -- vectors for test/glm53f_mla_proj_tb.v
(src/glm53f_mla_proj.v: the MLA projection front, Q8_0 off glm_matmul_q4k).

    c_q   = W_dq  @ x                [Q_LORA]
    q     = W_uq  @ rmsnorm(c_q)     [H*QK]
    qa[h] = q[h]  @ W_uk[h]          [KV_LORA]   <- the fold, one pass per head
    c_kv  = W_dkv @ x                [KV_LORA]   <- RAW: see below

BIT-EXACT, like `make mla-score`, because every piece already has an exact twin:
  * the GEMV          -> matmul_q4k_col (tools/q4k_ref.py): sequential fp32
                         accumulation of bf16 activation x dequantised weight,
                         rounded to bf16 once -- the RTL's contract exactly
  * rmsnorm at LANES=1-> rmsnorm_bf16 (tools/glm53f_hc_block_gen.py)
  * Q8_0              -> q80_rows (tools/glm53f_kda_attn_gen.py)
The golden runs on the DEQUANTISED weights, so the Q8_0 round trip is part of the
INPUT rather than of the error -- the same contract `make kda-attn` uses.

THE WEIGHT LAYOUT IS THE INTERESTING PART.  Every projection is stored as "one
output column per row, quantised along the reduction axis", which is what the
engine streams. For the FOLD that means W_uk is [H][KV_LORA][QK]: for head h and
output k, the reduction runs over d, i.e. down W_uk's rows. Getting that axis
wrong is a transpose that still has the right shape, which is why the TB carries
an injection for it.

c_kv IS EMITTED RAW.  rmsnorm belongs on the read side, once, inside
glm53f_mla_score. Normalising here as well applies it twice; rmsnorm is nearly
idempotent (measured 2.8e-06 relative) so shape checks do not notice, which is
exactly how it got into tools/glm53f_mla_ref.py before being caught. The self-test
asserts the emitted latent's RMS is NOT 1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4k_ref import matmul_q4k_col                       # noqa: E402
from glm53f_hc_block_gen import rmsnorm_bf16, bf16, bf16b  # noqa: E402
from glm53f_kda_attn_gen import q80_rows, fp16b           # noqa: E402

F32 = np.float32


def project(x, Wdq, Wuq, Wuk, Wdkv):
    """The whole front, bit-for-bit. Wuk is [H][KVL][QK]."""
    H = Wuk.shape[0]
    QK = Wuk.shape[2]
    cq = np.array([matmul_q4k_col(x, Wdq[i]) for i in range(Wdq.shape[0])], F32)
    cqn = rmsnorm_bf16(cq, np.ones(cq.shape[0], F32))
    q = np.array([matmul_q4k_col(cqn, Wuq[i]) for i in range(Wuq.shape[0])], F32)
    qa = np.zeros((H, Wuk.shape[1]), F32)
    for h in range(H):
        qh = q[h * QK:(h + 1) * QK]
        for k in range(Wuk.shape[1]):
            qa[h, k] = matmul_q4k_col(qh, Wuk[h, k])
    ckv = np.array([matmul_q4k_col(x, Wdkv[i]) for i in range(Wdkv.shape[0])], F32)
    return qa, ckv, cq, q


def _draw(rng, MD, H, QK, QLORA, KVL):
    x = np.array([bf16(v) for v in rng.normal(size=MD) * 0.9], F32)
    raw = dict(
        Wdq=(rng.normal(size=(QLORA, MD)) * 0.5).astype(F32),
        Wuq=(rng.normal(size=(H * QK, QLORA)) * 0.5).astype(F32),
        Wuk=(rng.normal(size=(H, KVL, QK)) * 0.5).astype(F32),
        Wdkv=(rng.normal(size=(KVL, MD)) * 0.5).astype(F32))
    codes, scales, deq = {}, {}, {}
    for nm, W in raw.items():
        flat = W.reshape(-1, W.shape[-1])
        q, d, dq = q80_rows(flat)
        codes[nm], scales[nm] = q, d
        deq[nm] = dq.reshape(W.shape)
    return x, codes, scales, deq


def gen(ntest, MD=16, H=2, QK=8, QLORA=8, KVL=8, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {MD} {H} {QK} {QLORA} {KVL}"]
    worst_q = 0.0
    for _ in range(ntest):
        x, codes, scales, deq = _draw(rng, MD, H, QK, QLORA, KVL)
        qa, ckv, _, _ = project(x, deq["Wdq"], deq["Wuq"], deq["Wuk"], deq["Wdkv"])
        out.append(" ".join(f"{bf16b(v):04x}" for v in x))
        for nm in ("Wdq", "Wuq", "Wuk", "Wdkv"):
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in codes[nm].reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(scales[nm]).reshape(-1)))
        out.append(" ".join(f"{bf16b(v):04x}" for row in qa for v in row))
        out.append(" ".join(f"{bf16b(v):04x}" for v in ckv))
    if report:
        print(f"mla-proj: {ntest} tokens, MD={MD} H={H} QK={QK} q_lora={QLORA} "
              f"kv_lora={KVL}; golden on the DEQUANTISED weights", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    n = 0
    fails = []

    def chk(c, m):
        nonlocal n
        n += 1
        if not c:
            fails.append(m)

    rng = np.random.default_rng(0xB33)
    MD, H, QK, QLORA, KVL = 16, 2, 8, 8, 8
    x, codes, scales, deq = _draw(rng, MD, H, QK, QLORA, KVL)
    qa, ckv, cq, q = project(x, deq["Wdq"], deq["Wuq"], deq["Wuk"], deq["Wdkv"])

    chk(qa.shape == (H, KVL), "qa must be [H, kv_lora] -- q already folded through W_uk")
    chk(ckv.shape == (KVL,), "the latent must be kv_lora wide")
    chk(np.isfinite(qa).all() and np.isfinite(ckv).all(), "non-finite output")

    # THE LATENT IS RAW. A normalised one has RMS 1; this must not.
    rms = float(np.sqrt((ckv.astype(np.float64) ** 2).mean()))
    chk(abs(rms - 1.0) > 1e-3,
        f"the emitted latent has RMS {rms:.6f} ~ 1 -- it is being normalised here, "
        f"so rmsnorm would be applied twice once the reader normalises it")

    # the c_q rmsnorm is LIVE: skipping it changes q
    q_nonorm = np.array([matmul_q4k_col(cq, deq["Wuq"][i])
                         for i in range(H * QK)], F32)
    chk(not np.allclose(q, q_nonorm),
        "q is the same with and without the c_q rmsnorm -- the norm is inert")

    # THE FOLD AXIS. Reducing over the wrong axis of W_uk has the SAME shape, so
    # this is the check that the transpose would fail.
    qa_t = np.zeros((H, KVL), F32)
    for h in range(H):
        qh = q[h * QK:(h + 1) * QK]
        Wt = deq["Wuk"][h].T                    # [QK][KVL] -- wrong way round
        for k in range(min(KVL, QK)):
            qa_t[h, k] = matmul_q4k_col(qh[:min(KVL, QK)], Wt[k][:min(KVL, QK)])
    chk(not np.allclose(qa[:, :min(KVL, QK)], qa_t[:, :min(KVL, QK)]),
        "folding over the transposed axis of W_uk gives the same answer -- then the "
        "axis is untested and a transpose would ship")

    # the fold really is q[h] @ W_uk[h]: perturbing head 1's q must not move head 0
    q2 = q.copy()
    q2[QK:] = (q2[QK:] + F32(3.0)).astype(F32)
    qa2 = np.zeros((H, KVL), F32)
    for h in range(H):
        qh = q2[h * QK:(h + 1) * QK]
        for k in range(KVL):
            qa2[h, k] = matmul_q4k_col(qh, deq["Wuk"][h, k])
    chk(np.array_equal(qa[0], qa2[0]), "head 1's q changed head 0's qa -- heads are mixed")
    chk(not np.array_equal(qa[1], qa2[1]), "head 1's qa did not move when its q did")

    # Q8_0 round trip is part of the INPUT: the dequantised weights are what the
    # golden uses, so the codes must actually reconstruct them
    q8, d8, dq8 = q80_rows(deq["Wdkv"])
    chk(np.allclose(dq8, deq["Wdkv"], atol=2e-2),
        "re-quantising an already-dequantised weight moved it a lot -- the round "
        "trip is not idempotent and the corpus is not self-consistent")

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_mla_proj_gen self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (qa is [H,kv_lora] with q folded through W_uk; the "
          f"latent is emitted RAW, RMS != 1; the c_q rmsnorm is live; the fold "
          f"reduces over the right axis of W_uk and a transpose would differ; heads "
          f"do not mix; the Q8_0 round trip is idempotent so the corpus is "
          f"self-consistent)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 4
    outp = a[1] if len(a) > 1 else "build/glm53f_mla_proj_vec.txt"
    seed = int(a[2]) if len(a) > 2 else 0
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest, seed=seed))
    print(f"wrote {outp}: {ntest} tests")
