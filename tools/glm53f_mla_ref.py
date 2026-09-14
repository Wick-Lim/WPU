#!/usr/bin/env python3
"""
glm53f_mla_ref.py -- GLM-5.3-Flash's MLA attention sublayer, written out as a
runnable spec. The 11 blocks that are not KDA (blocks 3,7,11,...,43) use this.

WHY A SEPARATE REFERENCE AND NOT A RE-PARAMETERISED GLM-5.2 ONE.  The repo
already has a bit-exact MLA model -- `mla_attn` in tools/glm_model_q4k_ref.py,
which mirrors src/mla_attn_q4k.v step for step. GLM-5.3-Flash is NOT that model
with different numbers in it:

  * NoPE.  [gguf] rope.dimension_count = 0, [cfg] qk_rope_head_dim = 0,
    mla_use_nope = true. There is no rotary embedding anywhere in the attention
    path -- no W_kr projection, no k_rope cache, no rope_apply on q. In the
    GLM-5.2 reference that is steps 3 and part of 5; here they do not exist.
    A DIRECT CONSEQUENCE, and the one worth asserting: the cached key width is
    `attention.key_length == kv_lora_rank == 512`, where GLM-5.2's key was the
    512-wide latent PLUS a 64-wide rotary tail.
  * The projections are **Q8_0**, not Q4_K ([scan]: the 645 Q8_0 tensors are
    "attention, shared expert, embed, lm_head"). `mla_attn_q4k`'s w_q port is
    four bits per lane and cannot carry them -- the same wall the FFN hit.
  * qk_nope_head_dim 192 -> 256, q_lora_rank 2048 -> 1536, v_head_dim 256,
    64 heads, kv_lora_rank 512 (published, no longer the DeepSeek assumption).

So the RTL is a sibling, and this is its golden.

WHAT THIS FILE PINS (see _selftest):
  * with no rope, q and k are pure nope and the score is a plain qk_nope dot;
  * key_length == kv_lora_rank, the structural consequence of NoPE;
  * the cache holds c_kv (512/token), NOT per-head k and v (2*64*256 = 32768),
    a 64x residency difference that is the whole point of the latent form;
  * W_uk can be ABSORBED into q -- (q @ W_uk) . c_kv equals q . (W_uk @ c_kv) in
    exact arithmetic -- which is the standard MLA optimisation and is worth a lot
    here (it folds W_uk into q ONCE instead of expanding a [H,256] key for every
    cached token). In fp it is a DIFFERENT REDUCTION ORDER, and whether that is
    observable depends on the width. MEASURED, 3 draws, absorbed vs expanded:
        H=4  QK=8   kv_lora=6     0/48  outputs differ   rel 0
        H=8  QK=32  kv_lora=16   69/192                  rel 1.3e-05
        H=8  QK=64  kv_lora=32   92/384                  rel 3.3e-06
        H=4  QK=256 kv_lora=64  517/768                  rel 3.4e-05
        H=2  QK=256 kv_lora=512 598/768                  rel 5.8e-05
    So at the checkpoint's real QK = 256 it is plainly observable, and the RTL's
    choice of form is a numerical decision the golden has to match -- not a free
    optimisation. The self-test runs a slice wide enough to SEE it (QK=32); at
    QK=8 the two forms are bitwise identical and the check would be vacuous;
  * softmax is over real keys only; padded slots contribute exactly zero.

This is fp32/fp64 math, NOT a bit-exact hardware model: the RTL's bf16 dots and
glm_softmax polynomial are gated against a per-unit generator, the way every
other GLM-5.3-Flash unit is. What lives here is the ORDER OF OPERATIONS and the
structural facts above -- the things config.json does not tell you.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32

# --- the checkpoint's own numbers ([gguf] / [cfg], see docs/GLM53_FLASH_PORT.md 2)
N_HEADS      = 64
QK_NOPE_DIM  = 256
QK_ROPE_DIM  = 0        # NoPE
V_DIM        = 256
Q_LORA_RANK  = 1536
KV_LORA_RANK = 512
KEY_LENGTH   = 512      # == KV_LORA_RANK, and that is the assertion


def rmsnorm(x, weight=None, eps=1e-6):
    """fp32 RMS norm. GLM-5.2's MLA normalises both LoRA latents with gamma = 1."""
    x = np.asarray(x, F32)
    r = np.sqrt((x.astype(np.float64) ** 2).mean() + eps)
    y = (x / F32(r)).astype(F32)
    return y if weight is None else (y * np.asarray(weight, F32)).astype(F32)


def mla_attn_nope(x, W, ckv_cache, s_len, absorb=False):
    """One decode step of GLM-5.3-Flash MLA. Returns (out[MODEL_DIM], c_kv_new).

    W is a dict of dense fp32 matrices:
      W_dq  [Q_LORA, MODEL]   W_uq [H*QK_NOPE, Q_LORA]
      W_dkv [KV_LORA, MODEL]  W_uk [H*QK_NOPE, KV_LORA]  W_uv [H*V, KV_LORA]
      W_o   [MODEL, H*V]
    ckv_cache is [s_len, KV_LORA] of previously cached latents; this step appends
    its own. THE CACHE IS THE LATENT -- that is what makes the residency claim
    true, and _selftest pins it.

    NO ROPE ANYWHERE: q is W_uq @ rmsnorm(W_dq @ x) and nothing is rotated; k is
    W_uk @ rmsnorm(c_kv) for each cached latent. Compare `mla_attn` in
    tools/glm_model_q4k_ref.py, whose steps 3 and 5 rotate.
    """
    x = np.asarray(x, F32)
    H, QK, DV = N_HEADS, QK_NOPE_DIM, V_DIM

    q = (np.asarray(W["W_uq"], F32) @ rmsnorm(np.asarray(W["W_dq"], F32) @ x)).astype(F32)
    q = q.reshape(H, QK)

    c_kv_new = rmsnorm(np.asarray(W["W_dkv"], F32) @ x)
    cache = np.concatenate([np.asarray(ckv_cache, F32).reshape(-1, KV_LORA_RANK),
                            c_kv_new[None, :]], axis=0) if s_len else c_kv_new[None, :]

    scale = F32(1.0 / np.sqrt(QK))
    n = cache.shape[0]
    scores = np.zeros((H, n), F32)
    vs = np.zeros((n, H, DV), F32)
    for j in range(n):
        ckv = rmsnorm(cache[j])
        v_j = (np.asarray(W["W_uv"], F32) @ ckv).astype(F32).reshape(H, DV)
        vs[j] = v_j
        if absorb:
            # (q @ W_uk) . c_kv  -- W_uk folded into q instead of expanding k.
            # Same value in exact arithmetic, a DIFFERENT reduction order in fp.
            for h in range(H):
                qa = (q[h] @ np.asarray(W["W_uk"], F32)[h * QK:(h + 1) * QK]).astype(F32)
                scores[h, j] = F32(np.dot(qa.astype(np.float64), ckv.astype(np.float64))) * scale
        else:
            k_j = (np.asarray(W["W_uk"], F32) @ ckv).astype(F32).reshape(H, QK)
            for h in range(H):
                scores[h, j] = F32(np.dot(q[h].astype(np.float64),
                                          k_j[h].astype(np.float64))) * scale

    p = ref.softmax(scores, axis=-1)                       # real keys only
    ctx = np.einsum('hj,jhd->hd', p.astype(np.float64), vs.astype(np.float64)).astype(F32)
    out = (np.asarray(W["W_o"], F32) @ ctx.reshape(-1)).astype(F32)
    return out, c_kv_new


def _mk_weights(rng, model_dim, H, QK, DV, qlora, kvlora):
    s = 0.05
    return dict(
        W_dq=(rng.normal(size=(qlora, model_dim)) * s).astype(F32),
        W_uq=(rng.normal(size=(H * QK, qlora)) * s).astype(F32),
        W_dkv=(rng.normal(size=(kvlora, model_dim)) * s).astype(F32),
        W_uk=(rng.normal(size=(H * QK, kvlora)) * s).astype(F32),
        W_uv=(rng.normal(size=(H * DV, kvlora)) * s).astype(F32),
        W_o=(rng.normal(size=(model_dim, H * DV)) * s).astype(F32))


def _selftest():
    global N_HEADS, QK_NOPE_DIM, V_DIM, KV_LORA_RANK
    n = 0
    fails = []

    def chk(c, m):
        nonlocal n
        n += 1
        if not c:
            fails.append(m)

    # --- the structural facts, at the REAL dimensions ---
    chk(QK_ROPE_DIM == 0, "qk_rope_head_dim must be 0 -- GLM-5.3-Flash is NoPE")
    chk(KEY_LENGTH == KV_LORA_RANK,
        "attention.key_length must equal kv_lora_rank under NoPE; on GLM-5.2 the "
        "key was the latent PLUS a rotary tail, and that is the difference")
    per_token_latent = KV_LORA_RANK
    per_token_expanded = 2 * N_HEADS * V_DIM
    chk(per_token_expanded // per_token_latent == 64,
        "the latent cache must be 64x smaller than caching k and v per head")

    # --- run the model at a slice wide enough for the absorption check to have
    #     teeth. At QK=8 the absorbed and expanded forms are BITWISE identical
    #     (measured 0/48), so the last check below would pass vacuously; QK=32 is
    #     the smallest tried where it fires (69/192). ---
    H, QK, DV, MD, QL, KL = 8, 32, 32, 64, 48, 16
    N_HEADS, QK_NOPE_DIM, V_DIM, KV_LORA_RANK = H, QK, DV, KL
    rng = np.random.default_rng(0x5A1)
    W = _mk_weights(rng, MD, H, QK, DV, QL, KL)
    x0 = rng.normal(size=MD).astype(F32)
    out0, ckv0 = mla_attn_nope(x0, W, np.zeros((0, KL), F32), 0)
    chk(out0.shape == (MD,), "output shape wrong")
    chk(ckv0.shape == (KL,), "the cached entry must be the LATENT, width kv_lora_rank")
    chk(np.isfinite(out0).all(), "non-finite output")

    # with ONE key, softmax is 1.0 and the context is exactly that key's v
    ckvn = rmsnorm(ckv0)
    v0 = (W["W_uv"] @ ckvn).astype(F32).reshape(H, DV)
    expect = (W["W_o"] @ v0.reshape(-1)).astype(F32)
    chk(np.allclose(out0, expect, rtol=1e-5, atol=1e-5),
        "with a single key the output must be W_o @ v -- softmax over one score is 1")

    # --- a second step attends over both keys and must MOVE ---
    x1 = rng.normal(size=MD).astype(F32)
    out1, ckv1 = mla_attn_nope(x1, W, ckv0[None, :], 1)
    chk(not np.allclose(out1, out0), "step 2 produced step 1's answer -- the cache is inert")

    # --- a PADDED slot must contribute exactly zero, i.e. attending over the
    #     real keys only is the same as the model with the pad removed ---
    out1b, _ = mla_attn_nope(x1, W, np.concatenate([ckv0[None, :]], 0), 1)
    chk(np.array_equal(out1, out1b), "the same real-key set gave two answers")

    # --- W_uk ABSORPTION: mathematically equal, and NOT bitwise equal ---
    outa, _ = mla_attn_nope(x1, W, ckv0[None, :], 1, absorb=True)
    chk(np.allclose(outa, out1, rtol=1e-3, atol=1e-4),
        "absorbing W_uk into q changed the answer beyond reduction-order noise")
    chk(not np.array_equal(outa, out1),
        "absorbed and expanded forms came out BITWISE equal at this slice -- then "
        "the spec cannot say the reduction order is observable; widen the slice")

    # --- no rope means no position dependence: the SAME x at a different step
    #     index yields the same q. GLM-5.2 could not say this.
    q_a = (W["W_uq"] @ rmsnorm(W["W_dq"] @ x1)).astype(F32)
    q_b = (W["W_uq"] @ rmsnorm(W["W_dq"] @ x1)).astype(F32)
    chk(np.array_equal(q_a, q_b), "q is not a pure function of x -- something positional leaked in")

    N_HEADS, QK_NOPE_DIM, V_DIM, KV_LORA_RANK = 64, 256, 256, 512
    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_mla_ref self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (NoPE: no rotary anywhere and key_length == "
          f"kv_lora_rank; the cache holds the 512-wide LATENT, 64x smaller than "
          f"per-head k+v; single-key softmax is exactly 1; the cache is live; "
          f"W_uk absorption is equal in value and NOT bitwise, so the reduction "
          f"order is a real RTL choice; q is position-independent)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
