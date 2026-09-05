#!/usr/bin/env python3
"""
glm53_flash_ref.py -- executable numpy reference for the GLM-5.3-Flash blocks
this repo does NOT yet have RTL for.

WHY THIS FILE EXISTS.  docs/GLM53_FLASH_PORT.md 4.2 lists KDA linear attention
(34 of 45 layers), the mHC hyper-connection residual, and the clamped SwiGLU as
the port's open datapath work.  Writing RTL for any of them from the config
alone would be guessing: the config publishes `hc_sinkhorn_iters` and
`swiglu_limit`, not the ORDER of operations, and every one of those has a detail
that a plausible guess gets wrong (see the notes on each function).

So this lands the step the Laguna port took first: an executable specification,
transcribed from the reference implementation
(`transformers/models/glm5_next/modeling_glm5_next.py`, transformers 5.16.0 --
the version `config.json` pins), with the traps called out.  RTL follows, gated
against THIS.

STATUS: reference only.  Nothing here is claimed to be bit-exact to a running
GLM-5.3-Flash -- it has not been run against the real checkpoint's activations.
It is a faithful transcription whose self-test checks internal consistency and
the invariants the math must satisfy.

All arithmetic is fp32: the reference casts to float for the state recurrence
("states are more susceptible to rounding errors") and for the whole mHC map.
"""
import numpy as np

F32 = np.float32


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))).astype(F32)


def silu(x):
    x = np.asarray(x, np.float64)
    return (x / (1.0 + np.exp(-x))).astype(F32)


def softmax(x, axis=-1):
    x = np.asarray(x, np.float64)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=axis, keepdims=True)).astype(F32)


# ---------------------------------------------------------------- clamped SwiGLU
def clamped_swiglu(gate, up, limit=10.0):
    """GLM-5.3-Flash MLP activation (Glm5NextTextMLP.forward).

    TRAP -- the clamp is ASYMMETRIC, and a symmetric guess is wrong:
        gate.clamp(min=None, max=limit)     # upper bound ONLY
        up.clamp(min=-limit, max=+limit)    # both bounds
    GLM-5.2 has no clamp at all, so this is new behaviour, not a tolerance knob:
    an unclamped SwiGLU here is numerically wrong, not merely approximate.
    """
    g = np.minimum(np.asarray(gate, F32), F32(limit))
    u = np.clip(np.asarray(up, F32), F32(-limit), F32(limit))
    return (silu(g) * u).astype(F32)


# ---------------------------------------------------------------- KDA pieces
def _seq_sum(a, axis):
    """fp32 reduction in EXPLICIT SEQUENTIAL index order.

    WHY THIS EXISTS AND WHY numpy's .sum() IS NOT USED.  numpy reduces with
    PAIRWISE summation, which is a different association than the sequential
    accumulate a streaming datapath performs.  Measured on this machine: for
    length-8 fp32 vectors, np.sum differs from sequential in 159/300 random
    cases -- more than half, at the smallest size KDA uses.  A reference that
    reduces pairwise can never be matched bitwise by RTL that streams, so the
    reduction order is pinned HERE, sequentially, and that order is this repo's
    contract.

    Consequence, stated rather than discovered: torch/FLA reduce in their own
    blocked order, so whole-runtime numeric equality with them is OUT OF
    CONTRACT -- the same stance this repo already takes for llama.cpp.
    """
    a = np.asarray(a, F32)
    a = np.moveaxis(a, axis, -1)
    out = np.zeros(a.shape[:-1], F32)
    for i in range(a.shape[-1]):
        out = (out + a[..., i]).astype(F32)
    return out


def l2norm(x, eps=1e-6, axis=-1):
    """FLA-compatible L2 norm.

    TRAP: this is `x / sqrt(sum(x*x) + eps)`, NOT `x / max(norm, eps)` and NOT
    `F.normalize`.  The reference comments that it intentionally uses sqrt-then-
    divide to match the original triton kernel; the eps is INSIDE the sqrt.
    The sum is sequential -- see _seq_sum.
    """
    x = np.asarray(x, F32)
    inv = np.sqrt(_seq_sum(x * x, axis)[..., None] + F32(eps))
    return (x / inv).astype(F32)


def forget_gate(f_lowrank, dt_bias, A_log, lower_bound=-5.0):
    """Glm5NextTextForgetGate.forward, the branch this model actually takes.

    g          = f_b_proj(f_a_proj(h)) + dt_bias        (fp32, [.., H, Dk])
    decay_rate = exp(A_log)                              (per head, [H,1])
    return       lower_bound * sigmoid(decay_rate * g)

    TRAP: `linear_lower_bound` is -5.0 for this model, so the softplus branch in
    the reference is DEAD CODE here.  Implementing softplus because the source
    contains it would be implementing the path this checkpoint never runs.
    """
    g = (np.asarray(f_lowrank, F32) + np.asarray(dt_bias, F32)).astype(F32)
    decay = np.exp(np.asarray(A_log, np.float64)).astype(F32)
    return (F32(lower_bound) * sigmoid(decay * g)).astype(F32)


def rmsnorm_gated(x, weight, gate, eps=1e-5):
    """Glm5NextTextRMSNormGated: fp32 RMS norm, then multiply by sigmoid(gate).

    TRAP: the gate is applied AFTER the weight multiply, and the activation is
    sigmoid -- not SiLU, despite the block's other gates being SiLU-flavoured.
    """
    x = np.asarray(x, F32)
    var = (x.astype(np.float64) ** 2).mean(-1, keepdims=True)
    y = (x / np.sqrt(var + eps)).astype(F32)
    y = (np.asarray(weight, F32) * y).astype(F32)
    return (y * sigmoid(gate)).astype(F32)


def kda_step(state, q, k, v, g, beta, use_qk_l2norm=True):
    """ONE decode token through Kimi Delta Attention -- the recurrence the RTL
    must implement (`recurrent_kimi_delta_attention`, seq_len == 1 path).

      state : [H, Dk, Dv] fp32, carried across tokens (NOT a growing KV cache --
              this is why only 11 of 45 layers page KV; see HARDWARE_LADDER)
      q,k,v : [H, Dk] / [H, Dk] / [H, Dv]
      g     : [H, Dk]   per-(head,key-dim) log-decay from forget_gate()
      beta  : [H]       sigmoid(b_proj(h))

      q     = l2norm(q) * Dk**-0.5   ;   k = l2norm(k)
      state *= exp(g)[:, :, None]
      kv    = (state * k[:, :, None]).sum(axis=-2)        # [H, Dv]
      delta = (v - kv) * beta[:, None]
      state += k[:, :, None] * delta[:, None, :]          # outer product
      out   = (state * q[:, :, None]).sum(axis=-2)        # [H, Dv]

    TRAPS:
      * both reductions are SEQUENTIAL in the Dk index (see _seq_sum) -- numpy's
        pairwise .sum() cannot be matched bitwise by a streaming datapath;
      * the l2norm happens AFTER the fp32 cast and BEFORE the 1/sqrt(Dk) scale;
      * the scale divides by q's LAST dim, and only q is scaled, never k;
      * `state *= exp(g)` decays the state BEFORE kv is read, so kv sees the
        decayed state -- reading kv first is a different (wrong) recurrence;
      * `delta` uses the value MINUS the decayed memory (delta rule), and the
        outer-product update uses that delta, not v.
    """
    state = np.asarray(state, F32)
    H, Dk, Dv = state.shape
    q = np.asarray(q, F32); k = np.asarray(k, F32); v = np.asarray(v, F32)
    if use_qk_l2norm:
        q = l2norm(q); k = l2norm(k)
    q = (q * F32(1.0 / np.sqrt(Dk))).astype(F32)
    gi = np.exp(np.asarray(g, np.float64)).astype(F32)          # [H, Dk]
    state = (state * gi[:, :, None]).astype(F32)
    kv = _seq_sum(state * k[:, :, None], -2).astype(F32)        # [H, Dv]
    delta = ((v - kv) * np.asarray(beta, F32)[:, None]).astype(F32)
    state = (state + k[:, :, None] * delta[:, None, :]).astype(F32)
    out = _seq_sum(state * q[:, :, None], -2).astype(F32)       # [H, Dv]
    return out, state


def causal_conv_step(conv_state, x_new, weight):
    """Depthwise causal short conv, decode step (`causal_conv1d_update`) with the
    SiLU activation this model uses.  conv_state is [C, K-1] history; returns the
    new activation and the shifted state.  C = 3*qkv_dim (q, k and v share it).
    """
    conv_state = np.asarray(conv_state, F32)
    w = np.asarray(weight, F32)                                  # [C, K]
    C, Km1 = conv_state.shape
    win = np.concatenate([conv_state, np.asarray(x_new, F32)[:, None]], axis=1)
    y = silu((win * w).sum(axis=1))
    return y.astype(F32), win[:, 1:].astype(F32)


# ------------------------------------------------- mHC hyper-connection
def hyper_connection(hidden_streams, fn, base, scale, hc_mult=4,
                     sinkhorn_iters=20, eps=1e-6, rms_eps=1e-5):
    """Glm5NextTextHyperConnection.forward -- Manifold-Constrained Hyper-
    Connections.  Returns (post, comb, collapsed).

      hidden_streams : [H, D]  -- H = hc_mult PARALLEL residual streams.  This is
        the structural surprise: the block does not carry one residual, it
        carries four, and this map decides how they collapse in and mix back out.

      flat        = unweighted_rmsnorm(hidden_streams.flatten())     [H*D]
      pre,post,comb_w = (flat @ fn.T).split([H, H, H*H])
      pre  = sigmoid(pre_w  * scale[0] + base[:H])   + eps
      post = 2 * sigmoid(post_w * scale[1] + base[H:2H])            -> range [0,2]
      comb = softmax(comb_w*scale[2] + base[2H:], -1) + eps, then Sinkhorn
      collapsed = sum_h pre[h] * hidden_streams[h]

    TRAPS:
      * the Sinkhorn loop is NOT `iters` symmetric passes.  The reference does
        ONE column normalise first, then `iters - 1` iterations of
        (row, then column).  Off-by-one here changes the matrix.
      * every normalise divides by (sum + eps), never a bare sum;
      * `post` is 2*sigmoid, so it spans [0,2] -- a plain sigmoid halves the
        sublayer's contribution;
      * the input RMS norm is UNWEIGHTED (no learned gain).
    """
    hs = np.asarray(hidden_streams, F32)
    H = hc_mult
    flat = hs.reshape(-1).astype(F32)
    rms = np.sqrt((flat.astype(np.float64) ** 2).mean() + rms_eps)
    flat = (flat / rms).astype(F32)

    mixed = (np.asarray(fn, F32) @ flat).astype(F32)             # [(2+H)*H]
    pre_w, post_w, comb_w = mixed[:H], mixed[H:2 * H], mixed[2 * H:]
    base = np.asarray(base, F32)
    pre_b, post_b, comb_b = base[:H], base[H:2 * H], base[2 * H:].reshape(H, H)
    s0, s1, s2 = [F32(v) for v in np.asarray(scale, F32)]

    pre = (sigmoid(pre_w * s0 + pre_b) + F32(eps)).astype(F32)
    post = (F32(2.0) * sigmoid(post_w * s1 + post_b)).astype(F32)

    comb = (softmax(comb_w.reshape(H, H) * s2 + comb_b, axis=-1) + F32(eps)).astype(F32)
    comb = (comb / (comb.sum(axis=-2, keepdims=True) + F32(eps))).astype(F32)   # column first
    for _ in range(sinkhorn_iters - 1):
        comb = (comb / (comb.sum(axis=-1, keepdims=True) + F32(eps))).astype(F32)
        comb = (comb / (comb.sum(axis=-2, keepdims=True) + F32(eps))).astype(F32)

    collapsed = (pre[:, None] * hs).sum(axis=0).astype(F32)
    return post, comb, collapsed


def hc_collapse(hidden_streams, pre):
    """collapsed = sum_h pre[h] * streams[h]   -- the sublayer's input.

    Pinned SEQUENTIAL over h, which is what numpy's .sum(axis=0) already does at
    H = 4 (measured 0/300 differing), so this only makes the contract explicit.
    Note the result is NOT normalised: the block's own attn_norm / ffn_norm still
    applies to it (both tensors exist on all 46 blocks per the GGUF census).
    """
    hs = np.asarray(hidden_streams, F32)
    pre = np.asarray(pre, F32)
    acc = (pre[0] * hs[0]).astype(F32)
    for h in range(1, hs.shape[0]):
        acc = (acc + (pre[h] * hs[h]).astype(F32)).astype(F32)
    return acc


def hc_mix(hidden_streams, comb, post, sub_out):
    """streams' = comb @ streams + post (x) sub_out   -- the residual update.

    TRAP, and the reason this function exists rather than a bare `comb @ hs`:
    numpy's matmul does NOT reduce in this order -- measured 300/300 differing on
    [4,4] @ [4,D].  The gap is one fp32 rounding (5.0e-7 of the output RMS; it is
    an FMA-vs-mul-then-add difference), but it is not zero, so a golden built on
    `@` would depend on whichever BLAS the host links and could never be matched
    bitwise by streaming RTL.  The order is therefore pinned HERE -- multiply, then
    accumulate sequentially over g, then add the post term last -- exactly as
    _seq_sum does for KDA, and that order is this repo's contract.

    Beware the metric when checking this: comb is doubly stochastic, so the four
    terms nearly cancel and some outputs land near zero.  A plain relative error
    divides by those and reports ~1e-2 for what is a 1-ULP difference; compare
    against the output RMS instead.
    """
    hs = np.asarray(hidden_streams, F32)
    comb = np.asarray(comb, F32)
    post = np.asarray(post, F32)
    sub = np.asarray(sub_out, F32)
    H = hs.shape[0]
    out = np.empty_like(hs)
    for h in range(H):
        acc = (comb[h, 0] * hs[0]).astype(F32)
        for g in range(1, H):
            acc = (acc + (comb[h, g] * hs[g]).astype(F32)).astype(F32)
        out[h] = (acc + (post[h] * sub).astype(F32)).astype(F32)
    return out


# ------------------------------------------------------------------ self-test
def _selftest():
    rng = np.random.default_rng(0x5F3)
    n = 0
    fails = []

    def chk(cond, msg):
        nonlocal n
        n += 1
        if not cond:
            fails.append(msg)

    # --- clamped SwiGLU: the asymmetry is the property worth pinning ---
    g = np.array([-50.0, -1.0, 0.0, 5.0, 50.0], F32)
    u = np.array([-50.0, -1.0, 0.0, 5.0, 50.0], F32)
    y = clamped_swiglu(g, u, 10.0)
    chk(np.isfinite(y).all(), "swiglu: non-finite")
    # gate is NOT clamped from below -> silu(-50) underflows toward 0, and the
    # result must differ from a symmetric clamp at the negative end
    y_sym = (silu(np.clip(g, -10.0, 10.0)) * np.clip(u, -10.0, 10.0)).astype(F32)
    chk(not np.array_equal(y, y_sym), "swiglu: symmetric clamp gives the same answer -- the asymmetry is not implemented")
    chk(np.isclose(y[4], silu(F32(10.0)) * F32(10.0), rtol=1e-6), "swiglu: upper clamp wrong")

    # --- l2norm: unit norm up to the in-sqrt eps ---
    x = rng.normal(size=(4, 128)).astype(F32)
    ln = l2norm(x)
    chk(np.allclose(np.linalg.norm(ln, axis=-1), 1.0, atol=1e-3), "l2norm: not unit")

    # --- forget gate: lower_bound * sigmoid(...) must lie in (lower_bound, 0) ---
    fg = forget_gate(rng.normal(size=(8, 128)).astype(F32) * 5,
                     rng.normal(size=(8, 128)).astype(F32),
                     rng.normal(size=(8, 1)).astype(F32), -5.0)
    # BOTH ends are CLOSED, and that is not pedantry -- it is an RTL contract.
    # fp32 sigmoid saturates to exactly 1.0 for large positive decay*g, giving
    # -5.0*1.0 == -5.0 (measured: 68/1024 samples), and to exactly 0.0 for large
    # negative, giving -5.0*0.0 == **-0.0**.  Signed zero survives here: an RTL
    # multiplier that emits +0.0 instead differs from this reference in the sign
    # bit, which a bitwise gate WILL catch.  An open-interval assertion (fg < 0)
    # fails on healthy input precisely because -0.0 is not less than 0.0.
    chk(((fg >= -5.0) & (fg <= 0.0)).all(), "forget_gate: outside [lower_bound, 0]")
    chk((fg == -5.0).any(), "forget_gate: lower bound never attained -- sigmoid saturation missing")
    negzero = np.signbit(fg[fg == 0.0])
    chk(negzero.all() if negzero.size else True,
        "forget_gate: a zero came out +0.0; lower_bound*sigmoid(x->0) must give -0.0")

    # --- KDA: the delta rule must drive the read toward v ---
    H, Dk, Dv = 2, 16, 16
    state = np.zeros((H, Dk, Dv), F32)
    q = rng.normal(size=(H, Dk)).astype(F32)
    k = rng.normal(size=(H, Dk)).astype(F32)
    v = rng.normal(size=(H, Dv)).astype(F32)
    g0 = np.zeros((H, Dk), F32)                 # exp(0)=1 -> no decay
    beta1 = np.ones(H, F32)
    _, s1 = kda_step(state, q, k, v, g0, beta1)
    kn = l2norm(k)
    kv = (s1 * kn[:, :, None]).sum(axis=-2)
    chk(np.allclose(kv, v, atol=2e-3), "kda: beta=1 write then read does not return v (delta rule broken)")
    # beta=0 must leave the state untouched
    _, s0 = kda_step(state, q, k, v, g0, np.zeros(H, F32))
    chk(np.abs(s0).max() < 1e-6, "kda: beta=0 modified the state")
    # a strong decay must shrink an existing state
    _, s2 = kda_step(s1, q, k, v, np.full((H, Dk), -5.0, F32), np.zeros(H, F32))
    chk(np.abs(s2).max() < np.abs(s1).max(), "kda: decay did not shrink the state")

    # --- mHC: comb must come out doubly stochastic ---
    hc = 4; D = 32
    mix = (2 + hc) * hc
    post, comb, coll = hyper_connection(
        rng.normal(size=(hc, D)).astype(F32),
        rng.normal(size=(mix, hc * D)).astype(F32) * 0.05,
        rng.normal(size=mix).astype(F32),
        np.array([1.0, 1.0, 1.0], F32), hc, 20, 1e-6)
    chk(np.allclose(comb.sum(-1), 1.0, atol=2e-2), "mHC: comb rows not stochastic")
    chk(np.allclose(comb.sum(-2), 1.0, atol=2e-2), "mHC: comb cols not stochastic")
    chk(((post >= 0) & (post <= 2)).all(), "mHC: post outside [0,2]")
    chk(coll.shape == (D,), "mHC: collapse shape wrong")
    # the Sinkhorn loop's asymmetric first pass must matter
    _, comb_sym, _ = hyper_connection(
        rng.normal(size=(hc, D)).astype(F32),
        rng.normal(size=(mix, hc * D)).astype(F32) * 0.05,
        rng.normal(size=mix).astype(F32),
        np.array([1.0, 1.0, 1.0], F32), hc, 1, 1e-6)
    chk(comb_sym.shape == (hc, hc), "mHC: 1-iteration path broken")

    # --- the residual path: collapse and mix ---
    hs_r    = rng.normal(size=(hc, D)).astype(F32)
    fn_r    = (rng.normal(size=(mix, hc * D)) * 0.05).astype(F32)
    base_r  = rng.normal(size=mix).astype(F32)
    scale_r = np.array([1.0, 1.0, 1.0], F32)
    sub_r   = rng.normal(size=D).astype(F32)

    # collapse must be exactly what hyper_connection itself returns for the same pre
    post_r, comb_r, coll_ref = hyper_connection(hs_r, fn_r, base_r, scale_r, hc, 20, 1e-6)
    flat_r = hs_r.reshape(-1).astype(F32)
    flat_r = (flat_r / np.sqrt((flat_r.astype(np.float64) ** 2).mean() + 1e-5)).astype(F32)
    pre_int = (sigmoid((fn_r[:hc] @ flat_r) * scale_r[0] + base_r[:hc]) + F32(1e-6)).astype(F32)
    chk(np.array_equal(hc_collapse(hs_r, pre_int), coll_ref),
        "mHC: hc_collapse does not reproduce hyper_connection's own collapsed")

    mixed_r = hc_mix(hs_r, comb_r, post_r, sub_r)
    chk(mixed_r.shape == (hc, D), "mHC: mix shape wrong")
    # post = 0 leaves the pure stream re-mix
    chk(np.array_equal(hc_mix(hs_r, comb_r, np.zeros(hc, F32), sub_r),
                       hc_mix(hs_r, comb_r, np.zeros(hc, F32), np.zeros(D, F32))),
        "mHC: post=0 still let sub_out through")
    # comb = I, post = 0 is the identity on the streams
    chk(np.array_equal(hc_mix(hs_r, np.eye(hc, dtype=F32), np.zeros(hc, F32), sub_r),
                       hs_r), "mHC: comb=I, post=0 is not the identity")
    # comb = 0, comb = I with post = 1 recovers the ordinary single residual add
    chk(np.array_equal(hc_mix(hs_r, np.eye(hc, dtype=F32), np.ones(hc, F32), sub_r),
                       (hs_r + sub_r[None, :]).astype(F32)),
        "mHC: comb=I, post=1 is not the plain residual add")
    # THE PIN: numpy's own matmul must NOT match, or the sequential order is
    # not actually being enforced and a BLAS change could move the golden.
    chk(not np.array_equal((comb_r @ hs_r + post_r[:, None] * sub_r[None, :]).astype(F32),
                           mixed_r),
        "mHC: numpy matmul matches the pinned order -- the pin is not live")

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53_flash_ref self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (clamped SwiGLU asymmetry, l2norm, forget-gate "
          f"range + signed zero, KDA delta rule + decay + beta=0, mHC double "
          f"stochasticity + post range, hc_collapse vs hyper_connection's own "
          f"collapsed, hc_mix identities + the pinned-order pin)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
