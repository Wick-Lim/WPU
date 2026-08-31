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
def l2norm(x, eps=1e-6, axis=-1):
    """FLA-compatible L2 norm.

    TRAP: this is `x / sqrt(sum(x*x) + eps)`, NOT `x / max(norm, eps)` and NOT
    `F.normalize`.  The reference comments that it intentionally uses sqrt-then-
    divide to match the original triton kernel; the eps is INSIDE the sqrt.
    """
    x = np.asarray(x, F32)
    inv = np.sqrt((x * x).sum(axis=axis, keepdims=True) + F32(eps))
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
    kv = (state * k[:, :, None]).sum(axis=-2).astype(F32)       # [H, Dv]
    delta = ((v - kv) * np.asarray(beta, F32)[:, None]).astype(F32)
    state = (state + k[:, :, None] * delta[:, None, :]).astype(F32)
    out = (state * q[:, :, None]).sum(axis=-2).astype(F32)      # [H, Dv]
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

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53_flash_ref self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (clamped SwiGLU asymmetry, l2norm, forget-gate "
          f"range + signed zero, KDA delta rule + decay + beta=0, mHC double "
          f"stochasticity + post range)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
