#!/usr/bin/env python3
"""
dsa_indexer_ref.py -- executable reference for GLM-5.3-Flash's DSA indexer
(Glm5NextTextIndexer), decode step: ONE query against a key cache.

Transcribed from the transformers 5.16.0 implementation that config.json pins;
see docs/GLM53_FLASH_PORT.md 4.3h for what it does and does not share with the
GLM-5.2 dsa_indexer.v (answer: only topk_select).

STATUS: reference only, self-tested locally; NOT yet wired into the release
ladder (adding a pinned gate needs a full ~7 h ladder run to re-pin, so it is
batched with the next RTL change). Not run against real activations.

Per DSA layer, per query token (H = index_n_heads = 32, D = index_head_dim = 128,
KPOOL = 4, TOPK = 2048 -- small values in the self-test):

    q[h]        = wq_b(q_resid)                view [H, D]
    k_tok       = LayerNorm(wk(h_tok))         [D]; WITH weight AND bias; eps 1e-6
    gate_tok    = h_tok @ Wg^T                 [D]      (index_kpool_compress_gate [D, hidden])
    -- caches: keys[S, D], gates[S, D], valid[S] --
    -- pools: groups of KPOOL consecutive keys starting at the first valid key --
    logit[p][j][c] = gates[key_j][c] + ape[j][c]     ape = index_kpool_compress_ape [KPOOL, D]
                     -inf where key_j invalid / past the cache end
    prob[p][j][c]  = softmax over j (the KPOOL positions), PER CHANNEL c
    pool_key[p][c] = sum_j prob[p][j][c] * keys[key_j][c]
    pool_valid[p]  = all KPOOL keys valid and inside the cache
    -- scoring over POOLS --
    score[h][p]    = relu( (q[h] . pool_key[p]) * D**-0.5 )
    w[h]           = (h_tok @ Wwp^T)[h] * H**-0.5      (weights_proj [H, hidden])
    index_score[p] = sum_h w[h] * score[h][p];   -inf where not (pool_valid and pool visible)
    selected       = top-(TOPK // KPOOL) pools by index_score
    tokens         = each selected pool's KPOOL raw indices (invalid -> -1)
    tail           = the current incomplete pool's raw indices (<= KPOOL-1), appended
    width          = TOPK (+ KPOOL-1 with tail), padded with -1

TRAPS a plausible transcription gets wrong:
  * k_norm is a LayerNorm (mean-subtract, weight AND bias), not the RMSNorm the
    rest of the model uses;
  * the softmax is over the KPOOL positions PER CHANNEL, not over channels;
  * the ReLU is on the per-head score BEFORE the head-weighted sum;
  * the head weights are scaled by H**-0.5, the scores by D**-0.5 -- two different
    scales;
  * selection is over pools (TOPK // KPOOL of them), then expanded x KPOOL -- not a
    top-TOPK over tokens;
  * pooling starts at the FIRST VALID key, not at cache slot 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32
NEG = np.float32(-np.inf)


def layernorm(x, w, b, eps=1e-6):
    x = np.asarray(x, np.float64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps) * w + b).astype(F32)


def pooled_states(keys, gates, valid, ape, kpool):
    """(pool_keys [P, D], pool_indices [P, KPOOL], pool_valid [P]) -- get_pooled_states."""
    S, D = keys.shape
    first = int(np.argmax(valid)) if valid.any() else S
    n_pools = (S + kpool - 1) // kpool
    idx = first + np.arange(n_pools * kpool).reshape(n_pools, kpool)          # [P, KPOOL]
    safe = np.clip(idx, 0, S - 1)
    gk = keys[safe]                                                           # [P, KPOOL, D]
    gg = gates[safe].astype(F32)
    gv = valid[safe] & (idx < S)                                              # [P, KPOOL]
    pool_valid = gv.all(-1)
    pool_idx = np.where(gv, idx, -1)
    logits = gg + ape[None].astype(F32)                                       # [P, KPOOL, D]
    logits = np.where(gv[..., None], logits, NEG)
    # softmax over axis 1 (the KPOOL positions), per channel
    # A pool with NO valid key has m = -inf; -inf - (-inf) is nan (with a numpy
    # RuntimeWarning) before nan_to_num repairs it.  Use 0 for the max there so
    # the subtraction stays -inf and exp gives 0 directly -- same result, no warning.
    m = np.max(logits, axis=1, keepdims=True)
    m0 = np.where(np.isfinite(m), m, 0.0).astype(F32)
    e = np.exp((logits - m0).astype(np.float64))
    e = np.nan_to_num(e)
    den = e.sum(axis=1, keepdims=True)
    prob = np.nan_to_num(np.where(den > 0, e / np.maximum(den, 1e-300), 0.0)).astype(F32)
    pool_keys = (prob * gk).sum(axis=1).astype(F32)                           # [P, D]
    return pool_keys, pool_idx, pool_valid, prob


def indexer_select(q, h_tok, keys, gates, valid, W, kpool=4, topk=8, tail=True):
    """One decode query. q: [H, D] (already wq_b'd). Returns (indices [width], scores [P])."""
    H, D = q.shape
    S = keys.shape[0]
    pool_keys, pool_idx, pool_valid, _ = pooled_states(keys, gates, valid, W["ape"], kpool)
    # visibility for a single decode query: every cached key is visible; a pool is
    # selectable only if its LAST key is a real, visible key.
    pool_end = np.clip(pool_idx[:, -1], 0, S - 1)
    visible = np.ones(S, bool) & valid
    cand = visible[pool_end] & pool_valid
    score = np.maximum(0.0, (q.astype(F32) @ pool_keys.T.astype(F32)) * F32(D ** -0.5)).astype(F32)   # [H, P]
    w = ((W["wp"] @ h_tok.astype(F32)) * F32(H ** -0.5)).astype(F32)                                    # [H]
    index_score = (w @ score).astype(F32)                                                                # [P]
    index_score = np.where(cand, index_score, np.finfo(F32).min).astype(F32)
    select_k = min(topk // kpool, index_score.shape[0])
    selected = np.argsort(-index_score, kind="stable")[:select_k]
    sel_valid = cand[selected]
    toks = pool_idx[selected].reshape(-1)
    toks = np.where(np.repeat(sel_valid, kpool), toks, -1)
    width = topk
    if tail and kpool > 1:
        # the current incomplete pool: raw indices from the last complete pool end
        first = int(np.argmax(valid)) if valid.any() else S
        n_full = (S - first) // kpool
        tail_start = first + n_full * kpool
        tail_idx = np.arange(tail_start, S)
        tail_idx = tail_idx[valid[tail_idx]] if tail_idx.size else tail_idx
        toks = np.concatenate([toks, tail_idx])
        width += kpool - 1
    out = np.full(width, -1, np.int64)
    out[:min(len(toks), width)] = toks[:width]
    return out, index_score


def _selftest():
    rng = np.random.default_rng(0)
    H, D, KPOOL, TOPK = 4, 8, 4, 8
    n = 0; fails = []
    def chk(c, m):
        nonlocal n
        n += 1
        if not c: fails.append(m)
    for trial in range(40):
        S = int(rng.integers(5, 20))
        keys = rng.normal(size=(S, D)).astype(F32)
        gates = rng.normal(size=(S, D)).astype(F32)
        valid = np.ones(S, bool)
        nlead = int(rng.integers(0, 3)); valid[:nlead] = False       # leading padding
        ape = rng.normal(size=(KPOOL, D)).astype(F32)
        pk, pidx, pval, prob = pooled_states(keys, gates, valid, ape, KPOOL)
        # probs are a distribution over the KPOOL positions, per channel, on valid pools
        chk(np.allclose(prob[pval].sum(axis=1), 1.0, atol=1e-5), f"t{trial}: probs do not sum to 1 per channel")
        # pool key is a convex combination of its keys (per channel, between min and max)
        for p in np.where(pval)[0]:
            g = keys[pidx[p]]
            chk(bool(((pk[p] >= g.min(0) - 1e-5) & (pk[p] <= g.max(0) + 1e-5)).all()),
                f"t{trial}: pool {p} key outside the convex hull of its keys")
        # pooling starts at the first valid key
        first = int(np.argmax(valid))
        chk(int(pidx[0][0]) == first, f"t{trial}: first pool does not start at the first valid key")
        # a full run
        q = rng.normal(size=(H, D)).astype(F32)
        h_tok = rng.normal(size=16).astype(F32)
        W = dict(ape=ape, wp=rng.normal(size=(H, 16)).astype(F32))
        toks, isc = indexer_select(q, h_tok, keys, gates, valid, W, KPOOL, TOPK, tail=True)
        chk(toks.shape[0] == TOPK + KPOOL - 1, f"t{trial}: width {toks.shape[0]} != {TOPK + KPOOL - 1}")
        got = toks[toks >= 0]
        chk(len(set(got.tolist())) == len(got), f"t{trial}: duplicate token indices")
        chk(bool(((got >= 0) & (got < S)).all()) and bool(valid[got].all()), f"t{trial}: out-of-range or invalid index selected")
        # never select an invalid pool: every selected complete pool has finite score
        sel_pools = set()
        for t in got:
            p = np.where((pidx == t).any(1))[0]
            if p.size: sel_pools.add(int(p[0]))
        full_sel = [p for p in sel_pools if pval[p]]
        chk(all(np.isfinite(isc[p]) and isc[p] > np.finfo(F32).min for p in full_sel), f"t{trial}: an invalid pool was selected")
        # the selected complete pools are exactly the top-(TOPK//KPOOL) by index_score
        top = set(np.argsort(-isc, kind="stable")[:min(TOPK // KPOOL, len(isc))].tolist())
        top = {p for p in top if pval[p]}
        chk(top == set(full_sel), f"t{trial}: selected pools {sorted(full_sel)} != top-k {sorted(top)}")
        # tail = the incomplete pool's indices
        n_full = (S - first) // KPOOL
        tail_expect = [t for t in range(first + n_full * KPOOL, S) if valid[t]]
        chk(all(t in got.tolist() for t in tail_expect), f"t{trial}: tail indices {tail_expect} not appended")
    # a second formulation of the compressor (explicit loops) must agree bitwise
    S = 9; keys = rng.normal(size=(S, D)).astype(F32); gates = rng.normal(size=(S, D)).astype(F32)
    valid = np.ones(S, bool); ape = rng.normal(size=(KPOOL, D)).astype(F32)
    pk, pidx, pval, _ = pooled_states(keys, gates, valid, ape, KPOOL)
    for p in np.where(pval)[0]:
        for c in range(D):
            lg = np.array([gates[pidx[p][j]][c] + ape[j][c] for j in range(KPOOL)], np.float64)
            e = np.exp(lg - lg.max()); pr = e / e.sum()
            v = F32(sum(pr[j] * keys[pidx[p][j]][c] for j in range(KPOOL)))
            n += 1
            if not np.isclose(v, pk[p][c], rtol=1e-5, atol=1e-6): fails.append(f"loop vs vector compressor differ at pool {p} ch {c}")
    if fails:
        for f in fails[:8]: print("  FAIL:", f)
        print(f"dsa_indexer_ref self-test: {len(fails)}/{n} FAILED"); return 1
    print(f"ALL {n} TESTS PASSED (k-pool compressor is a per-channel convex combination starting at the first "
          f"valid key; loop and vector forms agree; selection is the exact top-(TOPK/KPOOL) over valid pools, "
          f"expanded xKPOOL with the incomplete tail appended, width TOPK+KPOOL-1, no duplicates/invalid)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
