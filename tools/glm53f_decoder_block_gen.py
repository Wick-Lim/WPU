#!/usr/bin/env python3
"""
glm53f_decoder_block_gen.py -- vectors for test/glm53f_decoder_block_tb.v
(src/glm53f_decoder_block.v: KDA attention wired INSIDE a two-site mHC block).

This is the first golden that composes both machines. `make hc-block` checks the
two mHC sites with stub sublayers; `make kda-attn` checks the KDA sublayer on its
own. This checks that putting the second inside the first still lands -- the
attention site's collapse -> attn_norm -> KDA -> mix round trip, with the KDA
recurrence and conv history advancing across it.

BOTH sites now carry a real sublayer, so this is a COMPLETE decoder layer for
blocks 0-2 (the dense front, and KDA since the first MLA block is 3): KDA in the
attention site, glm53f_swiglu_q8 -- clamped SwiGLU over Q8_0 -- in the FFN site.
swiglu_expert_q4k could not have been used: its w_q port is four bits per lane and
this FFN is Q8_0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402
from mhc_map_gen import ref_map  # noqa: E402
from mhc_fn_gemv_gen import q80_pack, dut_emulate  # noqa: E402
from mhc_precision_study import bf16  # noqa: E402
from glm53f_hc_block_gen import rmsnorm_bf16  # noqa: E402
from glm53f_kda_layer_gen import layer as kda_layer, _draw as kda_draw, ORDER  # noqa: E402
from glm53f_kda_attn_gen import q80_rows  # noqa: E402
from glm53f_swiglu_q8_gen import ffn as swiglu_ffn  # noqa: E402

F32 = np.float32


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def bf16b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16


def fp16b(v):
    return int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0])


def nudge(x, n):
    x = np.asarray(x, F32)
    b = np.frombuffer(np.ascontiguousarray(x).tobytes(), np.int32).astype(np.int64)
    return np.frombuffer(((b + n) & 0xFFFFFFFF).astype(np.uint32).tobytes(),
                         F32).reshape(x.shape)


MAP_ULP = {"pre": 1024, "post": 1024, "comb": 16384}
KDA_ABS_ENV = 0.03      # `make kda-attn` gates the attention sublayer at abs 0.03


def block(streams, hc, kda, ffnw, H, MD, KH, DK, DV, pert=None, sub_pert=0.0):
    """One decode step through the block: mHC attn site (KDA inside), then FFN."""
    s = streams
    normeds = []
    # Both sublayers' OWN published bounds, each scaled by the `post` that places
    # it into the streams. The attention site's term matters even though it is the
    # smaller one: KDA's error enters at the first site and then rides through
    # `comb @ streams` at the second, so a bound that only carries the FFN's is
    # short by exactly that much -- which is what the first two failures were.
    KDA_ABS = 0.03          # `make kda-attn` gates the sublayer at rel 6% + abs 0.03
    stol = None
    state, hist = kda["state"], kda["hist"]
    for site in ("attn", "ffn"):
        q, d, base, scale, gam = hc[site]
        mixed = dut_emulate(s.reshape(-1), q, d)
        pre, post, comb = ref_map(mixed, base, scale, H, 20)
        if pert is not None:
            pre = nudge(pre, pert * MAP_ULP["pre"])
            post = nudge(post, pert * MAP_ULP["post"])
            comb = nudge(comb, pert * MAP_ULP["comb"])
        coll = ref.hc_collapse(s, pre)
        normed = rmsnorm_bf16(bf16(coll), gam)
        normeds.append(normed)
        if site == "attn":
            y, state, hist, _ = kda_layer(normed, kda["W"], kda["cw"], kda["dtb"],
                                          kda["a_log"], kda["onw"], state, hist, KH, DK, DV)
            sub = bf16(y)
            # The attention sublayer's OWN gated bound, injected HERE so the
            # envelope sees it amplified by everything downstream -- the mix, the
            # next collapse, the norm, and the FFN -- rather than merely added at
            # the end. Measured, that path has a gain of ~29x, so a sum of the
            # parts' bounds is not a bound on the composition.
            if sub_pert:
                sub = bf16((np.asarray(sub, F32) + F32(sub_pert)).astype(F32))
            # `comb @ streams` at the NEXT site mixes every input row into every
            # output row, and comb's rows sum to ~1, so row h inherits at most
            # max_g(post[g]) * KDA_ABS -- not post[h] * KDA_ABS. Using the per-row
            # value under-counts, which is what the last failure was.
            stol = np.full((post.shape[0], y.shape[0]),
                           float(np.abs(post).max()) * KDA_ABS, np.float64)
        else:
            yf, act, _ = swiglu_ffn(normed, ffnw["Wg"], ffnw["Wu"], ffnw["Wd"])
            sub = bf16(yf)
            # The FFN's own gate does not claim better than this per element, and
            # the mix scales it by post (range [0,2]). Carrying that through is
            # what keeps the block's bound derived from its sublayers rather than
            # picked: a flat tolerance here failed on exactly the elements where
            # the DOWN reduction is large, which is where 0.02*mag lives.
            ftol = np.array([max(0.06 * abs(float(yf[o])),
                                 0.02 * float(np.sum(np.abs(act) * np.abs(ffnw["Wd"][o]))),
                                 0.03) for o in range(yf.shape[0])], np.float64)
            stol = stol + (np.abs(np.asarray(post, np.float64))[:, None] * ftol[None, :])
        s = ref.hc_mix(s, comb, post, sub.astype(F32))
    return normeds, s, state, hist, stol


def _draw(rng, MD, H, KH, DK, DV, RANK, CK, INTER=32):
    MIXN = (2 + H) * H
    hc = {}
    for site in ("attn", "ffn"):
        fn = (rng.normal(size=(MIXN, H * MD)) * (1.0 / np.sqrt(H * MD))).astype(F32)
        q, d = q80_pack(fn)
        base = rng.normal(size=MIXN).astype(F32)
        scale = (F32(1.0) + rng.normal(size=3).astype(F32) * F32(0.3)).astype(F32)
        gam = bf16((F32(1.0) + rng.normal(size=MD).astype(F32) * F32(0.3)).astype(F32))
        hc[site] = (q, d, base, scale, gam)
    x, W, cw, dtb, a_log, onw, state, hist = kda_draw(rng, MD, KH, DK, DV, RANK, CK)
    codes, scales, deq = {}, {}, {}
    for nm in ORDER:
        cq, cd, dq = q80_rows(W[nm])
        codes[nm], scales[nm], deq[nm] = cq, cd, dq
    kda = dict(W=deq, codes=codes, scales=scales, cw=cw, dtb=dtb, a_log=a_log,
               onw=onw, state=state, hist=hist)
    # dense FFN, Q8_0. Scales chosen so the SwiGLU clamp actually fires -- an FFN
    # corpus that never reaches +/-10 would not exercise the thing that makes this
    # FFN GLM-5.3-Flash's rather than GLM-5.2's.
    Wg = (rng.normal(size=(INTER, MD)) * 1.1).astype(F32)
    Wu = (rng.normal(size=(INTER, MD)) * 1.1).astype(F32)
    Wd = (rng.normal(size=(MD, INTER)) * 0.3).astype(F32)
    fq = {}
    ffnw = {}
    for nm, W in (("Wg", Wg), ("Wu", Wu), ("Wd", Wd)):
        cq, cd, dq = q80_rows(W)
        fq[nm] = (cq, cd)
        ffnw[nm] = dq
    ffnw["codes"] = fq
    streams = (rng.normal(size=(H, MD)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
    return hc, kda, ffnw, streams


def gen(ntest, MD=16, H=4, KH=2, DK=4, DV=4, RANK=4, CK=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {MD} {H} {KH} {DK} {DV} {RANK} {CK}"]
    moved_s = moved_h = 0
    for _ in range(ntest):
        hc, kda, ffnw, streams = _draw(rng, MD, H, KH, DK, DV, RANK, CK)
        normeds, final, st2, hi2, stol = block(streams, hc, kda, ffnw, H, MD, KH, DK, DV)
        # ENVELOPE, not a sum of the parts' bounds.  Perturbing pre/post/comb by
        # exactly mhc_map_step's own gated ULP bounds and re-running the WHOLE
        # block captures something a sum cannot: the dense FFN AMPLIFIES its input
        # error.  Measured here, a 1.2 % change in the normed vector moved an FFN
        # output by 29x that -- silu and a 32-term DOWN reduction do that -- so a
        # composed block is NOT as tight as its sublayers' bounds added up.
        env = np.zeros_like(final)
        for sg in (+1, -1):
            for sp in (0.0, sg * KDA_ABS_ENV):
                _, f2, _, _, _ = block(streams, hc, kda, ffnw, H, MD, KH, DK, DV,
                                       pert=sg, sub_pert=sp)
                env = np.maximum(env, np.abs(f2 - final))
        stol = np.maximum(stol, env.astype(np.float64))
        moved_s += 0 if np.array_equal(st2, kda["state"]) else 1
        moved_h += 0 if np.array_equal(hi2, kda["hist"]) else 1
        out.append(" ".join(f"{f32b(v):08x}" for v in streams.reshape(-1)))
        for site in ("attn", "ffn"):
            q, d, base, scale, gam = hc[site]
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in q.reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(d).reshape(-1)))
            out.append(" ".join(f"{f32b(v):08x}" for v in base))
            out.append(" ".join(f"{f32b(v):08x}" for v in scale))
            out.append(" ".join(f"{bf16b(v):04x}" for v in gam))
        for nm in ORDER:
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in kda["codes"][nm].reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(kda["scales"][nm]).reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in np.exp(kda["a_log"].astype(np.float64)).astype(F32)))
        out.append(" ".join(f"{f32b(v):08x}" for v in kda["dtb"]))
        out.append(" ".join(f"{f32b(v):08x}" for v in kda["cw"].reshape(-1)))
        out.append(" ".join(f"{bf16b(v):04x}" for v in kda["onw"]))
        out.append(" ".join(f"{f32b(v):08x}" for v in kda["state"].reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in kda["hist"].reshape(-1)))
        for nm in ("Wg", "Wu", "Wd"):
            cq, cd = ffnw["codes"][nm]
            out.append(" ".join(f"{int(np.uint8(v)):02x}" for v in cq.reshape(-1)))
            out.append(" ".join(f"{fp16b(v):04x}" for v in np.asarray(cd).reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in final.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in st2.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in hi2.reshape(-1)))
        out.append(" ".join(f"{t:.6f}" for t in stol.reshape(-1)))
    if report:
        print(f"the block advances the KDA state in {moved_s}/{ntest} and the conv "
              f"history in {moved_h}/{ntest} steps", file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    rng = np.random.default_rng(0xDB)
    MD, H, KH, DK, DV, RANK, CK = 16, 4, 2, 4, 4, 4, 4
    n = 0
    fails = []
    live = {k: 0 for k in ("site_swap", "no_kda")}
    trials = 6
    for _ in range(trials):
        hc, kda, ffnw, streams = _draw(rng, MD, H, KH, DK, DV, RANK, CK)
        normeds, final, st2, hi2, _ = block(streams, hc, kda, ffnw, H, MD, KH, DK, DV)
        n += 3
        if final.shape != (H, MD):
            fails.append("streams shape")
        if np.array_equal(st2, kda["state"]):
            fails.append("KDA state did not advance inside the block")
        if np.array_equal(hi2, kda["hist"]):
            fails.append("conv history did not advance inside the block")

        # swapping which sublayer serves which site must change the result
        hc_sw = {"attn": hc["ffn"], "ffn": hc["attn"]}
        if not np.array_equal(block(streams, hc_sw, kda, ffnw, H, MD, KH, DK, DV)[1], final):
            live["site_swap"] += 1
        # running the FFN in BOTH sites must change it, or KDA is not reaching the
        # residual stream at all
        s = streams
        for site in ("attn", "ffn"):
            q, d, base, scale, gam = hc[site]
            mixed = dut_emulate(s.reshape(-1), q, d)
            pre, post, comb = ref_map(mixed, base, scale, H, 20)
            coll = ref.hc_collapse(s, pre)
            nm_ = rmsnorm_bf16(bf16(coll), gam)
            yf, _, _ = swiglu_ffn(nm_, ffnw["Wg"], ffnw["Wu"], ffnw["Wd"])
            s = ref.hc_mix(s, comb, post, bf16(yf).astype(F32))
        if not np.array_equal(s, final):
            live["no_kda"] += 1
    for k, v in live.items():
        n += 1
        if v == 0:
            fails.append(f"trap '{k}' never changed the result -- injection is dead")
    if fails:
        print(f"glm53f_decoder_block_gen self-test: {len(fails)}/{n} FAILED")
        for f in sorted(set(fails))[:8]:
            print("   " + f)
        return 1
    print(f"ALL {n} TESTS PASSED (both KDA state pieces advance inside the block; "
          f"trap liveness over {trials} draws: " +
          ", ".join(f"{k} {v}/{trials}" for k, v in live.items()) + ")")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 4
    outp = a[1] if len(a) > 1 else "build/glm53f_decoder_block_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen(ntest))
    print(f"wrote {outp}: {ntest} tests")
