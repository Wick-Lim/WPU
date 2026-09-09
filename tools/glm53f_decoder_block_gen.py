#!/usr/bin/env python3
"""
glm53f_decoder_block_gen.py -- vectors for test/glm53f_decoder_block_tb.v
(src/glm53f_decoder_block.v: KDA attention wired INSIDE a two-site mHC block).

This is the first golden that composes both machines. `make hc-block` checks the
two mHC sites with stub sublayers; `make kda-attn` checks the KDA sublayer on its
own. This checks that putting the second inside the first still lands -- the
attention site's collapse -> attn_norm -> KDA -> mix round trip, with the KDA
recurrence and conv history advancing across it.

The FFN site keeps the stub (0.5 * normed): GLM-5.3-Flash's dense FFN is Q8_0 and
its MoE experts are a Q4_K/Q5_K/Q6_K mix, so swiglu_expert_q4k -- 4 bits per lane
-- cannot carry either. Wiring it anyway to make the block look finished is the
silent-wrong-weights failure this repo builds must-fail pairs against.
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

F32 = np.float32


def f32b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0])


def bf16b(v):
    return int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16


def fp16b(v):
    return int(np.frombuffer(np.float16(v).tobytes(), np.uint16)[0])


def block(streams, hc, kda, H, MD, KH, DK, DV):
    """One decode step through the block: mHC attn site (KDA inside), then FFN."""
    s = streams
    normeds = []
    state, hist = kda["state"], kda["hist"]
    for site in ("attn", "ffn"):
        q, d, base, scale, gam = hc[site]
        mixed = dut_emulate(s.reshape(-1), q, d)
        pre, post, comb = ref_map(mixed, base, scale, H, 20)
        coll = ref.hc_collapse(s, pre)
        normed = rmsnorm_bf16(bf16(coll), gam)
        normeds.append(normed)
        if site == "attn":
            y, state, hist, _ = kda_layer(normed, kda["W"], kda["cw"], kda["dtb"],
                                          kda["a_log"], kda["onw"], state, hist, KH, DK, DV)
            sub = bf16(y)
        else:
            sub = bf16((normed * F32(0.5)).astype(F32))
        s = ref.hc_mix(s, comb, post, sub.astype(F32))
    return normeds, s, state, hist


def _draw(rng, MD, H, KH, DK, DV, RANK, CK):
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
    streams = (rng.normal(size=(H, MD)) * rng.choice([0.3, 1.0, 3.0])).astype(F32)
    return hc, kda, streams


def gen(ntest, MD=16, H=4, KH=2, DK=4, DV=4, RANK=4, CK=4, seed=0, report=True):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {MD} {H} {KH} {DK} {DV} {RANK} {CK}"]
    moved_s = moved_h = 0
    for _ in range(ntest):
        hc, kda, streams = _draw(rng, MD, H, KH, DK, DV, RANK, CK)
        normeds, final, st2, hi2 = block(streams, hc, kda, H, MD, KH, DK, DV)
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
        out.append(" ".join(f"{f32b(v):08x}" for v in final.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in st2.reshape(-1)))
        out.append(" ".join(f"{f32b(v):08x}" for v in hi2.reshape(-1)))
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
        hc, kda, streams = _draw(rng, MD, H, KH, DK, DV, RANK, CK)
        normeds, final, st2, hi2 = block(streams, hc, kda, H, MD, KH, DK, DV)
        n += 3
        if final.shape != (H, MD):
            fails.append("streams shape")
        if np.array_equal(st2, kda["state"]):
            fails.append("KDA state did not advance inside the block")
        if np.array_equal(hi2, kda["hist"]):
            fails.append("conv history did not advance inside the block")

        # swapping which sublayer serves which site must change the result
        hc_sw = {"attn": hc["ffn"], "ffn": hc["attn"]}
        if not np.array_equal(block(streams, hc_sw, kda, H, MD, KH, DK, DV)[1], final):
            live["site_swap"] += 1
        # replacing KDA with the stub must change it too, or KDA is not reaching
        # the residual stream at all
        s = streams
        for site in ("attn", "ffn"):
            q, d, base, scale, gam = hc[site]
            mixed = dut_emulate(s.reshape(-1), q, d)
            pre, post, comb = ref_map(mixed, base, scale, H, 20)
            coll = ref.hc_collapse(s, pre)
            nm_ = rmsnorm_bf16(bf16(coll), gam)
            s = ref.hc_mix(s, comb, post, bf16((nm_ * F32(0.5)).astype(F32)).astype(F32))
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
