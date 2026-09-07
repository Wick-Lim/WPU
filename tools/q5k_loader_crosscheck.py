#!/usr/bin/env python3
"""q5k_loader_crosscheck.py -- emit a PACKER-BUILT Q5_K image for the RTL loader.

WHY THIS EXISTS
  glm_matmul_q4k has consumed Q5_K since the GEMM arm landed (`make mixedtype`,
  bit-exact vs tools/q4k_ref.py). What did not exist was a way to GET a Q5_K tile
  to it: weight_loader_q4k `$fatal`ed on a Q5_K descriptor because nothing had
  ever laid one out, and streaming Q4_K geometry instead would have been "same
  widths, wrong bytes, no error".

  Reading the two sides settled it: a Q5_K tile needs NOTHING new from the loader.
  Its header fields ARE Q4_K's (d, dmin, 12-byte packed scales), so the loader's
  Q4_K default branch decodes it correctly and `ns = nsblk*PE_N` is unchanged; and
  its code rides mm_w_hp, the same 16-bit-per-column lane Q6_K and Q8_0 already
  use. The only thing missing was the PACKER emitting that layout -- which
  ckpt_pack_q4k.pack_q5k_weight now does.

  This is the cross-TOOL gate for that claim: the file the packer writes is the
  file the RTL reads. Expectations come from the SOURCE arrays before packing, so
  they are independent of the layout under test.

  --inj-noqh packs the codes with the fifth bit DROPPED while leaving the
  expectations at their true 5-bit values. That is the whole point of the exercise
  -- a Q5_K image that is byte-plausible but is really Q4_K data -- and the gate
  must catch it.

GEOMETRY: N=8 (2 tiles at PE_N=4), K=768 (nb=3). nb>1 matters for the same reason
it does in packer_rtl_crosscheck: the col-outer/sb-inner header order only
coincides with sb-outer at nb==1.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import ckpt_pack_q4k as pk  # noqa: E402
import q4k_ref as ref  # noqa: E402

PE_N = 4
N = 8
K = 768
NB = K // 256


def build_weight(rng):
    raw = bytearray()
    src = {"d": [], "dmin": [], "sc96": [], "codes": []}
    for col in range(N):
        ds, dms, scs_l, codes_row = [], [], [], []
        for sb in range(NB):
            scs = [int(v) for v in rng.integers(0, 64, 8)]
            mns = [int(v) for v in rng.integers(0, 64, 8)]
            scales12 = ref._pack_6bit_scales(scs, mns)
            qh = [int(v) for v in rng.integers(0, 256, 32)]
            qs = [int(v) for v in rng.integers(0, 256, 128)]
            d_h = ref._f32_to_f16bits(float(rng.uniform(0.003, 0.05)))
            dm_h = ref._f32_to_f16bits(float(rng.uniform(0.001, 0.02)))
            raw += pk.q5k_pack_block(d_h, dm_h, scales12, qh, qs)
            ds.append(d_h); dms.append(dm_h)
            scs_l.append(int.from_bytes(bytes(scales12), "little"))
            codes_row.extend(pk.q5k_qs_to_codes(qh, qs))
        src["d"].append(ds); src["dmin"].append(dms)
        src["sc96"].append(scs_l); src["codes"].append(codes_row)
    return {"name": "q5k_cross", "N": N, "K": K, "raw": bytes(raw)}, src


def main(inj_noqh=False):
    os.makedirs("build", exist_ok=True)
    rng = np.random.default_rng(0x5C0DE5)
    w, src = build_weight(rng)
    words, descs = pk.pack_q5k_weight(w, pe_n=PE_N)

    if inj_noqh:
        # must FAIL: re-pack the CODE region with the fifth bit dropped, leaving
        # the header region and the expectations untouched.
        words = list(words)
        n_tiles = (N + PE_N - 1) // PE_N
        for ct in range(n_tiles):
            cbase = ct * (NB * PE_N + K) + NB * PE_N
            for k in range(K):
                wd = 0
                for pj in range(PE_N):
                    col = ct * PE_N + pj
                    c = src["codes"][col][k] if col < N else 0
                    wd |= (c & 0x0F) << (16 * pj)      # 0x0F, not 0x1F
                words[cbase + k] = wd

    with open("build/q5k_cross_img.hex", "w") as f:
        for wd in words:
            f.write(f"{wd:064x}\n")

    n_tiles = (N + PE_N - 1) // PE_N
    with open("build/q5k_cross_exp.txt", "w") as f:
        f.write(f"{n_tiles} {PE_N} {NB} {K}\n")
        for ct in range(n_tiles):
            f.write(f"{ct * (NB * PE_N + K)}\n")
            for pj in range(PE_N):
                col = ct * PE_N + pj
                for sb in range(NB):
                    f.write(f"{src['d'][col][sb]:04x} {src['dmin'][col][sb]:04x} "
                            f"{src['sc96'][col][sb]:024x}\n")
            for k in range(K):
                lane = 0
                for pj in range(PE_N):
                    lane |= (src["codes"][ct * PE_N + pj][k] & 0x1F) << (16 * pj)
                f.write(f"{lane:016x}\n")

    for ct, t in enumerate(descs["tiles"]):
        assert t["base"] == ct * (NB * PE_N + K), "tile base disagrees with the TB's derivation"
    assert descs["k_len"] == K and descs["n_sblk"] == NB
    # the corpus must actually exercise the fifth bit, or the gate proves nothing
    hi = sum(1 for col in range(N) for c in src["codes"][col] if c >= 16)
    tot = N * K
    print(f"q5k-cross: image {len(words)} words, {n_tiles} tiles, nb={NB}; "
          f"{hi}/{tot} codes have the fifth bit set ({100.0*hi/tot:.1f} %)"
          + ("  [INJ: qh dropped]" if inj_noqh else ""))
    if not inj_noqh and hi * 4 < tot:
        print("q5k-cross: FAILED -- fewer than 25 % of codes use the fifth bit; "
              "this corpus would barely distinguish Q5_K from Q4_K")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main("--inj-noqh" in sys.argv))
