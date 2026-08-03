#!/usr/bin/env python3
"""packer_rtl_crosscheck.py -- emit a PACKER-BUILT image for the RTL to consume.

WHY THIS EXISTS
  tools/ckpt_pack_q4k.py and weight_loader_q4k disagreed on the HEADER word order:
  the packer emitted sb-outer (entry sb*PE_N+pj) while the loader reads memory at
  pj*n_sblk+sb (col-outer).  The two orders COINCIDE at nb==1 -- which is every sim
  geometry the repo ever ran -- and silently diverge at real K>256 (GLM-5.2 has
  K=6144, nb=24).  The packer's own round-trip selftest could never see it because
  its unpack side mirrored the same order.

  This script is the missing half of the fix: it packs a deterministic synthetic
  Q4_K weight at nb=3 (the diverging regime) with the REAL pack_q4k_weight code
  path, and writes
      build/pk_cross_img.hex   the packed words, one per memory word ($readmemh)
      build/pk_cross_exp.txt   the EXPECTED per-(col,sb) d/dmin/scales and the
                               per-k code nibbles, taken from the SOURCE arrays
                               BEFORE packing -- independent of the order under test
  test/packer_rtl_crosscheck_tb.v then runs the actual weight_loader_q4k over the
  image and compares the mm_w_* buses against the expectations.  A cross-TOOL
  gate: the file the packer writes is the file the RTL reads.

GEOMETRY: N=8 (2 tiles at PE_N=4), K=768 (nb=3).  nb>1 is the whole point.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ckpt_pack_q4k as pk
import q4k_ref as ref

QK_K = pk.QK_K
PE_N, N, K = 4, 8, 768
NB = K // QK_K            # 3 super-blocks -- the regime where the orders diverge


def build_weight():
    rng = np.random.default_rng(0x51C0DE)
    raw = bytearray()
    src = {"d": [], "dmin": [], "sc96": [], "codes": []}
    for col in range(N):
        ds, dms, scs_l, codes_row = [], [], [], []
        for sb in range(NB):
            scs = [int(v) for v in rng.integers(0, 64, 8)]
            mns = [int(v) for v in rng.integers(0, 64, 8)]
            scales12 = ref._pack_6bit_scales(scs, mns)
            qs = [int(v) for v in rng.integers(0, 256, 128)]
            d_h = ref._f32_to_f16bits(float(rng.uniform(0.003, 0.05)))
            dm_h = ref._f32_to_f16bits(float(rng.uniform(0.001, 0.02)))
            raw += pk.q4k_pack_block(d_h, dm_h, scales12, qs)
            ds.append(d_h); dms.append(dm_h)
            scs_l.append(int.from_bytes(bytes(scales12), "little"))
            codes_row.extend(pk.qs_to_codes(qs))
        src["d"].append(ds); src["dmin"].append(dms)
        src["sc96"].append(scs_l); src["codes"].append(codes_row)
    return {"name": "pk_cross", "N": N, "K": K, "raw": bytes(raw)}, src


def main():
    os.makedirs("build", exist_ok=True)
    w, src = build_weight()
    words, descs = pk.pack_q4k_weight(w, pe_n=PE_N)

    # image: one packed word per memory word, 256-bit hex lines for $readmemh
    with open("build/pk_cross_img.hex", "w") as f:
        for wd in words:
            f.write(f"{wd:064x}\n")

    # expectations from the SOURCE arrays (never through the pack order)
    with open("build/pk_cross_exp.txt", "w") as f:
        n_tiles = (N + PE_N - 1) // PE_N
        f.write(f"{n_tiles} {PE_N} {NB} {K}\n")
        for ct in range(n_tiles):
            base = ct * (NB * PE_N + K)
            f.write(f"{base}\n")
            for pj in range(PE_N):
                col = ct * PE_N + pj
                for sb in range(NB):
                    f.write(f"{src['d'][col][sb]:04x} {src['dmin'][col][sb]:04x} "
                            f"{src['sc96'][col][sb]:024x}\n")
            for k in range(K):
                nib = 0
                for pj in range(PE_N):
                    nib |= src["codes"][ct * PE_N + pj][k] << (4 * pj)
                f.write(f"{nib:04x}\n")

    # sanity: the per-tile bases must match what the TB derives
    assert descs["k_len"] == K and descs["n_sblk"] == NB and descs["pe_n"] == PE_N
    for ct, t in enumerate(descs["tiles"]):
        exp_base = ct * (NB * PE_N + K)
        assert t["base"] == exp_base, f"tile base {t['base']} != {exp_base}"
    print(f"pk-cross: image {len(words)} words, {n_tiles} tiles, nb={NB} (divergent regime)")


if __name__ == "__main__":
    n_tiles = (N + PE_N - 1) // PE_N
    main()
