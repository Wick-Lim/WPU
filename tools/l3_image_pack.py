#!/usr/bin/env python3
"""l3_image_pack.py -- build every image the L3 board top boots from.

WHAT THIS EMITS (all under build/):
  l3_boot.hex        the flat SPI-NOR byte image, segments packed back-to-back in
                     the EXACT order/lengths fpga/l3_top.v derives (weights, em,
                     fn, aw-store, fw-routed, fw-shared, rw-store)
  l3_lb_aw.hex ...   per-family loopback DDR images, indexed by the KEY bits of
                     the marked address (aw/fw/rw/lw/gn) -- the sim AXI model
                     marker-decodes ARADDR and serves img[key]

WHY THE VALUES ARE HASH-DEFINED
  The end-to-end gate's reference is a standalone glm_model_q4k fed by the
  loopback TBs' Verilog stub functions.  This file MIRRORS those functions in
  Python (f_h / gen_bf16 / gen_fp16 / gen_q4 / gen_s32 and the per-family f_*).
  If a single mirrored bit differs, the DUT (fed from these images) and the
  reference (fed from the Verilog functions) decode different tokens and the
  gate fails.  `make l3-hash-mirror` proves the mirrors bit-exact on a probe set
  BEFORE the expensive end-to-end sim ever runs, so a mirror bug fails in
  seconds, not hours.

GEOMETRY
  Parameterised; defaults mirror the E2E TB's tiny config, NOT the fitted one --
  the fitted config's boot alone is ~131 K words (~25 M sim cycles), which is a
  board timescale, not a gate timescale.  The layout logic is identical at every
  geometry; the E2E gate proves it at the tiny one.
"""
import argparse, os

M32 = 0xFFFFFFFF


def f_h(seed):
    # Verilog: (seed*2654435761)^(seed<<13)^(seed*40503), 32-bit integer wrap
    s = seed & M32
    return ((s * 2654435761) ^ ((s << 13) & M32) ^ (s * 40503)) & M32


def gen_bf16(seed):
    h = f_h(seed)
    s = (h >> 3) & 1
    e = (124 + ((h >> 4) & 3)) & 0xFF
    m = (h >> 6) & 0x7F
    return (s << 15) | (e << 7) | m


def gen_fp16(seed):
    h = f_h(seed)
    e = (12 + ((h >> 4) & 1)) & 0x1F
    m = (h >> 5) & 0x3FF
    return (e << 10) | m


def gen_q4(seed):
    return (f_h(seed) >> 8) & 0xF


def gen_s32(seed):
    return f_h(((seed & M32) * 97 + 5) & M32)


# ---- per-family value functions (mirrors of the loopback TBs') --------------
def f_awq(ly, sel, fo, kk):  return gen_q4(ly * 7919 + sel * 104729 + fo * 611953 + kk * 13 + 101)
def f_awd(ly, sel, fo):      return gen_fp16(ly * 7919 + sel * 104729 + fo * 611953 + 211)
def f_awdm(ly, sel, fo):     return gen_fp16(ly * 7919 + sel * 104729 + fo * 611953 + 307)
def f_awsc(ly, sel, fo, w):  return gen_s32(ly * 7919 + sel * 104729 + fo * 611953 + 601 + w)
def f_rwq(ly, e, kk):        return gen_q4(ly * 7919 + e * 350377 + kk * 13 + 401)
def f_rwd(ly, e):            return gen_fp16(ly * 7919 + e * 350377 + 421)
def f_rwdm(ly, e):           return gen_fp16(ly * 7919 + e * 350377 + 431)
def f_rwsc(ly, e, w):        return gen_s32(ly * 7919 + e * 350377 + 441 + w)
def f_fwq(ly, sel, shr, ei, fo, kk):
    return gen_q4(ly * 7919 + sel * 104729 + shr * 15485863 + ei * 350377 + fo * 611953 + kk * 13 + 503)
def f_fwd(ly, sel, shr, ei, fo):
    return gen_fp16(ly * 7919 + sel * 104729 + shr * 15485863 + ei * 350377 + fo * 611953 + 521)
def f_fwdm(ly, sel, shr, ei, fo):
    return gen_fp16(ly * 7919 + sel * 104729 + shr * 15485863 + ei * 350377 + fo * 611953 + 531)
def f_fwsc(ly, sel, shr, ei, fo, w):
    return gen_s32(ly * 7919 + sel * 104729 + shr * 15485863 + ei * 350377 + fo * 611953 + 541 + w)
def f_lw(vt, t, kk, dim):    return gen_bf16((vt * -1 + (vt + 0)) or 0)  # replaced below
def f_em(tok, idx, dim):     return gen_bf16(tok * dim + idx + 7001)
def f_fn(idx):               return gen_bf16(idx + 7207)
def f_gn(ly, which, idx):    return gen_bf16(ly * 1024 + which * 512 + idx + 7411)


def f_lwcol(vt, t, kk, lm_tn, dim):
    return gen_bf16((vt * lm_tn + t) * dim + kk + 7603)


def clog2(n):
    b = 0
    while (1 << b) < n:
        b += 1
    return max(b, 1)


def pack_scales96(w0, w1, w2):
    return w0 | (w1 << 32) | (w2 << 64)


class Geo:
    def __init__(self, a):
        self.__dict__.update(vars(a))
        self.LAYW  = clog2(self.L)
        self.EIDXW = clog2(self.N_EXPERT)
        self.A_NSB = 1                      # tiny/fitted configs: KMAX <= 256
        self.FF_NSB_D = 1
        self.R_NSB = 1
        # aw grp width mirrors l3_top's A_GRPW derivation
        hqk   = self.H_HEADS * (self.NOPE + self.ROPE)
        hnope = self.H_HEADS * self.NOPE
        hv    = self.H_HEADS * self.V_DIM
        a_omax = max(hqk, self.MODEL_DIM, hnope, hv)
        self.A_GRPW = clog2((a_omax + self.PE_N - 1) // self.PE_N)
        self.AWW  = (16 + 16 + 96) * self.PE_N * self.A_NSB
        self.FWW  = (16 + 16 + 96) * self.TN * self.FF_NSB_D * 2
        self.RWW  = (16 + 16 + 96) * self.N_EXPERT * self.R_NSB
        self.AW_AB = self.LAYW + 4 + self.A_GRPW
        self.FR_AB = self.LAYW + 1 + self.EIDXW + 5
        self.FS_AB = self.LAYW + 1 + 6
        self.RW_AB = max(self.LAYW, 3)
        self.BOOT_DW = 64


def aw_word(g, ly, sel, grp):
    d = dm = sc = 0
    for t in range(g.PE_N):
        fo = grp * g.PE_N + t
        for sb in range(g.A_NSB):
            slot = sb * g.PE_N + t
            d  |= f_awd(ly, sel, fo)  << (16 * slot)
            dm |= f_awdm(ly, sel, fo) << (16 * slot)
            sc |= pack_scales96(f_awsc(ly, sel, fo, 0), f_awsc(ly, sel, fo, 1),
                                f_awsc(ly, sel, fo, 2)) << (96 * slot)
    db = 16 * g.PE_N * g.A_NSB
    return d | (dm << db) | (sc << (2 * db))


def fw_word(g, ly, sel, shr, ei, grp):
    def half(s):
        d = dm = sc = 0
        for t in range(g.TN):
            fo = grp * g.TN + t
            for sb in range(g.FF_NSB_D):
                slot = sb * g.TN + t
                d  |= f_fwd(ly, s, shr, ei, fo)  << (16 * slot)
                dm |= f_fwdm(ly, s, shr, ei, fo) << (16 * slot)
                sc |= pack_scales96(f_fwsc(ly, s, shr, ei, fo, 0),
                                    f_fwsc(ly, s, shr, ei, fo, 1),
                                    f_fwsc(ly, s, shr, ei, fo, 2)) << (96 * slot)
        db = 16 * g.TN * g.FF_NSB_D
        return d | (dm << db) | (sc << (2 * db))
    gate = half(sel)          # the pass sel (gate or down)
    up   = half(3)            # the UP pass reuses sel=3 in the stub functions
    return gate | (up << (g.FWW // 2))


def rw_word(g, ly):
    d = dm = sc = 0
    for e in range(g.N_EXPERT):
        for sb in range(g.R_NSB):
            slot = sb * g.N_EXPERT + e
            d  |= f_rwd(ly, e)  << (16 * slot)
            dm |= f_rwdm(ly, e) << (16 * slot)
            sc |= pack_scales96(f_rwsc(ly, e, 0), f_rwsc(ly, e, 1),
                                f_rwsc(ly, e, 2)) << (96 * slot)
    db = 16 * g.N_EXPERT * g.R_NSB
    return d | (dm << db) | (sc << (2 * db))


def words_to_boot(words, width, bdw=64):
    """split each wide store word into MSB-first 64b boot sub-words --
       the exact assembly order l3_top's b_sh shifter expects."""
    out = []
    n = width // bdw
    for w in words:
        for i in range(n):
            out.append((w >> (bdw * (n - 1 - i))) & ((1 << bdw) - 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--MODEL_DIM", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--VOCAB", type=int, default=64)
    ap.add_argument("--H_HEADS", type=int, default=2)
    ap.add_argument("--NOPE", type=int, default=4)
    ap.add_argument("--ROPE", type=int, default=4)
    ap.add_argument("--V_DIM", type=int, default=8)
    ap.add_argument("--PE_N", type=int, default=2)
    ap.add_argument("--N_EXPERT", type=int, default=4)
    ap.add_argument("--TN", type=int, default=4)
    ap.add_argument("--LM_TN", type=int, default=4)
    ap.add_argument("--INTER_MOE", type=int, default=16)
    ap.add_argument("--INTER_DENSE", type=int, default=32)
    ap.add_argument("--KV_LORA", type=int, default=8)
    ap.add_argument("--out", default="build")
    g = Geo(ap.parse_args())
    os.makedirs(g.out, exist_ok=True)

    # ---- store segments (boot image) ----------------------------------------
    aw_words = [aw_word(g, (a >> (4 + g.A_GRPW)), (a >> g.A_GRPW) & 15, a & ((1 << g.A_GRPW) - 1))
                for a in range(1 << g.AW_AB)]
    fwr_words = [fw_word(g, (a >> (1 + g.EIDXW + 5)),
                         2 if ((a >> (g.EIDXW + 5)) & 1) else 0,   # sel bit: 0=gate,1=down
                         0, (a >> 5) & ((1 << g.EIDXW) - 1), a & 31)
                 for a in range(1 << g.FR_AB)]
    fws_words = [fw_word(g, (a >> 7), 2 if ((a >> 6) & 1) else 0, 1, 0, a & 63)
                 for a in range(1 << g.FS_AB)]
    rw_words  = [rw_word(g, a % g.L) for a in range(1 << g.RW_AB)]

    em_elems = [f_em(t, i, g.MODEL_DIM) for t in range(g.VOCAB) for i in range(g.MODEL_DIM)]
    fn_elems = [f_fn(i) for i in range(g.MODEL_DIM)]

    def elems_to_boot(elems):     # 4 bf16 per 64b word, MSB-first
        out = []
        for i in range(0, len(elems), 4):
            w = 0
            for j in range(4):
                w |= elems[i + j] << (16 * (3 - j))
            out.append(w)
        return out

    wt_seg  = [0] * 1024                       # DDR weight seg: unused at RESIDENT=1+loopback
    segs = [wt_seg, elems_to_boot(em_elems), elems_to_boot(fn_elems),
            words_to_boot(aw_words, g.AWW), words_to_boot(fwr_words, g.FWW),
            words_to_boot(fws_words, g.FWW), words_to_boot(rw_words, g.RWW)]

    with open(f"{g.out}/l3_boot.hex", "w") as f:   # flat byte image, MSB-first per word
        for seg in segs:
            for w in seg:
                for b in range(8):
                    f.write(f"{(w >> (8 * (7 - b))) & 0xFF:02x}\n")

    with open(f"{g.out}/l3_boot_layout.txt", "w") as f:
        base = 0
        for i, seg in enumerate(segs):
            f.write(f"seg{i} flash_word_base={base} len={len(seg)}\n")
            base += len(seg)

    print("boot segs (words):", [len(x) for x in segs], "total", sum(len(x) for x in segs))

    # ---- loopback DDR images (per-family, key-indexed) ----------------------
    DIMW = clog2(g.MODEL_DIM)
    NVT  = g.VOCAB // g.LM_TN
    VTW  = clog2(NVT)
    # aw: key {ly, sel, grp, k} -> beat low PE_N*4 = codes
    #     (mirrors the aw loopback address encoding: k | grp | sel | layer)
    # kept as a probe-value emitter for the E2E TB's AXI model, which decodes the
    # marked address itself and calls the SAME functions -- so what the image
    # files must pin down is ONLY the VALUE functions, proven by l3-hash-mirror.
    probes = []
    for i in range(64):
        probes.append(("awq",  f_awq(i % g.L, i % 7, i % 16, i % 8)))
        probes.append(("awd",  f_awd(i % g.L, i % 7, i % 16)))
        probes.append(("fwq",  f_fwq(i % g.L, i % 3, i & 1, i % g.N_EXPERT, i % 8, i % 8)))
        probes.append(("fwd",  f_fwd(i % g.L, i % 3, i & 1, i % g.N_EXPERT, i % 8)))
        probes.append(("rwq",  f_rwq(i % g.L, i % g.N_EXPERT, i % 8)))
        probes.append(("rwd",  f_rwd(i % g.L, i % g.N_EXPERT)))
        probes.append(("lw",   f_lwcol(i % NVT, i % g.LM_TN, i % g.MODEL_DIM, g.LM_TN, g.MODEL_DIM)))
        probes.append(("em",   f_em(i % g.VOCAB, i % g.MODEL_DIM, g.MODEL_DIM)))
        probes.append(("fn",   f_fn(i % g.MODEL_DIM)))
        probes.append(("gn",   f_gn(i % g.L, i & 1, i % g.MODEL_DIM)))
        probes.append(("sc",   f_awsc(i % g.L, i % 7, i % 16, i % 3) & 0xFFFFFFFF))
    with open(f"{g.out}/l3_hash_probe.txt", "w") as f:
        for name, v in probes:
            f.write(f"{name} {v:08x}\n")
    print(f"hash probes: {len(probes)} written")


if __name__ == "__main__":
    main()
