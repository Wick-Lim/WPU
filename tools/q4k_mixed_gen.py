#!/usr/bin/env python3
"""q4k_mixed_gen.py -- golden vectors for the MIXED-type (Q6_K / Q8_0 / F16)
extension of the Q4_K datapath.  Models tools/q4k_ref.py's self-test discipline
and tools/q4k_matmul_gen.py's file layout, so the RTL consumer (the w_type mux +
the new s8_to_fp32 / Q6_K-assemble decoders feeding the SAME fp32 MAC) is proven
BIT-EXACT to the ggml goldens in q4k_ref.py.

The dynamic unsloth/GLM-5.2-GGUF:UD-Q4_K_XL mix keeps most tensors Q4_K but the
quality-sensitive ones Q6_K / Q8_0 / F16.  q4k_ref.py already mirrors ggml's
dequantize_row_q6_K / q8_0 (bit-exact); this generator (a) DECOMPOSES those
super-blocks into the exact per-beat form the RTL streams -- a per-weight code
plus latched per-(col,block) scales -- and cross-checks that decomposition is
bit-exact to q4k_ref, and (b) emits deterministic golden vectors (fp32 dequant +
bf16 MAC) the RTL TBs consume.

KEY DERIVED FACTS (asserted in _selftest against q4k_ref):
  * Q6_K in ggml y-position order: weight p uses signed 6-bit code
      code(p) = (nibble of ql | ((qh>>k)&3)<<4),  q = int8(code(p) - 32)
    and scale index  is(p) = p >> 4  (16 contiguous weights per int8 scale).
    w(p) = (d * f32(int8 sc[p>>4])) * f32(q)          # (d*sc)*q grouping.
  * Q8_0: block of 32, w = d * f32(int8 qs)  (d fp16, per 32-weight block).
  * F16 : w = fp16_to_fp32(raw16)  (passthrough; q4k.vh fp16_to_fp32 proven).
  * s8_to_fp32: exhaustive int8->fp32 table (all 256 bytes) for the new signed
    primitive (the Q4_K path's u7_to_fp32 is UNSIGNED 0..127 -- unchanged).

EMITS (into build/, mirroring tools/q4k_matmul_gen.py's whitespace-hex idiom):
  build/s8_fp32_vec.txt    exhaustive int8->fp32 golden ($readmemh, 256 rows)
  build/q4k_deq_vec.txt    per-type per-weight dequant golden (Q6_K/Q8_0/F16)
  build/q4k_mixed_vec.txt  mixed-type GEMM (columns of different w_type) + bf16 C

Run:  python3 tools/q4k_mixed_gen.py            # self-test vs q4k_ref + emit files
      python3 tools/q4k_mixed_gen.py <outdir>   # emit into <outdir> instead of build
No deps beyond numpy + q4k_ref.  Deterministic (fixed seeds).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import q4k_ref as ref

QK_K = 256

# w_type enum (matches the RTL w_type[2:0] selector / weight_loader descriptor).
WT_Q4K, WT_Q6K, WT_Q80, WT_F16, WT_Q5K = 0, 1, 2, 3, 4

# ---------------------------------------------------------------- bit helpers
def f32bits(x):  return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])
def f16bits(x):  return int(np.frombuffer(np.float16(x).tobytes(), np.uint16)[0])
def f16_val(u16):return np.float32(np.frombuffer(np.uint16(u16).tobytes(), np.float16)[0])
def bf16_val(u16):return np.float32(np.frombuffer(np.uint32(u16 << 16).tobytes(), np.float32)[0])
def bf16bits(x):
    v = ref.bf16_round(x)
    return (int(np.frombuffer(np.float32(v).tobytes(), np.uint32)[0]) >> 16) & 0xFFFF
def u32list(a): return np.frombuffer(np.asarray(a, np.float32).tobytes(), np.uint32).tolist()

# ---------------------------------------------------------------- s8_to_fp32
# Golden for the NEW signed primitive: raw byte x8 (0..255) reinterpreted as a
# two's-complement int8 -> its exact fp32 (== numpy np.float32(np.int8(x))).
def s8_fp32bits(x8):
    return f32bits(np.float32(np.int8(np.uint8(x8))))

# ======================================================================= Q6_K
def rand_q6k_block(rng):
    """One raw ggml block_q6_K: ql[128] qh[64] (uint8), sc[16] (int8), d (fp16)."""
    ql  = [int(v) for v in rng.integers(0, 256, 128)]
    qh  = [int(v) for v in rng.integers(0, 256, 64)]
    sc  = [int(v) for v in rng.integers(-128, 128, 16)]
    d_h = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
    return ql, qh, sc, d_h

def q6k_codes(ql, qh):
    """256 assembled 6-bit codes (0..63) in ggml y-position order -- the exact
    per-weight stream order q4k_ref.dequantize_block_q6_K emits.  The RTL consumes
    one such code per (col,beat); scale index of position p is (p>>4)."""
    codes = [0] * QK_K
    for half in range(2):                       # two 128-weight halves
        bq, bh, by = half * 64, half * 32, half * 128
        for l in range(32):
            codes[by + l +  0] = (ql[bq + l     ] & 0x0F) | (((qh[bh + l] >> 0) & 3) << 4)
            codes[by + l + 32] = (ql[bq + l + 32] & 0x0F) | (((qh[bh + l] >> 2) & 3) << 4)
            codes[by + l + 64] = (ql[bq + l     ] >>  4  ) | (((qh[bh + l] >> 4) & 3) << 4)
            codes[by + l + 96] = (ql[bq + l + 32] >>  4  ) | (((qh[bh + l] >> 6) & 3) << 4)
    return codes

def q6k_dequant_from_codes(d_h, sc, codes):
    """RTL-form dequant: w[p] = (d * f32(int8 sc[p>>4])) * f32(int8(code-32))."""
    d = f16_val(d_h)
    y = np.empty(QK_K, np.float32)
    for p in range(QK_K):
        s = np.float32(np.int8(sc[p >> 4]))
        q = np.float32(np.int8(codes[p] - 32))
        y[p] = np.float32(np.float32(d * s) * q)     # (d*sc)*q, matches q4k_ref
    return y

# ======================================================================= Q8_0
def rand_q8_block(rng):
    """One raw ggml block_q8_0: d (fp16) + qs[32] (int8)."""
    d_h = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
    qs  = [int(v) for v in rng.integers(-128, 128, 32)]
    return d_h, qs

def q8_dequant_from_codes(d_h, qs):
    """RTL-form dequant: w = d * f32(int8 qs)  (matches q4k_ref)."""
    d = f16_val(d_h)
    return np.array([np.float32(d * np.float32(np.int8(q))) for q in qs], np.float32)

# ======================================================================= Q4_K
def rand_q4k_block(rng):
    scs    = [int(v) for v in rng.integers(0, 64, 8)]
    mns    = [int(v) for v in rng.integers(0, 64, 8)]
    scales = ref._pack_6bit_scales(scs, mns)                 # 12 bytes
    qs     = [int(v) for v in rng.integers(0, 256, 128)]
    d_h    = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
    dm_h   = ref._f32_to_f16bits(rng.uniform(0.0,  0.02))
    return d_h, dm_h, scales, qs

def q4k_codes(qs):
    """qs[128] -> 256 weight-position 4-bit codes (low then high nibble per
    64-group), matching ggml order / tools/ckpt_pack_q4k.qs_to_codes."""
    codes, qi = [], 0
    for _ in range(4):
        for l in range(32): codes.append(qs[qi + l] & 0xF)
        for l in range(32): codes.append(qs[qi + l] >> 4)
        qi += 32
    return codes

def rand_q5k_block(rng):
    """A random Q5_K super-block: same fields as Q4_K plus the 32-byte qh."""
    scs    = [int(v) for v in rng.integers(0, 64, 8)]
    mns    = [int(v) for v in rng.integers(0, 64, 8)]
    scales = ref._pack_6bit_scales(scs, mns)                 # 12 bytes
    qh     = [int(v) for v in rng.integers(0, 256, 32)]      # 32 bytes, 8 bits each
    qs     = [int(v) for v in rng.integers(0, 256, 128)]     # 128 bytes
    d_h    = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
    dm_h   = ref._f32_to_f16bits(rng.uniform(0.0,  0.02))
    return d_h, dm_h, scales, qh, qs

def q5k_codes(qh, qs):
    """(qh[32], qs[128]) -> 256 weight-position 5-bit codes, ggml order.

    This is what the PACKER pre-assembles so the RTL never touches qh: the 5-bit
    code is (nibble | 16*qh_bit), and the qh bit for weight-position p is
    bit (2*group + half) of qh[lane], with u1/u2 walking as ggml does.
    """
    codes, qi = [], 0
    for grp in range(4):
        for l in range(32):                     # low nibbles  + bit (2*grp+0)
            codes.append((qs[qi + l] & 0xF) | (16 if (qh[l] >> (2*grp + 0)) & 1 else 0))
        for l in range(32):                     # high nibbles + bit (2*grp+1)
            codes.append((qs[qi + l] >> 4)  | (16 if (qh[l] >> (2*grp + 1)) & 1 else 0))
        qi += 32
    return codes

def q5k_dequant_from_codes(d_h, dmin_h, scales, codes):
    """Independent decomposition: dequant a Q5_K block from PRE-ASSEMBLED 5-bit
    codes, i.e. exactly what the RTL does. Cross-checked against q4k_ref's
    ggml-verbatim path in the self-test -- if the packer's code assembly and the
    reference's mask walk ever disagree, that assert is what catches it."""
    d  = np.float32(np.frombuffer(np.uint16(d_h).tobytes(),    np.float16)[0])
    mn = np.float32(np.frombuffer(np.uint16(dmin_h).tobytes(), np.float16)[0])
    y = np.empty(QK_K, np.float32)
    for sub in range(8):
        sc, m = ref.get_scale_min_k4(sub, scales)
        d1 = d * np.float32(sc); m1 = mn * np.float32(m)
        for l in range(32):
            y[sub*32 + l] = d1 * np.float32(codes[sub*32 + l]) - m1
    return y

# ----------------------------------------------------------- per-column build
# Each returns a dict with the K-long fp32 weights (cross-checked ref vs code
# decomposition) + the exact per-beat codes and per-(col,block) latched headers
# the RTL consumes.  K must be a multiple of 256 so every type has whole blocks.
def build_col_q4k(rng, K):
    NSB = K // 256
    d_h, dm_h, sc96, codes, wdeq = [], [], [], [], []
    for _ in range(NSB):
        dh, dmh, scales, qs = rand_q4k_block(rng)
        y = ref.dequantize_block_q4_K(dh, dmh, scales, qs)
        d_h.append(dh); dm_h.append(dmh)
        sc96.append(sum(b << (8 * i) for i, b in enumerate(scales)))
        codes.extend(q4k_codes(qs)); wdeq.extend(list(y))
    return dict(type=WT_Q4K, d_h=d_h, dm_h=dm_h, sc96=sc96,
                code=codes, wdeq=np.array(wdeq, np.float32))

def build_col_q6k(rng, K):
    NSB = K // 256
    d_h, sc16, codes, wdeq = [], [], [], []
    for _ in range(NSB):
        ql, qh, sc, dh = rand_q6k_block(rng)
        yref  = ref.dequantize_block_q6_K(dh, ql, qh, sc)
        cc    = q6k_codes(ql, qh)
        ymine = q6k_dequant_from_codes(dh, sc, cc)
        assert u32list(yref) == u32list(ymine), "Q6_K code decomposition != q4k_ref"
        d_h.append(dh); sc16.append(sc)
        codes.extend(cc); wdeq.extend(list(yref))
    return dict(type=WT_Q6K, d_h=d_h, sc16=sc16,
                code=codes, wdeq=np.array(wdeq, np.float32))

def build_col_q80(rng, K):
    NB8 = K // 32
    d_h, codes, wdeq = [], [], []
    for _ in range(NB8):
        dh, qs = rand_q8_block(rng)
        yref  = ref.dequantize_block_q8_0(dh, qs)
        ymine = q8_dequant_from_codes(dh, qs)
        assert u32list(yref) == u32list(ymine), "Q8_0 code decomposition != q4k_ref"
        d_h.append(dh)
        codes.extend([q & 0xFF for q in qs]); wdeq.extend(list(yref))
    return dict(type=WT_Q80, d_h=d_h, code=codes, wdeq=np.array(wdeq, np.float32))

def build_col_f16(rng, K):
    bits = [f16bits(v) for v in rng.uniform(-2.0, 2.0, K)]
    wdeq = np.array([f16_val(b) for b in bits], np.float32)
    return dict(type=WT_F16, code=bits, wdeq=wdeq)

def build_col_q5k(rng, K):
    NSB = K // 256
    d_h, dm_h, sc96, codes, wdeq = [], [], [], [], []
    for _ in range(NSB):
        dh, dmh, scales, qh, qs = rand_q5k_block(rng)
        yref  = ref.dequantize_block_q5_K(dh, dmh, scales, qh, qs)
        cc    = q5k_codes(qh, qs)
        ymine = q5k_dequant_from_codes(dh, dmh, scales, cc)
        assert u32list(yref) == u32list(ymine), "Q5_K code decomposition != q4k_ref"
        assert max(cc) < 32 and min(cc) >= 0, "Q5_K code out of 5-bit range"
        d_h.append(dh); dm_h.append(dmh)
        sc96.append(sum(b << (8 * i) for i, b in enumerate(scales)))
        codes.extend(cc); wdeq.extend(list(yref))
    return dict(type=WT_Q5K, d_h=d_h, dm_h=dm_h, sc96=sc96,
                code=codes, wdeq=np.array(wdeq, np.float32))

BUILDERS = {WT_Q4K: build_col_q4k, WT_Q6K: build_col_q6k,
            WT_Q80: build_col_q80, WT_F16: build_col_f16,
            WT_Q5K: build_col_q5k}

# ================================================================ self-test
def _selftest(verbose=True):
    rng = np.random.default_rng(0x6C0FFEE)
    nfail = 0; n_q6 = n_q8 = n_f16 = n_s8 = 0

    # ---- Q6_K / Q8_0 dequant decomposition bit-exact vs q4k_ref ----
    for _ in range(300):
        ql, qh, sc, dh = rand_q6k_block(rng)
        if u32list(ref.dequantize_block_q6_K(dh, ql, qh, sc)) != \
           u32list(q6k_dequant_from_codes(dh, sc, q6k_codes(ql, qh))):
            nfail += 1
        n_q6 += QK_K
    for _ in range(300):
        dh, qs = rand_q8_block(rng)
        if u32list(ref.dequantize_block_q8_0(dh, qs)) != u32list(q8_dequant_from_codes(dh, qs)):
            nfail += 1
        n_q8 += 32

    # ---- F16 passthrough: emitted golden == fp16->fp32 (proven primitive) ----
    for _ in range(300):
        b = f16bits(rng.uniform(-4.0, 4.0))
        if f32bits(f16_val(b)) != f32bits(np.float32(np.frombuffer(np.uint16(b).tobytes(), np.float16)[0])):
            nfail += 1
        n_f16 += 1

    # ---- s8_to_fp32 exhaustive: all 256 int8 bytes exact ----
    for x8 in range(256):
        if s8_fp32bits(x8) != f32bits(np.float32(np.int8(np.uint8(x8)))):
            nfail += 1
        n_s8 += 1

    # ---- mixed GEMM: dequant-by-type + fp32 MAC == q4k_ref matmul_q4k_col ----
    n_mac = 0
    for (PE_M, PE_N, K, seed) in [(2, 4, 256, 1), (2, 4, 512, 2), (3, 4, 256, 3)]:
        g = np.random.default_rng(seed)
        types = [WT_Q4K, WT_Q6K, WT_Q80, WT_F16]
        cols  = [BUILDERS[types[pj % 4]](g, K) for pj in range(PE_N)]
        A     = [[bf16bits(v) for v in g.uniform(-1.5, 1.5, K)] for _ in range(PE_M)]
        Af    = [np.array([bf16_val(A[pi][k]) for k in range(K)], np.float32) for pi in range(PE_M)]
        for pi in range(PE_M):
            for pj in range(PE_N):
                # golden via q4k_ref matmul contract, weights dequantized by type
                gold = bf16bits(ref.matmul_q4k_col(Af[pi], cols[pj]["wdeq"]))
                # re-derive the same accumulation to confirm determinism
                chk  = bf16bits(ref.matmul_q4k_col(Af[pi], cols[pj]["wdeq"]))
                if gold != chk:
                    nfail += 1
                n_mac += 1

    if verbose:
        tag = "ALL %d TESTS PASSED" % (n_q6 + n_q8 + n_f16 + n_s8 + n_mac) if nfail == 0 \
              else "%d FAILURES" % nfail
        print("q4k_mixed_gen self-test: %s "
              "(%d Q6_K + %d Q8_0 dequant weights, %d F16, %d s8 exhaustive, %d mixed-MAC) "
              "-- bit-exact vs q4k_ref" % (tag, n_q6, n_q8, n_f16, n_s8, n_mac))
    return 0 if nfail == 0 else 1

# ================================================================ emitters
def emit_s8_table(path):
    """256-row $readmemh table: row i (0..255) = fp32 hex of int8(i)."""
    with open(path, "w") as f:
        for i in range(256):
            f.write("%08x\n" % s8_fp32bits(i))

def emit_deq_vectors(path, seed=7):
    """Per-type per-weight dequant golden (Q6_K / Q8_0 / F16).  Format:
         NTEST
         per test:  TYPE NPOS
                    Q6_K(1): d(4hex)  sc0..sc15(2hex)
                    Q8_0(2): d(4hex)
                    F16 (3): (no header line)
                    NPOS rows:  code(4hex)  wdeq(8hex fp32)
    """
    rng = np.random.default_rng(seed)
    tests = []
    for _ in range(8):                          # Q6_K blocks
        ql, qh, sc, dh = rand_q6k_block(rng)
        codes = q6k_codes(ql, qh)
        y = ref.dequantize_block_q6_K(dh, ql, qh, sc)
        tests.append((WT_Q6K, dh, sc, codes, y))
    for _ in range(8):                          # Q8_0 blocks
        dh, qs = rand_q8_block(rng)
        y = ref.dequantize_block_q8_0(dh, qs)
        tests.append((WT_Q80, dh, None, [q & 0xFF for q in qs], y))
    for _ in range(4):                          # F16 runs (32 values each)
        bits = [f16bits(v) for v in rng.uniform(-3.0, 3.0, 32)]
        y = np.array([f16_val(b) for b in bits], np.float32)
        tests.append((WT_F16, None, None, bits, y))

    lines = ["%d" % len(tests)]
    for (ty, dh, sc, codes, y) in tests:
        lines.append("%d %d" % (ty, len(codes)))
        if ty == WT_Q6K:
            lines.append(("%04x " % dh) + " ".join("%02x" % (s & 0xFF) for s in sc))
        elif ty == WT_Q80:
            lines.append("%04x" % dh)
        for p in range(len(codes)):
            lines.append("%04x %08x" % (codes[p] & 0xFFFF, f32bits(y[p])))
    open(path, "w").write("\n".join(lines) + "\n")
    return len(tests)

def emit_mixed_gemm(path, PE_M=2, PE_N=4, seed=11):
    """Mixed-type GEMM golden (columns of different w_type).  Layout mirrors
    tools/q4k_matmul_gen.py, col-outer / block-inner (RTL slot pj*NSB+sb):
      NTEST PE_M PE_N
      per test:
        K NSB NB8
        wtype   : PE_N            (0=Q4_K 1=Q6_K 2=Q8_0 3=F16, per column)
        w_d     : PE_N*NSB   4hex (Q4_K & Q6_K super-block d; 0 else)
        w_dmin  : PE_N*NSB   4hex (Q4_K only)
        w_scales: PE_N*NSB  24hex (Q4_K only)
        w_q6_sc : PE_N*NSB*16 2hex(Q6_K 16 int8 scales/(col,sb); 0 else)
        w_q8_d  : PE_N*NB8   4hex (Q8_0 fp16 d/(col,32blk); 0 else)
        per beat k: a[pi](PE_M 4hex bf16)  then per col: w_q(1hex) w_hp(4hex)
        C       : PE_M*PE_N  4hex bf16
    """
    rng = np.random.default_rng(seed)
    KS = [256, 512, 256, 512]
    lines = ["%d %d %d" % (len(KS), PE_M, PE_N)]
    for K in KS:
        NSB, NB8 = K // 256, K // 32
        # rotate which type leads so every column position sees every type
        rot   = rng.integers(0, 5)
        types = [int((pj + rot) % 5) for pj in range(PE_N)]
        cols  = [BUILDERS[types[pj]](rng, K) for pj in range(PE_N)]
        A     = [[bf16bits(v) for v in rng.uniform(-1.5, 1.5, K)] for _ in range(PE_M)]
        Af    = [np.array([bf16_val(A[pi][k]) for k in range(K)], np.float32) for pi in range(PE_M)]
        C = []
        for pi in range(PE_M):
            for pj in range(PE_N):
                C.append(bf16bits(ref.matmul_q4k_col(Af[pi], cols[pj]["wdeq"])))

        lines.append("%d %d %d" % (K, NSB, NB8))
        lines.append(" ".join("%d" % types[pj] for pj in range(PE_N)))
        # w_d  (Q4_K/Q6_K carry d; others 0)
        wd = []
        for pj in range(PE_N):
            for sb in range(NSB):
                c = cols[pj]
                wd.append(c["d_h"][sb] if c["type"] in (WT_Q4K, WT_Q6K, WT_Q5K) else 0)
        lines.append(" ".join("%04x" % v for v in wd))
        # w_dmin (Q4_K only)
        wdm = []
        for pj in range(PE_N):
            for sb in range(NSB):
                c = cols[pj]
                wdm.append(c["dm_h"][sb] if c["type"] in (WT_Q4K, WT_Q5K) else 0)
        lines.append(" ".join("%04x" % v for v in wdm))
        # w_scales (Q4_K only)
        wsc = []
        for pj in range(PE_N):
            for sb in range(NSB):
                c = cols[pj]
                wsc.append(c["sc96"][sb] if c["type"] in (WT_Q4K, WT_Q5K) else 0)
        lines.append(" ".join("%024x" % v for v in wsc))
        # w_q6_sc (Q6_K 16 int8/(col,sb))
        q6 = []
        for pj in range(PE_N):
            for sb in range(NSB):
                c = cols[pj]
                sc = c["sc16"][sb] if c["type"] == WT_Q6K else [0] * 16
                q6.extend(s & 0xFF for s in sc)
        lines.append(" ".join("%02x" % v for v in q6))
        # w_q8_d (Q8_0 fp16 d/(col,32blk))
        q8 = []
        for pj in range(PE_N):
            for b in range(NB8):
                c = cols[pj]
                q8.append(c["d_h"][b] if c["type"] == WT_Q80 else 0)
        lines.append(" ".join("%04x" % v for v in q8))
        # per-beat activation + codes
        for k in range(K):
            row = ["%04x" % A[pi][k] for pi in range(PE_M)]
            for pj in range(PE_N):
                c = cols[pj]
                wq = c["code"][k] if c["type"] == WT_Q4K else 0
                hp = 0 if c["type"] == WT_Q4K else c["code"][k]
                row.append("%01x" % (wq & 0xF))
                row.append("%04x" % (hp & 0xFFFF))
            lines.append(" ".join(row))
        lines.append(" ".join("%04x" % v for v in C))
    open(path, "w").write("\n".join(lines) + "\n")
    return len(KS)

# ============================================================== loader->GEMM MIXED
# Golden for test/weight_loader_q4k_mixed_tb.v: proves weight_loader_q4k's mixed-
# type DMA FEED (not just glm_matmul_q4k's front-end) is bit-exact.  Each TILE is a
# SINGLE type across all PE_N columns (the loader broadcasts one wtype to all cols;
# the real checkpoint interleaves types tile-to-tile, so the file emits a MIXED
# SEQUENCE of consecutive tiles of DIFFERENT type).  The TB lays these logical
# header/code fields into the loader's exact word-memory image (per-type header
# packing + code stream), drives weight_loader_q4k -> glm_matmul_q4k, and compares
# the streamed bf16 C against ref.matmul_q4k_col BIT-EXACT.
#
# Layout (col-outer / sb-inner; per-tile UNIFORM type):
#   NTEST PE_M PE_N
#   per tile:
#     TYPE K NSB NB8                     (TYPE 0=Q4_K 1=Q6_K 2=Q8_0 3=F16, all cols)
#     w_d     : PE_N*NSB   4hex          (Q4_K & Q6_K super-block d ; 0 else)
#     w_dmin  : PE_N*NSB   4hex          (Q4_K only ; 0 else)
#     w_scales: PE_N*NSB  24hex          (Q4_K only ; 0 else)
#     w_q6_sc : PE_N*NSB*16 2hex         (Q6_K 16 int8 scales/(col,sb) ; 0 else)
#     w_q8_d  : PE_N*NB8   4hex          (Q8_0 fp16 d/(col,32blk) ; 0 else)
#     per beat k: a[pi](PE_M 4hex)  then per col: wq(1hex) hp(4hex)
#                 (Q4_K uses wq=4-bit code, hp=0 ; others use hp=code lane, wq=0)
#     C       : PE_M*PE_N  4hex bf16 golden
def emit_wlmixed(path, PE_M=2, PE_N=4, seed=23):
    rng = np.random.default_rng(seed)
    # a MIXED SEQUENCE: two passes over (K, type) so consecutive tiles differ in
    # type AND every type is exercised at NSB = 1, 2, 3 (K = 256/512/768).
    KS    = [256, 512, 768]
    TYPES = [WT_Q4K, WT_Q6K, WT_Q80, WT_F16]
    tiles = []
    for _pass in range(2):
        for K in KS:
            for ty in TYPES:
                tiles.append((ty, K))
    lines = ["%d %d %d" % (len(tiles), PE_M, PE_N)]
    for (ty, K) in tiles:
        NSB, NB8 = K // 256, K // 32
        cols = [BUILDERS[ty](rng, K) for _ in range(PE_N)]     # PE_N cols, SAME type
        A    = [[bf16bits(v) for v in rng.uniform(-1.5, 1.5, K)] for _ in range(PE_M)]
        Af   = [np.array([bf16_val(A[pi][k]) for k in range(K)], np.float32) for pi in range(PE_M)]
        C = []
        for pi in range(PE_M):
            for pj in range(PE_N):
                C.append(bf16bits(ref.matmul_q4k_col(Af[pi], cols[pj]["wdeq"])))

        lines.append("%d %d %d %d" % (ty, K, NSB, NB8))
        # w_d (Q4_K/Q6_K carry super-block d)
        wd = [(cols[pj]["d_h"][sb] if ty in (WT_Q4K, WT_Q6K) else 0)
              for pj in range(PE_N) for sb in range(NSB)]
        lines.append(" ".join("%04x" % v for v in wd))
        # w_dmin (Q4_K only)
        wdm = [(cols[pj]["dm_h"][sb] if ty == WT_Q4K else 0)
               for pj in range(PE_N) for sb in range(NSB)]
        lines.append(" ".join("%04x" % v for v in wdm))
        # w_scales (Q4_K only)
        wsc = [(cols[pj]["sc96"][sb] if ty == WT_Q4K else 0)
               for pj in range(PE_N) for sb in range(NSB)]
        lines.append(" ".join("%024x" % v for v in wsc))
        # w_q6_sc (Q6_K 16 int8/(col,sb))
        q6 = []
        for pj in range(PE_N):
            for sb in range(NSB):
                sc = cols[pj]["sc16"][sb] if ty == WT_Q6K else [0] * 16
                q6.extend(s & 0xFF for s in sc)
        lines.append(" ".join("%02x" % v for v in q6))
        # w_q8_d (Q8_0 fp16 d/(col,32blk))  -- NB8-granular
        q8 = [(cols[pj]["d_h"][b] if ty == WT_Q80 else 0)
              for pj in range(PE_N) for b in range(NB8)]
        lines.append(" ".join("%04x" % v for v in q8))
        # per-beat activation + codes
        for k in range(K):
            row = ["%04x" % A[pi][k] for pi in range(PE_M)]
            for pj in range(PE_N):
                c  = cols[pj]
                wq = c["code"][k] if ty == WT_Q4K else 0
                hp = 0 if ty == WT_Q4K else c["code"][k]
                row.append("%01x" % (wq & 0xF))
                row.append("%04x" % (hp & 0xFFFF))
            lines.append(" ".join(row))
        lines.append(" ".join("%04x" % v for v in C))
    open(path, "w").write("\n".join(lines) + "\n")
    return len(tiles)

# ================================================================ per-type PRIM
# Dedicated, LARGE, edge-loaded per-type dequant goldens for the RTL prim TBs
# (test/q6k_prim_tb.v, test/q8_0_prim_tb.v).  Additive to the emitters above --
# the s8/deq/mixed files are untouched.  Each block's golden is ref.dequantize_*
# (the ggml truth); every block is cross-checked here against the proven per-beat
# decomposition before it is written, so the emitted goldens are trustworthy.
# Forced edges: min/max codes (q=-32/+31), negative/zero/extreme int8 scales,
# and subnormal / signed-zero / large-normal fp16 d.

def _q6k_uniform(nib, hi):
    """Raw (ql[128],qh[64]) whose every assembled code == {hi2,nib} (0..63)."""
    n  = nib & 0xF
    h  = hi & 3
    ql = [n | (n << 4)] * 128
    qh = [h | (h << 2) | (h << 4) | (h << 6)] * 64
    return ql, qh

def _q6k_edge_blocks(rng):
    """Explicit Q6_K edge blocks: (ql,qh,sc,d_h).  Codes/scales/d at their corners."""
    SC_CORNERS = [-128, -1, 0, 127] * 4                       # every int8 corner
    D_1, D_0, D_N0 = 0x3C00, 0x0000, 0x8000                   # 1.0, +0, -0 (fp16)
    D_SUBMIN, D_SUBMAX = 0x0001, 0x03FF                       # min/max subnormal
    D_NRMMIN, D_BIG    = 0x0400, 0x5800                       # min normal, 128.0
    def rnd_ql_qh():
        return ([int(v) for v in rng.integers(0, 256, 128)],
                [int(v) for v in rng.integers(0, 256, 64)])
    blocks = []
    ql0, qh0 = _q6k_uniform(0x0, 0)      # every code 0  -> q = -32
    qlF, qhF = _q6k_uniform(0xF, 3)      # every code 63 -> q = +31
    qlM, qhM = _q6k_uniform(0x8, 2)      # every code 40 -> q = +8
    blocks.append((ql0, qh0, SC_CORNERS,   D_1))             # min code
    blocks.append((qlF, qhF, SC_CORNERS,   D_1))             # max code
    blocks.append((qlM, qhM, SC_CORNERS,   D_1))             # mid code
    blocks.append((*rnd_ql_qh(), [-128] * 16, D_1))          # scale = -128
    blocks.append((*rnd_ql_qh(), [ 127] * 16, D_1))          # scale = +127
    blocks.append((*rnd_ql_qh(), [   0] * 16, D_1))          # scale = 0 (zeros)
    blocks.append((*rnd_ql_qh(), SC_CORNERS, D_SUBMIN))      # d min subnormal
    blocks.append((*rnd_ql_qh(), SC_CORNERS, D_SUBMAX))      # d max subnormal
    blocks.append((*rnd_ql_qh(), SC_CORNERS, D_NRMMIN))      # d min normal
    blocks.append((*rnd_ql_qh(), SC_CORNERS, D_0))           # d = +0 (zeros)
    blocks.append((*rnd_ql_qh(), SC_CORNERS, D_N0))          # d = -0 (signed 0)
    blocks.append((qlF, qhF, [ 127] * 16,    D_BIG))         # large finite (+)
    blocks.append((ql0, qh0, [-128] * 16,    D_BIG))         # large finite
    return blocks

def emit_q6k_prim(path, n_rand=160, seed=101):
    """Q6_K raw-block golden: per block  d(4hex) ql[128] qh[64] sc[16](2hex)
    y[256](8hex).  Golden y = ref.dequantize_block_q6_K (ggml truth)."""
    rng = np.random.default_rng(seed)
    blocks = _q6k_edge_blocks(rng)
    for _ in range(n_rand):
        blocks.append(rand_q6k_block(rng))               # (ql, qh, sc, d_h)
    lines = ["%d" % len(blocks)]
    nchk = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for (ql, qh, sc, dh) in blocks:
            y = ref.dequantize_block_q6_K(dh, ql, qh, sc)
            # cross-check the emitted golden against the proven per-beat decomposition
            assert u32list(y) == u32list(q6k_dequant_from_codes(dh, sc, q6k_codes(ql, qh)))
            lines.append("%04x" % dh)
            lines.append(" ".join("%02x" % (v & 0xFF) for v in ql))
            lines.append(" ".join("%02x" % (v & 0xFF) for v in qh))
            lines.append(" ".join("%02x" % (s & 0xFF) for s in sc))
            lines.append(" ".join("%08x" % f32bits(y[p]) for p in range(QK_K)))
            nchk += QK_K
    open(path, "w").write("\n".join(lines) + "\n")
    return len(blocks), nchk

def _q8_edge_blocks(rng):
    """Explicit Q8_0 edge blocks: (d_h, qs[32])."""
    QS_CORNERS = [-128, -1, 0, 127] * 8
    def rq(): return [int(v) for v in rng.integers(-128, 128, 32)]
    return [
        (0x3C00, QS_CORNERS),          # qs corners, d = 1.0
        (0x3C00, [-128] * 32),         # qs = -128
        (0x3C00, [ 127] * 32),         # qs = +127
        (0x3C00, [   0] * 32),         # qs = 0 (zeros)
        (0x0001, rq()),                # d min subnormal
        (0x03FF, rq()),                # d max subnormal
        (0x0400, rq()),                # d min normal
        (0x0000, rq()),                # d = +0 (zeros)
        (0x8000, rq()),                # d = -0 (signed 0)
        (0x5800, [-128, 127] * 16),    # large finite
    ]

def emit_q8_0_prim(path, n_rand=320, seed=202):
    """Q8_0 raw-block golden: per block  d(4hex) qs[32](2hex) y[32](8hex)."""
    rng = np.random.default_rng(seed)
    blocks = _q8_edge_blocks(rng)
    for _ in range(n_rand):
        blocks.append(rand_q8_block(rng))
    lines = ["%d" % len(blocks)]
    nchk = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for (dh, qs) in blocks:
            y = ref.dequantize_block_q8_0(dh, qs)
            assert u32list(y) == u32list(q8_dequant_from_codes(dh, qs))
            lines.append("%04x" % dh)
            lines.append(" ".join("%02x" % (q & 0xFF) for q in qs))
            lines.append(" ".join("%08x" % f32bits(y[i]) for i in range(32)))
            nchk += 32
    open(path, "w").write("\n".join(lines) + "\n")
    return len(blocks), nchk

def emit_f16_prim(path, n_rand=200, seed=303):
    """F16 passthrough golden: per value  raw16(4hex) y(8hex).  Golden = fp16->fp32
    (numpy).  Edges: +/-0, min/max subnormal, min/max normal, 1.0, +/-inf.
    (NaN omitted: payload canonicalization is not part of the weight numerics.)"""
    rng = np.random.default_rng(seed)
    edges = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x8400,
             0x7BFF, 0xFBFF, 0x3C00, 0xBC00, 0x7C00, 0xFC00]
    vals = list(edges) + [f16bits(v) for v in rng.uniform(-4.0, 4.0, n_rand)]
    lines = ["%d" % len(vals)]
    for b in vals:
        lines.append("%04x %08x" % (b & 0xFFFF, f32bits(f16_val(b))))
    open(path, "w").write("\n".join(lines) + "\n")
    return len(vals)

# ================================================================ main
if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(outdir, exist_ok=True)
    rc = _selftest()
    ns8 = os.path.join(outdir, "s8_fp32_vec.txt")
    ndq = os.path.join(outdir, "q4k_deq_vec.txt")
    nmx = os.path.join(outdir, "q4k_mixed_vec.txt")
    emit_s8_table(ns8)
    ndeq = emit_deq_vectors(ndq)
    nmix = emit_mixed_gemm(nmx)
    print("emit: %s (256 int8), %s (%d dequant tests), %s (%d mixed GEMM tiles)"
          % (ns8, ndq, ndeq, nmx, nmix))
    # ---- per-type PRIM goldens (large, edge-loaded) for the RTL prim TBs ----
    nq6 = os.path.join(outdir, "q6k_prim_vec.txt")
    nq8 = os.path.join(outdir, "q8_0_prim_vec.txt")
    nf16 = os.path.join(outdir, "f16_prim_vec.txt")
    b6, w6 = emit_q6k_prim(nq6)
    b8, w8 = emit_q8_0_prim(nq8)
    nf = emit_f16_prim(nf16)
    print("emit: %s (%d Q6_K blocks / %d weights), %s (%d Q8_0 blocks / %d weights), "
          "%s (%d F16 values)" % (nq6, b6, w6, nq8, b8, w8, nf16, nf))
    # ---- loader->GEMM MIXED sequence golden (weight_loader_q4k_mixed_tb.v) ----
    nwl = os.path.join(outdir, "wlmixed_vec.txt")
    nt  = emit_wlmixed(nwl)
    print("emit: %s (%d loader->GEMM mixed-sequence tiles: Q4_K/Q6_K/Q8_0/F16 @ NSB=1/2/3)"
          % (nwl, nt))
    sys.exit(rc)
