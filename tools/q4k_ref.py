#!/usr/bin/env python3
"""q4k_ref.py -- bit-exact reference for GGML Q4_K dequantization.

The verification golden for the Q4_K hardware core (src/q4k.vh, glm_matmul_q4k.v),
targeting the published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` model. This reimplements
ggml's `dequantize_row_q4_K` (ggml/src/ggml-quants.c) EXACTLY, so the RTL's Q4_K path
can be proven bit-exact to THIS reference. NOTE: this file IS the golden, so that
proof is bit-exactness vs this reimplementation -- matching the real published GGUF
bytes end-to-end (including the Q6_K/Q8_0/F16 tensors of the dynamic UD-Q4_K_XL mix,
which this Q4_K-only datapath does not yet consume) is a separate, OPEN validation.

GGML Q4_K super-block (QK_K = 256 weights, 144 bytes):
    ggml_half d;         # fp16 super-block scale
    ggml_half dmin;      # fp16 super-block min
    uint8_t   scales[12] # 8x 6-bit block-scales + 8x 6-bit block-mins, packed
    uint8_t   qs[128]    # 256x 4-bit quant codes (low then high nibble per byte-group)
Dequant per weight:  w = (d*sc)*q - (dmin*m)
  where d,dmin are fp16->fp32, sc/m are the 6-bit block scale/min (as ints), q in [0,15].
All products/subtracts are fp32 -- reuses the datapath's existing fp32 pipes.

No external deps beyond numpy (for exact IEEE fp16<->fp32). Run: python3 tools/q4k_ref.py
"""
import numpy as np

QK_K = 256

def get_scale_min_k4(j, scales):
    """ggml get_scale_min_k4: unpack the j-th 6-bit block scale `d` and min `m`
    from the 12-byte `scales` array. Bit-identical to the C."""
    q = scales
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4)
    return d, m

def dequantize_block_q4_K(d_h, dmin_h, scales, qs):
    """Dequantize ONE Q4_K super-block -> 256 fp32 values, exactly as ggml does.
      d_h, dmin_h : uint16 fp16 bit-patterns (super-block scale / min)
      scales      : list/bytes of 12 uint8
      qs          : list/bytes of 128 uint8
    """
    d   = np.float32(np.frombuffer(np.uint16(d_h).tobytes(),   dtype=np.float16)[0])
    mn  = np.float32(np.frombuffer(np.uint16(dmin_h).tobytes(), dtype=np.float16)[0])
    y = np.empty(QK_K, dtype=np.float32)
    yi = 0
    qi = 0
    is_ = 0
    for _ in range(0, QK_K, 64):          # 4 groups of 64 = 8 sub-blocks of 32
        sc, m = get_scale_min_k4(is_ + 0, scales)
        d1 = d * np.float32(sc); m1 = mn * np.float32(m)
        sc, m = get_scale_min_k4(is_ + 1, scales)
        d2 = d * np.float32(sc); m2 = mn * np.float32(m)
        for l in range(32):               # low nibbles, scale index is+0
            y[yi] = d1 * np.float32(qs[qi + l] & 0xF) - m1; yi += 1
        for l in range(32):               # high nibbles, scale index is+1
            y[yi] = d2 * np.float32(qs[qi + l] >> 4) - m2; yi += 1
        qi += 32; is_ += 2
    return y

def dequantize_row_q4_K(blocks):
    """blocks: iterable of (d_h, dmin_h, scales, qs). -> concatenated fp32 row."""
    return np.concatenate([dequantize_block_q4_K(*b) for b in blocks])

# ---------------------------------------------------------------- matmul contract
# The GEMM contract the hardware core (glm_matmul_q4k.v) computes, bit-exact:
#   weights: the Q4_K-typed tensors, dequantized EXACTLY (above), NO re-quantization.
#   activations: bf16 (same interface as glm_matmul_pipe).
#   per output: out[m][n] = bf16( SUM_k fp32(a[m][k]) * w_deq[k][n] ), the fp32
#   products sequentially fp32-accumulated in K order (the streaming order the RTL
#   uses), then rounded to bf16 (round-to-nearest-even). Same accumulate structure
#   as the proven bf16 glm_matmul_pipe -- only the weight source changes (Q4_K deq).

def bf16_round(x):
    """fp32 -> bf16 value (kept in fp32), round-to-nearest-even on the low 16 bits."""
    u = int(np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0])
    if (u & 0x7F800000) == 0x7F800000:      # inf/nan: truncate mantissa, keep
        u2 = u & 0xFFFF0000
    else:
        lsb = (u >> 16) & 1
        u2 = (u + 0x7FFF + lsb) & 0xFFFF0000
    return np.float32(np.frombuffer(np.uint32(u2).tobytes(), dtype=np.float32)[0])

def matmul_q4k_col(a_row_f32, wdeq_col_f32):
    """One output = sequential fp32 dot of a bf16 activation row with a dequantized
    Q4_K weight column, rounded to bf16. a_row_f32 entries are bf16-valued fp32."""
    acc = np.float32(0.0)
    for k in range(len(a_row_f32)):
        acc = np.float32(acc + np.float32(a_row_f32[k]) * np.float32(wdeq_col_f32[k]))
    return bf16_round(acc)

# ------------------------------------------------- other UD-Q4_K_XL k-quants ----
# UD-Q4_K_XL is a DYNAMIC mix: most tensors Q4_K, sensitive ones kept at higher
# precision (Q6_K / Q8_0 / F16). Same GEMM contract (dequant exactly -> fp32 MAC);
# only the per-weight dequant differs. All bit-exact to ggml.

def _s8(b):
    """Raw byte (0..255) -> signed int8 value (two's complement). Explicit wrap:
    np.int8(x) on x>127 wrapped silently on numpy<2 but raises OverflowError on
    numpy>=2 -- this keeps the golden bit-identical on both."""
    return b - 256 if b > 127 else b

def dequantize_block_q6_K(d_h, ql, qh, sc):
    """One Q6_K super-block (256 weights, 210 B: ql[128] qh[64] int8 sc[16] + fp16 d)
    -> 256 fp32, exactly as ggml dequantize_row_q6_K.
      q = ((ql&0xF)|((qh>>k&3)<<4)) - 32 (signed 6-bit), w = d * sc[is] * q."""
    d = np.float32(np.frombuffer(np.uint16(d_h).tobytes(), dtype=np.float16)[0])
    y = np.empty(QK_K, dtype=np.float32)
    qlb, qhb, scb, yb = 0, 0, 0, 0
    for _ in range(0, QK_K, 128):
        for l in range(32):
            is_ = l // 16
            q1 = np.int8(((ql[qlb+l+ 0] & 0xF) | (((qh[qhb+l] >> 0) & 3) << 4)) - 32)
            q2 = np.int8(((ql[qlb+l+32] & 0xF) | (((qh[qhb+l] >> 2) & 3) << 4)) - 32)
            q3 = np.int8(((ql[qlb+l+ 0] >>  4) | (((qh[qhb+l] >> 4) & 3) << 4)) - 32)
            q4 = np.int8(((ql[qlb+l+32] >>  4) | (((qh[qhb+l] >> 6) & 3) << 4)) - 32)
            y[yb+l+ 0] = d * np.float32(np.int8(_s8(sc[scb+is_+0]))) * np.float32(q1)
            y[yb+l+32] = d * np.float32(np.int8(_s8(sc[scb+is_+2]))) * np.float32(q2)
            y[yb+l+64] = d * np.float32(np.int8(_s8(sc[scb+is_+4]))) * np.float32(q3)
            y[yb+l+96] = d * np.float32(np.int8(_s8(sc[scb+is_+6]))) * np.float32(q4)
        yb += 128; qlb += 64; qhb += 32; scb += 8
    return y

def dequantize_block_q5_K(d_h, dmin_h, scales, qh, qs):
    """One Q5_K super-block (256 weights, 176 B) -> 256 fp32, exactly as ggml
    dequantize_row_q5_K (ggml/src/ggml-quants.c).

    Q5_K is Q4_K plus a fifth bit: the same fp16 d/dmin, the same 12-byte packed
    6-bit block scales/mins read by get_scale_min_k4, the same 128-byte nibble
    array -- and a 32-byte `qh` supplying one extra high bit per weight, which
    lifts q from [0,15] to [0,31].
      d_h, dmin_h : uint16 fp16 bit-patterns (super-block scale / min)
      scales      : 12 uint8   (shared unpack with Q4_K)
      qh          : 32 uint8   (bit l of the group's mask selects +16)
      qs          : 128 uint8  (low nibble, then high nibble, per 32-lane half)
    Dequant per weight:  w = (d*sc) * (q4 + 16*qh_bit) - (dmin*m)

    NOTE on the masks: ggml walks qh with u1=1,u2=2 shifted left by 2 per
    64-group, and does NOT advance the qh pointer -- so each of the 32 qh bytes
    supplies 8 bits, two per 64-group across the 4 groups. Reproduced verbatim;
    "simplifying" the mask walk is how this gets silently wrong for groups 1-3.
    """
    d  = np.float32(np.frombuffer(np.uint16(d_h).tobytes(),    dtype=np.float16)[0])
    mn = np.float32(np.frombuffer(np.uint16(dmin_h).tobytes(), dtype=np.float16)[0])
    y = np.empty(QK_K, dtype=np.float32)
    yi = 0
    qi = 0
    is_ = 0
    u1, u2 = 1, 2
    for _ in range(0, QK_K, 64):
        sc, m = get_scale_min_k4(is_ + 0, scales)
        d1 = d * np.float32(sc); m1 = mn * np.float32(m)
        sc, m = get_scale_min_k4(is_ + 1, scales)
        d2 = d * np.float32(sc); m2 = mn * np.float32(m)
        for l in range(32):                       # low nibbles  + bit u1
            hi = np.float32(16.0) if (qh[l] & u1) else np.float32(0.0)
            y[yi] = d1 * (np.float32(qs[qi + l] & 0xF) + hi) - m1; yi += 1
        for l in range(32):                       # high nibbles + bit u2
            hi = np.float32(16.0) if (qh[l] & u2) else np.float32(0.0)
            y[yi] = d2 * (np.float32(qs[qi + l] >> 4) + hi) - m2; yi += 1
        qi += 32; is_ += 2
        u1 <<= 2; u2 <<= 2
    return y

def dequantize_row_q5_K(blocks):
    """blocks: iterable of (d_h, dmin_h, scales, qh, qs). -> concatenated fp32 row."""
    return np.concatenate([dequantize_block_q5_K(*b) for b in blocks])

def dequantize_block_q8_0(d_h, qs):
    """One Q8_0 block (32 weights: fp16 d + 32 int8) -> 32 fp32. w = d * qs."""
    d = np.float32(np.frombuffer(np.uint16(d_h).tobytes(), dtype=np.float16)[0])
    return np.array([d * np.float32(np.int8(_s8(q))) for q in qs], dtype=np.float32)

# ---------------------------------------------------------------- self-test ----
def _pack_6bit_scales(scs, mns):
    """Inverse of get_scale_min_k4: pack 8 6-bit scales + 8 6-bit mins -> 12 bytes.
    Used only to build test vectors (the encoder side is llama.cpp's job in production)."""
    assert len(scs) == 8 and len(mns) == 8 and all(0 <= v < 64 for v in scs + mns)
    q = [0] * 12
    for j in range(4):
        q[j]     = scs[j]            # low 6 bits
        q[j + 4] = mns[j]
    for j in range(4, 8):
        # d = (q[j+4]&0xF) | ((q[j-4]>>6)<<4) ; m = (q[j+4]>>4) | ((q[j]>>6)<<4)
        q[j + 4] = (scs[j] & 0xF) | ((mns[j] & 0xF) << 4)
        q[j - 4] |= ((scs[j] >> 4) & 0x3) << 6
        q[j]     |= ((mns[j] >> 4) & 0x3) << 6
    return q

def _f32_to_f16bits(x):
    return int(np.frombuffer(np.float16(x).tobytes(), dtype=np.uint16)[0])

def _selftest():
    rng = np.random.default_rng(0)
    n_fail = 0; n = 0
    for t in range(200):
        scs = [int(v) for v in rng.integers(0, 64, 8)]
        mns = [int(v) for v in rng.integers(0, 64, 8)]
        scales = _pack_6bit_scales(scs, mns)
        # verify the pack<->unpack round-trips through get_scale_min_k4
        for j in range(8):
            d, m = get_scale_min_k4(j, scales)
            assert d == scs[j] and m == mns[j], f"scale unpack mismatch j={j}: {d}!={scs[j]} {m}!={mns[j]}"
        qs = [int(v) for v in rng.integers(0, 256, 128)]
        d_h  = _f32_to_f16bits(rng.uniform(0.001, 0.05))
        dm_h = _f32_to_f16bits(rng.uniform(0.0, 0.02))
        y = dequantize_block_q4_K(d_h, dm_h, scales, qs)
        # independent recompute of a few positions
        d  = np.float32(np.frombuffer(np.uint16(d_h).tobytes(),  dtype=np.float16)[0])
        mn = np.float32(np.frombuffer(np.uint16(dm_h).tobytes(), dtype=np.float16)[0])
        for pos in [0, 31, 32, 63, 64, 127, 200, 255]:
            sub = pos // 32                     # which 32-lane sub-block (0..7)
            grp = sub // 2                      # 64-group
            is_ = grp * 2 + (sub & 1)           # scale index
            sc, m = get_scale_min_k4(is_, scales)
            byte = (pos % 32) + (pos // 64) * 32
            q = (qs[byte] & 0xF) if (sub & 1) == 0 else (qs[byte] >> 4)
            exp = np.float32(d * np.float32(sc)) * np.float32(q) - np.float32(mn * np.float32(m))
            n += 1
            if np.frombuffer(y[pos].tobytes(),dtype=np.uint32)[0] != np.frombuffer(exp.tobytes(),dtype=np.uint32)[0]:
                n_fail += 1
    if n_fail == 0:
        print(f"q4k_ref self-test: ALL {n} TESTS PASSED (200 blocks, bit-exact dequant + 6-bit scale round-trip)")
        return 0
    print(f"q4k_ref self-test: {n_fail}/{n} FAILED"); return 1

if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
