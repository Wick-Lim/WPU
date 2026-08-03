#!/usr/bin/env python3
# ============================================================================
# ckpt_pack_q4k.py -- GLM-5.2 GGUF (UD-Q4_K_XL) -> OUR RTL weight-memory image
# ----------------------------------------------------------------------------
# WHAT THIS IS  (Q4K_SYSTEM_PLAN.md 2.2 / 2.5)
#   The BRIDGE from the published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` GGUF file
#   (the format llama.cpp and every local device actually run) to the exact
#   weight-memory layout `src/weight_loader_q4k.v` reads and feeds to
#   `src/glm_matmul_q4k.v`.  It is the Q4_K sibling of `tools/ckpt_pack.py`
#   (which bridged the FP8 safetensors checkpoint); the "moat" moves from
#   "bit-exact to the FP8 safetensors" to "bit-exact to the published GGUF".
#
#   The real 467 GB GGUF is NOT on the dev host (disk), so this packer runs its
#   whole gen -> pack -> unpack round-trip against a SYNTHETIC tiny GGUF it
#   fabricates in-memory (a few small tensors of each ggml type, in real GGUF
#   v3 container bytes).  `import q4k_ref` supplies the bit-exact ggml dequant
#   mirrors (Q4_K / Q6_K / Q8_0) so the packed image is proven to dequantize
#   exactly as ggml does.  Runs offline: `python3 tools/ckpt_pack_q4k.py` -> 0.
#
# THE GGUF CONTAINER WE PARSE  (spec v3, little-endian)
#   [magic "GGUF"=u32][version u32][tensor_count u64][kv_count u64]
#   kv_count x  { key:gguf_string, value_type:u32, value }
#   tensor_count x { name:gguf_string, n_dims:u32, dims[n_dims]:u64,
#                    ggml_type:u32, offset:u64 }
#   pad -> general.alignment (default 32);  then the aligned tensor blob.
#   gguf_string = [len u64][utf8 bytes].  The per-tensor ggml_type enum IS the
#   dynamic type map (2.5): Q4_K=12, Q6_K=14, Q8_0=8, F16=1 (+ F32=0 tail).
#
# THE RTL TARGET LAYOUT  (2.1 super-block header + nibble codes)
#   RTL contraction orientation is W_rtl[k][n] (n = output column) = the
#   TRANSPOSE of the GGUF [out,in] weight, so GGUF row n supplies RTL column n.
#   A "tile" = PE_N output columns over the full K.  Per Q4_K tile, from `base`,
#   with NSB = ceil(K/256) super-blocks:
#     HEADER region : entry (pj*NSB + sb), pj=0..PE_N-1, sb=0..NSB-1
#         (col-OUTER / super-block-INNER -- the order weight_loader_q4k
#         actually issues header reads in; see its rd_slot comment.  This
#         file used to emit sb-outer, which COINCIDES at NSB==1 -- every
#         test geometry -- and silently diverges at real K>256.)
#         word = d[15:0] | dmin[31:16] | scales[127:32]   (fp16 d, fp16 dmin,
#         96-bit packed 6-bit scales) -- assembled by the loader into the
#         mm_w_d / mm_w_dmin / mm_w_scales buses glm_matmul_q4k latches at start.
#     CODE   region : base + NSB*PE_N + k,  k=0..K-1
#         word[4*pj +: 4] = the 4-bit code of weight k in column (col0+pj).
#   (vs the FP8 image: 8-bit codes + one bf16 block-scale/column -- Q4_K is
#    4-bit codes + a per-super-block d/dmin/scales triple, ~44% fewer bytes.)
#   Q6_K / Q8_0 / F16 / F32 tensors are emitted as their native block bytes,
#   byte-packed into the same DATA_W words; the manifest's per-tensor type lets
#   the loader/`glm_matmul_q4k` w_type mux (2.5) pick the dequant.
#
# USAGE
#   python3 tools/ckpt_pack_q4k.py gen   <path.gguf>          # synthetic GGUF
#   python3 tools/ckpt_pack_q4k.py pack  <path.gguf> <outdir> # GGUF -> image
#   python3 tools/ckpt_pack_q4k.py check <path.gguf>          # gen+pack+unpack
#   python3 tools/ckpt_pack_q4k.py                            # self-test (exit 0)
# ============================================================================
import sys, os, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import q4k_ref as ref   # bit-exact ggml dequant mirrors + 6-bit scale codec

# ---- ggml type enums (the per-tensor dynamic type map, 2.5) ---------------
GGML_F32  = 0
GGML_F16  = 1
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q6_K = 14
TYPE_NAME = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q8_0: "Q8_0",
             GGML_Q4_K: "Q4_K", GGML_Q6_K: "Q6_K"}
# type -> (elements_per_block, bytes_per_block)
BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_Q8_0: (32, 34),
         GGML_Q4_K: (256, 144), GGML_Q6_K: (256, 210)}

# ---- GGUF metadata value-type enums ---------------------------------------
GV_UINT32 = 4
GV_STRING = 8
GV_ARRAY  = 9
_GV_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
             4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
             10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
GGUF_MAGIC = 0x46554747            # "GGUF" little-endian
GGUF_VERSION = 3

# ---- RTL geometry (weight_loader_q4k / glm_matmul_q4k image defaults) ------
PE_N   = 4
QK_K   = 256                        # Q4_K/Q6_K super-block width
DATA_W = 256                        # 2.1: stays 256 (== ddr5_xbar beat)
DATA_HEX = DATA_W // 4

def tnbytes(ttype, nelem):
    epb, bpb = BLOCK[ttype]
    assert nelem % epb == 0, f"nelem {nelem} not a multiple of block {epb}"
    return (nelem // epb) * bpb

# ============================================================================
# (1) MINIMAL GGUF v3 CODEC  (pure struct; faithful container bytes)
# ============================================================================
def _gstr(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def _kv_u32(key, val):
    return _gstr(key) + struct.pack("<I", GV_UINT32) + struct.pack("<I", val)

def _kv_str(key, val):
    return _gstr(key) + struct.pack("<I", GV_STRING) + _gstr(val)

def write_gguf(tensors, arch="glm", alignment=32):
    """tensors: list of (name, ggml_type, dims_ne, raw_bytes) in GGUF ne order
       (dims_ne[0] = fastest = K/in, dims_ne[1] = N/out).  Returns GGUF bytes."""
    kv = _kv_str("general.architecture", arch) + \
         _kv_u32("general.alignment", alignment) + \
         _kv_u32("general.quantization_version", 2)
    n_kv = 3
    hdr = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION) + \
          struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv) + kv
    # tensor-info table (offsets are relative to the aligned data section)
    info = bytearray()
    blob = bytearray()
    for (name, ttype, dims, raw) in tensors:
        nelem = 1
        for d in dims:
            nelem *= d
        assert len(raw) == tnbytes(ttype, nelem), \
            f"{name}: {len(raw)} != {tnbytes(ttype, nelem)} bytes for {TYPE_NAME[ttype]}"
        off = len(blob)
        info += _gstr(name) + struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", d)
        info += struct.pack("<I", ttype) + struct.pack("<Q", off)
        blob += raw
        pad = (-len(blob)) % alignment          # keep each tensor aligned
        blob += b"\x00" * pad
    pre = hdr + bytes(info)
    dpad = (-len(pre)) % alignment               # align the data section start
    return pre + b"\x00" * dpad + bytes(blob)

def read_gguf(buf):
    """Parse GGUF bytes -> (metadata dict, tensors dict name->(type,dims,raw))."""
    off = 0
    def take(fmt):
        nonlocal off
        v = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v
    def rstr():
        nonlocal off
        n = take("<Q")
        s = buf[off:off + n].decode("utf-8"); off += n
        return s
    def rval(vtype):
        nonlocal off
        if vtype in _GV_FIXED:
            fmt, _ = _GV_FIXED[vtype]
            return take(fmt)
        if vtype == GV_STRING:
            return rstr()
        if vtype == GV_ARRAY:
            atype = take("<I"); alen = take("<Q")
            return [rval(atype) for _ in range(alen)]
        raise ValueError(f"unknown gguf value_type {vtype}")

    magic = take("<I"); version = take("<I")
    assert magic == GGUF_MAGIC, f"bad GGUF magic {magic:#x}"
    assert version == GGUF_VERSION, f"unsupported GGUF version {version}"
    n_tensors = take("<Q"); n_kv = take("<Q")
    meta = {}
    for _ in range(n_kv):
        key = rstr(); vtype = take("<I")
        meta[key] = rval(vtype)
    infos = []
    for _ in range(n_tensors):
        name = rstr(); ndim = take("<I")
        dims = [take("<Q") for _ in range(ndim)]
        ttype = take("<I"); toff = take("<Q")
        infos.append((name, ttype, dims, toff))
    alignment = meta.get("general.alignment", 32)
    data_start = off + ((-off) % alignment)      # aligned data section
    tensors = {}
    for (name, ttype, dims, toff) in infos:
        nelem = 1
        for d in dims:
            nelem *= d
        s = data_start + toff
        tensors[name] = (ttype, dims, bytes(buf[s:s + tnbytes(ttype, nelem)]))
    return meta, tensors

# ============================================================================
# (2) Q4_K block <-> (d,dmin,scales,codes) in ggml WEIGHT-POSITION order
# ============================================================================
# ggml block_q4_K on disk: [fp16 d][fp16 dmin][u8 scales[12]][u8 qs[128]] =144B.
# dequantize_row_q4_K emits weights in the order: per 64-group, 32 low-nibbles
# (scale index is+0) then 32 high-nibbles (is+1).  We mirror that exact order so
# code position p in a super-block uses scale get_scale_min_k4(p//32, .).
def q4k_pack_block(d_h, dmin_h, scales12, qs128):
    return struct.pack("<HH", d_h & 0xFFFF, dmin_h & 0xFFFF) + \
           bytes(bytearray(scales12)) + bytes(bytearray(qs128))

def q4k_unpack_block(raw):
    d_h, dmin_h = struct.unpack_from("<HH", raw, 0)
    scales12 = list(raw[4:16])
    qs128 = list(raw[16:144])
    return d_h, dmin_h, scales12, qs128

def qs_to_codes(qs128):
    """qs[128] -> 256 codes in ggml weight-position order (matches q4k_ref)."""
    codes = []
    qi = 0
    for _ in range(4):                       # 4 groups of 64
        for l in range(32):
            codes.append(qs128[qi + l] & 0xF)      # low nibbles  (scale is+0)
        for l in range(32):
            codes.append(qs128[qi + l] >> 4)       # high nibbles (scale is+1)
        qi += 32
    return codes

def codes_to_qs(codes256):
    """Inverse of qs_to_codes: 256 weight-position codes -> qs[128]."""
    qs = [0] * 128
    qi = 0; ci = 0
    for _ in range(4):
        for l in range(32):
            qs[qi + l] |= codes256[ci] & 0xF; ci += 1
        for l in range(32):
            qs[qi + l] |= (codes256[ci] & 0xF) << 4; ci += 1
        qi += 32
    return qs

# ============================================================================
# (3) SYNTHETIC tiny GGUF generator (a few tensors of each ggml type)
# ============================================================================
def gen_synthetic(path):
    rng = np.random.default_rng(0xC0FFEE)
    tensors = []

    def q4k_tensor(name, N, K):
        assert K % QK_K == 0
        nb = K // QK_K
        raw = bytearray()
        for _ in range(N):                   # one row (K weights) per out-channel
            for _ in range(nb):
                scs = [int(v) for v in rng.integers(0, 64, 8)]
                mns = [int(v) for v in rng.integers(0, 64, 8)]
                scales = ref._pack_6bit_scales(scs, mns)
                qs = [int(v) for v in rng.integers(0, 256, 128)]
                d_h  = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
                dm_h = ref._f32_to_f16bits(rng.uniform(0.0, 0.02))
                raw += q4k_pack_block(d_h, dm_h, scales, qs)
        tensors.append((name, GGML_Q4_K, [K, N], bytes(raw)))

    def q6k_tensor(name, N, K):
        assert K % QK_K == 0
        nb = K // QK_K
        raw = bytearray()
        for _ in range(N * nb):
            ql = bytes(int(v) for v in rng.integers(0, 256, 128))
            qh = bytes(int(v) for v in rng.integers(0, 256, 64))
            sc = struct.pack("<16b", *[int(v) for v in rng.integers(-32, 32, 16)])
            d_h = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
            raw += ql + qh + sc + struct.pack("<H", d_h)     # ggml block_q6_K
        tensors.append((name, GGML_Q6_K, [K, N], bytes(raw)))

    def q8_0_tensor(name, N, K):
        assert K % 32 == 0
        nb = K // 32
        raw = bytearray()
        for _ in range(N * nb):
            d_h = ref._f32_to_f16bits(rng.uniform(0.003, 0.05))
            qs = struct.pack("<32b", *[int(v) for v in rng.integers(-128, 128, 32)])
            raw += struct.pack("<H", d_h) + qs              # ggml block_q8_0
        tensors.append((name, GGML_Q8_0, [K, N], bytes(raw)))

    def f16_tensor(name, N, K):
        vals = rng.uniform(-2.0, 2.0, N * K).astype(np.float16)
        tensors.append((name, GGML_F16, [K, N], vals.tobytes()))

    def f32_tensor(name, n):                 # a bf16-ish norm tail kept full prec
        vals = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        tensors.append((name, GGML_F32, [n], vals.tobytes()))

    # a faithful mini dynamic-mix: most Q4_K, sensitive ones higher precision.
    q4k_tensor("blk.0.ffn_down.weight",   8, 512)   # NSB=2 super-blocks/row
    q4k_tensor("blk.0.ffn_gate.weight",   8, 256)
    q6k_tensor("blk.0.attn_output.weight", 4, 256)  # sensitive proj -> Q6_K
    q8_0_tensor("blk.0.attn_q.weight",     4, 256)  # sensitive proj -> Q8_0
    f16_tensor("output.weight",            4, 256)  # lm head kept F16
    f32_tensor("blk.0.attn_norm.weight",   64)      # norm tail (passthrough)

    buf = write_gguf(tensors)
    if path:
        with open(path, "wb") as f:
            f.write(buf)
    return buf

# ============================================================================
# (4) classify a parsed GGUF into (quantized weights, tail)
# ============================================================================
def classify(tensors):
    """weights: dicts for the 2-D quantized (or F16) weight tensors we pack as
       tiles; tail: 1-D vectors (norms) passed through.  GGUF ne = [K, N]."""
    weights, tail = [], []
    for name, (ttype, dims, raw) in tensors.items():
        if len(dims) == 2 and ttype in (GGML_Q4_K, GGML_Q6_K, GGML_Q8_0, GGML_F16):
            K, N = dims[0], dims[1]
            weights.append(dict(name=name, ttype=ttype, N=N, K=K, raw=raw))
        else:
            tail.append(dict(name=name, ttype=ttype, dims=dims, raw=raw))
    return weights, tail

# ============================================================================
# (5) PACK -- emit the weight-memory image + per-tensor type manifest
# ============================================================================
def _bytes_to_words(raw):
    """Pack a raw byte stream into DATA_W-bit little-endian words (pad last)."""
    words = []
    step = DATA_W // 8
    for i in range(0, len(raw), step):
        chunk = raw[i:i + step]
        w = int.from_bytes(chunk + b"\x00" * (step - len(chunk)), "little")
        words.append(w)
    return words

def _words_to_bytes(words, nbytes):
    step = DATA_W // 8
    out = bytearray()
    for w in words:
        out += int(w).to_bytes(step, "little")
    return bytes(out[:nbytes])

def pack_q4k_weight(w, pe_n=PE_N):
    """Q4_K weight -> (words, descriptor).  HEADER (per super-block, per column
       d|dmin|scales triple) then CODE region (4-bit code/column per K-beat)."""
    N, K, raw = w["N"], w["K"], w["raw"]
    nb = K // QK_K                            # super-blocks per row (== NSB)
    # decode every row's super-blocks: per (col, sb) -> (d,dmin,sc96); codes[col][k]
    d_h  = [[0] * nb for _ in range(N)]
    dm_h = [[0] * nb for _ in range(N)]
    sc96 = [[0] * nb for _ in range(N)]
    codes = [[0] * K for _ in range(N)]
    for col in range(N):
        for sb in range(nb):
            blk = raw[(col * nb + sb) * 144:(col * nb + sb) * 144 + 144]
            dh, dmh, scales12, qs = q4k_unpack_block(blk)
            d_h[col][sb] = dh; dm_h[col][sb] = dmh
            sc96[col][sb] = int.from_bytes(bytes(scales12), "little")
            cc = qs_to_codes(qs)
            for i in range(QK_K):
                codes[col][sb * QK_K + i] = cc[i]
    n_tiles = (N + pe_n - 1) // pe_n
    words, descs = [], []
    for ct in range(n_tiles):
        col0 = ct * pe_n
        base = len(words)
        # HEADER : entry (pj*nb + sb) = d | dmin<<16 | scales<<32
        #   col-OUTER / super-block-INNER: weight_loader_q4k reads header word
        #   (pj, sb) from memory offset pj*n_sblk+sb (its issue loop is
        #   "col-outer / super-block-inner").  The old sb-outer order here only
        #   matched at nb==1, which is every sim geometry -- at real K (nb=24)
        #   each super-block's d/dmin/scales landed on the wrong column, which is
        #   a plausible-looking, silently wrong model, not an error.
        for pj in range(pe_n):
            for sb in range(nb):
                col = col0 + pj
                if col < N:
                    word = (d_h[col][sb] & 0xFFFF) | ((dm_h[col][sb] & 0xFFFF) << 16) \
                           | (sc96[col][sb] << 32)
                else:
                    word = 0                  # padding column
                words.append(word)
        # CODE : word[4*pj +: 4] = code[col][k]
        for k in range(K):
            word = 0
            for pj in range(pe_n):
                col = col0 + pj
                c = codes[col][k] if col < N else 0
                word |= (c & 0xF) << (4 * pj)
            words.append(word)
        descs.append(dict(base=base, col0=col0, n_tiles=n_tiles))
    desc = dict(name=w["name"], type="Q4_K", base=None, k_len=K, n_sblk=nb,
                N=N, pe_n=pe_n, tiles=descs)
    return words, desc

def pack_raw_weight(w):
    """Q6_K / Q8_0 / F16 weight -> native block bytes byte-packed into words."""
    words = _bytes_to_words(w["raw"])
    epb, bpb = BLOCK[w["ttype"]]
    n_sblk = (w["N"] * w["K"]) // epb
    desc = dict(name=w["name"], type=TYPE_NAME[w["ttype"]], base=None,
                k_len=w["K"], n_sblk=n_sblk, N=w["N"], nbytes=len(w["raw"]))
    return words, desc

def pack_gguf(buf, out_dir=None, pe_n=PE_N):
    meta, tensors = read_gguf(buf)
    weights, tail = classify(tensors)
    all_words, manifest_tensors = [], []
    for w in weights:
        if w["ttype"] == GGML_Q4_K:
            words, desc = pack_q4k_weight(w, pe_n)
        else:
            words, desc = pack_raw_weight(w)
        off = len(all_words)
        desc["base"] = off
        if desc["type"] == "Q4_K":            # relocate per-tile bases
            for t in desc["tiles"]:
                t["base"] += off
        all_words += words
        manifest_tensors.append(desc)
    manifest = dict(
        params=dict(PE_N=pe_n, QK_K=QK_K, DATA_W=DATA_W),
        arch=meta.get("general.architecture", "?"),
        n_words=len(all_words),
        tensors=manifest_tensors,                       # 2.5 per-tensor type map
        tail=[dict(name=t["name"], type=TYPE_NAME.get(t["ttype"], str(t["ttype"])),
                   dims=t["dims"]) for t in tail],
        note="weight_mem_q4k.hex = word image src/weight_loader_q4k.v reads.",
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        img = os.path.join(out_dir, "weight_mem_q4k.hex")
        with open(img, "w") as f:
            for wd in all_words:
                f.write(f"{wd & ((1 << DATA_W) - 1):0{DATA_HEX}x}\n")
        import json
        with open(os.path.join(out_dir, "manifest_q4k.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    return all_words, manifest, weights, tail

# ============================================================================
# (6) UNPACK -- reconstruct per-tensor bytes/codes from the image (round-trip)
# ============================================================================
def unpack_q4k_weight(words, desc, pe_n):
    N, K, nb = desc["N"], desc["k_len"], desc["n_sblk"]
    d_h  = [[0] * nb for _ in range(N)]
    dm_h = [[0] * nb for _ in range(N)]
    sc96 = [[0] * nb for _ in range(N)]
    codes = [[0] * K for _ in range(N)]
    for t in desc["tiles"]:
        base, col0 = t["base"], t["col0"]
        for pj in range(pe_n):
            for sb in range(nb):
                col = col0 + pj
                if col >= N:
                    continue
                word = words[base + pj * nb + sb]
                d_h[col][sb]  = word & 0xFFFF
                dm_h[col][sb] = (word >> 16) & 0xFFFF
                sc96[col][sb] = (word >> 32) & ((1 << 96) - 1)
        code_base = base + nb * pe_n
        for k in range(K):
            word = words[code_base + k]
            for pj in range(pe_n):
                col = col0 + pj
                if col < N:
                    codes[col][k] = (word >> (4 * pj)) & 0xF
    # reassemble native GGUF block bytes per (col, super-block)
    raw = bytearray()
    for col in range(N):
        for sb in range(nb):
            scales12 = list(sc96[col][sb].to_bytes(12, "little"))
            qs = codes_to_qs([codes[col][sb * QK_K + i] for i in range(QK_K)])
            raw += q4k_pack_block(d_h[col][sb], dm_h[col][sb], scales12, qs)
    return bytes(raw)

def roundtrip_check(buf, out_dir=None):
    all_words, manifest, weights, tail = pack_gguf(buf, out_dir)
    if out_dir:                               # prove the on-disk hex is faithful
        with open(os.path.join(out_dir, "weight_mem_q4k.hex")) as f:
            all_words = [int(l.strip(), 16) for l in f if l.strip()]
    by_name = {d["name"]: d for d in manifest["tensors"]}
    ok = True
    for w in weights:
        desc = by_name[w["name"]]
        if desc["type"] == "Q4_K":
            recon = unpack_q4k_weight(all_words, desc, manifest["params"]["PE_N"])
        else:
            recon = _words_to_bytes(all_words[desc["base"]:], desc["nbytes"])
        if recon != w["raw"]:
            ok = False
            print(f"  ROUND-TRIP MISMATCH {w['name']} ({desc['type']}): "
                  f"{len(recon)} vs {len(w['raw'])} bytes; "
                  f"first diff @ {next((i for i in range(min(len(recon),len(w['raw']))) if recon[i]!=w['raw'][i]), -1)}")
    return ok, manifest, weights

# ============================================================================
# (7) DEQUANT cross-check -- packed Q4_K image dequantizes bit-exact to ggml
# ============================================================================
def dequant_crosscheck(buf):
    """Pack, unpack, and confirm the reconstructed Q4_K super-blocks dequantize
       bit-exact to q4k_ref.dequantize_block_q4_K (the ggml golden)."""
    all_words, manifest, weights, _ = pack_gguf(buf)
    by_name = {d["name"]: d for d in manifest["tensors"]}
    checked = 0
    for w in weights:
        if w["ttype"] != GGML_Q4_K:
            continue
        desc = by_name[w["name"]]
        recon = unpack_q4k_weight(all_words, desc, manifest["params"]["PE_N"])
        nb = w["K"] // QK_K
        for col in range(w["N"]):
            for sb in range(nb):
                o = (col * nb + sb) * 144
                a = ref.dequantize_block_q4_K(*q4k_unpack_block(w["raw"][o:o+144]))
                b = ref.dequantize_block_q4_K(*q4k_unpack_block(recon[o:o+144]))
                if np.frombuffer(a.tobytes(), np.uint32).tolist() != \
                   np.frombuffer(b.tobytes(), np.uint32).tolist():
                    return False, checked
                checked += 1
    return True, checked

# ============================================================================
# (8) FOOTPRINT moat guard -- Q4_K is ~44% fewer bytes/weight than FP8
# ============================================================================
def footprint(weights):
    """Return (q4k_bpw, fp8_bpw, reduction) over the Q4_K weights, plus a blended
       total across every quantized tensor.  FP8 image = 1 code byte/weight + one
       bf16 (2 B) block-scale per 128 weights = 1.015625 B/weight."""
    fp8_bpw = 1.0 + 2.0 / 128.0
    q4k_w, q4k_b, tot_w, tot_b = 0, 0, 0, 0
    for w in weights:
        nel = w["N"] * w["K"]
        epb, bpb = BLOCK[w["ttype"]]
        b = (nel // epb) * bpb
        tot_w += nel; tot_b += b
        if w["ttype"] == GGML_Q4_K:
            q4k_w += nel; q4k_b += b
    q4k_bpw = q4k_b / q4k_w                    # 144/256 = 0.5625
    reduction = 1.0 - q4k_bpw / fp8_bpw        # ~0.446
    blended_bpw = tot_b / tot_w
    blended_fp8 = tot_w * fp8_bpw
    return dict(q4k_bpw=q4k_bpw, fp8_bpw=fp8_bpw, reduction=reduction,
                q4k_bytes=q4k_b, blended_bpw=blended_bpw, total_bytes=tot_b,
                total_weights=tot_w, blended_fp8_bytes=blended_fp8,
                blended_reduction=1.0 - tot_b / blended_fp8)

# ============================================================================
# SELF-TEST
# ============================================================================
def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ckpt_pack_q4k_")
    gg = os.path.join(tmp, "synthetic.gguf")
    outd = os.path.join(tmp, "rtl")

    buf = gen_synthetic(gg)
    meta, tensors = read_gguf(buf)
    print(f"gen_synthetic: wrote {gg} ({len(buf)} bytes GGUF v{GGUF_VERSION}, "
          f"arch={meta.get('general.architecture')})")
    weights, tail = classify(tensors)
    for w in weights:
        print(f"  weight {w['name']}: {TYPE_NAME[w['ttype']]} [N={w['N']},K={w['K']}]")
    for t in tail:
        print(f"  tail   {t['name']}: {TYPE_NAME.get(t['ttype'], t['ttype'])} {t['dims']}")

    ok, manifest, weights = roundtrip_check(buf, outd)
    print(f"pack: {manifest['n_words']} words -> {outd}/weight_mem_q4k.hex "
          f"({len(manifest['tensors'])} tensors, manifest_q4k.json = per-tensor type map)")
    print(f"round-trip (pack -> unpack == original GGUF blocks, bit-exact): "
          f"{'PASS' if ok else 'FAIL'}")

    deq_ok, ndeq = dequant_crosscheck(buf)
    print(f"dequant cross-check (packed Q4_K == ggml dequantize_block_q4_K): "
          f"{'PASS' if deq_ok else 'FAIL'} ({ndeq} super-blocks)")

    fp = footprint(weights)
    print(f"footprint: Q4_K {fp['q4k_bpw']:.4f} B/wt vs FP8 {fp['fp8_bpw']:.4f} B/wt "
          f"-> {100*fp['reduction']:.1f}% smaller/weight "
          f"(Q4_K image {fp['q4k_bytes']} B)")
    print(f"           blended {fp['total_bytes']} B over {fp['total_weights']} weights "
          f"({fp['blended_bpw']:.4f} B/wt) vs FP8-equiv {int(fp['blended_fp8_bytes'])} B "
          f"-> {100*fp['blended_reduction']:.1f}% smaller")

    # ---- moat regression guard (2.2): Q4_K ~44% fewer bytes/weight than FP8 ----
    moat = (0.43 <= fp["reduction"] <= 0.46)
    print(f"moat guard (Q4_K >=43% smaller/weight than FP8): "
          f"{'PASS' if moat else 'FAIL'}")

    allok = ok and deq_ok and moat
    print("SELFTEST", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "gen":
        gen_synthetic(sys.argv[2])
        print(f"wrote synthetic GGUF -> {sys.argv[2]}")
    elif len(sys.argv) >= 4 and sys.argv[1] == "pack":
        with open(sys.argv[2], "rb") as f:
            buf = f.read()
        _, man, _, _ = pack_gguf(buf, sys.argv[3])
        print(f"packed {man['n_words']} words -> {sys.argv[3]}/weight_mem_q4k.hex")
    elif len(sys.argv) >= 3 and sys.argv[1] == "check":
        with open(sys.argv[2], "rb") as f:
            buf = f.read()
        ok, _, _ = roundtrip_check(buf, sys.argv[2] + "_rtl")
        print("round-trip", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    else:
        sys.exit(_selftest())
