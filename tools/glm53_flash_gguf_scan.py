#!/usr/bin/env python3
"""
glm53_flash_gguf_scan.py -- re-derive EVERY GLM-5.3-Flash number this branch
cites, straight from the published GGUF's own bytes.

The branch contract is "read the published checkpoint bit-exactly", so the GGUF
metadata -- not a blog post, not config.json -- is the authority for the model
shape.  This script is what makes docs/GLM53_FLASH_PORT.md and
configs/full_glm53_flash.vh reproducible instead of assertions:

    tensor map  -> which layers are KDA vs MLA+DSA, and what each one holds
    quant mix   -> which ggml types the UD-Q4_K_XL build actually uses
    byte total  -> cross-checked against the real shard sizes (the honest test
                   that the parse is right: if the block-size math is wrong the
                   total will not land on the published file sizes)

It reads only the GGUF *headers* -- shard 1 in full (metadata-only, ~9 MB) and
the first few MB of shards 2..6 (KV + tensor-info section) -- so a full scan
costs ~30 MB, not 200 GB.

usage:
    python3 tools/glm53_flash_gguf_scan.py --fetch <workdir>   # download headers
    python3 tools/glm53_flash_gguf_scan.py <workdir>           # scan + report
"""
import collections
import json
import os
import re
import struct
import subprocess
import sys

REPO_URL = "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF"
QUANT = "UD-Q4_K_XL"
N_SHARDS = 6
HDR_BYTES = 4 << 20  # tensor-info section of a shard fits well inside 4 MiB

# published shard sizes (HTTP content-length), the cross-check target
SHARD_BYTES = [9429859, 49198039200, 49988626304, 48700701344, 49026026752, 2784497888]

# ---- GGUF v3 primitives -----------------------------------------------------
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FMT = {U8: ('<B', 1), I8: ('<b', 1), U16: ('<H', 2), I16: ('<h', 2),
        U32: ('<I', 4), I32: ('<i', 4), F32: ('<f', 4), BOOL: ('<?', 1),
        U64: ('<Q', 8), I64: ('<q', 8), F64: ('<d', 8)}
GGML_TYPE = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
             8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
             13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS',
             18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
             23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64',
             29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4'}
# bytes per ggml block / weights per block -- the k-quant format constants
BLOCK = {'F32': (4, 1), 'F16': (2, 1), 'BF16': (2, 1), 'Q8_0': (34, 32),
         'Q4_K': (144, 256), 'Q5_K': (176, 256), 'Q6_K': (210, 256)}


class _R:
    def __init__(self, b):
        self.b, self.o = b, 0

    def raw(self, n):
        v = self.b[self.o:self.o + n]
        if len(v) < n:
            raise EOFError("header truncated -- raise HDR_BYTES")
        self.o += n
        return v

    def sc(self, t):
        f, n = _FMT[t]
        return struct.unpack(f, self.raw(n))[0]

    def st(self):
        return self.raw(self.sc(U64)).decode('utf-8', 'replace')

    def val(self, t):
        if t == STR:
            return self.st()
        if t == ARR:
            et, n = self.sc(U32), self.sc(U64)
            return [self.val(et) for _ in range(n)]
        return self.sc(t)


def read_header(path):
    """Parse a GGUF header. Returns (version, kv, tensor_infos)."""
    with open(path, 'rb') as f:
        b = f.read()
    r = _R(b)
    if r.raw(4) != b'GGUF':
        raise ValueError(f"{path}: not a GGUF file")
    ver, n_tensors, n_kv = r.sc(U32), r.sc(U64), r.sc(U64)
    kv = {}
    for _ in range(n_kv):
        k = r.st()
        kv[k] = r.val(r.sc(U32))
    tensors = []
    for _ in range(n_tensors):
        name, nd = r.st(), r.sc(U32)
        dims = [r.sc(U64) for _ in range(nd)]
        tensors.append({'name': name, 'dims': dims,
                        'type': GGML_TYPE.get(r.sc(U32), '?'), 'offset': r.sc(U64)})
    return ver, kv, tensors


def shard_url(i):
    return f"{REPO_URL}/resolve/main/{QUANT}/GLM-5.3-Flash-{QUANT}-{i:05d}-of-{N_SHARDS:05d}.gguf"


def fetch(workdir):
    os.makedirs(workdir, exist_ok=True)
    for i in range(1, N_SHARDS + 1):
        dst = os.path.join(workdir, f"hdr{i}.gguf")
        if os.path.exists(dst):
            print(f"  hdr{i}: cached")
            continue
        # shard 1 is the metadata-only shard -- take it whole; the rest header-only
        rng = [] if i == 1 else ["-r", f"0-{HDR_BYTES - 1}"]
        print(f"  hdr{i}: fetching {'(full metadata shard)' if i == 1 else '(header range)'}")
        subprocess.run(["curl", "-sL", "--max-time", "300", *rng, "-o", dst, shard_url(i)],
                       check=True)
    print(f"fetched into {workdir}")


def numel(t):
    n = 1
    for d in t['dims']:
        n *= d
    return n


def nbytes(t):
    bpb, wpb = BLOCK[t['type']]
    return numel(t) * bpb // wpb


def scan(workdir):
    ver, kv, _ = read_header(os.path.join(workdir, "hdr1.gguf"))
    tensors = []
    for i in range(2, N_SHARDS + 1):
        tensors += read_header(os.path.join(workdir, f"hdr{i}.gguf"))[2]

    arch = kv['general.architecture']
    print(f"=== {kv['general.name']} :: {QUANT} ===")
    print(f"GGUF v{ver}   arch={arch}   size_label={kv['general.size_label']}   "
          f"quantized_by={kv.get('general.quantized_by')}")

    print("\n--- model shape (GGUF metadata is the authority) ---")
    for k in sorted(kv):
        if k.startswith(f"{arch}.") and not isinstance(kv[k], list):
            print(f"  {k:52s} {kv[k]}")

    kvh = kv[f'{arch}.attention.head_count_kv']
    kda_blocks = [i for i, v in enumerate(kvh) if v == 0]
    mla_blocks = [i for i, v in enumerate(kvh) if v != 0]
    print(f"\n--- layer schedule (head_count_kv: 0 = KDA linear-attn, 1 = MLA+DSA) ---")
    print(f"  blocks total : {len(kvh)}  (= {kv[f'{arch}.block_count']}, "
          f"incl. {kv[f'{arch}.nextn_predict_layers']} MTP block)")
    print(f"  KDA          : {len(kda_blocks)}  {kda_blocks}")
    print(f"  MLA+DSA      : {len(mla_blocks)}  {mla_blocks}")

    print(f"\n--- tensor census: {len(tensors)} tensors "
          f"(GGUF split.tensors.count={kv.get('split.tensors.count')}) ---")
    if len(tensors) != kv.get('split.tensors.count'):
        print("  !! MISMATCH -- header parse incomplete")

    fam = collections.defaultdict(collections.Counter)
    ex = {}
    for t in tensors:
        k = re.sub(r'\.\d+\.', '.N.', t['name'])
        fam[k][t['type']] += 1
        ex.setdefault(k, t)
    for k in sorted(fam):
        c = fam[k]
        mix = ' '.join(f"{ty}x{n}" for ty, n in c.most_common())
        print(f"  {k:44s} {str(ex[k]['dims']):24s} {mix}")

    print("\n--- quant mix (the dequant contract this branch must satisfy) ---")
    per = collections.defaultdict(lambda: [0, 0, 0])   # tensors, params, bytes
    for t in tensors:
        p = per[t['type']]
        p[0] += 1
        p[1] += numel(t)
        p[2] += nbytes(t)
    tot_b = sum(v[2] for v in per.values())
    tot_p = sum(v[1] for v in per.values())
    for ty, (n, p, b) in sorted(per.items(), key=lambda kv_: -kv_[1][2]):
        print(f"  {ty:6s} tensors={n:5d}  params={p / 1e9:8.3f} B  "
              f"bytes={b / 1e9:7.2f} GB ({b / tot_b * 100:5.1f}%)")
    print(f"  {'TOTAL':6s} tensors={len(tensors):5d}  params={tot_p / 1e9:8.3f} B  "
          f"bytes={tot_b / 1e9:7.2f} GB")

    published = sum(SHARD_BYTES)
    delta = published - tot_b
    print(f"\n  published shard bytes : {published / 1e9:7.2f} GB")
    print(f"  tensor-data accounted : {tot_b / 1e9:7.2f} GB")
    print(f"  delta (GGUF headers)  : {delta / 1e6:7.2f} MB  "
          f"-> {'OK, parse is consistent' if 0 <= delta < 40e6 else 'MISMATCH'}")

    # active params/token: top-k of the routed experts + every dense weight,
    # MTP block excluded (it only runs when speculating)
    topk, n_exp = kv[f'{arch}.expert_used_count'], kv[f'{arch}.expert_count']
    mtp_block = len(kvh) - 1
    act = mtp_act = mtp_tot = 0
    act_b = mtp_act_b = 0
    d_model = kv[f'{arch}.embedding_length']
    for t in tensors:
        n, nb = numel(t), nbytes(t)
        # a routed-expert tensor contributes only the top-k slice it actually reads
        exp = '_exps.' in t['name']
        a = n * topk // n_exp if exp else n
        ab = nb * topk // n_exp if exp else nb
        # token_embd is a ROW LOOKUP, not a GEMV: one token reads one row of it.
        # (output.weight / lm_head is different -- that one really is read whole
        # every token, and tie_word_embeddings is false so they are two tensors.)
        if t['name'] == 'token_embd.weight':
            a = d_model
            ab = nb * d_model // n
        if t['name'].startswith(f"blk.{mtp_block}."):
            mtp_act += a
            mtp_act_b += ab
            mtp_tot += n
        else:
            act += a
            act_b += ab
    print(f"\n--- active parameters / token ---")
    print(f"  total (all weights)      : {tot_p / 1e9:8.3f} B")
    print(f"  active, 1 token          : {act / 1e9:8.3f} B   "
          f"(top-{topk}/{n_exp} routed + every dense weight; token_embd counted as "
          f"one row; MTP block excluded)")
    print(f"  MTP block {mtp_block}: active     : {mtp_act / 1e9:8.3f} B   "
          f"(adds only while speculating; block holds {mtp_tot / 1e9:.3f} B total)")
    print(f"  active incl. MTP         : {(act + mtp_act) / 1e9:8.3f} B")
    # The roofline lever: tok/s ~= memory bandwidth / bytes-read-per-token.  This is
    # the *byte* figure, not the parameter figure -- the two differ here because the
    # routed experts are the 4-5 bit tensors while attention is Q8_0.
    print(f"\n--- weight BYTES read per token (the roofline denominator) ---")
    print(f"  1 token, no speculation  : {act_b / 1e9:8.3f} GB")
    print(f"  MTP block adds           : {mtp_act_b / 1e9:8.3f} GB")
    print(f"  1 token incl. MTP        : {(act_b + mtp_act_b) / 1e9:8.3f} GB")

    print("\n--- RTL coverage gap on this branch ---")
    # what src/ + tools/q4k_ref.py implement. Q5_K landed on this branch:
    # reference (q4k_ref.dequantize_block_q5_K, ggml-verbatim), RTL (WT_Q5K in
    # glm_matmul_q4k, 5-bit code off the existing w_hp bus), and a gate with a
    # must-fail injection (`make mixedtype`).
    have = {'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0', 'F32', 'F16'}
    missing = {ty for ty in per if ty not in have}
    for ty in sorted(missing):
        n, p, b = per[ty]
        print(f"  !! {ty}: NO dequant kernel in this repo -- {n} tensors, "
              f"{b / 1e9:.2f} GB ({b / tot_b * 100:.1f}% of the checkpoint)")
    print(f"  !! KDA linear attention: NO RTL -- {len(kda_blocks)}/{len(kvh) - 1} "
          f"of the model's layers")
    print(f"  !! hyper-connections (mult={kv.get(f'{arch}.hyper_connection.count')}, "
          f"sinkhorn={kv.get(f'{arch}.hyper_connection.sinkhorn_iterations')}): NO RTL")
    if not missing:
        print("  (dequant mix fully covered)")
    return 0


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == '--fetch':
        fetch(a[1])
    else:
        sys.exit(scan(a[0]))
