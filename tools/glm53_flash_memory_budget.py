#!/usr/bin/env python3
"""
glm53_flash_memory_budget.py -- what GLM-5.3-Flash actually needs to sit
resident, and what that does to the hardware ladder.

WHY THIS EXISTS.  docs/HARDWARE_LADDER.md sizes every rung against GLM-5.2:
467 GB of weights and ~94 GB of KV at 1M context, which is where the 512 GB
LPDDR5X rung-3 point and the 512 GB HBF + 96 GB HBM rung-4 split both come
from.  GLM-5.3-Flash changes BOTH terms, and not by the same factor -- weights
fall 2.3x but KV falls 7.6x, because only 11 of its 45 layers cache at all and
NoPE strips the rotary tail off the cached latent.  Re-quoting the old rung
sizes for this model would be wrong in a way that looks conservative.

Model constants are PARSED OUT OF configs/full_glm53_flash.vh, not retyped, so
this tool and the locked config cannot drift apart.

EVIDENCE CLASSES -- kept separate on purpose:
  [measured] weight bytes: the GGUF tensor census (tools/glm53_flash_gguf_scan.py),
             cross-checked against the published shard sizes
  [derived]  KV / indexer / recurrent-state bytes: arithmetic on the locked
             config under the precision assumptions named below
  [EST]      everything vendor-side: per-stack bandwidth and capacity, pJ/bit,
             and therefore every tok/s and watt figure printed here

usage: python3 tools/glm53_flash_memory_budget.py [--weights-gb 199.70]
"""
import argparse
import os
import re
import sys

GB, TB = 1e9, 1e12
GiB = float(1 << 30)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "configs", "full_glm53_flash.vh")

# ---- ASSUMPTIONS, all [derived]-class. Change here, not inline. ------------
KV_BYTES = 2      # cached MLA latent element width (fp16/bf16)
IDX_BYTES = 1     # pooled indexer key element width (int8-class)
STATE_BYTES = 4   # KDA delta-rule state kept fp32 for stability
# The indexer is modelled as scoring EVERY pooled position once per token.
# GLM-5.2 had an index_topk_freq to amortize that; GLM-5.3-Flash's config
# publishes no such field, so no amortization is assumed. If a future reading
# of the reference implementation shows the indexer runs less often, the
# indexer rows below are an upper bound.

# ---- vendor-side [EST]: capacity and bandwidth per package/stack ----------
# LPDDR5X: bus width follows PACKAGE COUNT (x64 per package), not capacity.
# HBF: ~1.6 TB/s per stack is the announced figure; per-stack capacity is
#      INFERRED from HARDWARE_LADDER's "512 GB HBF, 2-stack base" => ~256 GB.
# HBM: public-roadmap class figures, not vendor commitments.
TIERS = {
    "LPDDR5X": dict(unit="package", cap_gb=[16, 32], bw_per=1100 / 16 / 1000, nom=16),
    "HBF":     dict(unit="stack",   cap_gb=[256],    bw_per=1.6,              nom=2),
    "HBM3E":   dict(unit="stack",   cap_gb=[36],     bw_per=1.2,              nom=6),
    "HBM4":    dict(unit="stack",   cap_gb=[64],     bw_per=2.0,              nom=4),
}
PJ_BIT = (4e-12, 7e-12)   # DRAM-class memory-system energy [EST]


def load_cfg(path):
    """Parse `define GLM53F_<NAME> <int> out of the locked config header."""
    txt = open(path).read()
    cfg = {}
    for name, val in re.findall(r"^`define\s+GLM53F_(\w+)\s+(\d+)\s*(?://.*)?$",
                                txt, re.M):
        cfg[name] = int(val)
    need = ["N_MLA", "N_KDA", "L", "KV_LORA", "ROPE", "IDX_DIM", "IDX_KPOOL",
            "KDA_HEADS", "KDA_DIM", "KDA_CONV_K", "CTX", "MODEL_DIM"]
    missing = [k for k in need if k not in cfg]
    if missing:
        sys.exit(f"config header is missing {missing} -- has {path} been edited?")
    return cfg


def residency(c, ctx, weights_b):
    """Bytes that must be resident at a given context length."""
    # MLA: DeepSeek-style -- the compressed latent is cached once per token per
    # MLA layer. ROPE is 0 here, so there is no rotary tail on top of it.
    kv = c["N_MLA"] * ctx * (c["KV_LORA"] + c["ROPE"]) * KV_BYTES
    # DSA indexer keys, pooled by kpool
    idx = c["N_MLA"] * (ctx // c["IDX_KPOOL"]) * c["IDX_DIM"] * IDX_BYTES
    # KDA: the delta-rule state is head_dim x head_dim per head, plus the short
    # conv history. CONSTANT in context -- this is the whole point of the hybrid.
    hd, nh, k = c["KDA_DIM"], c["KDA_HEADS"], c["KDA_CONV_K"]
    state = c["N_KDA"] * (nh * hd * hd * STATE_BYTES
                          + 3 * (k - 1) * nh * hd * KV_BYTES)
    return dict(weights=weights_b, kv=kv, idx=idx, state=state,
                total=weights_b + kv + idx + state)


def indexer_macs(c, ctx):
    """MACs/token spent scoring pooled positions (see the amortization note)."""
    return c["N_MLA"] * (ctx // c["IDX_KPOOL"]) * c["IDX_HEADS"] * c["IDX_DIM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-gb", type=float, default=199.70,
                    help="[measured] UD-Q4_K_XL weight bytes, from the GGUF census")
    ap.add_argument("--active-params", type=float, default=16.742e9,
                    help="[measured] active params/token, from the GGUF census")
    ap.add_argument("--raw-gb-per-tok", type=float, default=14.118,
                    help="[measured] weight bytes read per token, from the GGUF census")
    a = ap.parse_args()
    c = load_cfg(CFG)
    c.setdefault("IDX_HEADS", 32)
    W = a.weights_gb * GB

    print(f"GLM-5.3-Flash residency budget   (config: configs/full_glm53_flash.vh)")
    print(f"  {c['L']} layers = {c['N_KDA']} KDA + {c['N_MLA']} MLA+DSA   "
          f"kv_lora={c['KV_LORA']}  rope={c['ROPE']} (NoPE)  kpool={c['IDX_KPOOL']}")
    print(f"  assumptions: KV {KV_BYTES}B, indexer key {IDX_BYTES}B, "
          f"KDA state {STATE_BYTES}B\n")

    print("=== 1. what must be resident, by context length ===")
    print(f"{'context':>10} {'weights':>10} {'KV':>10} {'DSA idx':>9} "
          f"{'KDA state':>10} {'TOTAL':>10} {'TOTAL':>10}")
    print(f"{'':>10} {'[measured]':>10} {'[derived]':>10} {'[derived]':>9} "
          f"{'[derived]':>10} {'(GB)':>10} {'(GiB)':>10}")
    print("-" * 74)
    rows = {}
    for ctx in (32768, 131072, 262144, 524288, c["CTX"]):
        r = residency(c, ctx, W)
        rows[ctx] = r
        print(f"{ctx:>10,} {r['weights']/GB:>9.1f} {r['kv']/GB:>9.2f} "
              f"{r['idx']/GB:>8.2f} {r['state']/GB:>9.3f} "
              f"{r['total']/GB:>9.1f} {r['total']/GiB:>9.1f}")
    full = rows[c["CTX"]]
    print(f"\n  Only {c['N_MLA']} of {c['L']} layers grow a KV cache; the other "
          f"{c['N_KDA']} carry {full['state']/1e6:.0f} MB of state that does NOT "
          f"grow with context.")

    print("\n=== 2. the coupling that decides a config: bandwidth is PER UNIT ===")
    print("  Capacity fell, so the temptation is to fit the model in fewer units.")
    print("  Bus width / aggregate bandwidth follows the UNIT COUNT, not the capacity,")
    print("  so units are kept for BANDWIDTH. Halving them halves throughput.\n")
    need = full["total"]
    print(f"{'tier':>9} {'units':>6} {'cap/unit':>9} {'capacity':>9} {'fits?':>6} "
          f"{'bandwidth':>11} {'tok/s [EST]':>12}")
    print("-" * 70)
    for name, t in TIERS.items():
        for cap in t["cap_gb"]:
            for n in sorted({1, 2, 4, 8, t["nom"], 16}):
                if name == "LPDDR5X" and n not in (8, 16):
                    continue
                if name != "LPDDR5X" and n > 8:
                    continue
                total_cap = n * cap
                if total_cap * GB < need:
                    continue
                bw = n * t["bw_per"] * TB
                # HARDWARE_LADDER deliberately refuses to quote unit counts
                # past the design point ("arithmetically possible but push into
                # chiplet/kW territory -- a different product bracket").  Print
                # them, but never unlabelled: a table that silently manufactures
                # a 1133 tok/s headline is the overclaim this repo exists to avoid.
                mark = (" <- design point" if n == t["nom"]
                        else "    beyond the quoted bracket" if n > t["nom"] else "")
                print(f"{name:>9} {n:>6} {cap:>7} GB {total_cap:>7} GB "
                      f"{'yes':>6} {bw/TB:>8.2f} TB/s "
                      f"{bw/GB/a.raw_gb_per_tok:>11.0f}{mark}")

    print(f"\n  Reference points: GLM-5.2 needs 467 GB weights + ~94 GB KV = ~561 GB,")
    print(f"  so an all-HBM residency was arithmetically out of reach for it and is")
    print(f"  reachable here ({need/GB:.0f} GB) -- a genuinely new option, at a cost")
    print(f"  priced in section 4.")

    # ---- bytes actually READ per token, not just weights --------------------
    # The roofline in HARDWARE_LADDER divides weight bytes by bandwidth. On this
    # model that under-counts, because two NEW terms are read every token:
    #   * the KDA recurrent state -- read AND written once per layer per token.
    #     It does not grow with context, but it is not free either.
    #   * the DSA indexer's pooled key cache -- the indexer scores every pooled
    #     position, so this term grows with context and, at 1M, is larger than
    #     the KV the attention itself reads.
    print("\n=== 3. bytes READ per token (the honest roofline denominator) ===")
    print(f"{'context':>10} {'weights':>10} {'KDA state':>10} {'DSA KV':>9} "
          f"{'indexer':>9} {'TOTAL':>10} {'tok/s @1.1TB/s':>15}")
    print("-" * 78)
    topk_attn = c.get("TOPK_ATTN", 2048)   # [gguf] attention.indexer.top_k, via the config header
    for ctx in (32768, 131072, 262144, c["CTX"]):
        w_b = a.raw_gb_per_tok * GB
        # state: read + write, every KDA layer, every token
        st_b = 2 * (c["N_KDA"] * c["KDA_HEADS"] * c["KDA_DIM"] * c["KDA_DIM"] * STATE_BYTES)
        # DSA reads only the top-k selected positions' latents
        kv_b = c["N_MLA"] * min(topk_attn, ctx) * (c["KV_LORA"] + c["ROPE"]) * KV_BYTES
        # the indexer must READ every pooled key to score it
        ix_b = c["N_MLA"] * (ctx // c["IDX_KPOOL"]) * c["IDX_DIM"] * IDX_BYTES
        tot = w_b + st_b + kv_b + ix_b
        print(f"{ctx:>10,} {w_b/GB:>9.3f} {st_b/GB:>9.3f} {kv_b/GB:>8.3f} "
              f"{ix_b/GB:>8.3f} {tot/GB:>9.3f} {1100*GB/tot:>14.1f}")
    print("  Weights dominate at every context, so the bandwidth-bound model holds --")
    print("  but the total is 2-5% above the weights-only figure, and the indexer term")
    print("  is what grows. These tok/s are UNAMORTIZED (no speculative decode).")

    print("\n=== 4. where the binding constraint leaves memory: the die ===")
    print(f"{'tok/s':>8} {'MAC/s':>10} {'lanes @1GHz':>13} {'lanes @2GHz':>13}")
    print("-" * 48)
    for t in (78, 227, 453, 510, 567):
        m = t * a.active_params
        print(f"{t:>8} {m/1e12:>7.1f} T {m/1e9:>12,.0f} {m/2e9:>12,.0f}")
    print("  Measured lane scaling on this RTL is SUBLINEAR (4x lanes -> ~2.40x),")
    print("  so these are lower bounds on the silicon, not a shopping list.")

    print("\n=== 5. two things that make the high-bandwidth rows optimistic ===")
    print("  (a) the DSA indexer stops being free at long context:")
    wm = a.active_params
    for ctx in (32768, 131072, 262144, c["CTX"]):
        im = indexer_macs(c, ctx)
        print(f"      ctx {ctx:>9,}: indexer {im/1e9:>6.2f} G MAC/tok = "
              f"{im/wm*100:>5.1f}% of the weight MACs")
    xo = wm / (c["N_MLA"] * c["IDX_HEADS"] * c["IDX_DIM"] / c["IDX_KPOOL"])
    print(f"      -> parity at ctx ~= {xo/1e6:.2f} M tokens. Below ~128K this is noise;")
    print(f"         at {c['CTX']:,} the 'weight-bandwidth-bound' model no longer holds,")
    print(f"         and every tok/s above is derived from exactly that model.")
    print("\n  (b) the memory rail's own power [EST], DRAM-class pJ/bit:")
    for label, bw in [("LPDDR5X 16-package", 1.1), ("HBF 2-stack", 3.2),
                      ("HBM3E 6-stack", 7.2), ("HBM4 4-stack", 8.0)]:
        bits = bw * TB * 8
        note = "   (HBF pJ/bit is UNPUBLISHED -- DRAM math, not an HBF claim)" \
               if "HBF" in label else ""
        print(f"      {label:20} {bw:>4.1f} TB/s -> "
              f"{bits*PJ_BIT[0]:>4.0f}-{bits*PJ_BIT[1]:>4.0f} W memory only{note}")
    print("      The appliance envelope in docs/HARDWARE_LADDER.md is >=50-78 W TOTAL,")
    print("      so the top rows are a different product bracket, not a faster appliance.")

    print("\n  No speculative-decode amortization is applied anywhere above: this model's")
    print("  MTP accept rate is unmeasured (docs/GLM53_FLASH_PORT.md 4.3), so every")
    print("  tok/s here is the UNAMORTIZED figure and would improve once it is measured.")


if __name__ == "__main__":
    main()
