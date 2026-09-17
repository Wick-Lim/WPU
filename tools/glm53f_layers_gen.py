#!/usr/bin/env python3
"""
glm53f_layers_gen.py -- the expected layer schedule for test/glm53f_layers_tb.v
(src/glm53f_layers.v: the 45-layer walk).

THE SCHEDULE IS READ FROM THE LOCKED CONFIG, not written here. The RTL takes
ATTN_PERIOD / ATTN_OFFSET / N_DENSE as parameters and this generator takes them
from configs/full_glm53_flash.vh, so both sides trace to the same cited line and a
disagreement is a real disagreement rather than two independent guesses.

    attention: block l is MLA+DSA  iff  (l % ATTN_PERIOD) == ATTN_OFFSET
               [gguf] attention.head_count_kv is a per-block list; [scan] confirms
               it is strictly periodic -> { 3, 7, 11, ..., 43 }
    FFN:       block l is MoE      iff  l >= N_DENSE
               [gguf] leading_dense_block_count = 3

WHAT THIS GATE CLAIMS, AND WHAT IT DOES NOT. It claims the WALK: that the block is
invoked exactly L times, that the streams are loaded exactly once before the first
one, and that the (layer, attn_sel, ffn_sel) triple on each invocation is the
checkpoint's schedule. It does NOT re-claim what one layer computes -- that is
`make dec-block`, which also proves (via its KIND=2 equivalence build) that the
selectors actually select. The three together are the argument: one layer is right,
selection works, and the schedule picks the right selection per layer.
"""
import os
import re
import sys

CFG = "configs/full_glm53_flash.vh"


def load_cfg(path=CFG):
    with open(path) as fh:
        txt = fh.read()
    cfg = {n: int(v) for n, v in
           re.findall(r"^`define\s+GLM53F_(\w+)\s+(\d+)\s*(?://.*)?$", txt, re.M)}
    need = ["L", "N_DENSE", "ATTN_PERIOD", "ATTN_OFFSET", "N_KDA", "N_MLA"]
    missing = [k for k in need if k not in cfg]
    if missing:
        sys.exit(f"{path} is missing {missing}")
    return cfg


def schedule(L, n_dense, period, offset):
    """(layer, attn_is_mla, ffn_is_moe) for every layer, in order."""
    return [(l, (l % period) == offset, l >= n_dense) for l in range(L)]


def gen(cfg=None, report=True):
    c = cfg or load_cfg()
    sch = schedule(c["L"], c["N_DENSE"], c["ATTN_PERIOD"], c["ATTN_OFFSET"])
    out = [f"{c['L']} {c['N_DENSE']} {c['ATTN_PERIOD']} {c['ATTN_OFFSET']}"]
    for l, mla, moe in sch:
        out.append(f"{l} {int(mla)} {int(moe)}")
    if report:
        mla = [l for l, m, _ in sch if m]
        print(f"layers: L={c['L']}  MLA blocks {mla[:4]}...{mla[-1]} "
              f"({len(mla)} of {c['L']}), dense front {c['N_DENSE']}",
              file=sys.stderr)
    return "\n".join(out) + "\n"


def _selftest():
    n = 0
    fails = []

    def chk(cond, m):
        nonlocal n
        n += 1
        if not cond:
            fails.append(m)

    c = load_cfg()
    sch = schedule(c["L"], c["N_DENSE"], c["ATTN_PERIOD"], c["ATTN_OFFSET"])
    mla = [l for l, m, _ in sch if m]
    moe = [l for l, _, e in sch if e]

    # --- the schedule must REPRODUCE the census counts, which are cited separately
    chk(len(mla) == c["N_MLA"],
        f"the periodic schedule yields {len(mla)} MLA blocks but [scan] says "
        f"N_MLA = {c['N_MLA']} -- the period/offset and the count disagree")
    chk(c["L"] - len(mla) == c["N_KDA"],
        f"KDA count {c['L'] - len(mla)} != [scan] N_KDA = {c['N_KDA']}")
    chk(mla[0] == c["ATTN_OFFSET"], "the first MLA block is not at the cited offset")
    chk(all(b - a == c["ATTN_PERIOD"] for a, b in zip(mla, mla[1:])),
        "the MLA blocks are not evenly spaced")

    # --- the dense front is a PREFIX, and the two schedules are INDEPENDENT ---
    chk(moe == list(range(c["N_DENSE"], c["L"])), "the MoE blocks are not the suffix")
    chk(all(not m for l, m, _ in sch if l < c["N_DENSE"]),
        "a dense-front block is MLA -- the first MLA block must be at or after the front")
    kinds = {(m, e) for _, m, e in sch}
    chk(len(kinds) >= 3,
        f"only {len(kinds)} attention x FFN combinations occur; the walk would not "
        f"exercise the block's arms")

    # --- the corpus must SEE the injections the TB carries, and the OFF-BY-ONE
    #     one has to be the QUIET kind: same count, different blocks. Shifting the
    #     phase by one keeps 11 MLA blocks at L = 45 ({2,6,..,42} vs {3,7,..,43}),
    #     so the census figures still agree and only the SEQUENCE disagrees.
    #     Dropping the phase entirely gives 12 and a count check would catch it --
    #     which is why the injection is the shift, not the drop. ---
    off1 = schedule(c["L"], c["N_DENSE"], c["ATTN_PERIOD"], c["ATTN_OFFSET"] - 1)
    chk(off1 != sch, "an off-by-one offset gives the same schedule -- "
                     "INJ_LAYERS_OFF_BY_ONE could not fail")
    chk(len([l for l, m, _ in off1 if m]) == len(mla),
        f"an off-by-one offset changes the MLA COUNT "
        f"({len([l for l, m, _ in off1 if m])} vs {len(mla)}) -- then a count check "
        f"would catch it and the injection is not testing the sequence")
    chk(any(not e for _, _, e in sch),
        "no dense block at all -- INJ_LAYERS_DENSE_OFF could not fail")

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_layers_gen self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (the periodic schedule reproduces [scan]'s "
          f"N_MLA={c['N_MLA']} / N_KDA={c['N_KDA']} exactly, the MLA blocks start at "
          f"the cited offset and are evenly spaced, the dense front is a prefix, at "
          f"least three attention x FFN combinations occur, and an off-by-one offset "
          f"changes WHICH blocks are MLA without changing HOW MANY -- so the census "
          f"counts still agree and only the sequence disagrees)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    outp = a[0] if a else "build/glm53f_layers_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen())
    print(f"wrote {outp}")
