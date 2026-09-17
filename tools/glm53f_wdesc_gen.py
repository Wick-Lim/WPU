#!/usr/bin/env python3
"""
glm53f_wdesc_gen.py -- vectors and parameters for test/glm53f_wdesc_tb.v
(src/glm53f_wdesc.v: the per-tensor weight descriptor).

    (kind, layer, expert) -> (base, klen, nsblk, wtype)

THE TABLE'S SHAPE IS THE CHECKPOINT'S, not a compression trick. [scan] describes
the quantisation mix as a regular assignment per tensor kind plus a handful of "UD
bumps" on named blocks -- `ffn_{gate,up}_exps` Q4_K x42 + Q5_K x1, `ffn_down_exps`
Q5_K x40 + **Q6_K on blk.{11,12,44}**. So the RTL carries a per-KIND row (base,
layer stride, expert stride, klen, nsblk, default type) and a short list of
(kind, layer) -> type EXCEPTIONS, which is exactly what that sentence is.

WHAT THE CORPUS HAS TO CONTAIN, and the generator asserts:
  * at least one kind whose type is overridden on SOME layers and not others --
    otherwise the exception mechanism is never exercised and a descriptor that
    ignored it would pass;
  * at least one kind with a non-zero EXPERT stride, and more than one expert
    queried -- otherwise dropping the stride is invisible;
  * at least one kind with a non-zero LAYER stride, queried on more than one layer.

THE REAL VALUES ARE NOT HERE, and that is deliberate. Turning the census into
actual byte offsets needs the GGUF tensor map, which needs the 199.7 GB checkpoint
(or at least its headers) -- not available on this branch. What this gates is the
MACHINE: that a (kind, layer, expert) triple is resolved by the documented
arithmetic and that the exceptions win. The table it is fed here is synthetic and
shaped like the real one.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WT_Q4K, WT_Q6K, WT_Q80, WT_F16, WT_Q5K = 0, 1, 2, 3, 4

# kind id -> (name, base, layer stride, expert stride, klen, nsblk, default type)
KINDS = [
    ("attn_q",        0x00000, 0x1000,      0, 256, 1, WT_Q80),
    ("attn_out",      0x40000, 0x1000,      0, 256, 1, WT_Q80),
    ("ffn_gate",      0x80000, 0x2000,      0, 256, 1, WT_Q80),   # dense front
    ("ffn_gate_exps", 0xC0000, 0x8000, 0x0400, 256, 1, WT_Q4K),   # per EXPERT
    ("ffn_down_exps", 0x180000, 0x8000, 0x0400, 256, 1, WT_Q5K),  # UD-bumped below
    ("ffn_gate_inp",  0x240000, 0x0100,      0,  32, 1, WT_F16),
    ("attn_norm",     0x250000, 0x0040,      0,  16, 1, WT_F16),
    ("shexp_down",    0x260000, 0x1000,      0, 256, 1, WT_Q80),
]
# the UD bumps: (kind name, layer) -> type. Shaped like blk.{11,12,44}.ffn_down_exps
EXCEPTIONS = [("ffn_down_exps", 1, WT_Q6K), ("ffn_down_exps", 3, WT_Q6K)]

NLAYER = 4
NEXP = 4


def kind_id(name):
    return [k[0] for k in KINDS].index(name)


def descriptor(kind, layer, expert):
    """The documented arithmetic, in Python. The RTL must agree bit for bit."""
    _, base, lstr, estr, klen, nsblk, wt = KINDS[kind]
    for xn, xl, xw in EXCEPTIONS:
        if kind_id(xn) == kind and xl == layer:
            wt = xw
    return base + lstr * layer + estr * expert, klen, nsblk, wt


def _packed(values, width):
    """Pack low-index-first into one Verilog literal, as the RTL indexes it."""
    v = 0
    for i, x in enumerate(values):
        assert 0 <= x < (1 << width), f"{x} does not fit in {width} bits"
        v |= x << (i * width)
    return v


def params(ADDR_W=32, KW=16, SBW=8):
    """The parameter block the TB passes to the DUT."""
    nk = len(KINDS)
    kidw = max(1, (nk - 1).bit_length())
    lidw = max(1, (NLAYER - 1).bit_length())
    out = {
        "K_BASE":  (_packed([k[1] for k in KINDS], ADDR_W), nk * ADDR_W),
        "K_LSTR":  (_packed([k[2] for k in KINDS], ADDR_W), nk * ADDR_W),
        "K_ESTR":  (_packed([k[3] for k in KINDS], ADDR_W), nk * ADDR_W),
        "K_KLEN":  (_packed([k[4] for k in KINDS], KW), nk * KW),
        "K_NSBLK": (_packed([k[5] for k in KINDS], SBW), nk * SBW),
        "K_WTYPE": (_packed([k[6] for k in KINDS], 3), nk * 3),
        "X_KIND":  (_packed([kind_id(x[0]) for x in EXCEPTIONS], kidw), len(EXCEPTIONS) * kidw),
        "X_LAYER": (_packed([x[1] for x in EXCEPTIONS], lidw), len(EXCEPTIONS) * lidw),
        "X_WTYPE": (_packed([x[2] for x in EXCEPTIONS], 3), len(EXCEPTIONS) * 3),
        "X_VALID": (_packed([1] * len(EXCEPTIONS), 1), len(EXCEPTIONS)),
    }
    return out


def gen(report=True):
    nk = len(KINDS)
    lines = [f"{nk} {NLAYER} {NEXP} {len(EXCEPTIONS)}"]
    for name, (val, width) in params().items():
        lines.append(f"{name} {width} {val:0{(width + 3) // 4}x}")
    n = 0
    for kind in range(nk):
        for layer in range(NLAYER):
            for expert in range(NEXP):
                b, kl, ns, wt = descriptor(kind, layer, expert)
                lines.append(f"{kind} {layer} {expert} {b:08x} {kl:04x} {ns:02x} {wt}")
                n += 1
    if report:
        exc_kinds = {kind_id(x[0]) for x in EXCEPTIONS}
        print(f"wdesc: {n} (kind,layer,expert) triples over {nk} kinds x {NLAYER} "
              f"layers x {NEXP} experts; {len(EXCEPTIONS)} UD-bump exceptions on "
              f"{len(exc_kinds)} kind(s)", file=sys.stderr)
    return "\n".join(lines) + "\n"


def _selftest():
    n = 0
    fails = []

    def chk(c, m):
        nonlocal n
        n += 1
        if not c:
            fails.append(m)

    # --- the corpus must EXERCISE each mechanism, or a leg proves nothing ---
    exc_kinds = {kind_id(x[0]) for x in EXCEPTIONS}
    chk(exc_kinds, "no exceptions at all -- the override mechanism is untested")
    for k in exc_kinds:
        bumped = {l for xn, l, _ in EXCEPTIONS if kind_id(xn) == k}
        chk(bumped and len(bumped) < NLAYER,
            f"kind {KINDS[k][0]} is bumped on every layer or none -- the default "
            f"and the exception are never both observed")
    chk(any(k[3] for k in KINDS), "no kind has an expert stride -- dropping it is invisible")
    chk(any(k[2] for k in KINDS), "no kind has a layer stride")
    chk(NEXP > 1 and NLAYER > 1, "one expert or one layer makes both strides invisible")

    # --- the arithmetic ---
    ke = next(i for i, k in enumerate(KINDS) if k[3])
    b0, _, _, _ = descriptor(ke, 0, 0)
    b1, _, _, _ = descriptor(ke, 0, 1)
    bl, _, _, _ = descriptor(ke, 1, 0)
    chk(b1 - b0 == KINDS[ke][3], "expert stride is not applied once per expert")
    chk(bl - b0 == KINDS[ke][2], "layer stride is not applied once per layer")
    chk(descriptor(ke, 1, 1)[0] - b0 == KINDS[ke][2] + KINDS[ke][3],
        "the two strides do not compose")

    # --- the exception WINS, and only where it applies ---
    for xn, xl, xw in EXCEPTIONS:
        k = kind_id(xn)
        chk(descriptor(k, xl, 0)[3] == xw, f"exception on {xn}@{xl} did not win")
        other = next(l for l in range(NLAYER)
                     if l not in {x[1] for x in EXCEPTIONS if kind_id(x[0]) == k})
        chk(descriptor(k, other, 0)[3] == KINDS[k][6],
            f"{xn} kept the exception type on layer {other}, which is not bumped")
        chk(xw != KINDS[k][6],
            f"the bump on {xn} is the SAME as its default -- the exception is a no-op "
            f"and INJ_WDESC_NO_EXC could not fail")

    # --- packing round-trip: the RTL indexes low-index-first ---
    p = params()
    val, width = p["K_BASE"]
    aw = width // len(KINDS)
    for i, k in enumerate(KINDS):
        chk(((val >> (i * aw)) & ((1 << aw) - 1)) == k[1],
            f"K_BASE packing is not low-index-first at kind {i}")

    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"glm53f_wdesc_gen self-test: {len(fails)}/{n} FAILED")
        return 1
    print(f"ALL {n} TESTS PASSED (the corpus exercises both strides and both sides "
          f"of every exception; base composes layer and expert strides; the UD bump "
          f"wins on its own layer and only there, and differs from the default so "
          f"the no-exception injection can fail; packing is low-index-first)")
    return 0


def params_vh():
    """The table as a Verilog include. It is DATA -- emitting it keeps the TB (and,
    later, the model top) free of hand-copied literals that can drift from this
    file silently."""
    out = ["// GENERATED by tools/glm53f_wdesc_gen.py -- do not edit.",
           f"localparam integer G_NKIND  = {len(KINDS)};",
           f"localparam integer G_NLAYER = {NLAYER};",
           f"localparam integer G_NEXP   = {NEXP};",
           f"localparam integer G_NEXC   = {len(EXCEPTIONS)};"]
    for name, (val, width) in params().items():
        out.append(f"localparam [{width-1}:0] G_{name} = "
                   f"{width}'h{val:0{(width + 3) // 4}x};")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    if "--params" in sys.argv:
        outp = a[0] if a else "build/glm53f_wdesc_params.vh"
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w") as fh:
            fh.write(params_vh())
        print(f"wrote {outp}")
        sys.exit(0)
    outp = a[0] if a else "build/glm53f_wdesc_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as fh:
        fh.write(gen())
    print(f"wrote {outp}")
