#!/usr/bin/env python3
"""swiglu_q4k_gen.py -- test vectors + golden for swiglu_expert_q4k.
y = ( silu(x @ W_gate) (.) (x @ W_up) ) @ W_down, weights in GGML Q4_K (UD-Q4_K_XL).

The GEMMs use the exact Q4_K dequant + fp32-MAC contract (tools/q4k_ref, proven
bit-exact); the silu/merge tail is the shared glm_act SILU + bf16_mul (proven in the
FP8 path). This golden pairs exact GEMMs with a true-silu reference and a per-output
tolerance covering glm_act's poly-silu approximation + bf16 grid -- the proven
methodology of swiglu_expert_fp8_tb. Emits build/swiglu_q4k_vec.txt for the TB.

File layout (HIDDEN, INTER, TN, PE_M fixed by the TB; NSB(K)=ceil(K/256)
super-blocks per weight column -- NSB=1 at the committed slice, byte-identical):
  line1: NTEST HIDDEN INTER TN
  per test:
    x[PE_M*HIDDEN]  (4hex bf16, row-major)
    per weight matrix in {GATE(N=INTER,K=HIDDEN), UP(N=INTER,K=HIDDEN), DOWN(N=HIDDEN,K=INTER)}:
      for n in 0..N-1:  {d(4hex) dmin(4hex) scales(24hex)} x NSB(K)  q[0..K-1](1hex each)
    y[PE_M*HIDDEN] (4hex bf16 golden)   tol[PE_M*HIDDEN] (float)
"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from q4k_ref import (get_scale_min_k4, _pack_6bit_scales, _f32_to_f16bits,
                     bf16_round, matmul_q4k_col)

def bf16bits(x):
    return (int(np.frombuffer(np.float32(bf16_round(x)).tobytes(), dtype=np.uint32)[0]) >> 16) & 0xFFFF
def bf16_val(u16):
    return np.float32(np.frombuffer(np.uint32(u16 << 16).tobytes(), dtype=np.float32)[0])
def bf16_mul(a, b):  # matches glm_fp.vh bf16_mul: fp32 product -> bf16
    return bf16_round(np.float32(a) * np.float32(b))
def silu(x):  # glm_act SILU semantics: sigmoid(clamp(x,+-16)) then x*sigmoid, bf16 out
    z = np.float32(min(16.0, max(-16.0, float(x))))
    s = np.float32(1.0 / (1.0 + np.exp(-z)))
    return bf16_round(np.float32(x) * s)

def mk_weight(rng, N, K):
    """random Q4_K weight matrix as N columns, each spanning NSB=ceil(K/256)
    super-blocks along K (NSB=1 at the slice K<=256 -- byte-identical output)."""
    NSB = (K + 255) // 256
    cols = []
    for _ in range(N):
        d, dm, s, s96 = [], [], [], []
        for _sb in range(NSB):
            d.append(_f32_to_f16bits(rng.uniform(0.01, 0.06)))
            dm.append(_f32_to_f16bits(rng.uniform(0.0, 0.03)))
            sc = _pack_6bit_scales([int(v) for v in rng.integers(0,64,8)],
                                   [int(v) for v in rng.integers(0,64,8)])
            s.append(sc); s96.append(sum(b << (8*i) for i, b in enumerate(sc)))
        q = [int(v) for v in rng.integers(0, 16, K)]
        cols.append((d, dm, s, s96, q))
    return cols

def dequant_col(col):
    d_h, dm_h, s, _, q = col
    out = np.empty(len(q), dtype=np.float32)
    for k, qv in enumerate(q):
        sb = k // 256
        d  = np.float32(np.frombuffer(np.uint16(d_h[sb]).tobytes(),  dtype=np.float16)[0])
        mn = np.float32(np.frombuffer(np.uint16(dm_h[sb]).tobytes(), dtype=np.float16)[0])
        sc, m = get_scale_min_k4((k % 256) // 32, s[sb])
        out[k] = (d * np.float32(sc)) * np.float32(qv) - mn * np.float32(m)
    return out

def emit_w(lines, cols):
    for (d, dm, _s, s96, q) in cols:
        hdr = " ".join(f"{d[sb]:04x} {dm[sb]:04x} {s96[sb]:024x}" for sb in range(len(d)))
        lines.append(hdr + " " + " ".join(f"{qq:01x}" for qq in q))

def gen(ntest, HIDDEN, INTER, TN, PE_M=1, seed=0, clamp_lim=None, xscale=1.0):
    """clamp_lim: None = the committed GLM-5.2 SwiGLU (no clamp). A float turns on
    GLM-5.3-Flash's clamp, which is ASYMMETRIC -- gate upper-bounded only, up
    bounded both ways (tools/glm53_flash_ref.py: clamped_swiglu).

    xscale exists because a clamp gate is VACUOUS unless the operands actually
    cross the limit: at the default input scale the GEMM outputs stay well inside
    +/-10, every clamp is a no-op, and the test would pass against an
    unclamped DUT. The emitter asserts the crossing really happened."""
    rng = np.random.default_rng(seed)
    tot_hi = tot_lo = 0
    lines = [f"{ntest} {HIDDEN} {INTER} {TN}"]
    for t in range(ntest):
        x = [[bf16bits(v) for v in rng.uniform(-1.2 * xscale, 1.2 * xscale, HIDDEN)]
             for _ in range(PE_M)]
        Wg = mk_weight(rng, INTER,  HIDDEN)
        Wu = mk_weight(rng, INTER,  HIDDEN)
        Wd = mk_weight(rng, HIDDEN, INTER)
        wg = [dequant_col(c) for c in Wg]
        wu = [dequant_col(c) for c in Wu]
        wd = [dequant_col(c) for c in Wd]
        Y, TOL = [], []
        n_hi = n_lo = 0                      # clamp-activity counters (see xscale)
        for r in range(PE_M):
            xf = np.array([bf16_val(x[r][k]) for k in range(HIDDEN)], dtype=np.float32)
            h  = np.empty(INTER, dtype=np.float32)
            for n in range(INTER):
                g = matmul_q4k_col(xf, wg[n])   # bf16-valued float32
                u = matmul_q4k_col(xf, wu[n])
                if clamp_lim is not None:
                    L = np.float32(clamp_lim)
                    if g > L:            g = L; n_hi += 1
                    if u >  L:           u = L; n_hi += 1
                    elif u < -L:         u = -L; n_lo += 1
                h[n] = bf16_mul(silu(g), u)      # bf16-valued float32
            for o in range(HIDDEN):
                yv = matmul_q4k_col(h, wd[o])
                Y.append(bf16bits(yv))
                # tolerance: covers glm_act poly-silu approx + bf16 grid, scaled by
                # the down reduction magnitude. Functional (plumbing) check; the
                # bit-exact gate is glm_matmul_q4k.
                mag = float(np.sum(np.abs(h) * np.abs(wd[o])))
                TOL.append(max(0.06 * abs(float(yv)), 0.02 * mag, 0.03))
        lines.append(" ".join(f"{x[r][k]:04x}" for r in range(PE_M) for k in range(HIDDEN)))
        emit_w(lines, Wg); emit_w(lines, Wu); emit_w(lines, Wd)
        lines.append(" ".join(f"{y:04x}" for y in Y))
        lines.append(" ".join(f"{tt:.6f}" for tt in TOL))
        if clamp_lim is not None:
            tot_hi += n_hi; tot_lo += n_lo
    if clamp_lim is not None:
        # A clamp gate that never clamps proves nothing. Both directions must fire:
        # the upper bound (gate AND up) and the lower bound (up only).
        assert tot_hi > 0 and tot_lo > 0, (
            f"clamp vectors are VACUOUS: upper fired {tot_hi}x, lower {tot_lo}x -- "
            f"raise xscale until both are nonzero, or the gate passes against an "
            f"unclamped DUT")
        print(f"clamp activity: upper {tot_hi}, lower {tot_lo} (both nonzero -> gate is live)",
              file=sys.stderr)
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    # --clamp[=LIMIT] turns on the GLM-5.3-Flash asymmetric SwiGLU clamp golden.
    # Flags are stripped before the positional parse so `--clamp` may appear
    # anywhere (int("--clamp") would otherwise abort the run).
    clamp = None
    for a in sys.argv[1:]:
        if a.startswith("--clamp"):
            clamp = float(a.split("=", 1)[1]) if "=" in a else 10.0
    argv   = [a for a in sys.argv[1:] if not a.startswith("--")]
    ntest  = int(argv[0]) if len(argv) > 0 else 30
    HIDDEN = int(argv[1]) if len(argv) > 1 else 8
    INTER  = int(argv[2]) if len(argv) > 2 else 8
    TN     = int(argv[3]) if len(argv) > 3 else 4
    out    = argv[4] if len(argv) > 4 else "build/swiglu_q4k_vec.txt"
    open(out, "w").write(gen(ntest, HIDDEN, INTER, TN, clamp_lim=clamp))
    print(f"wrote {out}: {ntest} tests HIDDEN={HIDDEN} INTER={INTER} TN={TN} "
          f"(NSB gate/up={(HIDDEN+255)//256} down={(INTER+255)//256})")
