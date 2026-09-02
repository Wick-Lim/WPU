#!/usr/bin/env python3
"""
kda_gen.py -- golden vectors for the KDA recurrence core (src/kda_recur.v).

The golden is tools/glm53_flash_ref.py: kda_step, which is a transcription of
GLM-5.3-Flash's `recurrent_kimi_delta_attention` (seq_len == 1 decode path).

ACCURACY CONTRACT -- read this before calling any KDA result "bit-exact".
The recurrence needs two transcendentals this repo implements as APPROXIMATIONS:
  * l2norm needs 1/sqrt(.)  -> src/glm_fp.vh fp32_rsqrt is the Quake fast inverse
    square root with 2 Newton iterations;
  * the decay needs exp(.)  -> src/glm_fp_pipe.v fp32_exp_pipe is a degree-4
    Horner polynomial with range reduction.
Neither is bitwise equal to numpy's exp / sqrt, so ANY value downstream of them
can only be checked to a TOLERANCE. That is the same status the repo already
gives swiglu_expert_q4k ("functional plumbing check; the bit-exact gate is
glm_matmul_q4k"), and it is stated here rather than discovered later.

To keep the tolerance from hiding real bugs, this emitter also produces an
EXACT-PATH variant: q, k pre-normalised and g pre-exponentiated on the host, so
the RTL consumes them directly and the whole remaining recurrence (decay-multiply,
kv reduction, delta, outer-product update, output reduction) is pure fp32
mul/add -- which the repo's fp32_mul/fp32_add ARE bit-exact for. That leg is
bit-exact; the transcendental leg is tolerance. Two legs, two honest claims.

Vector format (one file, both legs share it):
  NTEST H DK DV
  per test:
    beta      : H            8hex fp32
    g         : H*DK         8hex fp32   (log-decay, pre-exp)
    gexp      : H*DK         8hex fp32   (exp(g), host-computed -- exact leg)
    q,k       : H*DK each    8hex fp32   (raw)
    qn,kn     : H*DK each    8hex fp32   (l2-normed ONLY -- exact leg; the RTL
                                          applies the constant 1/sqrt(DK) to qn
                                          itself, exactly as kda_step does after
                                          its l2norm. Pre-multiplying it here and
                                          dividing it back out does NOT round-trip
                                          in fp32 -- the self-test caught that.)
    v         : H*DV         8hex fp32
    state_in  : H*DK*DV      8hex fp32
    out       : H*DV         8hex fp32   (golden)
    state_out : H*DK*DV      8hex fp32   (golden)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32


def f32b(x):
    return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])


def row(a):
    return " ".join(f"{f32b(v):08x}" for v in np.asarray(a, F32).reshape(-1))


def gen(ntest, H, DK, DV, seed=0):
    rng = np.random.default_rng(seed)
    out = [f"{ntest} {H} {DK} {DV}"]
    for _ in range(ntest):
        # Scales chosen so the state neither explodes nor decays to nothing over a
        # single step: g in [-2, 0) keeps exp(g) in (0.13, 1), beta in (0,1).
        beta = ref.sigmoid(rng.normal(size=H).astype(F32))
        g = (-rng.uniform(0.05, 2.0, size=(H, DK))).astype(F32)
        q = rng.normal(size=(H, DK)).astype(F32)
        k = rng.normal(size=(H, DK)).astype(F32)
        v = rng.normal(size=(H, DV)).astype(F32)
        st = (rng.normal(size=(H, DK, DV)) * 0.1).astype(F32)

        o, sn = ref.kda_step(st, q, k, v, g, beta, use_qk_l2norm=True)

        # pre-computed exact-leg operands. l2norm ONLY: the 1/sqrt(DK) scale is
        # the RTL's to apply, because kda_step applies it AFTER the l2norm and a
        # scale/unscale round-trip is not bit-preserving in fp32.
        qn = ref.l2norm(q).astype(F32)
        kn = ref.l2norm(k).astype(F32)
        gexp = np.exp(g.astype(np.float64)).astype(F32)

        out += [row(beta), row(g), row(gexp), row(q), row(k),
                row(qn), row(kn), row(v), row(st), row(o), row(sn)]
    return "\n".join(out) + "\n"


def _selftest():
    """Re-derive the golden a second way: run the recurrence from the PRE-NORMED
    operands with l2norm/exp disabled. The two must agree bitwise -- if they do
    not, the exact leg's operands do not describe the same computation the
    tolerance leg checks, and the two legs would be testing different things."""
    rng = np.random.default_rng(7)
    H, DK, DV = 3, 8, 8
    bad = 0
    for _ in range(40):
        beta = ref.sigmoid(rng.normal(size=H).astype(F32))
        g = (-rng.uniform(0.05, 2.0, size=(H, DK))).astype(F32)
        q = rng.normal(size=(H, DK)).astype(F32)
        k = rng.normal(size=(H, DK)).astype(F32)
        v = rng.normal(size=(H, DV)).astype(F32)
        st = (rng.normal(size=(H, DK, DV)) * 0.1).astype(F32)
        o1, s1 = ref.kda_step(st, q, k, v, g, beta, use_qk_l2norm=True)
        qn = ref.l2norm(q).astype(F32)
        kn = ref.l2norm(k).astype(F32)
        # feed the pre-normed operands with the in-kernel norm OFF; kda_step then
        # applies the same 1/sqrt(DK) scale it would have, so the two paths are
        # the identical computation.
        o2, s2 = ref.kda_step(st, qn, kn, v, g, beta, use_qk_l2norm=False)
        if (np.frombuffer(o1.tobytes(), np.uint32) != np.frombuffer(o2.tobytes(), np.uint32)).any():
            bad += 1
        if (np.frombuffer(s1.tobytes(), np.uint32) != np.frombuffer(s2.tobytes(), np.uint32)).any():
            bad += 1
    if bad:
        print(f"kda_gen self-test: {bad} mismatches -- the exact leg's operands do "
              f"not reproduce the tolerance leg's computation")
        return 1
    print("ALL 80 TESTS PASSED (pre-normed operands reproduce the in-kernel-norm "
          "recurrence bitwise, so both gate legs check the same math)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    ntest = int(a[0]) if len(a) > 0 else 24
    H = int(a[1]) if len(a) > 1 else 3
    DK = int(a[2]) if len(a) > 2 else 8
    DV = int(a[3]) if len(a) > 3 else 8
    outp = a[4] if len(a) > 4 else "build/kda_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w").write(gen(ntest, H, DK, DV))
    print(f"wrote {outp}: {ntest} tests H={H} DK={DK} DV={DV}")
