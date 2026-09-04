#!/usr/bin/env python3
"""
fp32_exp_acc_gen.py -- corpus for the fp32_exp_pipe accuracy gate.

WHY.  src/glm_fp_pipe.v's fp32_exp_pipe is a degree-4 Horner with range
reduction.  Nothing in this repo had ever measured how far that is from a
correctly-rounded exp -- three spot checks suggested ~10 ULP, and building
src/fp32_sigmoid_pipe.v on top of it made the real number matter.  Swept over
x in [-40, 40] (the range 1/(1+exp(-x)) actually needs) it is **1899 ULP =
2.3e-4 relative**, two orders worse than the spot checks implied.

That is the precision CEILING for everything built on it: the fp32 sigmoid, and
the mHC map, whose softmax is also an exp.  It is still ~50x better than the
bf16 glm_act path it replaces -- but it is not "fp32-exact", and the ledger now
says so instead of implying it.

This gate pins the number so a future change to the polynomial moves a measured
value rather than silently shifting every downstream accuracy claim.

Format:  NTEST \n  per test: x(8hex) golden_exp(8hex)
"""
import os
import sys

import numpy as np

F32 = np.float32


def b(x):
    return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])


def gen(seed=9, n=3000):
    rng = np.random.default_rng(seed)
    xs = list(rng.uniform(-40, 40, n))
    out = [str(len(xs))]
    for v in xs:
        v = F32(v)
        out.append(f"{b(v):08x} {b(F32(np.exp(np.float64(v)))):08x}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    outp = sys.argv[1] if len(sys.argv) > 1 else "build/fp32_exp_acc_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w").write(gen())
    print(f"wrote {outp}")
