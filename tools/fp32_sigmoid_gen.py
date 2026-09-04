#!/usr/bin/env python3
"""
fp32_sigmoid_gen.py -- golden vectors for src/fp32_sigmoid_pipe.v.

Golden: numpy fp32 of 1/(1+exp(-x)) computed in float64 then rounded once, i.e.
the correctly-rounded fp32 sigmoid.  The DUT is exp (a degree-4 Horner, ~10 ULP)
followed by a Newton reciprocal (<=1 ULP), so this is a MEASURED-ULP gate, not a
bit-exact one -- and the point of the unit is the four orders of magnitude it
wins over the bf16 glm_act sigmoid it replaces, not bit-exactness.

The corpus deliberately covers the three regimes the port depends on:
  * the steep transition   |x| <= 8       -- where the KDA gate and mHC operate
  * the shoulders          8 < |x| <= 40  -- where bf16 has already lost the tail
  * SATURATION             |x| >= 90      -- where the reference gives EXACTLY
    1.0 and EXACTLY 0.0.  Those two are checked BITWISE: they are the properties
    that make this unit worth building (mHC needs sigma == 1.0 so that
    sigma + 1e-6 is representable; the KDA forget gate needs sigma == +0.0 so
    that -5.0 * sigma is -0.0).  A unit that is accurate in the middle but cannot
    saturate would pass a pure-tolerance gate and still be useless for both.

Format:  NTEST \n  per test: x(8hex) golden(8hex)
"""
import os
import sys

import numpy as np

F32 = np.float32


def b(x):
    return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])


def gen(seed=0):
    rng = np.random.default_rng(seed)
    xs = []
    xs += list(rng.uniform(-8, 8, 600))                     # the steep transition
    xs += list(rng.uniform(-40, 40, 300))                   # shoulders
    xs += list(rng.uniform(-16, 16, 100))                   # where glm_act rails
    xs += [0.0, 1.0, -1.0, 16.0, -16.0, 8.0, -8.0]
    xs += list(rng.uniform(90, 200, 40))                    # saturate to 1.0
    xs += list(-rng.uniform(90, 200, 40))                   # saturate to 0.0
    out = [str(len(xs))]
    n_one = n_zero = 0
    for v in xs:
        v = F32(v)
        s = F32(1.0 / (1.0 + np.exp(-np.float64(v))))
        if b(s) == 0x3F800000:
            n_one += 1
        if b(s) == 0x00000000:
            n_zero += 1
        out.append(f"{b(v):08x} {b(s):08x}")
    # A corpus that never saturates would leave the two bitwise legs vacuous.
    assert n_one > 0 and n_zero > 0, (
        f"corpus never saturates (sigma==1.0: {n_one}, sigma==0.0: {n_zero})")
    print(f"corpus: {len(xs)} points; golden sigma is exactly 1.0 on {n_one} and "
          f"exactly 0.0 on {n_zero} (both nonzero -> the bitwise saturation legs are live)",
          file=sys.stderr)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    outp = sys.argv[1] if len(sys.argv) > 1 else "build/fp32_sigmoid_vec.txt"
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w").write(gen())
    print(f"wrote {outp}")
