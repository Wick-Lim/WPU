#!/usr/bin/env python3
"""
fp32_ieee_gen.py -- corpus for the fp32 primitive IEEE-conformance gate.

WHY THIS GATE EXISTS.  src/glm_fp.vh fp32_add is NOT exactly IEEE
round-to-nearest-even: measured, it differs from the correctly-rounded sum in
~0.04% of random cases, concentrated at moderate exponent gaps (4-5), always by
1 ULP.  That had been invisible because every proven path in this repo ends in
bf16, and a 1-ULP fp32 difference survives bf16_round in only ~0.001% of cases
-- so the Q4_K core's bit-exactness claim is intact and unaffected.

It became visible with src/kda_recur.v, the first consumer whose OUTPUT is fp32
(the KDA recurrent state). Rather than leave the gap latent, this gate MEASURES
it over a swept exponent-gap corpus and pins an upper bound, so a future change
to the adder shows up as a number moving rather than as a mystery ULP in some
downstream golden.

The bound is a CEILING on non-conformance, not a target: driving it to 0 (an
exactly-rounded adder) would be an improvement, and this gate would still pass.
"""
import sys

import numpy as np


def b(x):
    return int(np.frombuffer(np.float32(x).tobytes(), np.uint32)[0])


def gen(path, per_gap=400, max_gap=25, seed=5):
    rng = np.random.default_rng(seed)
    n = 0
    with open(path, "w") as f:
        for gap in range(max_gap):
            for _ in range(per_gap):
                a = np.float32(rng.uniform(1, 2) * rng.choice([-1, 1]))
                c = np.float32(rng.uniform(1, 2) * rng.choice([-1, 1]) * 2.0 ** (-gap))
                # numpy float32 arithmetic IS correctly rounded, so it is the oracle
                f.write(f"{b(a):08x} {b(c):08x} {b(np.float32(a + c)):08x} "
                        f"{b(np.float32(a * c)):08x}\n")
                n += 1
    return n


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "build/fp32_ieee_vec.txt"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"wrote {out}: {gen(out)} pairs (exponent gaps 0..24)")
