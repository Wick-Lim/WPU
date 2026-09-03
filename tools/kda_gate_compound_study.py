#!/usr/bin/env python3
"""
kda_gate_compound_study.py -- does the bf16 gate error compound in the recurrence?

make kda-gate measured, against the fp32 reference, worst-case on its corpus:
    ge   (decay)      1.24 % relative      beta (write gate)  1.34 % relative
    saturation floor: where the fp32 sigmoid gives exactly 0 (ge = 1.0, NO decay),
                      glm_act lands at g = -5.63e-7 -> ge = 0.99999944.
A single-token gate cannot say whether that matters once the recurrence runs for
thousands of tokens.  This study answers it with the reference recurrence itself
(tools/glm53_flash_ref.py kda_step -- no RTL involved): run T tokens with exact
gates, run the SAME tokens with the gates perturbed by the measured error, and
watch the divergence of `out` and of the state.

Three perturbation models, because the answer depends on the error's structure:
  random   : eps ~ U(-e, +e) per element per token  (uncorrelated, partly averages out)
  biased   : eps = +e on every element every token  (systematic; the worst plausible)
  floor    : only the saturation floor -- ge that should be exactly 1.0 is 0.99999944
Reported: relative error of `out` (RMS over heads/dims) and the state-norm ratio,
at T = 1, 16, 256, 2048.  Pure numpy; parameters are the real DK = DV = 128.

This is evidence for a DECISION ("build an fp32 sigmoid or not"), not a proof:
random q/k/v and random gates are not the model's activations.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_flash_ref as ref  # noqa: E402

F32 = np.float32
E_GE, E_BETA, FLOOR_GE = 0.0124, 0.0134, F32(0.99999944)


def run(T, H, DK, DV, mode, seed=0, checkpoints=(1, 16, 256, 2048)):
    rng = np.random.default_rng(seed)
    s_ref = np.zeros((H, DK, DV), F32)
    s_pert = np.zeros((H, DK, DV), F32)
    rows = {}
    for t in range(1, T + 1):
        q = rng.normal(size=(H, DK)).astype(F32)
        k = rng.normal(size=(H, DK)).astype(F32)
        v = rng.normal(size=(H, DV)).astype(F32)
        # realistic gate regime: mostly mild decay, a slice of "no decay" (saturated)
        g = (-rng.uniform(0.0, 1.5, size=(H, DK))).astype(F32)
        sat = rng.random(size=(H, DK)) < 0.15          # 15% of dims: no decay wanted
        g[sat] = F32(-0.0)
        beta = ref.sigmoid(rng.normal(size=H).astype(F32))

        o_ref, s_ref = ref.kda_step(s_ref, q, k, v, g, beta)

        # perturb the gates the way the RTL does, then run the identical recurrence
        ge = np.exp(g.astype(np.float64)).astype(F32)
        if mode == "random":
            ge_p = (ge * (1 + rng.uniform(-E_GE, E_GE, ge.shape))).astype(F32)
            beta_p = (beta * (1 + rng.uniform(-E_BETA, E_BETA, beta.shape))).astype(F32)
        elif mode == "biased":
            ge_p = (ge * (1 + E_GE)).astype(F32)
            beta_p = (beta * (1 + E_BETA)).astype(F32)
        elif mode == "floor":
            ge_p = ge.copy(); beta_p = beta
        else:
            raise ValueError(mode)
        ge_p = np.where(sat, np.minimum(ge_p, FLOOR_GE), ge_p).astype(F32)  # the rail floor
        beta_p = np.clip(beta_p, 0, 1).astype(F32)
        # kda_step takes log-decay; feed log(ge_p) so the recurrence is identical
        g_p = np.log(ge_p.astype(np.float64)).astype(F32)
        o_pert, s_pert = ref.kda_step(s_pert, q, k, v, g_p, beta_p)

        if t in checkpoints:
            rel_out = float(np.sqrt(np.mean((o_pert - o_ref) ** 2)) /
                            (np.sqrt(np.mean(o_ref ** 2)) + 1e-30))
            snorm = float(np.linalg.norm(s_pert) / (np.linalg.norm(s_ref) + 1e-30))
            rows[t] = (rel_out, snorm)
    return rows


def main():
    H, DK, DV, T = 8, 128, 128, 2048
    print(f"KDA recurrence, H={H} DK={DK} DV={DV}, T={T} tokens; perturbation = measured "
          f"make kda-gate error (ge {E_GE*100:.2f}%, beta {E_BETA*100:.2f}%, floor {FLOOR_GE})\n")
    print(f"{'mode':<8} {'T':>5} {'rel err of out (RMS)':>22} {'state-norm ratio':>18}")
    print("-" * 58)
    for mode in ("floor", "random", "biased"):
        rows = run(T, H, DK, DV, mode)
        for t in sorted(rows):
            r, sn = rows[t]
            print(f"{mode:<8} {t:>5} {r:>21.3e} {sn:>18.6f}")
        print()
    print("READING IT: the recurrence is contractive (decay < 1), so a per-token gate error")
    print("does NOT accumulate without bound -- it reaches a steady state set by the decay")
    print("horizon.  What matters is where that steady state lands relative to the 1.2-1.3%")
    print("per-token input error: near it (error stays a few percent) or far above it.")


if __name__ == "__main__":
    main()
