"""
wpu_sim_backend.py -- an WPUDevice backed by the RTL SIMULATOR (iverilog/vvp).

Drives the on-main product top `glm_model_q4k` (the committed VOCAB=256 slice, via its
`make model-q4k` iverilog/`vvp` build) and returns the REAL argmax next-token ids the
RTL forward pass produces, wired into the device protocol. This is the
"server -> real RTL -> real token" co-simulation path.

HONEST SCOPE (all real, all caveated -- this is a co-sim / datapath witness, NOT a
usable chatbot):
  * SLOW -- each forward is the full assembled model in an event simulator; the
    committed VOCAB=256 slice run is minutes-long (the SPEC_SLICE VOCAB=16 build is
    seconds). Fine for bring-up, NOT for interactive use. The result is cached per
    process.
  * SLICE model -- MODEL_DIM=128/L=6/VOCAB=256, small/untrained: the tokens are
    genuine bit-exact-vs-numpy-golden RTL outputs but NOT language. This validates
    the datapath + protocol, not fluency.
  * FIXED vectors -- runs the testbench's built-in stimulus (build/mq4k/stim.hex),
    NOT the user's prompt. The streamed tokens are the TB's golden argmax cases, so
    they are INDEPENDENT of whatever text a client sends. Driving glm_model_q4k from a
    live prompt needs the full weight/embedding/KV pull-port ROM harness (a larger TB
    effort); the committed TB already carries that harness for its own vectors, which
    we reuse here.

Build the binary + golden vectors first (once):
    make model-q4k            # -> build/glm_model_q4k_full_sim + build/mq4k/*.hex

History: the prior fp8-era backend targeted `build/glm_model_fp8_sim`; `glm_model_fp8`
was removed from main (it lives on tag `fp8-verified-baseline`), so this backend was retargeted to the
on-main `glm_model_q4k` product top.
"""

from __future__ import annotations

import os
import re
import subprocess

from wpu_device import WPUDevice, DeviceState

# glm_model_q4k_full_tb prints one line per golden case:
#   "case 0: token=3 pos=1 s_len=2 -> argmax=13 (golden 13)    MATCH"
# -- the `argmax=<id>` is the DUT's real next-token for that fixed vector.
_ARGMAX_RE = re.compile(r"->\s*argmax=(\d+)")
_REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))


class SimulatorBackend(WPUDevice):
    vocab_size = 256
    eos_token = 256                                  # out-of-band sentinel

    def __init__(self, vvp_binary: str = "build/glm_model_q4k_full_sim",
                 vvp: str = "vvp", timeout: float = 1800.0,
                 cwd: str | None = None) -> None:
        super().__init__()
        self.cwd = cwd or _REPO
        self.vvp_binary = vvp_binary
        self.vvp = vvp
        self.timeout = timeout
        self._tokens: list[int] | None = None
        self._cursor = 0

    def _boot_seconds(self) -> float:
        return 0.0                                   # the binary is already built

    # The slice sim REPLAYS the argmax stream of ONE fixed vvp run (prefill() below
    # ignores prompt_ids), so there is no position-addressed KV to reuse.
    supports_prefix_cache = False

    def reset_session(self) -> None:
        super().reset_session()
        self._cursor = 0                             # keep the cached RTL tokens

    def _binary_path(self) -> str:
        p = self.vvp_binary
        return p if os.path.isabs(p) else os.path.join(self.cwd, p)

    def _run_rtl(self) -> list[int]:
        """Run the vvp slice sim ONCE and parse the RTL argmax tokens (minutes)."""
        binp = self._binary_path()
        if not os.path.exists(binp):
            raise FileNotFoundError(
                f"vvp binary {binp} not built. Build it + its golden vectors first: "
                f"`make model-q4k` (builds build/glm_model_q4k_full_sim and "
                f"build/mq4k/*.hex).")
        proc = subprocess.run([self.vvp, binp], cwd=self.cwd,
                              capture_output=True, text=True, timeout=self.timeout)
        toks = [int(m) for m in _ARGMAX_RE.findall(proc.stdout)]
        if not toks:
            raise RuntimeError(
                f"no '-> argmax=<id>' lines in vvp output (rc={proc.returncode}); the "
                f"golden vectors (build/mq4k/*.hex) may be missing -- run `make model-q4k`. "
                f"tail: {proc.stdout[-400:]!r}")
        return toks

    def _ensure(self) -> None:
        if self._tokens is None:
            self.state = DeviceState.BUSY
            self._tokens = self._run_rtl()
            self.state = DeviceState.READY

    def prefill(self, prompt_ids, start_pos: int) -> int:
        self._ensure()
        self._cursor = 0
        out = self._tokens[0] if self._tokens else self.eos_token
        self._cursor = 1
        return out

    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        self._ensure()
        out = (self._tokens[self._cursor]
               if self._cursor < len(self._tokens) else self.eos_token)
        self._cursor += 1
        return out


if __name__ == "__main__":
    # Manual smoke (SLOW, minutes): needs `make model-q4k` first.
    #   python3 host/wpu_sim_backend.py
    dev = SimulatorBackend()
    dev.power_on()
    print("running the RTL glm_model_q4k slice sim (minutes; fixed TB vectors)...")
    toks = list(dev.generate(prompt_ids=[1, 2, 3], max_new_tokens=16))
    print(f"RTL argmax tokens: {toks}")
