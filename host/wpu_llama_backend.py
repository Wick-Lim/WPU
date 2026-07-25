"""wpu_llama_backend.py -- the v0.1 SOFTWARE full-model backend.

Honest scope (docs/PRODUCT_SPEC.md, Stage 1 / v0.1): the WPU silicon does not
exist yet and the FPGA is a VOCAB=256 slice, so it cannot emit real GLM-5.2 text.
To prove the END-TO-END PRODUCT PATH now -- a standard client -> our
OpenAI-compatible server -> a backend that emits REAL tokens -> back to the
client, fully offline -- this backend shells out to a local llama.cpp build
running a real GGUF. It is clearly a SOFTWARE backend, not the accelerator: the
same path, with the GGUF swapped for GLM-5.2 on the box, is the product.

This is a TEXT backend: llama.cpp owns its own tokenizer, so it produces text
directly (the server uses `stream_text`, bypassing the id-level WPU tokenizer).

It never fabricates tokens: if the llama.cpp binary or the model is missing it
raises at construction, so nobody mistakes a canned string for real inference.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

from wpu_device import DeviceState


def _find_llama_cli(explicit: str | None) -> str:
    """Locate a runnable llama.cpp CLI; raise clearly if absent (no fake fallback)."""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("LLAMA_CLI"):
        cands.append(os.environ["LLAMA_CLI"])
    # the checkout used for the GGUF cross-check, if present
    home = os.path.expanduser("~")
    cands += [
        os.path.join(home, ".claude/jobs/01dbb3de/tmp/llamacpp/build/bin/llama-cli"),
        "llama-cli",
        "llama",
    ]
    for c in cands:
        p = shutil.which(c) if os.path.basename(c) == c else (c if os.path.exists(c) else None)
        if p:
            return p
    raise FileNotFoundError(
        "llama.cpp CLI not found. Build it (cmake --build build --target llama-cli) "
        "and pass --llama-cli <path> or set $LLAMA_CLI. This backend runs a REAL "
        "GGUF via llama.cpp; it will not fabricate tokens.")


class LlamaCppBackend:
    """Text backend: streams real completion text from llama.cpp over a subprocess.

    Interface mirrors WPUDevice where the server needs it (`state`, `model_id`,
    `poll_ready`) and adds `stream_text` / `complete_text` used by the server's
    text path.
    """

    def __init__(self, model_path: str, llama_cli: str | None = None,
                 n_ctx: int = 4096, threads: int | None = None,
                 boot_seconds: float = 0.0):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model GGUF not found: {model_path!r}. Point --model at a real GGUF.")
        self.cli = _find_llama_cli(llama_cli)
        self.model_path = model_path
        self.model_id = os.path.splitext(os.path.basename(model_path))[0]
        self.n_ctx = n_ctx
        self.threads = threads or max(2, (os.cpu_count() or 4) - 2)
        self._boot_seconds = boot_seconds        # software backend loads fast; keep honest
        self._boot_at = time.time()
        self.state = DeviceState.BOOTING
        # simple telemetry the management console reads
        self.telemetry = {
            "tokens_total": 0, "requests": 0, "last_tok_s": 0.0,
            "backend": "llama.cpp (software)", "note": "software backend, not the accelerator",
        }

    # ---- device-ish surface --------------------------------------------------
    def poll_ready(self) -> bool:
        if self.state == DeviceState.BOOTING and (time.time() - self._boot_at) >= self._boot_seconds:
            self.state = DeviceState.READY
        return self.state in (DeviceState.READY, DeviceState.BUSY)

    # ---- text generation -----------------------------------------------------
    # This llama.cpp build prints its banner, the echoed "> prompt" turn, and a
    # "[ Prompt: ... ]" timing footer to STDOUT (single-turn, non-conversation is
    # the only reliably-terminating one-shot mode). Live char parsing is fragile,
    # so we run buffered, extract the completion, and let the SERVER stream it to
    # the client in chunks (imperceptible for a small model; honest for the demo).
    def _argv(self, prompt: str, sampling) -> list:
        temp = float(getattr(sampling, "temperature", 0.0) or 0.0)
        argv = [
            self.cli, "-m", self.model_path,
            "-p", prompt,
            "-n", str(int(getattr(sampling, "max_tokens", 256) or 256)),
            "-c", str(self.n_ctx),
            "-t", str(self.threads),
            "--no-conversation", "--single-turn",   # one-shot completion that exits
            "--no-display-prompt",
        ]
        if temp <= 0.0:
            argv += ["--temp", "0", "--top-k", "1"]     # greedy
        else:
            argv += ["--temp", str(temp)]
            if getattr(sampling, "top_p", None) is not None:
                argv += ["--top-p", str(float(sampling.top_p))]
            if getattr(sampling, "top_k", None):
                argv += ["--top-k", str(int(sampling.top_k))]
        seed = getattr(sampling, "seed", None)
        if seed is not None:
            argv += ["--seed", str(int(seed))]
        return argv

    @staticmethod
    def _extract(raw: str) -> str:
        """Strip the llama-cli banner + echoed prompt + timing/exit footer, leaving
           just the model's completion text."""
        for m in ("\n[ Prompt:", "[ Prompt:", "\nExiting"):
            i = raw.find(m)
            if i != -1:
                raw = raw[:i]
        j = raw.rfind("\n> ")                    # end of the echoed "> prompt" turn
        if j != -1:
            k = raw.find("\n", j + 1)
            raw = raw[k + 1:] if k != -1 else ""
        return raw.strip()

    def stream_text(self, prompt: str, sampling, chunk: int = 24):
        """Run llama.cpp, extract the real completion, and yield it in chunks."""
        self.poll_ready()
        self.state = DeviceState.BUSY
        self.telemetry["requests"] += 1
        t0 = time.time()
        try:
            r = subprocess.run(
                self._argv(prompt, sampling), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=max(60, int(getattr(sampling, "max_tokens", 256) or 256)))
            text = self._extract(r.stdout)
        except subprocess.TimeoutExpired:
            text = "[llama.cpp backend timed out]"
        finally:
            self.state = DeviceState.READY
        dt = max(1e-6, time.time() - t0)
        approx_tok = max(0, round(len(text) / 4))     # ~4 chars/token, dashboard ballpark
        self.telemetry["tokens_total"] += approx_tok
        self.telemetry["last_tok_s"] = round(approx_tok / dt, 1)
        for i in range(0, len(text), chunk):
            yield text[i:i + chunk]

    def complete_text(self, prompt: str, sampling) -> str:
        return "".join(self.stream_text(prompt, sampling))
