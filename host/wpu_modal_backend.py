"""wpu_modal_backend.py -- the v0.1 SOFTWARE full-model backend, via Modal GPU cloud.

Instead of a small local model (llama.cpp), this backend proxies chat completions
to the REAL GLM family served by vLLM on Modal (tools/modal_glm_server.py, 2x H100).
So a standard OpenAI client -> WPU host server -> this proxy -> Modal vLLM -> REAL
GLM-4.5-Air text, fully OpenAI-compatible and streaming.

Honest scope: this is SOFTWARE on cloud GPUs, NOT the WPU accelerator; it proves
the product EXPERIENCE with the real model family now. The accelerator replaces
this backend behind the same host API once silicon exists (docs/PRODUCT_SPEC.md).

It never fabricates: if the endpoint is unreachable it raises / surfaces the error,
so nobody mistakes a canned string for real inference. Any OpenAI-compatible
`/v1` base URL works here (a local vLLM, a Modal deploy, etc.) -- it is a thin proxy.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error

from wpu_device import DeviceState


class ModalBackend:
    """Text backend: proxies to an OpenAI-compatible /v1 endpoint (Modal vLLM by
       default). Mirrors the device surface the server needs (`state`, `model_id`,
       `poll_ready`) and provides `stream_text` used by the server's text path."""

    def __init__(self, base_url: str, api_key: str = "wpu-local",
                 model: str = "wpu-glm", boot_seconds: float = 0.0,
                 timeout: float = 300.0):
        if not base_url:
            raise ValueError("--modal-url is required (the Modal vLLM /v1 base URL). "
                             "Deploy tools/modal_glm_server.py first.")
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key
        self.model_id = model
        self.timeout = timeout
        self._boot_seconds = boot_seconds
        self._boot_at = time.time()
        self.state = DeviceState.BOOTING
        self.telemetry = {
            "tokens_total": 0, "requests": 0, "last_tok_s": 0.0,
            "backend": f"Modal vLLM ({model})", "endpoint": self.base_url,
            "note": "software on cloud GPUs (real GLM family), not the accelerator",
        }

    # ---- device-ish surface --------------------------------------------------
    def poll_ready(self) -> bool:
        if self.state == DeviceState.BOOTING and (time.time() - self._boot_at) >= self._boot_seconds:
            self.state = DeviceState.READY
        return self.state in (DeviceState.READY, DeviceState.BUSY)

    def _post(self, path: str, payload: dict):
        req = urllib.request.Request(
            self.base_url + path, method="POST",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.api_key}"})
        return urllib.request.urlopen(req, timeout=self.timeout)

    # ---- text generation (proxy, streaming) ----------------------------------
    def stream_text(self, prompt: str, sampling):
        """Yield real GLM completion-text deltas by streaming from the Modal vLLM
           OpenAI endpoint. `prompt` is the host-formatted text; we send it as a
           single user message (vLLM applies the model's own chat template)."""
        self.poll_ready()
        self.state = DeviceState.BUSY
        self.telemetry["requests"] += 1
        t0, n_chars = time.time(), 0
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(getattr(sampling, "max_tokens", 256) or 256),
            "temperature": float(getattr(sampling, "temperature", 0.0) or 0.0),
            "stream": True,
        }
        tp = getattr(sampling, "top_p", None)
        if tp is not None:
            payload["top_p"] = float(tp)
        seed = getattr(sampling, "seed", None)
        if seed is not None:
            payload["seed"] = int(seed)
        try:
            resp = self._post("/chat/completions", payload)
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    n_chars += len(delta)
                    yield delta
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Modal endpoint unreachable ({self.base_url}): {e}. Deploy "
                "tools/modal_glm_server.py and pass its URL via --modal-url. "
                "This backend proxies real GLM tokens; it will not fabricate.") from e
        finally:
            dt = max(1e-6, time.time() - t0)
            approx_tok = max(0, round(n_chars / 4))
            self.telemetry["tokens_total"] += approx_tok
            self.telemetry["last_tok_s"] = round(approx_tok / dt, 1)
            self.state = DeviceState.READY

    def complete_text(self, prompt: str, sampling) -> str:
        return "".join(self.stream_text(prompt, sampling))
