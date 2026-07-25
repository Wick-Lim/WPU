#!/usr/bin/env python3
"""
wpu_server.py -- a local OpenAI-compatible HTTP server backed by an WPU device.

Point any OpenAI-compatible client (chat UIs, VS Code extensions, `openai` SDK with
base_url=http://localhost:8000/v1) at this server and it drives the WPU device
through the exact RTL host protocol (wpu_device.py). This is the D2 software
scaffold from docs/USBC_PRODUCT_PLAN.md: the API surface, the device-protocol
plumbing, and token streaming are REAL and swappable to the hardware backend; the
default MockDevice returns a clearly-labelled canned response (proving the loop, not
the model). A byte-level tokenizer makes text round-trip exactly through token ids.

Run:   python3 host/wpu_server.py            # 0 external deps (stdlib only)
Test:  curl -s localhost:8000/v1/models
       curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
            -d '{"model":"wpu-glm-5.2-q4k","messages":[{"role":"user","content":"hi"}]}'
       # streaming:
       curl -sN localhost:8000/v1/chat/completions -H 'content-type: application/json' \
            -d '{"model":"wpu-glm-5.2-q4k","messages":[{"role":"user","content":"hi"}],"stream":true}'
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wpu_device import WPUDevice, MockDevice, SamplingParams   # noqa: E402
from wpu_chat_template import apply_chat_template      # noqa: E402
from wpu_tokenizer import make_tokenizer               # noqa: E402


# ---------------------------------------------------------------------------
# Stop-sequence scanning (host-side, on the decoded text)
# ---------------------------------------------------------------------------
def _stop_scan(text: str, stops: list) -> "tuple[int | None, int]":
    """Scan `text` for OpenAI `stop` sequences.

    Returns (cut, safe):
      * cut  -- index where the earliest stop string starts (emit text[:cut] and finish
                with reason "stop"), or None if no full stop appears yet.
      * safe -- number of leading chars safe to emit NOW while streaming: everything
                except a trailing suffix that could still grow into a stop string (so a
                stop straddling token boundaries is never partially emitted)."""
    if not stops:
        return None, len(text)
    cut = None
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = i if cut is None else min(cut, i)
    if cut is not None:
        return cut, cut
    hold = 0                                          # longest partial-stop suffix
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                hold = max(hold, k)
                break
    return None, len(text) - hold


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class WPUServer:
    def __init__(self, device, tokenizer, raw: bool = False,
                 backend_name: str = "", manifest_path: str | None = None):
        self.device = device
        self.tok = tokenizer
        #: raw=True forces the naive "role: content" flatten (byte scaffold / debug),
        #: bypassing the GLM chat template even when the GLM tokenizer is active.
        self.raw = raw
        self.backend_name = backend_name
        self.manifest_path = manifest_path
        #: control-plane settings surfaced/editable via the management console.
        self.settings = {
            "default_max_tokens": 256,
            "default_temperature": 0.0,
            "eco_mode": "device-only (no physical device in the software demo)",
            "caching": "prefix/KV reuse -- v0.2 (see docs/PRODUCT_SPEC.md)",
        }

    def provisioning_info(self) -> dict:
        """Read a provision_image.py manifest if one was supplied; else report the
           software-demo state honestly (host reads the GGUF directly, no NVMe image)."""
        if self.manifest_path and os.path.exists(self.manifest_path):
            try:
                m = json.load(open(self.manifest_path))
                return {"source": "provision manifest", "manifest": self.manifest_path,
                        "model": m.get("model"), "total_bytes": m.get("total_bytes"),
                        "image_sha256": m.get("image_sha256"),
                        "tensor_count": m.get("tensor_count"),
                        "resident_hot_bytes": m.get("resident_hot_bytes"),
                        "streamed_expert_bytes": m.get("streamed_expert_bytes")}
            except Exception as e:
                return {"source": "manifest error", "error": str(e)}
        return {"source": "software demo",
                "note": "no NVMe image in the software demo -- the host reads the GGUF "
                        "directly. On the box, tools/provision_image.py writes an image "
                        "+ manifest that this panel would show (model/size/sha/segments)."}

    # ---- prompt formatting ---------------------------------------------------
    def prompt_text(self, messages: list[dict]) -> str:
        """Build the prompt string. Applies the GLM-5.2 chat template when the GLM BPE
           tokenizer is active (real special tokens); falls back to the naive flatten
           for the byte scaffold or when raw=True (the mock just round-trips)."""
        if self.raw or getattr(self.tok, "name", None) != "glm":
            return self._flatten(messages)
        return apply_chat_template(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        """Naive "role: content" join -- the zero-dependency byte-scaffold path and the
           --raw fallback (a real GLM backend uses prompt_text()/the chat template)."""
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                         for m in messages)

    def _prime_mock(self, prompt: str) -> None:
        """For the MockDevice: encode a clearly-labelled canned reply with the ACTIVE
           tokenizer (byte or real GLM BPE) and hand the ids to the device to replay.
           A real backend produces its own ids -> this is a no-op."""
        if isinstance(self.device, MockDevice):
            reply = (f"[WPU mock device / {self.tok.name} tokenizer] protocol "
                     f"round-trip OK -- received {len(prompt)} chars over the host "
                     f"interface. Real tokens need the hardware / full-model backend "
                     f"(docs/USBC_PRODUCT_PLAN.md D1).")
            self.device.set_reply_ids(self.tok.encode(reply))

    # ---- core decode (host-side stop + finish_reason) ------------------------
    def _decode(self, messages, sampling: SamplingParams):
        """Core generator: yields decoded text deltas, then `return`s an info dict
           {"finish_reason", "prompt_tokens", "completion_tokens"} (read via
           StopIteration.value). Enforces `stop` sequences and `max_tokens` host-side;
           `temperature`/`top_p`/`top_k`/`seed` are programmed into the device (honored
           device-side; the mock is greedy)."""
        # Text backend (software full-model, e.g. llama.cpp): it owns its own
        # tokenizer and emits text directly -> use the text path, not the id path.
        if hasattr(self.device, "stream_text"):
            return (yield from self._decode_text(messages, sampling))
        prompt = self.prompt_text(messages)
        prompt_ids = self.tok.encode(prompt)
        self._prime_mock(prompt)
        st = self.tok.stream()
        stops = sampling.stop
        acc, emitted, produced = "", 0, 0

        def _finish(reason):
            return {"finish_reason": reason, "prompt_tokens": len(prompt_ids),
                    "completion_tokens": produced}

        for tok in self.device.generate(prompt_ids, sampling.max_tokens,
                                         sampling=sampling):
            produced += 1
            piece = st.push(tok)
            if piece:
                acc += piece
            cut, safe = _stop_scan(acc, stops)
            if cut is not None:                       # stop sequence hit -> truncate
                if cut > emitted:
                    yield acc[emitted:cut]
                return _finish("stop")
            if safe > emitted:
                yield acc[emitted:safe]
                emitted = safe

        tail = st.flush()                             # flush any buffered partial char
        if tail:
            acc += tail
        cut, _ = _stop_scan(acc, stops)
        if cut is not None:
            if cut > emitted:
                yield acc[emitted:cut]
            return _finish("stop")
        if len(acc) > emitted:                        # emit held-back suffix (gen over)
            yield acc[emitted:]
        # device stopped at max_tokens -> "length"; earlier (eos) -> "stop".
        return _finish("length" if produced >= sampling.max_tokens else "stop")

    def complete(self, messages, sampling: SamplingParams):
        """Non-streaming: return (full_text, prompt_tokens, finish_reason)."""
        gen = self._decode(messages, sampling)
        chunks, info = [], {}
        while True:
            try:
                chunks.append(next(gen))
            except StopIteration as e:
                info = e.value or {}
                break
        return ("".join(chunks), info.get("prompt_tokens", 0),
                info.get("finish_reason", "stop"))

    def stream(self, messages, sampling: SamplingParams):
        """Streaming: yield text deltas; `return`s the info dict (finish_reason etc.)."""
        return (yield from self._decode(messages, sampling))

    # ---- backward-compatible convenience wrappers ----------------------------
    def generate_text(self, messages, max_tokens: int = 256):
        """Legacy 2-tuple API (text, prompt_tokens) used by tests/simple callers."""
        text, n_prompt, _ = self.complete(messages, SamplingParams(max_tokens=max_tokens))
        return text, n_prompt

    def generate_stream(self, messages, max_tokens: int = 256):
        """Legacy streaming API: yield text deltas (finish_reason discarded)."""
        yield from self._decode(messages, SamplingParams(max_tokens=max_tokens))

    # ---- text-backend decode (software full-model; llama.cpp owns tokenization) ----
    def _decode_text(self, messages, sampling: SamplingParams):
        """Stream real text from a text backend (`device.stream_text`), enforcing
           `stop` sequences host-side. `return`s the same info dict as `_decode`."""
        prompt = self.prompt_text(messages)
        prompt_tok = len(self.tok.encode(prompt))
        stops = sampling.stop
        acc, emitted = "", 0

        def _finish(reason):
            return {"finish_reason": reason, "prompt_tokens": prompt_tok,
                    "completion_tokens": max(0, len(self.tok.encode(acc)))}

        for piece in self.device.stream_text(prompt, sampling):
            if not piece:
                continue
            acc += piece
            cut, safe = _stop_scan(acc, stops)
            if cut is not None:
                if cut > emitted:
                    yield acc[emitted:cut]
                return _finish("stop")
            if safe > emitted:
                yield acc[emitted:safe]
                emitted = safe
        if len(acc) > emitted:
            yield acc[emitted:]
        return _finish("stop")


def _now() -> int:
    return int(time.time())


def make_handler(server: WPUServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                 # quieter
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": server.device.model_id, "object": "model",
                     "created": _now(), "owned_by": "wpu"}]})
            elif path in ("/health", "/v1/health"):
                server.device.poll_ready()
                self._json(200, {"state": server.device.state,
                                 "model": server.device.model_id})
            # ---- management console (control plane; not a chat GUI) --------------
            elif path in ("", "/console"):
                self._serve_console()
            elif path == "/api/status":
                d = server.device
                d.poll_ready()
                tel = dict(getattr(d, "telemetry", {}) or {})
                # Prefix cache (D5): report the REUSED/FED split so the win is visible
                # rather than asserted. Text backends (Modal/llama.cpp) run their own
                # caching upstream and expose no such counters -- absent, not faked.
                ps = getattr(d, "prefix_stats", None)
                if ps and (ps["reused"] + ps["fed"]):
                    tel["prefix_cache"] = dict(
                        ps, hit_rate=round(ps["reused"] / (ps["reused"] + ps["fed"]), 3),
                        enabled=getattr(d, "supports_prefix_cache", False))
                self._json(200, {
                    "state": d.state, "model": d.model_id,
                    "backend": server.backend_name, "tokenizer": server.tok.name,
                    "telemetry": tel})
            elif path == "/api/provisioning":
                self._json(200, server.provisioning_info())
            elif path == "/api/settings":
                self._json(200, server.settings)
            else:
                self._json(404, {"error": {"message": f"no route {self.path}"}})

        def _serve_console(self):
            try:
                html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "console.html"), "rb").read()
            except OSError:
                self._json(500, {"error": {"message": "console.html missing"}})
                return
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, {"error": {"message": f"no route {self.path}"}})
                return
            length = int(self.headers.get("content-length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:
                self._json(400, {"error": {"message": f"bad json: {e}"}})
                return
            messages = req.get("messages", [])
            sampling = SamplingParams.from_request(req)   # temperature/top_p/top_k/
            stream = bool(req.get("stream", False))       # max_tokens/stop/seed/...
            cid = f"chatcmpl-wpu-{_now()}"
            model = server.device.model_id
            fp = f"wpu-seed-{sampling.seed}" if sampling.seed is not None else "wpu"

            if stream:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()

                def sse(obj):
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()

                sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                     "model": model, "system_fingerprint": fp, "choices": [{"index": 0,
                     "delta": {"role": "assistant"}, "finish_reason": None}]})
                try:
                    gen = server.stream(messages, sampling)
                    finish = "stop"
                    while True:
                        try:
                            piece = next(gen)
                        except StopIteration as e:
                            finish = (e.value or {}).get("finish_reason", "stop")
                            break
                        sse({"id": cid, "object": "chat.completion.chunk",
                             "created": _now(), "model": model, "choices": [{"index": 0,
                             "delta": {"content": piece}, "finish_reason": None}]})
                    sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                         "model": model, "choices": [{"index": 0, "delta": {},
                         "finish_reason": finish}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            text, n_prompt, finish = server.complete(messages, sampling)
            n_completion = len(server.tok.encode(text))
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": _now(),
                "model": model, "system_fingerprint": fp,
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                          "total_tokens": n_prompt + n_completion}})

    return Handler


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="WPU local OpenAI-compatible server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--boot-seconds", type=float, default=0.4,
                   help="modelled device boot (resident-set load) time")
    p.add_argument("--tokenizer", default=None,
                   help="path to GLM tokenizer.json (else byte-level fallback)")
    p.add_argument("--backend", choices=["mock", "sim", "modal", "llama"], default="mock",
                   help="mock (canned, default); modal (v0.1 SOFTWARE full-model demo: "
                        "REAL GLM family via vLLM on Modal GPU cloud -- needs --modal-url); "
                        "llama (local small GGUF via llama.cpp, offline -- needs --model); "
                        "sim (RTL glm_model_q4k slice via vvp -- SLOW, fixed-vector tokens). "
                        "modal/llama are SOFTWARE, not the accelerator.")
    p.add_argument("--modal-url", default=None,
                   help="OpenAI /v1 base URL of the Modal vLLM server "
                        "(deploy tools/modal_glm_server.py) for --backend modal")
    p.add_argument("--modal-key", default="wpu-local", help="bearer key for --backend modal")
    p.add_argument("--model", default=None, help="GGUF path for --backend llama")
    p.add_argument("--llama-cli", default=None, help="path to a llama.cpp llama-cli binary")
    p.add_argument("--manifest", default=None,
                   help="provision_image.py manifest json (shown in the console's "
                        "provisioning panel)")
    p.add_argument("--raw", action="store_true",
                   help="skip the GLM chat template; use the naive 'role: content' "
                        "flatten (the byte-scaffold / debug path)")
    args = p.parse_args(argv)

    if args.backend == "modal":
        from wpu_modal_backend import ModalBackend
        if not args.modal_url:
            p.error("--backend modal requires --modal-url (deploy tools/modal_glm_server.py)")
        device = ModalBackend(args.modal_url, api_key=args.modal_key,
                              boot_seconds=args.boot_seconds)
        tok = make_tokenizer(args.tokenizer)          # used only for prompt fmt + usage counts
        backend_name = f"ModalBackend(software cloud, {device.model_id})"
        print("NOTE: --backend modal is the v0.1 SOFTWARE full-model demo -- REAL GLM "
              "family text from vLLM on Modal GPU cloud. It is software on cloud GPUs, "
              "NOT the WPU accelerator; the accelerator replaces it behind this same "
              "API once silicon exists (docs/PRODUCT_SPEC.md).")
    elif args.backend == "llama":
        from wpu_llama_backend import LlamaCppBackend
        if not args.model:
            p.error("--backend llama requires --model <gguf>")
        device = LlamaCppBackend(args.model, llama_cli=args.llama_cli,
                                 boot_seconds=args.boot_seconds)
        tok = make_tokenizer(args.tokenizer)          # used only for prompt fmt + usage counts
        backend_name = f"LlamaCppBackend(software, {device.model_id})"
        print("NOTE: --backend llama is the v0.1 SOFTWARE full-model backend -- REAL tokens "
              "from your GGUF via llama.cpp. It is software (CPU/GPU), NOT the WPU "
              "accelerator; swap the GGUF for GLM-5.2 on the box for the product experience.")
    elif args.backend == "sim":
        from wpu_sim_backend import SimulatorBackend
        device = SimulatorBackend()                  # slice VOCAB=256 glm_model_q4k
        tok = make_tokenizer(args.tokenizer)          # decode is best-effort (slice tokens)
        backend_name = "SimulatorBackend(glm_model_q4k/vvp)"
        print("NOTE: --backend sim runs the on-main glm_model_q4k RTL slice via vvp. It is "
              "SLOW (minutes/run) and emits REAL but UNTRAINED slice argmax tokens from the "
              "testbench's FIXED golden vectors -- they are NOT a response to your prompt and "
              "NOT language. This is a datapath co-sim witness, not a chatbot. Needs "
              "`make model-q4k` first. Use the default (mock) for the API/plumbing loop.")
    else:
        tok = make_tokenizer(args.tokenizer)
        device = MockDevice(boot_seconds=args.boot_seconds,
                            eos_token=tok.eos_id, vocab_size=tok.vocab_size)
        backend_name = "MockDevice"
    if hasattr(device, "power_on"):
        device.power_on()
    server = WPUServer(device, tok, raw=args.raw, backend_name=backend_name,
                        manifest_path=args.manifest)
    template = "raw-flatten" if server.raw else ("glm" if tok.name == "glm" else "flatten")
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    base = f"http://{args.host}:{args.port}"
    print(f"WPU server on {base}/v1  "
          f"(model={device.model_id}, backend={backend_name}, tokenizer={tok.name}, "
          f"template={template})")
    print(f"  chat:    POST {base}/v1/chat/completions [stream]   GET {base}/v1/models")
    print(f"  console: {base}/console   (health / settings / provisioning -- the control plane)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
