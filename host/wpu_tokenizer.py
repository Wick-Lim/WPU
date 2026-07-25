"""
wpu_tokenizer.py -- tokenizers for the WPU host server.

Two implementations behind one interface (encode / stream-decode / eos_id):
  * ByteTokenizer -- stdlib-only, byte-level, exact round-trip. Pairs with the byte
    MockDevice for the zero-dependency scaffold.
  * GLMTokenizer  -- the REAL GLM-5.2 BPE tokenizer (tokenizer.json via the
    `tokenizers` lib). Pairs with a real GLM-vocab backend (simulator / USB device).

`make_tokenizer(path)` returns the GLM tokenizer when `tokenizers` + a tokenizer.json
are available, else falls back to the byte tokenizer (honest degradation).
"""

from __future__ import annotations

import os


class ByteTokenizer:
    name = "byte"
    vocab_size = 256
    eos_id = 256                                   # out-of-band (not a byte)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    class _Stream:
        """Incremental UTF-8 decoder: feed byte ids, get decodable text deltas."""
        def __init__(self):
            self._buf = bytearray()

        def push(self, byte_id: int) -> str:
            if byte_id < 0 or byte_id > 255:
                return ""
            self._buf.append(byte_id)
            try:
                out = self._buf.decode("utf-8")
                self._buf.clear()
                return out
            except UnicodeDecodeError:
                return ""                          # wait for the rest of the char

        def flush(self) -> str:
            out = self._buf.decode("utf-8", errors="replace")
            self._buf.clear()
            return out

    def stream(self) -> "ByteTokenizer._Stream":
        return self._Stream()


class GLMTokenizer:
    """The real GLM-5.2 BPE tokenizer. Requires `pip install tokenizers` and the
       repo's tokenizer.json (fetch: huggingface.co/zai-org/GLM-5.2-FP8)."""
    name = "glm"

    def __init__(self, path: str):
        from tokenizers import Tokenizer            # raises if lib absent
        self._tk = Tokenizer.from_file(path)
        self.vocab_size = self._tk.get_vocab_size()
        eid = self._tk.token_to_id("<|endoftext|>")
        self.eos_id = eid if eid is not None else self.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tk.encode(text).ids

    class _Stream:
        """BPE decode is not per-token concatenation (byte-level BPE handles spacing
           AND a multi-byte UTF-8 char can span tokens), so decode the accumulated id
           list each step. HOLD emission while the decode ends in the replacement char
           U+FFFD (an incomplete trailing char) -- so we only commit at complete-char
           boundaries and never emit a mangled partial char."""
        def __init__(self, tk):
            self._tk = tk
            self._ids: list[int] = []
            self._emitted = ""

        def _delta(self, full: str) -> str:
            if full.startswith(self._emitted):
                delta = full[len(self._emitted):]
            else:                                   # rare divergence: re-sync on prefix
                i = 0
                m = min(len(full), len(self._emitted))
                while i < m and full[i] == self._emitted[i]:
                    i += 1
                delta = full[i:]
            self._emitted = full
            return delta

        def push(self, tid: int) -> str:
            self._ids.append(tid)
            full = self._tk.decode(self._ids)
            if full.endswith("�"):
                return ""                           # incomplete trailing char; wait
            return self._delta(full)

        def flush(self) -> str:
            return self._delta(self._tk.decode(self._ids))

    def stream(self) -> "GLMTokenizer._Stream":
        return self._Stream(self._tk)


def _default_tokenizer_path() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("WPU_TOKENIZER_JSON"),
                 os.path.join(here, "tokenizer.json"),
                 os.path.join(here, "..", "tokenizer.json")):
        if cand and os.path.exists(cand):
            return cand
    return None


def make_tokenizer(path: str | None = None):
    """GLM tokenizer if (tokenizers lib + tokenizer.json) available, else byte."""
    path = path or _default_tokenizer_path()
    if path:
        try:
            return GLMTokenizer(path)
        except Exception as e:                      # lib missing / bad file
            print(f"[tokenizer] GLM unavailable ({type(e).__name__}: {e}); "
                  f"falling back to byte tokenizer")
    return ByteTokenizer()
