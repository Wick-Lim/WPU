"""
wpu_device.py -- host-side driver abstraction for the WPU USB-C device.

This mirrors the RTL host interface of `glm_q4k_system_cdc` (the production 2-clock
top) EXACTLY, so the same driver code that talks to a MockDevice today talks to the
real USB-C device tomorrow -- only the backend changes.

RTL host interface (host_clk domain, src/glm_q4k_system_cdc.v):
    in :  start, prompt_tok[TOKW], start_pos[POSW], s_len[IDXW+1]
    out:  busy, done, next_tok[TOKW], tok_valid
Boot: inference is released by `boot_loader.done` (the ~28 GB resident set is DMA'd
Flash->DDR5 first), NOT by power-on -- see docs/OPERATION_FLOW.md sec 1.

What crosses USB-C is ONLY these token IDs + control; the heavy weight/KV traffic is
internal to the device. Session KV lives in the device's DDR5 (per session).

Scope (honest): this is the D2 software scaffold (docs/USBC_PRODUCT_PLAN.md). The
protocol + generation loop are real; MockDevice returns a clearly-labelled canned
response (the plumbing, not the model). The real backend is either the RTL simulator
(slice model) or the shipped device / full-model runtime.
"""

from __future__ import annotations

import abc
import dataclasses
import time


@dataclasses.dataclass
class SamplingParams:
    """OpenAI-style sampling parameters, threaded from the HTTP request to the device.

    HONEST host-vs-device split (see host/README.md):
      * HOST-SIDE (this scaffold enforces): `max_tokens` (decode cap) and `stop`
        (truncate when a stop string appears in the decoded text) -- both real and
        applied in wpu_server.py. `seed` is threaded through to the device.
      * DEVICE-SIDE (`sampler.v` samples ON-DEVICE from logits): `temperature`,
        `top_p`, `top_k`, `seed`. The MockDevice returns the ARGMAX token (greedy) and
        IGNORES temperature/top_p/top_k -- true sampling needs a logits-capable
        backend, which programs these via `configure_sampling()`. Faking sampling on a
        canned/argmax stream would be dishonest, so we don't.
      * `presence_penalty`/`frequency_penalty` are accepted and ignored (no host-side
        logit bias in the scaffold).
    """

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0                                   # 0 = disabled
    stop: list = dataclasses.field(default_factory=list)
    seed: "int | None" = None
    presence_penalty: float = 0.0                    # accepted, ignored (device-side)
    frequency_penalty: float = 0.0                   # accepted, ignored (device-side)

    @classmethod
    def from_request(cls, req: dict) -> "SamplingParams":
        """Parse an OpenAI `/v1/chat/completions` request dict into SamplingParams,
           tolerating missing/malformed fields (falls back to defaults)."""
        def _num(key, default, cast):
            v = req.get(key, default)
            if v is None:
                return default
            try:
                return cast(v)
            except (TypeError, ValueError):
                return default

        stop = req.get("stop")
        if isinstance(stop, str):
            stops = [stop] if stop else []
        elif isinstance(stop, (list, tuple)):
            stops = [s for s in stop if isinstance(s, str) and s]
        else:
            stops = []

        seed = req.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None

        return cls(
            max_tokens=max(1, _num("max_tokens", 256, int)),
            temperature=_num("temperature", 1.0, float),
            top_p=_num("top_p", 1.0, float),
            top_k=_num("top_k", 0, int),
            stop=stops[:4],                          # OpenAI allows up to 4 stops
            seed=seed,
            presence_penalty=_num("presence_penalty", 0.0, float),
            frequency_penalty=_num("frequency_penalty", 0.0, float),
        )


class DeviceState:
    OFF = "off"
    BOOTING = "booting"          # boot_loader streaming resident set Flash->DDR5
    READY = "ready"              # boot_loader.done asserted -> inference released
    BUSY = "busy"                # a decode step in flight


class WPUDevice(abc.ABC):
    """Abstract WPU device. Concrete backends: MockDevice (here), a simulator-backed
       driver, or the real USB-C driver. The generation loop in wpu_server.py uses
       ONLY this interface."""

    #: vocabulary size the device decodes over (real GLM-5.2: 154880; scaffold: 256).
    vocab_size: int = 256
    #: model identifier surfaced to OpenAI clients.
    model_id: str = "wpu-glm-5.2-q4k"

    #: False on backends that cannot resume a session mid-sequence (replay stubs that
    #: ignore prompt_ids, or any device whose KV is not position-addressed). Setting
    #: this False makes generate() re-feed every prompt token, i.e. the pre-cache
    #: behaviour -- correctness never depends on the cache being ON.
    supports_prefix_cache: bool = True

    def __init__(self) -> None:
        self.state = DeviceState.OFF
        self._boot_t0 = None
        #: token ids whose KV is resident in the device for the CURRENT session, in
        #: position order: _kv_ids[i] was fed at position i, so its KV occupies row i.
        #: This is the ONLY thing the prefix cache trusts; reset_session() clears it.
        self._kv_ids: list[int] = []
        #: prefix-cache counters (surfaced by the console / telemetry).
        self.prefix_stats = {"reused": 0, "fed": 0, "turns": 0}

    # ---- lifecycle (mirrors power-on -> boot_loader.done -> ready) ---------------
    @abc.abstractmethod
    def _boot_seconds(self) -> float:
        """Modelled boot duration (resident-set load). Real device ~1-2 s [EST]."""

    def power_on(self) -> None:
        """Begin the boot sequence (async). Ready is gated by boot_loader.done."""
        if self.state in (DeviceState.READY, DeviceState.BUSY):
            return
        self.state = DeviceState.BOOTING
        self._boot_t0 = time.monotonic()

    def poll_ready(self) -> bool:
        """True once boot_loader.done would be asserted (resident set in DDR5)."""
        if self.state == DeviceState.OFF:
            self.power_on()
        if self.state == DeviceState.BOOTING:
            if time.monotonic() - (self._boot_t0 or 0) >= self._boot_seconds():
                self.state = DeviceState.READY
        return self.state in (DeviceState.READY, DeviceState.BUSY)

    def wait_ready(self, timeout: float = 30.0) -> None:
        t0 = time.monotonic()
        while not self.poll_ready():
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("WPU device did not reach READY (boot_loader.done)")
            time.sleep(0.02)

    # ---- session (KV lives in the device's DDR5, per session) --------------------
    @abc.abstractmethod
    def reset_session(self) -> None:
        """Clear the KV cache / conversation state for a fresh sequence.

        Implementations MUST call `super().reset_session()` (this body). Dropping the
        device's KV without clearing `_kv_ids` would leave the host believing a prefix
        is resident that the device just discarded -- the prefix cache would then skip
        re-feeding tokens whose KV no longer exists, silently corrupting attention."""
        self._kv_ids = []

    # ---- context bound (the ring aliases; see context_capacity) -------------------
    class ContextOverflow(RuntimeError):
        """The turn would run past the resident KV ring, where positions alias."""

    def _check_context_fits(self, n_prompt: int, max_new_tokens: int,
                            start_pos: int = 0) -> None:
        """Refuse a turn that would address a position beyond the ring.

        Checked against the LAST position the turn would touch -- prompt AND the tokens
        it is allowed to generate -- because the aliasing happens on write, so catching
        it after prefill would already have clobbered live rows. The caller's policy
        (clear the context and start fresh, or compact it) belongs in the host/UI; this
        layer's job is only to make the failure visible instead of silent."""
        if self.context_capacity is None:
            return
        last = start_pos + n_prompt + max(0, max_new_tokens)
        if last > self.context_capacity:
            raise self.ContextOverflow(
                f"turn needs positions up to {last:,} but the resident KV ring holds "
                f"{self.context_capacity:,} (prompt {n_prompt:,} + up to "
                f"{max_new_tokens:,} new, from {start_pos:,}). The ring addresses by "
                f"bit-slice modulo, so going past it would silently overwrite the "
                f"oldest rows and corrupt attention rather than error. Clear the "
                f"context (reset_session) and start a fresh one, or shorten the input.")

    # ---- prefix cache (D5: 캐싱 = 무조건, docs/PRODUCT_SPEC.md) -------------------
    def _prefix_reuse(self, prompt_ids: list[int]) -> int:
        """How many leading prompt tokens are ALREADY resident (so must not be re-fed).

        Chat re-sends the whole conversation every turn (wpu_server.py formats all
        messages), so turn N's prompt is turn N-1's prompt + reply + the new user
        message -- a long shared prefix. Reusing it is the single biggest lever on the
        box: prefill is ~25.3 GB of weight traffic PER TOKEN with no speculation to
        amortise it (tokens are known, so U=1/A=1), i.e. ~61 tok/s [EST] -- far slower
        per token than decode.

        Device contract this relies on (mirrors the RTL host handshake):
          * `step(tok, pos, s_len)` addresses KV BY POSITION -- token fed at `pos`
            occupies row `pos` -- so resuming at `pos = n` simply continues the row.
          * `s_len` bounds the attention window to rows [0, s_len), so on divergence we
            just overwrite from the fork point: rows beyond it are never attended and
            need no explicit invalidation.
        Capped at len(prompt_ids)-1 so at least ONE token is always fed: the device's
        next-token output is the response to the LAST token fed, so a fully-cached
        prompt would otherwise produce nothing to return."""
        if not self.supports_prefix_cache or not self._kv_ids:
            return 0
        limit = min(len(self._kv_ids), len(prompt_ids) - 1)
        n = 0
        while n < limit and self._kv_ids[n] == prompt_ids[n]:
            n += 1
        return max(0, n)

    # ---- one decode step (mirrors: assert start+inputs -> tok_valid+next_tok) ----
    @abc.abstractmethod
    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        """Feed one token at `start_pos` (with running length `s_len`) and return the
           device's next-token id. This is EXACTLY the RTL host handshake:
             start=1, prompt_tok, start_pos, s_len  ->  (busy) -> tok_valid, next_tok."""

    #: the token id that ends generation (EOS). Real value comes from the tokenizer.
    eos_token: int = -1

    #: Resident KV ring capacity, in token positions. None = unknown/unbounded (replay
    #: stubs). Set it on any backend whose KV is a real ring, because THE RING ALIASES:
    #: `kv_cache_pager.v:73-74` states "RESIDENT must be a POWER OF TWO (the ring uses
    #: bit-slice modulo slot = pos[RPTRW-1:0])", so position `capacity` lands on slot 0
    #: and SILENTLY OVERWRITES position 0. Nothing errors; the model keeps emitting
    #: fluent tokens while attention reads clobbered keys. For this product's buyers
    #: that reads as hallucination, which is why the guard below fails LOUDLY instead:
    #: a refusal is recoverable, a silently wrong answer is not.
    #: The box is single-context by construction -- the production top never passes
    #: NSEQ (it defaults to 1; multi-context lives only in the unshipped
    #: `glm_q4k_soc_ms`) -- so this one ring is the whole context budget.
    context_capacity: int | None = None

    #: sampling programmed for the current generate() (None => greedy/argmax).
    _sampling: "SamplingParams | None" = None

    def configure_sampling(self, sampling: "SamplingParams") -> None:
        """Program device-side sampling for the next generate() call.

        The RTL `sampler.v` samples ON-DEVICE from the model's logits
        (temperature/top_p/top_k with `seed`). The MockDevice (and the slice
        SimulatorBackend) return the ARGMAX token -- i.e. GREEDY -- so they IGNORE
        temperature/top_p/top_k here: there are no host-side logits to sample from a
        canned/argmax stream, and faking it would be dishonest. A logits-capable
        backend OVERRIDES this to write the sampler's config registers; the default
        just records the params so a real backend / inspector can read them."""
        self._sampling = sampling

    def prefill(self, prompt_ids, start_pos: int) -> int:
        """Feed the prompt tokens and return the FIRST next-token. Default = real
           device semantics: one `step` per prompt token, the last output is the
           first generated token (the model's KV is built along the way)."""
        last_out = None
        pos = start_pos
        for tok in prompt_ids:
            last_out = self.step(tok, pos, pos + 1)
            pos += 1
        return last_out

    def generate(self, prompt_ids, max_new_tokens: int, start_pos: int = 0,
                 sampling: "SamplingParams | None" = None):
        """Autoregressive loop: prefill `prompt_ids` -> first token, then feed each
           generated token back. Yields token ids as they decode (server streams).
           Stops at eos_token or max_new_tokens. `sampling` (optional) is programmed
           into the device via configure_sampling() -- honored device-side by a
           logits-capable backend; the mock is greedy and ignores it."""
        self.wait_ready()
        prompt_ids = list(prompt_ids)
        self._check_context_fits(len(prompt_ids), max_new_tokens, start_pos)
        # Prefix cache: re-feed ONLY the tokens the device does not already hold. The
        # cache is keyed on absolute position, so it applies to a sequence indexed from
        # 0; a caller placing the prompt elsewhere is managing positions itself.
        n_reuse = self._prefix_reuse(prompt_ids) if start_pos == 0 else 0
        # Clear ONLY for a fresh sequence that reuses nothing. start_pos != 0 says
        # "continue the session that already holds rows [0, start_pos)" -- resetting
        # there would destroy the very rows the caller is continuing from.
        if start_pos == 0 and n_reuse == 0:
            self.reset_session()                    # also clears _kv_ids (see contract)
        if sampling is not None:
            self.configure_sampling(sampling)
        self.prefix_stats["turns"] += 1
        self.prefix_stats["reused"] += n_reuse
        self.prefix_stats["fed"] += len(prompt_ids) - n_reuse
        # Feed the uncached tail at its true positions. On divergence (n_reuse < len(
        # _kv_ids)) this overwrites from the fork point; s_len bounds attention to the
        # live rows, so the stale tail beyond is never read.
        cur = self.prefill(prompt_ids[n_reuse:], start_pos + n_reuse)
        # Track what the device now holds, BY POSITION. Rows [0, start_pos) are whatever
        # the session already held (start_pos != 0 means "continue this session"); the
        # prompt occupies rows start_pos onward. If those leading rows are unaccounted
        # for we must not guess -- claiming a row we did not place would silently
        # corrupt a later turn's reuse -- so we drop the cache and stop tracking for
        # this turn (the decode loop below must honour that, or it would rebuild a
        # cache whose positions are wrong).
        track = len(self._kv_ids) >= start_pos
        self._kv_ids = self._kv_ids[:start_pos] + prompt_ids if track else []
        pos = start_pos + len(prompt_ids)
        produced = 0
        while produced < max_new_tokens and cur is not None and cur != self.eos_token:
            yield cur
            produced += 1
            fed = cur                               # the token whose KV this step lands
            cur = self.step(fed, pos, pos + 1)
            if track:
                self._kv_ids.append(fed)            # ...record `fed`, not the output
            pos += 1


class MockDevice(WPUDevice):
    """A backend that implements the full protocol but REPLAYS a fixed list of token
       ids (set by the server for this turn). Tokenizer-agnostic -- the server encodes
       a clearly-labelled canned reply with whatever tokenizer is active (byte OR the
       real GLM BPE) and the mock replays those ids -- so it proves the end-to-end
       plumbing (HTTP -> tokenize -> device protocol -> detokenize -> HTTP streaming)
       for BOTH vocabularies WITHOUT pretending to be the model. A real device would
       run glm_q4k_system_cdc and emit real next-token ids here."""

    def __init__(self, boot_seconds: float = 0.4, eos_token: int = 256,
                 vocab_size: int = 256) -> None:
        super().__init__()
        self._boot = boot_seconds
        self.eos_token = eos_token                  # server sets this to tok.eos_id
        self.vocab_size = vocab_size
        self._reply_ids: list[int] = []
        self._cursor = 0

    def _boot_seconds(self) -> float:
        return self._boot

    # This mock REPLAYS a canned list: its "session" is a decode cursor, not
    # position-addressed KV, and it ignores prompt_ids entirely. Skipping tokens would
    # therefore change what it replays rather than save real work -- so no prefix cache.
    supports_prefix_cache = False

    def reset_session(self) -> None:
        # New sequence: reset the decode cursor (analogous to clearing the device's
        # DDR5 KV). The per-turn reply ids set by the server survive (set just before
        # generate()).
        super().reset_session()
        self._cursor = 0

    def set_reply_ids(self, ids) -> None:
        """Server sets this turn's canned reply as token ids (in the active tokenizer's
           space), terminated by eos."""
        self._reply_ids = list(ids) + [self.eos_token]
        self._cursor = 0

    def prefill(self, prompt_ids, start_pos: int) -> int:
        # Mock: the prompt builds no real KV and does NOT consume the reply; the first
        # decode token is reply[0]. (A real device streams the prompt through
        # glm_q4k_system_cdc to build its DDR5 KV.)
        self._cursor = 0
        out = self._reply_ids[0] if self._reply_ids else self.eos_token
        self._cursor = 1
        return out

    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        # Mock decode: emit the next reply id.
        self.state = DeviceState.BUSY
        out = (self._reply_ids[self._cursor]
               if self._cursor < len(self._reply_ids) else self.eos_token)
        self._cursor += 1
        self.state = DeviceState.READY
        return out
