#!/usr/bin/env python3
"""Self-tests for the WPU host scaffold (no HTTP, no network): device protocol,
   tokenizers (byte + real GLM BPE if available), boot gate, and the server
   generation loop. Run:  python3 host/test_wpu.py   ->  exit 0 on PASS."""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wpu_device import (MockDevice, WPUDevice, DeviceState,   # noqa: E402
                         SamplingParams)
from wpu_tokenizer import ByteTokenizer, make_tokenizer, GLMTokenizer  # noqa: E402
from wpu_server import WPUServer                           # noqa: E402
from wpu_chat_template import apply_chat_template            # noqa: E402


def _byte_dev(**kw):
    return MockDevice(eos_token=ByteTokenizer.eos_id, vocab_size=256, **kw)


class _FixedDevice(WPUDevice):
    """Test device that yields a FIXED list of token ids -- bypasses MockDevice's
       canned-reply priming so tests can drive arbitrary decoded output (stop
       sequences, finish_reason). Records the SamplingParams it was configured with."""

    def __init__(self, ids, eos=ByteTokenizer.eos_id):
        super().__init__()
        self.eos_token = eos
        self._ids = list(ids)
        self._c = 0
        self.state = DeviceState.READY
        self.seen_sampling = None

    def _boot_seconds(self):
        return 0.0

    supports_prefix_cache = False        # replays a fixed list; no position-addressed KV

    def reset_session(self):
        super().reset_session()
        self._c = 0

    def configure_sampling(self, sampling):
        self.seen_sampling = sampling

    def prefill(self, prompt_ids, start_pos):
        self._c = 0
        out = self._ids[0] if self._ids else self.eos_token
        self._c = 1
        return out

    def step(self, prompt_tok, start_pos, s_len):
        out = self._ids[self._c] if self._c < len(self._ids) else self.eos_token
        self._c += 1
        return out


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello", "WPU 디바이스", "emoji 🚀 test", ""]:
        ids = tok.encode(s)
        st = tok.stream()
        out = "".join(st.push(i) for i in ids) + st.flush()
        assert out == s, (s, out)


def test_boot_gate():
    d = _byte_dev(boot_seconds=0.3)
    d.power_on()
    assert d.state == DeviceState.BOOTING
    assert not d.poll_ready()                    # not ready before boot_loader.done
    d.wait_ready(timeout=5)
    assert d.state in (DeviceState.READY, DeviceState.BUSY)


def test_generate_streams_full_reply():
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"ABCDE"))
    out = list(d.generate(prompt_ids=[104, 105], max_new_tokens=100))   # "hi"
    assert out == [65, 66, 67, 68, 69], out      # eos not yielded


def test_generate_respects_max_tokens():
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"A very long canned reply that exceeds the cap"))
    out = list(d.generate(prompt_ids=[104], max_new_tokens=5))
    assert len(out) == 5, out


def test_prefill_does_not_truncate():
    # regression: prefill steps must NOT consume the reply (the first-cut bug).
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"HELLO"))
    long_prompt = list(range(50))                # 50 prompt tokens
    out = bytes(d.generate(long_prompt, max_new_tokens=100)).decode()
    assert out == "HELLO", out


def test_server_end_to_end_byte():
    d = _byte_dev(boot_seconds=0.0)
    srv = WPUServer(d, ByteTokenizer())
    msgs = [{"role": "user", "content": "hello wpu"}]
    text, n_prompt = srv.generate_text(msgs, max_tokens=500)
    assert "protocol round-trip OK" in text, text
    assert "byte tokenizer" in text, text
    assert n_prompt > 0
    d.reset_session()
    streamed = "".join(srv.generate_stream(msgs, max_tokens=500))
    assert streamed == text, (streamed, text)


def test_glm_tokenizer_if_available():
    """Real GLM BPE round-trip + server end-to-end -- SKIPPED if `tokenizers` or
       tokenizer.json aren't present (byte fallback covers the plumbing)."""
    tok = make_tokenizer()
    if not isinstance(tok, GLMTokenizer):
        print("  SKIP test_glm_tokenizer_if_available (no tokenizers/tokenizer.json)")
        return
    for s in ["hello world", "WPU 디바이스 test", "def f(x):\n    return x+1"]:
        ids = tok.encode(s)
        st = tok.stream()
        out = "".join(st.push(i) for i in ids) + st.flush()
        assert out == s, (s, out)
    # server end-to-end through the REAL GLM vocab
    d = MockDevice(boot_seconds=0.0, eos_token=tok.eos_id, vocab_size=tok.vocab_size)
    srv = WPUServer(d, tok)
    msgs = [{"role": "user", "content": "hi"}]
    text, _ = srv.generate_text(msgs, max_tokens=500)
    assert "protocol round-trip OK" in text and "glm tokenizer" in text, text
    print(f"  (GLM vocab_size={tok.vocab_size}, eos_id={tok.eos_id})")


def test_simulator_backend_parse():
    """SimulatorBackend parse + protocol via a fake `vvp` (echo the per-case lines the
       real glm_model_q4k slice sim prints). The full vvp run is separately validated
       (`make model-q4k`) -- minutes-long, too slow for CI -- so we stub the subprocess
       with the exact TB output format and check the parse + device protocol. The stub
       tokens {13,3,13} are the real SPEC_SLICE argmax vectors (build/mq4k_s)."""
    import os
    import tempfile
    from wpu_sim_backend import SimulatorBackend
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("case 0: token=3 pos=1 s_len=2 -> argmax=13 (golden 13)    MATCH\n"
                "case 1: token=7 pos=3 s_len=2 -> argmax=3 (golden 3)    MATCH\n"
                "case 2: token=11 pos=0 s_len=1 -> argmax=13 (golden 13)    MATCH\n"
                "ALL 99 TESTS PASSED\n")
        path = f.name
    try:
        dev = SimulatorBackend(vvp="cat", vvp_binary=path, cwd="/tmp")
        dev.power_on()
        assert list(dev.generate([1, 2, 3], max_new_tokens=16)) == [13, 3, 13]
        dev.reset_session()
        assert list(dev.generate([9], max_new_tokens=16)) == [13, 3, 13]   # cached
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# TASK 1 -- GLM chat template
# ---------------------------------------------------------------------------
def test_chat_template_glm_structure():
    """GLM-5.2 template: leading [gMASK]<sop>, per-role special tokens in order, and
       the trailing generation prompt (thinking on -> <think>, off -> <think></think>)."""
    msgs = [{"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi there"}]
    p = apply_chat_template(msgs)
    assert p.startswith("[gMASK]<sop>"), p
    assert "<|system|>Reasoning Effort: Max" in p          # GLM-5.2 thinking-effort turn
    assert "<|system|>be nice" in p
    assert "<|user|>hi there" in p
    assert p.endswith("<|assistant|><think>"), p           # generation prompt (thinking)
    assert p.index("<|user|>hi there") < p.rindex("<|assistant|>")   # ordering
    # thinking disabled: no reasoning-effort turn, closed think block
    p2 = apply_chat_template(msgs, enable_thinking=False)
    assert "Reasoning Effort" not in p2
    assert p2.endswith("<|assistant|><think></think>"), p2
    # reasoning_effort=high -> "High"
    assert "Reasoning Effort: High" in apply_chat_template(msgs, reasoning_effort="high")
    # no generation prompt
    assert not apply_chat_template(msgs, add_generation_prompt=False).endswith("<think>")


def test_chat_template_clears_assistant_history():
    """Historical assistant turns render as <|assistant|><think></think>{answer} with
       prior <think> reasoning stripped (template default)."""
    p = apply_chat_template([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "<think>secret reasoning</think>the answer"},
        {"role": "user", "content": "b"}])
    assert "secret reasoning" not in p, p
    assert "<|assistant|><think></think>the answer" in p, p


def test_chat_template_multimodal_reminder():
    """Non-text content parts fall back to the template's media <reminder>."""
    p = apply_chat_template([{"role": "user", "content": [
        {"type": "text", "text": "look:"},
        {"type": "image_url", "image_url": {"url": "x"}}]}])
    assert "<|user|>look:" in p
    assert "unable to process this image" in p, p


def test_server_prompt_dispatch():
    """Server applies the GLM template when the tokenizer name is 'glm', else the naive
       flatten; --raw forces the flatten even for GLM."""
    d = _byte_dev(boot_seconds=0.0)
    msgs = [{"role": "user", "content": "hi"}]
    assert WPUServer(d, ByteTokenizer()).prompt_text(msgs) == "user: hi"   # byte->flatten

    class _FakeGLM(ByteTokenizer):
        name = "glm"
    glm_prompt = WPUServer(d, _FakeGLM()).prompt_text(msgs)
    assert glm_prompt.startswith("[gMASK]<sop>") and "<|user|>hi" in glm_prompt
    assert WPUServer(d, _FakeGLM(), raw=True).prompt_text(msgs) == "user: hi"  # --raw


def test_chat_template_encodes_special_tokens_if_glm():
    """Through the REAL GLM BPE, template special tokens encode as SINGLE ids -- SKIPPED
       without tokenizers/tokenizer.json."""
    tok = make_tokenizer()
    if not isinstance(tok, GLMTokenizer):
        print("  SKIP test_chat_template_encodes_special_tokens_if_glm (no GLM tok)")
        return
    ids = tok.encode(apply_chat_template([{"role": "user", "content": "hi"}]))
    for sid in (154822, 154824, 154826, 154827, 154828):   # gMASK/sop/system/user/asst
        assert sid in ids, (sid, ids)


# ---------------------------------------------------------------------------
# TASK 2 -- OpenAI sampling parameters
# ---------------------------------------------------------------------------
def test_sampling_params_from_request():
    sp = SamplingParams.from_request({
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 32,
        "stop": ["\n\n", "END"], "seed": 123,
        "presence_penalty": 0.5, "frequency_penalty": 0.2})
    assert (sp.temperature, sp.top_p, sp.top_k) == (0.7, 0.9, 40)
    assert sp.max_tokens == 32 and sp.seed == 123
    assert sp.stop == ["\n\n", "END"]
    assert sp.presence_penalty == 0.5 and sp.frequency_penalty == 0.2
    assert SamplingParams.from_request({"stop": "STOP"}).stop == ["STOP"]   # str->list
    assert SamplingParams.from_request({}).stop == []
    bad = SamplingParams.from_request({"temperature": "hot", "max_tokens": None})
    assert bad.temperature == 1.0 and bad.max_tokens == 256                 # defaults


def test_stop_sequence_truncates():
    """A stop string in the decoded output truncates it (exclusive) and yields
       finish_reason 'stop' -- streaming-safe across a multi-token stop."""
    d = _FixedDevice(list(b"hello STOP world"))
    srv = WPUServer(d, ByteTokenizer())
    sp = SamplingParams(max_tokens=100, stop=["STOP"])
    text, _, finish = srv.complete([{"role": "user", "content": "x"}], sp)
    assert text == "hello ", repr(text)
    assert finish == "stop", finish
    # streaming truncates identically (and never emits a partial stop)
    d.reset_session()
    streamed = "".join(srv.stream([{"role": "user", "content": "x"}], sp))
    assert streamed == "hello ", repr(streamed)


def test_finish_reason_length_vs_stop():
    """finish_reason: 'length' when max_tokens caps generation, 'stop' at natural eos."""
    d = _FixedDevice(list(b"ABCDEFGHIJ"))
    srv = WPUServer(d, ByteTokenizer())
    text, _, finish = srv.complete([{"role": "user", "content": "x"}],
                                   SamplingParams(max_tokens=4))
    assert text == "ABCD" and finish == "length", (text, finish)
    d2 = _FixedDevice(list(b"AB"))
    text2, _, finish2 = WPUServer(d2, ByteTokenizer()).complete(
        [{"role": "user", "content": "x"}], SamplingParams(max_tokens=100))
    assert text2 == "AB" and finish2 == "stop", (text2, finish2)


def test_sampling_params_reach_device():
    """temperature/top_p/top_k/seed are plumbed to the device (configure_sampling)."""
    d = _FixedDevice(list(b"hi"))
    sp = SamplingParams(max_tokens=8, temperature=0.3, top_k=5, seed=7)
    WPUServer(d, ByteTokenizer()).complete([{"role": "user", "content": "x"}], sp)
    assert d.seen_sampling is sp, "sampling not plumbed to device"
    assert d.seen_sampling.temperature == 0.3 and d.seen_sampling.seed == 7


def test_mockdevice_accepts_and_ignores_sampling():
    """MockDevice accepts SamplingParams (records them) but IGNORES them: greedy replay
       is unchanged (honest -- true sampling is device-side, not faked on canned ids)."""
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"XYZ"))
    sp = SamplingParams(max_tokens=100, temperature=0.0, top_k=1, seed=99)
    out = bytes(d.generate([1, 2, 3], max_new_tokens=100, sampling=sp)).decode()
    assert out == "XYZ", out                       # output unaffected by sampling
    assert d._sampling is sp                        # accepted/recorded


# ---------------------------------------------------------------------------
# PREFIX / KV CACHE  (D5 "캐싱 = 무조건", docs/PRODUCT_SPEC.md)
# ---------------------------------------------------------------------------
class _KVDevice(WPUDevice):
    """Device that models POSITION-ADDRESSED KV honestly, so these tests have real
       power to catch a cache bug.

       `step(tok, pos, s_len)` writes row `pos` and returns a next token derived from
       the WHOLE attended window rows[0:s_len] -- exactly the dependency a real
       attention has. So any prefix-cache error changes the output:
         * a row never fed        -> a HOLE in the window (asserted below),
         * a stale row reused     -> different window contents -> different token,
         * a token fed at the wrong position -> different window order -> different token.
       A cache that merely *skips work* cannot pass by accident."""
    supports_prefix_cache = True
    eos_token = -1                                   # unreachable: outputs are >= 1

    def __init__(self):
        super().__init__()
        self.rows: dict[int, int] = {}
        self.steps = 0                               # real work done (die passes)
        self.state = DeviceState.READY

    def _boot_seconds(self):
        return 0.0

    def reset_session(self):
        super().reset_session()
        self.rows = {}

    def step(self, prompt_tok, start_pos, s_len):
        self.rows[start_pos] = prompt_tok            # KV of `prompt_tok` lands at row pos
        self.steps += 1
        win = [self.rows.get(i) for i in range(s_len)]
        assert all(v is not None for v in win), \
            f"attention window [0,{s_len}) has a HOLE: {win} -- a token was never fed"
        h = 0
        for i, v in enumerate(win):                  # order-sensitive over the window
            h = (h * 31 + v * (i + 1)) % 251
        return h + 1


def _chat(dev, n_turns=4, sys_len=40, reply_len=5):
    """Drive a realistic chat: the server re-sends the WHOLE conversation each turn,
       so turn N's prompt = turn N-1's prompt + reply + the new user message."""
    convo, replies = list(range(1, sys_len + 1)), []
    for t in range(n_turns):
        convo = convo + [200 + t]                    # this turn's user message
        reply = list(dev.generate(convo, max_new_tokens=reply_len))
        replies.append(reply)
        convo = convo + reply                        # the reply joins the history
    return replies


def test_prefix_cache_is_output_equivalent():
    """THE load-bearing test: the cache must be BEHAVIOUR-PRESERVING -- identical
       tokens to the uncached path, just less work. Same discipline as the RTL
       equivalence gates: prove it changes nothing, THEN claim the win."""
    cached, plain = _KVDevice(), _KVDevice()
    plain.supports_prefix_cache = False              # the pre-cache behaviour
    assert _chat(cached) == _chat(plain), "prefix cache CHANGED the output"
    assert cached.steps < plain.steps, (cached.steps, plain.steps)


def test_prefix_cache_saves_real_work():
    """The win must be large and must grow with conversation length."""
    cached, plain = _KVDevice(), _KVDevice()
    plain.supports_prefix_cache = False
    _chat(cached, n_turns=6), _chat(plain, n_turns=6)
    assert cached.steps * 3 < plain.steps, (cached.steps, plain.steps)
    # Every prompt token past turn 1 should be a hit except the fork (user msg + the
    # one always-fed token), so reuse must dominate the tokens fed.
    assert cached.prefix_stats["reused"] > cached.prefix_stats["fed"]


def test_context_overflow_refuses_instead_of_aliasing():
    """The KV ring addresses by bit-slice modulo (kv_cache_pager.v:73-74), so a turn
       past its capacity would overwrite the oldest rows and corrupt attention while
       still emitting fluent tokens. Refuse loudly instead -- for this product's buyers
       a silently wrong answer is the one failure that cannot be recovered from."""
    d = _KVDevice()
    d.context_capacity = 64
    list(d.generate([1, 2, 3], max_new_tokens=4))                 # 7 <= 64: fine
    try:
        list(d.generate(list(range(1, 60)), max_new_tokens=10))   # 69 > 64
    except WPUDevice.ContextOverflow as e:
        assert "64" in str(e), e
    else:
        raise AssertionError("ran past the ring instead of refusing")


def test_context_overflow_counts_generated_tokens_too():
    """The bound must include what the turn will GENERATE, not just the prompt: the
       aliasing happens on write, so a prompt that fits but whose reply does not would
       clobber live rows mid-generation -- after the caller has already seen output."""
    d = _KVDevice()
    d.context_capacity = 20
    try:
        list(d.generate([1, 2, 3], max_new_tokens=100))           # prompt fits, reply doesn't
    except WPUDevice.ContextOverflow:
        pass
    else:
        raise AssertionError("only bounded the prompt; generation can still alias")


def test_context_capacity_unset_is_unbounded():
    """Replay stubs have no real ring; the guard must not invent a limit for them."""
    d = _KVDevice()
    assert d.context_capacity is None
    list(d.generate([1, 2, 3], max_new_tokens=4))                 # no raise


def test_prefix_cache_records_generated_tokens():
    """The reply must join the cache EXACTLY: after a turn the device holds the prompt
       followed by the tokens it generated, at those positions.

       Pinned as an invariant because drifting it is INVISIBLE to the equivalence test:
       if `_kv_ids` is off by one, reuse stops at the reply boundary and the whole reply
       is re-fed every turn -- output stays correct, most of the win silently vanishes.
       Chat re-sends the reply next turn, so this is where the win actually comes from."""
    d = _KVDevice()
    prompt = [1, 2, 3]
    reply = list(d.generate(prompt, max_new_tokens=4))
    assert d._kv_ids == prompt + reply, (d._kv_ids, prompt + reply)


def test_prefix_cache_reuses_the_previous_reply():
    """Turn 2 must re-feed only the fork: the new user message (+ the always-fed one)."""
    d = _KVDevice()
    convo = [1, 2, 3]
    reply = list(d.generate(convo, max_new_tokens=4))
    convo = convo + reply + [99]                     # history + reply + new user msg
    n0 = d.steps
    list(d.generate(convo, max_new_tokens=2))
    fed = d.steps - n0 - 2                           # minus the 2 decode steps
    assert fed <= 2, f"re-fed {fed} prompt tokens; the reply should have been cached"


def test_prefix_cache_tracks_positions_on_continuation():
    """start_pos != 0 means "continue this session": the prompt lands at rows start_pos
       onward, NOT row 0. Recording it as row 0 would corrupt the NEXT turn's reuse --
       an error invisible on the turn that causes it."""
    d = _KVDevice()
    list(d.generate([1, 2, 3], max_new_tokens=2))
    held = list(d._kv_ids)
    list(d.generate([7, 8], max_new_tokens=1, start_pos=len(held)))
    assert d._kv_ids[:len(held)] == held, "continuation clobbered the resident rows"
    assert d._kv_ids[len(held):len(held) + 2] == [7, 8], d._kv_ids


class _LenientKVDevice(_KVDevice):
    """As _KVDevice but does NOT police the caller (a real device need not assert on a
       hole). Used to test the host's own bookkeeping in isolation."""
    def step(self, prompt_tok, start_pos, s_len):
        self.rows[start_pos] = prompt_tok
        self.steps += 1
        return (prompt_tok + start_pos) % 251 + 1


def test_prefix_cache_drops_rather_than_guesses():
    """If the rows before start_pos are unaccounted for, the cache must DROP itself, not
       guess. Recording the prompt as if it began at row 0 would claim rows the host
       never placed -- and the device may not police that for us."""
    d = _LenientKVDevice()
    list(d.generate([1, 2], max_new_tokens=1, start_pos=9))    # rows 0..8 unaccounted
    assert d._kv_ids == [], f"cache claimed rows it never placed: {d._kv_ids}"


def test_prefix_cache_survives_reasoning_strip():
    """GLM's template CLEARS historical <think> reasoning, but the device generated it
       and so its KV holds it. Turn N+1's prompt therefore FORKS from the KV at the last
       reply's <think> -- structural to a thinking model, not a bug.

       What must hold: the fork costs only the LAST reply + the new message. Everything
       before it (system prompt + earlier history) must still hit, because after the
       fork the cache records the STRIPPED rendering that later turns re-send. If this
       ever regresses to re-feeding the whole conversation, the win mostly vanishes."""
    tok = ByteTokenizer()
    sys_prompt = "You are a helpful assistant. " * 40
    gen = "<think>" + ("reasoning. " * 20) + "</think>the answer"
    d = _LenientKVDevice()
    msgs = [{"role": "system", "content": sys_prompt}]
    fed = []
    for t in range(3):
        msgs = msgs + [{"role": "user", "content": f"q{t}"}]
        prompt_ids = tok.encode(apply_chat_template(msgs))
        n0 = d.steps
        gen_ids = tok.encode(gen)
        list(d.generate(prompt_ids, max_new_tokens=len(gen_ids)))
        fed.append(d.steps - n0 - len(gen_ids))
        d._kv_ids = d._kv_ids[:len(prompt_ids)] + gen_ids     # KV holds what was GENERATED
        msgs = msgs + [{"role": "assistant", "content": gen}]
    # Turn 1 pays full price; later turns must NOT -- and must not grow with history.
    assert fed[1] < fed[0] / 2, fed
    assert fed[2] <= fed[1] + 8, f"re-feed grew with history: {fed}"


def test_prefix_cache_handles_divergence():
    """A prompt that FORKS from the cached prefix must not reuse the stale tail."""
    d = _KVDevice()
    a = list(d.generate([1, 2, 3, 4, 5], max_new_tokens=3))
    forked = list(d.generate([1, 2, 9, 9], max_new_tokens=3))     # forks at index 2
    ref = _KVDevice()
    ref.supports_prefix_cache = False
    assert list(ref.generate([1, 2, 9, 9], max_new_tokens=3)) == forked, \
        "divergent prompt reused a stale row"
    assert a  # turn 1 produced output


def test_prefix_cache_invalidated_by_reset_session():
    """reset_session() drops device KV; the host must not then claim it is resident."""
    d = _KVDevice()
    list(d.generate([1, 2, 3, 4], max_new_tokens=2))
    d.reset_session()
    assert d._kv_ids == [], "reset_session() left a stale prefix claim"
    n0 = d.steps
    list(d.generate([1, 2, 3, 4], max_new_tokens=2))
    assert d.steps - n0 >= 4, "re-fed fewer tokens than the (now empty) device holds"


def test_prefix_cache_off_for_replay_backends():
    """Backends that replay canned ids have no position-addressed KV to reuse."""
    assert MockDevice().supports_prefix_cache is False
    assert _FixedDevice([1, 2]).supports_prefix_cache is False


def test_prefix_cache_always_feeds_one_token():
    """A fully-cached prompt still needs the last token fed: the device's output is the
       response to the LAST token fed, so reuse is capped at len(prompt)-1."""
    d = _KVDevice()
    list(d.generate([1, 2, 3], max_new_tokens=2))
    n0 = d.steps
    out = list(d.generate([1, 2, 3], max_new_tokens=2))           # identical prompt
    assert out, "fully-cached prompt produced nothing"
    assert d.steps > n0, "no token was fed at all"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED  (WPU host scaffold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
