"""wpu_chat_template.py -- the GLM-5.2 chat template for the WPU host server.

Formats OpenAI-style chat `messages` into the single prompt string the GLM-5.2
tokenizer expects, using GLM's special tokens: [gMASK], <sop>, <|system|>, <|user|>,
<|assistant|>, <|observation|>, <think>. Each of these is a SINGLE token id in the
GLM BPE vocab (verified against host/tokenizer.json), so encoding the string this
function returns composes exactly the right control tokens.

FIDELITY (honest -- I confirmed the EXACT template, I am not guessing):
This is a faithful Python port of the COMMON TEXT PATH of the official template at
  huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/chat_template.jinja
(downloaded + read verbatim, 2026-07). For plain system/user/assistant text turns it
reproduces the default rendering EXACTLY, including:
  * the leading `[gMASK]<sop>`;
  * GLM-5.2's auto-injected `<|system|>Reasoning Effort: {High|Max}` turn -- GLM-5.2
    is a thinking model and the template emits this whenever thinking is enabled and a
    reasoning effort is set (default `max`);
  * per-message `<|{role}|>{content}` with NO separating newline -- note GLM-5.2 drops
    the `\n` the older GLM-4 template placed after each role tag;
  * assistant history turns rendered `<|assistant|><think></think>{content}` (the
    template clears historical reasoning by default);
  * the trailing generation prompt `<|assistant|><think>` (thinking on) or
    `<|assistant|><think></think>` (thinking off).

NOT ported (kept as a standalone, easily-extended function): tool / function-calling
(<tool_call>/<|observation|> tool-def echo) and multi-modal image/video/audio content
-- those fall back to plain visible text (with the template's media <reminder> note).
The scaffold's MockDevice replies with canned tokens regardless of the prompt, so full
tool/vision rendering isn't exercised yet; wire it in when a real backend lands.
"""

from __future__ import annotations

_GMASK = "[gMASK]"
_SOP = "<sop>"

# media content-part types the template refuses (it has no multi-modal input ability).
_MEDIA_TYPES = ("image", "image_url", "video", "video_url",
                "audio", "audio_url", "input_audio")


def _visible_text(content) -> str:
    """Extract visible text from an OpenAI `content` field -- a plain string or a list
       of content parts (`{"type": "text", "text": ...}`) -- mirroring the template's
       visible_text() macro (including its media <reminder> for unsupported types)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                itype = item.get("type")
                if itype == "text" or (itype is None and "text" in item):
                    out.append(item.get("text", ""))
                elif itype in _MEDIA_TYPES:
                    media = itype.replace("_url", "").replace("input_", "")
                    out.append(f"<reminder>You are unable to process this {media} "
                               f"because you don't have multi-modal input ability. "
                               f"Try different methods.</reminder>")
        return "".join(out)
    return str(content)


def apply_chat_template(messages, *, add_generation_prompt: bool = True,
                        enable_thinking: bool = True,
                        reasoning_effort: str | None = "max") -> str:
    """Render `messages` to a GLM-5.2 prompt string (see module docstring for fidelity).

    Args:
      messages: OpenAI chat messages -- list of {"role", "content"}.
      add_generation_prompt: append the trailing `<|assistant|>` turn to prompt a reply
        (True for inference). GLM-5.2 default.
      enable_thinking: GLM-5.2 is a thinking model; when True (default) the reasoning
        effort system turn and a trailing `<think>` are emitted, matching the template.
      reasoning_effort: 'high' -> "High", anything else -> "Max" (the template only
        distinguishes 'high'); None suppresses the reasoning-effort turn.
    """
    parts = [_GMASK, _SOP]

    # Auto-injected reasoning-effort system turn (template lines 2-3).
    if enable_thinking and reasoning_effort is not None:
        effort = "High" if str(reasoning_effort).lower() == "high" else "Max"
        parts.append(f"<|system|>Reasoning Effort: {effort}")

    for m in messages:
        role = m.get("role", "user")
        content = _visible_text(m.get("content", ""))
        if role == "user":
            parts.append(f"<|user|>{content}")
        elif role == "system":
            parts.append(f"<|system|>{content}")
        elif role == "assistant":
            # Template default clears historical reasoning: strip a prior <think>...
            # </think> block, re-emit an empty one, then the (stripped) response.
            body = content.split("</think>")[-1] if "</think>" in content else content
            parts.append(f"<|assistant|><think></think>{body.strip()}")
        elif role == "tool":
            # Minimal tool-result rendering (no tool-def echo -- see module docstring).
            parts.append(f"<|observation|><tool_response>{content}</tool_response>")
        else:                                        # unknown role: best-effort
            parts.append(f"<|{role}|>{content}")

    if add_generation_prompt:
        parts.append("<|assistant|>" +
                     ("<think>" if enable_thinking else "<think></think>"))
    return "".join(parts)
