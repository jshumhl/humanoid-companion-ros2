"""The JSON reply format the model must produce, and spoken-text cleanup.

The model answers with exactly one JSON object, in one of two shapes:

    {"say": "要说的话"}
    {"tool": "look_around", "args": {}}

This is our own schema, parsed here, so it works with any chat model
regardless of whether its API supports native tool calling.
"""

import json
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Say:
    text: str
    gesture: str = ""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    gesture: str = ""


class ReplyFormatError(ValueError):
    """The model output does not match the schema."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_model_output(raw):
    """Parse raw model text into Say or ToolCall. Raises ReplyFormatError."""
    text = _FENCE.sub("", (raw or "").strip()).strip()
    if not text:
        raise ReplyFormatError("empty model output")

    start = text.find("{")
    if start == -1 and not text.startswith("["):
        # The model ignored the format and answered in plain text. Speaking it
        # is better than failing the turn.
        return Say(text)

    try:
        obj, _ = json.JSONDecoder().raw_decode(text if start == -1 else text[start:])
    except json.JSONDecodeError as e:
        raise ReplyFormatError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ReplyFormatError("top-level JSON value is not an object")

    has_say, has_tool = "say" in obj, "tool" in obj
    if has_say and has_tool:
        raise ReplyFormatError("object has both 'say' and 'tool'")

    gesture = _parse_gesture(obj)

    if has_tool:
        name = obj["tool"]
        args = obj.get("args", {})
        if not isinstance(name, str) or not name.strip():
            raise ReplyFormatError("'tool' must be a non-empty string")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ReplyFormatError("'args' must be an object")
        return ToolCall(name.strip(), args, gesture)

    if has_say:
        say = obj["say"]
        if not isinstance(say, str) or not say.strip():
            raise ReplyFormatError("'say' must be a non-empty string")
        return Say(say.strip(), gesture)

    raise ReplyFormatError("object has neither 'say' nor 'tool'")


def _parse_gesture(obj):
    """At most one gesture name. Missing or null means no gesture.

    Whether the name exists is decided against the catalogue later, so a made-up
    name costs the reply nothing.
    """
    gesture = obj.get("gesture")
    if gesture is None:
        return ""
    if not isinstance(gesture, str):
        raise ReplyFormatError("'gesture' must be a string")
    return gesture.strip()


_MARKDOWN = re.compile(r"[*#`_>|~]+")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]+"
)
_SENTENCE = re.compile(r"[^。！？!?；;]+[。！？!?；;]*")


def clean_spoken(text):
    """Make model text suitable for TTS: strip markdown, emoji and stray spacing."""
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([。！？!?，,、；;：:])", r"\1", text)


def split_sentences(text):
    """Split spoken text into sentences, so it can be delivered a piece at a time."""
    return [s.strip() for s in _SENTENCE.findall(text or "") if s.strip()]
