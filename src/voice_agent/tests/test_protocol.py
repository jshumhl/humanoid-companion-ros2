import pytest

from voice_agent.protocol import (
    ReplyFormatError, Say, ToolCall, clean_spoken, parse_model_output, split_sentences,
)


@pytest.mark.parametrize("raw, expected", [
    ('{"say": "我是巴克机器人。"}', Say("我是巴克机器人。")),
    ('{"tool": "look_around", "args": {}}', ToolCall("look_around", {})),
    ('{"tool": "look_around"}', ToolCall("look_around", {})),
    ('{"tool": "look_around", "args": null}', ToolCall("look_around", {})),
    ('```json\n{"say": "你好。"}\n```', Say("你好。")),
    ('好的：{"say": "你好。"} 以上', Say("你好。")),
    ('  {"say": "  你好。 "}  ', Say("你好。")),
    ("我是巴克机器人。", Say("我是巴克机器人。")),  # plain text, format ignored
    # A gesture may ride along with either kind of reply
    ('{"say": "你好。", "gesture": "hello"}', Say("你好。", "hello")),
    ('{"say": "你好。", "gesture": " hello "}', Say("你好。", "hello")),
    ('{"say": "你好。", "gesture": null}', Say("你好。", "")),
    ('{"say": "你好。", "gesture": ""}', Say("你好。", "")),
    ('{"tool": "look_around", "gesture": "point"}', ToolCall("look_around", {}, "point")),
    # Unknown names are the catalogue's business, not the parser's
    ('{"say": "你好。", "gesture": "backflip"}', Say("你好。", "backflip")),
])
def test_parse_valid(raw, expected):
    assert parse_model_output(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    None,
    '{"say": ',
    '["say", "hi"]',
    '{"say": ""}',
    '{"say": 42}',
    '{"tool": ""}',
    '{"tool": "look_around", "args": []}',
    '{"say": "hi", "tool": "look_around"}',
    '{"answer": "hi"}',
    '{"say": "你好。", "gesture": ["hello"]}',
    '{"say": "你好。", "gesture": 3}',
])
def test_parse_invalid(raw):
    with pytest.raises(ReplyFormatError):
        parse_model_output(raw)


def test_clean_spoken_keeps_the_whole_reply():
    """Trimming is the delivery layer's job now: it speaks N sentences, then asks."""
    text = "第一句。第二句！第三句？第四句。"
    assert clean_spoken(text) == text


def test_clean_spoken_strips_markdown_and_emoji():
    assert clean_spoken("**我是** 巴克机器人 😊。") == "我是 巴克机器人。"


def test_clean_spoken_empty():
    assert clean_spoken("**  **") == ""


@pytest.mark.parametrize("text, expected", [
    ("第一句。第二句！第三句？", ["第一句。", "第二句！", "第三句？"]),
    ("你好。很高兴见到你", ["你好。", "很高兴见到你"]),   # unterminated tail is kept
    ("只有一句话。", ["只有一句话。"]),
    ("", []),
    ("   ", []),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected
