import pytest

from voice_agent.protocol import ReplyFormatError, Say, ToolCall, clean_spoken, parse_model_output


@pytest.mark.parametrize("raw, expected", [
    ('{"say": "我是巴克机器人。"}', Say("我是巴克机器人。")),
    ('{"tool": "look_around", "args": {}}', ToolCall("look_around", {})),
    ('{"tool": "look_around"}', ToolCall("look_around", {})),
    ('{"tool": "look_around", "args": null}', ToolCall("look_around", {})),
    ('```json\n{"say": "你好。"}\n```', Say("你好。")),
    ('好的：{"say": "你好。"} 以上', Say("你好。")),
    ('  {"say": "  你好。 "}  ', Say("你好。")),
    ("我是巴克机器人。", Say("我是巴克机器人。")),  # plain text, format ignored
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
])
def test_parse_invalid(raw):
    with pytest.raises(ReplyFormatError):
        parse_model_output(raw)


def test_clean_spoken_caps_sentences():
    text = "第一句。第二句！第三句？第四句。"
    assert clean_spoken(text, 3) == "第一句。第二句！第三句？"


def test_clean_spoken_strips_markdown_and_emoji():
    assert clean_spoken("**我是** 巴克机器人 😊。", 3) == "我是 巴克机器人。"


def test_clean_spoken_keeps_unterminated_last_sentence():
    assert clean_spoken("你好。很高兴见到你", 3) == "你好。很高兴见到你"


def test_clean_spoken_empty():
    assert clean_spoken("**  **", 3) == ""
