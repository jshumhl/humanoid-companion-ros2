"""Grammar construction and offline matching. No model, microphone or network."""

import pytest

from voice_agent.config import MenuOption
from voice_agent.local_asr import (
    UNKNOWN, LocalRecognizerUnavailable, VoskMenuRecognizer, build_grammar, load_menu_recognizer,
)
from voice_agent.menu import match_choice

OPTIONS = [
    MenuOption("settings", "设置", "好"),
    MenuOption("retry", "重试", "好"),
    MenuOption("quit", "退出", "好"),
]


def test_grammar_is_built_from_the_configured_keywords():
    grammar = build_grammar(OPTIONS)
    assert "设置" in grammar and "重试" in grammar and "退出" in grammar
    assert grammar[-1] == UNKNOWN


def test_grammar_also_offers_keywords_split_into_characters():
    """The small Chinese model lacks some words, e.g. 重试, but knows the characters."""
    assert "重 试" in build_grammar(OPTIONS)


def test_grammar_includes_spoken_positions():
    grammar = build_grammar(OPTIONS)
    assert {"一", "二", "三"} <= set(grammar)
    assert "1" not in grammar  # digits are typed, not spoken


def test_grammar_follows_custom_options():
    grammar = build_grammar([MenuOption("quit", "再见", "好")])
    assert "再见" in grammar and "再 见" in grammar
    assert "二" not in grammar  # only one option, so only the first position
    assert grammar[-1] == UNKNOWN


def test_grammar_has_no_duplicates():
    grammar = build_grammar([MenuOption("retry", "一", "好"), MenuOption("quit", "退出", "好")])
    assert len(grammar) == len(set(grammar))


@pytest.mark.parametrize("heard, expected", [
    # Vosk returns space-separated words
    ("重 试", 1),
    ("设置", 0),
    ("退出", 2),
    ("二", 1),
    # Unrelated speech: "[unk] 一" must not select option 1
    (f"{UNKNOWN} 一", None),
    (UNKNOWN, None),
    ("", None),
    # A keyword next to unrecognized speech is still a clear choice
    (f"{UNKNOWN} 重 试 {UNKNOWN}", 1),
])
def test_matching_recognizer_output(heard, expected):
    assert match_choice(heard, OPTIONS) == expected


def test_missing_model_reports_where_to_get_it(tmp_path):
    with pytest.raises(LocalRecognizerUnavailable, match="SETUP.md"):
        VoskMenuRecognizer(tmp_path / "no-such-model", OPTIONS)


def test_loader_returns_none_instead_of_raising(tmp_path):
    class LocalAsr:
        enabled = True
        model_path = str(tmp_path / "missing")

    assert load_menu_recognizer(LocalAsr(), OPTIONS) is None


def test_loader_skips_when_disabled():
    class LocalAsr:
        enabled = False
        model_path = "/should/not/be/read"

    assert load_menu_recognizer(LocalAsr(), OPTIONS) is None
