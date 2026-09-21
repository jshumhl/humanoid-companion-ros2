import pytest

from voice_agent.config import MenuOption
from voice_agent.menu import IDLE, OfflineMenu, match_choice, menu_phrases
from voice_agent.providers import ProviderUnavailable

OPTIONS = [
    MenuOption("settings", "设置", "好的，我把设置情况显示在屏幕上了。"),
    MenuOption("retry", "重试", "好的，我再试一次。"),
    MenuOption("quit", "退出", "好的，我先休息一下。"),
]


@pytest.mark.parametrize("answer, expected", [
    # Keywords, alone or in a sentence
    ("设置", 0),
    ("重试", 1),
    ("退出", 2),
    ("我想检查设置。", 0),
    ("你再重试一次吧", 1),
    # Positions, spoken or typed
    ("一", 0), ("二", 1), ("三", 2),
    ("1", 0), ("2", 1), ("3", 2),
    ("第二", 1), ("两", 1), ("第3", 2), ("三号", 2),
    # ASR transcripts arrive with punctuation and spaces
    (" 二。", 1),
    ("一，谢谢", 0),
    # Not understood
    ("", None), ("   ", None), (None, None),
    ("今天天气怎么样", None),
    ("四", None),
])
def test_match_choice(answer, expected):
    assert match_choice(answer, OPTIONS) == expected


def test_keyword_wins_over_position_of_another_option():
    # "一" is position 1, but the keyword 退出 is an explicit choice.
    assert match_choice("一定要退出", OPTIONS) == 2


def test_match_choice_with_two_options_ignores_third_position():
    assert match_choice("三", OPTIONS[:2]) is None


class Recorder:
    """Collects spoken phrases and replays scripted answers."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.said = []

    def say(self, text):
        self.said.append(text)

    def listen(self):
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def run_menu(config, answers):
    recorder = Recorder(answers)
    action = OfflineMenu(config.offline_menu, recorder.say, recorder.listen).run()
    return action, recorder


def test_menu_speaks_prompt_then_acts_on_keyword(config):
    action, recorder = run_menu(config, ["重试"])
    assert action == "retry"
    assert recorder.said == [config.offline_menu.prompt, "好的，我再试一次。"]


def test_menu_accepts_a_number(config):
    action, _ = run_menu(config, ["三"])
    assert action == "quit"


def test_second_attempt_is_accepted(config):
    action, recorder = run_menu(config, ["今天天气怎么样", "二"])
    assert action == "retry"
    assert recorder.said[-1] == "好的，我再试一次。"


def test_two_failures_close_and_return_to_idle(config):
    action, recorder = run_menu(config, ["嗯嗯", "哦"])
    assert action == IDLE
    assert recorder.said == [config.offline_menu.prompt, config.offline_menu.closing]


def test_still_offline_counts_as_a_failed_attempt(config):
    """Speech recognition needs the network; failing to hear must not hang the menu."""
    action, recorder = run_menu(config, [ProviderUnavailable("DNS failure"),
                                         ProviderUnavailable("DNS failure")])
    assert action == IDLE
    assert recorder.said[-1] == config.offline_menu.closing


def test_programming_errors_are_not_swallowed(config):
    """A bug in the listen callback must crash, not count as "not understood"."""
    with pytest.raises(NameError):
        run_menu(config, [NameError("name 'to_wav_bytes' is not defined")])


def test_attempts_are_configurable(config):
    config.offline_menu.max_attempts = 1
    action, recorder = run_menu(config, ["不知道"])
    assert action == IDLE
    assert len(recorder.said) == 2  # prompt, closing


def test_menu_phrases_are_all_pre_cacheable(config):
    phrases = menu_phrases(config.offline_menu)
    assert config.offline_menu.prompt in phrases
    assert config.offline_menu.closing in phrases
    for option in config.offline_menu.options:
        assert option.reply in phrases


def test_disabled_menu_has_no_phrases(config):
    config.offline_menu.enabled = False
    assert menu_phrases(config.offline_menu) == []
