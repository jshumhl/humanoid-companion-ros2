from datetime import datetime, timezone

import pytest

from voice_agent.clock import UNSURE, chinese_number, clock_tools, spoken_date, spoken_time
from voice_agent.config import ClockConfig, ConfigError, load_config
from voice_agent.protocol import ToolCall
from voice_agent.tools import ToolRegistry, default_tools

from .test_config import write_config


def at(hour, minute):
    return datetime(2026, 10, 5, hour, minute)


@pytest.mark.parametrize("n, words", [
    (0, "零"), (2, "二"), (10, "十"), (11, "十一"), (17, "十七"), (20, "二十"), (59, "五十九"),
])
def test_chinese_number(n, words):
    assert chinese_number(n) == words


@pytest.mark.parametrize("hour, minute, words", [
    (0, 0, "凌晨零点整"),
    (0, 10, "凌晨零点十分"),
    (4, 59, "凌晨四点五十九分"),
    (5, 0, "早上五点整"),
    (7, 30, "早上七点半"),
    (8, 5, "上午八点零五分"),
    (11, 45, "上午十一点四十五分"),
    (12, 0, "中午十二点整"),
    (12, 30, "中午十二点半"),
    (13, 0, "下午一点整"),
    (14, 5, "下午两点零五分"),
    (15, 17, "下午三点十七分"),
    (17, 59, "下午五点五十九分"),
    (18, 0, "晚上六点整"),
    (22, 2, "晚上十点零二分"),
    (23, 59, "晚上十一点五十九分"),
])
def test_spoken_time(hour, minute, words):
    assert spoken_time(at(hour, minute)) == words


def test_spoken_time_has_no_digits_for_tts():
    for hour in range(24):
        for minute in range(60):
            words = spoken_time(at(hour, minute))
            assert not any(ch.isascii() for ch in words), words


@pytest.mark.parametrize("day, words", [
    (datetime(2026, 10, 5), "2026年10月5日，星期一"),
    (datetime(2026, 10, 10), "2026年10月10日，星期六"),
    (datetime(2026, 10, 11), "2026年10月11日，星期天"),
    (datetime(2027, 1, 1), "2027年1月1日，星期五"),
])
def test_spoken_date(day, words):
    assert spoken_date(day) == words


def registry(now, timezone=""):
    return ToolRegistry(clock_tools(ClockConfig(timezone=timezone), now=lambda tz: now))


def test_current_time_says_the_time():
    assert registry(at(15, 17)).dispatch(ToolCall("current_time")) == "现在是下午三点十七分。"


def test_today_says_the_date_and_weekday():
    assert registry(at(15, 17)).dispatch(ToolCall("today")) == "今天是2026年10月5日，星期一。"


@pytest.mark.parametrize("tool", ["current_time", "today"])
def test_unset_clock_says_it_is_unsure(tool):
    never_set = datetime(1970, 1, 1, 0, 3)
    assert registry(never_set).dispatch(ToolCall(tool)) == UNSURE


def test_empty_timezone_reads_local_time():
    seen = []
    tools = ToolRegistry(clock_tools(ClockConfig(), now=lambda tz: seen.append(tz) or at(9, 0)))
    tools.dispatch(ToolCall("current_time"))
    assert seen == [None]


def test_configured_timezone_is_used():
    utc_moment = datetime(2026, 10, 5, 23, 30, tzinfo=timezone.utc)
    tools = ToolRegistry(clock_tools(ClockConfig(timezone="Asia/Shanghai"),
                                     now=lambda tz: utc_moment.astimezone(tz)))
    # 23:30 UTC is 07:30 the next day in Shanghai.
    assert tools.dispatch(ToolCall("current_time")) == "现在是早上七点半。"
    assert tools.dispatch(ToolCall("today")) == "今天是2026年10月6日，星期二。"


def test_real_clock_answers():
    tools = ToolRegistry(clock_tools(ClockConfig()))
    assert tools.dispatch(ToolCall("current_time")).startswith("现在是")
    assert tools.dispatch(ToolCall("today")).startswith("今天是")


def test_default_tools_include_the_clock(config):
    tools = default_tools(config.narrator, config.clock)
    assert tools.names() == ["look_around", "current_time", "today"]
    assert "- current_time()：" in tools.catalog()
    assert "- today()：" in tools.catalog()


def test_shipped_config_uses_local_time_and_mentions_the_tools(config):
    assert config.clock.timezone == ""
    assert "current_time" in config.system_prompt
    assert "today" in config.system_prompt


def test_timezone_setting_loads(tmp_path):
    config = load_config(write_config(tmp_path, clock__timezone="Asia/Shanghai"))
    assert config.clock.timezone == "Asia/Shanghai"


def test_unknown_timezone_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="clock.timezone"):
        load_config(write_config(tmp_path, clock__timezone="Mars/Olympus"))
