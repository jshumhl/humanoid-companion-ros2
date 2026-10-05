"""Time and date tools, read from this machine's clock.

Answers are spoken Chinese ("下午三点十七分"), never "15:17", so every TTS
engine reads them the same way. No network call is involved.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from .tools import Tool

# A clock earlier than this was never set, as on a machine that booted without
# network time. Saying so beats announcing a date from years ago.
EARLIEST_PLAUSIBLE_YEAR = 2025
UNSURE = "我现在不太确定准确的时间。"

_DIGITS = "零一二三四五六七八九"
_WEEKDAYS = ("一", "二", "三", "四", "五", "六", "天")


def chinese_number(n):
    """0-59 as spoken Chinese: 5 -> 五, 10 -> 十, 17 -> 十七, 40 -> 四十."""
    if n < 10:
        return _DIGITS[n]
    tens, ones = divmod(n, 10)
    return ("十" if tens == 1 else _DIGITS[tens] + "十") + (_DIGITS[ones] if ones else "")


def _period(hour):
    if hour < 5:
        return "凌晨"
    if hour < 8:
        return "早上"
    if hour < 12:
        return "上午"
    if hour == 12:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def spoken_time(moment):
    """14:05 -> 下午两点零五分, 9:30 -> 上午九点半, 12:00 -> 中午十二点整."""
    hour = 12 if moment.hour == 12 else moment.hour % 12
    hour_word = "两" if hour == 2 else chinese_number(hour)
    minute = moment.minute
    if minute == 0:
        minute_word = "整"
    elif minute == 30:
        minute_word = "半"
    elif minute < 10:
        minute_word = f"零{chinese_number(minute)}分"
    else:
        minute_word = f"{chinese_number(minute)}分"
    return f"{_period(moment.hour)}{hour_word}点{minute_word}"


def spoken_date(moment):
    """2026-10-05 -> 2026年10月5日，星期一."""
    return f"{moment.year}年{moment.month}月{moment.day}日，星期{_WEEKDAYS[moment.weekday()]}"


def clock_tools(clock_config=None, now=datetime.now):
    """current_time() and today(). `now(tz)` returns the current datetime; tests replace it."""
    timezone = getattr(clock_config, "timezone", "")
    tz = ZoneInfo(timezone) if timezone else None  # None: this machine's local time

    def read_clock():
        moment = now(tz)
        return moment if moment.year >= EARLIEST_PLAUSIBLE_YEAR else None

    def current_time(args):
        moment = read_clock()
        return f"现在是{spoken_time(moment)}。" if moment else UNSURE

    def today(args):
        moment = read_clock()
        return f"今天是{spoken_date(moment)}。" if moment else UNSURE

    return [
        Tool(name="current_time", signature="current_time()",
             description="查看现在几点，返回一句报时的中文句子。", run=current_time),
        Tool(name="today", signature="today()",
             description="查看今天是几月几号、星期几，返回一句中文句子。", run=today),
    ]
