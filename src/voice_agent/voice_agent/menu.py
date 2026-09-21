"""Spoken menu offered when the provider is unreachable.

    "我现在连不上网络。你可以说 一 检查设置，二 重试，三 退出。"

The person answers with the keyword (设置 / 重试 / 退出) or the position
(一 / 1 / 第一). Every phrase is cached as audio at startup, so the menu can
always be spoken with the network down.

Hearing the answer still needs speech recognition, which is the very service
that is unreachable. Each attempt therefore retries it: short outages recover
by the time the menu has finished speaking. If recognition still fails, the
attempt counts as not understood, and after `max_attempts` the menu speaks its
closing phrase and gives up. In --text mode the answer is typed, so the menu
works with no network at all.
"""

import logging
import re

log = logging.getLogger(__name__)

IDLE = "idle"  # returned when nothing was understood, or the menu was skipped

# Ways of saying "the first/second/third option".
POSITION_WORDS = (
    ("一", "1", "１", "第一", "一号", "第1"),
    ("二", "两", "2", "２", "第二", "二号", "第2"),
    ("三", "3", "３", "第三", "三号", "第3"),
)

_IGNORED = re.compile(r"[\s，。！？、,.!?；;：:～~]+")


def normalize(text):
    return _IGNORED.sub("", (text or "").strip().lower())


def match_choice(text, options):
    """Index of the chosen option, or None if the answer was not understood.

    Accepts a keyword (设置), a position (一 / 1 / 第二), typed or spoken. Local
    recognition marks speech it could not place as `[unk]`; a keyword still
    counts next to one, but a bare position does not, because a single syllable
    next to unrecognized speech is too easily a mishearing.
    """
    raw = text or ""
    had_unknown = "[unk]" in raw
    cleaned = normalize(raw.replace("[unk]", " "))
    if not cleaned:
        return None

    for i, option in enumerate(options):
        keyword = normalize(option.keyword)
        if keyword and keyword in cleaned:
            return i
    if had_unknown:
        return None
    for i, _ in enumerate(options):
        if i < len(POSITION_WORDS) and any(word in cleaned for word in POSITION_WORDS[i]):
            return i
    return None


def menu_phrases(menu):
    """Every phrase the menu can speak, for pre-caching at startup."""
    if not menu.enabled:
        return []
    return [menu.prompt, menu.closing, *(option.reply for option in menu.options)]


class OfflineMenu:
    def __init__(self, menu_config, say, listen):
        """
        say(text):  speak (and print) one phrase
        listen():   return what the person said or typed, "" if nothing was understood
        """
        self._menu = menu_config
        self._say = say
        self._listen = listen

    def run(self):
        """Offer the menu. Returns the chosen action, or IDLE."""
        menu = self._menu
        self._say(menu.prompt)

        for attempt in range(1, menu.max_attempts + 1):
            try:
                answer = self._listen()
            except (KeyboardInterrupt, EOFError):
                raise
            except (NameError, AttributeError, ImportError, TypeError):
                # Bugs in our own code, not a failure to hear. Let them surface
                # instead of being reported as "not understood".
                raise
            except Exception as e:
                # Recognition is down, the microphone failed, and so on.
                log.warning("Menu input failed: %s", e)
                answer = ""

            choice = match_choice(answer, menu.options)
            if choice is not None:
                option = menu.options[choice]
                log.debug("Menu answer %r selected %s", answer, option.action)
                self._say(option.reply)
                return option.action
            log.info("Menu attempt %d of %d not understood: %r",
                     attempt, menu.max_attempts, answer)

        self._say(menu.closing)
        return IDLE
