"""A record of which gesture the model chose for each turn, for later tuning.

One JSON object per line, in a rotating file, so a long-running robot cannot
fill its disk:

    {"time": "2026-09-21T14:03:11", "user": "你好！", "gesture": "hello", ...}
    {"time": "2026-09-21T14:03:29", "user": "现在几点？", "gesture": null, ...}

Turns where no gesture was chosen are written too, with `gesture: null`: the
proportion of replies that gesture is the thing worth tuning, and it cannot be
seen from the gestures alone.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class GestureLog:
    """Appends one line per turn. Never raises: logging must not break a conversation."""

    def __init__(self, path, max_bytes=1_000_000, backups=3):
        self._logger = logging.getLogger("voice_agent.gesture_choices")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # this file only, not the terminal
        self._logger.handlers.clear()

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self.path = path

    def record(self, user_text, gesture, reply="", tool="", fallback=""):
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": user_text,
            "gesture": gesture or None,
            "reply": reply,
        }
        if tool:
            entry["tool"] = tool
        if fallback:
            entry["fallback"] = fallback
        self._logger.info(json.dumps(entry, ensure_ascii=False))

    def close(self):
        for handler in self._logger.handlers:
            handler.close()
        self._logger.handlers.clear()


def open_gesture_log(gestures_config):
    """Open the log, or return None with a warning if the path is unusable."""
    if not gestures_config.log_path:
        return None
    try:
        return GestureLog(gestures_config.log_path,
                          gestures_config.log_max_bytes,
                          gestures_config.log_backups)
    except OSError as e:
        log.warning("Cannot open the gesture log at %s: %s", gestures_config.log_path, e)
        return None
