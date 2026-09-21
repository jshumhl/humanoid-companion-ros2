"""The per-turn gesture log. No network, microphone or robot."""

import json

from voice_agent.config import GesturesConfig
from voice_agent.gesture_log import GestureLog, open_gesture_log


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_the_utterance_and_the_chosen_gesture(tmp_path):
    path = tmp_path / "gestures.log"
    log = GestureLog(path)
    log.record("你好！", "hello", "你好，我是巴克机器人。")
    log.close()

    entry = read_lines(path)[0]
    assert entry["user"] == "你好！"
    assert entry["gesture"] == "hello"
    assert entry["reply"] == "你好，我是巴克机器人。"
    assert entry["time"]


def test_turns_without_a_gesture_are_recorded_as_null(tmp_path):
    """The share of replies that gesture is what needs tuning, so nulls matter."""
    path = tmp_path / "gestures.log"
    log = GestureLog(path)
    log.record("现在几点？", "", "现在是三点。")
    log.record("你好！", "hello", "你好。")
    log.close()

    assert [e["gesture"] for e in read_lines(path)] == [None, "hello"]


def test_tool_and_fallback_turns_are_marked(tmp_path):
    path = tmp_path / "gestures.log"
    log = GestureLog(path)
    log.record("你看见什么？", "point", "我看到了两个人。", tool="look_around")
    log.record("你是谁？", "", "我现在连不上网络。", fallback="offline")
    log.close()

    first, second = read_lines(path)
    assert first["tool"] == "look_around"
    assert second["fallback"] == "offline"
    assert "tool" not in second


def test_file_rotates_instead_of_growing(tmp_path):
    path = tmp_path / "gestures.log"
    log = GestureLog(path, max_bytes=200, backups=2)
    for i in range(50):
        log.record(f"第{i}句话，这是一句比较长的话。", "nod", "好的。")
    log.close()

    assert path.exists()
    assert (tmp_path / "gestures.log.1").exists()
    assert not (tmp_path / "gestures.log.3").exists()  # older ones are discarded
    assert path.stat().st_size < 1000


def test_directory_is_created(tmp_path):
    path = tmp_path / "nested" / "dir" / "gestures.log"
    GestureLog(path).close()
    assert path.exists()


def test_open_gesture_log_disabled_by_empty_path():
    assert open_gesture_log(GesturesConfig(log_path="")) is None


def test_open_gesture_log_returns_none_when_path_is_unusable(tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    assert open_gesture_log(GesturesConfig(log_path=str(blocker / "gestures.log"))) is None
