"""Status reporting: phases, the speech wrapper, backends and the loops. No robot or ROS."""

import logging
import threading

import numpy as np
import pytest

from voice_agent.__main__ import text_loop, voice_loop
from voice_agent.config import ConfigError, StatusConfig, load_config
from voice_agent.delivery import Outcome
from voice_agent.status import (
    IDLE, LISTENING, SPEAKING, THINKING, Ros2Backend, StatusBackend, StatusBackendUnavailable,
    StatusReporter, StatusSpeech, StubBackend, build_reporter, make_backend,
)

from .conftest import SHIPPED_CONFIG


class RecordingBackend(StatusBackend):
    name = "recording"

    def __init__(self, fail=False):
        self.published = []
        self.closed = False
        self.fail = fail

    def publish(self, phase):
        if self.fail:
            raise RuntimeError("display offline")
        self.published.append(phase)

    def close(self):
        self.closed = True


class FakeSpeech:
    name = "fake"

    def __init__(self):
        self.interrupt = None

    def say(self, text, on_playback_start=None, interrupt=None):
        self.interrupt = interrupt
        if on_playback_start is not None:
            on_playback_start()
        return "result"

    def prefetch(self, text):
        self.prefetched = text


def make_reporter(**kwargs):
    backend = RecordingBackend(**kwargs)
    return StatusReporter(backend), backend


def test_each_change_is_published_once():
    status, backend = make_reporter()
    for phase in (IDLE, LISTENING, LISTENING, THINKING, SPEAKING, SPEAKING, IDLE):
        status.set(phase)
    assert backend.published == [IDLE, LISTENING, THINKING, SPEAKING, IDLE]
    assert status.phase == IDLE


def test_unknown_phase_is_a_programming_error():
    status, _ = make_reporter()
    with pytest.raises(ValueError, match="unknown phase"):
        status.set("dancing")


def test_backend_failure_never_raises(caplog):
    status, _ = make_reporter(fail=True)
    with caplog.at_level(logging.WARNING):
        status.set(SPEAKING)
    assert "could not be reported" in caplog.text


def test_close_reports_idle_then_closes_the_backend():
    status, backend = make_reporter()
    status.set(SPEAKING)
    status.close()
    assert backend.published == [SPEAKING, IDLE]
    assert backend.closed


def test_speaking_is_reported_when_playback_starts_before_the_gesture():
    status, backend = make_reporter()
    seen = []
    speech = StatusSpeech(FakeSpeech(), status)
    interrupt = threading.Event()

    result = speech.say("你好", on_playback_start=lambda: seen.append(status.phase),
                        interrupt=interrupt)

    assert result == "result"
    assert backend.published == [SPEAKING]
    assert seen == [SPEAKING]
    assert speech.interrupt is interrupt


def test_speech_wrapper_passes_other_calls_through():
    status, _ = make_reporter()
    inner = FakeSpeech()
    speech = StatusSpeech(inner, status)
    speech.prefetch("下一句")
    assert inner.prefetched == "下一句"
    assert speech.name == "fake"


def test_stub_is_the_default_and_needs_no_ros():
    assert make_backend(StatusConfig()).name == "stub"


def test_disabled_status_uses_the_stub():
    reporter = build_reporter(StatusConfig(enabled=False, backend="ros2"))
    reporter.set(LISTENING)
    assert reporter.phase == LISTENING


def test_ros2_backend_reports_missing_ros():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        with pytest.raises(StatusBackendUnavailable, match="ROS 2 is not available"):
            Ros2Backend("/voice_agent/state")
    else:
        pytest.skip("ROS 2 is installed here, so the failure path cannot run")


def test_falls_back_to_stub_when_ros_is_unavailable(caplog):
    try:
        import rclpy  # noqa: F401
    except ImportError:
        with caplog.at_level(logging.WARNING):
            backend = make_backend(StatusConfig(backend="ros2"))
        assert isinstance(backend, StubBackend)
        assert "unavailable" in caplog.text
    else:
        pytest.skip("ROS 2 is installed here, so the fallback cannot run")


def write_config(tmp_path, status_yaml):
    text = SHIPPED_CONFIG.read_text(encoding="utf-8")
    start = text.index("status:\n")
    end = text.index("\n\n", start)
    path = tmp_path / "config.yaml"
    path.write_text(text[:start] + status_yaml + text[end:], encoding="utf-8")
    for name in ("gestures.yaml", "narrator.yaml"):
        (tmp_path / name).write_text(
            (SHIPPED_CONFIG.parent / name).read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_shipped_config_reports_through_the_stub(config):
    assert config.status == StatusConfig(enabled=True, backend="stub",
                                         ros2_topic="/voice_agent/state")


def test_unknown_backend_is_rejected(tmp_path):
    path = write_config(tmp_path, "status:\n  backend: led")
    with pytest.raises(ConfigError, match="status.backend"):
        load_config(path)


def test_topic_must_be_a_ros2_name(tmp_path):
    path = write_config(tmp_path, "status:\n  backend: ros2\n  ros2_topic: voice state")
    with pytest.raises(ConfigError, match="status.ros2_topic"):
        load_config(path)


def test_disabled_status_skips_validation(tmp_path):
    path = write_config(tmp_path, "status:\n  enabled: false\n  backend: led")
    assert load_config(path).status.enabled is False


class FakeDelivery:
    """Speaks every reply through the (wrapped) speech output, like ReplyDelivery."""

    def __init__(self, speech, status):
        self.speech = speech
        self.status = status
        self.phase_at_respond = []

    def wants_continue(self, text):
        return False

    def respond_and_speak(self, respond):
        self.phase_at_respond.append(self.status.phase)
        self.speech.say("好的。")
        return Outcome(reply=None)


def run_text_loop(monkeypatch, lines, config):
    status, backend = make_reporter()
    speech = StatusSpeech(FakeSpeech(), status)
    delivery = FakeDelivery(speech, status)
    feed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *a: next(feed))
    text_loop(None, speech, config, delivery=delivery, status=status)
    return backend.published, delivery


def test_text_turn_goes_idle_thinking_speaking_idle(monkeypatch, config):
    published, delivery = run_text_loop(monkeypatch, ["你好", "q"], config)
    assert published == [IDLE, THINKING, SPEAKING, IDLE]
    assert delivery.phase_at_respond == [THINKING]


def test_empty_line_stays_idle(monkeypatch, config):
    published, delivery = run_text_loop(monkeypatch, ["", "q"], config)
    assert published == [IDLE]
    assert delivery.phase_at_respond == []


class OneShotRecorder:
    """Returns one utterance, then stops the loop the way Ctrl+C would."""

    def __init__(self, seconds, sample_rate=16000):
        self.takes = [np.ones(int(seconds * sample_rate), dtype=np.int16)]

    def record(self):
        if not self.takes:
            raise KeyboardInterrupt
        return self.takes.pop(0)


def test_always_on_turn_goes_listening_thinking_speaking_listening(monkeypatch, config):
    config.listening.mode = "always_on"
    status, backend = make_reporter()
    speech = StatusSpeech(FakeSpeech(), status)
    delivery = FakeDelivery(speech, status)
    monkeypatch.setattr("voice_agent.audio.make_recorder",
                        lambda audio, mode: OneShotRecorder(1.0))
    monkeypatch.setattr("voice_agent.__main__.hear", lambda agent, wav, rate: ("你好", None))
    monkeypatch.setattr("voice_agent.__main__.lazy_local_recognizer", lambda config: None)

    with pytest.raises(KeyboardInterrupt):
        voice_loop(None, speech, config, delivery=delivery, status=status)

    assert backend.published == [LISTENING, THINKING, SPEAKING, LISTENING]
    assert delivery.phase_at_respond == [THINKING]


def test_push_to_talk_is_idle_until_enter(monkeypatch, config):
    config.listening.mode = "push_to_talk"
    status, backend = make_reporter()
    speech = StatusSpeech(FakeSpeech(), status)
    delivery = FakeDelivery(speech, status)
    feed = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(feed))
    monkeypatch.setattr("voice_agent.audio.make_recorder",
                        lambda audio, mode: OneShotRecorder(1.0))
    monkeypatch.setattr("voice_agent.__main__.hear", lambda agent, wav, rate: ("你好", None))
    monkeypatch.setattr("voice_agent.__main__.lazy_local_recognizer", lambda config: None)

    voice_loop(None, speech, config, delivery=delivery, status=status)

    assert backend.published == [IDLE, LISTENING, THINKING, SPEAKING, IDLE]
