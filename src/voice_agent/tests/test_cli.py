"""Tests for the CLI glue: menu input in voice mode, and what gets spoken."""

import numpy as np

from voice_agent.__main__ import deliver, listen_once, menu_follows
from voice_agent.agent import Reply


class FakeRecorder:
    def __init__(self, seconds, sample_rate=16000):
        self.pcm = np.zeros(int(seconds * sample_rate), dtype=np.int16)

    def record(self):
        return self.pcm


class FakeLocalRecognizer:
    def __init__(self, text="重 试"):
        self.text = text
        self.heard = []

    def recognize(self, pcm):
        self.heard.append(len(pcm))
        return self.text


class FakeSpeech:
    def __init__(self):
        self.spoken = []

    def say(self, text):
        self.spoken.append(text)


def test_listen_once_uses_the_local_recognizer(config, monkeypatch):
    """Regression: this path once raised NameError instead of returning the answer."""
    monkeypatch.setattr("builtins.input", lambda *a: "")
    local = FakeLocalRecognizer("重 试")

    answer = listen_once(FakeRecorder(2.0), config.audio, local)

    assert answer == "重 试"
    assert local.heard == [2 * 16000]


def test_listen_once_ignores_too_short_recording(config, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    local = FakeLocalRecognizer()

    assert listen_once(FakeRecorder(0.05), config.audio, local) == ""
    assert local.heard == []


def test_typed_answer_is_used_without_recording(config, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: " 2 ")
    local = FakeLocalRecognizer()

    assert listen_once(FakeRecorder(2.0), config.audio, local) == "2"
    assert local.heard == []  # nothing was recorded


def test_typing_still_works_without_a_local_recognizer(config, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "3")
    assert listen_once(FakeRecorder(2.0), config.audio, None) == "3"


def test_without_local_recognizer_nothing_is_recorded(config, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert listen_once(FakeRecorder(2.0), config.audio, None) == ""
    assert "type 1, 2 or 3" in capsys.readouterr().out


def test_offline_reply_is_not_spoken_when_menu_follows(config, capsys):
    speech = FakeSpeech()
    reply = Reply(config.fallback_phrases.offline, heard="你是谁？", fallback="offline")

    assert menu_follows(reply, config) is True
    deliver(reply, speech, say_reply=False)

    out = capsys.readouterr().out
    assert "你：你是谁？" in out
    assert config.fallback_phrases.offline not in out
    assert speech.spoken == []


def test_offline_reply_is_spoken_when_menu_disabled(config):
    config.offline_menu.enabled = False
    reply = Reply(config.fallback_phrases.offline, fallback="offline")

    assert menu_follows(reply, config) is False
    speech = FakeSpeech()
    deliver(reply, speech, say_reply=True)
    assert speech.spoken == [config.fallback_phrases.offline]


def test_other_fallbacks_are_still_spoken(config):
    reply = Reply(config.fallback_phrases.not_heard, fallback="not_heard")
    assert menu_follows(reply, config) is False
