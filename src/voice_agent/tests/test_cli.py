"""Tests for the CLI glue: menu input in voice mode, and what gets spoken."""

import numpy as np

from voice_agent.__main__ import deliver, listen_once, menu_follows
from voice_agent.agent import Reply


class FakeRecorder:
    def __init__(self, seconds, sample_rate=16000):
        self.pcm = np.zeros(int(seconds * sample_rate), dtype=np.int16)

    def record(self):
        return self.pcm


class FakeAgent:
    def __init__(self, transcript="重试"):
        self.transcript = transcript
        self.wavs = []

    def transcribe(self, wav_bytes, sample_rate):
        self.wavs.append(wav_bytes)
        return self.transcript


class FakeSpeech:
    def __init__(self):
        self.spoken = []

    def say(self, text):
        self.spoken.append(text)


def test_listen_once_records_and_transcribes(config, monkeypatch):
    """Regression: this path once raised NameError instead of returning the answer."""
    monkeypatch.setattr("builtins.input", lambda *a: "")
    agent = FakeAgent("重试")

    answer = listen_once(agent, FakeRecorder(2.0), config.audio)

    assert answer == "重试"
    assert agent.wavs[0].startswith(b"RIFF")  # a real WAV was built


def test_listen_once_ignores_too_short_recording(config, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    agent = FakeAgent("重试")

    assert listen_once(agent, FakeRecorder(0.05), config.audio) == ""
    assert agent.wavs == []  # nothing sent to speech recognition


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
