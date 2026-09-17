from pathlib import Path

import pytest

from voice_agent.config import load_config

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = PACKAGE_DIR / "config.yaml"


@pytest.fixture
def config():
    """The shipped config.yaml, loaded fresh for each test."""
    return load_config(SHIPPED_CONFIG)


class FakeProvider:
    """Scripted provider: `chat_replies` are returned in order, or raised if exceptions."""

    name = "fake"

    def __init__(self, chat_replies=(), transcript="", transcribe_error=None):
        self.chat_replies = list(chat_replies)
        self.transcript = transcript
        self.transcribe_error = transcribe_error
        self.chat_calls = []

    def chat(self, messages, timeout_sec):
        self.chat_calls.append(messages)
        reply = self.chat_replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply()
        return reply

    def transcribe(self, wav_bytes, sample_rate, language, timeout_sec):
        if self.transcribe_error:
            raise self.transcribe_error
        return self.transcript


@pytest.fixture
def fake_provider():
    return FakeProvider
