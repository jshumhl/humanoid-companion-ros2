import subprocess
import threading
import time
from pathlib import Path

from voice_agent.speech import SpeechOutput


class FakeSpeaker:
    voice = "zh-CN-XiaoxiaoNeural"

    def __init__(self, fail=False):
        self.fail = fail
        self.synthesized = []
        self.played = []

    def synthesize(self, text, path):
        if self.fail:
            raise OSError("Cannot connect to host speech.platform.bing.com")
        self.synthesized.append(text)
        Path(path).write_bytes(b"ID3edge")

    def play(self, path):
        self.played.append(Path(path).read_bytes())


class FakeTtsProvider:
    name = "fake"

    def __init__(self, fail=False):
        self.fail = fail

    def synthesize(self, text, language, timeout_sec):
        if self.fail:
            raise RuntimeError("Engine error [411]")
        return b"ID3provider"


def test_cached_phrase_plays_without_network(tmp_path):
    online = FakeSpeaker()
    SpeechOutput(online, tmp_path, timeout_sec=5).prepare(["我现在连不上网络，请稍后再试。"])

    offline = FakeSpeaker(fail=True)
    SpeechOutput(offline, tmp_path, timeout_sec=5).say("我现在连不上网络，请稍后再试。")
    assert offline.played == [b"ID3edge"]


def test_say_never_raises_when_synthesis_fails(tmp_path):
    speaker = FakeSpeaker(fail=True)
    SpeechOutput(speaker, tmp_path, timeout_sec=5).say("你好。")
    assert speaker.played == []
    assert list(tmp_path.iterdir()) == []  # no half-written cache files


def test_provider_tts_used_when_configured(tmp_path):
    speaker = FakeSpeaker()
    SpeechOutput(speaker, tmp_path, timeout_sec=5, provider=FakeTtsProvider()).say("你好。")
    assert speaker.played == [b"ID3provider"]
    assert speaker.synthesized == []


def test_gesture_starts_with_playback_not_after(tmp_path):
    """The callback must fire before play(), so the movement runs with the voice."""
    events = []
    speaker = FakeSpeaker()
    speaker.play = lambda path: events.append("play")

    SpeechOutput(speaker, tmp_path, timeout_sec=5).say(
        "你好。", on_playback_start=lambda: events.append("gesture"))

    assert events == ["gesture", "play"]


def test_no_gesture_when_speech_cannot_be_produced(tmp_path):
    """A gesture without its sentence would be a robot waving in silence."""
    events = []
    SpeechOutput(FakeSpeaker(fail=True), tmp_path, timeout_sec=5).say(
        "你好。", on_playback_start=lambda: events.append("gesture"))
    assert events == []


def test_say_without_callback_still_speaks(tmp_path):
    speaker = FakeSpeaker()
    SpeechOutput(speaker, tmp_path, timeout_sec=5).say("你好。")
    assert speaker.played == [b"ID3edge"]


class FakePlayer:
    """A player process that keeps running until it is stopped."""

    def __init__(self, ignores_terminate=False):
        self.terminated = False
        self.killed = False
        self._ignores_terminate = ignores_terminate

    def poll(self):
        return 0 if self.terminated or self.killed else None

    def terminate(self):
        if not self._ignores_terminate:
            self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.poll() is None:
            raise subprocess.TimeoutExpired("player", timeout)
        return 0


class PlayingSpeaker(FakeSpeaker):
    def __init__(self, ignores_terminate=False):
        super().__init__()
        self.process = FakePlayer(ignores_terminate)
        self.started = None

    def start(self, path):
        self.started = path
        return self.process


def test_playback_stops_within_the_budget(tmp_path):
    """Interrupting mid-sentence must silence the robot in well under 200 ms."""
    speaker = PlayingSpeaker()
    interrupt = threading.Event()
    threading.Timer(0.03, interrupt.set).start()   # interrupt while it is playing

    started = time.monotonic()
    result = SpeechOutput(speaker, tmp_path, timeout_sec=5).say("第一句。", interrupt=interrupt)

    assert result.interrupted and not result.spoken
    assert speaker.process.terminated
    stop_latency_ms = (result.stopped_at - started) * 1000
    assert stop_latency_ms < 200


def test_player_that_ignores_terminate_is_killed(tmp_path):
    speaker = PlayingSpeaker(ignores_terminate=True)
    interrupt = threading.Event()
    threading.Timer(0.03, interrupt.set).start()

    result = SpeechOutput(speaker, tmp_path, timeout_sec=5).say("第一句。", interrupt=interrupt)

    assert speaker.process.killed
    assert result.interrupted


def test_nothing_is_played_when_the_interrupt_arrives_first(tmp_path):
    """Already interrupted: start no audio rather than starting and cutting it."""
    speaker = PlayingSpeaker()
    interrupt = threading.Event()
    interrupt.set()

    result = SpeechOutput(speaker, tmp_path, timeout_sec=5).say("第一句。", interrupt=interrupt)

    assert result.interrupted and not result.spoken
    assert speaker.started is None      # the player was never launched


def test_playback_runs_to_the_end_when_not_interrupted(tmp_path):
    speaker = PlayingSpeaker()
    speaker.process.terminated = True   # a process that has already finished
    result = SpeechOutput(speaker, tmp_path, timeout_sec=5).say(
        "第一句。", interrupt=threading.Event())
    assert result.spoken and not result.interrupted


def test_synthesis_is_abandoned_when_interrupted_first(tmp_path):
    """A sentence interrupted before its audio exists is never played."""
    interrupt = threading.Event()
    release = threading.Event()

    class SlowSpeaker(FakeSpeaker):
        def synthesize(self, text, path):
            interrupt.set()
            release.wait(5)
            super().synthesize(text, path)

    speaker = SlowSpeaker()
    try:
        result = SpeechOutput(speaker, tmp_path, timeout_sec=5).say("第一句。", interrupt=interrupt)
    finally:
        release.set()

    assert result.interrupted and not result.spoken
    assert speaker.played == []


def test_provider_tts_failure_falls_back_to_edge_tts(tmp_path):
    speaker = FakeSpeaker()
    SpeechOutput(speaker, tmp_path, timeout_sec=5, provider=FakeTtsProvider(fail=True)).say("你好。")
    assert speaker.played == [b"ID3edge"]
