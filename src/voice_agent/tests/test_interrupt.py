"""Interrupt sources and the watcher. No microphone: stdin is a pipe."""

import os
import threading
import time

import pytest

from voice_agent.config import AudioConfig, ListeningConfig
from voice_agent.interrupt import (
    SOURCE_ENTER, InterruptSource, InterruptWatcher, KeyboardSource, NullWatcher, SustainedSpeech,
    build_watcher,
)


@pytest.fixture
def pipe():
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r")
    writer = os.fdopen(write_fd, "w")
    yield reader, writer
    reader.close()
    writer.close()


def test_keyboard_source_is_quiet_until_enter(pipe):
    reader, writer = pipe
    source = KeyboardSource(reader)

    assert source.poll() is False

    writer.write("\n")
    writer.flush()
    assert source.poll() is True


def test_keyboard_source_consumes_the_line(pipe):
    """The keypress that stopped the robot must not become the next question."""
    reader, writer = pipe
    source = KeyboardSource(reader)
    writer.write("\n")
    writer.flush()

    assert source.poll() is True
    assert source.poll() is False   # the line was read, not left waiting


@pytest.mark.parametrize("frames, expected", [
    ([True] * 10, True),                    # 300 ms of speech
    ([True] * 9, False),                    # 270 ms is not enough
    ([True] * 5 + [False] + [True] * 5, False),   # a burst, a gap, another burst
    ([False] * 20, False),
])
def test_sustained_speech_needs_an_unbroken_run(frames, expected):
    """A syllable of the robot's own voice in the microphone must not count."""
    detector = SustainedSpeech(interrupt_ms=300, frame_ms=30)
    assert any(detector.push(frame) for frame in frames) is expected


def test_sustained_speech_resets_between_turns():
    detector = SustainedSpeech(interrupt_ms=60, frame_ms=30)
    detector.push(True)
    detector.reset()
    assert detector.push(True) is False     # the earlier frame no longer counts


class ScriptedSource(InterruptSource):
    def __init__(self, name, hits):
        self.name = name
        self._hits = list(hits)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def poll(self):
        return self._hits.pop(0) if self._hits else False

    def stop(self):
        self.stopped = True


def test_watcher_sets_the_event_and_names_the_source():
    source = ScriptedSource("enter", [False, True])
    watcher = InterruptWatcher([source], poll_sec=0.001)
    watcher.arm()
    try:
        assert watcher.event.wait(1.0) is True
        assert watcher.source == "enter"
        assert watcher.detected_at is not None
    finally:
        watcher.disarm()
    assert source.started and source.stopped


def test_watcher_stays_quiet_when_nothing_happens():
    watcher = InterruptWatcher([ScriptedSource("enter", [])], poll_sec=0.001)
    watcher.arm()
    try:
        assert watcher.event.wait(0.1) is False
    finally:
        watcher.disarm()


def test_arming_clears_a_previous_interruption():
    watcher = InterruptWatcher([ScriptedSource("enter", [True])], poll_sec=0.001)
    watcher.arm()
    watcher.event.wait(1.0)
    watcher.disarm()

    watcher.arm()
    try:
        assert watcher.event.is_set() is False
    finally:
        watcher.disarm()


def test_a_failing_source_does_not_stop_the_others(caplog):
    class Broken(InterruptSource):
        name = "broken"

        def poll(self):
            raise RuntimeError("device went away")

    watcher = InterruptWatcher([Broken(), ScriptedSource("enter", [True])], poll_sec=0.001)
    watcher.arm()
    try:
        assert watcher.event.wait(1.0) is True
        assert watcher.source == "enter"
    finally:
        watcher.disarm()


def test_push_to_talk_watcher_has_only_the_keyboard():
    watcher = build_watcher(ListeningConfig(mode="push_to_talk"), AudioConfig())
    assert [s.name for s in watcher._sources] == [SOURCE_ENTER]


def test_null_watcher_never_fires():
    watcher = NullWatcher()
    watcher.arm()
    assert watcher.event.is_set() is False
    watcher.disarm()


def test_watcher_thread_stops_promptly():
    watcher = InterruptWatcher([ScriptedSource("enter", [])], poll_sec=0.01)
    watcher.arm()
    started = time.monotonic()
    watcher.disarm()
    assert time.monotonic() - started < 0.5
    assert threading.active_count() >= 1
