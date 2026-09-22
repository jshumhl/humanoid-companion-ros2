"""Noticing that the person wants the robot to stop talking.

A watcher is armed only while the robot is speaking. It polls its sources
often, so that playback stops promptly rather than at the end of the sentence.

Sources:
    enter  pressing Enter, always available
    voice  sustained speech picked up by VAD, in always_on mode

The voice source hears the robot's own speaker as well as the person, since
there is no echo cancellation here. `interrupt_ms` is the defence: a short
burst of the robot's own voice is ignored, sustained speech is not. On a robot
whose microphone hears its speaker clearly, raise it or use Enter.
"""

import logging
import select
import sys
import threading
import time

log = logging.getLogger(__name__)

SOURCE_ENTER = "enter"
SOURCE_VOICE = "voice"
VAD_FRAME_MS = 30


class InterruptSource:
    name = "base"

    def start(self):
        pass

    def poll(self):
        """True when this source says the person wants to interrupt."""
        return False

    def stop(self):
        pass


class KeyboardSource(InterruptSource):
    """Enter while the robot speaks. Reads the line so it is not seen as the next turn."""

    name = SOURCE_ENTER

    def __init__(self, stream=None):
        self._stream = stream or sys.stdin

    def poll(self):
        try:
            if not self._stream.isatty() and self._stream.closed:
                return False
            ready, _, _ = select.select([self._stream], [], [], 0)
        except (ValueError, OSError):
            return False  # stdin is not selectable, e.g. under some test runners
        if not ready:
            return False
        self._stream.readline()
        return True


class SustainedSpeech:
    """Counts consecutive speech frames, to ignore a syllable of echo."""

    def __init__(self, interrupt_ms, frame_ms=VAD_FRAME_MS):
        self._needed = max(1, round(interrupt_ms / frame_ms))
        self._count = 0

    def push(self, is_speech):
        """Add one frame. True once speech has lasted long enough to count."""
        self._count = self._count + 1 if is_speech else 0
        return self._count >= self._needed

    def reset(self):
        self._count = 0


class VoiceSource(InterruptSource):
    """Speech heard through the microphone while the robot is speaking."""

    name = SOURCE_VOICE

    def __init__(self, audio_config, interrupt_ms):
        import webrtcvad  # always_on mode requires it; checked at startup

        self._config = audio_config
        self._vad = webrtcvad.Vad(audio_config.vad.aggressiveness)
        self._detector = SustainedSpeech(interrupt_ms)
        self._frame_len = audio_config.sample_rate * VAD_FRAME_MS // 1000
        self._stream = None

    def start(self):
        import sounddevice as sd

        self._detector.reset()
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate, channels=1, dtype="int16",
            device=self._config.input_device, blocksize=self._frame_len,
        )
        self._stream.start()

    def poll(self):
        if self._stream is None:
            return False
        interrupted = False
        # Drain whatever has arrived; never block, playback is waiting on this.
        while self._stream.read_available >= self._frame_len:
            data, _ = self._stream.read(self._frame_len)
            frame = data[:, 0].copy()
            if self._detector.push(self._vad.is_speech(frame.tobytes(), self._config.sample_rate)):
                interrupted = True
                break
        return interrupted

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class InterruptWatcher:
    """Watches all sources while armed, and sets an event on the first hit."""

    def __init__(self, sources, poll_sec=0.02):
        self._sources = list(sources)
        self._poll_sec = poll_sec
        self._thread = None
        self._stop = threading.Event()
        self.event = threading.Event()
        self.source = ""
        self.detected_at = None

    def arm(self):
        """Start watching. The event is cleared, so an earlier interrupt does not linger."""
        self.event.clear()
        self.source = ""
        self.detected_at = None
        self._stop.clear()
        for source in self._sources:
            try:
                source.start()
            except Exception as e:
                log.warning("Interrupt source %s could not start: %s", source.name, e)
        if self._sources:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def disarm(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        for source in self._sources:
            try:
                source.stop()
            except Exception as e:
                log.warning("Interrupt source %s could not stop: %s", source.name, e)

    def _watch(self):
        while not self._stop.is_set():
            for source in self._sources:
                try:
                    hit = source.poll()
                except Exception as e:
                    log.warning("Interrupt source %s failed: %s", source.name, e)
                    continue
                if hit:
                    self.source = source.name
                    self.detected_at = time.monotonic()
                    self.event.set()
                    return
            self._stop.wait(self._poll_sec)


class NullWatcher:
    """Used where interruption does not apply, e.g. the offline menu."""

    def __init__(self):
        self.event = threading.Event()
        self.source = ""
        self.detected_at = None

    def arm(self):
        self.event.clear()

    def disarm(self):
        pass


def build_watcher(listening_config, audio_config):
    """Enter always; voice too when listening in always_on mode."""
    sources = [KeyboardSource()]
    if listening_config.mode == "always_on":
        try:
            sources.append(VoiceSource(audio_config, listening_config.interrupt_ms))
        except ImportError as e:
            log.warning("Voice interruption needs webrtcvad (%s); Enter still interrupts", e)
    return InterruptWatcher(sources, listening_config.playback_poll_ms / 1000)
