"""Spoken output: synthesize, cache, and play through object_narrator's Speaker.

Every phrase that has been spoken once is cached as an MP3 keyed by engine,
voice and text. The fallback phrases are cached at startup, so the robot can
still say "我现在连不上网络" when the network is down.
"""

import hashlib
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .timeouts import CallCancelled, call_with_timeout

log = logging.getLogger(__name__)

PLAYBACK_POLL_SEC = 0.02   # how often playback checks for an interrupt
KILL_AFTER_SEC = 0.15      # a player that ignores terminate is killed


@dataclass
class SpeakResult:
    spoken: bool                  # the audio was played to the end
    interrupted: bool = False
    started_at: float = None      # time.monotonic() when playback began
    stopped_at: float = None      # time.monotonic() when the audio went silent

    @property
    def played_sec(self):
        if self.started_at is None or self.stopped_at is None:
            return 0.0
        return self.stopped_at - self.started_at


class SpeechOutput:
    def __init__(self, speaker, cache_dir, timeout_sec, provider=None, language="zh-CN"):
        """
        speaker:  object_narrator.speaker.Speaker (edge-tts synthesis + MP3 playback)
        provider: if given, its synthesize() is tried first, with edge-tts as fallback
        """
        self._speaker = speaker
        self._provider = provider
        self._language = language
        self._timeout = timeout_sec
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def say(self, text, on_playback_start=None, interrupt=None):
        """Speak text. Never raises: on failure the text is only printed.

        `on_playback_start` runs once the audio is ready and playback is about
        to begin, which is when a gesture should start so that it runs with the
        voice rather than after it.

        `interrupt` is an event watched during synthesis and playback. When it
        is set, the request is abandoned and playback is stopped. Returns a
        SpeakResult saying whether the sentence was finished.
        """
        path = self._cached_path(text)
        try:
            if not path.exists():
                self._synthesize_to(text, path, interrupt)
        except CallCancelled:
            return SpeakResult(spoken=False, interrupted=True)
        except Exception as e:
            log.warning("Could not speak %r: %s", text, e)
            return SpeakResult(spoken=False)

        if interrupt is not None and interrupt.is_set():
            return SpeakResult(spoken=False, interrupted=True)
        try:
            if on_playback_start is not None:
                on_playback_start()
            return self._play(path, interrupt)
        except Exception as e:
            log.warning("Could not play %r: %s", text, e)
            return SpeakResult(spoken=False)

    def _play(self, path, interrupt):
        if interrupt is None:
            self._speaker.play(str(path))
            return SpeakResult(spoken=True)

        started_at = time.monotonic()
        process = self._speaker.start(str(path))
        while process.poll() is None:
            if interrupt.is_set():
                stopped_at = self._stop_playback(process)
                return SpeakResult(spoken=False, interrupted=True,
                                   started_at=started_at, stopped_at=stopped_at)
            time.sleep(PLAYBACK_POLL_SEC)
        return SpeakResult(spoken=True, started_at=started_at, stopped_at=time.monotonic())

    def _stop_playback(self, process):
        """Stop the player and return when it was actually silent."""
        process.terminate()
        try:
            process.wait(timeout=KILL_AFTER_SEC)
        except subprocess.TimeoutExpired:
            log.warning("Player ignored terminate; killing it")
            process.kill()
            process.wait(timeout=KILL_AFTER_SEC)
        return time.monotonic()

    def prefetch(self, text):
        """Synthesize in the background, so the next sentence is ready when needed.

        Speaking sentence by sentence makes interruption possible, but it would
        otherwise leave half a second of silence at every full stop while the
        next sentence is synthesized. This overlaps that with playback.
        """
        if not text or self._cached_path(text).exists():
            return
        thread = threading.Thread(target=self._prefetch_one, args=(text,), daemon=True)
        thread.start()

    def _prefetch_one(self, text):
        path = self._cached_path(text)
        try:
            if not path.exists():
                self._synthesize_to(text, path)
        except Exception as e:
            log.debug("Prefetch of %r failed, will retry when spoken: %s", text, e)

    def prepare(self, phrases):
        """Cache phrases now, while the network is likely up. Best effort."""
        for text in phrases:
            path = self._cached_path(text)
            if path.exists():
                continue
            try:
                self._synthesize_to(text, path)
            except Exception as e:
                log.warning("Could not pre-cache %r: %s", text, e)

    def _cached_path(self, text):
        engine = f"{self._provider.name}" if self._provider else f"edge-{self._speaker.voice}"
        digest = hashlib.sha256(f"{engine}\n{text}".encode("utf-8")).hexdigest()[:24]
        return self._cache_dir / f"{digest}.mp3"

    def _synthesize_to(self, text, path, interrupt=None):
        fd, tmp = tempfile.mkstemp(suffix=".mp3", dir=self._cache_dir)
        os.close(fd)
        try:
            if self._provider is not None:
                try:
                    audio = call_with_timeout(self._provider.synthesize, self._timeout,
                                              text, self._language, self._timeout,
                                              cancel_event=interrupt)
                    Path(tmp).write_bytes(audio)
                except CallCancelled:
                    raise
                except Exception as e:
                    log.warning("Provider TTS failed, using edge-tts: %s", e)
                    call_with_timeout(self._speaker.synthesize, self._timeout, text, tmp,
                                      cancel_event=interrupt)
            else:
                call_with_timeout(self._speaker.synthesize, self._timeout, text, tmp,
                                  cancel_event=interrupt)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("synthesized audio is empty")
            os.replace(tmp, path)  # atomic: never leave a half-written cache file
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
