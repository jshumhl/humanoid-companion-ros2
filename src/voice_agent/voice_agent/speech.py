"""Spoken output: synthesize, cache, and play through object_narrator's Speaker.

Every phrase that has been spoken once is cached as an MP3 keyed by engine,
voice and text. The fallback phrases are cached at startup, so the robot can
still say "我现在连不上网络" when the network is down.
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

from .timeouts import call_with_timeout

log = logging.getLogger(__name__)


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

    def say(self, text):
        """Speak text. Never raises: on failure the text is only printed."""
        path = self._cached_path(text)
        try:
            if not path.exists():
                self._synthesize_to(text, path)
            self._speaker.play(str(path))
        except Exception as e:
            log.warning("Could not speak %r: %s", text, e)

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

    def _synthesize_to(self, text, path):
        fd, tmp = tempfile.mkstemp(suffix=".mp3", dir=self._cache_dir)
        os.close(fd)
        try:
            if self._provider is not None:
                try:
                    audio = call_with_timeout(self._provider.synthesize, self._timeout,
                                              text, self._language, self._timeout)
                    Path(tmp).write_bytes(audio)
                except Exception as e:
                    log.warning("Provider TTS failed, using edge-tts: %s", e)
                    call_with_timeout(self._speaker.synthesize, self._timeout, text, tmp)
            else:
                call_with_timeout(self._speaker.synthesize, self._timeout, text, tmp)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("synthesized audio is empty")
            os.replace(tmp, path)  # atomic: never leave a half-written cache file
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
