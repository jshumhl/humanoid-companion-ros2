"""Microphone capture: push-to-talk, or voice activity detection (optional)."""

import io
import logging
import threading
import time
import wave

import numpy as np

log = logging.getLogger(__name__)

VAD_FRAME_MS = 30


def to_wav_bytes(pcm, sample_rate):
    """Encode int16 mono samples as a WAV file in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buffer.getvalue()


def level_dbfs(pcm):
    """RMS level of int16 samples in dBFS (0 = full scale, about -90 = silence)."""
    samples = np.asarray(pcm, dtype=np.float64)
    if samples.size == 0:
        return -90.0
    rms = np.sqrt(np.mean(samples ** 2)) / 32768.0
    return max(-90.0, 20 * np.log10(rms)) if rms > 0 else -90.0


class PushToTalkRecorder:
    """Enter starts recording, Enter stops. Works in any terminal, no extra permissions."""

    def __init__(self, audio_config):
        self._config = audio_config

    def record(self, wait_for_stop=input):
        """Record until `wait_for_stop` returns. Returns int16 samples."""
        import sounddevice as sd

        cfg = self._config
        max_samples = int(cfg.max_record_sec * cfg.sample_rate)
        chunks, total = [], 0
        lock = threading.Lock()

        def callback(indata, frames, time_info, status):
            nonlocal total
            if status:
                log.debug("audio input status: %s", status)
            with lock:
                if total < max_samples:
                    chunks.append(indata[:, 0].copy())
                    total += frames

        with sd.InputStream(
            samplerate=cfg.sample_rate, channels=1, dtype="int16",
            device=cfg.input_device, callback=callback,
        ):
            wait_for_stop()
        with lock:
            pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return pcm[:max_samples]


class SpeechSegmenter:
    """Turns a stream of (frame, is_speech) decisions into one utterance.

    Recording starts at the first speech frame (keeping a little audio from
    just before it) and ends after `silence_ms` without speech.
    """

    def __init__(self, silence_ms, max_ms, pre_roll_ms=300):
        self._silence_frames = max(1, silence_ms // VAD_FRAME_MS)
        self._max_frames = max(1, max_ms // VAD_FRAME_MS)
        self._pre_roll = max(0, pre_roll_ms // VAD_FRAME_MS)
        self._before = []
        self._frames = []
        self._started = False
        self._silent = 0

    @property
    def started(self):
        return self._started

    def push(self, frame, is_speech):
        """Add a frame. Returns True when the utterance is complete."""
        if not self._started:
            if not is_speech:
                self._before.append(frame)
                excess = len(self._before) - self._pre_roll
                if excess > 0:
                    del self._before[:excess]
                return False
            self._started = True
            self._frames = self._before + [frame]
            self._before = []
            return False

        self._frames.append(frame)
        self._silent = 0 if is_speech else self._silent + 1
        return self._silent >= self._silence_frames or len(self._frames) >= self._max_frames

    def samples(self):
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.int16)


class VadRecorder:
    """Records one utterance using webrtcvad. Needs `pip install webrtcvad-wheels`."""

    def __init__(self, audio_config):
        try:
            import webrtcvad
        except ImportError as e:
            raise RuntimeError(
                "listening.mode is always_on but webrtcvad is not installed. "
                "Run: pip install webrtcvad-wheels"
            ) from e
        self._config = audio_config
        self._vad = webrtcvad.Vad(audio_config.vad.aggressiveness)

    def record(self):
        """Wait for speech and record it. Returns int16 samples (empty if nobody spoke)."""
        import sounddevice as sd

        cfg = self._config
        frame_len = cfg.sample_rate * VAD_FRAME_MS // 1000
        segmenter = SpeechSegmenter(cfg.vad.silence_ms, int(cfg.max_record_sec * 1000))
        deadline = time.monotonic() + cfg.vad.start_timeout_sec

        with sd.InputStream(
            samplerate=cfg.sample_rate, channels=1, dtype="int16",
            device=cfg.input_device, blocksize=frame_len,
        ) as stream:
            while True:
                data, _ = stream.read(frame_len)
                frame = data[:, 0].copy()
                if segmenter.push(frame, self._vad.is_speech(frame.tobytes(), cfg.sample_rate)):
                    break
                if not segmenter.started and time.monotonic() > deadline:
                    return np.zeros(0, dtype=np.int16)
        return segmenter.samples()


def make_recorder(audio_config, mode="push_to_talk"):
    """push_to_talk: Enter starts and stops. always_on: VAD finds the utterance."""
    if mode == "always_on":
        return VadRecorder(audio_config)
    return PushToTalkRecorder(audio_config)
