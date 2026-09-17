import io
import wave

import numpy as np

from voice_agent.audio import SpeechSegmenter, level_dbfs, to_wav_bytes
from voice_agent.env import load_dotenv


def frame(value, n=480):
    return np.full(n, value, dtype=np.int16)


def test_wav_bytes_roundtrip():
    pcm = np.arange(1600, dtype=np.int16)
    with wave.open(io.BytesIO(to_wav_bytes(pcm, 16000))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert np.array_equal(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), pcm)


def test_level_dbfs():
    assert level_dbfs(np.zeros(100, dtype=np.int16)) == -90.0
    assert -1.0 < level_dbfs(np.full(100, 32767, dtype=np.int16)) <= 0.0


def test_segmenter_waits_for_speech_and_ends_on_silence():
    seg = SpeechSegmenter(silence_ms=90, max_ms=10_000, pre_roll_ms=60)
    for _ in range(5):
        assert seg.push(frame(0), False) is False
    assert not seg.started
    assert seg.push(frame(1), True) is False
    assert seg.push(frame(1), True) is False
    assert seg.push(frame(0), False) is False
    assert seg.push(frame(0), False) is False
    assert seg.push(frame(0), False) is True  # 3 silent frames = 90 ms
    # 2 pre-roll + 2 speech + 3 silence frames
    assert len(seg.samples()) == 7 * 480


def test_segmenter_stops_at_max_length():
    seg = SpeechSegmenter(silence_ms=900, max_ms=150, pre_roll_ms=0)
    results = [seg.push(frame(1), True) for _ in range(5)]
    assert results == [False, False, False, False, True]


def test_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nVA_TEST_A="from file"\nexport VA_TEST_B=b\nVA_TEST_C=c\n', encoding="utf-8")
    monkeypatch.setenv("VA_TEST_C", "from shell")
    monkeypatch.delenv("VA_TEST_A", raising=False)
    monkeypatch.delenv("VA_TEST_B", raising=False)
    assert load_dotenv(env) == env
    import os
    assert (os.environ["VA_TEST_A"], os.environ["VA_TEST_B"], os.environ["VA_TEST_C"]) == ("from file", "b", "from shell")
