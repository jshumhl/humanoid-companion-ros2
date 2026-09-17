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


def test_provider_tts_failure_falls_back_to_edge_tts(tmp_path):
    speaker = FakeSpeaker()
    SpeechOutput(speaker, tmp_path, timeout_sec=5, provider=FakeTtsProvider(fail=True)).say("你好。")
    assert speaker.played == [b"ID3edge"]
