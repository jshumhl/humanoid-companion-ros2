"""Text-to-speech with edge-tts, played through a local command-line player."""

import os
import shlex
import shutil
import subprocess
import tempfile

import edge_tts

DEFAULT_VOICES = {
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "zh-HK": "zh-HK-HiuGaaiNeural",
}

# Players that can decode MP3, tried in order.
_PLAYERS = [
    ["afplay"],
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpv", "--no-video", "--really-quiet"],
]


def find_player():
    for command in _PLAYERS:
        if shutil.which(command[0]):
            return command
    raise RuntimeError(
        "No MP3 player found. Install mpg123 or ffmpeg, or set `player` in the config."
    )


class Speaker:
    def __init__(self, language, voice=None, player=None):
        self._voice = voice or DEFAULT_VOICES[language]
        self._player = shlex.split(player) if player else find_player()

    @property
    def voice(self):
        return self._voice

    def synthesize(self, text, path):
        """Write speech for `text` to an MP3 file. Needs network access."""
        edge_tts.Communicate(text, self._voice).save_sync(path)

    def play(self, path):
        subprocess.run([*self._player, path], check=True)

    def speak(self, text):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            self.synthesize(text, path)
            self.play(path)
        finally:
            os.remove(path)
