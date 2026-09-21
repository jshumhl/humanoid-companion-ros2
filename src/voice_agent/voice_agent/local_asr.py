"""Offline speech recognition for the menu, using Vosk.

Cloud recognition is unusable exactly when the menu is needed, so menu answers
are recognized on the machine instead. The grammar is built from the menu
options in config.yaml, so Vosk only has to choose between a handful of
phrases, plus `[unk]` for everything else. That is far more reliable than
open-ended recognition from a small model.

Two details of the Chinese model shape the grammar:

- Words missing from its vocabulary are dropped with a warning, and 重试 is
  one of them. Each keyword is therefore also offered split into characters
  ("重 试"), which the model does recognize.
- Recognition output is space-separated, e.g. "重 试". Matching normalizes it.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16000
UNKNOWN = "[unk]"

# Spoken position words. Digits are for typing, not speaking, so they are not
# part of the grammar.
SPOKEN_POSITIONS = (("一",), ("二", "两"), ("三",))


class LocalRecognizerUnavailable(RuntimeError):
    """Vosk is not installed, or the model is missing or unreadable."""


def build_grammar(options):
    """Grammar phrases for the configured menu options, plus [unk].

    Each keyword appears whole and split into characters, because the small
    Chinese model knows some multi-character words and not others.
    """
    phrases = []
    for index, option in enumerate(options):
        keyword = option.keyword.strip()
        if keyword:
            _add(phrases, keyword)
            if len(keyword) > 1:
                _add(phrases, " ".join(keyword))
        if index < len(SPOKEN_POSITIONS):
            for word in SPOKEN_POSITIONS[index]:
                _add(phrases, word)
    phrases.append(UNKNOWN)
    return phrases


def _add(phrases, phrase):
    if phrase not in phrases:
        phrases.append(phrase)


class VoskMenuRecognizer:
    """Recognizes one menu answer from int16 mono audio. No network."""

    def __init__(self, model_path, options, sample_rate=MODEL_SAMPLE_RATE):
        try:
            from vosk import Model, SetLogLevel
        except ImportError as e:
            raise LocalRecognizerUnavailable(
                "vosk is not installed. Run: pip install vosk"
            ) from e

        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise LocalRecognizerUnavailable(
                f"Vosk model not found at {path}. See SETUP.md, 'Offline menu recognizer'."
            )
        # -2 also silences "Ignoring word missing in vocabulary", which is
        # expected: keywords are offered whole and split into characters, and
        # the model knows only one of the two forms.
        SetLogLevel(-2)
        try:
            self._model = Model(str(path))
        except Exception as e:
            raise LocalRecognizerUnavailable(f"Cannot load the Vosk model at {path}: {e}") from e

        self._sample_rate = sample_rate
        self.grammar = build_grammar(options)
        self._grammar_json = json.dumps(self.grammar, ensure_ascii=False)
        log.debug("Local recognizer ready: %s, grammar %s", path, self.grammar)

    def recognize(self, pcm):
        """Return the recognized text, e.g. "重 试" or "[unk]". Never raises for audio."""
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self._model, self._sample_rate, self._grammar_json)
        recognizer.AcceptWaveform(bytes(memoryview(pcm).cast("B")))
        text = json.loads(recognizer.FinalResult()).get("text", "")
        log.debug("Local recognizer heard %r", text)
        return text


def load_menu_recognizer(local_asr_config, options, sample_rate=MODEL_SAMPLE_RATE):
    """Build the recognizer, or return None with a warning if it is unavailable."""
    if not local_asr_config.enabled:
        return None
    try:
        return VoskMenuRecognizer(local_asr_config.model_path, options, sample_rate)
    except LocalRecognizerUnavailable as e:
        log.warning("Offline menu recognizer unavailable: %s", e)
        return None
