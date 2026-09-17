"""Interface every speech/LLM backend implements.

Call sites only use `Provider` and the exceptions below. Vendor SDKs and
vendor-specific request formats live in adapter modules in this directory and
nowhere else.
"""

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """The provider answered, but the request failed (auth, quota, bad model...)."""


class ProviderUnavailable(ProviderError):
    """The provider could not be reached: network down, DNS, timeout, 5xx."""


class ProviderConfigError(ProviderError):
    """Missing API key or invalid provider settings. Raised at startup."""


class Provider(ABC):
    name = "base"

    @abstractmethod
    def chat(self, messages, timeout_sec):
        """Send chat messages and return the model's raw text reply.

        `messages` is a list of {"role": "system"|"user"|"assistant", "content": str}.
        """

    @abstractmethod
    def transcribe(self, wav_bytes, sample_rate, language, timeout_sec):
        """Transcribe mono 16-bit WAV audio and return the text ('' if nothing heard).

        `language` is an object_narrator language code: zh-CN, zh-TW or zh-HK.
        """

    def synthesize(self, text, language, timeout_sec):
        """Return MP3 bytes for `text`. Optional; edge-tts is used when unsupported."""
        raise NotImplementedError(f"{self.name} does not provide text-to-speech")
