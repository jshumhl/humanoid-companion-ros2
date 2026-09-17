"""DashScope (Alibaba Cloud Model Studio): Qwen chat, Qwen audio ASR and TTS.

Talks to the DashScope HTTP API directly with `requests`, which gives real
connect/read timeouts and readable error bodies.

Environment:
    DASHSCOPE_API_KEY   required
    DASHSCOPE_BASE_URL  optional, default https://dashscope-intl.aliyuncs.com/api/v1.
                        Mainland China: https://dashscope.aliyuncs.com/api/v1.
                        Workspace endpoint:
                        https://<workspace>.<region>.maas.aliyuncs.com/api/v1
"""

import base64
import logging
import os
from urllib.parse import urlparse

import requests

from .base import Provider, ProviderConfigError, ProviderError, ProviderUnavailable

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
CONNECT_TIMEOUT_SEC = 5

DEFAULTS = {
    "llm_model": "qwen-plus-character",
    "asr_model": "qwen-audio-3.0-asr-flash",
    "tts_model": "qwen-audio-3.0-tts-plus",
    "tts_voice": "longanyang",
}

_CHAT_PATH = "/services/aigc/text-generation/generation"
_ASR_PATH = "/services/aigc/multimodal-generation/generation"
_TTS_PATH = "/services/audio/tts/SpeechSynthesizer"


class DashScopeProvider(Provider):
    name = "dashscope"

    def __init__(self, settings):
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ProviderConfigError(
                f"Unknown keys in providers.dashscope: {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(DEFAULTS)}"
            )
        merged = {**DEFAULTS, **settings}
        for key, value in merged.items():
            if not isinstance(value, str) or not value.strip():
                raise ProviderConfigError(f"providers.dashscope.{key} must be a non-empty string")
        self.llm_model = merged["llm_model"]
        self.asr_model = merged["asr_model"]
        self.tts_model = merged["tts_model"]
        self.tts_voice = merged["tts_voice"]

        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ProviderConfigError(
                "DASHSCOPE_API_KEY is not set. Export it or put it in a .env file (see SETUP.md)."
            )
        self._headers = {"Authorization": f"Bearer {api_key}"}

        base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ProviderConfigError(
                f"DASHSCOPE_BASE_URL must look like https://host/api/v1, got {base_url!r}"
            )
        self.base_url = base_url
        self._session = requests.Session()

    def chat(self, messages, timeout_sec):
        body = {
            "model": self.llm_model,
            "input": {"messages": messages},
            "parameters": {"result_format": "message"},
        }
        data = self._post(_CHAT_PATH, body, timeout_sec).json()
        try:
            return data["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Unexpected chat response: {str(data)[:300]}") from e

    def transcribe(self, wav_bytes, sample_rate, language, timeout_sec):
        audio = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
        body = {
            "model": self.asr_model,
            "input": {"messages": [{"role": "user", "content": [{"audio": audio}]}]},
            "parameters": {"format": "wav", "sample_rate": sample_rate},
        }
        return extract_transcript(self._post(_ASR_PATH, body, timeout_sec).json())

    def synthesize(self, text, language, timeout_sec):
        body = {
            "model": self.tts_model,
            "input": {"text": text, "voice": self.tts_voice, "format": "mp3", "sample_rate": 24000},
        }
        data = self._post(_TTS_PATH, body, timeout_sec).json()
        url = ((data.get("output") or {}).get("audio") or {}).get("url")
        if not url:
            raise ProviderError(f"TTS response has no audio URL: {str(data)[:300]}")
        try:
            response = self._session.get(url, timeout=(CONNECT_TIMEOUT_SEC, timeout_sec))
            response.raise_for_status()
        except requests.RequestException as e:
            raise ProviderUnavailable(f"DashScope TTS audio download failed: {e}") from e
        return response.content

    def _post(self, path, body, timeout_sec):
        url = self.base_url + path
        try:
            response = self._session.post(
                url, json=body, headers=self._headers, timeout=(CONNECT_TIMEOUT_SEC, timeout_sec)
            )
        except requests.RequestException as e:
            # Connection refused, DNS failure, TLS failure, timeout.
            raise ProviderUnavailable(f"DashScope unreachable ({path}): {e}") from e

        if response.status_code >= 500:
            raise ProviderUnavailable(f"DashScope {response.status_code} ({path}): {response.text[:300]}")
        if response.status_code != 200:
            raise ProviderError(f"DashScope {response.status_code} ({path}): {response.text[:300]}")
        return response


def extract_transcript(data):
    """Collect sentence text from an ASR response.

    Endpoints differ in nesting: the regional endpoint returns
    {"output": {"output": {"sentence": ...}}}, workspace endpoints return
    {"sentence": ...}. `sentence` may be a dict or a list of dicts.
    """
    texts = []

    def walk(node):
        if isinstance(node, dict):
            sentence = node.get("sentence")
            if sentence is not None:
                for s in sentence if isinstance(sentence, list) else [sentence]:
                    if isinstance(s, dict) and s.get("text"):
                        texts.append(s["text"])
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return "".join(texts).strip()
