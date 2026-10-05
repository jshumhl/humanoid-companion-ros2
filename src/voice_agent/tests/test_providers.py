import re

import pytest

from voice_agent.providers import (
    REGISTRY, ProviderConfigError, ProviderError, ProviderUnavailable, create_provider,
)
from voice_agent.providers.dashscope import DashScopeProvider, extract_transcript

from .conftest import PACKAGE_DIR


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ProviderConfigError, match="LLM_PROVIDER='nope' is not supported"):
        create_provider({})


def test_dashscope_selected_by_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "DashScope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    provider = create_provider({"dashscope": {"llm_model": "qwen-plus-character"}})
    assert isinstance(provider, DashScopeProvider)
    assert provider.llm_model == "qwen-plus-character"


def test_dashscope_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError, match="DASHSCOPE_API_KEY is not set"):
        DashScopeProvider({})


def test_dashscope_rejects_unknown_settings(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    with pytest.raises(ProviderConfigError, match="Unknown keys in providers.dashscope: api_key"):
        DashScopeProvider({"api_key": "sk-should-not-be-here"})


def test_dashscope_base_url_from_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://ws-abc.ap-southeast-1.maas.aliyuncs.com/api/v1/")
    assert DashScopeProvider({}).base_url == "https://ws-abc.ap-southeast-1.maas.aliyuncs.com/api/v1"


def test_unreachable_host_is_provider_unavailable(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://127.0.0.1:9/api/v1")  # nothing listens on port 9
    with pytest.raises(ProviderUnavailable):
        DashScopeProvider({}).chat([{"role": "user", "content": "你好"}], timeout_sec=2)


@pytest.mark.parametrize("response, text", [
    # Regional endpoint
    ({"output": {"output": {"sentence": {"text": "你看见什么。", "sentence_end": True}}}}, "你看见什么。"),
    # Workspace endpoint
    ({"sentence": {"text": "你是谁？"}}, "你是谁？"),
    # Several sentences
    ({"output": {"sentence": [{"text": "你好。"}, {"text": "你是谁？"}]}}, "你好。你是谁？"),
    ({"output": {}}, ""),
])
def test_extract_transcript(response, text):
    assert extract_transcript(response) == text


def test_no_vendor_imports_outside_providers():
    """Vendor SDKs may only be imported inside voice_agent/providers/."""
    vendor = re.compile(r"^\s*(import|from)\s+(dashscope|openai|anthropic|google|zhipuai)\b", re.MULTILINE)
    offenders = [
        str(path.relative_to(PACKAGE_DIR))
        for path in (PACKAGE_DIR / "voice_agent").rglob("*.py")
        if "providers" not in path.parts and vendor.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    assert set(REGISTRY) == {"dashscope"}


# --- keeping the key and provider internals out of errors ---

@pytest.mark.parametrize("url", [
    "http://dashscope.aliyuncs.com/api/v1",                 # not HTTPS
    "https://user:secret@dashscope.aliyuncs.com/api/v1",    # credentials in the URL
    "https://dashscope.aliyuncs.com/api/v1?key=secret",     # query string
    "https://dashscope.aliyuncs.com/api/v1#frag",           # fragment
])
def test_dashscope_rejects_unsafe_base_urls(monkeypatch, url):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", url)
    with pytest.raises(ProviderConfigError) as excinfo:
        DashScopeProvider({})
    assert "secret" not in str(excinfo.value)


class FakeHttpResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("not JSON")
        return self._json


def make_dashscope(monkeypatch, response):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    provider = DashScopeProvider({})
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(provider._session, "post", post)
    return provider, calls


def test_dashscope_error_shows_code_and_message_not_the_body(monkeypatch):
    body = {"code": "InvalidApiKey", "message": "Invalid API-key provided.",
            "request_id": "abc-123", "echo": "SENSITIVE"}
    provider, _ = make_dashscope(monkeypatch, FakeHttpResponse(401, body, text=str(body)))
    with pytest.raises(ProviderError) as excinfo:
        provider.chat([{"role": "user", "content": "你好"}], timeout_sec=5)
    message = str(excinfo.value)
    assert "HTTP 401" in message and "InvalidApiKey" in message and "abc-123" in message
    assert "SENSITIVE" not in message


def test_dashscope_non_json_error_is_reduced_to_status(monkeypatch):
    page = "<html><body>502 Bad Gateway nginx internal-host-01</body></html>"
    provider, _ = make_dashscope(monkeypatch, FakeHttpResponse(502, text=page))
    with pytest.raises(ProviderUnavailable) as excinfo:
        provider.chat([{"role": "user", "content": "你好"}], timeout_sec=5)
    assert "HTTP 502" in str(excinfo.value)
    assert "internal-host-01" not in str(excinfo.value)


def test_dashscope_requests_do_not_follow_redirects(monkeypatch):
    ok = FakeHttpResponse(200, {"output": {"choices": [{"message": {"content": "好"}}]}})
    provider, calls = make_dashscope(monkeypatch, ok)
    provider.chat([{"role": "user", "content": "你好"}], timeout_sec=5)
    assert calls[0]["allow_redirects"] is False
