import re

import pytest

from voice_agent.providers import (
    REGISTRY, ProviderConfigError, ProviderUnavailable, create_provider,
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
