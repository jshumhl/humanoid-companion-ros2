"""Provider selection by the LLM_PROVIDER environment variable."""

import importlib
import os

from .base import Provider, ProviderConfigError, ProviderError, ProviderUnavailable

# LLM_PROVIDER value -> "module:class". Adapters are imported only when selected,
# so an unused provider's SDK does not need to be installed.
REGISTRY = {
    "dashscope": "voice_agent.providers.dashscope:DashScopeProvider",
}

DEFAULT_PROVIDER = "dashscope"

__all__ = [
    "Provider", "ProviderConfigError", "ProviderError", "ProviderUnavailable",
    "REGISTRY", "create_provider", "selected_provider_name",
]


def selected_provider_name():
    return os.environ.get("LLM_PROVIDER", "").strip().lower() or DEFAULT_PROVIDER


def create_provider(provider_settings):
    """Instantiate the provider named by LLM_PROVIDER.

    `provider_settings` is the `providers` section of config.yaml; each adapter
    receives only its own sub-section.
    """
    name = selected_provider_name()
    if name not in REGISTRY:
        raise ProviderConfigError(
            f"LLM_PROVIDER={name!r} is not supported. Choose one of: {', '.join(sorted(REGISTRY))}"
        )
    module_name, class_name = REGISTRY[name].split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ProviderConfigError(f"Provider {name!r} needs a package that is not installed: {e}") from e
    return getattr(module, class_name)(provider_settings.get(name) or {})
