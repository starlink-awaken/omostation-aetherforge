"""Unit tests for GeminiProvider.

本测试覆盖配置/默认值,不需要 google-generativeai SDK (它是 optional dep)。
"""

from __future__ import annotations

import pytest
from llm_gateway.providers.gemini_provider import GeminiProvider


class TestGeminiProviderConfig:
    def test_provider_name(self) -> None:
        assert GeminiProvider().provider_name == "gemini"

    def test_available_models(self) -> None:
        models = GeminiProvider().available_models()
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models
        assert "gemini-2.0-flash" in models


class TestGeminiProviderInit:
    """__init__ 从 env 或参数拉取 API key。"""

    def test_default_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GeminiProvider()
        assert provider._api_key == ""

    def test_explicit_key(self) -> None:
        provider = GeminiProvider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    def test_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        provider = GeminiProvider()
        assert provider._api_key == "env-key"


class TestGeminiProviderAvailability:
    def test_unavailable_without_api_key(self) -> None:
        provider = GeminiProvider(api_key="")
        assert provider.is_available() is False

    def test_unavailable_with_mock_key(self) -> None:
        provider = GeminiProvider(api_key="MOCK_KEY")
        assert provider.is_available() is False
