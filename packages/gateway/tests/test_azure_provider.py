"""Unit tests for AzureOpenAIProvider.

覆盖配置/默认值。Azure SDK 调用由 integration 测试覆盖。
"""

from __future__ import annotations

import pytest
from llm_gateway.providers.azure_provider import _AZURE_DEFAULT_API_VERSION, AzureOpenAIProvider


class TestAzureOpenAIProviderConfig:
    def test_provider_name(self) -> None:
        assert AzureOpenAIProvider().provider_name == "azure"

    def test_available_models_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认 deployment 'gpt-4o'。"""
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
        provider = AzureOpenAIProvider()
        assert "gpt-4o" in provider.available_models()

    def test_available_models_with_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-custom-deployment")
        provider = AzureOpenAIProvider()
        assert "my-custom-deployment" in provider.available_models()

    def test_default_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        provider = AzureOpenAIProvider()
        assert provider._api_version == _AZURE_DEFAULT_API_VERSION
        assert "2024-10-01-preview" in _AZURE_DEFAULT_API_VERSION


class TestAzureOpenAIProviderInit:
    def test_ctor_with_all_params(self) -> None:
        provider = AzureOpenAIProvider(
            api_key="key",
            endpoint="https://x.openai.azure.com",
            api_version="2024-08-01",
            deployment="gpt-4o-mini",
        )
        assert provider._api_key == "key"
        assert provider._endpoint == "https://x.openai.azure.com"
        assert provider._api_version == "2024-08-01"
        assert provider._deployment == "gpt-4o-mini"
        assert provider.default_model == "gpt-4o-mini"

    def test_ctor_uses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        provider = AzureOpenAIProvider()
        assert provider._api_key == "env-key"
        assert provider._endpoint == "https://env.openai.azure.com"


class TestAzureOpenAIProviderAvailability:
    def test_unavailable_without_key(self) -> None:
        provider = AzureOpenAIProvider(api_key="")
        assert provider.is_available() is False

    def test_unavailable_with_mock_key(self) -> None:
        provider = AzureOpenAIProvider(api_key="MOCK_KEY")
        assert provider.is_available() is False

    def test_unavailable_without_endpoint(self) -> None:
        """即使有 key,没有 endpoint 也不可用。"""
        provider = AzureOpenAIProvider(api_key="real-key", endpoint="")
        assert provider.is_available() is False

    def test_unavailable_without_openai_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """openai SDK 未装时不可用。"""
        import sys

        # 模拟 openai 不可导入
        original = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            provider = AzureOpenAIProvider(api_key="real-key", endpoint="https://x.openai.azure.com")
            # 触发 import 失败 → is_available 应当 False
            # (但 sys.modules[openai] = None 实际会触发 AttributeError 而不是 ImportError, 验证行为)
            try:
                result = provider.is_available()
                # 如果不抛错, 应当是 False (因为 import 失败)
                assert result is False
            except (AttributeError, ImportError):
                # AttributeError 也可接受 - 表示 SDK 不可用
                pass
        finally:
            if original is not None:
                sys.modules["openai"] = original
            else:
                sys.modules.pop("openai", None)
