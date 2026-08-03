"""Unit tests for DeepSeekProvider.

DeepSeekProvider 继承 OpenAIProvider,共享 OpenAI SDK 兼容性。
本测试覆盖配置/默认值,不实际调用 API。
"""

from __future__ import annotations

import pytest
from llm_gateway.providers.deepseek_provider import DeepSeekProvider


class TestDeepSeekProviderConfig:
    """Configuration / defaults — no SDK calls."""

    def test_provider_name(self) -> None:
        assert DeepSeekProvider().provider_name == "deepseek"

    def test_available_models(self) -> None:
        models = DeepSeekProvider().available_models()
        assert "deepseek-chat" in models
        assert "deepseek-reasoner" in models

    def test_default_base_url(self) -> None:
        """默认 base_url 应指向 deepseek API。"""
        provider = DeepSeekProvider(api_key="test-key")
        assert "deepseek.com" in provider.base_url  # type: ignore[reportOperatorIssue]

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认 model 来自 BOS_DEEPSEEK_MODEL env 或 hard-coded 'deepseek-chat'。"""
        monkeypatch.delenv("BOS_DEEPSEEK_MODEL", raising=False)
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.default_model == "deepseek-chat"

    def test_custom_model_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BOS_DEEPSEEK_MODEL env override 应生效。"""
        monkeypatch.setenv("BOS_DEEPSEEK_MODEL", "deepseek-coder")
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.default_model == "deepseek-coder"

    def test_custom_base_url_via_ctor(self) -> None:
        """__init__ 支持自定义 base_url override。"""
        provider = DeepSeekProvider(api_key="test-key", base_url="https://custom.api/v1")
        assert provider.base_url == "https://custom.api/v1"


class TestDeepSeekProviderAvailability:
    """is_available() 与 SDK / API key 状态关联。"""

    def test_unavailable_with_mock_key(self) -> None:
        """MOCK_KEY 是测试占位符,不算可用。"""
        provider = DeepSeekProvider(api_key="MOCK_KEY")
        assert provider.is_available() is False
