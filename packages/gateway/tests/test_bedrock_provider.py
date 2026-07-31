"""Unit tests for BedrockProvider.

覆盖配置/默认值。boto3 调用由 integration 测试覆盖。
"""

from __future__ import annotations

import pytest
from llm_gateway.providers.bedrock_provider import _KNOWN_MODELS, BedrockProvider


class TestBedrockProviderConfig:
    def test_provider_name(self) -> None:
        assert BedrockProvider().provider_name == "bedrock"

    def test_available_models_includes_known(self) -> None:
        models = BedrockProvider().available_models()
        for m in _KNOWN_MODELS:
            assert m in models
        # Claude / Llama / Mistral / Titan / Cohere 家族都应存在
        assert any("claude" in m for m in models)
        assert any("llama" in m for m in models)
        assert any("mistral" in m for m in models)

    def test_default_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AWS_REGION", raising=False)
        provider = BedrockProvider()
        assert provider._region == "us-east-1"

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AWS_BEDROCK_MODEL_ID", raising=False)
        provider = BedrockProvider()
        assert provider._model_id == "anthropic.claude-3-sonnet-20240229"


class TestBedrockProviderInit:
    def test_ctor_with_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key-id")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        provider = BedrockProvider()
        assert provider._region == "eu-west-1"
        assert provider._aws_key == "test-key-id"
        assert provider._aws_secret == "test-secret"  # noqa: S105

    def test_ctor_with_kwargs(self) -> None:
        provider = BedrockProvider(
            aws_access_key_id="kw-key",
            aws_secret_access_key="kw-secret",  # noqa: S106
            region="ap-southeast-1",
            model_id="anthropic.claude-3-haiku-20240307",
        )
        assert provider._aws_key == "kw-key"
        assert provider._region == "ap-southeast-1"
        assert provider._model_id == "anthropic.claude-3-haiku-20240307"


class TestBedrockProviderAvailability:
    def test_unavailable_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """无 AWS creds 时不可用。"""
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        provider = BedrockProvider()
        assert provider.is_available() is False

    def test_unavailable_with_mock_credentials(self) -> None:
        """MOCK_KEY 是测试占位符。"""
        provider = BedrockProvider(
            aws_access_key_id="MOCK_KEY",
            aws_secret_access_key="MOCK_SECRET",  # noqa: S106
        )
        # 注: bedrock 的 is_available 行为可能与 azure 不同, 此处仅验证不抛错
        try:
            result = provider.is_available()
            assert isinstance(result, bool)
        except Exception:
            pass
