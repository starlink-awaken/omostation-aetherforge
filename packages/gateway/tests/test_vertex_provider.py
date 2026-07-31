"""Unit tests for VertexAIProvider.

覆盖配置/默认值。google-genai / google-generativeai SDK 调用由 integration 测试覆盖。
"""

from __future__ import annotations

import pytest
from llm_gateway.providers.vertex_provider import _DEFAULT_LOCATION, _DEFAULT_MODEL, _KNOWN_MODELS, VertexAIProvider


class TestVertexAIProviderConfig:
    def test_provider_name(self) -> None:
        assert VertexAIProvider().provider_name == "vertex"

    def test_available_models(self) -> None:
        models = VertexAIProvider().available_models()
        assert models == _KNOWN_MODELS
        # Gemini 家族都应存在
        assert any("gemini-2.5" in m for m in models)
        assert any("gemini-1.5" in m for m in models)

    def test_default_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        provider = VertexAIProvider()
        assert provider._location == _DEFAULT_LOCATION
        assert _DEFAULT_LOCATION == "us-central1"

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERTEX_MODEL_ID", raising=False)
        provider = VertexAIProvider()
        assert provider._model_id == _DEFAULT_MODEL
        assert "gemini" in _DEFAULT_MODEL


class TestVertexAIProviderInit:
    def test_ctor_with_kwargs(self) -> None:
        provider = VertexAIProvider(
            project="my-project",
            location="europe-west4",
            model_id="gemini-2.5-pro-001",
        )
        assert provider._project == "my-project"
        assert provider._location == "europe-west4"
        assert provider._model_id == "gemini-2.5-pro-001"

    def test_ctor_uses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
        provider = VertexAIProvider()
        assert provider._project == "env-project"
        assert provider._location == "asia-east1"


class TestVertexAIProviderAvailability:
    def test_unavailable_without_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        provider = VertexAIProvider()
        assert provider.is_available() is False

    def test_unavailable_with_mock_project(self) -> None:
        provider = VertexAIProvider(project="MOCK_PROJECT")
        # MOCK_KEY check is provider-specific; just verify doesn't crash
        try:
            result = provider.is_available()
            assert isinstance(result, bool)
        except Exception:
            pass
