"""Unit tests for HitlLLMProvider.

AUDIT.md F-class: provider 单元测试缺口 — HITL (Human-in-the-Loop) provider
无单测。本测试覆盖配置 + 行为,不需要 SDK 依赖。
"""

from __future__ import annotations

import pytest
from llm_gateway.provider import LLMRequest
from llm_gateway.providers.hitl_provider import HitlLLMProvider


class TestHitlLLMProviderConfig:
    """Pure configuration / metadata checks (no SDK calls)."""

    def test_provider_name(self) -> None:
        assert HitlLLMProvider().provider_name == "hitl"

    def test_available_models(self) -> None:
        models = HitlLLMProvider().available_models()
        assert models == ["human-expert"]
        assert all(isinstance(m, str) for m in models)

    def test_is_available_returns_true(self) -> None:
        """HITL 是人工兜底,永远可用。"""
        assert HitlLLMProvider().is_available() is True


class TestHitlLLMProviderGenerate:
    """generate() / complete() 行为 — 不发实际请求,验证返回 mock 内容。"""

    @pytest.mark.asyncio
    async def test_generate_returns_hitl_marker(self) -> None:
        provider = HitlLLMProvider()
        req = LLMRequest(model="human-expert", prompt="emergency: system down")

        resp = await provider.generate(req)

        assert resp.provider == "hitl"
        assert resp.model == "human-expert"
        assert "HITL" in resp.content
        assert resp.metadata.get("hitl_triggered") is True
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0

    def test_complete_sync_returns_hitl_marker(self) -> None:
        provider = HitlLLMProvider()
        req = LLMRequest(model="human-expert", prompt="emergency: system down")

        resp = provider.complete(req)

        assert resp.provider == "hitl"
        assert "HITL" in resp.content
        assert resp.metadata.get("hitl_triggered") is True

    @pytest.mark.asyncio
    async def test_stream_generate_yields_single_chunk(self) -> None:
        """stream_generate() 实现为 generate() 后 yield content。"""
        provider = HitlLLMProvider()
        req = LLMRequest(model="human-expert", prompt="test")

        chunks: list[str] = []
        async for chunk in provider.stream_generate(req):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "HITL" in chunks[0]
