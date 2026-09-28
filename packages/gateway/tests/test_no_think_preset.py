"""no_think_preset_models: 已知默认思考的模型 (Gemma-4) 首发即关 thinking。

2026-09-28 实测: gemma-4-e2b 经门面时思考段带 <channel|> 标记漏进正文, 正文非空,
触发不了「正文空 → 关 thinking 重试」, 只能首发就带 reasoning_effort=none。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_gateway.gateway import GatewayRequest, ModelGateway
from llm_gateway.types import ChatResult


# fixture 与 test_model_gateway.py 保持一致 (测试目录为 importlib 模式, 不能跨模块 import)
@pytest.fixture
def gateway_config():
    from llm_gateway.gateway import GatewayConfig

    return GatewayConfig(
        omlx_bin="/bin/true",
        local_base_url="http://127.0.0.1",
        model_ports={"coding-fast": 8081, "test-model": 9999},
        model_sizes={"coding-fast": 28.0},
        lmstudio_fallback={"coding-fast": "pool/coding-fallback"},
        ollama_fallback={"coding-fast": "ollama-coding"},
        local_backend="legacy",
        fallback_chain=["coding-fast", "test-model"],
        warm_pool_ttl=60,
    )


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.get_all.return_value = []
    reg.list_models.return_value = []
    return reg


@pytest.fixture
def mock_scheduler():
    return MagicMock()


@pytest.fixture(autouse=True)
def stable_memory_budget(monkeypatch):
    from llm_gateway.gateway import MemoryGuard

    monkeypatch.setattr(MemoryGuard, "_get_free_memory_gb", lambda self: 100.0)


def _gateway(gateway_config, mock_registry, mock_scheduler, model_id, calls):
    async def chat(mid, messages, options=None):
        extra = dict(options.extra or {})
        calls.append(extra)
        if extra.get("reasoning_effort") == "none":
            return ChatResult(content="北京是中国的首都。", finish_reason="stop")
        return ChatResult(content="The user is asking...<channel|>北京", finish_reason="stop")

    mock_registry.chat = chat
    mock_registry.get.return_value = MagicMock(id=model_id)
    mock_registry.get_provider.return_value = MagicMock(name="p")
    gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)
    gw._port_reachable = AsyncMock(return_value=False)
    gw._ensure_model = AsyncMock(return_value=False)  # type: ignore[method-assign]
    return gw


def _ask(gw, model):
    return asyncio.run(
        gw.generate(
            GatewayRequest(messages=[{"role": "user", "content": "北京是哪国首都"}], model=model, max_tokens=80)
        )
    )


def test_gemma4_first_call_disables_thinking(gateway_config, mock_registry, mock_scheduler):
    # 与既有用例同一路由: coding-fast 端口不可达 → LM Studio 兜底, 兜底目标换成 gemma-4
    gateway_config.lmstudio_fallback = {"coding-fast": "google/gemma-4-e2b"}
    calls: list[dict] = []
    gw = _gateway(gateway_config, mock_registry, mock_scheduler, "google/gemma-4-e2b", calls)
    assert _ask(gw, "coding-fast").content == "北京是中国的首都。"
    assert calls == [gateway_config.no_think_param], f"首发应直接带关 thinking 参数, 实际 {calls}"


def test_other_models_unaffected(gateway_config, mock_registry, mock_scheduler):
    calls: list[dict] = []
    gw = _gateway(gateway_config, mock_registry, mock_scheduler, "pool/coding-fallback", calls)
    _ask(gw, "coding-fast")
    assert calls and "reasoning_effort" not in calls[0], f"非预设模型首发不该带关 thinking: {calls}"
