"""Local compute-hub routing: oMLX App → LM Link → Ollama, with hard deadline."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from llm_gateway.gateway import GatewayConfig, GatewayRequest, ModelGateway
from llm_gateway.ssot_loader import SSOTProviderAdapter
from llm_gateway.types import ChatOptions, ChatResult


def _registry(*model_ids: str):
    reg = MagicMock()
    descriptors = [SimpleNamespace(id=model_id, name=model_id.split("/", 1)[-1]) for model_id in model_ids]
    by_id = {m.id: m for m in descriptors}
    reg.list_models.return_value = descriptors
    reg.get.side_effect = by_id.get
    reg.get_provider.return_value = SimpleNamespace(name="test", base_url="http://127.0.0.1:1/v1")
    return reg


def _config(**overrides):
    values = dict(
        model_ports={"coding-fast": 8081},
        model_sizes={},
        lmstudio_fallback={"coding-fast": "qwen/qwen3.5-9b"},
        ollama_fallback={"coding-fast": "qwen3.5:9b"},
        local_backend="app",
        fallback_chain=["coding-fast"],
        complexity_chains={"simple": ["coding-fast"], "medium": ["coding-fast"], "complex": ["coding-fast"]},
        background_tasks_enabled=False,
    )
    values.update(overrides)
    return GatewayConfig(**values)


def test_app_then_lm_link_then_ollama_and_never_spawns_legacy_server():
    reg = _registry(
        "ENG-OMLX-LOCAL/coding-fast",
        "ENG-LMSTUDIO-MACMINI/qwen/qwen3.5-9b",
        "ENG-OLLAMA-MACMINI/qwen3.5:9b",
    )
    calls = []

    async def chat(model_id, messages, options=None):
        calls.append(model_id)
        if model_id.startswith("ENG-OLLAMA-"):
            return ChatResult(content="ollama-ok", finish_reason="stop")
        raise RuntimeError("backend down")

    reg.chat = chat
    gw = ModelGateway(reg, MagicMock(), _config())
    gw._ensure_model = AsyncMock(return_value=True)  # type: ignore[method-assign]

    response = asyncio.run(
        gw.generate(
            GatewayRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="coding-fast",
                timeout=1,
            )
        )
    )

    assert calls == [
        "ENG-OMLX-LOCAL/coding-fast",
        "ENG-LMSTUDIO-MACMINI/qwen/qwen3.5-9b",
        "ENG-OLLAMA-MACMINI/qwen3.5:9b",
    ]
    assert response.content == "ollama-ok"
    gw._ensure_model.assert_not_awaited()


def test_deadline_covers_the_whole_fallback_chain():
    reg = _registry("ENG-OMLX-LOCAL/coding-fast")

    async def slow_chat(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return ChatResult(content="too late")

    reg.chat = slow_chat
    gw = ModelGateway(reg, MagicMock(), _config(lmstudio_fallback={}, ollama_fallback={}))
    started = time.monotonic()
    response = asyncio.run(
        gw.generate(
            GatewayRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="coding-fast",
                timeout=0.03,
            )
        )
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert "deadline" in response.error.lower()


def test_cloud_is_opt_in_not_scheduler_accident():
    reg = _registry("ENG-CLOUD/cloud-model")
    reg.chat = AsyncMock(return_value=ChatResult(content="cloud"))
    gw = ModelGateway(
        reg,
        MagicMock(),
        _config(model_ports={}, fallback_chain=[], complexity_chains={}),
    )

    local = asyncio.run(
        gw.generate(
            GatewayRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="cloud-model",
                routing_mode="local",
            )
        )
    )
    hybrid = asyncio.run(
        gw.generate(
            GatewayRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="cloud-model",
                routing_mode="hybrid",
            )
        )
    )

    assert local.content == ""
    assert hybrid.content == "cloud"
    assert reg.chat.await_count == 1


def test_engine_request_defaults_are_merged_and_caller_wins():
    adapter = SSOTProviderAdapter(
        {
            "id": "ENG-OMLX-LOCAL",
            "engine_type": "local_daemon",
            "base_url": "http://127.0.0.1:8000/v1",
            "supported_protocols": ["openai"],
            "request_defaults": {
                "thinking_budget": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    )
    request = adapter._build_request(
        "ENG-OMLX-LOCAL/coding-fast",
        [{"role": "user", "content": "hi"}],
        ChatOptions(extra={"thinking_budget": 16}),
    )

    assert request.extra == {
        "thinking_budget": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_sync_provider_discovery_runs_concurrently_off_event_loop():
    def adapter(name):
        item = SSOTProviderAdapter(
            {
                "id": name,
                "engine_type": "local_server",
                "base_url": "http://127.0.0.1:1/v1",
                "supported_protocols": ["openai"],
            }
        )

        def slow_models():
            time.sleep(0.08)
            return ["m"]

        item._underlying.available_models = slow_models
        return item

    first, second = adapter("ENG-A"), adapter("ENG-B")

    async def run_both():
        return await asyncio.gather(first.discover(), second.discover())

    started = time.monotonic()
    results = asyncio.run(run_both())
    elapsed = time.monotonic() - started

    assert [models[0].name for models in results] == ["m", "m"]
    assert elapsed < 0.14, f"同步 discovery 串行阻塞了事件循环: {elapsed:.3f}s"


def test_embedding_uses_app_then_ollama_in_app_mode(monkeypatch):
    reg = _registry(
        "ENG-OMLX-LOCAL/embedding",
        "ENG-OLLAMA-MACBOOKPRO/nomic-embed-text:latest",
    )
    reg.get_provider.side_effect = lambda model_id: SimpleNamespace(
        name=model_id.partition("/")[0],
        base_url=("http://app/v1" if model_id.startswith("ENG-OMLX") else "http://ollama/v1"),
    )
    config = _config(
        model_ports={"embedding": 8183},
        lmstudio_fallback={},
        ollama_fallback={"embedding": "nomic-embed-text:latest"},
    )
    gw = ModelGateway(reg, MagicMock(), config)
    urls = []

    def response(status, data=None):
        item = AsyncMock()
        item.status = status
        item.json = AsyncMock(return_value=data or {})
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=item)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    session = AsyncMock()

    def post(url, **_kwargs):
        urls.append(url)
        if url.startswith("http://app"):
            return response(503)
        return response(200, {"data": [{"embedding": [0.1, 0.2]}]})

    session.post = post
    client = MagicMock()
    client.return_value.__aenter__ = AsyncMock(return_value=session)
    client.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("aiohttp.ClientSession", client)

    vectors = asyncio.run(gw.embed(["hello"], timeout=1))
    assert vectors == [[0.1, 0.2]]
    assert urls == ["http://app/v1/embeddings", "http://ollama/v1/embeddings"]


def test_embedding_unix_socket_engine_uses_omlxc_client(monkeypatch):
    """ENG-OMLX-LOCAL 的 base_url 是 unix://omlxc/api/v1: 必须走 omlxc 客户端, 不能交给 aiohttp。"""
    reg = _registry("ENG-OMLX-LOCAL/embedding", "ENG-OLLAMA-MACBOOKPRO/nomic-embed-text:latest")
    reg.get_provider.side_effect = lambda model_id: SimpleNamespace(
        name=model_id.partition("/")[0],
        base_url=("unix://omlxc/api/v1" if model_id.startswith("ENG-OMLX") else "http://ollama/v1"),
    )
    config = _config(
        model_ports={"embedding": 8183},
        lmstudio_fallback={},
        ollama_fallback={"embedding": "nomic-embed-text:latest"},
    )
    omlxc = MagicMock()
    omlxc.embed = AsyncMock(return_value=[[0.3, 0.4]])
    gw = ModelGateway(reg, MagicMock(), config, omlxc_client=omlxc)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(side_effect=AssertionError("aiohttp must not see unix://")))

    assert asyncio.run(gw.embed(["hello"], timeout=1)) == [[0.3, 0.4]]
    omlxc.embed.assert_awaited_once()
    assert omlxc.embed.await_args.kwargs["model"] == "embedding"


def test_embedding_is_recorded_in_usage_metrics(monkeypatch):
    """/stats 此前对 embeddings 失明: 成功与失败都要记进 metrics。"""
    reg = _registry("ENG-OMLX-LOCAL/embedding")
    reg.get_provider.side_effect = lambda model_id: SimpleNamespace(
        name="ENG-OMLX-LOCAL", base_url="unix://omlxc/api/v1"
    )
    config = _config(model_ports={"embedding": 8183}, lmstudio_fallback={}, ollama_fallback={})
    omlxc = MagicMock()
    omlxc.embed = AsyncMock(return_value=[[0.3, 0.4]])
    gw = ModelGateway(reg, MagicMock(), config, omlxc_client=omlxc)

    asyncio.run(gw.embed(["hello"], timeout=1))
    models = gw._metrics.report()["models"]
    assert models["embedding"]["requests"] == 1
