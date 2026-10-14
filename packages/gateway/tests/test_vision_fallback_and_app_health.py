"""视觉请求只在视觉档兜底; App 模式 health() 不探测已下线的每模型端口。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from llm_gateway.gateway import GatewayConfig, GatewayRequest, GatewayResponse, ModelGateway, _has_image


def _config(**overrides) -> GatewayConfig:
    base = GatewayConfig(
        omlx_bin="/bin/true",
        local_base_url="http://127.0.0.1",
        model_ports={},
        model_sizes={},
        lmstudio_fallback={},
        ollama_fallback={},
        local_backend="legacy",
        fallback_chain=["coding", "reasoning"],
        complexity_chains={"simple": ["coding"], "complex": ["reasoning"]},
        vision_fallback_chain=["vision", "minicpm"],
        omlxc_mode="legacy",
    )
    return replace(base, **overrides)


def _gateway(config: GatewayConfig) -> ModelGateway:
    reg = MagicMock()
    reg.get_all.return_value = []
    reg.list_models.return_value = []
    sched = MagicMock()
    sched.select_model.side_effect = RuntimeError("no scheduler in test")
    return ModelGateway(reg, sched, config)


IMAGE_MSG = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:,"}}],
    }
]


def test_has_image_detects_multimodal_parts() -> None:
    assert _has_image(IMAGE_MSG)
    assert not _has_image([{"role": "user", "content": "plain text"}])


def test_image_request_falls_back_only_within_vision_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway(_config())
    tried: list[str] = []

    async def fake_try(model_name: str, request: GatewayRequest) -> GatewayResponse:
        tried.append(model_name)
        raise RuntimeError("down")

    monkeypatch.setattr(gw, "_try_generate", fake_try)
    monkeypatch.setattr(gw, "_routing_allows_model", lambda *_a: True)

    asyncio.run(gw.generate(GatewayRequest(messages=IMAGE_MSG, model="vision-think", timeout=5)))

    assert tried == ["vision-think", "vision", "minicpm"]  # 从不落到纯文本的 coding/reasoning


def test_text_request_keeps_general_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway(_config())
    tried: list[str] = []

    async def fake_try(model_name: str, request: GatewayRequest) -> GatewayResponse:
        tried.append(model_name)
        raise RuntimeError("down")

    monkeypatch.setattr(gw, "_try_generate", fake_try)
    monkeypatch.setattr(gw, "_routing_allows_model", lambda *_a: True)

    asyncio.run(gw.generate(GatewayRequest(messages=[{"role": "user", "content": "hi"}], model="x", timeout=5)))

    assert tried[0] == "x" and "coding" in tried


def test_app_mode_health_does_not_probe_ports() -> None:
    gw = _gateway(_config(local_backend="app", model_ports={"mythos-fast": 8185}))
    assert asyncio.run(gw.health()) == {}
