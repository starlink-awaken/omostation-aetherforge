from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from llm_gateway.gateway import GatewayConfig, GatewayRequest, GatewayResponse, ModelGateway
from llm_gateway.omlxc_client import (
    OmlxcChatResult,
    OmlxcError,
    OmlxcErrorCode,
    OmlxcRoutePlan,
    TokenUsage,
)


class FakeRegistry:
    def list_models(self) -> list[object]:
        return [object()]


class FakeScheduler:
    async def select_model(self, _request: object) -> None:
        return None


class FakeOmlxc:
    def __init__(self, error: OmlxcError | None = None) -> None:
        self.error = error
        self.plan_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    async def route_plan(self, model_id: str, **kwargs: Any) -> OmlxcRoutePlan:
        self.plan_calls.append({"model_id": model_id, **kwargs})
        if self.error:
            raise self.error
        return OmlxcRoutePlan(
            request_id="route-1",
            selected="placement-a",
            candidates=("placement-a",),
            scores={"placement-a": 1.0},
            rejected={},
            fallback=("placement-a",),
            config_version="v1",
            explanation="selected",
        )

    async def chat(self, **kwargs: Any) -> OmlxcChatResult:
        self.chat_calls.append(kwargs)
        if self.error:
            raise self.error
        return OmlxcChatResult(
            content="omlxc answer",
            model=kwargs["model"],
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            request_id="infer-1",
            placement="placement-a",
            backend="backend-a",
        )

    async def embed(self, **kwargs: Any) -> list[list[float]]:
        self.embed_calls.append(kwargs)
        if self.error:
            raise self.error
        return [[1.0, 2.0] for _ in kwargs["inputs"]]


def _gateway(mode: str, client: FakeOmlxc) -> ModelGateway:
    config = GatewayConfig(
        omlxc_mode=mode,
        aliases={"logical": "local/resolved"},
        model_ports={},
        model_sizes={},
        lmstudio_fallback={},
        ollama_fallback={},
        fallback_chain=["logical"],
        complexity_chains={"simple": ["logical"], "complex": ["logical"]},
        background_tasks_enabled=False,
    )
    return ModelGateway(FakeRegistry(), FakeScheduler(), config, omlxc_client=client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_legacy_never_calls_omlxc(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOmlxc()
    gateway = _gateway("legacy", client)
    calls = 0

    async def legacy(_request: GatewayRequest) -> GatewayResponse:
        nonlocal calls
        calls += 1
        return GatewayResponse("legacy answer", "logical", 1)

    monkeypatch.setattr(gateway, "_generate_legacy", legacy)
    result = await gateway.generate(GatewayRequest(messages=[{"role": "user", "content": "hi"}], model="logical"))
    assert result.content == "legacy answer"
    assert calls == 1
    assert not client.plan_calls and not client.chat_calls


@pytest.mark.asyncio
async def test_shadow_plans_once_and_infers_only_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOmlxc()
    gateway = _gateway("shadow", client)
    calls = 0

    async def legacy(_request: GatewayRequest) -> GatewayResponse:
        nonlocal calls
        calls += 1
        return GatewayResponse("legacy answer", "legacy-model", 1, provider="legacy-provider")

    monkeypatch.setattr(gateway, "_generate_legacy", legacy)
    result = await gateway.generate(GatewayRequest(messages=[{"role": "user", "content": "hi"}], model="logical"))
    assert result.content == "legacy answer"
    assert calls == 1
    assert [call["model_id"] for call in client.plan_calls] == ["local/resolved"]
    assert not client.chat_calls


@pytest.mark.asyncio
async def test_active_local_resolves_alias_once_and_uses_only_omlxc(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOmlxc()
    gateway = _gateway("active", client)
    alias_calls = 0

    original = gateway.resolve_alias

    def resolve_once(name: str) -> str:
        nonlocal alias_calls
        alias_calls += 1
        return original(name)

    monkeypatch.setattr(gateway, "resolve_alias", resolve_once)
    for forbidden in ("_generate_legacy", "_ensure_model", "_port_reachable"):
        monkeypatch.setattr(gateway, forbidden, lambda *_a, _name=forbidden, **_k: pytest.fail(_name))

    result = await gateway.generate(GatewayRequest(messages=[{"role": "user", "content": "hi"}], model="logical"))
    assert result.content == "omlxc answer"
    assert result.provider == "omlxc:backend-a"
    assert alias_calls == 1
    assert client.chat_calls[0]["model"] == "local/resolved"
    assert client.chat_calls[0]["thinking"] is False


@pytest.mark.asyncio
async def test_cloud_never_calls_omlxc(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOmlxc()
    gateway = _gateway("active", client)

    async def legacy(request: GatewayRequest) -> GatewayResponse:
        assert request.routing_mode == "cloud"
        return GatewayResponse("cloud answer", "cloud/model", 1, provider="cloud")

    monkeypatch.setattr(gateway, "_generate_legacy", legacy)
    result = await gateway.generate(GatewayRequest(messages=[{"role": "user", "content": "hi"}], routing_mode="cloud"))
    assert result.content == "cloud answer"
    assert not client.plan_calls and not client.chat_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [OmlxcErrorCode.UNAVAILABLE, OmlxcErrorCode.NO_CAPACITY, OmlxcErrorCode.TIMEOUT],
)
async def test_hybrid_falls_back_to_cloud_only_for_typed_local_failure(
    code: OmlxcErrorCode, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOmlxc(OmlxcError(code))
    gateway = _gateway("active", client)
    cloud_calls = 0

    async def legacy(request: GatewayRequest) -> GatewayResponse:
        nonlocal cloud_calls
        cloud_calls += 1
        assert request.routing_mode == "cloud"
        return GatewayResponse("cloud answer", "cloud/model", 1, provider="cloud")

    monkeypatch.setattr(gateway, "_generate_legacy", legacy)
    result = await gateway.generate(
        GatewayRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="logical",
            routing_mode="hybrid",
        )
    )
    assert result.content == "cloud answer"
    assert len(client.chat_calls) == 1
    assert cloud_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [OmlxcErrorCode.SECURITY, OmlxcErrorCode.INVALID, OmlxcErrorCode.INTERNAL])
async def test_hybrid_fails_closed_for_non_fallbackable_error(
    code: OmlxcErrorCode, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOmlxc(OmlxcError(code))
    gateway = _gateway("active", client)
    monkeypatch.setattr(gateway, "_generate_legacy", lambda *_a, **_k: pytest.fail("cloud"))
    result = await gateway.generate(
        GatewayRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="logical",
            routing_mode="hybrid",
        )
    )
    assert not result.content
    assert "local inference failed" in result.error


@pytest.mark.asyncio
async def test_k1_never_falls_back_to_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOmlxc(OmlxcError(OmlxcErrorCode.UNAVAILABLE))
    gateway = _gateway("active", client)
    monkeypatch.setattr(gateway, "_generate_legacy", lambda *_a, **_k: pytest.fail("cloud"))
    result = await gateway.generate(
        GatewayRequest(
            messages=[{"role": "user", "content": "private"}],
            model="logical",
            routing_mode="hybrid",
            content_title="公文",
        )
    )
    assert not result.content
    assert result.error == "[K1] local inference unavailable"


def test_invalid_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="AETHERFORGE_OMLXC_MODE"):
        replace(GatewayConfig(), omlxc_mode="typo")


@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_nonlegacy_default_config_never_reads_legacy_models_json(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHERFORGE_OMLXC_MODE", mode)
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: pytest.fail("legacy JSON read"))
    config = GatewayConfig()
    assert config.omlxc_mode == mode


@pytest.mark.asyncio
async def test_active_embed_uses_omlxc_without_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOmlxc()
    gateway = _gateway("active", client)
    monkeypatch.setattr(gateway, "_embed_legacy", lambda *_a, **_k: pytest.fail("legacy embed"))
    vectors = await gateway.embed(["a", "b"], model="logical", timeout=2)
    assert vectors == [[1.0, 2.0], [1.0, 2.0]]
    assert client.embed_calls[0]["model"] == "local/resolved"
