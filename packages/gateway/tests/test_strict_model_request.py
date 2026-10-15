"""Strict requests cross the public parser and one real local provider HTTP hop."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from llm_gateway import events, openai_proxy, ssot_loader
from llm_gateway.gateway import GatewayConfig, ModelGateway
from llm_gateway.provider import LLMRequest
from llm_gateway.providers.openai_provider import OpenAIProvider
from llm_gateway.registry import ModelRegistry
from llm_gateway.retry import RetryConfig
from llm_gateway.ssot_loader import SSOTProviderAdapter
from openai import APIStatusError

MODEL = "ENG-LMSTUDIO-TEST/exact-model"
ALTERNATE = "ENG-LMSTUDIO-TEST/alternate-model"


async def _serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _exercise_http(
    monkeypatch: pytest.MonkeyPatch,
    provider_reply: Callable[[web.Request], web.StreamResponse],
    body: dict,
    *,
    mode: str = "legacy",
    engine: str = "ENG-LMSTUDIO-TEST",
    request_defaults: dict | None = None,
    registry_retries: bool = False,
    unsupported_adapter: bool = False,
) -> tuple[int, dict, list[dict]]:
    provider_posts: list[dict] = []

    async def generation(request: web.Request) -> web.StreamResponse:
        provider_posts.append(await request.json())
        reply = provider_reply(request)
        return await reply if asyncio.iscoroutine(reply) else reply

    provider_app = web.Application()
    provider_app.router.add_post("/v1/chat/completions", generation)
    provider_runner, provider_url = await _serve(provider_app)
    facade_runner = None
    try:
        monkeypatch.setattr(ssot_loader, "_get_credentials_for", lambda _name: None)
        adapter = SSOTProviderAdapter(
            {
                "id": engine,
                "engine_type": "local_daemon",
                "base_url": f"{provider_url}/v1",
                "supported_protocols": ["openai"],
                "request_defaults": request_defaults if request_defaults is not None else {"reasoning_effort": "none"},
            },
            model_defs=[{"model_id": "exact-model"}, {"model_id": "alternate-model"}],
        )
        registry = ModelRegistry()
        registry.register(adapter)
        await registry.refresh()
        if registry_retries:
            registry.retry_config = RetryConfig(max_retries=3, base_delay_ms=1)
        if unsupported_adapter:
            adapter._provider_type = "unsupported"
        config = GatewayConfig(
            omlxc_mode=mode,
            model_ports={},
            model_sizes={},
            aliases={},
            fallback_chain=["alternate-model"],
            complexity_chains={"simple": ["alternate-model"], "complex": ["alternate-model"]},
            background_tasks_enabled=False,
        )
        gateway = ModelGateway(registry, SimpleNamespace(select_model=AsyncMock(return_value=None)), config)
        monkeypatch.setattr(openai_proxy, "get_gateway", lambda: gateway)
        facade_app = web.Application()
        facade_app.router.add_post("/v1/chat/completions", openai_proxy.handle_chat_completions)
        facade_runner, facade_url = await _serve(facade_app)
        async with aiohttp.ClientSession() as client:
            async with client.post(f"{facade_url}/v1/chat/completions", json=body) as response:
                return response.status, await response.json(), provider_posts
    finally:
        if facade_runner is not None:
            await facade_runner.cleanup()
        await provider_runner.cleanup()


def _completion(*, content: str, finish_reason: str, model: str = "exact-model") -> web.Response:
    return web.json_response(
        {
            "id": "fake-provider-1",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 8, "total_tokens": 10},
        }
    )


def _strict_body(**overrides: object) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply briefly."}],
        "strict_model": True,
        "max_tokens": 8,
        "timeout": 4,
        "routing_mode": "local",
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_strict_length_reply_never_retries_or_lifts_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch,
        lambda _request: _completion(content="", finish_reason="length"),
        _strict_body(),
    )
    assert status >= 400, payload
    assert len(posts) == 1, posts
    assert [post["model"] for post in posts] == ["exact-model"]
    assert all(post["max_tokens"] == 8 for post in posts)


@pytest.mark.asyncio
async def test_strict_provider_5xx_is_one_http_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch,
        lambda _request: web.json_response({"error": "unavailable"}, status=503),
        _strict_body(),
        registry_retries=True,
    )
    assert status >= 400, payload
    assert len(posts) == 1, posts
    assert posts[0]["model"] == "exact-model"
    assert posts[0]["max_tokens"] == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_status", [307, 308])
async def test_strict_redirect_is_one_http_attempt(
    monkeypatch: pytest.MonkeyPatch, redirect_status: int
) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch,
        lambda _request: web.Response(
            status=redirect_status, headers={"Location": "/v1/chat/completions"}
        ),
        _strict_body(),
        registry_retries=True,
    )
    assert status >= 400, payload
    assert len(posts) == 1, posts


@pytest.mark.asyncio
async def test_strict_redirect_does_not_change_shared_clients_default_behavior() -> None:
    posts: list[dict] = []

    async def generation(request: web.Request) -> web.Response:
        posts.append(await request.json())
        if len(posts) in (1, 2):
            return web.Response(status=307, headers={"Location": "/v1/chat/completions"})
        return _completion(content="okay", finish_reason="stop")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", generation)
    runner, url = await _serve(app)
    try:
        provider = OpenAIProvider(api_key="test-only", base_url=f"{url}/v1")
        strict_request = LLMRequest(
            prompt="Reply briefly.", model="exact-model", max_tokens=8,
            metadata={"strict_model": True, "strict_timeout": 4},
        )
        with pytest.raises(APIStatusError):
            await provider.generate(strict_request)
        assert len(posts) == 1
        default_reply = await provider.generate(
            LLMRequest(prompt="Reply briefly.", model="exact-model", max_tokens=8)
        )
        assert default_reply.content == "okay"
        assert len(posts) == 3
        assert provider._get_async_client()._client.follow_redirects is True
    finally:
        await provider._get_async_client().close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "finish_reason", "expected_status"),
    [("partial answer", "length", 503), ("", "stop", 503), ("okay", "stop", 200)],
)
async def test_strict_records_returned_usage_once_on_failure_and_success(
    monkeypatch: pytest.MonkeyPatch, content: str, finish_reason: str, expected_status: int
) -> None:
    usage_records: list[dict] = []
    monkeypatch.setattr(ModelGateway, "_budget_guard", lambda *args: None)
    monkeypatch.setattr(
        ModelGateway, "_record_cloud_usage",
        lambda _self, _model, usage, **_kwargs: usage_records.append(dict(usage or {})),
    )
    monkeypatch.setattr(events, "emit", lambda *args, **kwargs: None)
    status, payload, posts = await _exercise_http(
        monkeypatch,
        lambda _request: _completion(content=content, finish_reason=finish_reason),
        _strict_body(),
    )
    assert status == expected_status, payload
    assert len(posts) == 1
    assert usage_records == [{"prompt_tokens": 2, "completion_tokens": 8}]


@pytest.mark.asyncio
async def test_strict_empty_reply_has_no_alternate(monkeypatch: pytest.MonkeyPatch) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="", finish_reason="stop"), _strict_body()
    )
    assert status >= 400, payload
    assert [post["model"] for post in posts] == ["exact-model"]


@pytest.mark.asyncio
async def test_strict_server_deadline_stops_before_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(_request: web.Request) -> web.Response:
        await asyncio.sleep(0.6)
        return _completion(content="late", finish_reason="stop")

    status, payload, posts = await _exercise_http(monkeypatch, slow, _strict_body(timeout=0.2))
    assert status >= 400, payload
    assert [post["model"] for post in posts] == ["exact-model"]


@pytest.mark.asyncio
async def test_strict_network_failure_is_one_http_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    def disconnect(request: web.Request) -> web.Response:
        request.transport.close()
        return web.Response()

    status, payload, posts = await _exercise_http(monkeypatch, disconnect, _strict_body())
    assert status >= 400, payload
    assert [post["model"] for post in posts] == ["exact-model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"strict_model": "true"},
        {"model": ""},
        {"model": "exact-model"},
        {"model": "ENG-LMSTUDIO-TEST/unknown"},
        {"max_tokens": None},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"timeout": None},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"stream": True},
        {"extra_body": {"max_tokens": 999}},
        {"extra_body": {"max_completion_tokens": 999}},
        {"extra_body": {"timeout": 99}},
        {"extra_body": {"strict_model": False}},
        {"max_output_tokens": 999},
    ],
)
async def test_strict_invalid_input_never_generates(monkeypatch: pytest.MonkeyPatch, overrides: dict) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="unexpected", finish_reason="stop"),
        _strict_body(**overrides),
    )
    assert status >= 400, payload
    assert posts == []


@pytest.mark.asyncio
async def test_strict_requires_max_tokens_even_when_alias_is_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _strict_body(max_completion_tokens=8)
    body.pop("max_tokens")
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="unexpected", finish_reason="stop"), body
    )
    assert status >= 400, payload
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "shadow", "active"])
async def test_strict_mode_boundary(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="okay", finish_reason="stop"),
        _strict_body(), mode=mode,
    )
    if mode == "active":
        assert status >= 400, payload
        assert posts == []
    else:
        assert status == 200, payload
        assert payload["model"] == MODEL
        assert [post["model"] for post in posts] == ["exact-model"]


@pytest.mark.asyncio
async def test_strict_k1_cloud_target_rejected_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="unexpected", finish_reason="stop"),
        _strict_body(model="ENG-CLOUD-TEST/exact-model", routing_mode="hybrid", content_title="公文"),
        engine="ENG-CLOUD-TEST",
    )
    assert status >= 400, payload
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_mode", ["local", "cloud", "invalid"])
async def test_strict_routing_boundary(monkeypatch: pytest.MonkeyPatch, routing_mode: str) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="unexpected", finish_reason="stop"),
        _strict_body(routing_mode=routing_mode, model="ENG-CLOUD-TEST/exact-model"),
        engine="ENG-CLOUD-TEST",
    )
    if routing_mode == "cloud":
        assert status == 200, payload
        assert [post["model"] for post in posts] == ["exact-model"]
    else:
        assert status >= 400, payload
        assert posts == []


@pytest.mark.asyncio
async def test_strict_unsupported_adapter_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="unexpected", finish_reason="stop"),
        _strict_body(), unsupported_adapter=True,
    )
    assert status >= 400, payload
    assert posts == []


@pytest.mark.asyncio
async def test_strict_rejects_conflicting_request_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="unexpected", finish_reason="stop"),
        _strict_body(), request_defaults={"max_tokens": 999},
    )
    assert status >= 400, payload
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, False])
async def test_omitted_or_false_retains_legacy_fallback(monkeypatch: pytest.MonkeyPatch, flag: bool | None) -> None:
    body = _strict_body()
    if flag is None:
        body.pop("strict_model")
    else:
        body["strict_model"] = flag
    status, payload, posts = await _exercise_http(
        monkeypatch, lambda _request: _completion(content="", finish_reason="length"), body
    )
    assert status >= 400, payload
    assert len(posts) > 1
    assert any(post["model"] == "alternate-model" for post in posts)
