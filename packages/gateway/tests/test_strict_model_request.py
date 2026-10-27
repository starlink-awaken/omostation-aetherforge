"""Strict requests cross the public parser and one real local provider HTTP hop."""

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from aiohttp import web
from llm_gateway import events, omlxc_client, openai_proxy, ssot_loader, tracing
from llm_gateway.gateway import GatewayConfig, GatewayRequest, ModelGateway
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


@pytest.fixture
def offline_openai(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Exercise the real lazy getter with fake SDK/HTTP boundaries; never open a socket."""
    import httpx
    import openai

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = SimpleNamespace(getter=0, socket=0, transport=0, sdk=0, payloads=[], options=[], http=[])

    async def create(**payload: object) -> SimpleNamespace:
        calls.payloads.append(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="okay", tool_calls=[]), finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )

    def with_options(**options: object) -> SimpleNamespace:
        calls.options.append(options)
        return client

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), with_options=with_options,
    )

    def sdk(**_kwargs: object) -> SimpleNamespace:
        calls.sdk += 1
        return client

    def socket_path() -> str:
        calls.socket += 1
        return "/offline-only/unused.sock"

    def transport(**_kwargs: object) -> object:
        calls.transport += 1
        return object()

    def http_client(**kwargs: object) -> SimpleNamespace:
        result = SimpleNamespace(**kwargs, aclose=AsyncMock())
        calls.http.append(result)
        return result

    getter = OpenAIProvider._get_async_client

    def counted_getter(provider: OpenAIProvider) -> object:
        calls.getter += 1
        return getter(provider)

    monkeypatch.setattr(OpenAIProvider, "_get_async_client", counted_getter)
    monkeypatch.setattr(omlxc_client, "default_omlxc_socket", socket_path)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncClient", http_client)
    monkeypatch.setattr(openai, "AsyncOpenAI", sdk)
    return calls


def _assert_no_provider_io(calls: SimpleNamespace) -> None:
    assert (calls.getter, calls.socket, calls.transport, calls.sdk) == (0, 0, 0, 0)
    assert calls.payloads == []
    assert calls.options == []
    assert calls.http == []


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [None, {"strict_model": False}, {"max_tokens": 999}, {"timeout": 99}])
async def test_offline_strict_unix_rejected_before_client(
    offline_openai: SimpleNamespace, extra: dict | None, caplog: pytest.LogCaptureFixture,
) -> None:
    provider = OpenAIProvider(api_key="offline-key-marker", base_url="unix://private-url-marker/api/v1")
    request = LLMRequest(
        prompt="private-prompt-marker", model="exact-model", max_tokens=8,
        metadata={"strict_model": True, "strict_timeout": 4}, extra=extra,
    )
    with pytest.raises(ValueError, match="^strict_model unsupported for unix transport$") as error:
        await provider.generate(request)
    _assert_no_provider_io(offline_openai)
    assert provider._async_client is None
    for marker in ("offline-key-marker", "private-url-marker", "private-prompt-marker"):
        assert marker not in str(error.value)
        assert marker not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, False])
async def test_offline_nonstrict_unix_payload_unchanged(offline_openai: SimpleNamespace, flag: bool | None) -> None:
    provider = OpenAIProvider(base_url="unix://private-url-marker/api/v1")
    request = LLMRequest(
        prompt="Reply briefly.", system_prompt="Be brief.", model="exact-model", max_tokens=8,
        temperature=0.2, stop_sequences=["END"], context=[{"role": "assistant", "content": "Hi"}],
        metadata={} if flag is None else {"strict_model": flag}, extra={"reasoning_effort": "none"},
    )
    response = await provider.generate(request)
    assert response.content == "okay"
    assert (offline_openai.getter, offline_openai.socket, offline_openai.transport, offline_openai.sdk) == (1, 1, 1, 1)
    assert offline_openai.options == []
    assert offline_openai.payloads == [{
        "model": "exact-model",
        "messages": [
            {"role": "system", "content": "Be brief."}, {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Reply briefly."},
        ],
        "max_tokens": 8, "temperature": 0.2, "stop": ["END"],
        "extra_body": {"reasoning_effort": "none"},
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "https"])
async def test_offline_direct_http_strict_controls_unchanged(offline_openai: SimpleNamespace, scheme: str) -> None:
    provider = OpenAIProvider(api_key="offline-only", base_url=f"{scheme}://offline.invalid/v1")
    response = await provider.generate(LLMRequest(
        prompt="Reply briefly.", model="exact-model", max_tokens=8,
        metadata={"strict_model": True, "strict_timeout": 4}, extra={"reasoning_effort": "none"},
    ))
    assert response.content == "okay"
    assert (offline_openai.getter, offline_openai.socket, offline_openai.transport, offline_openai.sdk) == (1, 0, 0, 1)
    assert len(offline_openai.payloads) == 1
    assert offline_openai.payloads[0]["model"] == "exact-model"
    assert offline_openai.payloads[0]["max_tokens"] == 8
    assert offline_openai.options == [{"max_retries": 0, "timeout": 4, "http_client": offline_openai.http[0]}]
    assert offline_openai.http[0].follow_redirects is False
    offline_openai.http[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "shadow", "active"])
@pytest.mark.parametrize("defaults", [{"reasoning_effort": "none"}, {"strict_model": False}, {"max_tokens": 999}, {"timeout": 99}])
@pytest.mark.parametrize("extra", [{}, {"strict_model": False}, {"max_tokens": 999}, {"timeout": 99}])
async def test_offline_strict_unix_gateway_terminal(
    monkeypatch: pytest.MonkeyPatch, offline_openai: SimpleNamespace, mode: str, defaults: dict, extra: dict,
) -> None:
    monkeypatch.setattr(ssot_loader, "_get_credentials_for", lambda _name: None)
    monkeypatch.setattr(tracing, "trace_llm_call", lambda *_args, **_kwargs: nullcontext(None))
    monkeypatch.setattr(events, "emit", lambda *_args, **_kwargs: None)
    adapter = SSOTProviderAdapter({
        "id": "ENG-OMLX-OFFLINE", "engine_type": "local_daemon", "base_url": "unix://offline/api/v1",
        "supported_protocols": ["openai"], "request_defaults": defaults,
    }, model_defs=[{"model_id": "exact-model"}, {"model_id": "alternate-model"}])
    provider_generate = AsyncMock(wraps=adapter._underlying.generate)
    monkeypatch.setattr(adapter._underlying, "generate", provider_generate)
    adapter_chat = AsyncMock(wraps=adapter.chat)
    monkeypatch.setattr(adapter, "chat", adapter_chat)
    registry = ModelRegistry()
    registry.register(adapter)
    await registry.refresh()
    registry.retry_config = RetryConfig(max_retries=3, base_delay_ms=1)
    scheduler = SimpleNamespace(select_model=AsyncMock(side_effect=AssertionError("scheduler must not run")))
    daemon = SimpleNamespace(
        plan=AsyncMock(side_effect=AssertionError("daemon must not run")),
        infer=AsyncMock(side_effect=AssertionError("daemon must not run")),
    )
    gateway = ModelGateway(registry, scheduler, GatewayConfig(
        omlxc_mode=mode, model_ports={}, model_sizes={}, aliases={},
        fallback_chain=["alternate-model"],
        complexity_chains={"simple": ["alternate-model"], "complex": ["alternate-model"]},
        background_tasks_enabled=False,
    ), omlxc_client=daemon)
    monkeypatch.setattr(gateway, "_budget_guard", lambda *_args: None)
    usage = Mock()
    monkeypatch.setattr(gateway, "_record_cloud_usage", usage)
    response = await gateway.generate(GatewayRequest(
        messages=[{"role": "user", "content": "Reply briefly."}], model="ENG-OMLX-OFFLINE/exact-model",
        strict_model=True, max_tokens=8, timeout=4, routing_mode="local", extra=extra,
    ))
    assert response.content == ""
    assert response.finish_reason == "error"
    assert response.error
    if extra:
        assert response.error == "Invalid strict_model bounds or model"
    elif mode == "active":
        assert response.error == "strict_model unsupported in active omlxcd mode"
    elif defaults == {"reasoning_effort": "none"}:
        assert response.error == "strict_model unsupported for unix transport"
        provider_generate.assert_awaited_once()
        request = provider_generate.call_args.args[0]
        assert request.metadata == {"strict_model": True, "strict_timeout": 4}
        assert request.model == "exact-model" and request.max_tokens == 8
    else:
        assert "forbids provider defaults" in response.error
    assert provider_generate.await_count == (mode != "active" and not extra and defaults == {"reasoning_effort": "none"})
    assert adapter_chat.await_count == (mode != "active" and not extra)
    assert all(call.args[0] == "ENG-OMLX-OFFLINE/exact-model" for call in adapter_chat.call_args_list)
    scheduler.select_model.assert_not_awaited()
    daemon.plan.assert_not_awaited()
    daemon.infer.assert_not_awaited()
    usage.assert_not_called()
    _assert_no_provider_io(offline_openai)
