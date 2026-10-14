"""AetherForge facade parameter fidelity and BOS infer bridge."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from llm_gateway import openai_proxy
from llm_gateway.gateway import GatewayConfig, GatewayResponse, ModelGateway
from llm_gateway.omlxc_client import OmlxcError, OmlxcErrorCode
from openai.types import Model


def test_proxy_preserves_engine_specific_fields(monkeypatch):
    captured = {}

    class Gateway:
        _registry = SimpleNamespace(list_models=lambda: [object()])

        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    request = SimpleNamespace(
        json=lambda: None,
    )

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "ping"}}],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
            "routing_mode": "local",
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(openai_proxy.handle_chat_completions(request))
    gateway_request = captured["request"]

    assert response.status == 200
    assert gateway_request.extra["tools"][0]["function"]["name"] == "ping"
    assert gateway_request.extra["response_format"] == {"type": "json_object"}
    assert gateway_request.extra["reasoning_effort"] == "none"


def test_active_model_directory_only_exposes_omlxc_logical_models(monkeypatch):
    class Gateway:
        _config = SimpleNamespace(omlxc_mode="active")
        _registry = SimpleNamespace(
            list_models=lambda: [SimpleNamespace(id="legacy-provider/unsafe-choice", provider="legacy-provider")]
        )

        async def _ensure_registry_ready(self):
            raise AssertionError("active model directory must not refresh the legacy registry")

        async def list_omlxc_models(self):
            return [SimpleNamespace(id="coding", capabilities=frozenset({"chat"}))]

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_list_models(SimpleNamespace()))
    payload = json.loads(response.body)

    assert response.status == 200
    assert payload["data"][0]["id"] == "coding"
    assert payload["data"][0]["object"] == "model"
    assert payload["data"][0]["owned_by"] == "omlxc"
    assert isinstance(payload["data"][0]["created"], int)
    assert Model.model_validate(payload["data"][0]).id == "coding"


def test_active_model_directory_scope_all_merges_registry_cloud_models(monkeypatch):
    """?scope=all 显式 opt-in: 合并 registry 云端清单(本地重名优先), 并允许刷新 registry。"""

    class Gateway:
        _config = SimpleNamespace(omlxc_mode="active")
        _registry = SimpleNamespace(
            list_models=lambda: [
                SimpleNamespace(id="coding", provider="omlxc"),
                SimpleNamespace(id="ENG-LONGCAT-CLOUD/LongCat-2.0", provider="ENG-LONGCAT-CLOUD"),
            ]
        )

        async def _ensure_registry_ready(self):
            return None  # scope=all 时允许刷新

        async def list_omlxc_models(self):
            return [SimpleNamespace(id="coding", capabilities=frozenset({"chat"}))]

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_list_models(SimpleNamespace(query={"scope": "all"})))
    payload = json.loads(response.body)

    assert response.status == 200
    ids = [item["id"] for item in payload["data"]]
    assert ids.count("coding") == 1  # 本地重名优先, registry 的 coding 不重复
    assert "ENG-LONGCAT-CLOUD/LongCat-2.0" in ids
    by_id = {item["id"]: item for item in payload["data"]}
    assert by_id["coding"]["owned_by"] == "omlxc"
    assert by_id["ENG-LONGCAT-CLOUD/LongCat-2.0"]["owned_by"] == "ENG-LONGCAT-CLOUD"


def test_active_model_directory_fails_closed_when_omlxc_catalog_is_unavailable(monkeypatch):
    class Gateway:
        _config = SimpleNamespace(omlxc_mode="active")
        _registry = SimpleNamespace(list_models=lambda: [object()])

        async def list_omlxc_models(self):
            raise OmlxcError(OmlxcErrorCode.UNAVAILABLE)

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_list_models(SimpleNamespace()))
    payload = json.loads(response.body)

    assert response.status == 503
    assert payload["error"]["code"] == "unavailable"


def test_non_active_model_directory_preserves_registry_compatibility(monkeypatch):
    class Gateway:
        _config = SimpleNamespace(omlxc_mode="shadow")
        _registry = SimpleNamespace(
            list_models=lambda: [SimpleNamespace(id="legacy-model", provider="legacy-provider")]
        )

        async def _ensure_registry_ready(self):
            return None

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_list_models(SimpleNamespace()))
    payload = json.loads(response.body)

    assert response.status == 200
    assert payload["data"][0]["id"] == "legacy-model"


def test_proxy_streams_openai_chunks_without_buffering(monkeypatch):
    class Gateway:
        async def generate_stream(self, request):
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            assert request.model == "coding"
            yield OmlxcStreamChunk(content="hel", model="coding", request_id="req-1")
            yield OmlxcStreamChunk(
                content="lo",
                model="coding",
                request_id="req-1",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            )

    request = SimpleNamespace(transport=SimpleNamespace(is_closing=lambda: False))

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def collect():
        response = await openai_proxy.handle_chat_completions(request)
        payload = b"".join([chunk async for chunk in response.body._value]).decode()
        return response, payload

    response, payload = asyncio.run(collect())
    assert response.status == 200
    assert payload.endswith("data: [DONE]\n\n")
    assert '"content":"hel"' in payload
    assert '"content":"lo"' in payload
    assert '"total_tokens":3' in payload


def test_proxy_disconnect_cancels_underlying_stream(monkeypatch):
    entered = asyncio.Event()
    closed = False

    class Gateway:
        async def generate_stream(self, _request):
            nonlocal closed
            try:
                entered.set()
                yield SimpleNamespace(
                    content="first",
                    model="coding",
                    request_id="req-1",
                    finish_reason=None,
                    usage=None,
                )
                await asyncio.Event().wait()
            finally:
                closed = True

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def disconnect() -> None:
        response = await openai_proxy.handle_chat_completions(request)
        stream = response.body._value
        await anext(stream)
        await entered.wait()
        await stream.aclose()

    asyncio.run(disconnect())
    assert closed


def test_proxy_embeddings_preserve_routing_and_sensitive_context(monkeypatch):
    captured = {}

    class Gateway:
        async def embed(self, texts, **kwargs):
            captured["texts"] = texts
            captured.update(kwargs)
            return [[1.0, 2.0]]

    request = SimpleNamespace()

    async def body():
        return {
            "model": "embedding",
            "input": "hello",
            "timeout": 4,
            "routing_mode": "cloud",
            "content_title": "public",
            "content_url": "https://example.com",
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(openai_proxy.handle_embeddings(request))
    payload = json.loads(response.body)

    assert captured == {
        "texts": ["hello"],
        "model": "embedding",
        "timeout": 4.0,
        "routing_mode": "cloud",
        "content_title": "public",
        "content_url": "https://example.com",
    }
    assert payload["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        (OmlxcErrorCode.NO_CAPACITY, 409),
        (OmlxcErrorCode.TIMEOUT, 504),
        (OmlxcErrorCode.SECURITY, 403),
        (OmlxcErrorCode.INVALID, 400),
        (OmlxcErrorCode.UNAVAILABLE, 503),
        (OmlxcErrorCode.INTERNAL, 502),
    ],
)
def test_proxy_maps_whitelisted_omlxc_errors_for_nonstream(code, expected_status, monkeypatch):
    class Gateway:
        async def generate(self, _request):
            return SimpleNamespace(
                content="",
                model="coding",
                finish_reason="error",
                tokens_in=0,
                tokens_out=0,
                provider="",
                error="local inference failed",
                error_code=code,
            )

    request = SimpleNamespace()

    async def body():
        return {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_chat_completions(request))
    payload = json.loads(response.body)

    assert response.status == expected_status
    assert payload["error"]["code"] == code.value


def test_proxy_does_not_trust_unknown_downstream_error_status(monkeypatch):
    class Gateway:
        async def generate(self, _request):
            return SimpleNamespace(
                content="",
                model="coding",
                finish_reason="error",
                tokens_in=0,
                tokens_out=0,
                provider="",
                error="private downstream detail",
                error_code="teapot:418",
            )

    request = SimpleNamespace()

    async def body():
        return {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_chat_completions(request))
    payload = json.loads(response.body)

    assert response.status == 502
    assert payload["error"]["code"] == "internal"
    assert "private downstream detail" not in response.body.decode()


def test_proxy_maps_pretoken_stream_error_to_http_status(monkeypatch):
    class Gateway:
        async def generate_stream(self, _request):
            if False:
                yield
            raise OmlxcError(OmlxcErrorCode.NO_CAPACITY)

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_chat_completions(request))
    payload = json.loads(response.body)

    assert response.status == 409
    assert payload["error"]["code"] == "no_capacity"


def test_proxy_preserves_gateway_pretoken_error_when_inner_stream_close_repeats_it(
    monkeypatch,
):
    class Client:
        async def stream_chat(self, **_kwargs):
            if False:
                yield
            raise OmlxcError(OmlxcErrorCode.NO_CAPACITY)

    config = GatewayConfig(
        omlxc_mode="active",
        aliases={},
        model_ports={},
        model_sizes={},
        lmstudio_fallback={},
        ollama_fallback={},
        fallback_chain=["coding"],
        complexity_chains={"simple": ["coding"], "complex": ["coding"]},
        background_tasks_enabled=False,
    )
    gateway = ModelGateway(
        SimpleNamespace(),
        SimpleNamespace(),
        config,
        omlxc_client=Client(),
    )
    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: gateway)

    response = asyncio.run(openai_proxy.handle_chat_completions(request))

    assert response.status == 409
    assert json.loads(response.body)["error"]["code"] == "no_capacity"


@pytest.mark.parametrize(
    "metadata",
    [
        {"usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}},
        {"finish_reason": "stop"},
    ],
    ids=["usage", "finish"],
)
def test_proxy_maps_stream_error_after_metadata_but_before_content_to_http_status(
    metadata,
    monkeypatch,
):
    closed = 0

    class Gateway:
        async def generate_stream(self, _request):
            nonlocal closed
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            try:
                yield OmlxcStreamChunk(
                    content="",
                    model="coding",
                    request_id="req-1",
                    **metadata,
                )
                raise OmlxcError(OmlxcErrorCode.NO_CAPACITY)
            finally:
                closed += 1

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_chat_completions(request))

    assert response.status == 409
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "no_capacity"
    assert closed == 1


def test_proxy_preserves_metadata_chunk_order_before_first_content(monkeypatch):
    class Gateway:
        async def generate_stream(self, _request):
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            yield OmlxcStreamChunk(
                content="",
                model="coding",
                request_id="req-1",
                usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            )
            yield OmlxcStreamChunk(content="token", model="coding", request_id="req-1")
            yield OmlxcStreamChunk(
                content="",
                model="coding",
                request_id="req-1",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def collect():
        response = await openai_proxy.handle_chat_completions(request)
        payload = b"".join([chunk async for chunk in response.body._value]).decode()
        return response, payload

    response, payload = asyncio.run(collect())
    first_usage = payload.index('"total_tokens":1')
    content = payload.index('"content":"token"')
    final_usage = payload.index('"total_tokens":2')

    assert response.status == 200
    assert first_usage < content < final_usage
    assert payload.endswith("data: [DONE]\n\n")


def test_proxy_preserves_normal_metadata_only_stream(monkeypatch):
    class Gateway:
        async def generate_stream(self, _request):
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            yield OmlxcStreamChunk(
                content="",
                model="coding",
                request_id="req-1",
                usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            )
            yield OmlxcStreamChunk(
                content="",
                model="coding",
                request_id="req-1",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 2},
            )

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def collect():
        response = await openai_proxy.handle_chat_completions(request)
        payload = b"".join([chunk async for chunk in response.body._value]).decode()
        return response, payload

    response, payload = asyncio.run(collect())

    assert response.status == 200
    assert payload.index('"total_tokens":1') < payload.index('"total_tokens":2')
    assert '"finish_reason":"stop"' in payload
    assert payload.endswith("data: [DONE]\n\n")


def test_proxy_bounds_pretoken_metadata_wait_and_closes_stream_once(monkeypatch):
    closed = 0

    class Gateway:
        async def generate_stream(self, _request):
            nonlocal closed
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            try:
                yield OmlxcStreamChunk(
                    content="",
                    model="coding",
                    request_id="req-1",
                    usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                )
                await asyncio.Event().wait()
            finally:
                closed += 1

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "timeout": 0.01,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def invoke():
        return await asyncio.wait_for(openai_proxy.handle_chat_completions(request), timeout=0.5)

    response = asyncio.run(invoke())

    assert response.status == 504
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "timeout"
    assert closed == 1


def test_proxy_rejects_high_volume_pretoken_metadata_events_before_http_200(monkeypatch):
    produced = 0
    closed = 0

    class Gateway:
        async def generate_stream(self, _request):
            nonlocal closed, produced
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            try:
                for _ in range(65):
                    produced += 1
                    yield OmlxcStreamChunk(
                        content="",
                        model="coding",
                        request_id="req-1",
                        usage={"prompt_tokens": produced},
                    )
                produced += 1
                yield OmlxcStreamChunk(content="must-not-emit", model="coding", request_id="req-1")
            finally:
                closed += 1

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "timeout": 10,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def invoke():
        return await asyncio.wait_for(openai_proxy.handle_chat_completions(request), timeout=0.5)

    response = asyncio.run(invoke())

    assert response.status == 502
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "internal"
    assert produced == 65
    assert closed == 1


def test_proxy_accepts_event_limit_with_content_and_preserves_order(monkeypatch):
    class Gateway:
        async def generate_stream(self, _request):
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            for index in range(63):
                yield OmlxcStreamChunk(
                    content="",
                    model="coding",
                    request_id="req-1",
                    usage={"prompt_tokens": index},
                )
            yield OmlxcStreamChunk(content="token", model="coding", request_id="req-1")

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def collect():
        response = await openai_proxy.handle_chat_completions(request)
        payload = b"".join([chunk async for chunk in response.body._value]).decode()
        return response, payload

    response, payload = asyncio.run(collect())

    assert response.status == 200
    assert payload.index('"prompt_tokens":0') < payload.index('"content":"token"')
    assert payload.count('"content":""') == 63
    assert payload.endswith("data: [DONE]\n\n")


def test_proxy_rejects_oversized_nested_usage_without_serializing_or_echoing_it(monkeypatch):
    closed = 0
    marker = "private-metadata-marker"

    class Gateway:
        async def generate_stream(self, _request):
            nonlocal closed
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            try:
                yield OmlxcStreamChunk(
                    content="",
                    model="coding",
                    request_id="req-1",
                    usage={"nested": {"payload": marker + ("x" * 65_536)}},
                )
                yield OmlxcStreamChunk(content="must-not-emit", model="coding", request_id="req-1")
            finally:
                closed += 1

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    response = asyncio.run(openai_proxy.handle_chat_completions(request))

    assert response.status == 502
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "internal"
    assert marker not in response.body.decode()
    assert "must-not-emit" not in response.body.decode()
    assert closed == 1


def test_proxy_keeps_posttoken_stream_error_in_sse_without_replay(monkeypatch):
    class Gateway:
        async def generate_stream(self, _request):
            from llm_gateway.omlxc_client import OmlxcStreamChunk

            yield OmlxcStreamChunk(content="once", model="coding", request_id="req-1")
            raise OmlxcError(OmlxcErrorCode.NO_CAPACITY, emitted_content=True)

    request = SimpleNamespace()

    async def body():
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def collect():
        response = await openai_proxy.handle_chat_completions(request)
        payload = b"".join([chunk async for chunk in response.body._value]).decode()
        return response, payload

    response, payload = asyncio.run(collect())
    assert response.status == 200
    assert payload.count('"content":"once"') == 1
    assert '"code":"no_capacity"' in payload
    assert '"emitted_content":true' in payload
    assert "data: [DONE]" not in payload


def test_proxy_maps_embedding_omlxc_error(monkeypatch):
    class Gateway:
        async def embed(self, _texts, **_kwargs):
            raise OmlxcError(OmlxcErrorCode.TIMEOUT)

    request = SimpleNamespace()

    async def body():
        return {"model": "embedding", "input": "hello"}

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(openai_proxy.handle_embeddings(request))
    payload = json.loads(response.body)

    assert response.status == 504
    assert payload["error"]["code"] == "timeout"


def test_proxy_keeps_unknown_embedding_error_sanitized_502(monkeypatch):
    class Gateway:
        async def embed(self, _texts, **_kwargs):
            raise RuntimeError("private downstream detail")

    request = SimpleNamespace()

    async def body():
        return {"model": "embedding", "input": "hello"}

    request.json = body
    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(openai_proxy.handle_embeddings(request))

    assert response.status == 502
    assert "private downstream detail" not in response.body.decode()


def test_bos_infer_reads_agora_envelope_and_calls_running_facade(monkeypatch, capsys):
    from aetherforge import cli

    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"kwargs": {"prompt": "hi", "model": "coding", "timeout": 5}})),
    )
    monkeypatch.setenv("AETHERFORGE_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert cli.cmd_infer([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["choices"][0]["message"]["content"] == "ok"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["routing_mode"] == "local"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_bos_infer_rejects_non_http_base_url(monkeypatch, capsys):
    from aetherforge import cli

    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"hi"}'))
    monkeypatch.setenv("AETHERFORGE_BASE_URL", "file:///tmp/not-a-gateway")
    assert cli.cmd_infer([]) == 2
    assert "http(s)" in json.loads(capsys.readouterr().out)["error"]


def _ready(monkeypatch, served_model: str, content: str = "OK", query: dict | None = None):
    class Gateway:
        async def generate(self, request):
            return GatewayResponse(content=content, model=served_model, latency_ms=1)

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace(query={"model": "mythos-fast", **(query or {})})
    response = asyncio.run(openai_proxy.handle_ready(request))
    return response.status, json.loads(response.body)


def test_ready_reports_ready_when_requested_tier_serves(monkeypatch):
    status, body = _ready(monkeypatch, "mythos-fast")
    assert status == 200
    assert body["status"] == "ready" and body["fallback"] is False


def test_ready_flags_fallback_as_degraded(monkeypatch):
    """兜底承接不能报 ready(此前报 ready, 判活方看不出档已挂); 默认仍 200 保住 gw-resolve 选站。"""
    status, body = _ready(monkeypatch, "coding")
    assert status == 200
    assert body["status"] == "degraded" and body["fallback"] is True
    assert body["requested"] == "mythos-fast" and body["model"] == "coding"


def test_ready_strict_rejects_fallback(monkeypatch):
    status, body = _ready(monkeypatch, "coding", query={"strict": "1"})
    assert status == 503 and body["fallback"] is True


def test_ready_not_ready_without_content(monkeypatch):
    status, body = _ready(monkeypatch, "mythos-fast", content="")
    assert status == 503 and body["status"] == "not_ready"
