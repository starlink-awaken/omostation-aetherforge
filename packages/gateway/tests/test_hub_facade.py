"""AetherForge facade parameter fidelity and BOS infer bridge."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from llm_gateway import openai_proxy
from llm_gateway.gateway import GatewayResponse
from llm_gateway.omlxc_client import OmlxcError, OmlxcErrorCode


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
