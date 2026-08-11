"""AetherForge facade parameter fidelity and BOS infer bridge."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

from llm_gateway import openai_proxy
from llm_gateway.gateway import GatewayResponse


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
    response = asyncio.run(openai_proxy.handle_chat_completions(request))
    assert response.status == 200

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in response.body._value])

    payload = asyncio.run(collect()).decode()
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
