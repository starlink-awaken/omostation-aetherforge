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


def test_proxy_rejects_stream_instead_of_returning_fake_non_stream(monkeypatch):
    request = SimpleNamespace()

    async def body():
        return {"messages": [{"role": "user", "content": "hi"}], "stream": True}

    request.json = body
    response = asyncio.run(openai_proxy.handle_chat_completions(request))
    assert response.status == 501


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
