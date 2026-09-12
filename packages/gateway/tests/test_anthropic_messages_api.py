"""/v1/messages — Anthropic Messages API translation layer."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from llm_gateway import anthropic_proxy
from llm_gateway.gateway import GatewayResponse
from llm_gateway.omlxc_client import OmlxcError, OmlxcErrorCode, OmlxcStreamChunk


def _request(body: dict) -> SimpleNamespace:
    async def _json():
        return body

    return SimpleNamespace(json=_json)


async def _achunks(*chunks):
    for chunk in chunks:
        yield chunk


def _sse_events(raw: bytes) -> list[tuple[str, dict]]:
    events = []
    for block in raw.decode().split("\n\n"):
        if not block.strip():
            continue
        event_line, data_line = block.split("\n", 1)
        event_type = event_line.removeprefix("event: ")
        data = json.loads(data_line.removeprefix("data: "))
        events.append((event_type, data))
    return events


def test_plain_string_content_returns_text_block(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="hello there", model="coding", latency_ms=1, tokens_in=3, tokens_out=2)

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        anthropic_proxy.handle_messages(
            _request({"model": "coding", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
        )
    )
    payload = json.loads(response.body)

    assert response.status == 200
    assert payload["stop_reason"] == "end_turn"
    assert payload["content"] == [{"type": "text", "text": "hello there"}]
    assert payload["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert captured["request"].messages == [{"role": "user", "content": "hi"}]


def test_system_prompt_becomes_system_message(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    asyncio.run(
        anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "system": "be terse",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
        )
    )

    assert captured["request"].messages[0] == {"role": "system", "content": "be terse"}
    assert captured["request"].messages[1] == {"role": "user", "content": "hi"}


def test_max_tokens_is_required(monkeypatch):
    class Gateway:
        async def generate(self, request):
            raise AssertionError("must not reach the model")

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        anthropic_proxy.handle_messages(_request({"model": "coding", "messages": [{"role": "user", "content": "hi"}]}))
    )
    payload = json.loads(response.body)

    assert response.status == 400
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"


def test_tool_use_and_tool_result_round_trip(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(
                content="",
                model="coding",
                latency_ms=1,
                finish_reason="tool_calls",
                tool_calls=(
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"sf"}'},
                    },
                ),
            )

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "messages": [
                        {"role": "user", "content": "weather in sf?"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "sf"}}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F and sunny"}],
                        },
                    ],
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "get weather",
                            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                        }
                    ],
                }
            )
        )
    )
    payload = json.loads(response.body)
    req = captured["request"]

    assert req.messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert req.messages[2] == {"role": "tool", "tool_call_id": "toolu_1", "content": "72F and sunny"}
    assert req.extra["tools"][0]["function"]["name"] == "get_weather"
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["type"] == "tool_use"
    assert payload["content"][0]["id"] == "toolu_1"
    assert payload["content"][0]["input"] == {"city": "sf"}


def test_tool_choice_translation(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    asyncio.run(
        anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tool_choice": {"type": "tool", "name": "get_weather"},
                }
            )
        )
    )

    assert captured["request"].extra["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_unsupported_role_is_400(monkeypatch):
    class Gateway:
        async def generate(self, request):
            raise AssertionError("must not reach the model on a malformed request")

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "messages": [{"role": "function", "content": "nope"}],
                }
            )
        )
    )
    payload = json.loads(response.body)

    assert response.status == 400
    assert payload["error"]["type"] == "invalid_request_error"


def test_inline_system_role_message_is_accepted(monkeypatch):
    """Anthropic's public spec only documents user/assistant inside `messages`
    (system prompts belong in the dedicated top-level `system` field), but real
    clients -- Claude Code itself, for some internal calls like session-title
    generation -- do send role="system" inline. Confirmed via a live
    `claude -p` run against this endpoint: it 400'd here before this was
    accepted."""
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "messages": [{"role": "system", "content": "Summarize this conversation in five words."}],
                }
            )
        )
    )

    assert response.status == 200
    assert captured["request"].messages == [
        {"role": "system", "content": "Summarize this conversation in five words."}
    ]


def test_error_response_maps_to_anthropic_error_envelope(monkeypatch):
    class Gateway:
        async def generate(self, request):
            return GatewayResponse(
                content="",
                model="coding",
                latency_ms=1,
                error="no capacity",
                error_code=OmlxcErrorCode.NO_CAPACITY,
            )

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        anthropic_proxy.handle_messages(
            _request({"model": "coding", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
        )
    )
    payload = json.loads(response.body)

    assert response.status == 409
    assert payload["type"] == "error"
    assert payload["error"]["message"] == "no capacity"


def test_stream_text_emits_content_block_and_message_events(monkeypatch):
    class Gateway:
        def generate_stream(self, request):
            return _achunks(
                OmlxcStreamChunk(content="Hel", model="coding", request_id="r1"),
                OmlxcStreamChunk(
                    content="lo",
                    model="coding",
                    request_id="r1",
                    finish_reason="stop",
                    usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                ),
            )

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())

    async def go():
        response = await anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
        )
        raw = b"".join([chunk async for chunk in response.body._value])
        return response, raw

    response, raw = asyncio.run(go())
    assert response.status == 200
    events = _sse_events(raw)
    types = [t for t, _ in events]

    assert types[0] == "message_start"
    assert "content_block_delta" in types
    deltas = [d["delta"]["text"] for t, d in events if t == "content_block_delta"]
    assert "".join(deltas) == "Hello"
    assert types[-1] == "message_stop"
    message_delta = next(d for t, d in events if t == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"
    assert message_delta["usage"]["output_tokens"] == 2


def test_stream_tool_use_emits_input_json_delta(monkeypatch):
    class Gateway:
        def generate_stream(self, request):
            return _achunks(
                OmlxcStreamChunk(
                    content="",
                    model="coding",
                    request_id="r1",
                    tool_calls=(
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"sf"}'},
                        },
                    ),
                ),
            )

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())

    async def go():
        response = await anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "weather?"}],
                }
            )
        )
        raw = b"".join([chunk async for chunk in response.body._value])
        return raw

    events = _sse_events(asyncio.run(go()))
    types = [t for t, _ in events]

    assert "content_block_start" in types
    start = next(d for t, d in events if t == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["id"] == "toolu_1"
    delta = next(d for t, d in events if t == "content_block_delta")
    assert delta["delta"]["type"] == "input_json_delta"
    assert delta["delta"]["partial_json"] == '{"city":"sf"}'
    message_delta = next(d for t, d in events if t == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_stream_error_mid_stream_emits_error_event(monkeypatch):
    async def _failing_source():
        yield OmlxcStreamChunk(content="partial", model="coding", request_id="r1")
        raise OmlxcError(OmlxcErrorCode.TIMEOUT)

    class Gateway:
        def generate_stream(self, request):
            return _failing_source()

    monkeypatch.setattr(anthropic_proxy, "get_gateway", lambda: Gateway())

    async def go():
        response = await anthropic_proxy.handle_messages(
            _request(
                {
                    "model": "coding",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
        )
        raw = b"".join([chunk async for chunk in response.body._value])
        return raw

    events = _sse_events(asyncio.run(go()))
    assert events[-1][0] == "error"
    assert events[-1][1]["type"] == "error"
