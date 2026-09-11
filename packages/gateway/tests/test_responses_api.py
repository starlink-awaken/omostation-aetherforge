"""/v1/responses — OpenAI Responses API translation layer."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from llm_gateway import openai_proxy
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


def test_plain_string_input_returns_text(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="hello there", model="coding", latency_ms=1, tokens_in=3, tokens_out=2)

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(openai_proxy.handle_responses(_request({"model": "coding", "input": "hi"})))
    payload = json.loads(response.body)

    assert response.status == 200
    assert payload["status"] == "completed"
    assert payload["output"][0]["type"] == "message"
    assert payload["output"][0]["content"][0]["text"] == "hello there"
    assert payload["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert captured["request"].messages == [{"role": "user", "content": "hi"}]


def test_instructions_become_system_message(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    asyncio.run(openai_proxy.handle_responses(_request({"model": "coding", "instructions": "be terse", "input": "hi"})))

    assert captured["request"].messages[0] == {"role": "system", "content": "be terse"}
    assert captured["request"].messages[1] == {"role": "user", "content": "hi"}


def test_function_call_round_trip(monkeypatch):
    captured = {}

    class Gateway:
        async def generate(self, request):
            captured["request"] = request
            return GatewayResponse(
                content="",
                model="coding",
                latency_ms=1,
                tool_calls=(
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"sf"}'},
                    },
                ),
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        openai_proxy.handle_responses(
            _request(
                {
                    "model": "coding",
                    "input": [
                        {"type": "message", "role": "user", "content": "weather in sf?"},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": '{"city":"sf"}',
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "72F and sunny",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "get weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        }
                    ],
                }
            )
        )
    )
    payload = json.loads(response.body)
    req = captured["request"]

    assert req.messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert req.messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "72F and sunny"}
    assert req.extra["tools"][0]["function"]["name"] == "get_weather"
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["call_id"] == "call_1"
    assert payload["output"][0]["name"] == "get_weather"


def test_reasoning_item_is_skipped_not_rejected(monkeypatch):
    class Gateway:
        async def generate(self, request):
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        openai_proxy.handle_responses(
            _request(
                {
                    "model": "coding",
                    "input": [
                        {"type": "reasoning", "summary": []},
                        {"type": "message", "role": "user", "content": "hi"},
                    ],
                }
            )
        )
    )
    assert response.status == 200


def test_unsupported_input_item_type_is_400(monkeypatch):
    class Gateway:
        async def generate(self, request):
            raise AssertionError("must not reach the model on a malformed request")

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        openai_proxy.handle_responses(_request({"model": "coding", "input": [{"type": "computer_call"}]}))
    )
    payload = json.loads(response.body)

    assert response.status == 400
    assert payload["error"]["type"] == "invalid_request_error"


def test_previous_response_id_is_rejected(monkeypatch):
    class Gateway:
        async def generate(self, request):
            raise AssertionError("must not reach the model")

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(
        openai_proxy.handle_responses(_request({"model": "coding", "input": "hi", "previous_response_id": "resp_1"}))
    )
    assert response.status == 400


def test_error_response_maps_to_failed_status(monkeypatch):
    class Gateway:
        async def generate(self, request):
            return GatewayResponse(
                content="",
                model="coding",
                latency_ms=1,
                error="no capacity",
                error_code=OmlxcErrorCode.NO_CAPACITY,
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    response = asyncio.run(openai_proxy.handle_responses(_request({"model": "coding", "input": "hi"})))
    payload = json.loads(response.body)

    assert response.status == 409
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "no_capacity"


def test_stream_text_emits_delta_and_completed_events(monkeypatch):
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

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def go():
        response = await openai_proxy.handle_responses(_request({"model": "coding", "input": "hi", "stream": True}))
        raw = b"".join([chunk async for chunk in response.body._value])
        return response, raw

    response, raw = asyncio.run(go())
    assert response.status == 200
    events = _sse_events(raw)
    types = [t for t, _ in events]

    assert types[0] == "response.created"
    assert "response.output_text.delta" in types
    deltas = [d["delta"] for t, d in events if t == "response.output_text.delta"]
    assert "".join(deltas) == "Hello"
    assert types[-1] == "response.completed"
    final = events[-1][1]["response"]
    assert final["status"] == "completed"
    assert final["output"][0]["content"][0]["text"] == "Hello"
    assert final["usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}


def test_stream_function_call_emits_arguments_events(monkeypatch):
    class Gateway:
        def generate_stream(self, request):
            return _achunks(
                OmlxcStreamChunk(
                    content="",
                    model="coding",
                    request_id="r1",
                    tool_calls=(
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"sf"}'},
                        },
                    ),
                ),
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def go():
        response = await openai_proxy.handle_responses(
            _request({"model": "coding", "input": "weather?", "stream": True})
        )
        raw = b"".join([chunk async for chunk in response.body._value])
        return raw

    events = _sse_events(asyncio.run(go()))
    types = [t for t, _ in events]

    assert "response.function_call_arguments.done" in types
    done_event = next(d for t, d in events if t == "response.function_call_arguments.done")
    assert done_event["arguments"] == '{"city":"sf"}'
    final = events[-1][1]["response"]
    assert final["output"][0]["type"] == "function_call"
    assert final["output"][0]["call_id"] == "call_1"


def test_stream_error_mid_stream_emits_failed_event(monkeypatch):
    async def _failing_source():
        yield OmlxcStreamChunk(content="partial", model="coding", request_id="r1")
        raise OmlxcError(OmlxcErrorCode.TIMEOUT)

    class Gateway:
        def generate_stream(self, request):
            return _failing_source()

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def go():
        response = await openai_proxy.handle_responses(_request({"model": "coding", "input": "hi", "stream": True}))
        raw = b"".join([chunk async for chunk in response.body._value])
        return raw

    events = _sse_events(asyncio.run(go()))
    assert events[-1][0] == "response.failed"
    assert events[-1][1]["error"]["code"] == "timeout"
