from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from llm_gateway import openai_proxy
from llm_gateway.gateway import GatewayConfig, GatewayRequest, GatewayResponse, ModelGateway
from llm_gateway.omlxc_client import (
    OmlxcChatResult,
    OmlxcClient,
    OmlxcError,
    OmlxcErrorCode,
    OmlxcStreamChunk,
    TokenUsage,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "read a file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
TOOL_CALL = {
    "id": "call_read",
    "type": "function",
    "function": {"name": "read", "arguments": '{"filePath":"README.md"}'},
}


@pytest.mark.asyncio
async def test_omlxc_client_forwards_agent_fields_and_parses_nonstream_tool_calls() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "X-OMLXC-Request-ID": "req-1",
                "X-OMLXC-Placement": "placement-local",
                "X-OMLXC-Backend": "backend-local",
            },
            json={
                "model": "coding",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    result = await client.chat(
        model="coding",
        messages=[
            {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_read", "content": "first line"},
        ],
        tools=TOOLS,
        tool_choice="auto",
        top_p=0.8,
        top_k=20,
        stop=["END"],
    )

    assert captured["tools"] == TOOLS
    assert captured["tool_choice"] == "auto"
    assert captured["top_p"] == 0.8
    assert captured["top_k"] == 20
    assert captured["stop"] == ["END"]
    assert captured["messages"][1]["tool_call_id"] == "call_read"  # type: ignore[index]
    assert result.content == ""
    assert result.tool_calls == (TOOL_CALL,)
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_omlxc_client_forwards_bounded_omp_tool_catalog() -> None:
    captured: dict[str, object] = {}
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": "bounded tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(223)
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "X-OMLXC-Request-ID": "req-1",
                "X-OMLXC-Placement": "placement-local",
                "X-OMLXC-Backend": "backend-local",
            },
            json={
                "model": "coding",
                "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    result = await client.chat(
        model="coding",
        messages=[{"role": "user", "content": "inspect"}],
        tools=tools,
    )

    assert result.content == "OK"
    assert len(captured["tools"]) == 223  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_omlxc_client_omits_empty_agent_tool_catalog() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "X-OMLXC-Request-ID": "req-1",
                "X-OMLXC-Placement": "placement-local",
                "X-OMLXC-Backend": "backend-local",
            },
            json={
                "model": "coding",
                "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    result = await client.chat(model="coding", messages=[{"role": "user", "content": "inspect"}], tools=[])

    assert result.content == "OK"
    assert "tools" not in captured


@pytest.mark.asyncio
async def test_omlxc_client_rejects_tool_choice_without_nonempty_tools() -> None:
    client = OmlxcClient(transport=httpx.MockTransport(lambda _request: pytest.fail("request must not be sent")))

    with pytest.raises(OmlxcError) as raised:
        await client.chat(
            model="coding",
            messages=[{"role": "user", "content": "inspect"}],
            tools=[],
            tool_choice="auto",
        )

    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_omlxc_client_omits_empty_agent_tool_catalog_from_stream() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "req-1",
                "X-OMLXC-Placement": "placement-local",
                "X-OMLXC-Backend": "backend-local",
            },
            content=b"data: [DONE]\n\n",
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    chunks = [
        chunk
        async for chunk in client.stream_chat(
            model="coding", messages=[{"role": "user", "content": "inspect"}], tools=[]
        )
    ]

    assert chunks == []
    assert "tools" not in captured


@pytest.mark.asyncio
async def test_omlxc_client_parses_stream_tool_call_deltas() -> None:
    body = (
        b'data: {"model":"coding","choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":0,"id":"call_read","type":"function","function":{"name":"read","arguments":"{}"}}]},'
        b'"finish_reason":null}]}\n\n'
        b'data: {"model":"coding","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "req-1",
                "X-OMLXC-Placement": "placement-local",
                "X-OMLXC-Backend": "backend-local",
            },
            content=body,
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    chunks = [
        chunk
        async for chunk in client.stream_chat(
            model="coding", messages=[{"role": "user", "content": "inspect"}], tools=TOOLS
        )
    ]

    assert chunks[0].tool_calls[0]["function"]["name"] == "read"
    assert chunks[1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_omlxc_client_rejects_malformed_tool_call_delta() -> None:
    body = b'data: {"model":"coding","choices":[{"delta":{"tool_calls":[{"index":-1}]}}]}\n\ndata: [DONE]\n\n'

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "req-1",
                "X-OMLXC-Placement": "placement-local",
                "X-OMLXC-Backend": "backend-local",
            },
            content=body,
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        async for _chunk in client.stream_chat(model="coding", messages=[{"role": "user", "content": "inspect"}]):
            pass
    assert raised.value.code is OmlxcErrorCode.INVALID


class _FakeOmlxc:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> OmlxcChatResult:
        self.chat_calls.append(kwargs)
        return OmlxcChatResult(
            content="",
            tool_calls=(TOOL_CALL,),
            model=kwargs["model"],
            finish_reason="tool_calls",
            usage=TokenUsage(),
            request_id="req-1",
            placement="placement-local",
            backend="backend-local",
        )

    async def stream_chat(self, **kwargs: Any) -> AsyncIterator[OmlxcStreamChunk]:
        self.stream_calls.append(kwargs)
        yield OmlxcStreamChunk(
            model=kwargs["model"],
            request_id="req-1",
            tool_calls=({"index": 0, **TOOL_CALL},),
        )
        yield OmlxcStreamChunk(model=kwargs["model"], request_id="req-1", finish_reason="tool_calls")


def _gateway(client: _FakeOmlxc) -> ModelGateway:
    config = GatewayConfig(
        omlxc_mode="active",
        aliases={"coding": "local/coding"},
        model_ports={},
        model_sizes={},
        lmstudio_fallback={},
        ollama_fallback={},
        fallback_chain=["coding"],
        complexity_chains={"simple": ["coding"], "complex": ["coding"]},
        background_tasks_enabled=False,
    )
    return ModelGateway(SimpleNamespace(), SimpleNamespace(), config, omlxc_client=client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_active_gateway_forwards_only_agent_fields_and_preserves_tool_calls() -> None:
    client = _FakeOmlxc()
    gateway = _gateway(client)
    request = GatewayRequest(
        messages=[{"role": "user", "content": "inspect"}],
        model="coding",
        extra={
            "tools": TOOLS,
            "tool_choice": "auto",
            "top_p": 0.8,
            "top_k": 20,
            "stop": ["END"],
            "response_format": {"type": "json"},
        },
    )

    response = await gateway.generate(request)
    chunks = [chunk async for chunk in gateway.generate_stream(request)]

    assert client.chat_calls[0]["tools"] == TOOLS
    assert client.chat_calls[0]["tool_choice"] == "auto"
    assert client.chat_calls[0]["top_p"] == 0.8
    assert client.chat_calls[0]["top_k"] == 20
    assert client.chat_calls[0]["stop"] == ["END"]
    assert "response_format" not in client.chat_calls[0]
    assert client.stream_calls[0]["tools"] == TOOLS
    assert response.tool_calls == (TOOL_CALL,)
    assert chunks[0].tool_calls[0]["function"]["name"] == "read"


def test_openai_facade_emits_nonstream_and_stream_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gateway:
        async def generate(self, _request: GatewayRequest) -> GatewayResponse:
            return GatewayResponse(
                content="", model="coding", latency_ms=1, tool_calls=(TOOL_CALL,), finish_reason="tool_calls"
            )

        async def generate_stream(self, _request: GatewayRequest) -> AsyncIterator[OmlxcStreamChunk]:
            yield OmlxcStreamChunk(
                model="coding",
                request_id="req-1",
                tool_calls=({"index": 0, **TOOL_CALL},),
            )
            yield OmlxcStreamChunk(model="coding", request_id="req-1", finish_reason="tool_calls")

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())

    async def invoke(stream: bool) -> object:
        request = SimpleNamespace()

        async def body() -> dict[str, object]:
            return {
                "model": "coding",
                "messages": [{"role": "user", "content": "inspect"}],
                "tools": TOOLS,
                "tool_choice": "auto",
                "stream": stream,
            }

        request.json = body
        response = await openai_proxy.handle_chat_completions(request)
        if not stream:
            return json.loads(response.body)
        return b"".join([chunk async for chunk in response.body._value]).decode()

    nonstream = asyncio.run(invoke(False))
    stream = asyncio.run(invoke(True))

    assert nonstream["choices"][0]["message"]["tool_calls"] == [TOOL_CALL]  # type: ignore[index]
    assert '"tool_calls"' in stream
    assert '"finish_reason":"tool_calls"' in stream
    assert stream.count("data: [DONE]") == 1


def test_openai_facade_normalizes_pi_completion_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[GatewayRequest] = []

    class Gateway:
        async def generate(self, request: GatewayRequest) -> GatewayResponse:
            captured.append(request)
            return GatewayResponse(content="ok", model="coding", latency_ms=1)

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace()

    async def body() -> dict[str, object]:
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "inspect"}],
            "max_completion_tokens": 321,
            "parallel_tool_calls": True,
            "store": False,
            "tools": TOOLS,
        }

    request.json = body
    response = asyncio.run(openai_proxy.handle_chat_completions(request))

    assert response.status == 200
    assert captured[0].max_tokens == 321
    assert captured[0].extra == {
        "parallel_tool_calls": True,
        "store": False,
        "tools": TOOLS,
    }


def test_openai_facade_rejects_conflicting_completion_token_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gateway:
        async def generate(self, _request: GatewayRequest) -> GatewayResponse:
            raise AssertionError("conflicting request must not reach the gateway")

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace()

    async def body() -> dict[str, object]:
        return {
            "model": "coding",
            "messages": [{"role": "user", "content": "inspect"}],
            "max_tokens": 100,
            "max_completion_tokens": 200,
        }

    request.json = body
    response = asyncio.run(openai_proxy.handle_chat_completions(request))

    assert response.status == 400


def test_gemini_generate_content_translates_text_system_and_generation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Gateway:
        async def generate(self, request: GatewayRequest) -> GatewayResponse:
            captured["request"] = request
            return GatewayResponse(
                content="hello from local",
                model="coding",
                latency_ms=1,
                tokens_in=7,
                tokens_out=3,
                finish_reason="stop",
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace(
        match_info={"model": "coding"},
        path="/v1beta/models/coding:generateContent",
    )

    async def body() -> dict[str, object]:
        return {
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
            "contents": [{"role": "user", "parts": [{"text": "Say hello"}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 64,
                "topP": 0.9,
                "topK": 20,
                "stopSequences": ["END"],
            },
        }

    request.json = body
    response = asyncio.run(openai_proxy.handle_gemini_generate_content(request))

    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["candidates"][0]["content"] == {
        "role": "model",
        "parts": [{"text": "hello from local"}],
    }
    assert payload["usageMetadata"] == {
        "promptTokenCount": 7,
        "candidatesTokenCount": 3,
        "totalTokenCount": 10,
    }
    gateway_request = captured["request"]
    assert gateway_request.messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello"},
    ]
    assert gateway_request.temperature == 0.2
    assert gateway_request.max_tokens == 64
    assert gateway_request.extra["top_p"] == 0.9
    assert gateway_request.extra["top_k"] == 20
    assert gateway_request.extra["stop"] == ["END"]


def test_gemini_function_declarations_and_function_call_response_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Gateway:
        async def generate(self, request: GatewayRequest) -> GatewayResponse:
            captured["request"] = request
            return GatewayResponse(
                content="",
                model="coding",
                latency_ms=1,
                finish_reason="tool_calls",
                tool_calls=(
                    {
                        "id": "gemini_get_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    },
                ),
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace(
        match_info={"model": "coding"},
        path="/v1beta/models/coding:generateContent",
    )

    async def body() -> dict[str, object]:
        return {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Paris"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_weather",
                                "response": {"temperature": 18},
                            }
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                }
            ],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        }

    request.json = body
    response = asyncio.run(openai_proxy.handle_gemini_generate_content(request))

    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["candidates"][0]["content"]["parts"] == [
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
    ]
    gateway_request = captured["request"]
    assert gateway_request.messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert gateway_request.messages[1]["role"] == "tool"
    assert gateway_request.messages[1]["tool_call_id"] == "gemini_get_weather"
    assert gateway_request.extra["tools"][0]["function"]["name"] == "get_weather"
    assert gateway_request.extra["tool_choice"] == "auto"


def test_gemini_stream_emits_sse_chunks_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gateway:
        async def generate_stream(self, _request: GatewayRequest) -> AsyncIterator[OmlxcStreamChunk]:
            yield OmlxcStreamChunk(content="Hel", model="coding", request_id="r1")
            yield OmlxcStreamChunk(
                content="lo",
                model="coding",
                request_id="r1",
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace(
        match_info={"model": "coding"},
        path="/v1beta/models/coding:streamGenerateContent",
    )

    async def body() -> dict[str, object]:
        return {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}

    async def collect() -> bytes:
        request.json = body
        response = await openai_proxy.handle_gemini_stream_generate_content(request)
        return b"".join([chunk async for chunk in response.body._value])

    raw = asyncio.run(collect()).decode()
    events = [
        json.loads(block.removeprefix("data: "))
        for block in raw.split("\n\n")
        if block.strip()
    ]

    assert [event["candidates"][0]["content"]["parts"][0]["text"] for event in events] == ["Hel", "lo"]
    assert events[-1]["candidates"][0]["finishReason"] == "STOP"
    assert events[-1]["usageMetadata"]["totalTokenCount"] == 7


def test_gemini_models_endpoint_uses_gemini_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gateway:
        async def list_omlxc_models(self):
            return (
                SimpleNamespace(id="ENG-OMLX-LOCAL/coding"),
                SimpleNamespace(id="ENG-OMLX-LOCAL/vision"),
            )

    monkeypatch.setattr(openai_proxy, "get_gateway", lambda: Gateway())
    request = SimpleNamespace(query={})
    response = asyncio.run(openai_proxy.handle_gemini_list_models(request))
    payload = json.loads(response.body)

    assert [model["name"] for model in payload["models"]] == [
        "models/ENG-OMLX-LOCAL/coding",
        "models/ENG-OMLX-LOCAL/vision",
    ]
