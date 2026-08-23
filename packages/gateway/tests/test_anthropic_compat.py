"""AnthropicCompatProvider 流式与响应解析回归测试。

背景(2026-08-23):
- _parse_response 的 fallback 曾对 list 类型 content 调 .get, 空内容响应
  直接 AttributeError 炸 provider(真实故障: longcat 空响应触发)。
- stream_generate 原继承基类默认(非流式聚合一次吐), Anthropic 系引擎
  流式 TTFT 等于整个生成时长; 现在实现真 SSE 解析。
"""

from __future__ import annotations

import json

import pytest

from llm_gateway.providers.anthropic_compat import AnthropicCompatProvider
from llm_gateway.provider import LLMRequest


def _provider() -> AnthropicCompatProvider:
    return AnthropicCompatProvider(
        name="test-anthropic",
        api_key="test-key",
        base_url="https://example.com/anthropic",
        default_model="test-model",
    )


def _sse_body() -> str:
    events = [
        '{"type":"message_start","message":{"usage":{"input_tokens":5}}}',
        '{"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"，世界"}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}',
        "not-json-garbage",
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        '{"type":"message_stop"}',
    ]
    return "".join(f"event: e\ndata: {e}\n\n" for e in events)


@pytest.mark.asyncio
async def test_stream_generate_yields_only_text_deltas_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """真 SSE 解析: 只吐 text_delta, 忽略 input_json_delta/垃圾行/其他事件。"""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/anthropic/messages"
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(200, text=_sse_body(), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    original_init = httpx.AsyncClient.__init__

    def patched_init(client: httpx.AsyncClient, **kwargs: object) -> None:
        kwargs["transport"] = transport  # type: ignore[assignment]
        original_init(client, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    chunks = [c async for c in _provider().stream_generate(LLMRequest(prompt="hi"))]

    assert chunks == ["你好", "，世界"]


@pytest.mark.asyncio
async def test_stream_detailed_carries_usage_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """detailed 流: 文本块 + 结束块带 usage(prompt/completion/total)与 finish_reason。"""
    import httpx

    body_events = [
        '{"type":"message_start","message":{"usage":{"input_tokens":16}}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你"}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好"}}',
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}',
        '{"type":"message_stop"}',
    ]
    sse = "".join(f"event: e\ndata: {e}\n\n" for e in body_events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    original_init = httpx.AsyncClient.__init__

    def patched_init(client: httpx.AsyncClient, **kwargs: object) -> None:
        kwargs["transport"] = transport  # type: ignore[assignment]
        original_init(client, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    from llm_gateway.provider import LLMRequest

    events = [e async for e in _provider().stream_generate_detailed(LLMRequest(prompt="hi"))]
    assert [(e.text, e.finish_reason, e.usage) for e in events] == [
        ("你", None, None),
        ("好", None, None),
        ("", "stop", {"prompt_tokens": 16, "completion_tokens": 7, "total_tokens": 23}),
    ]


@pytest.mark.asyncio
async def test_parse_response_empty_content_no_crash() -> None:
    """空 content 数组(限流/异常路径)不得 AttributeError —— 2026-08-23 真实故障。"""
    p = _provider()
    result = p._parse_response({"content": [], "stop_reason": "end_turn"}, "test-model")
    assert result.content == ""
    assert result.finish_reason == "stop"


def test_parse_response_tool_use_becomes_openai_tool_calls() -> None:
    p = _provider()
    result = p._parse_response(
        {
            "content": [
                {"type": "text", "text": "我需要查天气"},
                {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "北京"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        "test-model",
    )
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["id"] == "toolu_01"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "北京"}


def test_build_body_converts_openai_tools_to_anthropic() -> None:
    p = _provider()
    req = LLMRequest(
        prompt="hi",
        extra={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查天气",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    body = p._build_body(req)
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "查天气",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert body["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_stream_detailed_aggregates_tool_use_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    # 事件行用 json.dumps 构造 —— 手拼转义第一版就拼出了畸形 JSON, 被解析器
    # 静默跳过导致分片丢失, 白查一轮。
    events = [
        json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
        json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_02", "name": "search", "input": {}}}),
        json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"q":'}}),
        json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '"x"}'}}),
        json.dumps({"type": "content_block_stop", "index": 0}),
        json.dumps({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}),
        json.dumps({"type": "message_stop"}),
    ]
    sse = "".join(f"event: e\ndata: {e}\n\n" for e in events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    original_init = httpx.AsyncClient.__init__

    def patched_init(client: httpx.AsyncClient, **kwargs: object) -> None:
        kwargs["transport"] = transport  # type: ignore[assignment]
        original_init(client, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    events_out = [e async for e in _provider().stream_generate_detailed(LLMRequest(prompt="hi"))]
    assert len(events_out) == 1  # 无文本块, 只有 meta
    meta = events_out[0]
    assert meta.finish_reason == "tool_calls"
    assert len(meta.tool_calls) == 1
    assert meta.tool_calls[0]["function"]["name"] == "search"
    assert json.loads(meta.tool_calls[0]["function"]["arguments"]) == {"q": "x"}
    assert meta.usage is not None and meta.usage["prompt_tokens"] == 9
