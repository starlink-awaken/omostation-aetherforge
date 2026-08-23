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
async def test_parse_response_empty_content_no_crash() -> None:
    """空 content 数组(限流/异常路径)不得 AttributeError —— 2026-08-23 真实故障。"""
    p = _provider()
    result = p._parse_response({"content": [], "stop_reason": "end_turn"}, "test-model")
    assert result.content == ""
    assert result.finish_reason == "stop"
