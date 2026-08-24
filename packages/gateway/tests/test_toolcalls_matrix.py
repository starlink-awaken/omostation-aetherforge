"""tool_calls 复杂场景矩阵(治理 P1.3) — 生产前只验过最简闭环。

四类此前未覆盖的形态, 双协议(Anthropic SSE / OpenAI delta):
- 并行多工具: 一个响应里多个工具调用交错分片
- 文本+工具混合: text block 与 tool_use block 并存
- 超大 arguments: 大量 input_json_delta 分片长拼接
- 畸形分片: JSON 拼接不完整时不炸(原样保留)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_gateway.provider import LLMRequest
from llm_gateway.providers.anthropic_compat import AnthropicCompatProvider


def _provider() -> AnthropicCompatProvider:
    return AnthropicCompatProvider(name="m", api_key="k", base_url="https://example.com/a", default_model="mm")


def _sse_from_events(events: list[dict]) -> str:
    return "".join("event: e\ndata: " + json.dumps(e) + "\n\n" for e in events)


def _patch(monkeypatch: pytest.MonkeyPatch, sse_text: str) -> None:
    import httpx

    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=sse_text, headers={"content-type": "text/event-stream"})
    )
    original_init = httpx.AsyncClient.__init__

    def patched_init(client, **kwargs):
        kwargs["transport"] = transport
        original_init(client, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


@pytest.mark.asyncio
async def test_parallel_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """两个 tool_use blocks(index 0/1 交错) → 两个完整 tool_calls。"""
    a1 = json.dumps({"a": 1})
    a2 = json.dumps({"b": 2})
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "f1", "input": {}}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t2", "name": "f2", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": a1}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": a2}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    _patch(monkeypatch, _sse_from_events(events))
    evs = [e async for e in _provider().stream_generate_detailed(LLMRequest(prompt="p"))]
    meta = [e for e in evs if e.tool_calls][0]
    assert len(meta.tool_calls) == 2
    by_id = {tc["id"]: tc for tc in meta.tool_calls}
    assert json.loads(by_id["t1"]["function"]["arguments"]) == {"a": 1}
    assert json.loads(by_id["t2"]["function"]["arguments"]) == {"b": 2}
    assert meta.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_text_and_tool_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """text block + tool_use block 并存 → 文本流照吐, 工具调用也在。"""
    q = json.dumps({"q": "x"})
    events = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "我需要查"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t9", "name": "search", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": q}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    _patch(monkeypatch, _sse_from_events(events))
    evs = [e async for e in _provider().stream_generate_detailed(LLMRequest(prompt="p"))]
    texts = [e.text for e in evs if e.text]
    assert texts == ["我需要查"]
    meta = [e for e in evs if e.tool_calls][0]
    assert meta.tool_calls[0]["function"]["name"] == "search"


@pytest.mark.asyncio
async def test_huge_arguments_across_many_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """超大 arguments: 50 个分片长拼接, 最终 JSON 完整可解析。"""
    payload = {"blob": "x" * 5000}
    blob_json = json.dumps(payload)
    chunk_size = len(blob_json) // 50 + 1
    chunks = [blob_json[i : i + chunk_size] for i in range(0, len(blob_json), chunk_size)]
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tbig", "name": "save", "input": {}}},
    ]
    for c in chunks:
        events.append({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": c}})
    events.extend(
        [
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )
    _patch(monkeypatch, _sse_from_events(events))
    evs = [e async for e in _provider().stream_generate_detailed(LLMRequest(prompt="p"))]
    meta = [e for e in evs if e.tool_calls][0]
    assert json.loads(meta.tool_calls[0]["function"]["arguments"]) == payload


@pytest.mark.asyncio
async def test_malformed_partial_json_no_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """畸形分片(拼接不完整)不炸: arguments 原样保留, 消费方按协议容错。"""
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tbad", "name": "f", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"broken": '}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    _patch(monkeypatch, _sse_from_events(events))
    evs = [e async for e in _provider().stream_generate_detailed(LLMRequest(prompt="p"))]
    meta = [e for e in evs if e.tool_calls][0]
    assert meta.tool_calls[0]["function"]["arguments"] == '{"broken": '


def _tc_delta(index: int, id: str | None = None, name: str | None = None, args_chunk: str = "") -> SimpleNamespace:
    fn = SimpleNamespace(name=name, arguments=args_chunk or None)
    return SimpleNamespace(index=index, id=id, function=fn)


def _fake_openai_stream(items: list, usage: dict | None = None):
    chunks = []
    for it in items:
        delta = SimpleNamespace(content=None, tool_calls=[it])
        chunks.append(SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", delta=delta)], usage=None))
    if usage:
        chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(**usage)))

    class _Stream:
        def __aiter__(self):
            self._iter = iter(chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    return _Stream()


def _run_openai(items: list, usage: dict):
    import asyncio

    from llm_gateway.providers.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="k", base_url="https://x/v1")
    fake = _fake_openai_stream(items, usage)

    async def fake_create(**kw):
        return fake

    p._get_async_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    return asyncio.run(_collect(p))


async def _collect(provider):
    return [e async for e in provider.stream_generate_detailed(LLMRequest(prompt="p"))]


def test_openai_parallel_tool_delta_aggregation() -> None:
    """OpenAI 系: 两个 index 的 delta 交错到达, 按 index 正确聚合。"""
    evs = _run_openai(
        [
            _tc_delta(0, id="c1", name="f1", args_chunk='{"a":'),
            _tc_delta(1, id="c2", name="f2", args_chunk='{"b":'),
            _tc_delta(0, args_chunk="1}"),
            _tc_delta(1, args_chunk="2}"),
        ],
        {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )
    meta = [e for e in evs if e.tool_calls][0]
    calls = {tc["id"]: tc for tc in meta.tool_calls}
    assert json.loads(calls["c1"]["function"]["arguments"]) == {"a": 1}
    assert json.loads(calls["c2"]["function"]["arguments"]) == {"b": 2}


def test_openai_args_chunked_across_chunks() -> None:
    """OpenAI 系: 单个工具的 arguments 拆多个 chunk, 拼接完整。"""
    evs = _run_openai(
        [
            _tc_delta(0, id="c9", name="big", args_chunk='{"k":'),
            _tc_delta(0, args_chunk='"v'),
            _tc_delta(0, args_chunk='alue"}'),
        ],
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    meta = [e for e in evs if e.tool_calls][0]
    assert json.loads(meta.tool_calls[0]["function"]["arguments"]) == {"k": "value"}
