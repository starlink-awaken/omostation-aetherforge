from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator

import httpx
import pytest
from llm_gateway.omlxc_client import (
    OmlxcClient,
    OmlxcError,
    OmlxcErrorCode,
    OmlxcStreamChunk,
)


def _envelope(data: object, request_id: str = "req-1") -> dict[str, object]:
    return {"schema_version": 1, "request_id": request_id, "data": data}


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...], *, error: BaseException | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_list_models_uses_versioned_envelope_and_requests_a_bounded_page() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "catalog-req"},
            json=_envelope(
                {
                    "items": [
                        {"id": "coding", "capabilities": ["chat", "streaming"]},
                        {"id": "embedding", "capabilities": ["embedding"]},
                    ],
                    "next_cursor": None,
                },
                request_id="catalog-req",
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    models = await client.list_models(timeout=2)

    assert captured == {"path": "/api/v1/models", "params": {"limit": "100"}}
    assert [model.id for model in models] == ["coding", "embedding"]
    assert models[0].capabilities == frozenset({"chat", "streaming"})


@pytest.mark.asyncio
async def test_list_models_reads_all_pages_without_repeating_a_cursor() -> None:
    requests: list[dict[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(dict(request.url.params))
        after = request.url.params.get("after")
        if after is None:
            data = {"items": [{"id": "coding", "capabilities": ["chat"]}], "next_cursor": "coding"}
        elif after == "coding":
            data = {
                "items": [{"id": "embedding", "capabilities": ["embedding"]}],
                "next_cursor": "embedding",
            }
        elif after == "embedding":
            data = {"items": [], "next_cursor": None}
        else:
            pytest.fail(f"unexpected cursor: {after}")
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "catalog-req"},
            json=_envelope(data, request_id="catalog-req"),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    models = await client.list_models(timeout=2)

    assert [model.id for model in models] == ["coding", "embedding"]
    assert requests == [
        {"limit": "100"},
        {"after": "coding", "limit": "100"},
        {"after": "embedding", "limit": "100"},
    ]


@pytest.mark.asyncio
async def test_list_models_rejects_a_repeated_pagination_cursor() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "catalog-req"},
            json=_envelope(
                {"items": [{"id": "coding", "capabilities": ["chat"]}], "next_cursor": "coding"},
                request_id="catalog-req",
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    with pytest.raises(OmlxcError) as raised:
        await client.list_models(timeout=2)

    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_list_models_uses_one_total_timeout_budget_across_pages() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.07)
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "catalog-req"},
            json=_envelope(
                {"items": [{"id": "coding", "capabilities": ["chat"]}], "next_cursor": "coding"},
                request_id="catalog-req",
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    with pytest.raises(OmlxcError) as raised:
        await client.list_models(timeout=0.1)

    assert raised.value.code is OmlxcErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_list_models_rejects_a_malformed_catalog_item() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "catalog-req"},
            json=_envelope(
                {"items": [{"id": ["not-a-model-id"], "capabilities": ["chat"]}], "next_cursor": None},
                request_id="catalog-req",
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    with pytest.raises(OmlxcError) as raised:
        await client.list_models(timeout=2)

    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_list_models_rejects_a_mismatched_envelope_request_id() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "header-request"},
            json=_envelope(
                {"items": [{"id": "coding", "capabilities": ["chat"]}], "next_cursor": None},
                request_id="body-request",
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    with pytest.raises(OmlxcError) as raised:
        await client.list_models(timeout=2)

    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_list_models_rejects_duplicate_logical_ids_across_pages() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        after = request.url.params.get("after")
        if after is None:
            data = {"items": [{"id": "coding", "capabilities": ["chat"]}], "next_cursor": "one"}
        elif after == "one":
            data = {"items": [{"id": "coding", "capabilities": ["chat"]}], "next_cursor": None}
        else:
            pytest.fail(f"unexpected cursor: {after}")
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "catalog-req"},
            json=_envelope(data, request_id="catalog-req"),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    with pytest.raises(OmlxcError) as raised:
        await client.list_models(timeout=2)

    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_route_plan_uses_versioned_envelope_and_resolved_model() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "req-1"},
            json=_envelope(
                {
                    "request_id": "req-1",
                    "selected_placement_id": "placement-a",
                    "candidates": ["placement-a", "placement-b"],
                    "candidate_scores": {"placement-a": 1.0},
                    "rejected": {"placement-b": "capacity"},
                    "fallback_chain": ["placement-a"],
                    "config_version": "v1",
                    "explanation": "local candidate selected",
                    "thinking_authorized": False,
                }
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    plan = await client.route_plan(
        "local/resolved",
        profile="interactive",
        capabilities={"chat", "streaming"},
        context_tokens=123,
        thinking=False,
    )

    assert captured == {
        "path": "/api/v1/routes/plan",
        "body": {
            "model_id": "local/resolved",
            "profile": "interactive",
            "required_capabilities": ["chat", "streaming"],
            "context_tokens": 123,
            "thinking_requested": False,
        },
    }
    assert plan.request_id == "req-1"
    assert plan.selected == "placement-a"
    assert plan.fallback == ("placement-a",)
    assert plan.rejected == {"placement-b": "capacity"}
    assert plan.explanation == "local candidate selected"


@pytest.mark.asyncio
async def test_chat_and_embeddings_validate_shapes_and_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "local/resolved"
        assert body["timeout_seconds"] == 7.5
        if request.url.path.endswith("embeddings"):
            return httpx.Response(
                200,
                headers={"X-OMLXC-Request-ID": "backend-req"},
                json={
                    "object": "list",
                    "model": "local/resolved",
                    "data": [
                        {"object": "embedding", "index": 1, "embedding": [3.0, 4.0]},
                        {"object": "embedding", "index": 0, "embedding": [1.0, 2.0]},
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        return httpx.Response(
            200,
            headers={
                "X-OMLXC-Request-ID": "backend-req",
                "X-OMLXC-Placement": "placement-a",
                "X-OMLXC-Backend": "backend-a",
            },
            json={
                "id": "chatcmpl-backend-req",
                "object": "chat.completion",
                "model": "local/resolved",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    result = await client.chat(
        model="local/resolved",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        max_tokens=16,
        timeout=7.5,
    )
    vectors = await client.embed(model="local/resolved", inputs=["a", "b"], timeout=7.5)

    assert result.content == "answer"
    assert result.request_id == "backend-req"
    assert result.placement == "placement-a"
    assert result.backend == "backend-a"
    assert result.usage.total_tokens == 3
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.asyncio
async def test_embedding_rejects_non_finite_or_ragged_vectors() -> None:
    payloads = [
        [[1.0, math.nan]],
        [[1.0], [2.0, 3.0]],
    ]
    for vectors in payloads:

        async def handler(_request: httpx.Request, current=vectors) -> httpx.Response:
            payload = {
                "object": "list",
                "model": "m",
                "data": [{"object": "embedding", "index": i, "embedding": vector} for i, vector in enumerate(current)],
            }
            return httpx.Response(
                200,
                content=json.dumps(payload, allow_nan=True).encode(),
                headers={
                    "content-type": "application/json",
                    "X-OMLXC-Request-ID": "embed-req",
                },
            )

        client = OmlxcClient(transport=httpx.MockTransport(handler))
        with pytest.raises(OmlxcError) as raised:
            await client.embed(model="m", inputs=["x"] * len(vectors), timeout=1)
        assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_stream_parses_utf8_chunks_crlf_multiline_usage_and_done() -> None:
    snowman = "雪".encode()
    body = (
        b'data: {"id":"c","object":"chat.completion.chunk","model":"m",'
        b'"choices":[{"index":0,"delta":{"content":"' + snowman + b'"},"finish_reason":null}]}\r\n\r\n'
        b'data: {"id":"c","object":"chat.completion.chunk","model":"m",\r\n'
        b'data: "choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    stream = ChunkStream((body[:67], body[67:101], body[101:]))

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "req-stream",
                "X-OMLXC-Placement": "placement-a",
                "X-OMLXC-Backend": "backend-a",
            },
            stream=stream,
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    chunks = [
        item async for item in client.stream_chat(model="m", messages=[{"role": "user", "content": "hi"}], timeout=2)
    ]

    assert chunks == [
        OmlxcStreamChunk(
            content="雪",
            model="m",
            request_id="req-stream",
            placement="placement-a",
            backend="backend-a",
        ),
        OmlxcStreamChunk(
            model="m",
            request_id="req-stream",
            placement="placement-a",
            backend="backend-a",
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        ),
    ]
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("with_token", [False, True])
async def test_stream_maps_pre_and_post_token_disconnect_and_closes(with_token: bool) -> None:
    chunks = (
        (b'data: {"object":"chat.completion.chunk","model":"m","choices":[{"delta":{"content":"partial"}}]}\n\n',)
        if with_token
        else ()
    )
    stream = ChunkStream(chunks, error=httpx.ReadError("Authorization: Bearer secret"))

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "stream-req",
                "X-OMLXC-Placement": "placement-a",
                "X-OMLXC-Backend": "backend-a",
            },
            stream=stream,
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    observed: list[OmlxcStreamChunk] = []
    with pytest.raises(OmlxcError) as raised:
        async for chunk in client.stream_chat(
            model="m", messages=[{"role": "user", "content": "secret prompt"}], timeout=1
        ):
            observed.append(chunk)

    assert raised.value.emitted_content is with_token
    assert "secret" not in str(raised.value).lower()
    assert stream.closed
    assert bool(observed) is with_token


@pytest.mark.asyncio
async def test_stream_cancellation_closes_response() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await release.wait()
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            self.closed = True
            release.set()

    stream = BlockingStream()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "stream-req",
                "X-OMLXC-Placement": "placement-a",
                "X-OMLXC-Backend": "backend-a",
            },
            stream=stream,
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))

    async def consume() -> None:
        async for _ in client.stream_chat(model="m", messages=[{"role": "user", "content": "hi"}], timeout=5):
            pass

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type", "code"),
    [
        (409, "insufficient_capacity", OmlxcErrorCode.NO_CAPACITY),
        (504, "timeout", OmlxcErrorCode.TIMEOUT),
        (400, "unsupported_feature", OmlxcErrorCode.INVALID),
        (403, "security", OmlxcErrorCode.SECURITY),
        (503, "backend_unavailable", OmlxcErrorCode.UNAVAILABLE),
        (500, "internal", OmlxcErrorCode.INTERNAL),
    ],
)
async def test_errors_are_typed_and_sanitized(status: int, error_type: str, code: OmlxcErrorCode) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "Authorization: Bearer do-not-leak",
                    "type": error_type,
                    "code": error_type,
                }
            },
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        await client.chat(model="m", messages=[{"role": "user", "content": "secret"}], timeout=1)
    assert raised.value.code is code
    assert "do-not-leak" not in str(raised.value)


@pytest.mark.asyncio
async def test_malformed_envelope_fails_closed() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-OMLXC-Request-ID": "req"},
            json={"schema_version": 2, "request_id": "req", "data": {}},
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        await client.route_plan("m")
    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
async def test_transport_timeout_is_typed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("socket and prompt must not leak", request=request)

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        await client.chat(model="m", messages=[{"role": "user", "content": "secret"}], timeout=1)
    assert raised.value.code is OmlxcErrorCode.TIMEOUT
    assert str(raised.value) == "local inference timed out"


@pytest.mark.asyncio
async def test_malformed_sse_is_invalid_and_closed() -> None:
    stream = ChunkStream((b"data: {not-json}\n\ndata: [DONE]\n\n",))

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "X-OMLXC-Request-ID": "stream-req",
                "X-OMLXC-Placement": "placement-a",
                "X-OMLXC-Backend": "backend-a",
            },
            stream=stream,
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        async for _ in client.stream_chat(model="m", messages=[{"role": "user", "content": "hi"}], timeout=1):
            pass
    assert raised.value.code is OmlxcErrorCode.INVALID
    assert stream.closed


@pytest.mark.asyncio
async def test_client_reuses_transport_until_explicit_close() -> None:
    class ReusableTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        async def handle_async_request(self, _request: httpx.Request) -> httpx.Response:
            assert not self.closed
            self.calls += 1
            return httpx.Response(
                200,
                headers={"X-OMLXC-Request-ID": "req-1"},
                json=_envelope(
                    {
                        "request_id": "req-1",
                        "selected_placement_id": "p",
                        "candidates": ["p"],
                        "candidate_scores": {"p": 1.0},
                        "rejected": {},
                        "fallback_chain": ["p"],
                        "config_version": "v1",
                        "explanation": "selected",
                    }
                ),
            )

        async def aclose(self) -> None:
            self.closed = True

    transport = ReusableTransport()
    client = OmlxcClient(transport=transport)
    await client.route_plan("m")
    await client.route_plan("m")
    assert transport.calls == 2
    assert not transport.closed
    await client.aclose()
    assert transport.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"content-type": "text/plain", "X-OMLXC-Request-ID": "req-1"},
        {"content-type": "application/jsonish", "X-OMLXC-Request-ID": "req-1"},
        {"content-type": "application/json", "X-OMLXC-Request-ID": ""},
    ],
)
async def test_json_success_requires_json_content_type_and_request_id(
    headers: dict[str, str],
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=headers,
            json=_envelope(
                {
                    "request_id": "req-1",
                    "selected_placement_id": "p",
                    "candidates": ["p"],
                    "candidate_scores": {"p": 1.0},
                    "rejected": {},
                    "fallback_chain": ["p"],
                    "config_version": "v1",
                    "explanation": "selected",
                }
            ),
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        await client.route_plan("m")
    assert raised.value.code is OmlxcErrorCode.INVALID
    assert str(raised.value) == "local inference returned an invalid response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"X-OMLXC-Request-ID": "req", "X-OMLXC-Placement": "p"},
        {"X-OMLXC-Request-ID": "req", "X-OMLXC-Backend": "b"},
        {"X-OMLXC-Placement": "p", "X-OMLXC-Backend": "b"},
    ],
)
async def test_chat_success_requires_request_placement_and_backend_headers(
    headers: dict[str, str],
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=headers,
            json={
                "object": "chat.completion",
                "model": "m",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OmlxcError) as raised:
        await client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert raised.value.code is OmlxcErrorCode.INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {
            "content-type": "application/json",
            "X-OMLXC-Request-ID": "r",
            "X-OMLXC-Placement": "p",
            "X-OMLXC-Backend": "b",
        },
        {
            "content-type": "application/x-text/event-stream",
            "X-OMLXC-Request-ID": "r",
            "X-OMLXC-Placement": "p",
            "X-OMLXC-Backend": "b",
        },
        {"content-type": "text/event-stream", "X-OMLXC-Placement": "p", "X-OMLXC-Backend": "b"},
        {"content-type": "text/event-stream", "X-OMLXC-Request-ID": "r", "X-OMLXC-Backend": "b"},
        {"content-type": "text/event-stream", "X-OMLXC-Request-ID": "r", "X-OMLXC-Placement": "p"},
    ],
)
async def test_stream_success_requires_sse_and_all_metadata_before_yield(
    headers: dict[str, str],
) -> None:
    stream = ChunkStream((b'data: {"model":"m","choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n',))

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=stream)

    client = OmlxcClient(transport=httpx.MockTransport(handler))
    observed: list[OmlxcStreamChunk] = []
    with pytest.raises(OmlxcError) as raised:
        async for chunk in client.stream_chat(model="m", messages=[{"role": "user", "content": "hi"}]):
            observed.append(chunk)
    assert raised.value.code is OmlxcErrorCode.INVALID
    assert observed == []
    assert stream.closed
