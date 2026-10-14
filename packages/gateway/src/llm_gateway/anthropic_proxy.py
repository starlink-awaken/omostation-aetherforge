"""Anthropic Messages API facade — exposes ModelGateway as POST /v1/messages.

Any tool using the anthropic Python/TS SDK, or Claude Code itself via
ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, can point at this server:
    client = anthropic.Anthropic(base_url="http://127.0.0.1:9290", api_key="local")

Translates one request/response shape to Chat Completions and reuses the
exact same ModelGateway.generate()/generate_stream() calls as
openai_proxy.handle_chat_completions -- every fix already made there
(circuit isolation, health-failure decay, ...) applies here for free.

Deliberately NOT implemented (fails loud with 400, never silently drops
data a caller would expect to round-trip):
  - prompt caching (`cache_control` on content blocks / system blocks) --
    accepted and ignored, since there is no cache to hit locally; ignoring
    is safe because it only ever affects cost/latency, never correctness.
  - non-text, non-image content blocks (document, search_result, ...) and
    extended-thinking blocks (`thinking`/`redacted_thinking`) on replay --
    dropped the same way openai_proxy drops unsupported Responses input,
    since none of the physical models here consume them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping

from aiohttp import web

from .gateway import DEFAULT_REQUEST_TIMEOUT, GatewayRequest, get_gateway
from .openai_proxy import _close_stream, _omlxc_http_status, _prime_stream

_log = logging.getLogger(__name__)


def _anthropic_error_payload(status: int, message: str) -> dict[str, object]:
    error_type = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        409: "conflict_error",
        413: "request_too_large",
        429: "rate_limit_error",
        504: "timeout_error",
    }.get(status, "api_error")
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _anthropic_block_text(block: Mapping[str, object]) -> str:
    block_type = block.get("type")
    if block_type == "text":
        return str(block.get("text", ""))
    # image / document / search_result / thinking / redacted_thinking / server_tool_use /
    # web_search_tool_result / other future block types: no physical capability to act on
    # them here, and dropping is safer than raising on every multimodal turn that still
    # carries text worth answering.
    return ""


def _anthropic_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_anthropic_block_text(block) for block in content if isinstance(block, Mapping)]
        return "\n".join(part for part in parts if part)
    return ""


def _anthropic_tool_result_text(content: object) -> str:
    """`tool_result.content` is a string, or a list of blocks (usually text,
    sometimes image) mirroring message content -- flatten the same way."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _anthropic_content_to_text(content)
    if content is None:
        return ""
    return json.dumps(content)


def _anthropic_system_to_text(system: object) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return _anthropic_content_to_text(system)
    return ""


def _anthropic_messages_to_chat(
    body: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Translate one Messages API request body into (messages, tools) in
    Chat Completions shape. Raises ValueError (→ 400) for anything genuinely
    malformed, so a caller learns immediately instead of getting a
    silently-wrong answer."""
    messages: list[dict[str, object]] = []
    system = body.get("system")
    system_text = _anthropic_system_to_text(system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be an array")

    for item in raw_messages:
        if not isinstance(item, Mapping):
            raise ValueError("each message must be an object")
        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            # Anthropic's public spec only documents user/assistant in the messages
            # array (system prompts belong in the dedicated top-level `system`
            # field), but real clients (Claude Code itself, for some internal
            # calls like session-title generation) do send role="system" inline.
            # Accepting it is strictly more permissive than the spec, never less.
            raise ValueError(f"unsupported message role: {role!r}")
        content = item.get("content")
        if isinstance(content, str) or not isinstance(content, list):
            messages.append({"role": role, "content": _anthropic_content_to_text(content)})
            continue

        # Block-array content: text/image blocks fold into one text message, but
        # tool_use / tool_result blocks are structurally distinct turns in Chat
        # Completions (assistant tool_calls, tool-role results) and must split out.
        text_parts: list[str] = []
        tool_calls: list[dict[str, object]] = []
        tool_results: list[dict[str, object]] = []
        for block in content:
            if not isinstance(block, Mapping):
                raise ValueError("each content block must be an object")
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _anthropic_tool_result_text(block.get("content")),
                    }
                )
            else:
                text = _anthropic_block_text(block)
                if text:
                    text_parts.append(text)

        if tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        elif text_parts or not tool_results:
            messages.append({"role": role, "content": "\n".join(text_parts)})
        messages.extend(tool_results)

    tools: list[dict[str, object]] = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, Mapping) or "input_schema" not in tool:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return messages, tools


def _anthropic_tool_choice(tool_choice: object) -> object:
    if not isinstance(tool_choice, Mapping):
        return None
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    if kind == "none":
        return "none"
    return None


def _anthropic_content_blocks(resp) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    if resp.content:
        blocks.append({"type": "text", "text": resp.content})
    for call in getattr(resp, "tool_calls", ()) or ():
        fn = call.get("function") if isinstance(call, Mapping) else None
        fn = fn if isinstance(fn, Mapping) else {}
        arguments = fn.get("arguments", "{}")
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id", "") if isinstance(call, Mapping) else "",
                "name": fn.get("name", ""),
                "input": tool_input,
            }
        )
    return blocks


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def _anthropic_stop_reason(resp) -> str:
    if getattr(resp, "tool_calls", ()):
        return "tool_use"
    return _STOP_REASON_MAP.get(resp.finish_reason, "end_turn")


def _anthropic_payload(resp, model: str) -> tuple[dict[str, object], int]:
    blocks = _anthropic_content_blocks(resp)
    if resp.error and not blocks:
        status = _omlxc_http_status(getattr(resp, "error_code", None))
        return _anthropic_error_payload(status, resp.error or "local inference failed"), status
    payload: dict[str, object] = {
        "id": f"msg-aetherforge-{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "model": resp.model or model,
        "content": blocks,
        "stop_reason": _anthropic_stop_reason(resp),
        "stop_sequence": None,
        "usage": {"input_tokens": resp.tokens_in, "output_tokens": resp.tokens_out},
    }
    return payload, 200


def _anthropic_sse_event(event_type: str, data: dict[str, object]) -> bytes:
    encoded = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    return f"event: {event_type}\ndata: {encoded}\n\n".encode()


async def _anthropic_sse(gateway, request: GatewayRequest, model: str, *, source=None, pending_chunks=()):
    """Translate gateway chunks into Messages-API SSE events as they arrive.

    Local backends emit each tool call as one complete chunk rather than
    incremental argument tokens, so a tool_use block's input_json_delta fires
    once with the whole arguments string rather than token-by-token -- still
    spec-shaped, just not incremental for that one block type."""
    from .openai_proxy import _chain_stream

    message_id = f"msg-aetherforge-{int(time.time())}"
    stream = source if source is not None else gateway.generate_stream(request)

    yield _anthropic_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    block_index = 0
    text_open = False
    output_tokens = 0
    stop_reason = "end_turn"
    emitted = False

    try:
        async for chunk in _chain_stream(pending_chunks, stream):
            if chunk.model:
                model = chunk.model
            if chunk.usage is not None:
                output_tokens = int(dict(chunk.usage).get("completion_tokens") or output_tokens)
            if chunk.content:
                emitted = True
                if not text_open:
                    yield _anthropic_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_open = True
                yield _anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": chunk.content},
                    },
                )
            if chunk.tool_calls:
                emitted = True
                stop_reason = "tool_use"
                if text_open:
                    yield _anthropic_sse_event(
                        "content_block_stop", {"type": "content_block_stop", "index": block_index}
                    )
                    text_open = False
                    block_index += 1
                for call in chunk.tool_calls:
                    fn = call.get("function") if isinstance(call, Mapping) else None
                    fn = fn if isinstance(fn, Mapping) else {}
                    call_id = call.get("id", "") if isinstance(call, Mapping) else ""
                    arguments = fn.get("arguments", "{}")
                    yield _anthropic_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": fn.get("name", ""),
                                "input": {},
                            },
                        },
                    )
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": arguments},
                        },
                    )
                    yield _anthropic_sse_event(
                        "content_block_stop", {"type": "content_block_stop", "index": block_index}
                    )
                    block_index += 1
            if chunk.finish_reason == "length":
                stop_reason = "max_tokens"
    except Exception as error:
        status = _omlxc_http_status(getattr(error, "code", None))
        payload = _anthropic_error_payload(status, "local inference failed" if emitted else str(error))
        yield _anthropic_sse_event("error", payload)
        return
    finally:
        await _close_stream(stream)

    if text_open:
        yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})

    yield _anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _anthropic_sse_event("message_stop", {"type": "message_stop"})


async def handle_messages(request: web.Request) -> web.Response:
    """POST /v1/messages — translates to Chat Completions semantics and
    reuses ModelGateway.generate()/generate_stream(). See the module-level
    notes above for what is and is not covered."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response(_anthropic_error_payload(400, "Invalid JSON"), status=400)

    if "max_tokens" not in body:
        return web.json_response(_anthropic_error_payload(400, "max_tokens is required"), status=400)

    try:
        messages, tools = _anthropic_messages_to_chat(body)
    except ValueError as exc:
        return web.json_response(_anthropic_error_payload(400, str(exc)), status=400)

    model = body.get("model", "")
    extra: dict[str, object] = {}
    if tools:
        extra["tools"] = tools
    tool_choice = _anthropic_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    if body.get("stop_sequences"):
        extra["stop"] = body["stop_sequences"]

    gw = get_gateway()
    req = GatewayRequest(
        messages=messages,
        model=model,
        timeout=float(body.get("timeout", DEFAULT_REQUEST_TIMEOUT)),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        task="chat",
        extra=extra,
        routing_mode=str(body.get("routing_mode") or "local"),
    )

    if body.get("stream"):
        source = gw.generate_stream(req)
        pending_chunks, error_response = await _prime_stream(source, req.timeout)
        if error_response is not None:
            status = error_response.status
            message = "local inference failed"
            try:
                payload = json.loads(error_response.body or b"{}")
            except (json.JSONDecodeError, TypeError):
                payload = None
            if isinstance(payload, Mapping):
                inner = payload.get("error")
                if isinstance(inner, Mapping):
                    message = str(inner.get("message", message))
            return web.json_response(_anthropic_error_payload(status, message), status=status)
        return web.Response(
            body=_anthropic_sse(gw, req, model, source=source, pending_chunks=tuple(pending_chunks)),
            status=200,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            content_type="text/event-stream",
        )

    t0 = time.time()
    resp = await gw.generate(req)
    latency = (time.time() - t0) * 1000
    _log.info("messages model=%s latency=%.0fms", getattr(resp, "model", "") or model or "?", latency)

    payload, status = _anthropic_payload(resp, model)
    return web.json_response(payload, status=status)
