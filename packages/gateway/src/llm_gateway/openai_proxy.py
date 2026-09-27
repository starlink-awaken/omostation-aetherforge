"""OpenAI-compatible proxy server — exposes ModelGateway as /v1/* endpoints.

Any tool using the openai Python library can point to this server:
    client = openai.OpenAI(base_url="http://127.0.0.1:9290/v1", api_key="local")

Endpoints:
    POST /v1/chat/completions  → ModelGateway.generate()
    GET  /v1/models            → active: omlxcd logical catalog; shadow/legacy: registry
    GET  /v1beta/models        → Gemini-compatible model catalog
    POST /v1beta/models/{model}:generateContent
    POST /v1beta/models/{model}:streamGenerateContent
    POST /v1/embeddings        → ModelGateway.embed()
    GET  /v1/compute           → omlxcd inventory observe (warnings only; not liveness)
    GET  /health               → simple health check

Usage:
    python -m llm_gateway.openai_proxy                    # default :9290
    python -m llm_gateway.openai_proxy --port 9290        # custom port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from urllib.parse import unquote

from aiohttp import web

from .gateway import DEFAULT_REQUEST_TIMEOUT, GatewayRequest, get_gateway
from .omlxc_client import OmlxcError, OmlxcErrorCode

_log = logging.getLogger(__name__)
API_KEY = web.AppKey("aetherforge_api_key", str)
_PRETOKEN_MAX_EVENTS = 64
_PRETOKEN_MAX_BYTES = 64 * 1024

_OMLXC_HTTP_STATUS = {
    OmlxcErrorCode.NO_CAPACITY: 409,
    OmlxcErrorCode.TIMEOUT: 504,
    OmlxcErrorCode.SECURITY: 403,
    OmlxcErrorCode.INVALID: 400,
    OmlxcErrorCode.UNAVAILABLE: 503,
    OmlxcErrorCode.INTERNAL: 502,
}


def _known_omlxc_code(code: object) -> OmlxcErrorCode | None:
    return code if isinstance(code, OmlxcErrorCode) else None


def _omlxc_http_status(code: object) -> int:
    known = _known_omlxc_code(code)
    return _OMLXC_HTTP_STATUS[known] if known is not None else 502


def _openai_error_payload(
    code: object,
    *,
    stream: bool = False,
    emitted_content: bool = False,
) -> dict[str, object]:
    known = _known_omlxc_code(code)
    error: dict[str, object] = {
        "message": str(OmlxcError(known)) if known is not None else "local inference failed",
        "type": "stream_error" if stream else "local_inference_error",
        "code": known.value if known is not None else "internal",
    }
    if stream:
        error["emitted_content"] = emitted_content
    return {"error": error}


async def _close_stream(source: object) -> None:
    close = getattr(source, "aclose", None)
    if close is not None:
        await asyncio.shield(close())


def _bounded_value_size(value: object, limit: int, seen: set[int]) -> int:
    """Estimate JSON-like size without serializing or retaining metadata values."""
    if limit < 0:
        return 1
    if value is None:
        return 4 if limit >= 4 else limit + 1
    if isinstance(value, bool):
        size = 4 if value else 5
        return size if size <= limit else limit + 1
    if isinstance(value, int):
        digits = max(1, (abs(value).bit_length() * 30103) // 100000 + 1)
        size = digits + int(value < 0)
        return size if size <= limit else limit + 1
    if isinstance(value, float):
        return 32 if limit >= 32 else limit + 1
    if isinstance(value, str):
        if len(value) > limit:
            return limit + 1
        size = 2
        for character in value:
            codepoint = ord(character)
            if codepoint < 0x20:
                size += 6
            elif character in {'"', "\\"}:
                size += 2
            elif codepoint <= 0x7F:
                size += 1
            elif codepoint <= 0xFFFF:
                size += 6
            else:
                size += 12
            if size > limit:
                return limit + 1
        return size
    if isinstance(value, bytes):
        return limit + 1

    identity = id(value)
    if identity in seen:
        return limit + 1
    if isinstance(value, Mapping):
        seen.add(identity)
        total = 2
        try:
            for key, nested in value.items():
                if not isinstance(key, str) or total > limit:
                    return limit + 1
                key_size = _bounded_value_size(key, limit - total, seen)
                total += key_size + 1
                if total > limit:
                    return limit + 1
                total += _bounded_value_size(nested, limit - total, seen) + 1
            return total if total <= limit else limit + 1
        finally:
            seen.remove(identity)
    if isinstance(value, Sequence):
        seen.add(identity)
        total = 2
        try:
            for nested in value:
                if total > limit:
                    return limit + 1
                total += _bounded_value_size(nested, limit - total, seen) + 1
            return total if total <= limit else limit + 1
        finally:
            seen.remove(identity)
    return limit + 1


def _pretoken_chunk_size(chunk: object, limit: int) -> int:
    # Fixed envelope overhead plus all values that can later enter the SSE payload.
    total = 128
    for value in (
        getattr(chunk, "content", None),
        getattr(chunk, "model", None),
        getattr(chunk, "request_id", None),
        getattr(chunk, "placement", None),
        getattr(chunk, "backend", None),
        getattr(chunk, "finish_reason", None),
        getattr(chunk, "usage", None),
        getattr(chunk, "tool_calls", None),
    ):
        if total > limit:
            return limit + 1
        total += _bounded_value_size(value, limit - total, set())
    return total if total <= limit else limit + 1


async def _prime_stream(source, timeout: float) -> tuple[list, web.Response | None]:
    """Pull chunks until real content/tool_calls appear or the stream ends, so
    an immediate backend failure surfaces as a normal HTTP error status instead
    of a 200 whose SSE body then errors out mid-stream. Shared by
    /v1/chat/completions and /v1/responses streaming, which differ only in how
    they render the primed chunks into events."""
    pending_chunks: list = []
    pending_bytes = 0
    try:
        async with asyncio.timeout(max(0.1, timeout)):
            while True:
                try:
                    chunk = await anext(source)
                except StopAsyncIteration:
                    break
                remaining = _PRETOKEN_MAX_BYTES - pending_bytes
                chunk_size = _pretoken_chunk_size(chunk, remaining)
                if len(pending_chunks) >= _PRETOKEN_MAX_EVENTS or chunk_size > remaining:
                    raise OmlxcError(OmlxcErrorCode.INTERNAL)
                pending_chunks.append(chunk)
                pending_bytes += chunk_size
                if chunk.content or chunk.tool_calls:
                    break
    except TimeoutError:
        await _close_stream(source)
        return [], web.json_response(
            _openai_error_payload(OmlxcErrorCode.TIMEOUT),
            status=_omlxc_http_status(OmlxcErrorCode.TIMEOUT),
        )
    except OmlxcError as error:
        await _close_stream(source)
        return [], web.json_response(
            _openai_error_payload(error.code),
            status=_omlxc_http_status(error.code),
        )
    except asyncio.CancelledError:
        await _close_stream(source)
        raise
    except Exception:
        await _close_stream(source)
        return [], web.json_response(_openai_error_payload(None), status=502)
    return pending_chunks, None


# ============================================================
# Gemini generateContent compatibility facade
#
# Gemini clients (including agy) use a different wire shape from the
# OpenAI-compatible physical gateway.  Keep this translation layer at the HTTP
# boundary and reuse ModelGateway for all routing, policy and accounting.
# ============================================================

_GEMINI_FINISH_REASONS = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
    "error": "OTHER",
}


class _GeminiRequestError(ValueError):
    """A client-visible Gemini request validation failure."""


def _gemini_error_payload(message: str, status: int, *, status_name: str | None = None) -> dict[str, object]:
    """Build the REST error envelope used by Gemini's generateContent API."""
    return {
        "error": {
            "code": status,
            "message": message,
            "status": status_name
            or {
                400: "INVALID_ARGUMENT",
                401: "UNAUTHENTICATED",
                404: "NOT_FOUND",
                409: "ABORTED",
                502: "BAD_GATEWAY",
                503: "UNAVAILABLE",
                504: "DEADLINE_EXCEEDED",
            }.get(status, "INTERNAL"),
        }
    }


def _gemini_model_from_request(request: web.Request) -> str:
    match_info = getattr(request, "match_info", {})
    model = str(match_info.get("model") or "")
    for suffix in (":generateContent", ":streamGenerateContent"):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break
    model = unquote(model)
    if model.startswith("models/"):
        model = model.removeprefix("models/")
    if not model:
        raise _GeminiRequestError("model is required in the request path")
    return model


def _gemini_text_from_parts(parts: object, *, field: str) -> str:
    if isinstance(parts, Mapping):
        parts = parts.get("parts")
    if not isinstance(parts, list):
        raise _GeminiRequestError(f"{field}.parts must be an array")
    text: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise _GeminiRequestError(f"{field}.parts entries must be objects")
        value = part.get("text")
        if value is not None:
            if not isinstance(value, str):
                raise _GeminiRequestError(f"{field}.parts.text must be a string")
            text.append(value)
            continue
        unsupported = next(
            (key for key in ("inlineData", "fileData", "functionCall", "functionResponse") if key in part),
            None,
        )
        if unsupported:
            raise _GeminiRequestError(f"{field} does not support {unsupported}")
        raise _GeminiRequestError(f"{field}.parts must contain text parts")
    return "\n".join(text)


def _gemini_part_to_openai(part: Mapping[str, object]) -> tuple[object | None, Mapping[str, object] | None, Mapping[str, object] | None]:
    """Translate one Gemini part into content, a tool call, or a tool result."""
    if "text" in part:
        text = part["text"]
        if not isinstance(text, str):
            raise _GeminiRequestError("parts.text must be a string")
        return {"type": "text", "text": text}, None, None

    inline_data = part.get("inlineData")
    if inline_data is not None:
        if not isinstance(inline_data, Mapping):
            raise _GeminiRequestError("inlineData must be an object")
        mime_type = inline_data.get("mimeType")
        data = inline_data.get("data")
        if not isinstance(mime_type, str) or not mime_type:
            raise _GeminiRequestError("inlineData.mimeType is required")
        if not isinstance(data, str) or not data:
            raise _GeminiRequestError("inlineData.data is required")
        if not mime_type.startswith("image/"):
            raise _GeminiRequestError(f"inlineData mime type is not supported locally: {mime_type}")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}}, None, None

    if "fileData" in part:
        raise _GeminiRequestError("fileData is not supported by the local Gemini facade; use inlineData")

    function_call = part.get("functionCall")
    if function_call is not None:
        if not isinstance(function_call, Mapping):
            raise _GeminiRequestError("functionCall must be an object")
        name = function_call.get("name")
        args = function_call.get("args", {})
        if not isinstance(name, str) or not name:
            raise _GeminiRequestError("functionCall.name is required")
        if not isinstance(args, Mapping):
            raise _GeminiRequestError("functionCall.args must be an object")
        return None, {
            "id": f"gemini_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, separators=(",", ":"))},
        }, None

    function_response = part.get("functionResponse")
    if function_response is not None:
        if not isinstance(function_response, Mapping):
            raise _GeminiRequestError("functionResponse must be an object")
        name = function_response.get("name")
        response = function_response.get("response", {})
        if not isinstance(name, str) or not name:
            raise _GeminiRequestError("functionResponse.name is required")
        if not isinstance(response, Mapping):
            raise _GeminiRequestError("functionResponse.response must be an object")
        return None, None, {
            "role": "tool",
            "tool_call_id": f"gemini_{name}",
            "content": json.dumps(response, ensure_ascii=True, separators=(",", ":")),
        }

    raise _GeminiRequestError("each part must contain text, inlineData, functionCall, or functionResponse")


def _gemini_content_to_openai(content: object, index: int) -> list[dict[str, object]]:
    if not isinstance(content, Mapping):
        raise _GeminiRequestError(f"contents[{index}] must be an object")
    role = content.get("role", "user")
    if role == "model":
        openai_role = "assistant"
    elif role == "user":
        openai_role = "user"
    else:
        raise _GeminiRequestError(f"contents[{index}].role must be user or model")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise _GeminiRequestError(f"contents[{index}].parts must be a non-empty array")

    content_parts: list[object] = []
    tool_calls: list[Mapping[str, object]] = []
    messages: list[dict[str, object]] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise _GeminiRequestError(f"contents[{index}].parts entries must be objects")
        translated, tool_call, tool_result = _gemini_part_to_openai(part)
        if translated is not None:
            content_parts.append(translated)
        if tool_call is not None:
            tool_calls.append(tool_call)
        if tool_result is not None:
            messages.append(dict(tool_result))

    if content_parts or tool_calls:
        if all(isinstance(part, Mapping) and part.get("type") == "text" for part in content_parts):
            message_content: object = "\n".join(str(part["text"]) for part in content_parts)
        else:
            message_content = content_parts
        message: dict[str, object] = {"role": openai_role, "content": message_content}
        if tool_calls:
            message["content"] = message_content if content_parts else None
            message["tool_calls"] = tool_calls
        messages.insert(0, message)
    return messages


def _gemini_contents_to_messages(contents: object) -> list[dict[str, object]]:
    if not isinstance(contents, list) or not contents:
        raise _GeminiRequestError("contents must be a non-empty array")
    messages: list[dict[str, object]] = []
    for index, content in enumerate(contents):
        messages.extend(_gemini_content_to_openai(content, index))
    return messages


def _gemini_system_message(system_instruction: object) -> dict[str, object] | None:
    if system_instruction is None:
        return None
    text = _gemini_text_from_parts(system_instruction, field="systemInstruction")
    return {"role": "system", "content": text}


def _gemini_tools(body: Mapping[str, object]) -> list[dict[str, object]]:
    raw_tools = body.get("tools") or []
    if not isinstance(raw_tools, list):
        raise _GeminiRequestError("tools must be an array")
    tools: list[dict[str, object]] = []
    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, Mapping):
            raise _GeminiRequestError(f"tools[{index}] must be an object")
        declarations = raw_tool.get("functionDeclarations")
        if not isinstance(declarations, list):
            raise _GeminiRequestError("only functionDeclarations tools are supported")
        for declaration in declarations:
            if not isinstance(declaration, Mapping):
                raise _GeminiRequestError("functionDeclarations entries must be objects")
            name = declaration.get("name")
            if not isinstance(name, str) or not name:
                raise _GeminiRequestError("function declaration name is required")
            parameters = declaration.get("parameters") or {"type": "object", "properties": {}}
            if not isinstance(parameters, Mapping):
                raise _GeminiRequestError(f"function declaration {name!r} parameters must be an object")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(declaration.get("description") or ""),
                        "parameters": dict(parameters),
                    },
                }
            )
    return tools


def _gemini_tool_choice(body: Mapping[str, object], tools: Sequence[Mapping[str, object]]) -> object | None:
    config = body.get("toolConfig")
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise _GeminiRequestError("toolConfig must be an object")
    calling = config.get("functionCallingConfig")
    if not isinstance(calling, Mapping):
        raise _GeminiRequestError("toolConfig.functionCallingConfig must be an object")
    mode = calling.get("mode", "AUTO")
    if mode == "AUTO":
        choice: object = "auto"
    elif mode == "NONE":
        choice = "none"
    elif mode == "ANY":
        choice = "required"
    else:
        raise _GeminiRequestError(f"unsupported function calling mode: {mode!r}")
    allowed = calling.get("allowedFunctionNames")
    if allowed is not None:
        if not isinstance(allowed, list) or not all(isinstance(name, str) for name in allowed):
            raise _GeminiRequestError("allowedFunctionNames must be an array of strings")
        if len(allowed) > 1:
            raise _GeminiRequestError("multiple allowedFunctionNames are not supported by the local facade")
        if allowed:
            if not any(
                isinstance(tool.get("function"), Mapping) and tool["function"].get("name") == allowed[0]
                for tool in tools
            ):
                raise _GeminiRequestError(f"allowed function is not declared: {allowed[0]!r}")
            choice = {"type": "function", "function": {"name": allowed[0]}}
    return choice


def _gemini_request_to_gateway(body: Mapping[str, object], model: str) -> GatewayRequest:
    if body.get("cachedContent") is not None:
        raise _GeminiRequestError("cachedContent is not supported by the local facade")
    if body.get("safetySettings") is not None:
        raise _GeminiRequestError("safetySettings are not supported by the local facade")
    messages = _gemini_contents_to_messages(body.get("contents"))
    system_message = _gemini_system_message(body.get("systemInstruction"))
    if system_message is not None:
        messages.insert(0, system_message)

    tools = _gemini_tools(body)
    extra: dict[str, object] = {}
    if tools:
        extra["tools"] = tools
    tool_choice = _gemini_tool_choice(body, tools)
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice

    generation = body.get("generationConfig") or {}
    if not isinstance(generation, Mapping):
        raise _GeminiRequestError("generationConfig must be an object")
    temperature = generation.get("temperature")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))):
        raise _GeminiRequestError("generationConfig.temperature must be a number")
    max_tokens = generation.get("maxOutputTokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
        raise _GeminiRequestError("generationConfig.maxOutputTokens must be a positive integer")
    stop_sequences = generation.get("stopSequences")
    if stop_sequences is not None:
        if not isinstance(stop_sequences, list) or not all(isinstance(value, str) for value in stop_sequences):
            raise _GeminiRequestError("generationConfig.stopSequences must be an array of strings")
        extra["stop"] = list(stop_sequences)
    for source, target in (("topP", "top_p"), ("topK", "top_k")):
        value = generation.get(source)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _GeminiRequestError(f"generationConfig.{source} must be a number")
            extra[target] = value

    response_mime = generation.get("responseMimeType")
    if response_mime is not None:
        if response_mime != "application/json":
            raise _GeminiRequestError("only responseMimeType=application/json is supported")
        extra["response_format"] = {"type": "json_object"}
    if generation.get("responseSchema") is not None:
        schema = generation["responseSchema"]
        if not isinstance(schema, Mapping):
            raise _GeminiRequestError("generationConfig.responseSchema must be an object")
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "gemini_response", "schema": dict(schema)},
        }

    return GatewayRequest(
        messages=messages,
        model=model,
        timeout=float(body.get("timeout", DEFAULT_REQUEST_TIMEOUT)),
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
        task="chat",
        extra=extra,
        routing_mode=str(body.get("routing_mode") or "local"),
    )


def _gemini_tool_arguments(call: Mapping[str, object]) -> object:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {}
    raw = function.get("arguments", "{}")
    try:
        parsed = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {"raw_arguments": str(raw)}
    return parsed if isinstance(parsed, Mapping) else {"value": parsed}


def _gemini_finish_reason(reason: object) -> str:
    return _GEMINI_FINISH_REASONS.get(str(reason or "stop"), str(reason).upper())


def _gemini_usage(resp: object) -> dict[str, int]:
    prompt = int(getattr(resp, "tokens_in", 0) or 0)
    completion = int(getattr(resp, "tokens_out", 0) or 0)
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": completion,
        "totalTokenCount": prompt + completion,
    }


def _gemini_response_payload(resp: object, model: str) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    content = str(getattr(resp, "content", "") or "")
    if content:
        parts.append({"text": content})
    for call in getattr(resp, "tool_calls", ()) or ():
        if isinstance(call, Mapping):
            function = call.get("function")
            if isinstance(function, Mapping):
                parts.append(
                    {
                        "functionCall": {
                            "name": str(function.get("name") or ""),
                            "args": _gemini_tool_arguments(call),
                        }
                    }
                )
    candidate: dict[str, object] = {
        "content": {"role": "model", "parts": parts},
        "index": 0,
        "finishReason": _gemini_finish_reason(getattr(resp, "finish_reason", "stop")),
    }
    return {
        "candidates": [candidate],
        "usageMetadata": _gemini_usage(resp),
        "modelVersion": str(getattr(resp, "model", "") or model),
    }


async def _gemini_sse(gateway, request: GatewayRequest, model: str):
    stream = gateway.generate_stream(request)
    emitted = False
    try:
        async for chunk in stream:
            parts: list[dict[str, object]] = []
            if chunk.content:
                parts.append({"text": chunk.content})
                emitted = True
            for call in chunk.tool_calls or ():
                if isinstance(call, Mapping):
                    function = call.get("function")
                    if isinstance(function, Mapping):
                        parts.append(
                            {
                                "functionCall": {
                                    "name": str(function.get("name") or ""),
                                    "args": _gemini_tool_arguments(call),
                                }
                            }
                        )
                        emitted = True
            candidate: dict[str, object] = {
                "content": {"role": "model", "parts": parts},
                "index": 0,
            }
            if chunk.finish_reason:
                candidate["finishReason"] = _gemini_finish_reason(chunk.finish_reason)
            payload: dict[str, object] = {
                "candidates": [candidate],
                "modelVersion": chunk.model or model,
            }
            if chunk.usage is not None:
                usage = dict(chunk.usage)
                payload["usageMetadata"] = {
                    "promptTokenCount": int(usage.get("prompt_tokens") or 0),
                    "candidatesTokenCount": int(usage.get("completion_tokens") or 0),
                    "totalTokenCount": int(usage.get("total_tokens") or 0),
                }
            yield f"data: {json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}\n\n".encode()
    except OmlxcError as error:
        yield f"data: {json.dumps(_gemini_error_payload(str(error), _omlxc_http_status(error.code)), separators=(',', ':'))}\n\n".encode()
    except Exception:
        _log.exception("Gemini stream translation failed emitted=%s", emitted)
        yield f"data: {json.dumps(_gemini_error_payload('local inference failed', 502), separators=(',', ':'))}\n\n".encode()
    finally:
        await _close_stream(stream)


async def handle_gemini_generate_content(request: web.Request) -> web.Response:
    """POST /v1beta/models/{model}:generateContent."""
    try:
        body = await request.json()
        if not isinstance(body, Mapping):
            raise _GeminiRequestError("request body must be an object")
        model = _gemini_model_from_request(request)
        gateway_request = _gemini_request_to_gateway(body, model)
    except _GeminiRequestError as error:
        return web.json_response(_gemini_error_payload(str(error), 400), status=400)
    except Exception:
        return web.json_response(_gemini_error_payload("Invalid JSON", 400), status=400)

    response = await get_gateway().generate(gateway_request)
    if response.error and not response.content and not response.tool_calls:
        code = getattr(response, "error_code", None)
        status = _omlxc_http_status(code)
        return web.json_response(_gemini_error_payload(response.error, status), status=status)
    return web.json_response(_gemini_response_payload(response, model))


async def handle_gemini_stream_generate_content(request: web.Request) -> web.Response:
    """POST /v1beta/models/{model}:streamGenerateContent (SSE)."""
    try:
        body = await request.json()
        if not isinstance(body, Mapping):
            raise _GeminiRequestError("request body must be an object")
        model = _gemini_model_from_request(request)
        gateway_request = _gemini_request_to_gateway(body, model)
    except _GeminiRequestError as error:
        return web.json_response(_gemini_error_payload(str(error), 400), status=400)
    except Exception:
        return web.json_response(_gemini_error_payload("Invalid JSON", 400), status=400)
    return web.Response(
        body=_gemini_sse(get_gateway(), gateway_request, model),
        status=200,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        content_type="text/event-stream",
    )


async def handle_gemini_list_models(request: web.Request) -> web.Response:
    """GET /v1beta/models — expose the gateway catalog in Gemini shape."""
    try:
        models = await get_gateway().list_omlxc_models()
    except OmlxcError as error:
        status = _omlxc_http_status(error.code)
        return web.json_response(_gemini_error_payload(str(error), status), status=status)
    return web.json_response(
        {
            "models": [
                {
                    "name": f"models/{model.id}",
                    "baseModelId": model.id,
                    "version": "aetherforge-local",
                    "displayName": model.id,
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
                }
                for model in models
            ]
        }
    )


async def handle_gemini_get_model(request: web.Request) -> web.Response:
    """GET /v1beta/models/{model} — return one catalog entry."""
    model_id = _gemini_model_from_request(request)
    try:
        models = await get_gateway().list_omlxc_models()
    except OmlxcError as error:
        status = _omlxc_http_status(error.code)
        return web.json_response(_gemini_error_payload(str(error), status), status=status)
    if not any(model.id == model_id for model in models):
        return web.json_response(_gemini_error_payload(f"model {model_id!r} was not found", 404), status=404)
    return web.json_response(
        {
            "name": f"models/{model_id}",
            "baseModelId": model_id,
            "version": "aetherforge-local",
            "displayName": model_id,
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        }
    )


async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions — OpenAI-compatible chat endpoint."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

    messages = body.get("messages", [])
    model = body.get("model", "")
    temperature = body.get("temperature")  # None = 不指定, 用下游默认
    max_tokens = body.get("max_tokens")  # 同上
    if max_tokens is not None and body.get("max_completion_tokens") is not None:
        return web.json_response(
            {"error": {"message": "Conflicting token limits", "type": "invalid_request_error"}},
            status=400,
        )
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    gateway_fields = {
        "messages",
        "model",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "timeout",
        "task",
        "routing_mode",
        "stream",
        "content_title",
        "content_url",
        "extra_body",
    }
    extra = dict(body.get("extra_body") or {})
    # OpenAI SDK 的 extra_body 会摊平进顶层；工具、结构化输出、采样参数等均
    # 原样交给物理引擎，不在门面静默吞掉。
    extra.update({k: v for k, v in body.items() if k not in gateway_fields})

    gw = get_gateway()

    req = GatewayRequest(
        messages=messages,
        model=model,
        timeout=float(body.get("timeout", DEFAULT_REQUEST_TIMEOUT)),
        temperature=temperature,
        max_tokens=max_tokens,
        task=str(body.get("task") or "chat"),
        extra=extra,
        routing_mode=str(body.get("routing_mode") or "local"),
        content_title=str(body.get("content_title") or ""),
        content_url=str(body.get("content_url") or ""),
    )

    if body.get("stream"):
        source = gw.generate_stream(req)
        pending_chunks, error_response = await _prime_stream(source, req.timeout)
        if error_response is not None:
            return error_response
        return web.Response(
            body=_openai_sse(gw, req, source=source, pending_chunks=tuple(pending_chunks)),
            status=200,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            content_type="text/event-stream",
        )

    t0 = time.time()
    resp = await gw.generate(req)
    latency = (time.time() - t0) * 1000
    _log.info(
        "chat.completions model=%s latency=%.0fms",
        getattr(resp, "model", "") or model or "?",
        latency,
    )

    failed = bool(resp.error and not resp.content)
    # Build OpenAI-compatible response
    response_body = {
        "id": f"chatcmpl-aetherforge-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model or model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp.content,
                    **(
                        {"tool_calls": list(getattr(resp, "tool_calls", ()))} if getattr(resp, "tool_calls", ()) else {}
                    ),
                },
                # 失败响应在序列化层统一标 error, 不依赖每个构造点记得设 finish_reason
                "finish_reason": "error" if failed else resp.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resp.tokens_in,
            "completion_tokens": resp.tokens_out,
            "total_tokens": resp.tokens_in + resp.tokens_out,
        },
        "provider": resp.provider,
    }

    if failed:
        code = getattr(resp, "error_code", None)
        response_body.update(_openai_error_payload(code))
        return web.json_response(response_body, status=_omlxc_http_status(code))

    return web.json_response(response_body)


async def _openai_sse(
    gateway,
    request: GatewayRequest,
    *,
    source=None,
    pending_chunks=(),
):
    """Translate gateway chunks as they arrive; cancellation closes the UDS stream."""
    emitted = False
    stream = source if source is not None else gateway.generate_stream(request)
    try:
        async for chunk in _chain_stream(pending_chunks, stream):
            delta: dict[str, object] = {"content": chunk.content}
            if chunk.tool_calls:
                delta["tool_calls"] = list(chunk.tool_calls)
            payload: dict[str, object] = {
                "id": f"chatcmpl-aetherforge-{chunk.request_id or int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chunk.model or request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            if chunk.usage is not None:
                payload["usage"] = dict(chunk.usage)
            emitted = emitted or bool(chunk.content) or bool(chunk.tool_calls)
            encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            yield f"data: {encoded}\n\n".encode()
    except OmlxcError as error:
        payload = _openai_error_payload(
            error.code,
            stream=True,
            emitted_content=emitted or error.emitted_content,
        )
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
        return
    except Exception:
        payload = _openai_error_payload(None, stream=True, emitted_content=emitted)
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
        return
    finally:
        await _close_stream(stream)
    yield b"data: [DONE]\n\n"


async def _chain_stream(first_chunks, source):
    for chunk in first_chunks:
        yield chunk
    async for chunk in source:
        yield chunk


# ============================================================
# /v1/responses — OpenAI Responses API translation
#
# 2026-09-12: Codex CLI (and anything else built against the newer `openai`
# SDK's Responses surface) no longer accepts `wire_api = "chat"`; it requires
# "responses", and this gateway only ever implemented Chat Completions.
# Rather than a second physical inference path, this layer translates one
# request/response shape to the other and reuses the exact same
# ModelGateway.generate()/generate_stream() calls as handle_chat_completions
# -- every fix already made there (circuit isolation, health-failure decay,
# ...) applies here for free. Confirmed live: codex-rs's Responses client
# always sends stream=true (no non-streaming mode to fall back to), so
# streaming is not optional here -- it is what actually unblocks Codex.
#
# Deliberately NOT implemented (fails loud with 400, never silently drops
# data a caller would expect to round-trip):
#   - previous_response_id (server-side conversation state / caching).
#     Codex itself resends full history every turn, so this has not been
#     needed in practice; a client that relies on it must be told, not
#     silently given a fresh context.
#   - non-function tool types (web_search, code_interpreter, computer_use,
#     ...) and non-text input parts (input_image, input_file). Function
#     tools and text are what the physical models here actually serve.
#   - "reasoning" input items (a prior turn's redacted reasoning trace) are
#     accepted and skipped -- they carry no content a Chat Completions
#     message can represent, and codex does not require them to be replayed
#     for the turn to make sense.
# ============================================================


def _responses_content_to_text(content: object) -> str:
    """Flatten a Responses `content` field (str, or a list of typed parts) to
    plain text. Only text-bearing part types are kept; anything else
    (input_image, input_file, ...) is silently dropped -- there is no
    physical capability here to act on it, and dropping is safer than
    raising on every multimodal-capable client that sends an image alongside
    text it still wants answered."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return ""


def _responses_output_to_text(output: object) -> str:
    """`function_call_output.output` is usually a string, but the spec also
    allows a list of content parts (mirroring tool-result content blocks) --
    flatten those the same way as message content."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _responses_content_to_text(output)
    if output is None:
        return ""
    return json.dumps(output)


def _responses_input_to_messages(
    body: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Translate one Responses API request body into (messages, tools) in
    Chat Completions shape. Raises ValueError (→ 400) for anything genuinely
    unsupported, so a caller learns immediately instead of getting a
    silently-wrong answer."""
    if body.get("previous_response_id"):
        raise ValueError(
            "previous_response_id is not supported: this facade keeps no server-side "
            "conversation state, send the full input history on every request"
        )

    messages: list[dict[str, object]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                raise ValueError("each input item must be an object")
            item_type = item.get("type")
            if item_type in (None, "message"):
                role = item.get("role") or "user"
                messages.append({"role": role, "content": _responses_content_to_text(item.get("content"))})
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item.get("call_id") or item.get("id") or "",
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments") or "{}",
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": _responses_output_to_text(item.get("output")),
                    }
                )
            elif item_type == "reasoning":
                continue
            else:
                raise ValueError(f"unsupported input item type: {item_type!r}")
    elif raw_input is not None:
        raise ValueError("input must be a string or an array of items")

    tools: list[dict[str, object]] = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return messages, tools


def _responses_output_items(resp) -> list[dict[str, object]]:
    """Build the `output` array from a GatewayResponse. Tool calls surface
    first (matching the order a Chat Completions message would imply: the
    model decided to call tools, optionally alongside a text remark)."""
    output: list[dict[str, object]] = []
    for call in getattr(resp, "tool_calls", ()) or ():
        fn = call.get("function") if isinstance(call, Mapping) else None
        fn = fn if isinstance(fn, Mapping) else {}
        call_id = call.get("id", "") if isinstance(call, Mapping) else ""
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
                "status": "completed",
            }
        )
    if resp.content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{int(time.time() * 1000)}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": resp.content, "annotations": []}],
            }
        )
    return output


def _responses_payload(resp, model: str) -> dict[str, object]:
    output = _responses_output_items(resp)
    failed = bool(resp.error) and not output
    payload: dict[str, object] = {
        "id": f"resp-aetherforge-{int(time.time())}",
        "object": "response",
        "created_at": int(time.time()),
        "model": resp.model or model,
        "status": "failed" if failed else "completed",
        "output": output,
        "usage": {
            "input_tokens": resp.tokens_in,
            "output_tokens": resp.tokens_out,
            "total_tokens": resp.tokens_in + resp.tokens_out,
        },
    }
    if failed:
        code = getattr(resp, "error_code", None)
        payload["error"] = _openai_error_payload(code)["error"]
    return payload


def _responses_sse_event(event_type: str, data: dict[str, object]) -> bytes:
    encoded = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    return f"event: {event_type}\ndata: {encoded}\n\n".encode()


def _responses_message_close_events(
    item_id: str, output_index: int, text: str
) -> tuple[list[bytes], dict[str, object]]:
    """Shared tail for closing a text message item, whether it closes because
    the stream ended or because a tool call interrupted it. Returns the SSE
    events to emit plus the finalized item (for the eventual response.completed
    payload's `output` array)."""
    item = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    events = [
        _responses_sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        ),
        _responses_sse_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        ),
        _responses_sse_event(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": output_index, "item": item},
        ),
    ]
    return events, item


async def _responses_sse(gateway, request: GatewayRequest, model: str, *, source=None, pending_chunks=()):
    """Translate gateway chunks into Responses-API SSE events as they arrive.

    Local backends emit each tool call as one complete chunk rather than
    incremental argument tokens, so a function_call item's delta/done pair
    fires back-to-back the moment its chunk lands -- still spec-shaped, just
    not token-by-token for that one item type."""
    response_id = f"resp-aetherforge-{int(time.time())}"
    created_at = int(time.time())
    stream = source if source is not None else gateway.generate_stream(request)

    def _skeleton(status: str, output: list) -> dict[str, object]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": status,
            "output": output,
        }

    yield _responses_sse_event(
        "response.created", {"type": "response.created", "response": _skeleton("in_progress", [])}
    )

    output_index = 0
    message_item_id = f"msg_{response_id}"
    text_accum = ""
    message_open = False
    finalized_items: list[dict[str, object]] = []
    usage: dict[str, int] | None = None
    emitted = False

    try:
        async for chunk in _chain_stream(pending_chunks, stream):
            if chunk.model:
                model = chunk.model
            if chunk.usage is not None:
                raw_usage = dict(chunk.usage)
                usage = {
                    "input_tokens": int(raw_usage.get("prompt_tokens") or 0),
                    "output_tokens": int(raw_usage.get("completion_tokens") or 0),
                    "total_tokens": int(raw_usage.get("total_tokens") or 0),
                }
            if chunk.content:
                emitted = True
                if not message_open:
                    yield _responses_sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": message_item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        },
                    )
                    yield _responses_sse_event(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "item_id": message_item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        },
                    )
                    message_open = True
                text_accum += chunk.content
                yield _responses_sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": chunk.content,
                    },
                )
            if chunk.tool_calls:
                emitted = True
                if message_open:
                    events, item = _responses_message_close_events(message_item_id, output_index, text_accum)
                    for event in events:
                        yield event
                    finalized_items.append(item)
                    output_index += 1
                    message_open = False
                    text_accum = ""
                    message_item_id = f"msg_{response_id}_{output_index}"
                for call in chunk.tool_calls:
                    fn = call.get("function") if isinstance(call, Mapping) else None
                    fn = fn if isinstance(fn, Mapping) else {}
                    call_id = call.get("id", "") if isinstance(call, Mapping) else ""
                    item_id = f"fc_{call_id or output_index}"
                    name = fn.get("name", "")
                    arguments = fn.get("arguments", "{}")
                    yield _responses_sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "call_id": call_id,
                                "name": name,
                                "arguments": "",
                            },
                        },
                    )
                    yield _responses_sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": arguments,
                        },
                    )
                    yield _responses_sse_event(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "arguments": arguments,
                        },
                    )
                    tool_item = {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                    yield _responses_sse_event(
                        "response.output_item.done",
                        {"type": "response.output_item.done", "output_index": output_index, "item": tool_item},
                    )
                    finalized_items.append(tool_item)
                    output_index += 1
    except OmlxcError as error:
        payload = _openai_error_payload(error.code, stream=True, emitted_content=emitted or error.emitted_content)
        yield _responses_sse_event(
            "response.failed", {"type": "response.failed", "response": _skeleton("failed", finalized_items), **payload}
        )
        return
    except Exception:
        payload = _openai_error_payload(None, stream=True, emitted_content=emitted)
        yield _responses_sse_event(
            "response.failed", {"type": "response.failed", "response": _skeleton("failed", finalized_items), **payload}
        )
        return
    finally:
        await _close_stream(stream)

    if message_open:
        events, item = _responses_message_close_events(message_item_id, output_index, text_accum)
        for event in events:
            yield event
        finalized_items.append(item)

    final_response = _skeleton("completed", finalized_items)
    final_response["usage"] = usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    yield _responses_sse_event("response.completed", {"type": "response.completed", "response": final_response})


async def handle_responses(request: web.Request) -> web.Response:
    """POST /v1/responses — translates to Chat Completions semantics and
    reuses ModelGateway.generate()/generate_stream(). See the module-level
    notes above for what is and is not covered."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

    try:
        messages, tools = _responses_input_to_messages(body)
    except ValueError as exc:
        return web.json_response({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)

    model = body.get("model", "")
    extra: dict[str, object] = {}
    if tools:
        extra["tools"] = tools
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice

    gw = get_gateway()
    req = GatewayRequest(
        messages=messages,
        model=model,
        timeout=float(body.get("timeout", DEFAULT_REQUEST_TIMEOUT)),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_output_tokens"),
        task="chat",
        extra=extra,
        routing_mode=str(body.get("routing_mode") or "local"),
    )

    if body.get("stream"):
        source = gw.generate_stream(req)
        pending_chunks, error_response = await _prime_stream(source, req.timeout)
        if error_response is not None:
            return error_response
        return web.Response(
            body=_responses_sse(gw, req, model, source=source, pending_chunks=tuple(pending_chunks)),
            status=200,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            content_type="text/event-stream",
        )

    t0 = time.time()
    resp = await gw.generate(req)
    latency = (time.time() - t0) * 1000
    _log.info("responses model=%s latency=%.0fms", getattr(resp, "model", "") or model or "?", latency)

    payload = _responses_payload(resp, model)
    status = 200 if payload["status"] == "completed" else _omlxc_http_status(getattr(resp, "error_code", None))
    return web.json_response(payload, status=status)


async def handle_list_models(request: web.Request) -> web.Response:
    """GET /v1/models — list models executable in the current gateway mode.

    active 模式默认只列 omlxc 本地模型(既有消费者依赖此契约); ?scope=all
    额外合并 registry 云端引擎清单(owned_by=引擎id) —— 否则 22 引擎 369
    模型对客户端完全不可见, 只能手写完整 ID。本地与 registry 重名时本地优先。
    """
    gw = get_gateway()
    if gw._config.omlxc_mode == "active":
        try:
            models = await gw.list_omlxc_models()
        except OmlxcError as error:
            return web.json_response(
                _openai_error_payload(error.code),
                status=_omlxc_http_status(error.code),
            )
        data = [
            {
                "id": model.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "omlxc",
            }
            for model in models
        ]
        # 显式 opt-in 才碰 legacy registry(默认路径保持 fail-closed 契约:
        # active 目录不刷新 registry); getattr 兼容无 query 的最小 request。
        query = getattr(request, "query", None)
        if query is not None and query.get("scope") == "all":
            await gw._ensure_registry_ready()
            seen = {item["id"] for item in data}
            data.extend(
                {
                    "id": m.id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": m.provider,
                }
                for m in gw._registry.list_models()
                if m.id not in seen
            )
        return web.json_response({"object": "list", "data": data})
    await gw._ensure_registry_ready()

    models = gw._registry.list_models()
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": m.provider,
                }
                for m in models
            ],
        }
    )


async def handle_embeddings(request: web.Request) -> web.Response:
    """POST /v1/embeddings — embedding endpoint."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

    texts = body.get("input", [])
    if isinstance(texts, str):
        texts = [texts]

    gw = get_gateway()
    try:
        embeddings = await gw.embed(
            texts,
            model=str(body.get("model") or "embedding"),
            timeout=float(body.get("timeout", 30)),
            routing_mode=str(body.get("routing_mode") or "local"),
            content_title=str(body.get("content_title") or ""),
            content_url=str(body.get("content_url") or ""),
        )
        return web.json_response(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embeddings)],
                "model": body.get("model", "embedding"),
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )
    except OmlxcError as error:
        return web.json_response(
            _openai_error_payload(error.code),
            status=_omlxc_http_status(error.code),
        )
    except Exception:
        return web.json_response(_openai_error_payload(None), status=502)


async def handle_stats(request: web.Request) -> web.Response:
    """GET /stats — 治理观测聚合: metrics/凭据健康/预算/registry/近期事件。"""
    payload = get_gateway().observe_stats()
    return web.json_response(payload)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({"status": "ok", "service": "aetherforge-openai-proxy"})


async def handle_compute(request: web.Request) -> web.Response:
    """GET /v1/compute — omlxcd reachability + inventory_drop warnings."""
    payload = await get_gateway().observe_omlxc_compute()
    return web.json_response(payload)


async def handle_ready(request: web.Request) -> web.Response:
    """GET /ready — 真生成探针，能识别“端口活着但后端卡死”。"""
    model = request.query.get("model", "mythos-fast")
    try:
        timeout = min(30.0, max(1.0, float(request.query.get("timeout", "15"))))
    except ValueError:
        return web.json_response({"error": {"message": "invalid timeout"}}, status=400)
    resp = await get_gateway().generate(
        GatewayRequest(
            messages=[{"role": "user", "content": "Reply with OK only."}],
            model=model,
            max_tokens=8,
            timeout=timeout,
            routing_mode="local",
        )
    )
    # 兜底承接(响应 model ≠ 请求档)时 /ready 此前照报 ready —— 判活方看不出"这个档其实挂了"。
    # 标 degraded + fallback=true; 默认仍 200(备用站本就靠兜底承接, gw-resolve 只关心能否出字),
    # strict=1 时兜底即 503(部署校验/e2e 要求精确到档)。
    served = resp.model or model
    fallback = bool(resp.content) and served != model
    strict = request.query.get("strict", "").lower() in {"1", "true", "yes"}
    if not resp.content:
        status = "not_ready"
    else:
        status = "degraded" if fallback else "ready"
    payload = {
        "status": status,
        "model": served,
        "requested": model,
        "fallback": fallback,
        "provider": resp.provider,
        "latency_ms": round(resp.latency_ms, 1),
    }
    if resp.error:
        payload["error"] = resp.error
    ok = bool(resp.content) and not (strict and fallback)
    return web.json_response(payload, status=200 if ok else 503)


@web.middleware
async def trace_middleware(request: web.Request, handler):
    """P2.3 可追溯: 每请求生成短 trace_id, 响应头返回 + 事件留痕。

    客户端报障时提供 X-Request-ID, 从 events.jsonl 按 trace 检索即可
    对齐该请求的路由与记账(request_complete/request_failed 的 ts+model
    关联), 不再靠时间戳猜。
    """
    import uuid

    from .events import emit

    trace_id = uuid.uuid4().hex[:12]
    request["trace_id"] = trace_id
    resp = await handler(request)
    resp.headers["X-Request-ID"] = trace_id
    if (request.path.startswith("/v1/") or request.path.startswith("/v1beta/")) and request.method == "POST":
        emit("http_request", {"trace": trace_id, "path": request.path, "method": request.method})
    return resp


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Bearer 鉴权。未配 key 时整体放行(仅 loopback 场景, 见 serve 的守卫)。"""
    key = request.app.get(API_KEY)
    path = getattr(request, "path", "")
    if not key or path in ("/health", "/"):
        return await handler(request)
    got = request.headers.get("Authorization", "")
    if got.startswith("Bearer "):
        got = got[7:]
    if not got and path.startswith("/v1beta/"):
        got = request.headers.get("x-goog-api-key", "")
        if not got:
            query = getattr(request, "query", None)
            if query is not None:
                got = query.get("key", "")
    # 常数时间比较, 免得把 key 的前缀通过时间差漏出去
    import hmac

    if not hmac.compare_digest(got, key):
        return web.json_response(
            {"error": {"message": "Unauthorized", "type": "invalid_request_error"}},
            status=401,
        )
    return await handler(request)


def create_app(api_key: str | None = None) -> web.Application:
    """Create the aiohttp application."""
    # Deferred import: anthropic_proxy imports shared streaming helpers back
    # from this module, so a top-level import here would be circular.
    from .anthropic_proxy import handle_messages

    app = web.Application(middlewares=[trace_middleware, auth_middleware])
    if api_key:
        app[API_KEY] = api_key

    async def _startup(_app: web.Application) -> None:
        await get_gateway().start_background_tasks()

    async def _cleanup(_app: web.Application) -> None:
        await get_gateway().stop_background_tasks()

    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/responses", handle_responses)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_get("/v1/models", handle_list_models)
    app.router.add_post("/v1/embeddings", handle_embeddings)
    # 图像生成 / 语音合成 / 语音识别 → 本地多媒体后端(Unsloth Studio), 别名与文本路由共用一张表
    from .media_proxy import register_media_routes

    register_media_routes(app, resolve_alias=lambda name: get_gateway().resolve_alias(name))
    app.router.add_post("/v1beta/models/{model:.*}:generateContent", handle_gemini_generate_content)
    app.router.add_post("/v1beta/models/{model:.*}:streamGenerateContent", handle_gemini_stream_generate_content)
    app.router.add_get("/v1beta/models", handle_gemini_list_models)
    app.router.add_get("/v1beta/models/{model:.*}", handle_gemini_get_model)
    app.router.add_get("/v1/compute", handle_compute)
    app.router.add_get("/stats", handle_stats)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/ready", handle_ready)
    app.router.add_get("/", handle_health)
    return app


def _tailnet_ip() -> str | None:
    """本机的 tailnet 地址。拿不到就返回 None(只绑 loopback, 不猜)。

    先问 tailscale CLI; 问不到再读网卡。CLI 不可靠: Homebrew 版 tailscaled 常跑在
    非默认 socket 上, App 版的 CLI 又不在 PATH —— 此时 `tailscale ip` 失败, 但
    utun 网卡上的地址是真的。两条路都只认 100.64.0.0/10。
    """
    import ipaddress
    import re
    import shutil
    import subprocess

    tailnet = ipaddress.ip_network("100.64.0.0/10")

    def first_tailnet(text: str) -> str | None:
        for cand in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            try:
                if ipaddress.ip_address(cand) in tailnet:
                    return cand
            except ValueError:
                pass
        return None

    probes = [[exe, "ip", "-4"] for exe in (shutil.which("tailscale"),) if exe]
    probes.append(["ifconfig"])
    for cmd in probes:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        # ifconfig 里 inet 行形如 "inet 100.68.80.44 --> 100.68.80.44", 只取 inet 后的本端地址
        text = out if cmd[0] != "ifconfig" else "\n".join(re.findall(r"\binet (\S+)", out))
        ip = first_tailnet(text)
        if ip:
            return ip
    return None


def resolve_bind_hosts(bind: str) -> list[str]:
    """把 --bind 选项翻成具体地址列表。

    local   只有 127.0.0.1(默认, 最安全)
    tailnet 127.0.0.1 + 本机 tailnet 地址 —— 给跨机调用方用, 但**不含**
            其它局域网段; 拿不到 tailnet 地址时退回 local 并如实告警。
    其它值  当成字面地址, 逗号分隔。
    """
    if bind == "local":
        return ["127.0.0.1"]
    if bind == "tailnet":
        ip = _tailnet_ip()
        if not ip:
            _log.warning("拿不到 tailnet 地址, 退回只绑 127.0.0.1")
            return ["127.0.0.1"]
        return ["127.0.0.1", ip]
    return [h.strip() for h in bind.split(",") if h.strip()]


def parse_ports(spec: str | int) -> list[int]:
    """ "9290" / "9290,4000" / 9290 → [9290] / [9290, 4000] / [9290]"""
    if isinstance(spec, int):
        return [spec]
    out: list[int] = []
    for p in str(spec).split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out or [9290]


async def _run_sites(app: web.Application, hosts: list[str], ports: list[int]) -> None:
    """在 hosts × ports 的每个组合上开一个 site, 然后一直挂着。"""
    import asyncio

    runner = web.AppRunner(app)
    await runner.setup()
    for h in hosts:
        for p in ports:
            await web.TCPSite(runner, h, p).start()
    await asyncio.Event().wait()  # 交给信号处理去中断


def serve(port: int | str = 9290, bind: str = "local") -> None:
    """Start the OpenAI-compatible proxy server."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    _log.info("Starting AetherForge OpenAI proxy on :%s", port)
    _log.info("  POST /v1/chat/completions  — LLM inference")
    _log.info("  GET  /v1/models            — list models")
    _log.info("  GET  /v1beta/models        — Gemini-compatible model catalog")
    _log.info("  POST /v1beta/models/{model}:generateContent — Gemini inference")
    _log.info("  GET  /v1/compute           — omlxc inventory observe")
    _log.info("  POST /v1/embeddings        — embeddings")
    import asyncio

    hosts = resolve_bind_hosts(bind)
    ports = parse_ports(port)
    api_key = os.environ.get("AETHERFORGE_API_KEY") or None
    non_loopback = [h for h in hosts if h not in ("127.0.0.1", "::1", "localhost")]
    if non_loopback and not api_key:
        raise SystemExit(
            f"拒绝启动: 要绑到 {', '.join(non_loopback)} (loopback 之外), 但没配 "
            "AETHERFORGE_API_KEY。\n"
            "  绑出去就等于把本机全部模型对该网段敞开, 不能无鉴权裸奔。\n"
            "  要么设 AETHERFORGE_API_KEY=<key>, 要么用 --bind local。"
        )
    for h in hosts:
        for p in ports:
            _log.info("  base_url=http://%s:%d/v1", h, p)
    _log.info("  鉴权: %s", "Bearer key 已启用" if api_key else "无(仅 loopback)")
    try:
        asyncio.run(_run_sites(create_app(api_key), hosts, ports))
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherForge OpenAI-compatible proxy")
    parser.add_argument(
        "--port",
        default=os.environ.get("AETHERFORGE_PORTS", "9290"),
        help="端口, 逗号分隔可多个。过渡期用 9290,4000 接管 LiteLLM 的位置。",
    )
    parser.add_argument(
        "--bind",
        default=os.environ.get("AETHERFORGE_BIND", "local"),
        help="local(仅 127.0.0.1, 默认) | tailnet(127.0.0.1 + 本机 tailnet 地址) | 逗号分隔的字面地址",
    )
    args = parser.parse_args(argv)
    serve(args.port, args.bind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
