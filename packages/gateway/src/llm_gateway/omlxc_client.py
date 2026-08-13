"""Typed, bounded client for the private omlxcd Unix-socket API."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import httpx

_BASE_URL = "http://omlxc"
_SCHEMA_VERSION = 1
_MAX_SSE_EVENT_BYTES = 1_048_576
_CATALOG_PAGE_LIMIT = 100
_MAX_CATALOG_PAGES = 100


class OmlxcErrorCode(StrEnum):
    UNAVAILABLE = "unavailable"
    NO_CAPACITY = "no_capacity"
    TIMEOUT = "timeout"
    SECURITY = "security"
    INVALID = "invalid"
    INTERNAL = "internal"


_SAFE_MESSAGES = {
    OmlxcErrorCode.UNAVAILABLE: "local inference unavailable",
    OmlxcErrorCode.NO_CAPACITY: "local inference has no capacity",
    OmlxcErrorCode.TIMEOUT: "local inference timed out",
    OmlxcErrorCode.SECURITY: "local inference rejected by security policy",
    OmlxcErrorCode.INVALID: "local inference returned an invalid response",
    OmlxcErrorCode.INTERNAL: "local inference failed",
}


class OmlxcError(RuntimeError):
    """Sanitized typed failure from the private local-compute boundary."""

    def __init__(self, code: OmlxcErrorCode, *, emitted_content: bool = False) -> None:
        self.code = code
        self.emitted_content = emitted_content
        super().__init__(_SAFE_MESSAGES[code])

    @property
    def cloud_fallback_allowed(self) -> bool:
        return self.code in {
            OmlxcErrorCode.UNAVAILABLE,
            OmlxcErrorCode.NO_CAPACITY,
            OmlxcErrorCode.TIMEOUT,
        }


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class OmlxcRoutePlan:
    request_id: str
    selected: str | None
    candidates: tuple[str, ...]
    scores: Mapping[str, float]
    rejected: Mapping[str, str]
    fallback: tuple[str, ...]
    config_version: str
    explanation: str
    thinking_authorized: bool = False


@dataclass(frozen=True)
class OmlxcCatalogModel:
    """A logical model advertised by the private omlxcd control plane."""

    id: str
    capabilities: frozenset[str]


@dataclass(frozen=True)
class OmlxcChatResult:
    content: str
    model: str
    finish_reason: str
    usage: TokenUsage
    request_id: str
    placement: str | None = None
    backend: str | None = None
    tool_calls: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class OmlxcStreamChunk:
    content: str = ""
    model: str = ""
    request_id: str = ""
    placement: str | None = None
    backend: str | None = None
    finish_reason: str | None = None
    usage: Mapping[str, int] | None = field(default=None)
    tool_calls: tuple[Mapping[str, object], ...] = ()


def default_omlxc_socket() -> Path:
    override = os.environ.get("OMLXC_SOCKET")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/omlxc/omlxcd.sock"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "omlxc/omlxcd.sock"


class OmlxcClient:
    """Calls omlxcd without exposing its socket or physical topology upstream."""

    def __init__(
        self,
        socket_path: str | Path | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Any | None = None,
    ) -> None:
        self.socket_path = Path(socket_path) if socket_path is not None else default_omlxc_socket()
        self._transport = transport or httpx.AsyncHTTPTransport(uds=str(self.socket_path))
        self._clock = clock
        self._http = httpx.AsyncClient(transport=self._transport, base_url=_BASE_URL)

    @staticmethod
    def _timeout(timeout: float) -> httpx.Timeout:
        bounded = max(0.1, min(float(timeout), 3600.0))
        return httpx.Timeout(bounded)

    async def aclose(self) -> None:
        await self._http.aclose()

    @asynccontextmanager
    async def _client(self, _timeout: float) -> AsyncIterator[httpx.AsyncClient]:
        """Borrow the long-lived pool without closing it after one request."""
        yield self._http

    async def route_plan(
        self,
        model_id: str,
        *,
        profile: str = "interactive",
        capabilities: set[str] | frozenset[str] = frozenset({"chat"}),
        context_tokens: int = 0,
        thinking: bool = False,
        timeout: float = 2.0,
    ) -> OmlxcRoutePlan:
        payload = {
            "model_id": model_id,
            "profile": profile,
            "required_capabilities": sorted(capabilities),
            "context_tokens": context_tokens,
            "thinking_requested": thinking,
        }
        response = await self._post_json("/api/v1/routes/plan", payload, timeout)
        envelope = _envelope(response)
        if _required_header(response, "X-OMLXC-Request-ID") != envelope["request_id"]:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        data = _mapping(envelope["data"])
        required = {
            "request_id",
            "selected_placement_id",
            "candidates",
            "candidate_scores",
            "rejected",
            "fallback_chain",
            "config_version",
            "explanation",
        }
        if not required.issubset(data):
            raise OmlxcError(OmlxcErrorCode.INVALID)
        request_id = _string(data["request_id"])
        if request_id != envelope["request_id"]:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        selected_raw = data["selected_placement_id"]
        selected = None if selected_raw is None else _string(selected_raw)
        return OmlxcRoutePlan(
            request_id=request_id,
            selected=selected,
            candidates=_strings(data["candidates"]),
            scores=_finite_float_mapping(data["candidate_scores"]),
            rejected=_string_mapping(data["rejected"]),
            fallback=_strings(data["fallback_chain"]),
            config_version=_string(data["config_version"]),
            explanation=_string(data["explanation"]),
            thinking_authorized=_bool(data.get("thinking_authorized", False)),
        )

    async def list_models(self, *, timeout: float = 2.0) -> tuple[OmlxcCatalogModel, ...]:
        """List the logical models that omlxcd can accept in active mode."""
        bounded_timeout = max(0.1, min(float(timeout), 3600.0))
        try:
            async with asyncio.timeout(bounded_timeout):
                cursor: str | None = None
                seen_cursors: set[str] = set()
                seen_model_ids: set[str] = set()
                models: list[OmlxcCatalogModel] = []
                for _page in range(_MAX_CATALOG_PAGES):
                    params: dict[str, str | int] = {"limit": _CATALOG_PAGE_LIMIT}
                    if cursor is not None:
                        params["after"] = cursor
                    response = await self._get_json("/api/v1/models", timeout, params=params)
                    envelope = _envelope(response)
                    if _required_header(response, "X-OMLXC-Request-ID") != envelope["request_id"]:
                        raise OmlxcError(OmlxcErrorCode.INVALID)
                    data = _mapping(envelope["data"])
                    for item in _sequence(data.get("items")):
                        model = OmlxcCatalogModel(
                            id=_nonempty_string(_mapping(item).get("id")),
                            capabilities=frozenset(_strings(_mapping(item).get("capabilities", ()))),
                        )
                        if model.id in seen_model_ids:
                            raise OmlxcError(OmlxcErrorCode.INVALID)
                        seen_model_ids.add(model.id)
                        models.append(model)
                    next_cursor = data.get("next_cursor")
                    if next_cursor is None:
                        return tuple(models)
                    cursor = _nonempty_string(next_cursor)
                    if cursor in seen_cursors:
                        raise OmlxcError(OmlxcErrorCode.INVALID)
                    seen_cursors.add(cursor)
        except TimeoutError as exc:
            raise OmlxcError(OmlxcErrorCode.TIMEOUT) from exc
        raise OmlxcError(OmlxcErrorCode.INVALID)

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        profile: str = "interactive",
        thinking: bool = False,
        tools: Sequence[Mapping[str, object]] | None = None,
        tool_choice: object | None = None,
    ) -> OmlxcChatResult:
        payload = _chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            profile=profile,
            thinking=thinking,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
        )
        response = await self._post_json("/openai/v1/chat/completions", payload, timeout)
        body = _response_mapping(response)
        choices = _sequence(body.get("choices"))
        if len(choices) != 1:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        choice = _mapping(choices[0])
        message = _mapping(choice.get("message"))
        content = _string(message.get("content"))
        tool_calls = _tool_calls(message.get("tool_calls"))
        if not content and not tool_calls:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        usage = _usage(body.get("usage"))
        request_id = _required_header(response, "X-OMLXC-Request-ID")
        placement = _required_header(response, "X-OMLXC-Placement")
        backend = _required_header(response, "X-OMLXC-Backend")
        return OmlxcChatResult(
            content=content,
            tool_calls=tool_calls,
            model=_string(body.get("model")),
            finish_reason=_string(choice.get("finish_reason")),
            usage=usage,
            request_id=request_id,
            placement=placement,
            backend=backend,
        )

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        profile: str = "interactive",
        thinking: bool = False,
        tools: Sequence[Mapping[str, object]] | None = None,
        tool_choice: object | None = None,
    ) -> AsyncIterator[OmlxcStreamChunk]:
        payload = _chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            profile=profile,
            thinking=thinking,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
        )
        emitted = False
        saw_done = False
        try:
            async with self._client(timeout) as client:
                async with client.stream(
                    "POST",
                    "/openai/v1/chat/completions",
                    json=payload,
                    timeout=self._timeout(timeout),
                ) as response:
                    if response.status_code >= 400:
                        await _raise_http_error(response)
                    if _media_type(response) != "text/event-stream":
                        raise OmlxcError(OmlxcErrorCode.INVALID)
                    metadata = {
                        "request_id": _required_header(response, "X-OMLXC-Request-ID"),
                        "placement": _required_header(response, "X-OMLXC-Placement"),
                        "backend": _required_header(response, "X-OMLXC-Backend"),
                    }
                    async for data in _sse_data(response.aiter_bytes()):
                        if data == "[DONE]":
                            saw_done = True
                            break
                        try:
                            payload_obj = json.loads(data)
                        except (json.JSONDecodeError, UnicodeError) as exc:
                            raise OmlxcError(OmlxcErrorCode.INVALID, emitted_content=emitted) from exc
                        body = _mapping(payload_obj)
                        if "error" in body:
                            raise OmlxcError(_payload_error_code(body["error"]), emitted_content=emitted)
                        choices = _sequence(body.get("choices", []))
                        usage_value = body.get("usage")
                        content = ""
                        tool_calls: tuple[Mapping[str, object], ...] = ()
                        finish_reason: str | None = None
                        if choices:
                            if len(choices) != 1:
                                raise OmlxcError(OmlxcErrorCode.INVALID, emitted_content=emitted)
                            choice = _mapping(choices[0])
                            delta = _mapping(choice.get("delta", {}))
                            raw_content = delta.get("content", "")
                            content = _string(raw_content)
                            tool_calls = _tool_call_deltas(delta.get("tool_calls"))
                            raw_finish = choice.get("finish_reason")
                            finish_reason = None if raw_finish is None else _string(raw_finish)
                        usage = None
                        if usage_value is not None:
                            parsed = _usage(usage_value)
                            usage = {
                                "prompt_tokens": parsed.prompt_tokens,
                                "completion_tokens": parsed.completion_tokens,
                                "total_tokens": parsed.total_tokens,
                            }
                        if not content and not tool_calls and usage is None and finish_reason is None:
                            continue
                        emitted = emitted or bool(content) or bool(tool_calls)
                        yield OmlxcStreamChunk(
                            content=content,
                            model=_string(body.get("model", model)),
                            finish_reason=finish_reason,
                            usage=usage,
                            tool_calls=tool_calls,
                            **metadata,
                        )
                    if not saw_done:
                        raise OmlxcError(OmlxcErrorCode.UNAVAILABLE, emitted_content=emitted)
        except OmlxcError:
            raise
        except httpx.TimeoutException as exc:
            raise OmlxcError(OmlxcErrorCode.TIMEOUT, emitted_content=emitted) from exc
        except httpx.HTTPError as exc:
            raise OmlxcError(OmlxcErrorCode.UNAVAILABLE, emitted_content=emitted) from exc

    async def embed(
        self,
        *,
        model: str,
        inputs: list[str],
        timeout: float = 120.0,
        profile: str = "interactive",
    ) -> list[list[float]]:
        response = await self._post_json(
            "/openai/v1/embeddings",
            {
                "model": model,
                "input": inputs,
                "profile": profile,
                "timeout_seconds": timeout,
            },
            timeout,
        )
        body = _response_mapping(response)
        rows = _sequence(body.get("data"))
        if len(rows) != len(inputs):
            raise OmlxcError(OmlxcErrorCode.INVALID)
        indexed: dict[int, list[float]] = {}
        width: int | None = None
        for item in rows:
            row = _mapping(item)
            index = _integer(row.get("index"))
            raw_vector = _sequence(row.get("embedding"))
            vector = [_finite_float(value) for value in raw_vector]
            if not vector or index in indexed or not 0 <= index < len(inputs):
                raise OmlxcError(OmlxcErrorCode.INVALID)
            width = len(vector) if width is None else width
            if len(vector) != width:
                raise OmlxcError(OmlxcErrorCode.INVALID)
            indexed[index] = vector
        if set(indexed) != set(range(len(inputs))):
            raise OmlxcError(OmlxcErrorCode.INVALID)
        return [indexed[index] for index in range(len(inputs))]

    async def _post_json(self, path: str, payload: Mapping[str, object], timeout: float) -> httpx.Response:
        try:
            response = await self._http.post(path, json=payload, timeout=self._timeout(timeout))
            if response.status_code >= 400:
                await _raise_http_error(response)
            if _media_type(response) != "application/json":
                raise OmlxcError(OmlxcErrorCode.INVALID)
            _required_header(response, "X-OMLXC-Request-ID")
            return response
        except OmlxcError:
            raise
        except httpx.TimeoutException as exc:
            raise OmlxcError(OmlxcErrorCode.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise OmlxcError(OmlxcErrorCode.UNAVAILABLE) from exc

    async def _get_json(
        self, path: str, timeout: float, *, params: Mapping[str, str | int] | None = None
    ) -> httpx.Response:
        try:
            response = await self._http.get(path, params=params, timeout=self._timeout(timeout))
            if response.status_code >= 400:
                await _raise_http_error(response)
            if _media_type(response) != "application/json":
                raise OmlxcError(OmlxcErrorCode.INVALID)
            _required_header(response, "X-OMLXC-Request-ID")
            return response
        except OmlxcError:
            raise
        except httpx.TimeoutException as exc:
            raise OmlxcError(OmlxcErrorCode.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise OmlxcError(OmlxcErrorCode.UNAVAILABLE) from exc


def _chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
    profile: str,
    thinking: bool,
    stream: bool,
    tools: Sequence[Mapping[str, object]] | None,
    tool_choice: object | None,
) -> dict[str, object]:
    _validate_agent_fields(tools, tool_choice)
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "profile": profile,
        "thinking": thinking,
        "timeout_seconds": timeout,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = list(tools)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def _validate_agent_fields(
    tools: Sequence[Mapping[str, object]] | None,
    tool_choice: object | None,
) -> None:
    if tools is not None:
        if not tools or len(tools) > 128:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        try:
            encoded = json.dumps(tools, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise OmlxcError(OmlxcErrorCode.INVALID) from exc
        if len(encoded) > 512_000:
            raise OmlxcError(OmlxcErrorCode.INVALID)
    if tool_choice is not None and not tools:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    if tool_choice is not None and not isinstance(tool_choice, (str, Mapping)):
        raise OmlxcError(OmlxcErrorCode.INVALID)


def _tool_calls(value: object) -> tuple[Mapping[str, object], ...]:
    calls = _sequence(value) if value is not None else ()
    if len(calls) > 128:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    result: list[Mapping[str, object]] = []
    for item in calls:
        call = _mapping(item)
        function = _mapping(call.get("function"))
        call_id = call.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if (
            not isinstance(call_id, str)
            or not 1 <= len(call_id) <= 256
            or call.get("type") != "function"
            or not isinstance(name, str)
            or not 1 <= len(name) <= 128
            or not isinstance(arguments, str)
            or len(arguments) > 262_144
        ):
            raise OmlxcError(OmlxcErrorCode.INVALID)
        try:
            if not isinstance(json.loads(arguments), dict):
                raise OmlxcError(OmlxcErrorCode.INVALID)
        except json.JSONDecodeError as exc:
            raise OmlxcError(OmlxcErrorCode.INVALID) from exc
        result.append(call)
    return tuple(result)


def _tool_call_deltas(value: object) -> tuple[Mapping[str, object], ...]:
    calls = _sequence(value) if value is not None else ()
    if len(calls) > 128:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    result: list[Mapping[str, object]] = []
    for item in calls:
        call = _mapping(item)
        index = call.get("index")
        call_id = call.get("id")
        call_type = call.get("type")
        function = call.get("function")
        if not isinstance(index, int) or not 0 <= index <= 127:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        if call_id is not None and (not isinstance(call_id, str) or not 1 <= len(call_id) <= 256):
            raise OmlxcError(OmlxcErrorCode.INVALID)
        if call_type is not None and call_type != "function":
            raise OmlxcError(OmlxcErrorCode.INVALID)
        if function is not None:
            typed_function = _mapping(function)
            name = typed_function.get("name")
            arguments = typed_function.get("arguments")
            if name is not None and (not isinstance(name, str) or not 1 <= len(name) <= 128):
                raise OmlxcError(OmlxcErrorCode.INVALID)
            if arguments is not None and (not isinstance(arguments, str) or len(arguments) > 262_144):
                raise OmlxcError(OmlxcErrorCode.INVALID)
        result.append(call)
    return tuple(result)


async def _raise_http_error(response: httpx.Response) -> None:
    try:
        if response.is_stream_consumed:
            raw = response.content
        else:
            chunks: list[bytes] = []
            size = 0
            oversized = False
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 64 * 1024:
                    oversized = True
                    break
                chunks.append(chunk)
            raw = b"" if oversized else b"".join(chunks)
        body = {} if not raw else _mapping(json.loads(raw))
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError):
        body = {}
    error = body.get("error")
    raise OmlxcError(_http_error_code(response.status_code, error))


def _http_error_code(status: int, error: object) -> OmlxcErrorCode:
    if status in {401, 403}:
        return OmlxcErrorCode.SECURITY
    if status == 408 or status == 504:
        return OmlxcErrorCode.TIMEOUT
    if status == 409:
        return OmlxcErrorCode.NO_CAPACITY
    if status in {400, 404, 413, 422}:
        return OmlxcErrorCode.INVALID
    if status == 503:
        return _payload_error_code(error)
    return OmlxcErrorCode.INTERNAL if status >= 500 else OmlxcErrorCode.INVALID


def _payload_error_code(error: object) -> OmlxcErrorCode:
    mapping = _mapping_or_empty(error)
    raw = str(mapping.get("type") or mapping.get("code") or "").lower()
    if raw in {"insufficient_capacity", "no_capacity", "no_candidate"}:
        return OmlxcErrorCode.NO_CAPACITY
    if raw == "timeout":
        return OmlxcErrorCode.TIMEOUT
    if raw in {"security", "forbidden", "unauthorized"}:
        return OmlxcErrorCode.SECURITY
    if raw in {"unsupported_feature", "invalid_request", "validation_error"}:
        return OmlxcErrorCode.INVALID
    if raw in {"backend_unavailable", "unavailable", "stream_error"}:
        return OmlxcErrorCode.UNAVAILABLE
    return OmlxcErrorCode.INTERNAL


def _envelope(response: httpx.Response) -> Mapping[str, object]:
    body = _response_mapping(response)
    if set(body) != {"schema_version", "request_id", "data"}:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    if body["schema_version"] != _SCHEMA_VERSION or not _string(body["request_id"]):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return body


def _response_mapping(response: httpx.Response) -> Mapping[str, object]:
    try:
        return _mapping(response.json())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OmlxcError(OmlxcErrorCode.INVALID) from exc


def _required_header(response: httpx.Response, name: str) -> str:
    value = response.headers.get(name, "").strip()
    if not value:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return value


def _media_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").partition(";")[0].strip().lower()


async def _sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    buffer = bytearray()
    data_lines: list[bytes] = []
    async for chunk in chunks:
        buffer.extend(chunk)
        if len(buffer) > _MAX_SSE_EVENT_BYTES:
            raise OmlxcError(OmlxcErrorCode.INVALID)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                if data_lines:
                    try:
                        yield b"\n".join(data_lines).decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise OmlxcError(OmlxcErrorCode.INVALID) from exc
                    data_lines.clear()
                continue
            if line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
                if sum(map(len, data_lines)) > _MAX_SSE_EVENT_BYTES:
                    raise OmlxcError(OmlxcErrorCode.INVALID)
    if buffer or data_lines:
        raise OmlxcError(OmlxcErrorCode.UNAVAILABLE)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return cast(Mapping[str, object], value)


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return value


def _nonempty_string(value: object) -> str:
    result = _string(value)
    if not result:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return result


def _strings(value: object) -> tuple[str, ...]:
    return tuple(_string(item) for item in _sequence(value))


def _bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return value


def _finite_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    number = float(value)
    if not math.isfinite(number):
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return number


def _finite_float_mapping(value: object) -> dict[str, float]:
    return {str(key): _finite_float(item) for key, item in _mapping(value).items()}


def _string_mapping(value: object) -> dict[str, str]:
    return {str(key): _string(item) for key, item in _mapping(value).items()}


def _usage(value: object) -> TokenUsage:
    data = _mapping(value)
    prompt = _integer(data.get("prompt_tokens", 0))
    completion = _integer(data.get("completion_tokens", 0))
    total = _integer(data.get("total_tokens", prompt + completion))
    if min(prompt, completion, total) < 0 or total < prompt + completion:
        raise OmlxcError(OmlxcErrorCode.INVALID)
    return TokenUsage(prompt, completion, total)
