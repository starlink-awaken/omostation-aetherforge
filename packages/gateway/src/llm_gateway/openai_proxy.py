"""OpenAI-compatible proxy server — exposes ModelGateway as /v1/* endpoints.

Any tool using the openai Python library can point to this server:
    client = openai.OpenAI(base_url="http://127.0.0.1:9290/v1", api_key="local")

Endpoints:
    POST /v1/chat/completions  → ModelGateway.generate()
    GET  /v1/models            → registry.list_models()
    POST /v1/embeddings        → ModelGateway.embed()
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

from aiohttp import web

from .gateway import GatewayRequest, get_gateway
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
    gateway_fields = {
        "messages",
        "model",
        "temperature",
        "max_tokens",
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
        timeout=float(body.get("timeout", 120)),
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
        pending_chunks = []
        pending_bytes = 0
        try:
            async with asyncio.timeout(max(0.1, req.timeout)):
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
            return web.json_response(
                _openai_error_payload(OmlxcErrorCode.TIMEOUT),
                status=_omlxc_http_status(OmlxcErrorCode.TIMEOUT),
            )
        except OmlxcError as error:
            await _close_stream(source)
            return web.json_response(
                _openai_error_payload(error.code),
                status=_omlxc_http_status(error.code),
            )
        except asyncio.CancelledError:
            await _close_stream(source)
            raise
        except Exception:
            await _close_stream(source)
            return web.json_response(_openai_error_payload(None), status=502)
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
                "finish_reason": resp.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resp.tokens_in,
            "completion_tokens": resp.tokens_out,
            "total_tokens": resp.tokens_in + resp.tokens_out,
        },
        "provider": resp.provider,
    }

    if resp.error and not resp.content:
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


async def handle_list_models(request: web.Request) -> web.Response:
    """GET /v1/models — list all discovered models."""
    gw = get_gateway()
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


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({"status": "ok", "service": "aetherforge-openai-proxy"})


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
    payload = {
        "status": "ready" if resp.content else "not_ready",
        "model": resp.model or model,
        "provider": resp.provider,
        "latency_ms": round(resp.latency_ms, 1),
    }
    if resp.error:
        payload["error"] = resp.error
    return web.json_response(payload, status=200 if resp.content else 503)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Bearer 鉴权。未配 key 时整体放行(仅 loopback 场景, 见 serve 的守卫)。"""
    key = request.app.get(API_KEY)
    if not key or request.path in ("/health", "/"):
        return await handler(request)
    got = request.headers.get("Authorization", "")
    if got.startswith("Bearer "):
        got = got[7:]
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
    app = web.Application(middlewares=[auth_middleware])
    if api_key:
        app[API_KEY] = api_key

    async def _startup(_app: web.Application) -> None:
        await get_gateway().start_background_tasks()

    async def _cleanup(_app: web.Application) -> None:
        await get_gateway().stop_background_tasks()

    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_list_models)
    app.router.add_post("/v1/embeddings", handle_embeddings)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/ready", handle_ready)
    app.router.add_get("/", handle_health)
    return app


def _tailnet_ip() -> str | None:
    """本机的 tailnet 地址。拿不到就返回 None(只绑 loopback, 不猜)。"""
    import shutil
    import subprocess

    exe = shutil.which("tailscale") or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    if not os.path.exists(exe):
        return None
    try:
        out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for ln in out.splitlines():
        ip = ln.strip()
        # tailnet 是 100.64.0.0/10; 只认这个段, 免得把别的网卡地址绑出去
        if ip.startswith("100.") and ip.count(".") == 3:
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
