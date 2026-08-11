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
import json
import logging
import os
import time

from aiohttp import web

from .gateway import GatewayRequest, get_gateway
from .omlxc_client import OmlxcError

_log = logging.getLogger(__name__)
API_KEY = web.AppKey("aetherforge_api_key", str)


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
        return web.Response(
            body=_openai_sse(gw, req),
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
                "message": {"role": "assistant", "content": resp.content},
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
        response_body["error"] = {"message": resp.error}
        return web.json_response(response_body, status=502)

    return web.json_response(response_body)


async def _openai_sse(gateway, request: GatewayRequest):
    """Translate gateway chunks as they arrive; cancellation closes the UDS stream."""
    emitted = False
    try:
        async for chunk in gateway.generate_stream(request):
            payload: dict[str, object] = {
                "id": f"chatcmpl-aetherforge-{chunk.request_id or int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chunk.model or request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.content},
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            if chunk.usage is not None:
                payload["usage"] = dict(chunk.usage)
            emitted = emitted or bool(chunk.content)
            encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            yield f"data: {encoded}\n\n".encode()
    except OmlxcError:
        payload = {
            "error": {
                "message": "local stream failed",
                "type": "stream_error",
                "code": "stream_error",
                "emitted_content": emitted,
            }
        }
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
        return
    yield b"data: [DONE]\n\n"


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
    except Exception:
        return web.json_response({"error": {"message": "embedding failed"}}, status=502)


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
