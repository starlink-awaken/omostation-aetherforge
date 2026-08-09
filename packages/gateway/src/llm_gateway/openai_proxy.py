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
import logging
import time

from aiohttp import web

from .gateway import GatewayRequest, get_gateway

_log = logging.getLogger(__name__)


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

    gw = get_gateway()

    # Ensure registry is populated
    if not gw._registry.list_models():
        await gw._registry.refresh()

    req = GatewayRequest(
        messages=messages,
        model=model,
        timeout=float(body.get("timeout", 120)),
        temperature=temperature,
        max_tokens=max_tokens,
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
                "finish_reason": "stop",
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


async def handle_list_models(request: web.Request) -> web.Response:
    """GET /v1/models — list all discovered models."""
    gw = get_gateway()
    if not gw._registry.list_models():
        await gw._registry.refresh()

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
        embeddings = await gw.embed(texts)
        return web.json_response(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embeddings)],
                "model": body.get("model", "embedding"),
            }
        )
    except Exception as e:
        return web.json_response({"error": {"message": str(e)[:100]}}, status=502)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({"status": "ok", "service": "aetherforge-openai-proxy"})


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_list_models)
    app.router.add_post("/v1/embeddings", handle_embeddings)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/", handle_health)
    return app


def serve(port: int = 9290) -> None:
    """Start the OpenAI-compatible proxy server."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    _log.info("Starting AetherForge OpenAI proxy on :%d", port)
    _log.info("  POST /v1/chat/completions  — LLM inference")
    _log.info("  GET  /v1/models            — list models")
    _log.info("  POST /v1/embeddings        — embeddings")
    _log.info("  Set base_url=http://127.0.0.1:%d/v1 in any OpenAI client", port)
    web.run_app(create_app(), host="127.0.0.1", port=port, print=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherForge OpenAI-compatible proxy")
    parser.add_argument("--port", type=int, default=9290)
    args = parser.parse_args(argv)
    serve(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
