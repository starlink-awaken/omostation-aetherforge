"""多媒体透传 — 图像生成 / 语音合成 / 语音识别 转给本地多媒体后端(Unsloth Studio)。

为什么是透传而不是走 ModelGateway:
  ModelGateway 的路由/降级/记账都围绕"文本进、文本出"; 图像与音频是二进制、
  多为 multipart 上传, 且只有一个后端能做 —— 没有可降级的同能力候选
  (禁跨能力兜底)。所以这里只做三件事: 鉴权边界换 key、别名解析、原样转发。

鉴权: 调用方用门面自己的 key(由 auth_middleware 校验); 转发时换成后端的 key
(AETHERFORGE_MEDIA_API_KEY, 由 gw-guard 从 Keychain 注入), 后端 key 不暴露给调用方。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

import aiohttp
from aiohttp import web

_log = logging.getLogger(__name__)

MEDIA_ROUTES = (
    "/v1/images/generations",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
)
DEFAULT_MEDIA_BASE_URL = "http://127.0.0.1:8888"
# 图像/视频生成可以跑几分钟; 语音很快。取宽松上限, 超时交给调用方自己的 timeout。
_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=900, sock_connect=5)

RESOLVE_ALIAS = web.AppKey("media_resolve_alias", Callable[[str], str])


def media_upstream() -> tuple[str, str | None]:
    base = os.environ.get("AETHERFORGE_MEDIA_BASE_URL", DEFAULT_MEDIA_BASE_URL).rstrip("/")
    return base, os.environ.get("AETHERFORGE_MEDIA_API_KEY") or None


def _error(message: str, status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "media_backend_error", "code": code}}, status=status
    )


async def handle_media(request: web.Request) -> web.Response:
    base, key = media_upstream()
    content_type = request.headers.get("Content-Type", "")
    body = await request.read()

    # JSON 请求(图像生成/语音合成)解析别名; multipart(语音识别上传)原样转发
    if content_type.startswith("application/json"):
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            return _error("Invalid JSON", 400, "invalid_json")
        model = payload.get("model")
        resolve = request.app.get(RESOLVE_ALIAS)
        if isinstance(model, str) and model and resolve is not None:
            payload["model"] = resolve(model)
        body = json.dumps(payload).encode()

    headers = {"Content-Type": content_type} if content_type else {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        # trust_env=False: 本地后端必须直连, 不能被系统代理(Clash)接走
        async with aiohttp.ClientSession(timeout=_UPSTREAM_TIMEOUT, trust_env=False) as session:
            async with session.post(base + request.path, data=body, headers=headers) as upstream:
                data = await upstream.read()
                return web.Response(
                    body=data,
                    status=upstream.status,
                    content_type=upstream.content_type,
                    charset=upstream.charset,
                )
    except (aiohttp.ClientError, TimeoutError) as exc:
        _log.warning("media backend %s%s unavailable: %s", base, request.path, exc)
        return _error("media backend unavailable", 502, "media_backend_unavailable")


def register_media_routes(app: web.Application, resolve_alias: Callable[[str], str] | None = None) -> None:
    if resolve_alias is not None:
        app[RESOLVE_ALIAS] = resolve_alias
    for path in MEDIA_ROUTES:
        app.router.add_post(path, handle_media)
