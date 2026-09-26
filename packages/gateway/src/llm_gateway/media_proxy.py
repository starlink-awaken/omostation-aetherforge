"""多媒体透传 — 图像生成 / 语音合成 / 语音识别 转给本地多媒体后端。

后端按能力分:
  图像  /v1/images/generations              → Unsloth Studio(:8888, 扩散模型)
  语音  /v1/audio/{speech,transcriptions,voices} → oMLX App(:8000, mlx-audio: Qwen3-TTS/Kokoro/Whisper)
  (2026-09-25 实测: Unsloth 在 Apple Silicon 上只认 higgs/moss/minimax 几个原生音频模型且
   需 remote code; oMLX 直接读模型盘上的 MLX 权重, Kokoro 0.15s/4.7s 音频, 中文走 Qwen3-TTS。)

为什么是透传而不是走 ModelGateway:
  ModelGateway 的路由/降级/记账都围绕"文本进、文本出"; 图像与音频是二进制、
  多为 multipart 上传, 且每类只有一个后端能做 —— 没有可降级的同能力候选
  (禁跨能力兜底)。所以这里只做三件事: 鉴权边界换 key、别名解析、原样转发。

鉴权: 调用方用门面自己的 key(由 auth_middleware 校验); 转发时换成对应后端的 key
(图像: AETHERFORGE_MEDIA_API_KEY, 由 gw-guard 从 Keychain 注入), 后端 key 不暴露给调用方。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp
from aiohttp import web

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Backend:
    base_env: str
    base_default: str
    key_env: str

    def resolve(self) -> tuple[str, str | None]:
        base = os.environ.get(self.base_env, self.base_default).rstrip("/")
        return base, os.environ.get(self.key_env) or None


IMAGE_BACKEND = _Backend("AETHERFORGE_MEDIA_IMAGE_BASE_URL", "http://127.0.0.1:8888", "AETHERFORGE_MEDIA_API_KEY")
AUDIO_BACKEND = _Backend("AETHERFORGE_MEDIA_AUDIO_BASE_URL", "http://127.0.0.1:8000", "AETHERFORGE_MEDIA_AUDIO_API_KEY")

MEDIA_ROUTES: dict[str, tuple[str, _Backend]] = {
    "/v1/images/generations": ("POST", IMAGE_BACKEND),
    "/v1/audio/speech": ("POST", AUDIO_BACKEND),
    "/v1/audio/transcriptions": ("POST", AUDIO_BACKEND),
    "/v1/audio/voices": ("GET", AUDIO_BACKEND),
}
# 图像/视频生成可以跑几分钟; 语音很快。取宽松上限, 超时交给调用方自己的 timeout。
_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=900, sock_connect=5)

RESOLVE_ALIAS = web.AppKey("media_resolve_alias", Callable[[str], str])


def _error(message: str, status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "media_backend_error", "code": code}}, status=status
    )


def _resolve(request: web.Request, model: object) -> object:
    resolve = request.app.get(RESOLVE_ALIAS)
    return resolve(model) if isinstance(model, str) and model and resolve is not None else model


async def _outgoing_body(request: web.Request) -> tuple[bytes | aiohttp.FormData | None, dict[str, str]] | web.Response:
    """按内容类型重建上游请求体, 并把 model 字段过一遍别名表。"""
    content_type = request.headers.get("Content-Type", "")
    if request.method == "GET":
        return None, {}
    if content_type.startswith("application/json"):
        try:
            payload = json.loads(await request.read() or b"{}")
        except ValueError:
            return _error("Invalid JSON", 400, "invalid_json")
        if "model" in payload:
            payload["model"] = _resolve(request, payload["model"])
        return json.dumps(payload).encode(), {"Content-Type": "application/json"}
    if content_type.startswith("multipart/form-data"):
        # 语音识别上传: 逐字段重组, 文件字节原样, 仅 model 字段换别名
        form = aiohttp.FormData()
        reader = await request.multipart()
        while (part := await reader.next()) is not None:
            data = await part.read()
            if not part.filename:
                # 普通表单字段按文本转发(字节会被当成文件 part); 只有 model 走别名表
                text = data.decode()
                form.add_field(part.name or "", str(_resolve(request, text)) if part.name == "model" else text)
            else:
                form.add_field(
                    part.name or "file",
                    data,
                    filename=part.filename,
                    content_type=part.headers.get("Content-Type"),
                )
        return form, {}
    return await request.read(), ({"Content-Type": content_type} if content_type else {})


async def handle_media(request: web.Request) -> web.Response:
    _method, backend = MEDIA_ROUTES[request.path]
    base, key = backend.resolve()
    built = await _outgoing_body(request)
    if isinstance(built, web.Response):
        return built
    body, headers = built
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        # trust_env=False: 本地后端必须直连, 不能被系统代理(Clash)接走
        async with aiohttp.ClientSession(timeout=_UPSTREAM_TIMEOUT, trust_env=False) as session:
            async with session.request(
                request.method, base + request.path, params=request.query, data=body, headers=headers
            ) as upstream:
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
    for path, (method, _backend) in MEDIA_ROUTES.items():
        app.router.add_route(method, path, handle_media)
