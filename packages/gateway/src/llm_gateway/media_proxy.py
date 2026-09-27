"""多媒体透传 — 图像生成 / 语音合成 / 语音识别 转给本地多媒体后端。

后端按能力分:
  图像  /v1/images/generations              → Unsloth Studio(:8888, 扩散模型)
  语音  /v1/audio/{speech,transcriptions,voices} → oMLX App(:8000, mlx-audio: Qwen3-TTS/Kokoro/Whisper)
  决策  /v1/decisions                        → 本地决策服务(:8890, Laya / Jev-style, 返回校准概率)
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

import asyncio
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
# 决策服务(Laya 多语言 ~10ms / Jev v2 校准概率): 结构化选择/打分/真假判断, 不生成文本
DECISION_BACKEND = _Backend("AETHERFORGE_DECISION_BASE_URL", "http://127.0.0.1:8890", "AETHERFORGE_DECISION_API_KEY")
# 重排(cross-encoder): oMLX 原生 /v1/rerank。LM Studio / Ollama 都没有重排接口, 别名表里的
# rerank 档此前指向它们的模型 ID, 门面又无此路由 —— 声明了能力却从未调通。
RERANK_BACKEND = _Backend("AETHERFORGE_RERANK_BASE_URL", "http://127.0.0.1:8000", "AETHERFORGE_RERANK_API_KEY")

MEDIA_ROUTES: dict[str, tuple[str, _Backend]] = {
    "/v1/images/generations": ("POST", IMAGE_BACKEND),
    "/v1/audio/speech": ("POST", AUDIO_BACKEND),
    "/v1/audio/transcriptions": ("POST", AUDIO_BACKEND),
    "/v1/audio/voices": ("GET", AUDIO_BACKEND),
    "/v1/decisions": ("POST", DECISION_BACKEND),
    "/v1/rerank": ("POST", RERANK_BACKEND),
}
# 图像/视频生成可以跑几分钟; 语音很快。取宽松上限, 超时交给调用方自己的 timeout。
_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=900, sock_connect=5)

RESOLVE_ALIAS = web.AppKey("media_resolve_alias", Callable[[str], str])

# Unsloth 生成前必须显式加载图像模型, 未加载时回 503 "No image model loaded"。
# 门面代为加载一次再重试, 调用方不必关心后端生命周期(加载 ~30s, 之后常驻)。
_IMAGE_NOT_LOADED = b"No image model loaded"
_IMAGE_LOAD_TIMEOUT = 600.0
_IMAGE_LOAD_POLL = 3.0


def default_image_model() -> str:
    return os.environ.get("AETHERFORGE_MEDIA_IMAGE_MODEL", "unsloth/Qwen-Image-2.1")


async def _ensure_image_model(session: aiohttp.ClientSession, base: str, headers: dict[str, str]) -> bool:
    """触发加载并轮询到就绪; 失败/超时返回 False(交回原 503 让调用方如实看到)。"""
    auth = {k: v for k, v in headers.items() if k == "Authorization"}
    async with session.post(
        base + "/api/inference/images/load", json={"model_path": default_image_model()}, headers=auth
    ) as r:
        if r.status >= 400:
            _log.warning("image model load rejected: %s %s", r.status, (await r.text())[:200])
            return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _IMAGE_LOAD_TIMEOUT
    while loop.time() < deadline:
        async with session.get(base + "/api/inference/images/status", headers=auth) as r:
            if r.status == 200 and (await r.json()).get("loaded"):
                return True
        async with session.get(base + "/api/inference/images/load-progress", headers=auth) as r:
            if r.status == 200 and (await r.json()).get("error"):
                _log.warning("image model load failed: %s", (await r.json()).get("error"))
                return False
        await asyncio.sleep(_IMAGE_LOAD_POLL)
    return False


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
            status, data, ctype, charset = await _forward(session, request, base, body, headers)
            if (
                backend is IMAGE_BACKEND
                and status == 503
                and _IMAGE_NOT_LOADED in data
                and await _ensure_image_model(session, base, headers)
            ):
                status, data, ctype, charset = await _forward(session, request, base, body, headers)
            return web.Response(body=data, status=status, content_type=ctype, charset=charset)
    except (aiohttp.ClientError, TimeoutError) as exc:
        _log.warning("media backend %s%s unavailable: %s", base, request.path, exc)
        return _error("media backend unavailable", 502, "media_backend_unavailable")


async def _forward(session, request: web.Request, base: str, body, headers: dict[str, str]):
    # 查询串里的 model(如 GET /v1/audio/voices?model=tts-zh)同样过别名表, 否则后端按原样找不到
    params = dict(request.query)
    if "model" in params:
        params["model"] = str(_resolve(request, params["model"]))
    async with session.request(
        request.method, base + request.path, params=params, data=body, headers=headers
    ) as upstream:
        return upstream.status, await upstream.read(), upstream.content_type, upstream.charset


def image_backend_kind() -> str:
    """图像后端: phosphene(默认, FLUX.2-klein 22s/张且遵循提示词) | unsloth(透传, 旧路径)。"""
    return os.environ.get("AETHERFORGE_MEDIA_IMAGE_KIND", "phosphene").strip().lower()


def register_media_routes(app: web.Application, resolve_alias: Callable[[str], str] | None = None) -> None:
    from .phosphene_adapter import register_phosphene_routes

    if resolve_alias is not None:
        app[RESOLVE_ALIAS] = resolve_alias
    phosphene_images = image_backend_kind() == "phosphene"
    for path, (method, _backend) in MEDIA_ROUTES.items():
        if path == "/v1/images/generations" and phosphene_images:
            continue
        app.router.add_route(method, path, handle_media)
    # 视频恒走 phosphene(LTX-2.5 / MiniMax H3); 图像按开关二选一
    register_phosphene_routes(app, resolve_alias, images=phosphene_images)
