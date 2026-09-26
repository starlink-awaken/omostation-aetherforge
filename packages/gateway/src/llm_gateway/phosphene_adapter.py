"""phosphene 适配 — 图像 / 视频生成走本地 phosphene 面板(Apple Silicon 原生 MLX)。

phosphene(Pinokio 装, 127.0.0.1:8198, 仅回环、无鉴权)提供:
  图像  POST /image/generate   mflux: FLUX.2-klein-4B 4bit(1024² ~22s), Qwen-Image-Edit, Ideogram4
  视频  POST /queue/add        LTX-2.5(t2v/i2v, 音画联合) 或 MiniMax H3(engine=h3, 对白+音效+配乐)
        GET  /status           current / queue / history, 终态 done|failed|cancelled|stopped|error

门面对外保持 OpenAI 形状:
  POST /v1/images/generations           → b64_json(面板与门面同机, 直接读 PNG)
  POST /v1/videos                       → {"id","status":"queued"}   (异步, 分钟级)
  GET  /v1/videos/{id}                  → status / progress / 终态
  GET  /v1/videos/{id}/content          → video/mp4

2026-09-26 实测: 同一提示词 phosphene FLUX.2-klein 22s 出图且遵循风格; Unsloth Qwen-Image-2.1
5 分钟且不遵循提示词 → 图像默认走 phosphene(AETHERFORGE_MEDIA_IMAGE_KIND=unsloth 可切回)。
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web

_log = logging.getLogger(__name__)
_TIMEOUT = aiohttp.ClientTimeout(total=1500, sock_connect=5)  # 首次出图可能含模型下载

_ASPECTS = {"1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}
_TERMINAL = {"done", "failed", "cancelled", "stopped", "error"}
_H3_QUALITIES = ("draft", "standard", "high", "native")
_H3_LENGTHS = (3, 5, 10, 15)


def phosphene_base() -> str:
    return os.environ.get("AETHERFORGE_PHOSPHENE_BASE_URL", "http://127.0.0.1:8198").rstrip("/")


def _err(message: str, status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "media_backend_error", "code": code}}, status=status
    )


def size_to_aspect(size: str | None) -> str:
    """OpenAI size("1024x1024"/"1792x1024"/"auto") → phosphene 最接近的画幅。"""
    try:
        w, h = (int(x) for x in str(size).lower().split("x"))
        ratio = w / h
    except (ValueError, ZeroDivisionError):
        return "1:1"
    return min(_ASPECTS, key=lambda k: abs(_ASPECTS[k] - ratio))


def ltx_frames(seconds: float) -> int:
    """LTX 要求 frames % 8 == 1, 24fps: 5s → 121。"""
    return max(1, round(seconds * 24 / 8)) * 8 + 1


def h3_tier(quality: str, seconds: float) -> str:
    q = quality if quality in _H3_QUALITIES else "draft"
    # 取不短于请求时长的最近一档(要 4s 给 5s, 别偷短), 超出最长档则封顶
    length = next((s for s in _H3_LENGTHS if s >= seconds), _H3_LENGTHS[-1])
    return f"{q}_{length}s"


async def _phosphene(session: aiohttp.ClientSession, method: str, path: str, **kw):
    async with session.request(method, phosphene_base() + path, **kw) as r:
        try:
            return r.status, await r.json(content_type=None)
        except ValueError:
            return r.status, {"error": (await r.text())[:300]}


# ── 图像 ────────────────────────────────────────────────────────────────


async def handle_image(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except ValueError:
        return _err("Invalid JSON", 400, "invalid_json")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt is required", 400, "invalid_request")
    n = max(1, min(int(body.get("n") or 1), 4))
    payload = {"prompt": prompt, "n": n, "aspect": size_to_aspect(body.get("size"))}
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT, trust_env=False) as session:
            status, data = await _phosphene(session, "POST", "/image/generate", json=payload)
    except (aiohttp.ClientError, TimeoutError) as exc:
        _log.warning("phosphene image unavailable: %s", exc)
        return _err("image backend unavailable", 502, "media_backend_unavailable")
    if not data.get("ok"):
        # 面板在渲视频/训练时拒绝出图(GPU 显存互斥) → 503 让调用方稍后重试
        busy = "busy" in str(data.get("error", "")).lower()
        return _err(
            str(data.get("error") or "image generation failed"),
            503 if busy else 502,
            "backend_busy" if busy else "generation_failed",
        )
    images = []
    for cand in data.get("candidates", []):
        path = Path(cand.get("png_path") or "")
        if not path.is_file():
            return _err(f"image file missing: {path}", 502, "generation_failed")
        images.append({"b64_json": base64.b64encode(path.read_bytes()).decode(), "revised_prompt": None})
    return web.json_response({"created": int(time.time()), "data": images, "model": data.get("model")})


# ── 视频 ────────────────────────────────────────────────────────────────


def _video_form(body: dict, model: str) -> dict[str, str]:
    seconds = float(body.get("seconds") or 5)
    quality = str(body.get("quality") or "")
    if "h3" in model:
        return {
            "mode": "t2v",
            "engine": "h3",
            "prompt": body["prompt"],
            "h3_tier": h3_tier(quality or "draft", seconds),
            "h3_upscale": "fit_720p",
            "seed": str(body.get("seed", -1)),
            "label": "gateway",
        }
    return {
        "mode": "t2v",
        "prompt": body["prompt"],
        "width": "1024",
        "height": "576",
        "frames": str(ltx_frames(seconds)),
        "frame_rate": "24",
        "quality": quality if quality in ("quick", "balanced", "standard", "high") else "balanced",
        "temporal_mode": "native",
        "upscale": "fit_720p",
        "upscale_method": "lanczos",
        "enhance": "false",
        "seed": str(body.get("seed", -1)),
        "label": "gateway",
    }


async def handle_video_create(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except ValueError:
        return _err("Invalid JSON", 400, "invalid_json")
    if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
        return _err("prompt is required", 400, "invalid_request")
    model = str(body.get("model") or "video")
    resolve = request.app.get(_RESOLVE)
    if resolve is not None:
        model = resolve(model)
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT, trust_env=False) as session:
            status, data = await _phosphene(session, "POST", "/queue/add", data=_video_form(body, model))
    except (aiohttp.ClientError, TimeoutError) as exc:
        _log.warning("phosphene video unavailable: %s", exc)
        return _err("video backend unavailable", 502, "media_backend_unavailable")
    if not data.get("ok") or not data.get("id"):
        return _err(str(data.get("error") or data), 400 if status == 400 else 502, "generation_failed")
    return web.json_response(
        {"id": data["id"], "object": "video", "model": model, "status": "queued", "created_at": int(time.time())}
    )


async def _find_job(job_id: str) -> tuple[dict | None, str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), trust_env=False) as session:
        _status, data = await _phosphene(session, "GET", "/status")
    cur = data.get("current") or {}
    if cur.get("id") == job_id:
        return cur, "in_progress"
    for job in data.get("queue") or []:
        if job.get("id") == job_id:
            return job, "queued"
    for job in data.get("history") or []:
        if job.get("id") == job_id:
            return job, "completed" if job.get("status") == "done" else "failed"
    return None, "not_found"


async def handle_video_get(request: web.Request) -> web.Response:
    try:
        job, state = await _find_job(request.match_info["video_id"])
    except (aiohttp.ClientError, TimeoutError):
        return _err("video backend unavailable", 502, "media_backend_unavailable")
    if job is None:
        return _err("video not found", 404, "not_found")
    progress = job.get("progress") or {}
    out = {
        "id": job["id"],
        "object": "video",
        "status": state,
        "progress": progress.get("pct") or progress.get("percent"),
        "phase": progress.get("phase_label") or progress.get("phase"),
        "elapsed_sec": job.get("elapsed_sec"),
    }
    if state == "failed":
        out["error"] = {"message": str(job.get("error") or job.get("status"))}
    return web.json_response(out)


async def handle_video_content(request: web.Request) -> web.StreamResponse:
    try:
        job, state = await _find_job(request.match_info["video_id"])
    except (aiohttp.ClientError, TimeoutError):
        return _err("video backend unavailable", 502, "media_backend_unavailable")
    if job is None:
        return _err("video not found", 404, "not_found")
    if state != "completed":
        return _err(f"video is {state}", 409, "not_ready")
    path = Path(job.get("output_path") or job.get("upscaled_path") or "")
    if not path.is_file():
        return _err("video file missing", 410, "gone")
    return web.FileResponse(path, headers={"Content-Type": "video/mp4"})


_RESOLVE = web.AppKey("phosphene_resolve_alias", object)


def register_phosphene_routes(app: web.Application, resolve_alias=None, *, images: bool = True) -> None:
    if resolve_alias is not None:
        app[_RESOLVE] = resolve_alias
    if images:
        app.router.add_post("/v1/images/generations", handle_image)
    app.router.add_post("/v1/videos", handle_video_create)
    app.router.add_get("/v1/videos/{video_id}", handle_video_get)
    app.router.add_get("/v1/videos/{video_id}/content", handle_video_content)
