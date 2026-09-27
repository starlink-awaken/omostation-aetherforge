"""多媒体透传: 按能力分后端、别名解析(JSON 与 multipart)、换 key、二进制原样、
后端不可达如实 502、门面鉴权覆盖。

用 aiohttp.test_utils 起真实的假后端与门面(仓里没有 pytest-aiohttp)。
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from llm_gateway import media_proxy
from llm_gateway import openai_proxy as proxy

ALIASES = {"image": "qwen-image", "tts": "tts-qwen3", "asr": "asr-whisper"}


def _run(coro):
    return asyncio.run(coro)


def _fake_backend(name: str, seen: list):
    async def handler(request: web.Request) -> web.Response:
        rec = {
            "backend": name,
            "path": request.path,
            "query": dict(request.query),
            "auth": request.headers.get("Authorization"),
            "ctype": request.headers.get("Content-Type", ""),
        }
        if request.headers.get("Content-Type", "").startswith("multipart/"):
            post = await request.post()
            rec["form"] = {k: (v.file.read() if hasattr(v, "file") else v) for k, v in post.items()}
        else:
            rec["body"] = await request.read()
        seen.append(rec)
        if request.path == "/v1/audio/speech":
            return web.Response(body=b"RIFF\x00\x01fake-wav", content_type="audio/wav")
        return web.json_response({"ok": True})

    app = web.Application()
    for path, (method, _backend) in media_proxy.MEDIA_ROUTES.items():
        app.router.add_route(method, path, handler)
    return app


async def _with_facade(monkeypatch, seen, fn, *, gateway_key=None):
    image = TestServer(_fake_backend("image", seen))
    audio = TestServer(_fake_backend("audio", seen))
    await image.start_server()
    await audio.start_server()
    monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_KIND", "unsloth")
    monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_BASE_URL", str(image.make_url("")).rstrip("/"))
    monkeypatch.setenv("AETHERFORGE_MEDIA_AUDIO_BASE_URL", str(audio.make_url("")).rstrip("/"))
    monkeypatch.setenv("AETHERFORGE_MEDIA_API_KEY", "sk-unsloth-backend")
    monkeypatch.delenv("AETHERFORGE_MEDIA_AUDIO_API_KEY", raising=False)
    facade = web.Application(middlewares=[proxy.auth_middleware])
    if gateway_key:
        facade[proxy.API_KEY] = gateway_key
    media_proxy.register_media_routes(facade, resolve_alias=lambda m: ALIASES.get(m, m))
    client = TestClient(TestServer(facade))
    await client.start_server()
    try:
        return await fn(client)
    finally:
        await client.close()
        await image.close()
        await audio.close()


def test_image_goes_to_image_backend_with_its_key_and_alias(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "a cat"},
            headers={"Authorization": "Bearer gw-key"},
        )
        return r.status

    assert _run(_with_facade(monkeypatch, seen, go, gateway_key="gw-key")) == 200
    assert seen[0]["backend"] == "image"
    # 门面 key 不外泄, 换成图像后端自己的 key
    assert seen[0]["auth"] == "Bearer sk-unsloth-backend"
    assert json.loads(seen[0]["body"]) == {"model": "qwen-image", "prompt": "a cat"}


def test_speech_goes_to_audio_backend_without_image_key(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.post("/v1/audio/speech", json={"model": "tts", "input": "hi"})
        return r.status, r.headers["Content-Type"], await r.read()

    status, ctype, body = _run(_with_facade(monkeypatch, seen, go))
    assert status == 200
    assert ctype.startswith("audio/wav")
    assert body == b"RIFF\x00\x01fake-wav"
    assert seen[0]["backend"] == "audio"
    assert seen[0]["auth"] is None  # 图像后端的 key 不能漏给语音后端
    assert json.loads(seen[0]["body"])["model"] == "tts-qwen3"


def test_transcription_multipart_resolves_model_and_keeps_file_bytes(monkeypatch):
    seen: list = []
    audio = b"\x00\x01binary-audio\xff"

    async def go(client):
        form = FormData()
        form.add_field("model", "asr")
        form.add_field("language", "zh")
        form.add_field("file", audio, filename="a.wav", content_type="audio/wav")
        r = await client.post("/v1/audio/transcriptions", data=form)
        return r.status

    assert _run(_with_facade(monkeypatch, seen, go)) == 200
    assert seen[0]["backend"] == "audio"
    assert seen[0]["form"]["model"] == "asr-whisper"
    assert seen[0]["form"]["language"] == "zh"
    assert seen[0]["form"]["file"] == audio


def test_voices_get_passes_query(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.get("/v1/audio/voices", params={"model": "tts-qwen3"})
        return r.status

    assert _run(_with_facade(monkeypatch, seen, go)) == 200
    assert seen[0]["backend"] == "audio"
    assert seen[0]["query"] == {"model": "tts-qwen3"}


def test_voices_get_resolves_model_alias_in_query(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.get("/v1/audio/voices", params={"model": "tts"})
        return r.status

    assert _run(_with_facade(monkeypatch, seen, go)) == 200
    assert seen[0]["query"] == {"model": "tts-qwen3"}


def test_backend_down_is_502_not_silent(monkeypatch):
    async def go():
        monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_KIND", "unsloth")
        monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_BASE_URL", "http://127.0.0.1:9")  # 丢弃端口, 必连不上
        facade = web.Application()
        media_proxy.register_media_routes(facade)
        client = TestClient(TestServer(facade))
        await client.start_server()
        try:
            r = await client.post("/v1/images/generations", json={"prompt": "x"})
            return r.status, await r.json()
        finally:
            await client.close()

    status, body = _run(go())
    assert status == 502
    assert body["error"]["code"] == "media_backend_unavailable"


def test_media_routes_require_gateway_key(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.post("/v1/audio/speech", json={"input": "hi"})
        return r.status

    assert _run(_with_facade(monkeypatch, seen, go, gateway_key="gw-key")) == 401
    assert seen == []


def _lazy_image_backend(state: dict):
    """未加载时 503; /images/load 后按 state['load_ok'] 变为已加载或报错。"""

    async def generate(request: web.Request) -> web.Response:
        state["gen_calls"] += 1
        if not state["loaded"]:
            return web.json_response(
                {"error": {"message": "No image model loaded. Load an image model first."}}, status=503
            )
        return web.json_response({"data": [{"b64_json": "aW1n"}]})

    async def load(request: web.Request) -> web.Response:
        state["load_body"] = await request.json()
        state["load_auth"] = request.headers.get("Authorization")
        state["loaded"] = state["load_ok"]
        return web.json_response({"loaded": False})

    async def status(request: web.Request) -> web.Response:
        return web.json_response({"loaded": state["loaded"]})

    async def progress(request: web.Request) -> web.Response:
        return web.json_response({"error": None if state["load_ok"] else "boom"})

    app = web.Application()
    app.router.add_post("/v1/images/generations", generate)
    app.router.add_post("/api/inference/images/load", load)
    app.router.add_get("/api/inference/images/status", status)
    app.router.add_get("/api/inference/images/load-progress", progress)
    return app


def _run_lazy(monkeypatch, state):
    async def go():
        backend = TestServer(_lazy_image_backend(state))
        await backend.start_server()
        monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_KIND", "unsloth")
        monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_BASE_URL", str(backend.make_url("")).rstrip("/"))
        monkeypatch.setenv("AETHERFORGE_MEDIA_API_KEY", "sk-unsloth-backend")
        monkeypatch.setenv("AETHERFORGE_MEDIA_IMAGE_MODEL", "unsloth/Qwen-Image-2.1")
        monkeypatch.setattr(media_proxy, "_IMAGE_LOAD_POLL", 0.01)
        facade = web.Application()
        media_proxy.register_media_routes(facade)
        client = TestClient(TestServer(facade))
        await client.start_server()
        try:
            r = await client.post("/v1/images/generations", json={"prompt": "a cat"})
            return r.status, await r.json()
        finally:
            await client.close()
            await backend.close()

    return _run(go())


def test_image_autoloads_default_model_then_retries(monkeypatch):
    state = {"loaded": False, "load_ok": True, "gen_calls": 0}
    status, body = _run_lazy(monkeypatch, state)
    assert status == 200 and body["data"][0]["b64_json"] == "aW1n"
    assert state["load_body"] == {"model_path": "unsloth/Qwen-Image-2.1"}
    assert state["load_auth"] == "Bearer sk-unsloth-backend"
    assert state["gen_calls"] == 2  # 一次 503 + 加载后一次重试


def test_image_autoload_failure_surfaces_original_503(monkeypatch):
    state = {"loaded": False, "load_ok": False, "gen_calls": 0}
    status, body = _run_lazy(monkeypatch, state)
    assert status == 503
    assert "No image model loaded" in body["error"]["message"]
    assert state["gen_calls"] == 1  # 加载失败不再重试


def test_decisions_route_to_decision_backend_with_alias(monkeypatch):
    seen: list = []

    async def go():
        backend_app = web.Application()

        async def handler(request):
            seen.append(await request.json())
            return web.json_response({"model": "laya-multilingual", "answers": {}})

        backend_app.router.add_post("/v1/decisions", handler)
        backend = TestServer(backend_app)
        await backend.start_server()
        monkeypatch.setenv("AETHERFORGE_DECISION_BASE_URL", str(backend.make_url("")).rstrip("/"))
        facade = web.Application()
        media_proxy.register_media_routes(facade, resolve_alias=lambda m: {"decide": "laya-multilingual"}.get(m, m))
        client = TestClient(TestServer(facade))
        await client.start_server()
        try:
            r = await client.post("/v1/decisions", json={"model": "decide", "state": "x", "questions": {"q": {}}})
            return r.status
        finally:
            await client.close()
            await backend.close()

    assert _run(go()) == 200
    assert seen[0]["model"] == "laya-multilingual"


def test_rerank_routes_to_omlx_backend_with_alias(monkeypatch):
    seen: list = []

    async def go():
        backend_app = web.Application()

        async def handler(request):
            seen.append(await request.json())
            return web.json_response({"results": [{"index": 0, "relevance_score": 0.9}]})

        backend_app.router.add_post("/v1/rerank", handler)
        backend = TestServer(backend_app)
        await backend.start_server()
        monkeypatch.setenv("AETHERFORGE_RERANK_BASE_URL", str(backend.make_url("")).rstrip("/"))
        facade = web.Application()
        media_proxy.register_media_routes(
            facade, resolve_alias=lambda m: {"rerank": "baai-bge-reranker-v2-m3-mlx-fp16"}.get(m, m)
        )
        client = TestClient(TestServer(facade))
        await client.start_server()
        try:
            r = await client.post("/v1/rerank", json={"model": "rerank", "query": "q", "documents": ["a"]})
            return r.status
        finally:
            await client.close()
            await backend.close()

    assert _run(go()) == 200
    assert seen[0]["model"] == "baai-bge-reranker-v2-m3-mlx-fp16"
