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


def test_backend_down_is_502_not_silent(monkeypatch):
    async def go():
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
