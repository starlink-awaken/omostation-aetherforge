"""多媒体透传: 别名解析、换 key、二进制/multipart 原样、后端不可达如实 502、门面鉴权覆盖。

用 aiohttp.test_utils 起真实的假后端与门面(仓里没有 pytest-aiohttp)。
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from llm_gateway import media_proxy
from llm_gateway import openai_proxy as proxy


def _run(coro):
    return asyncio.run(coro)


def _fake_backend(seen: list):
    async def handler(request: web.Request) -> web.Response:
        seen.append(
            {
                "path": request.path,
                "auth": request.headers.get("Authorization"),
                "ctype": request.headers.get("Content-Type", ""),
                "body": await request.read(),
            }
        )
        if request.path == "/v1/audio/speech":
            return web.Response(body=b"RIFF\x00\x01fake-wav", content_type="audio/wav")
        return web.json_response({"ok": True})

    app = web.Application()
    for path in media_proxy.MEDIA_ROUTES:
        app.router.add_post(path, handler)
    return app


async def _with_facade(monkeypatch, backend_app, fn, *, gateway_key=None, resolve=None):
    backend = TestServer(backend_app)
    await backend.start_server()
    monkeypatch.setenv("AETHERFORGE_MEDIA_BASE_URL", str(backend.make_url("")).rstrip("/"))
    monkeypatch.setenv("AETHERFORGE_MEDIA_API_KEY", "sk-unsloth-backend")
    facade = web.Application(middlewares=[proxy.auth_middleware])
    if gateway_key:
        facade[proxy.API_KEY] = gateway_key
    media_proxy.register_media_routes(facade, resolve_alias=resolve)
    client = TestClient(TestServer(facade))
    await client.start_server()
    try:
        return await fn(client)
    finally:
        await client.close()
        await backend.close()


def test_json_request_resolves_alias_and_swaps_key(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "a cat"},
            headers={"Authorization": "Bearer gw-key"},
        )
        return r.status

    status = _run(
        _with_facade(
            monkeypatch,
            _fake_backend(seen),
            go,
            gateway_key="gw-key",
            resolve=lambda m: {"image": "qwen-image"}.get(m, m),
        )
    )
    assert status == 200
    assert seen[0]["path"] == "/v1/images/generations"
    # 门面 key 不外泄, 换成后端 key
    assert seen[0]["auth"] == "Bearer sk-unsloth-backend"
    assert json.loads(seen[0]["body"]) == {"model": "qwen-image", "prompt": "a cat"}


def test_binary_audio_response_passes_through(monkeypatch):
    seen: list = []

    async def go(client):
        r = await client.post("/v1/audio/speech", json={"model": "tts", "input": "hi"})
        return r.status, r.headers["Content-Type"], await r.read()

    status, ctype, body = _run(_with_facade(monkeypatch, _fake_backend(seen), go))
    assert status == 200
    assert ctype.startswith("audio/wav")
    assert body == b"RIFF\x00\x01fake-wav"


def test_multipart_upload_forwarded_untouched(monkeypatch):
    seen: list = []
    audio = b"\x00\x01binary-audio\xff"

    async def go(client):
        form = FormData()
        form.add_field("model", "asr")
        form.add_field("file", audio, filename="a.wav", content_type="audio/wav")
        r = await client.post("/v1/audio/transcriptions", data=form)
        return r.status

    assert _run(_with_facade(monkeypatch, _fake_backend(seen), go)) == 200
    assert seen[0]["ctype"].startswith("multipart/form-data")
    assert audio in seen[0]["body"]


def test_backend_down_is_502_not_silent(monkeypatch):
    async def go():
        monkeypatch.setenv("AETHERFORGE_MEDIA_BASE_URL", "http://127.0.0.1:9")  # 丢弃端口, 必连不上
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

    assert _run(_with_facade(monkeypatch, _fake_backend(seen), go, gateway_key="gw-key")) == 401
    assert seen == []
