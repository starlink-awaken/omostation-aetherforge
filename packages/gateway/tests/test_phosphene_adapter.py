"""phosphene 适配: 图像 OpenAI↔phosphene 形状互转、GPU 忙如实 503、视频提交/轮询/取内容。"""

from __future__ import annotations

import asyncio
import base64

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from llm_gateway import phosphene_adapter as ph


def _run(coro):
    return asyncio.run(coro)


def test_size_to_aspect_and_frame_math():
    assert ph.size_to_aspect("1024x1024") == "1:1"
    assert ph.size_to_aspect("1792x1024") == "16:9"
    assert ph.size_to_aspect("1024x1792") == "9:16"
    assert ph.size_to_aspect("auto") == "1:1"
    assert ph.ltx_frames(5) == 121  # LTX: frames % 8 == 1
    assert ph.ltx_frames(7) % 8 == 1
    assert ph.h3_tier("standard", 5) == "standard_5s"
    assert ph.h3_tier("bogus", 4) == "draft_5s"  # 未知质量落 draft; 时长向上取档, 不偷短
    assert ph.h3_tier("high", 12) == "high_15s"
    assert ph.h3_tier("draft", 60) == "draft_15s"  # 超长封顶


def _fake_phosphene(state: dict, tmp_path):
    png = tmp_path / "cand.png"
    png.write_bytes(b"\x89PNG-fake")
    mp4 = tmp_path / "out.mp4"
    mp4.write_bytes(b"fake-mp4")

    async def image(request):
        state["image_req"] = await request.json()
        if state.get("busy"):
            return web.json_response({"error": "the GPU is busy with a render"})
        return web.json_response({"ok": True, "model": "flux2-klein", "candidates": [{"png_path": str(png)}]})

    async def queue_add(request):
        state["form"] = dict(await request.post())
        return web.json_response({"ok": True, "id": "j-1"})

    async def status(request):
        job = {
            "id": "j-1",
            "status": state.get("job_status", "running"),
            "output_path": str(mp4),
            "progress": {"pct": 40},
        }
        if job["status"] == "running":
            return web.json_response({"current": job, "queue": [], "history": []})
        return web.json_response({"current": {}, "queue": [], "history": [job]})

    app = web.Application()
    app.router.add_post("/image/generate", image)
    app.router.add_post("/queue/add", queue_add)
    app.router.add_get("/status", status)
    return app


async def _facade(monkeypatch, state, tmp_path, fn):
    backend = TestServer(_fake_phosphene(state, tmp_path))
    await backend.start_server()
    monkeypatch.setenv("AETHERFORGE_PHOSPHENE_BASE_URL", str(backend.make_url("")).rstrip("/"))
    app = web.Application()
    ph.register_phosphene_routes(app, resolve_alias=lambda m: {"video-h3": "phosphene-h3"}.get(m, m))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        return await fn(client)
    finally:
        await client.close()
        await backend.close()


def test_image_maps_openai_shape_and_returns_b64(monkeypatch, tmp_path):
    state: dict = {}

    async def go(client):
        r = await client.post("/v1/images/generations", json={"prompt": "a lantern", "size": "1792x1024", "n": 1})
        return r.status, await r.json()

    status, body = _run(_facade(monkeypatch, state, tmp_path, go))
    assert status == 200
    assert state["image_req"] == {"prompt": "a lantern", "n": 1, "aspect": "16:9"}
    assert base64.b64decode(body["data"][0]["b64_json"]) == b"\x89PNG-fake"


def test_image_gpu_busy_is_503(monkeypatch, tmp_path):
    state = {"busy": True}

    async def go(client):
        r = await client.post("/v1/images/generations", json={"prompt": "x"})
        return r.status, await r.json()

    status, body = _run(_facade(monkeypatch, state, tmp_path, go))
    assert status == 503 and body["error"]["code"] == "backend_busy"


def test_video_ltx_and_h3_forms(monkeypatch, tmp_path):
    state: dict = {}

    async def go(client):
        r1 = await client.post("/v1/videos", json={"model": "video", "prompt": "a crow", "seconds": 5})
        ltx_form = dict(state["form"])
        r2 = await client.post(
            "/v1/videos", json={"model": "video-h3", "prompt": "two people talk", "seconds": 5, "quality": "standard"}
        )
        return (await r1.json()), ltx_form, (await r2.json()), dict(state["form"])

    ltx_resp, ltx, h3_resp, h3 = _run(_facade(monkeypatch, state, tmp_path, go))
    assert ltx_resp["id"] == "j-1" and ltx_resp["status"] == "queued"
    assert ltx["mode"] == "t2v" and ltx["frames"] == "121" and "engine" not in ltx
    assert h3["engine"] == "h3" and h3["h3_tier"] == "standard_5s"
    assert h3_resp["model"] == "phosphene-h3"


def test_video_poll_then_content(monkeypatch, tmp_path):
    state: dict = {}

    async def go(client):
        running = await (await client.get("/v1/videos/j-1")).json()
        early = await client.get("/v1/videos/j-1/content")
        state["job_status"] = "done"
        done = await (await client.get("/v1/videos/j-1")).json()
        content = await client.get("/v1/videos/j-1/content")
        missing = await client.get("/v1/videos/nope")
        return running, early.status, done, content.status, await content.read(), missing.status

    running, early, done, cstatus, data, missing = _run(_facade(monkeypatch, state, tmp_path, go))
    assert running["status"] == "in_progress" and running["progress"] == 40
    assert early == 409  # 未完成不能取内容
    assert done["status"] == "completed"
    assert cstatus == 200 and data == b"fake-mp4"
    assert missing == 404
