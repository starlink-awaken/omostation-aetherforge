"""门面绑定范围与鉴权。

背景: 门面要接 LiteLLM 的班, 就得像它一样能被 tailnet 上的机器访问
(kairon / cockpit / gac 里散着一堆 http://100.96.126.35:4000)。
但"绑定范围变大"不能和"鉴权消失"同时发生, 所以有了这两条闸。
"""

from __future__ import annotations

import pytest
from aiohttp import web

from llm_gateway import openai_proxy as P


class TestBindResolution:
    def test_local_is_loopback_only(self):
        assert P.resolve_bind_hosts("local") == ["127.0.0.1"]

    def test_tailnet_includes_loopback_and_tailnet_ip(self, monkeypatch):
        monkeypatch.setattr(P, "_tailnet_ip", lambda: "100.96.126.35")
        assert P.resolve_bind_hosts("tailnet") == ["127.0.0.1", "100.96.126.35"]

    def test_tailnet_falls_back_to_local_when_unavailable(self, monkeypatch):
        """拿不到 tailnet 地址时收缩到 loopback, 而不是退成 0.0.0.0。"""
        monkeypatch.setattr(P, "_tailnet_ip", lambda: None)
        assert P.resolve_bind_hosts("tailnet") == ["127.0.0.1"]

    def test_never_defaults_to_all_interfaces(self, monkeypatch):
        """0.0.0.0 会把模型开给同网段任意机器 —— 只能显式指定, 不能是任何默认值。

        (omlxc gw start 给 LiteLLM 下的正是 --host 0.0.0.0, 注释却写"绑 tailnet"。)
        """
        monkeypatch.setattr(P, "_tailnet_ip", lambda: "100.96.126.35")
        for spec in ("local", "tailnet"):
            assert "0.0.0.0" not in P.resolve_bind_hosts(spec)


class TestAuth:
    """直接驱动中间件, 不起真 HTTP 服务(仓里没有 pytest-aiohttp)。"""

    @staticmethod
    def _run(api_key, path="/v1/chat/completions", header=None):
        import asyncio
        from types import SimpleNamespace

        app = {"api_key": api_key}
        req = SimpleNamespace(app=app, path=path, headers=header or {})

        async def handler(_):
            return web.json_response({"ok": True})

        return asyncio.run(P.auth_middleware(req, handler))

    def test_no_key_means_open(self):
        assert self._run(None).status == 200

    def test_health_stays_open_even_with_key(self):
        """探活不该要 key, 否则 wsvc / launchd 的健康检查全变红。"""
        assert self._run("secret", path="/health").status == 200

    def test_missing_key_rejected(self):
        assert self._run("secret").status == 401

    def test_wrong_key_rejected(self):
        assert self._run("secret", header={"Authorization": "Bearer nope"}).status == 401

    def test_right_key_passes(self):
        assert self._run("secret", header={"Authorization": "Bearer secret"}).status == 200

    def test_bare_key_without_bearer_prefix_also_works(self):
        """有些客户端只填 key 不带 Bearer, 别让它们卡在这。"""
        assert self._run("secret", header={"Authorization": "secret"}).status == 200


def test_serve_refuses_non_loopback_without_key(monkeypatch):
    """绑到 loopback 之外却没配 key → 拒绝启动, 而不是静默裸奔。"""
    monkeypatch.setattr(P, "_tailnet_ip", lambda: "100.96.126.35")
    monkeypatch.delenv("AETHERFORGE_API_KEY", raising=False)
    monkeypatch.setattr(web, "run_app", lambda *a, **k: pytest.fail("不该走到 run_app"))
    with pytest.raises(SystemExit) as ei:
        P.serve(port=59290, bind="tailnet")
    assert "AETHERFORGE_API_KEY" in str(ei.value)


def test_serve_allows_loopback_without_key(monkeypatch):
    """只绑 loopback 时不强制 key —— 本机进程本来就能直接打各模型端口。"""
    monkeypatch.delenv("AETHERFORGE_API_KEY", raising=False)
    called = {}
    monkeypatch.setattr(web, "run_app", lambda *a, **k: called.update(k))
    P.serve(port=59290, bind="local")
    assert called.get("host") == ["127.0.0.1"]
