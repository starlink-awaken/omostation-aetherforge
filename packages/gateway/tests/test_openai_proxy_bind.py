"""门面绑定范围与鉴权。

背景: 门面要接 LiteLLM 的班, 就得像它一样能被 tailnet 上的机器访问
(kairon / cockpit / gac 里散着一堆 http://100.96.126.35:4000)。
但"绑定范围变大"不能和"鉴权消失"同时发生, 所以有了这两条闸。
"""

from __future__ import annotations

import pytest
from aiohttp import web
from llm_gateway import openai_proxy as proxy


class TestBindResolution:
    def test_local_is_loopback_only(self):
        assert proxy.resolve_bind_hosts("local") == ["127.0.0.1"]

    def test_tailnet_includes_loopback_and_tailnet_ip(self, monkeypatch):
        monkeypatch.setattr(proxy, "_tailnet_ip", lambda: "100.96.126.35")
        assert proxy.resolve_bind_hosts("tailnet") == ["127.0.0.1", "100.96.126.35"]

    def test_tailnet_falls_back_to_local_when_unavailable(self, monkeypatch):
        """拿不到 tailnet 地址时收缩到 loopback, 而不是退成 0.0.0.0。"""
        monkeypatch.setattr(proxy, "_tailnet_ip", lambda: None)
        assert proxy.resolve_bind_hosts("tailnet") == ["127.0.0.1"]

    def test_never_defaults_to_all_interfaces(self, monkeypatch):
        """0.0.0.0 会把模型开给同网段任意机器 —— 只能显式指定, 不能是任何默认值。

        (omlxc gw start 给 LiteLLM 下的正是 --host 0.0.0.0, 注释却写"绑 tailnet"。)
        """
        monkeypatch.setattr(proxy, "_tailnet_ip", lambda: "100.96.126.35")
        for spec in ("local", "tailnet"):
            assert "0.0.0.0" not in proxy.resolve_bind_hosts(spec)  # noqa: S104


class TestAuth:
    """直接驱动中间件, 不起真 HTTP 服务(仓里没有 pytest-aiohttp)。"""

    @staticmethod
    def _run(api_key, path="/v1/chat/completions", header=None):
        import asyncio
        from types import SimpleNamespace

        app = {proxy.API_KEY: api_key}
        req = SimpleNamespace(app=app, path=path, headers=header or {})

        async def handler(_):
            return web.json_response({"ok": True})

        return asyncio.run(proxy.auth_middleware(req, handler))

    def test_no_key_means_open(self):
        assert self._run(None).status == 200

    def test_health_stays_open_even_with_key(self):
        """探活不该要 key, 否则 wsvc / launchd 的健康检查全变红。"""
        assert self._run("secret", path="/health").status == 200

    def test_missing_key_rejected(self):
        assert self._run("secret").status == 401

    def test_model_directory_requires_the_same_key(self):
        """模型发现不能成为绕过公开门面认证的侧门。"""
        assert self._run("secret", path="/v1/models").status == 401

    def test_wrong_key_rejected(self):
        assert self._run("secret", header={"Authorization": "Bearer nope"}).status == 401

    def test_right_key_passes(self):
        assert self._run("secret", header={"Authorization": "Bearer secret"}).status == 200

    def test_bare_key_without_bearer_prefix_also_works(self):
        """有些客户端只填 key 不带 Bearer, 别让它们卡在这。"""
        assert self._run("secret", header={"Authorization": "secret"}).status == 200


def test_serve_refuses_non_loopback_without_key(monkeypatch):
    """绑到 loopback 之外却没配 key → 拒绝启动, 而不是静默裸奔。"""
    monkeypatch.setattr(proxy, "_tailnet_ip", lambda: "100.96.126.35")
    monkeypatch.delenv("AETHERFORGE_API_KEY", raising=False)
    monkeypatch.setattr(proxy, "_run_sites", lambda *a: pytest.fail("不该起服务"))
    with pytest.raises(SystemExit) as ei:
        proxy.serve(port=59290, bind="tailnet")
    assert "AETHERFORGE_API_KEY" in str(ei.value)


def test_serve_allows_loopback_without_key(monkeypatch):
    """只绑 loopback 时不强制 key —— 本机进程本来就能直接打各模型端口。

    注意拦的是 _run_sites 而不是 web.run_app: serve 改成多端口后走的是
    AppRunner + 多个 TCPSite。上一版还拦着 run_app, 结果测试真起了服务并
    永久阻塞 —— 整个测试套跑不完。
    """
    monkeypatch.delenv("AETHERFORGE_API_KEY", raising=False)
    seen = {}

    async def fake_sites(app, hosts, ports):
        seen["app"] = app
        seen["hosts"] = hosts
        seen["ports"] = ports

    monkeypatch.setattr(proxy, "_run_sites", fake_sites)
    proxy.serve(port="59290,59291", bind="local")
    assert seen["hosts"] == ["127.0.0.1"]
    assert seen["ports"] == [59290, 59291]
    assert isinstance(seen["app"], web.Application)


def test_ports_parse():
    assert proxy.parse_ports(9290) == [9290]
    assert proxy.parse_ports("9290") == [9290]
    # 过渡期同时占 LiteLLM 的 4000, 调用方零改动
    assert proxy.parse_ports("9290,4000") == [9290, 4000]
    assert proxy.parse_ports(" 9290 , 4000 ") == [9290, 4000]
