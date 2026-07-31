"""Tests for ModelGateway — unified gateway, K1 hard block, MemoryGuard, strip_thinking.

这些测试用 mock 隔离外部依赖 (omlxc / HTTP), 聚焦网关本身的逻辑.

运行:
    cd packages/gateway && uv run pytest tests/test_model_gateway.py -v
    # 或
    python3 packages/gateway/tests/test_model_gateway.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def gateway_config():
    """最小化网关配置 (测试用)."""
    from llm_gateway.gateway import GatewayConfig

    return GatewayConfig(
        omlx_bin="/bin/true",  # mock
        local_base_url="http://127.0.0.1",
        model_ports={
            "coding-fast": 8081,
            "test-model": 9999,
        },
        fallback_chain=["coding-fast", "test-model"],
        warm_pool_ttl=60,
    )


@pytest.fixture
def mock_registry():
    """Mock ModelRegistry."""
    reg = MagicMock()
    reg.get_all.return_value = []
    reg.list_models.return_value = []
    return reg


@pytest.fixture
def mock_scheduler():
    """Mock ModelScheduler."""
    return MagicMock()


# ── strip_thinking ──────────────────────────────────────────────────────────


class TestStripThinking:
    """网关层 thinking 剥离."""

    def test_strip_full_think_block(self):
        from llm_gateway.gateway import _strip_thinking

        text = "<think>这是思考过程, 很长很长</think>这是最终回答"
        result = _strip_thinking(text)
        assert "<think>" not in result
        assert "这是最终回答" in result
        assert "思考过程" not in result

    def test_strip_unclosed_think_tag(self):
        from llm_gateway.gateway import _strip_thinking

        text = "</think>回答开头有残留标签"
        result = _strip_thinking(text)
        assert "</think>" not in result
        assert "回答开头" in result

    def test_strip_multiple_think_blocks(self):
        from llm_gateway.gateway import _strip_thinking

        text = "<think>思考1</think>回答1<think>思考2</think>回答2"
        result = _strip_thinking(text)
        assert "思考" not in result
        assert "回答1" in result
        assert "回答2" in result

    def test_empty_input(self):
        from llm_gateway.gateway import _strip_thinking

        assert _strip_thinking("") == ""
        assert _strip_thinking(None) is None

    def test_no_thinking_unchanged(self):
        from llm_gateway.gateway import _strip_thinking

        text = "这是一段普通回答, 没有 thinking"
        assert _strip_thinking(text) == text

    def test_case_insensitive(self):
        from llm_gateway.gateway import _strip_thinking

        text = "<THINK>大写标签</THINK>回答"
        result = _strip_thinking(text)
        assert "<THINK>" not in result
        assert "回答" in result


# ── _is_sensitive (K1) ───────────────────────────────────────────────────────


class TestIsSensitive:
    """K1 敏感流判断."""

    def test_gov_domain_sensitive(self):
        from llm_gateway.gateway import _is_sensitive

        assert _is_sensitive("通知", "https://www.gov.cn/some/page") is True

    def test_mail_domain_sensitive(self):
        from llm_gateway.gateway import _is_sensitive

        assert _is_sensitive("邮件", "https://mail.google.com/inbox") is True

    def test_oa_domain_sensitive(self):
        from llm_gateway.gateway import _is_sensitive

        assert _is_sensitive("公文", "https://oa.company.com/doc/123") is True

    def test_private_ip_sensitive(self):
        from llm_gateway.gateway import _is_sensitive

        assert _is_sensitive("", "http://10.0.0.1/admin") is True
        assert _is_sensitive("", "http://192.168.1.1/oa") is True

    def test_sensitive_keyword_in_title(self):
        from llm_gateway.gateway import _is_sensitive

        assert _is_sensitive("关于报送总结的通知", "https://example.com") is True
        assert _is_sensitive("会议纪要", "https://example.com") is True

    def test_public_domain_not_sensitive(self):
        from llm_gateway.gateway import _is_sensitive

        # 公开域名 + 非敏感标题 = 不敏感
        assert _is_sensitive("Python教程", "https://docs.python.org/3") is False
        assert _is_sensitive("知乎首页", "https://www.zhihu.com/hot") is False

    def test_empty_title_and_url(self):
        from llm_gateway.gateway import _is_sensitive

        # 空 = 保守判敏感
        assert _is_sensitive("", "") is True


# ── MemoryGuard ──────────────────────────────────────────────────────────────


class TestMemoryGuard:
    """内存守卫."""

    def test_can_load_sufficient_memory(self):
        from llm_gateway.gateway import MemoryGuard

        guard = MemoryGuard(safety_factor=1.2)
        with patch.object(guard, "_get_free_memory_gb", return_value=100.0):
            assert guard.can_load(10.0) is True  # 10*1.2=12 < 100

    def test_can_load_insufficient_memory(self):
        from llm_gateway.gateway import MemoryGuard

        guard = MemoryGuard(safety_factor=1.2)
        with patch.object(guard, "_get_free_memory_gb", return_value=10.0):
            assert guard.can_load(10.0) is False  # 10*1.2=12 > 10

    def test_safety_factor_respected(self):
        from llm_gateway.gateway import MemoryGuard

        # safety_factor=2.0 更保守
        guard = MemoryGuard(safety_factor=2.0)
        with patch.object(guard, "_get_free_memory_gb", return_value=15.0):
            assert guard.can_load(10.0) is False  # 10*2=20 > 15


# ── GatewayRequest / GatewayResponse ─────────────────────────────────────────


class TestDataClasses:
    """数据结构创建."""

    def test_request_defaults(self):
        from llm_gateway.gateway import GatewayRequest

        req = GatewayRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.model == ""
        assert req.task == ""
        assert req.timeout == 30.0

    def test_request_with_sensitive_context(self):
        from llm_gateway.gateway import GatewayRequest

        req = GatewayRequest(
            messages=[{"role": "user", "content": "hi"}],
            content_title="通知",
            content_url="https://oa.com/doc",
        )
        assert req.content_title == "通知"

    def test_response_defaults(self):
        from llm_gateway.gateway import GatewayResponse

        resp = GatewayResponse(content="hi", model="test", latency_ms=100)
        assert resp.tokens_in == 0
        assert resp.cost_usd == 0.0
        assert resp.error == ""
        assert resp.stripped_thinking is False


# ── ModelGateway._resolve_model_id ───────────────────────────────────────────


class TestResolveModelId:
    """模型 ID 解析."""

    def test_exact_match(self, gateway_config, mock_registry, mock_scheduler):
        from llm_gateway.gateway import ModelGateway
        from llm_gateway.types import ModelDescriptor

        mock_registry.get.return_value = ModelDescriptor(id="coding-fast", name="coding-fast")
        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        assert gw._resolve_model_id("coding-fast") == "coding-fast"

    def test_bare_name_with_engine_prefix(self, gateway_config, mock_registry, mock_scheduler):
        from llm_gateway.gateway import ModelGateway
        from llm_gateway.types import ModelDescriptor

        # registry.get("coding-fast") = None
        # registry.get("ENG-OMLX-LOCAL/coding-fast") = descriptor
        def mock_get(model_id):
            if model_id == "ENG-OMLX-LOCAL/coding-fast":
                return ModelDescriptor(id="ENG-OMLX-LOCAL/coding-fast")
            return None

        mock_registry.get.side_effect = mock_get
        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        result = gw._resolve_model_id("coding-fast")
        assert result == "ENG-OMLX-LOCAL/coding-fast"

    def test_no_match_returns_none(self, gateway_config, mock_registry, mock_scheduler):
        from llm_gateway.gateway import ModelGateway

        mock_registry.get.return_value = None
        mock_registry.list_models.return_value = []
        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        assert gw._resolve_model_id("nonexistent") is None


# ── ModelGateway.generate (K1 hard block) ────────────────────────────────────


class TestGenerateWithKI:
    """K1 敏感流硬拦在网关层."""

    def test_sensitive_local_only(self, gateway_config, mock_registry, mock_scheduler):
        """敏感流: 只用本地模型, 不 fallback 到云端."""
        from llm_gateway.gateway import GatewayRequest, ModelGateway
        from llm_gateway.types import ChatResult

        # Mock chat result
        mock_registry.chat = AsyncMock(return_value=ChatResult(
            content="本地回答", model="coding-fast",
        ))
        mock_registry.get_provider.return_value = MagicMock(name="ENG-OMLX-LOCAL")
        mock_registry.get.return_value = MagicMock(id="ENG-OMLX-LOCAL/coding-fast")

        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        # Mock _ensure_model to always succeed
        async def fake_ensure(model_name):
            gw._loaded_models[model_name] = time.time()
            return True

        gw._ensure_model = fake_ensure  # type: ignore[method-assign]

        req = GatewayRequest(
            messages=[{"role": "user", "content": "hello"}],
            model="coding-fast",
            content_title="关于报送的通知",
            content_url="https://oa.gov.cn/doc/123",
        )

        resp = asyncio.run(gw.generate(req))

        # 敏感流应该成功 (用本地模型)
        assert resp.error == ""
        assert resp.content == "本地回答"

    def test_strip_thinking_applied(self, gateway_config, mock_registry, mock_scheduler):
        """网关出口强制 strip_thinking."""
        from llm_gateway.gateway import GatewayRequest, ModelGateway
        from llm_gateway.types import ChatResult

        mock_registry.chat = AsyncMock(return_value=ChatResult(
            content="<think>思考中...</think>最终回答",
            model="coding-fast",
        ))
        mock_registry.get_provider.return_value = MagicMock(name="test")
        mock_registry.get.return_value = MagicMock(id="coding-fast")

        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        async def fake_ensure(model_name):
            gw._loaded_models[model_name] = time.time()
            return True

        gw._ensure_model = fake_ensure  # type: ignore[method-assign]

        req = GatewayRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="coding-fast",
        )

        resp = asyncio.run(gw.generate(req))

        assert "<think>" not in resp.content
        assert "最终回答" in resp.content
        assert resp.stripped_thinking is True


# ── ModelGateway.embed (降级链) ──────────────────────────────────────────────


class TestEmbedFallback:
    """Embedding 降级链."""

    def test_fallback_to_8188_when_8183_fails(self, gateway_config, mock_registry, mock_scheduler):
        """8183 失败时自动降级到 8188."""
        from llm_gateway.gateway import ModelGateway

        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        # Build async context manager mock for response
        def make_response_mock(status, json_data=None):
            resp = AsyncMock()
            resp.status = status
            if json_data:
                resp.json = AsyncMock(return_value=json_data)
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        resp_8183 = make_response_mock(503)
        resp_8188 = make_response_mock(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

        def mock_post(url, **kwargs):
            if "8183" in url:
                return resp_8183
            return resp_8188

        mock_session = AsyncMock()
        mock_session.post = mock_post

        with patch("aiohttp.ClientSession") as MockSession:
            MockSession.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = asyncio.run(gw.embed(["test text"]))

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]


# ── ModelGateway.health ──────────────────────────────────────────────────────


class TestHealthCheck:
    """健康检查."""

    def test_all_healthy(self, gateway_config, mock_registry, mock_scheduler):
        from llm_gateway.gateway import ModelGateway

        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)

        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as MockSession:
            MockSession.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = asyncio.run(gw.health())

        assert "coding-fast" in result
        assert result["coding-fast"]["status"] == "healthy"


# ── run_async (嵌事件池兼容) ─────────────────────────────────────────────────


class TestRunAsync:
    """run_async 辅助函数."""

    def test_run_async_no_loop(self):
        """无事件循环时正常运行."""
        from llm_gateway.gateway import run_async

        async def _coro():
            return "ok"

        result = run_async(_coro())
        assert result == "ok"

    def test_run_async_nested_loop(self):
        """已有事件循环时不崩."""
        from llm_gateway.gateway import run_async

        async def _inner():
            return "nested"

        async def _outer():
            # 在已有事件循环中调用 run_async
            return run_async(_inner())

        result = asyncio.run(_outer())
        assert result == "nested"


# ── get_gateway / reset_gateway (单例) ────────────────────────────────────────


class TestSingleton:
    """gateway 单例."""

    def test_get_gateway_returns_same_instance(self):
        from llm_gateway.gateway import get_gateway, reset_gateway, GatewayConfig

        reset_gateway()
        config = GatewayConfig(background_tasks_enabled=False)
        gw1 = get_gateway(config)
        gw2 = get_gateway()
        assert gw1 is gw2
        reset_gateway()

    def test_reset_gateway_creates_new(self):
        from llm_gateway.gateway import get_gateway, reset_gateway, GatewayConfig

        reset_gateway()
        config = GatewayConfig(background_tasks_enabled=False)
        gw1 = get_gateway(config)
        reset_gateway()
        gw2 = get_gateway()
        assert gw1 is not gw2
        reset_gateway()


# ── MemoryGuard 接入 _ensure_model ────────────────────────────────────────────


class TestMemoryGuardIntegration:
    """MemoryGuard 在 _ensure_model 中实际生效."""

    def test_refuse_load_when_memory_low(self, gateway_config, mock_registry, mock_scheduler):
        """内存不足时拒绝加载."""
        from llm_gateway.gateway import ModelGateway

        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)
        # Mock MemoryGuard 返回 False
        gw._memory_guard = MagicMock()
        gw._memory_guard.can_load.return_value = False

        # coding-fast 在 model_sizes 中有 18GB
        result = asyncio.run(gw._ensure_model("coding-fast"))
        assert result is False
        gw._memory_guard.can_load.assert_called_once_with(18.0)

    def test_skip_check_when_disabled(self, gateway_config, mock_registry, mock_scheduler):
        """memory_check_enabled=False 时跳过检查."""
        from llm_gateway.gateway import ModelGateway

        gateway_config.memory_check_enabled = False
        gw = ModelGateway(mock_registry, mock_scheduler, gateway_config)
        gw._memory_guard = MagicMock()

        # Mock _wait_healthy 和 subprocess
        async def fake_wait(*args, **kwargs):
            return True

        gw._wait_healthy = fake_wait

        with patch("asyncio.create_subprocess_exec") as mock_proc:
            proc_mock = AsyncMock()
            proc_mock.communicate = AsyncMock(return_value=(b"", b""))
            proc_mock.returncode = 0
            mock_proc.return_value = proc_mock

            result = asyncio.run(gw._ensure_model("coding-fast"))

        # 内存检查被跳过, can_load 不应被调用
        gw._memory_guard.can_load.assert_not_called()


# ── Runner ───────────────────────────────────────────────────────────────────


def run_all():
    """Run all tests without pytest (fallback)."""
    import sys

    test_classes = [
        TestStripThinking,
        TestIsSensitive,
        TestMemoryGuard,
        TestDataClasses,
        TestResolveModelId,
        TestGenerateWithKI,
        TestEmbedFallback,
        TestHealthCheck,
        TestRunAsync,
        TestSingleton,
        TestMemoryGuardIntegration,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    passed += 1
                    print(f"  ✅ {cls.__name__}.{name}")
                except Exception as e:
                    failed += 1
                    print(f"  ❌ {cls.__name__}.{name}: {e}")

    total = passed + failed
    print(f"\n  ModelGateway tests: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
