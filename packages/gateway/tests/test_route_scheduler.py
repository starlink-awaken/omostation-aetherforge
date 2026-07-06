"""Tests for llm_gateway.route_scheduler — RouteStrategies, Route, and scoring logic."""

from __future__ import annotations

import pytest


class TestRouteStrategies:
    def test_balanced_weights(self):
        from llm_gateway.route_scheduler import RouteStrategies

        w = RouteStrategies.BALANCED
        assert w["cost"] == 0.35
        assert w["quota"] == 0.35
        assert w["speed"] == 0.30
        assert sum(w.values()) == pytest.approx(1.0)

    def test_cost_first_weights(self):
        from llm_gateway.route_scheduler import RouteStrategies

        w = RouteStrategies.COST_FIRST
        assert w["cost"] == 0.70
        assert w["speed"] == 0.10
        assert sum(w.values()) == pytest.approx(1.0)

    def test_speed_first_weights(self):
        from llm_gateway.route_scheduler import RouteStrategies

        w = RouteStrategies.SPEED_FIRST
        assert w["cost"] == 0.10
        assert w["speed"] == 0.80
        assert sum(w.values()) == pytest.approx(1.0)

    def test_quota_first_weights(self):
        from llm_gateway.route_scheduler import RouteStrategies

        w = RouteStrategies.QUOTA_FIRST
        assert w["cost"] == 0.10
        assert w["quota"] == 0.70
        assert sum(w.values()) == pytest.approx(1.0)

    def test_get_by_name(self):
        from llm_gateway.route_scheduler import RouteStrategies

        assert RouteStrategies.get("balanced") == RouteStrategies.BALANCED
        assert RouteStrategies.get("COST_FIRST") == RouteStrategies.COST_FIRST
        assert RouteStrategies.get("unknown") == RouteStrategies.BALANCED  # fallback


class TestRoute:
    def test_default_route(self):
        from llm_gateway.route_scheduler import Route

        r = Route()
        assert r.provider == ""
        assert r.model == ""
        assert r.cost_per_1k_input == 0.0
        assert r.score == 0.0
        assert r.strategy == "balanced"
        assert r.quota_pct == 100.0

    def test_custom_route(self):
        from llm_gateway.route_scheduler import Route

        r = Route(
            provider="openai",
            model="gpt-4o",
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            score=0.85,
            strategy="cost_first",
            quota_pct=80.0,
            quota_source="codexbar",
        )
        assert r.provider == "openai"
        assert r.model == "gpt-4o"
        assert r.score == 0.85
        assert r.strategy == "cost_first"
        assert r.quota_pct == 80.0


class TestRouteSchedulerDynamic:
    def test_load_policies(self, tmp_path, monkeypatch):
        # Mock _load_model_provider_map to decouple tests from physical L0 MOF model directory
        from llm_gateway import route_scheduler
        monkeypatch.setattr(route_scheduler, "_load_model_provider_map", lambda: {})

        # 1. 模拟 M1_ROUTING_POLICY_DIR
        policy_dir = tmp_path / "routing_policy"
        policy_dir.mkdir()

        rp_data = """
id: RP-TEST
name: RP-TEST
type: RoutingPolicy
strategy: test_strategy
weights:
  cost: 0.50
  quota: 0.30
  speed: 0.20
constraints:
  max_cost_per_1k_input: 0.10
  min_quota_pct: 15
"""
        (policy_dir / "RP-TEST.yaml").write_text(rp_data, encoding="utf-8")

        # Mock M1_ROUTING_POLICY_DIR
        from llm_gateway import route_scheduler
        monkeypatch.setattr(route_scheduler, "M1_ROUTING_POLICY_DIR", policy_dir)

        # 2. Mock QuotaEngine
        class MockQuotaEngine:
            def __init__(self):
                pass
            def start(self):
                pass
            def stop(self):
                pass
            def wait_ready(self, timeout=8):
                return True
            def get_all_status(self):
                from llm_gateway.quota_engine import ProviderData
                # provider_a: 配额 10% < 15% (不符合 min_quota 15)
                # provider_b: 配额 50% > 15% (全部符合)
                # provider_c: 配额 10% < 15% (不符合 min_quota 15)
                return {
                    "provider_a": ProviderData(provider="provider_a", available=True, has_key=True, quota_pct=10.0),
                    "provider_b": ProviderData(provider="provider_b", available=True, has_key=True, quota_pct=50.0),
                    "provider_c": ProviderData(provider="provider_c", available=True, has_key=True, quota_pct=10.0),
                }

        monkeypatch.setattr(route_scheduler, "QuotaEngine", MockQuotaEngine)

        # 3. Mock PricingRegistry
        class MockPricingRegistry:
            def get_cost(self, model):
                if model == "test_model":
                    # provider A 成本高为 0.08
                    return {"input": 0.08, "output": 0.05}
                return {"input": 0.01, "output": 0.02}
            def get_price(self, model, provider):
                return "test_model"

        monkeypatch.setattr(route_scheduler, "PricingRegistry", MockPricingRegistry)

        # 4. 执行测试
        scheduler = route_scheduler.RouteScheduler()
        assert "test_strategy" in scheduler._policies

        # 校验权重
        p = scheduler._policies["test_strategy"]
        assert p["weights"]["cost"] == 0.50

        # 验证 select() 过滤和排序
        route = scheduler.select(model="test_model", strategy="test_strategy")
        assert route is not None
        assert route.provider == "provider_b"
