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
