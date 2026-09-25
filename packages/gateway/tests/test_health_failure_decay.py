"""Regression test: per-model health-failure tracking must decay over time.

2026-09-11 incident: `coding` hit three consecutive `insufficient_capacity`
errors during a burst of concurrent local testing (real, transient resource
contention on the shared node). Because the failure counter never expired,
every subsequent request for `coding` was skipped without even being
attempted for the rest of the gateway process's life — including minutes
later, once the contention had cleared and a direct request to the same
backend succeeded in under 1.5s. See ModelGateway._record_health_failure /
_is_health_blocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def gateway_config():
    from llm_gateway.gateway import GatewayConfig

    return GatewayConfig(
        omlx_bin="/bin/true",
        local_base_url="http://127.0.0.1",
        model_ports={},
        model_sizes={},
        lmstudio_fallback={},
        ollama_fallback={},
        local_backend="legacy",
        fallback_chain=[],
        warm_pool_ttl=60,
        omlxc_mode="legacy",
    )


@pytest.fixture
def gw(gateway_config):
    from llm_gateway.gateway import ModelGateway

    registry = MagicMock()
    scheduler = MagicMock()
    scheduler.select.return_value = None
    return ModelGateway(registry, scheduler, gateway_config)


class TestHealthFailureDecay:
    def test_not_blocked_below_threshold(self, gw) -> None:
        gw._record_health_failure("coding")
        gw._record_health_failure("coding")
        assert gw._is_health_blocked("coding") is False
        assert gw._health_failure_count("coding") == 2

    def test_blocked_at_threshold(self, gw) -> None:
        for _ in range(3):
            gw._record_health_failure("coding")
        assert gw._is_health_blocked("coding") is True

    def test_transient_failures_decay_after_ttl(self, gw, monkeypatch) -> None:
        """Three failures block the model; once the TTL window has fully
        elapsed since the last one, it must be tried again instead of
        staying blacklisted for the rest of the process's life."""
        import llm_gateway.gateway as gateway_module

        t = [1_000_000.0]
        monkeypatch.setattr(gateway_module.time, "time", lambda: t[0])

        for _ in range(3):
            gw._record_health_failure("coding")
        assert gw._is_health_blocked("coding") is True

        t[0] += gateway_module._HEALTH_FAILURE_TTL_SECONDS + 1
        assert gw._is_health_blocked("coding") is False
        # Checking also clears the stale entry so a fresh streak starts clean.
        assert gw._health_failure_count("coding") == 0

    def test_success_resets_count(self, gw) -> None:
        gw._record_health_failure("coding")
        gw._record_health_failure("coding")
        gw._health_failures.pop("coding", None)  # mirrors the success path in generate()
        assert gw._health_failure_count("coding") == 0
        assert gw._is_health_blocked("coding") is False

    def test_permanent_failure_never_decays(self, gw, monkeypatch) -> None:
        """`no capacity` (a genuinely undeployed model) must stay blocked
        even after the TTL window — unlike transient failures."""
        import llm_gateway.gateway as gateway_module

        t = [1_000_000.0]
        monkeypatch.setattr(gateway_module.time, "time", lambda: t[0])

        gw._record_health_failure("coding-fast", permanent=True)
        assert gw._is_health_blocked("coding-fast") is True

        t[0] += gateway_module._HEALTH_FAILURE_TTL_SECONDS * 100
    def test_no_capacity_failure_decays_after_ttl(self, gw, monkeypatch) -> None:
        """A transient local capacity miss must not blacklist a model forever."""
        import llm_gateway.gateway as gateway_module

        t = [1_000_000.0]
        monkeypatch.setattr(gateway_module.time, "time", lambda: t[0])

        for _ in range(gateway_module._HEALTH_FAILURE_THRESHOLD):
            gw._record_generation_failure("coding-fast", RuntimeError("no capacity"))
        assert gw._is_health_blocked("coding-fast") is True

        t[0] += gateway_module._HEALTH_FAILURE_TTL_SECONDS + 1
        assert gw._is_health_blocked("coding-fast") is False

    def test_missing_model_failure_remains_permanent(self, gw, monkeypatch) -> None:
        import llm_gateway.gateway as gateway_module

        t = [1_000_000.0]
        monkeypatch.setattr(gateway_module.time, "time", lambda: t[0])

        gw._record_generation_failure("missing", RuntimeError("model not found"))
        assert gw._is_health_blocked("missing") is True
        t[0] += gateway_module._HEALTH_FAILURE_TTL_SECONDS + 1
        assert gw._is_health_blocked("missing") is True
        t[0] += gateway_module._HEALTH_FAILURE_TTL_SECONDS + 1
        assert gw._is_health_blocked("coding-fast") is False

    def test_other_models_unaffected(self, gw) -> None:
        for _ in range(3):
            gw._record_health_failure("coding-fast")
        assert gw._is_health_blocked("coding-fast") is True
        assert gw._is_health_blocked("coding") is False
