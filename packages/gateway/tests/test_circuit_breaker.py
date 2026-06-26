"""Tests for llm_gateway.circuit_breaker — CircuitBreakerRegistry state machine."""

from __future__ import annotations

import time


class TestCircuitBreakerConfig:
    def test_default_config(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig

        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 3
        assert cfg.reset_timeout_ms == 30_000
        assert cfg.half_open_max_requests == 1
        assert cfg.recovery_timeout == 30.0

    def test_custom_config(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig

        cfg = CircuitBreakerConfig(failure_threshold=5, reset_timeout_ms=10_000, half_open_max_requests=2)
        assert cfg.failure_threshold == 5
        assert cfg.reset_timeout_ms == 10_000
        assert cfg.half_open_max_requests == 2

    def test_recovery_timeout_backward_compat(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig

        cfg = CircuitBreakerConfig(recovery_timeout=15.0)
        assert cfg.reset_timeout_ms == 15_000
        assert cfg.recovery_timeout == 15.0


class TestCircuitBreakerRegistry:
    def test_initial_state_closed(self):
        from llm_gateway.circuit_breaker import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        assert reg.get_state("p1") == "CLOSED"
        assert reg.can_request("p1") is True

    def test_configure_custom_config(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        cfg = CircuitBreakerConfig(failure_threshold=2, reset_timeout_ms=100)
        reg.configure("p1", cfg)
        assert reg.can_request("p1") is True

    def test_opens_after_failure_threshold(self):
        from llm_gateway.circuit_breaker import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        reg.configure("p1")

        reg.record_failure("p1")
        assert reg.get_state("p1") == "CLOSED"

        reg.record_failure("p1")
        assert reg.get_state("p1") == "CLOSED"

        reg.record_failure("p1")
        assert reg.get_state("p1") == "OPEN"
        assert reg.can_request("p1") is False

    def test_record_success_resets_failures(self):
        from llm_gateway.circuit_breaker import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        reg.configure("p1")

        reg.record_failure("p1")
        reg.record_failure("p1")
        reg.record_success("p1")
        assert reg.get_state("p1") == "CLOSED"

        # Failures reset, so 3 more are needed to open
        reg.record_failure("p1")
        reg.record_failure("p1")
        reg.record_failure("p1")
        assert reg.get_state("p1") == "OPEN"

    def test_half_open_after_timeout(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        cfg = CircuitBreakerConfig(failure_threshold=1, reset_timeout_ms=1)
        reg.configure("p1", cfg)

        reg.record_failure("p1")
        assert reg.get_state("p1") == "OPEN"
        assert reg.can_request("p1") is False

        time.sleep(0.01)
        assert reg.get_state("p1") == "HALF_OPEN"
        assert reg.can_request("p1") is True

    def test_half_open_success_resets_circuit(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        cfg = CircuitBreakerConfig(failure_threshold=1, reset_timeout_ms=1)
        reg.configure("p1", cfg)

        reg.record_failure("p1")
        time.sleep(0.01)
        assert reg.get_state("p1") == "HALF_OPEN"

        reg.record_success("p1")
        assert reg.get_state("p1") == "CLOSED"

    def test_half_open_failure_goes_back_to_open(self):
        from llm_gateway.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        cfg = CircuitBreakerConfig(failure_threshold=1, reset_timeout_ms=1)
        reg.configure("p1", cfg)

        reg.record_failure("p1")
        time.sleep(0.01)
        assert reg.get_state("p1") == "HALF_OPEN"

        reg.record_failure("p1")
        assert reg.get_state("p1") == "OPEN"

    def test_provider_isolation(self):
        from llm_gateway.circuit_breaker import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        reg.configure("p1")
        reg.configure("p2")

        reg.record_failure("p1")
        reg.record_failure("p1")
        reg.record_failure("p1")
        assert reg.get_state("p1") == "OPEN"
        assert reg.get_state("p2") == "CLOSED"
        assert reg.can_request("p1") is False
        assert reg.can_request("p2") is True

    def test_get_status(self):
        from llm_gateway.circuit_breaker import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        reg.configure("p1")
        reg.configure("p2")

        reg.record_failure("p1")
        reg.record_failure("p1")
        reg.record_failure("p1")

        status = reg.get_status()
        assert "p1" in status
        assert "p2" in status
        assert status["p1"]["state"] == "OPEN"
        assert status["p1"]["failures"] == 3
        assert status["p2"]["state"] == "CLOSED"


class TestLegacyCircuitBreaker:
    def test_legacy_wrapper(self):
        from llm_gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", cfg)
        assert cb.name == "test"
        assert cb.state == "closed"

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        cb.record_success()
        assert cb.state == "open"  # still open, half_open needs timeout
