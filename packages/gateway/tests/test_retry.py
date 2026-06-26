"""Tests for llm_gateway.retry — RetryConfig, _backoff, _is_retryable."""

from __future__ import annotations

import pytest


class TestRetryConfig:
    def test_default_config(self):
        from llm_gateway.retry import RetryConfig

        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay_ms == 500.0
        assert cfg.max_delay_ms == 10_000.0
        assert cfg.retryable_statuses == [429, 500, 502, 503, 504]
        assert cfg.call_timeout_ms == 30_000.0
        assert cfg.total_timeout_ms == 120_000.0

    def test_custom_config(self):
        from llm_gateway.retry import RetryConfig

        cfg = RetryConfig(max_retries=5, base_delay_ms=1000, retryable_statuses=[500, 502])
        assert cfg.max_retries == 5
        assert cfg.base_delay_ms == 1000
        assert cfg.retryable_statuses == [500, 502]

    def test_base_delay_backward_compat(self):
        from llm_gateway.retry import RetryConfig

        cfg = RetryConfig(base_delay=2.0)
        assert cfg.base_delay_ms == 2000.0
        assert cfg.base_delay == 2.0


class TestIsRetryable:
    def test_retryable_statuses(self):
        from llm_gateway.retry import _is_retryable

        retryable = [429, 500, 502, 503, 504]
        assert _is_retryable(429, retryable) is True
        assert _is_retryable(500, retryable) is True
        assert _is_retryable(503, retryable) is True

    def test_non_retryable_statuses(self):
        from llm_gateway.retry import _is_retryable

        retryable = [429, 500, 502, 503, 504]
        assert _is_retryable(400, retryable) is False
        assert _is_retryable(401, retryable) is False
        assert _is_retryable(404, retryable) is False
        assert _is_retryable(200, retryable) is False


class TestBackoff:
    def test_exponential_backoff(self):
        from llm_gateway.retry import RetryConfig, _backoff

        cfg = RetryConfig(base_delay_ms=500, max_delay_ms=10_000)
        assert _backoff(1, cfg) == 500
        assert _backoff(2, cfg) == 1000
        assert _backoff(3, cfg) == 2000
        assert _backoff(4, cfg) == 4000

    def test_backoff_capped_at_max(self):
        from llm_gateway.retry import RetryConfig, _backoff

        cfg = RetryConfig(base_delay_ms=500, max_delay_ms=2000)
        assert _backoff(5, cfg) == 2000  # 500 * 2^4 = 8000, capped at 2000
        assert _backoff(10, cfg) == 2000


@pytest.mark.asyncio
class TestWithRetry:
    async def test_success_no_retry(self):
        from llm_gateway.retry import with_retry

        call_count = 0

        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(succeed)
        assert result == "ok"
        assert call_count == 1

    async def test_retry_on_value_error(self):
        from llm_gateway.retry import RetryConfig, with_retry

        call_count = 0
        cfg = RetryConfig(max_retries=3, base_delay_ms=1)

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "recovered"

        result = await with_retry(fail_then_succeed, config=cfg)
        assert result == "recovered"
        assert call_count == 3

    async def test_non_retryable_status_raises_immediately(self):
        from llm_gateway.retry import RetryConfig, with_retry

        call_count = 0
        cfg = RetryConfig(max_retries=3, base_delay_ms=1)

        class HttpError(Exception):
            def __init__(self):
                self.status_code = 400

        async def fail():
            nonlocal call_count
            call_count += 1
            raise HttpError()

        with pytest.raises(HttpError):
            await with_retry(fail, config=cfg)
        assert call_count == 1  # no retries for non-retryable status

    async def test_exhausted_retries_raises_last_error(self):
        from llm_gateway.retry import RetryConfig, with_retry

        call_count = 0
        cfg = RetryConfig(max_retries=2, base_delay_ms=1)

        class HttpError(Exception):
            def __init__(self):
                self.status_code = 500

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise HttpError()

        with pytest.raises(HttpError):
            await with_retry(always_fail, config=cfg)
        assert call_count == 2
