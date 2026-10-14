"""熔断只记后端故障: 调用方 4xx(非 408/429)不计失败(2026-09-27 reasoning_effort 422 熔断 oMLX)。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from llm_gateway.registry import _counts_as_backend_failure


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_client_errors_do_not_trip_breaker(status):
    assert _counts_as_backend_failure(_StatusError(status)) is False


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_backend_and_throttle_errors_count(status):
    assert _counts_as_backend_failure(_StatusError(status)) is True


def test_status_from_response_attribute_and_plain_errors():
    exc = Exception("x")
    exc.response = SimpleNamespace(status_code=422)
    assert _counts_as_backend_failure(exc) is False
    assert _counts_as_backend_failure(TimeoutError()) is True
    assert _counts_as_backend_failure(ConnectionError("down")) is True
