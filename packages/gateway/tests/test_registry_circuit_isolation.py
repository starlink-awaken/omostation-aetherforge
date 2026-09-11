"""Regression test: circuit breaker must be scoped per (provider, model), not
just per provider.

2026-09-11 incident: a provider serving several models (e.g. omlx_app) had
its breaker trip from repeated failures on one never-deployed model
(coding-fast), which then blocked every *other*, healthy model on the same
provider (coding) for the reset-timeout window -- even though the healthy
model never itself failed. See registry.py::ModelRegistry._circuit_key.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from llm_gateway.circuit_breaker import CircuitBreakerConfig
from llm_gateway.providers.base import BaseLLMProvider
from llm_gateway.registry import ModelRegistry
from llm_gateway.types import ChatResult, ModelDescriptor


class _TwoModelProvider(BaseLLMProvider):
    """One provider serving a always-broken model and an always-healthy one."""

    def __init__(self) -> None:
        self._broken = {"coding-fast"}

    @property
    def name(self) -> str:
        return "ENG-OMLX-LOCAL"

    @property
    def provider_type(self) -> str:
        return "local"

    async def discover(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor(id="coding", provider=self.name),
            ModelDescriptor(id="coding-fast", provider=self.name),
        ]

    async def chat(self, model: str, messages: list[dict[str, Any]], options: Any = None) -> ChatResult:
        if model in self._broken:
            raise RuntimeError(f"{model}: no placement on this provider")
        return ChatResult(id="x", model=model, content="ok")


@pytest.fixture
def registry() -> ModelRegistry:
    reg = ModelRegistry()
    provider = _TwoModelProvider()
    reg.register(provider)
    reg._models["coding"] = (ModelDescriptor(id="coding", provider=provider.name), provider.name)
    reg._models["coding-fast"] = (ModelDescriptor(id="coding-fast", provider=provider.name), provider.name)
    reg.circuit_breaker.configure(
        ModelRegistry._circuit_key(provider.name, "coding"),
        CircuitBreakerConfig(failure_threshold=3),
    )
    reg.circuit_breaker.configure(
        ModelRegistry._circuit_key(provider.name, "coding-fast"),
        CircuitBreakerConfig(failure_threshold=3),
    )
    return reg


class TestCircuitBreakerPerModelIsolation:
    def test_broken_model_does_not_trip_healthy_sibling(self, registry: ModelRegistry) -> None:
        async def _run() -> Any:
            # coding-fast fails repeatedly past the failure threshold.
            for _ in range(5):
                with pytest.raises(RuntimeError):
                    await registry.chat("coding-fast", [{"role": "user", "content": "hi"}])

            # coding-fast's own breaker is now open.
            assert (
                registry.circuit_breaker.get_state(ModelRegistry._circuit_key("ENG-OMLX-LOCAL", "coding-fast"))
                == "OPEN"
            )

            # coding shares the same provider but must remain unaffected.
            assert (
                registry.circuit_breaker.get_state(ModelRegistry._circuit_key("ENG-OMLX-LOCAL", "coding")) == "CLOSED"
            )
            result = await registry.chat("coding", [{"role": "user", "content": "hi"}])
            assert result is not None
            assert result.content == "ok"

        asyncio.run(_run())
