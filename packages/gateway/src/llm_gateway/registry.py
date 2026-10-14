"""Model Registry — provider and model lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from .circuit_breaker import CircuitBreakerRegistry
from .providers.base import BaseLLMProvider
from .retry import RetryConfig, with_retry
from .types import ChatOptions, ChatResult, ModelDescriptor, StreamChunk

_log = logging.getLogger(__name__)


def _counts_as_backend_failure(exc: BaseException) -> bool:
    """熔断只该记"后端坏了", 不该记"请求本身有问题"。

    4xx(408/429 除外)是调用方参数/格式问题, 换个时刻重试结果一样 —— 此前照记失败,
    一个带非标准参数的客户端(2026-09-27: reasoning_effort → omlxc 422)就能把整个
    ENG-OMLX-LOCAL 熔断, 殃及同后端所有正常请求。异常照常上抛, 兜底链行为不变。
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return False
    return True


class ModelRegistry:
    """Registry managing providers, discovered models, circuit breakers, and retry.

    Integrates circuit breakers per (provider, model) pair and optional retry
    logic. See ``_circuit_key`` for why the breaker is not keyed by provider
    alone.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}
        self._models: dict[str, tuple[ModelDescriptor, str]] = {}
        self.circuit_breaker = CircuitBreakerRegistry()
        self.retry_config: RetryConfig | None = None
        self._scheduler_ref: Any = None
        self.metrics: Any = None
        self.rate_limiter: Any = None

    # ------------------------------------------------------------------
    # Provider registration
    # ------------------------------------------------------------------

    def register(self, provider: BaseLLMProvider) -> None:
        """Register a single provider."""
        self._providers[provider.name] = provider

    def register_many(self, providers: list[BaseLLMProvider]) -> None:
        """Register multiple providers."""
        for p in providers:
            self.register(p)

    def provider_names(self) -> tuple[str, ...]:
        """All registered provider/engine ids, sorted for deterministic iteration.

        2026-08-23: exists so callers building candidate model-id prefixes
        (see gateway._resolve_model_id) can enumerate what's actually
        registered instead of maintaining a hand-written engine list that
        silently drifts out of sync with SSOT (a real incident: 10 cloud
        engines were missing from such a list, so any client requesting a
        cloud model by its short name got "not in registry" even though the
        provider and model were correctly configured).
        """
        return tuple(sorted(self._providers))

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    async def refresh(self, discover_timeout: float = 10.0) -> list[ModelDescriptor]:
        """Discover models from all registered providers.

        Each provider's discovery is bounded by ``discover_timeout`` seconds so a
        single slow/hanging endpoint (e.g. an unreachable cloud provider) cannot
        block discovery of the rest.
        """
        all_models: list[ModelDescriptor] = []
        discovered: dict[str, tuple[ModelDescriptor, str]] = {}

        async def _discover(p: BaseLLMProvider) -> list[ModelDescriptor]:
            return await asyncio.wait_for(p.discover(), timeout=discover_timeout)

        tasks = [(name, _discover(provider)) for name, provider in self._providers.items()]
        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        for (name, _), result in zip(tasks, results):
            if isinstance(result, BaseException):
                _log.warning("[ModelRegistry] discover failed for %s: %s", name, result)
                continue
            for m in result:
                discovered[m.id] = (m, name)
                all_models.append(m)
        # 原子替换，避免周期 refresh 时并发请求看到“registry 突然清空”。
        self._models = discovered
        return all_models

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------

    def get_all(self) -> list[ModelDescriptor]:
        """Return all discovered models."""
        return [entry[0] for entry in self._models.values()]

    def list_models(self) -> list[ModelDescriptor]:
        """Legacy alias for get_all()."""
        return self.get_all()

    def get(self, model_id: str) -> ModelDescriptor | None:
        """Get a single model descriptor by ID."""
        entry = self._models.get(model_id)
        return entry[0] if entry else None

    # ------------------------------------------------------------------
    # Chat / streaming
    # ------------------------------------------------------------------

    @staticmethod
    def _circuit_key(provider_name: str, model_id: str) -> str:
        """Circuit breaker key scoped to (provider, model), not the provider alone.

        A provider often serves several models with independent availability
        (e.g. omlx_app only has real placements for some of its catalog).
        Keying purely by provider_name lets failures on an unrelated,
        never-deployed model (e.g. coding-fast) trip the breaker for every
        other model on that same provider (e.g. coding), even though that
        other model is healthy and reachable directly.
        """
        return f"{provider_name}::{model_id}"

    def _get_provider_for(self, model_id: str) -> tuple[BaseLLMProvider, str] | None:
        entry = self._models.get(model_id)
        if not entry:
            return None
        _, provider_name = entry
        provider = self._providers.get(provider_name)
        if not provider:
            return None
        circuit_key = self._circuit_key(provider_name, model_id)
        if not self.circuit_breaker.can_request(circuit_key):
            raise RuntimeError(f"Circuit breaker open for provider {provider_name} (model {model_id})")
        return provider, provider_name

    async def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult | None:
        """Send a chat request through the registry with circuit breaker and retry."""
        p = self._get_provider_for(model_id)
        if not p:
            return None
        provider, provider_name = p
        circuit_key = self._circuit_key(provider_name, model_id)

        from .tracing import trace_llm_call

        with trace_llm_call(model_id, messages, options) as gen:
            try:
                if self.retry_config:
                    result = await with_retry(
                        lambda: provider.chat(model_id, messages, options),
                        config=self.retry_config,
                    )
                else:
                    result = await provider.chat(model_id, messages, options)

                if gen and result:
                    input_tok = getattr(result, "input_tokens", 0)
                    output_tok = getattr(result, "output_tokens", 0)
                    gen.end(
                        output=result.content or "",
                        usage={
                            "input": input_tok,
                            "output": output_tok,
                        },
                    )

                self.circuit_breaker.record_success(circuit_key)
                return result
            except Exception as e:
                if gen:
                    gen.end(status_message=str(e))
                if _counts_as_backend_failure(e):
                    self.circuit_breaker.record_failure(circuit_key)
                raise
            finally:
                if self._scheduler_ref is not None:
                    self._scheduler_ref.release_load(model_id)

    async def chat_stream(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat response through the registry."""
        p = self._get_provider_for(model_id)
        if not p:
            raise RuntimeError(f"Model {model_id} not found or provider unavailable")
        provider, provider_name = p
        circuit_key = self._circuit_key(provider_name, model_id)
        try:
            try:
                # 2026-08-23: 原来调 provider.stream(...) —— BaseLLMProvider 的
                # 接口是 stream_chat, SSOTProviderAdapter 没有 stream 方法,
                # 这条流式路径对所有 SSOT 引擎一直 AttributeError 进 circuit
                # breaker 记失败, 是死链。
                async for chunk in provider.stream_chat(model_id, messages, options):
                    yield chunk
            except NotImplementedError:
                result = await provider.chat(model_id, messages, options)
                yield StreamChunk(
                    id=result.id,
                    model=result.model,
                    content=result.content,
                    finish_reason=result.finish_reason,
                )
            self.circuit_breaker.record_success(circuit_key)
        except Exception as e:
            if _counts_as_backend_failure(e):
                self.circuit_breaker.record_failure(circuit_key)
            raise
        finally:
            if self._scheduler_ref is not None:
                self._scheduler_ref.release_load(model_id)

    # ------------------------------------------------------------------
    # Scheduler linkage
    # ------------------------------------------------------------------

    def set_scheduler(self, scheduler: Any) -> None:
        """Attach a scheduler for load tracking."""
        self._scheduler_ref = scheduler

    def set_metrics_collector(self, metrics: Any) -> None:
        """Attach a metrics collector for observability."""
        self.metrics = metrics

    def set_rate_limiter(self, rate_limiter: Any) -> None:
        """Attach a rate limiter for request throttling."""
        self.rate_limiter = rate_limiter

    # ------------------------------------------------------------------
    # Provider access
    # ------------------------------------------------------------------

    def get_provider(self, model_id: str) -> BaseLLMProvider | None:
        """Get the provider that owns a model."""
        entry = self._models.get(model_id)
        if not entry:
            return None
        return self._providers.get(entry[1])

    def get_providers(self) -> list[BaseLLMProvider]:
        """Return all registered providers."""
        return list(self._providers.values())

    async def health_check(self, provider_name: str) -> bool:
        """Check health of a specific provider by name."""
        provider = self._providers.get(provider_name)
        if not provider:
            return False
        return await provider.health()
