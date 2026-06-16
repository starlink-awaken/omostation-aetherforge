"""llm-gateway bus adapter (R61, Month 3).

llm-gateway is the unified LLM provider abstraction layer (Ollama / OpenAI /
Anthropic / Gemini / DeepSeek / HITL). Every LLM call should be visible in
the agora I0 bus so:
  * omo agents can correlate agent decisions with token usage
  * metaos governance can audit cost / rate-limit policies
  * runtime can throttle per-model based on cross-pipeline signals

This adapter wraps the existing LLM call site events. It is intentionally
minimal: emit one event per call with the provider/model/tokens/result shape,
and let consumers subscribe to `llm:call:*` patterns.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _try_import_bus():
    try:
        from bus_foundation import BusEnvelope, publish  # type: ignore

        return BusEnvelope, publish
    except ImportError:
        return None, None


def emit_event(
    event_type: str,
    source: str = "llm-gateway",
    payload: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> str | None:
    bus_envelope, bus_publish = _try_import_bus()
    if bus_envelope is None or bus_publish is None:
        logger.debug("agora_bus_unavailable_skipping_event type=%s", event_type)
        return None
    envelope = bus_envelope(
        type=event_type,
        source=source,
        payload=payload or {},
        trace_id=trace_id,
    )
    try:
        return bus_publish(envelope)
    except Exception as e:
        logger.warning("llm_gateway_bus_emit_failed type=%s err=%s", event_type, e)
        return None


def emit_llm_call_started(
    request_id: str, provider: str, model: str, **extra: Any
) -> str | None:
    return emit_event(
        event_type="llm:call:started",
        payload={"request_id": request_id, "provider": provider, "model": model, **extra},
    )


def emit_llm_call_completed(
    request_id: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    success: bool = True,
    error: str | None = None,
    **extra: Any,
) -> str | None:
    return emit_event(
        event_type="llm:call:completed",
        payload={
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
            **extra,
        },
    )


def emit_provider_rate_limited(provider: str, retry_after_ms: int, **extra: Any) -> str | None:
    return emit_event(
        event_type="llm:provider:rate_limited",
        payload={"provider": provider, "retry_after_ms": retry_after_ms, **extra},
    )
