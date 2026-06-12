"""aetherforge bus adapter (R60, Month 2).

AetherForge is a compute-mesh + LLM-gateway + swarm-engine fusion stack.
It produces pipeline / agent-runtime / mesh-routing events that need to
flow into the agora I0 bus so other eCOS projects (omo, metaos, runtime)
can subscribe without aetherforge having to know about each consumer.

This adapter is a thin shim: aetherforge's internal event publisher
already produces dict payloads shaped like BusEnvelope; we wrap it
with the official agora.bus facade so:
  * Envelope shape is canonical (id, time, type, source, schema_version, payload)
  * Errors route to agora's DLQ instead of crashing the swarm loop
  * Other eCOS consumers can subscribe via `from agora.bus import subscribe`

We do NOT replace aetherforge's internal event bus (it has its own
back-pressure semantics tuned for swarm loops); we mirror high-value
events outward.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _try_import_bus():
    """Lazy import: agora may not be on the path during isolated aetherforge tests."""
    try:
        from bus_foundation import BusEnvelope, publish  # type: ignore

        return BusEnvelope, publish
    except ImportError:
        return None, None


def emit_event(
    event_type: str,
    source: str,
    payload: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> str | None:
    """Emit a single aetherforge event into the agora bus.

    Returns the event id, or None if agora.bus is not importable (e.g.
    running aetherforge in isolation before agora is installed).
    """
    BusEnvelope, publish = _try_import_bus()
    if BusEnvelope is None or publish is None:
        logger.debug("agora_bus_unavailable_skipping_event type=%s", event_type)
        return None
    envelope = BusEnvelope(
        type=event_type,
        source=source,
        payload=payload or {},
        trace_id=trace_id,
    )
    try:
        return publish(envelope)
    except Exception as e:  # agora bus already DLQs internally, but be defensive
        logger.warning("aetherforge_bus_emit_failed type=%s err=%s", event_type, e)
        return None


# Convenience helpers for the most common aetherforge event types.
# Call sites in aetherforge's swarm_engine / mesh can use these to
# keep type strings consistent and discoverable.
def emit_mesh_route(peer_id: str, hop_count: int, **extra: Any) -> str | None:
    return emit_event(
        event_type="mesh:route",
        source="aetherforge.mesh",
        payload={"peer_id": peer_id, "hop_count": hop_count, **extra},
    )


def emit_swarm_step(swarm_id: str, step: int, **extra: Any) -> str | None:
    return emit_event(
        event_type="swarm:step",
        source="aetherforge.swarm",
        payload={"swarm_id": swarm_id, "step": step, **extra},
    )


def emit_llm_call(model: str, prompt_tokens: int, completion_tokens: int, **extra: Any) -> str | None:
    return emit_event(
        event_type="llm:call",
        source="aetherforge.gateway",
        payload={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **extra,
        },
    )
