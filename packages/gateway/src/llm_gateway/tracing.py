"""Tracing utilities for AetherForge using Langfuse SDK.

Provides a soft-dependent trace context manager that handles connection failures
gracefully and bypasses execution if Langfuse credentials are not provided.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

_log = logging.getLogger(__name__)

# Soft import
try:
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

_client = None

def get_langfuse_client() -> Langfuse | None:
    """Lazy initialize and return a Langfuse client if configured."""
    global _client
    if _client is not None:
        return _client

    if not _LANGFUSE_AVAILABLE:
        return None

    # Load credentials (supports both public/secret pair and unified single API key format)
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3050")
    api_key = os.environ.get("LANGFUSE_API_KEY")

    if not (public_key and secret_key) and not api_key:
        # Silently bypass if no credentials are configured
        return None

    try:
        if api_key:
            _client = Langfuse(api_key=api_key, host=host)
        else:
            _client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        return _client
    except Exception as e:  # noqa: BLE001
        _log.debug("Failed to initialize Langfuse client: %s", e)
        return None


@contextmanager
def trace_llm_call(model_id: str, messages: list[dict[str, Any]], options: Any = None):
    """Context manager for tracing LLM execution via Langfuse (soft dependency)."""
    client = get_langfuse_client()
    if not client:
        yield None
        return

    try:
        # Create trace
        strategy = getattr(options, "strategy", "balanced") if options else "balanced"
        trace = client.trace(
            name="aetherforge-gateway-call",
            metadata={
                "model_id": model_id,
                "strategy": strategy,
            }
        )

        # Standard formatted input for Langfuse dashboard
        prompt_formatted = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

        generation = trace.generation(
            name="chat-completion",
            model=model_id.split("/")[-1],
            input=prompt_formatted,
            metadata={"messages": messages}
        )
        yield generation
    except Exception as e:  # noqa: BLE001
        _log.debug("Langfuse tracing setup failed: %s", e)
        yield None
