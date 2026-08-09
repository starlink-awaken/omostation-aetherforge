"""AetherForge Bridge — Unified LLM access for all workspace consumers.

This is the SINGLE entry point for any Python module in the workspace
that needs LLM inference, embedding, or model discovery.

Usage::

    from aetherforge.bridge import llm_generate, llm_embed, llm_list_models

    # Generate text (auto-routes to best available model)
    response = llm_generate("What is 1+1?")
    print(response["content"])

    # Generate with specific model
    response = llm_generate("Write a haiku", model="mythos-fast")

    # Embed texts
    vectors = llm_embed(["hello", "world"])

    # List available models
    models = llm_list_models()

Design:
  - In-process Python call (zero network overhead)
  - Unified routing via ModelGateway (direct port for local, API for cloud)
  - Unified credentials via CredentialsManager (synced from cc-switch)
  - Sensitive flow detection (K1 hard gate)
  - Thinking strip (Qwen3 thinking tags removed)

Replaces:
  - Direct openai.OpenAI() / anthropic.Anthropic() calls
  - Hardcoded http://127.0.0.1:9290 URLs (dead LiteLLM proxy)
  - Environment variable API key management
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _ensure_gateway_paths() -> None:
    """Inject sys.path for aetherforge gateway/swarm packages."""
    _workspace = Path(os.environ.get(
        "WORKSPACE_ROOT",
        str(Path(__file__).resolve().parents[3]),  # src/aetherforge/ → projects/aetherforge/ → projects/ → workspace
    ))
    candidates = [
        str(_workspace / "projects" / "aetherforge" / "packages" / "gateway" / "src"),
        str(_workspace / "projects" / "aetherforge" / "packages" / "swarm" / "src"),
        str(_workspace / "projects" / "aetherforge" / "src"),
        str(_workspace / "projects" / "aetherforge"),
    ]
    for p in candidates:
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)


_ensure_gateway_paths()


def _get_gateway():
    """Get or create ModelGateway singleton."""
    from llm_gateway import get_gateway
    return get_gateway()


def llm_generate(
    prompt: str,
    *,
    model: str = "",
    system: str = "",
    timeout: float = 60.0,
    title: str = "",
    url: str = "",
) -> dict[str, Any]:
    """Generate text via AetherForge ModelGateway.

    Args:
        prompt: User prompt text.
        model: Preferred model name (e.g., "coding", "mythos-fast"). Empty = auto-select.
        system: Optional system prompt.
        timeout: Request timeout in seconds.
        title: Content title for K1 sensitive flow detection.
        url: Content URL for K1 sensitive flow detection.

    Returns:
        Dict with: content, model, provider, latency_ms, tokens_in, tokens_out, error.
    """
    from llm_gateway import GatewayRequest
    from llm_gateway.gateway import run_async

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    gw = _get_gateway()
    resp = run_async(gw.generate(GatewayRequest(
        messages=messages,
        model=model,
        timeout=timeout,
        content_title=title,
        content_url=url,
    )))

    return {
        "content": resp.content,
        "model": resp.model,
        "provider": resp.provider,
        "latency_ms": resp.latency_ms,
        "tokens_in": resp.tokens_in,
        "tokens_out": resp.tokens_out,
        "error": resp.error,
    }


def llm_embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via AetherForge ModelGateway.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (one per input text).
    """
    gw = _get_gateway()
    from llm_gateway.gateway import run_async
    return run_async(gw.embed(texts))


def llm_list_models() -> list[dict[str, Any]]:
    """List all discovered models in the registry.

    Returns:
        List of dicts with: id, provider, name.
    """
    import asyncio
    gw = _get_gateway()
    if not gw._registry.list_models():
        asyncio.run(gw._registry.refresh())
    return [
        {"id": m.id, "provider": m.provider, "name": m.name}
        for m in gw._registry.list_models()
    ]


def llm_health() -> dict[str, Any]:
    """Check health of all model providers.

    Returns:
        Dict mapping provider name to health status.
    """
    gw = _get_gateway()
    from llm_gateway.gateway import run_async
    return run_async(gw.health())
