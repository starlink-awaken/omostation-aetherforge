import json
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

from .paths import M1_COMPUTE_ENGINE_DIR as M1_ENGINE_DIR
from .paths import M1_MODEL_DIR
from .registry import ModelRegistry
from .scheduler import ModelScheduler
from .ssot_loader import load_ssot_models
from .types import ChatOptions

# Heavy loading is moved to the server startup hook.

mcp = FastMCP("llm-gateway")


@mcp.prompt()
def get_prompt() -> str:
    return "This is the LLM Gateway MCP Server."


class GenerateRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None


# Module-level SSOT registry (lazy, shared across tool calls)
_REGISTRY: ModelRegistry | None = None
_REGISTRY_REFRESHED = False


def _get_registry() -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        reg = ModelRegistry()
        if M1_ENGINE_DIR.exists():
            load_ssot_models(reg, str(M1_ENGINE_DIR), str(M1_MODEL_DIR) if M1_MODEL_DIR.exists() else None)
        _REGISTRY = reg
    return _REGISTRY


def _resolve_model_id(reg: ModelRegistry, model: str) -> str | None:
    """Resolve a user-facing model name to a registry model id.

    Accepts exact ids (``ENG-OMLX-LOCAL/coder``), bare names (``coder``),
    or fuzzy substrings. Prefers the local omlx engine for bare names.
    """
    if reg.get(model):
        return model
    for engine in ("ENG-OMLX-LOCAL", "ENG-CC-SWITCH"):
        if reg.get(f"{engine}/{model}"):
            return f"{engine}/{model}"
    matches = [m for m in reg.list_models() if model.lower() in m.id.lower()]
    return matches[0].id if matches else None


@mcp.tool()
async def llm_generate(req: GenerateRequest) -> str:
    """Generate an LLM response via the unified SSOT gateway (routes to omlx / local / cloud engines)."""
    global _REGISTRY_REFRESHED
    reg = _get_registry()
    if not _REGISTRY_REFRESHED:
        try:
            await reg.refresh()
            _REGISTRY_REFRESHED = True
        except Exception as e:
            return json.dumps({"error": f"registry refresh failed: {e}"})

    model_id = _resolve_model_id(reg, req.model)
    if not model_id:
        return json.dumps({"error": f"Model '{req.model}' not found in SSOT registry."})

    try:
        result = await reg.chat(model_id, req.messages, ChatOptions())
        if not result:
            return json.dumps({"error": "No response from provider."})
        return json.dumps(
            {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [],
                "finish_reason": result.finish_reason or "stop",
                "model": model_id,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    if M1_ENGINE_DIR.exists():
        import asyncio

        from .ssot_loader import load_ssot_models

        _registry = ModelRegistry()
        load_ssot_models(_registry, str(M1_ENGINE_DIR), str(M1_MODEL_DIR) if M1_MODEL_DIR.exists() else None)
        _scheduler = ModelScheduler(_registry)
        try:
            _models = asyncio.run(_registry.refresh())
            _loaded_rates = _scheduler.load_quota_rates()
            print(f"[llm-gateway] Loaded {len(_models)} models, {_loaded_rates} with real prices from quota_rates.json")
        except Exception as e:
            print(f"[llm-gateway] M1 nodes loaded but refresh failed: {e}")
    else:
        print(f"[llm-gateway] M1 engine dir not found: {M1_ENGINE_DIR}")

    mcp.run()


if __name__ == "__main__":
    main()
