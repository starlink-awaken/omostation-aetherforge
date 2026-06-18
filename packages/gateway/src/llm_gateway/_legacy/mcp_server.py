import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

from .budget import check_budget_limit, get_remaining_budget, BudgetExhausted
from .detection import detect_backends
from .provider import LLMRequest, ToolSchema
from .registry import ModelRegistry
from .registry_data_loader import build_static_registry
from .scheduler import ModelScheduler

# Load L0 M1 compute_engine nodes
M1_ENGINE_DIR = Path.home() / "Workspace" / "projects" / "ecos" / "src" / "ecos" / "ssot" / "mof" / "m1" / "compute_engine"
_registry = ModelRegistry()
_scheduler: ModelScheduler | None = None

if M1_ENGINE_DIR.exists():
    import asyncio
    from .ssot_loader import load_ssot_models
    load_ssot_models(_registry, str(M1_ENGINE_DIR))
    _scheduler = ModelScheduler(_registry)
    try:
        _count = asyncio.run(_registry.refresh())
        _loaded_rates = _scheduler.load_quota_rates()
        _remaining = get_remaining_budget()
        _budget_info = f" (Remaining Budget: ${_remaining:.4f})" if _remaining is not None else ""
        print(f"[llm-gateway] Loaded {_count} models, {_loaded_rates} with real prices. {_budget_info}")
    except Exception as e:
        print(f"[llm-gateway] M1 nodes loaded but refresh failed: {e}")
else:
    _registry, _count = build_static_registry()
    _scheduler = ModelScheduler(_registry)
    print(f"[llm-gateway] M1 engine dir not found: {M1_ENGINE_DIR}; loaded {_count} static registry_data models instead")

mcp = FastMCP("llm-gateway")


class GenerateRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    budget_usd: float | None = None


@mcp.tool()
async def llm_generate(req: GenerateRequest) -> str:
    """Generate LLM response using the unified gateway with full schema support."""
    if _scheduler is not None and _scheduler._refresh_task is None:
        _scheduler.start_auto_refresh()

    providers = detect_backends()
    if not providers:
        return json.dumps({"error": "No LLM backend available."})

    provider = providers[0]
    model_id = req.model or provider.default_model

    # ── Phase 4: Pre-call Budget Check ──
    try:
        # Estimate input tokens
        prompt_text = "\n".join(str(m.get("content", "")) for m in req.messages)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(enc.encode(prompt_text))
        except ImportError:
            input_tokens = (len(prompt_text) + 3) // 4

        # Check against global and local budget
        check_budget_limit(
            model_id=model_id,
            input_tokens=input_tokens,
            max_output_tokens=512,  # Default safety margin
            local_budget_limit=req.budget_usd
        )
    except BudgetExhausted as e:
        return json.dumps({"error": f"BUDGET_EXHAUSTED: {str(e)}", "status": "blocked"})
    except Exception as e:
        # Budget check failure shouldn't necessarily block if it's a technical error,
        # but we log it.
        print(f"[llm-gateway] Budget check failed: {e}")

    # Convert tools
    mapped_tools = None
    if req.tools:
        mapped_tools = []
        for t in req.tools:
            if "function" in t:
                f = t["function"]
                mapped_tools.append(
                    ToolSchema(name=f["name"], description=f.get("description", ""), parameters=f.get("parameters", {}))
                )

    llm_req = LLMRequest(model=req.model or provider.default_model, messages=req.messages, tools=mapped_tools)

    try:
        resp = await provider.generate(llm_req)

        # Format response back to OpenAI style
        result = {"role": "assistant", "content": resp.content, "tool_calls": [], "finish_reason": "stop"}

        if resp.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in resp.tool_calls
            ]
            result["finish_reason"] = "tool_calls"

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    mcp.run()


if __name__ == "__main__":
    main()
