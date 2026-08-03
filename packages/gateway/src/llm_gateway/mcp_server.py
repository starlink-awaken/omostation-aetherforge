import json
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

from .gateway import GatewayRequest, get_gateway
from .paths import M1_COMPUTE_ENGINE_DIR as M1_ENGINE_DIR
from .paths import M1_MODEL_DIR
from .registry import ModelRegistry
from .scheduler import ModelScheduler
from .ssot_loader import load_ssot_models

# Heavy loading is moved to the server startup hook.

mcp = FastMCP("llm-gateway")


@mcp.prompt()
def get_prompt() -> str:
    return "This is the LLM Gateway MCP Server."


class GenerateRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None


class GatewayGenerateRequest(BaseModel):
    """ModelGateway 统一请求格式."""

    model: str = ""
    messages: list[dict[str, Any]]
    task: str = "mcp"
    timeout: float = 30.0
    content_title: str = ""
    content_url: str = ""


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
    """Generate an LLM response via the unified SSOT gateway (routes to omlx / local / cloud engines).

    内部已切换到 ModelGateway (统一入口, 含 MemoryGuard + WarmPool + K1 硬拦).
    """
    try:
        gateway = get_gateway()
        gw_req = GatewayRequest(
            messages=req.messages,
            model=req.model,
            task="mcp",
        )
        resp = await gateway.generate(gw_req)

        if resp.error:
            return json.dumps({"error": resp.error})
        return json.dumps(
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [],
                "finish_reason": "stop",
                "model": resp.model or req.model,
                "stripped_thinking": resp.stripped_thinking,
            },
            ensure_ascii=False,
        )
    except PermissionError as e:
        return json.dumps({"error": f"[K1] {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateway_generate(req: GatewayGenerateRequest) -> str:
    """Generate via ModelGateway (统一入口, 支持 K1 敏感检查 + content filtering).

    与 llm_generate 区别:
      - 支持 content_title / content_url → 触发 K1 敏感硬拦
      - 支持 task 标签 (triage/rpc/mcp) 用于指标分类
    """
    try:
        gateway = get_gateway()
        gw_req = GatewayRequest(
            messages=req.messages,
            model=req.model,
            task=req.task,
            timeout=req.timeout,
            content_title=req.content_title,
            content_url=req.content_url,
        )
        resp = await gateway.generate(gw_req)

        return json.dumps(
            {
                "role": "assistant",
                "content": resp.content,
                "model": resp.model or req.model,
                "provider": resp.provider,
                "latency_ms": resp.latency_ms,
                "tokens_in": resp.tokens_in,
                "tokens_out": resp.tokens_out,
                "stripped_thinking": resp.stripped_thinking,
                "error": resp.error or None,
            },
            ensure_ascii=False,
        )
    except PermissionError as e:
        return json.dumps({"error": f"[K1] {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateway_health() -> str:
    """Health check all configured models."""
    try:
        gateway = get_gateway()
        health = await gateway.health()
        return json.dumps(health, ensure_ascii=False)
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
