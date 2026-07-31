"""AetherForge Gateway — BOS RPC 入口 (internal 同进程直调).

暴露: bos://capability/compute/generate
经 ModelGateway 统一入口生成文本 (自动 load/unload + fallback + K1 硬拦).

架构变更 (2026-07-31):
- 旧: 直接调 llm_gateway.mcp_server.llm_generate (SSOT registry)
- 新: 通过 ModelGateway.generate() (统一入口, 含 MemoryGuard + WarmPool)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# 补齐 aetherforge / gateway 包路径
_af_dir = Path(__file__).resolve().parents[3]
_src_p = str(_af_dir / "src")
if _src_p not in sys.path:
    sys.path.insert(0, _src_p)
for _sub in ("gateway", "mesh"):
    _p = str(_af_dir / "packages" / _sub / "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_gateway.gateway import GatewayRequest, get_gateway, run_async

_log = logging.getLogger(__name__)


def run_generate(prompt: str = "", model: str = "coding-fast", **kwargs: Any) -> dict[str, Any]:
    """BOS RPC 入口: bos://capability/compute/generate.

    Args:
        prompt: 用户提示 (或用 kwargs['messages'] 传完整消息列表)
        model: 网关模型名 (裸名如 coding-fast / mid-local; 空则用 gateway fallback_chain)
    """
    messages = kwargs.get("messages") or ([{"role": "user", "content": prompt}] if prompt else [])
    if not messages:
        return {"status": "failed", "error": "prompt or messages is required"}

    # K1 敏感检查上下文
    content_title = kwargs.get("title", "")
    content_url = kwargs.get("url", "")

    try:
        gateway = get_gateway()
        req = GatewayRequest(
            messages=messages,
            model=model,
            task="rpc",
            content_title=content_title,
            content_url=content_url,
        )
        resp = run_async(gateway.generate(req))

        if resp.error:
            return {"status": "failed", "error": resp.error}
        return {
            "status": "success",
            "content": resp.content,
            "model": resp.model or model,
            "provider": resp.provider,
            "latency_ms": round(resp.latency_ms, 1),
            "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out,
            "stripped_thinking": resp.stripped_thinking,
        }
    except PermissionError as e:
        return {"status": "denied", "error": f"[K1] {e}"}
    except Exception as e:
        _log.warning("[Compute RPC] generate failed: %s", e)
        return {"status": "failed", "error": str(e)}


def health_check() -> dict[str, Any]:
    """RPC 健康检查入口."""
    try:
        gateway = get_gateway()
        return run_async(gateway.health())
    except Exception as e:
        return {"status": "error", "error": str(e)}
