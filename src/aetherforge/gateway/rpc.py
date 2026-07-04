"""AetherForge Gateway — BOS RPC 入口 (internal 同进程直调)。

暴露: bos://capability/compute/generate
经统一网关 (SSOT registry → omlx / 本地 / 云) 生成文本。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any

# 补齐 internal 同进程调用时子包的 sys.path
_af_dir = Path(__file__).resolve().parents[3]
for _sub in ("gateway", "mesh"):
    _p = str(_af_dir / "packages" / _sub / "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_gateway.mcp_server import GenerateRequest, llm_generate

_log = logging.getLogger(__name__)


def _sync_call(model: str, messages: list[dict[str, Any]]) -> str:
    """在独立线程里跑 async llm_generate, 无论调用方是否已在事件循环中都安全。"""

    def _work() -> str:
        return asyncio.run(llm_generate(GenerateRequest(model=model, messages=messages)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()


def run_generate(prompt: str = "", model: str = "coder", **kwargs: Any) -> dict[str, Any]:
    """BOS RPC 入口: bos://capability/compute/generate.

    Args:
        prompt: 用户提示 (或用 kwargs['messages'] 传完整消息列表)
        model: 网关模型名 (裸名如 coder / mini-9b, 或 ENG-*/model 全名; 裸名优先本地 omlx)
    """
    messages = kwargs.get("messages") or ([{"role": "user", "content": prompt}] if prompt else [])
    if not messages:
        return {"status": "failed", "error": "prompt or messages is required"}
    try:
        data = json.loads(_sync_call(model, messages))
        if data.get("error"):
            return {"status": "failed", "error": data["error"]}
        return {
            "status": "success",
            "content": data.get("content", ""),
            "model": data.get("model", model),
        }
    except Exception as e:
        _log.warning("[Compute RPC] generate failed: %s", e)
        return {"status": "failed", "error": str(e)}
