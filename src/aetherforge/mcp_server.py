"""AetherForge 统一 MCP Server — 整合 gateway + mesh + triage 的所有 MCP tools。

启动方式:
    aetherforge-mcp

暴露工具:
    forge_generate        → gateway LLM 生成
    forge_list_nodes      → mesh 节点列表
    forge_mesh_status     → mesh 健康状态
    forge_health_check    → mesh 批量健康检查
    forge_cost_report     → mesh 成本报告
    forge_triage          → 单条分诊 (丢弃/沉淀/提醒)
    forge_triage_consensus → 共识分诊 (两级: stage1并行 + stage2复核)
    forge_triage_batch    → 批量分诊
    forge_triage_status   → 分诊系统状态
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from aetherforge.gateway import llm_generate
from aetherforge.mesh import (
    mesh_cost_report,
    mesh_generate,
    mesh_health_check,
    mesh_list_nodes,
    mesh_status,
)

mcp = FastMCP("aetherforge")


# ── Gateway tools ──────────────────────────────────────────────────────────
mcp.tool(name="forge_generate")(llm_generate)

# ── Mesh tools ─────────────────────────────────────────────────────────────
mcp.tool(name="forge_list_nodes")(mesh_list_nodes)
mcp.tool(name="forge_mesh_status")(mesh_status)
mcp.tool(name="forge_health_check")(mesh_health_check)
mcp.tool(name="forge_cost_report")(mesh_cost_report)
mcp.tool(name="forge_generate_mesh")(mesh_generate)


# ── Triage tools ───────────────────────────────────────────────────────────


class _DirectHTTPGateway:
    """轻量级网关包装 — 直接 HTTP 调用, 不依赖 ModelGateway."""

    def __init__(self, url: str = "http://127.0.0.1:9000/coding/v1/chat/completions", key: str = "sk-omlx-admin"):
        self.url = url
        self.key = key

    async def generate(self, request):
        """模拟 ModelGateway.generate 接口."""
        import json
        import time
        import urllib.request

        payload = {
            "model": request.model or "triage",
            "messages": request.messages,
            "max_tokens": 50,
            "temperature": 0,
        }
        if hasattr(request, "extra_body") and request.extra_body:
            payload["extra_body"] = request.extra_body
        else:
            payload["extra_body"] = {"reasoning_effort": "none"}

        data = json.dumps(payload).encode()
        req = urllib.request.Request(  # noqa: S310  (internal gateway call)
            self.url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310  (internal gateway call)
                d = json.loads(resp.read())
            latency = (time.time() - t0) * 1000
            content = d["choices"][0]["message"]["content"].strip()
            usage = d.get("usage", {})

            # 模拟 GatewayResponse
            class _Response:  # type: ignore[reportRedeclaration]
                pass

            resp = _Response()
            resp.content = content  # type: ignore[reportAttributeAccessIssue]
            resp.model = d.get("model", request.model)  # type: ignore[reportAttributeAccessIssue]
            resp.latency_ms = latency  # type: ignore[reportAttributeAccessIssue]
            resp.tokens_in = usage.get("prompt_tokens", 0)  # type: ignore[reportAttributeAccessIssue]
            resp.tokens_out = usage.get("completion_tokens", 0)  # type: ignore[reportAttributeAccessIssue]
            resp.cost_usd = 0.0  # type: ignore[reportAttributeAccessIssue]
            resp.error = None  # type: ignore[reportAttributeAccessIssue]
            return resp
        except Exception as e:

            class _Response:
                pass

            resp = _Response()
            resp.content = ""  # type: ignore[reportAttributeAccessIssue]
            resp.model = request.model  # type: ignore[reportAttributeAccessIssue]
            resp.latency_ms = (time.time() - t0) * 1000  # type: ignore[reportAttributeAccessIssue]
            resp.tokens_in = 0  # type: ignore[reportAttributeAccessIssue]
            resp.tokens_out = 0  # type: ignore[reportAttributeAccessIssue]
            resp.cost_usd = 0.0  # type: ignore[reportAttributeAccessIssue]
            resp.error = str(e)[:50]  # type: ignore[reportAttributeAccessIssue]
            return resp


def _get_triage_router():
    """获取分诊路由器 (懒加载, 直接 HTTP 调用网关)."""
    if not hasattr(_get_triage_router, "_instance"):
        from aetherforge.triage.router import TriageRouter

        # 使用轻量级 HTTP 网关
        gateway = _DirectHTTPGateway()
        _get_triage_router._instance = TriageRouter(gateway=gateway)  # type: ignore[reportArgumentType]
    return _get_triage_router._instance  # type: ignore[reportFunctionMemberAccess]


def forge_triage(text: str) -> dict:
    """单条分诊 — 判断信息应 丢弃/沉淀/提醒.

    Args:
        text: 待分诊的信息内容

    Returns:
        包含 verdict(丢弃/沉淀/提醒), model, latency 的字典
    """
    router = _get_triage_router()
    result = router.triage_one(text)
    return {
        "verdict": result.verdict,
        "model": result.model,
        "latency": round(result.latency, 3),
        "error": result.error,
    }


def forge_triage_consensus(text: str) -> dict:
    """共识分诊 — 两级投票 (stage1 双模型并行 + stage2 分歧复核).

    Args:
        text: 待分诊的信息内容

    Returns:
        包含 verdict, votes, agreement, status, latency 的字典
    """
    router = _get_triage_router()
    result = router.consensus_triage(text)
    return {
        "verdict": result.verdict,
        "votes": result.votes,
        "agreement": result.agreement,
        "status": result.status,
        "latency": round(result.latency, 3),
        "cost_usd": result.cost_usd,
    }


def forge_triage_batch(texts: list[str], consensus: bool = False) -> dict:
    """批量分诊 — 对多条信息进行分类.

    Args:
        texts: 待分诊的信息列表
        consensus: 是否使用共识模式

    Returns:
        包含 results 列表和 summary 的字典
    """
    router = _get_triage_router()
    results = []
    for text in texts:
        if consensus:
            r = router.consensus_triage(text)
            results.append(
                {
                    "text": text[:50],
                    "verdict": r.verdict,
                    "status": r.status,
                    "latency": round(r.latency, 3),
                }
            )
        else:
            r = router.triage_one(text)
            results.append(
                {
                    "text": text[:50],
                    "verdict": r.verdict,
                    "latency": round(r.latency, 3),
                }
            )

    # 统计
    verdicts = {}
    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    return {
        "total": len(results),
        "verdicts": verdicts,
        "results": results,
    }


def forge_triage_status() -> dict:
    """分诊系统状态 — 网关健康 + 模型可用性.

    Returns:
        包含 gateway, models, tracker 状态的字典
    """
    import urllib.request

    gateway = "http://127.0.0.1:9000"
    status = {"gateway": "unknown", "models": []}

    # 检查网关
    try:
        req = urllib.request.Request(f"{gateway}/v1/models", headers={"Authorization": "Bearer sk-omlx-admin"})  # noqa: S310  (internal gateway call)
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310  (internal gateway call)
            import json

            data = json.loads(resp.read())
            status["gateway"] = "ok"
            status["models"] = [m["id"] for m in data.get("data", [])]
    except Exception as e:
        status["gateway"] = f"error: {str(e)[:50]}"

    return status


def forge_fabric_inspect() -> dict:
    """采集异构节点温控、模型架构与两级缓存状态 (omlxc 数据面)."""
    import json
    import subprocess

    from aetherforge._paths import _resolve_workspace_root

    ws_root = _resolve_workspace_root()
    omlxc_root = ws_root / "projects" / "omlxc"
    try:
        r = subprocess.run(
            ["uv", "run", "omlxc", "fabric", "inspect", "--json"],
            cwd=str(omlxc_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return json.loads(r.stdout)
        return {"status": "error", "error": r.stderr}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def forge_fabric_warm(model_id: str = "coding") -> dict:
    """预热常用系统 Prompt 前缀以实现 0ms TTFT (omlxc 前缀预热引擎)."""
    import json
    import subprocess

    from aetherforge._paths import _resolve_workspace_root

    ws_root = _resolve_workspace_root()
    omlxc_root = ws_root / "projects" / "omlxc"
    try:
        r = subprocess.run(
            ["uv", "run", "omlxc", "fabric", "warm", "--model", model_id, "--json"],
            cwd=str(omlxc_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return json.loads(r.stdout)
        return {"status": "error", "error": r.stderr}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def forge_fabric_vram(model_id: str = "coding", context_tokens: int = 32768) -> dict:
    """计算模型动态 KV Cache 显存预算与准入/压缩建议 (omlxc 显存估算器)."""
    import json
    import subprocess

    from aetherforge._paths import _resolve_workspace_root

    ws_root = _resolve_workspace_root()
    omlxc_root = ws_root / "projects" / "omlxc"
    try:
        r = subprocess.run(
            ["uv", "run", "omlxc", "fabric", "vram", model_id, str(context_tokens), "--json"],
            cwd=str(omlxc_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return json.loads(r.stdout)
        return {"status": "error", "error": r.stderr}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


mcp.tool(name="forge_triage")(forge_triage)
mcp.tool(name="forge_triage_consensus")(forge_triage_consensus)
mcp.tool(name="forge_triage_batch")(forge_triage_batch)
mcp.tool(name="forge_triage_status")(forge_triage_status)
mcp.tool(name="forge_fabric_inspect")(forge_fabric_inspect)
mcp.tool(name="forge_fabric_warm")(forge_fabric_warm)
mcp.tool(name="forge_fabric_vram")(forge_fabric_vram)


def main() -> None:
    transport = os.getenv("AETHERFORGE_MCP_TRANSPORT", "stdio").strip().lower()
    port = int(os.getenv("AETHERFORGE_MCP_PORT", "0"))

    if transport == "stdio" or port <= 0:
        mcp.run(transport="stdio")
        return

    mcp.run(
        transport=transport,  # type: ignore[reportArgumentType]
        host=os.getenv("AETHERFORGE_MCP_HOST", "0.0.0.0"),  # noqa: S104  (MCP server binds all interfaces by design)
        port=port,
    )


if __name__ == "__main__":
    main()
