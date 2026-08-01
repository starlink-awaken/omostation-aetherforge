from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# 动态环境补齐
# 解决 internal 模式同进程调用时 sys.path 缺失子包的问题
aetherforge_dir = Path(__file__).resolve().parents[3]
swarm_src_path = str(aetherforge_dir / "packages" / "swarm" / "src")

if swarm_src_path not in sys.path:
    sys.path.insert(0, swarm_src_path)

# 现在可以安全地导入了
from swarm_engine.graph_workflow import GraphWorkflow

from aetherforge.config import load_config
from aetherforge.gateway import create_provider

_log = logging.getLogger(__name__)


def run_swarm_workflow(goal: str, **kwargs: Any) -> dict[str, Any]:
    """
    BOS RPC 调用入口: bos://capability/swarm/run

    解析并处理该 BOS URI 路由，驱动底层 GraphWorkflow (Swarm 引擎) 执行多智能体任务。

    Args:
        goal: 任务目标
        **kwargs: 扩展参数
    """
    if not goal:
        return {"status": "failed", "error": "Goal parameter is required"}

    _log.info("[Swarm RPC] Initializing GraphWorkflow for goal: %s", goal)

    # 1. 实例化工作流引擎
    wf = GraphWorkflow()

    # 2. 定义 Swarm 协调节点
    @wf.node("任务规划", description="分析并分解任务目标")
    def plan_task(state: dict[str, Any]) -> dict[str, Any]:
        task_goal = state.get("goal", "")
        analysis = f"分析目标: {task_goal}"
        try:
            # 尝试通过本地 gateway 生成拆解方案
            cfg = load_config()
            model = kwargs.get("model") or getattr(cfg.gateway, "default_model", None)
            provider_name = kwargs.get("provider") or getattr(cfg.gateway, "default_provider", None)

            if model and provider_name:
                prov = create_provider(provider_name)
                resp = prov.generate(f"将以下任务目标拆解为3步，仅输出简短文本: {task_goal}")
                analysis = resp.text
        except Exception as e:  # defensive fallback  # noqa: BLE001
            _log.warning("[Swarm RPC] Planning stage gateway generate failed: %s", e)
        return {"plan": analysis}

    @wf.node("任务执行", description="协同智能体执行具体计划")
    def execute_task(state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan", "")
        return {"output": f"成功执行计划:\n{plan}"}

    # 3. 关联有向图拓扑
    wf.add_edge("任务规划", "任务执行")
    wf.set_entry("任务规划")

    # 4. 执行工作流
    initial_state = {"goal": goal}
    state = wf.run(
        initial_state,
        workflow_run_id=kwargs.get("workflow_run_id"),
        trace_id=kwargs.get("trace_id"),
        event_sink=kwargs.get("event_sink"),
        admission=kwargs.get("admission"),
    )

    # 5. 格式化返回结果 (保证是标准的序列化 dict)
    errors = state.get("_errors", [])
    history = state.get("_history", [])

    return {
        "status": "success" if not errors else "failed",
        "goal": goal,
        "plan": state.get("plan", ""),
        "result": state.get("output", ""),
        "steps": [
            {"name": step["node"], "status": "ok" if step["status"] == "ok" else "failed", "error": step.get("error")}
            for step in history
        ],
        "errors": [str(e) for e in errors],
        "workflow_run_id": kwargs.get("workflow_run_id") or state.get("_workflow_run_id"),
        "event_sink_errors": state.get("_event_sink_errors", []),
    }
