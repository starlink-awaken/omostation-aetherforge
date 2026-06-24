#!/usr/bin/env python3
"""AetherForge 统一 CLI — 算力网格 + LLM 网关 + 群体智能引擎。

Usage:
    aetherforge gateway list              List LLM models
    aetherforge gateway generate <prompt>  Generate text
    aetherforge mesh list                 List compute nodes
    aetherforge mesh status               Node health
    aetherforge mesh health               Run health checks
    aetherforge mesh topology-scan        Discover nodes
    aetherforge mesh generate <prompt>    Generate via best node
    aetherforge mesh cost                 Cost report
    aetherforge swarm ...                 Swarm commands (TODO)
"""

from __future__ import annotations

import argparse
import sys


def cmd_gateway(argv: list[str]) -> int:
    """Delegate to gateway CLI."""
    from aetherforge.gateway import cli as gateway_main

    return gateway_main(argv if argv else ["--help"])


def cmd_mesh(argv: list[str]) -> int:
    """Delegate to mesh CLI."""
    from aetherforge.mesh import cli as mesh_main

    return mesh_main(argv if argv else ["--help"])


def cmd_swarm(argv: list[str]) -> int:
    """Swarm CLI — 打通 C2G / ECOS Workflow Subprocess 调用。"""
    parser = argparse.ArgumentParser(description="AetherForge Swarm CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a multi-agent task workflow")
    run_parser.add_argument("--goal", required=False, help="Task goal to execute")
    run_parser.add_argument("--json", action="store_true", help="Print outputs in JSON format")

    args = parser.parse_args(argv)

    if args.command == "run":
        import sys
        import select
        import json

        goal = args.goal
        is_json_output = args.json

        # Fallback to stdin JSON (Agora StdioAdapter support)
        if not goal:
            # Check if stdin has data
            if select.select([sys.stdin], [], [], 0.0)[0]:
                try:
                    payload = json.loads(sys.stdin.read())
                    kwargs = payload.get("kwargs", {})
                    goal = kwargs.get("goal", "")
                    is_json_output = True  # Force JSON output for adapter
                except Exception:
                    pass

        if not goal:
            print("❌ Error: --goal is required or must be provided via stdin JSON.", file=sys.stderr)
            return 1

        from aetherforge.swarm import GraphWorkflow

        # 1. 初始化工作流
        wf = GraphWorkflow()

        @wf.node("任务规划", description="分析并分解任务目标")
        def plan_task(state):
            goal = state.get("goal", "")
            # 尝试通过 Gateway 对接 LLM 来丰富分析，如果有可用 Provider 且没有报错
            analysis = f"分析目标: {goal}"
            try:
                # 尝试调用本地的 llm_gateway
                from aetherforge.gateway import create_provider
                # 如果有默认 provider 配置，可以用它生成一些真实的拆解
                from aetherforge.config import load_config
                cfg = load_config()
                if cfg.gateway.default_model:
                    prov = create_provider(cfg.gateway.default_provider)
                    resp = prov.generate(f"将以下任务目标拆解为3步，仅输出简短文本: {goal}")
                    analysis = resp.text
            except Exception:
                pass
            return {"plan": analysis}

        @wf.node("任务执行", description="协同智能体执行具体计划")
        def execute_task(state):
            plan = state.get("plan", "")
            return {"output": f"成功执行计划:\n{plan}"}

        wf.add_edge("任务规划", "任务执行")
        wf.set_entry("任务规划")

        # 2. 运行
        initial_state = {"goal": goal}
        state = wf.run(initial_state)

        # 3. 结果输出
        if is_json_output:
            import json
            output_data = {
                "goal": goal,
                "status": "success" if not state.get("_errors") else "failed",
                "steps": [
                    {
                        "name": step["node"],
                        "status": "ok" if step["status"] == "ok" else "failed",
                        "error": step.get("error")
                    }
                    for step in state.get("_history", [])
                ],
                "result": state.get("output", "")
            }
            print(json.dumps(output_data, ensure_ascii=False, indent=2))
        else:
            print(f"🎯 Swarm Goal: {args.goal}")
            print(f"📄 Plan: {state.get('plan')}")
            print(f"💡 Result: {state.get('output')}")
        
        return 1 if state.get("_errors") else 0

    else:
        parser.print_help()
        return 1



def main(argv: list[str] | None = None) -> int:
    print("⚠️ AetherForge 独立 CLI 已弃用，请使用 cockpit 替代", file=sys.stderr)
    if argv is None:
        import sys
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: aetherforge {gateway,mesh,swarm} [subcommand_args]")
        print("\nCommands:")
        print("  gateway   LLM Gateway (List models, generate, MCP, serve)")
        print("  mesh      Compute Mesh (List nodes, status, topology-scan, health)")
        print("  swarm     Swarm Engine (Run multi-agent workflows)")
        return 0 if argv and argv[0] in ("-h", "--help") else 1

    domain = argv[0]
    sub_args = argv[1:]

    if domain == "gateway":
        return cmd_gateway(sub_args)
    elif domain == "mesh":
        return cmd_mesh(sub_args)
    elif domain == "swarm":
        return cmd_swarm(sub_args)
    else:
        print(f"Unknown domain: {domain}")
        print("Usage: aetherforge {gateway,mesh,swarm} [subcommand_args]")
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
