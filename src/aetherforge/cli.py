#!/usr/bin/env python3
"""AetherForge 统一 CLI — 算力网格 + LLM 网关 + 群体智能引擎 + 分诊。

Usage:
    aetherforge gateway list              List LLM models
    aetherforge gateway generate <prompt>  Generate text
    aetherforge mesh list                 List compute nodes
    aetherforge mesh status               Node health
    aetherforge mesh health               Run health checks
    aetherforge mesh topology-scan        Discover nodes
    aetherforge mesh generate <prompt>    Generate via best node
    aetherforge mesh cost                 Cost report
    aetherforge triage <text>             单条分诊
    aetherforge triage --consensus <text> 共识分诊 (两级)
    aetherforge triage --batch <file>     批量分诊
    aetherforge triage --benchmark        标准 benchmark
    aetherforge triage --monitor          准确率监控检查
    aetherforge triage --server           启动 HTTP 服务
    aetherforge swarm ...                 Swarm commands (TODO)
"""

from __future__ import annotations

import argparse


def cmd_gateway(argv: list[str]) -> int:
    """Delegate to gateway CLI."""
    from aetherforge.gateway import cli as gateway_main

    return gateway_main(argv if argv else ["--help"])


def cmd_infer(argv: list[str]) -> int:
    """BOS one-shot inference bridge → running AetherForge data plane.

    stdin accepts either a plain request object or Agora's {"kwargs": {...}} envelope.
    stdout is always JSON so the subprocess transport can parse it deterministically.
    """
    import json
    import os
    import sys
    import urllib.error
    import urllib.parse
    import urllib.request

    try:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid JSON: {exc}"}, ensure_ascii=False))
        return 2

    if isinstance(payload, dict) and isinstance(payload.get("kwargs"), dict):
        payload = payload["kwargs"]
    if not isinstance(payload, dict):
        print(json.dumps({"error": "request must be a JSON object"}, ensure_ascii=False))
        return 2

    if not payload.get("messages"):
        prompt = payload.pop("prompt", "")
        if not prompt and argv:
            prompt = " ".join(argv)
        if not prompt:
            print(json.dumps({"error": "messages or prompt is required"}, ensure_ascii=False))
            return 2
        payload["messages"] = [{"role": "user", "content": str(prompt)}]

    payload.setdefault("routing_mode", "local")
    payload.setdefault("stream", False)
    base = os.environ.get("AETHERFORGE_BASE_URL", "http://127.0.0.1:9290/v1").rstrip("/")
    parsed_base = urllib.parse.urlparse(base)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        print(json.dumps({"error": "AETHERFORGE_BASE_URL must be an http(s) URL"}, ensure_ascii=False))
        return 2
    req = urllib.request.Request(  # noqa: S310 -- scheme validated above
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    key = os.environ.get("AETHERFORGE_API_KEY")
    if not key:
        # LaunchAgent/Agora 子进程不该把密钥写进 YAML 或命令行；macOS 上从
        # Keychain 取，与 omlxc gw 共用 service 名。
        try:
            import subprocess

            found = subprocess.run(
                ["security", "find-generic-password", "-s", "aetherforge-gateway", "-w"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if found.returncode == 0:
                key = found.stdout.strip()
        except Exception:
            pass
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    timeout = min(900.0, max(1.0, float(payload.get("timeout", 120)))) + 2.0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- validated above
            sys.stdout.write(resp.read().decode("utf-8"))
            return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(body)
        except json.JSONDecodeError:
            error = {"error": {"message": body[:500]}}
        error["status"] = exc.code
        print(json.dumps(error, ensure_ascii=False))
        return 1
    except Exception as exc:
        print(json.dumps({"error": {"message": str(exc)}}, ensure_ascii=False))
        return 1


def cmd_mesh(argv: list[str]) -> int:
    """Delegate to mesh CLI."""
    from aetherforge.mesh import cli as mesh_main

    return mesh_main(argv if argv else ["--help"])


def cmd_swarm(argv: list[str]) -> int:
    """Swarm CLI — fail-closed shim (Y1Q4-T6-01)."""
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser(description="AetherForge Swarm CLI (disabled, Y1Q4-T6-01)")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="DISABLED: swarm_engine removed (Y1Q4-T6-01)")
    # args is unused: this fail-closed shim always exits 1 (F841).
    parser.parse_args(argv)
    msg = "swarm_engine removed (Y1Q4-T6-01); swarm CLI disabled"
    print(json.dumps({"status": "failed", "error": msg}), file=sys.stderr)
    return 1


def cmd_triage(argv: list[str]) -> int:
    """Triage CLI — 信息分诊 (单条/共识/批量/benchmark/监控/服务)."""
    import json as _json
    import sys as _sys

    def _make_router():
        """创建带 HTTP 网关的路由器."""
        from aetherforge.triage.router import TriageRouter

        class _DirectHTTPGateway:
            def __init__(self, url="http://127.0.0.1:9000/v1/chat/completions", key="sk-omlx-admin"):
                self.url = url
                self.key = key

            async def generate(self, request):
                import time
                import urllib.request

                payload = {
                    "model": request.model or "triage",
                    "messages": request.messages,
                    "max_tokens": 50,
                    "temperature": 0,
                    "extra_body": {"reasoning_effort": "none"},
                }
                data = _json.dumps(payload).encode()
                req = urllib.request.Request(  # noqa: S310  (internal gateway call)
                    self.url,
                    data=data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
                )
                t0 = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310  (internal gateway call)
                        d = _json.loads(resp.read())

                    class _R:  # type: ignore[reportRedeclaration]
                        pass

                    r = _R()
                    r.content = d["choices"][0]["message"]["content"].strip()  # type: ignore[reportAttributeAccessIssue]
                    r.model = d.get("model", request.model)  # type: ignore[reportAttributeAccessIssue]
                    r.latency_ms = (time.time() - t0) * 1000  # type: ignore[reportAttributeAccessIssue]
                    r.tokens_in = d.get("usage", {}).get("prompt_tokens", 0)  # type: ignore[reportAttributeAccessIssue]
                    r.tokens_out = d.get("usage", {}).get("completion_tokens", 0)  # type: ignore[reportAttributeAccessIssue]
                    r.cost_usd = 0.0  # type: ignore[reportAttributeAccessIssue]
                    r.error = None  # type: ignore[reportAttributeAccessIssue]
                    return r
                except Exception as e:

                    class _R:
                        pass

                    r = _R()
                    r.content = ""  # type: ignore[reportAttributeAccessIssue]
                    r.model = request.model  # type: ignore[reportAttributeAccessIssue]
                    r.latency_ms = (time.time() - t0) * 1000  # type: ignore[reportAttributeAccessIssue]
                    r.tokens_in = 0  # type: ignore[reportAttributeAccessIssue]
                    r.tokens_out = 0  # type: ignore[reportAttributeAccessIssue]
                    r.cost_usd = 0.0  # type: ignore[reportAttributeAccessIssue]
                    r.error = str(e)[:50]  # type: ignore[reportAttributeAccessIssue]
                    return r

        return TriageRouter(gateway=_DirectHTTPGateway())  # type: ignore[reportArgumentType]

    parser = argparse.ArgumentParser(
        description="AetherForge 分诊 — 信息自动分类 (丢弃/沉淀/提醒)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  aetherforge triage "【日历】明天10点开会"
  aetherforge triage --consensus "【GitHub】新PR merged"
  aetherforge triage --batch samples.txt
  aetherforge triage --benchmark
  aetherforge triage --monitor
  aetherforge triage --server --port 8095
        """,
    )
    parser.add_argument("text", nargs="?", help="单条分诊文本")
    parser.add_argument("--consensus", action="store_true", help="共识模式 (两级: stage1并行 + stage2复核)")
    parser.add_argument("--batch", help="批量分诊文件 (每行一条)")
    parser.add_argument("--benchmark", action="store_true", help="标准 20 条 benchmark")
    parser.add_argument("--monitor", action="store_true", help="准确率监控检查")
    parser.add_argument("--server", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--port", type=int, default=8095, help="HTTP 服务端口")
    parser.add_argument("--log", help="记账日志路径 (JSONL)")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args(argv)

    # --server 模式
    if args.server:
        from aetherforge.triage.server import create_server

        server = create_server(args.port)
        print(f"分诊服务启动: http://0.0.0.0:{args.port}", file=_sys.stderr)
        print("  POST /triage — 单条分诊", file=_sys.stderr)
        print("  POST /triage/consensus — 共识分诊", file=_sys.stderr)
        print("  GET /health — 健康检查", file=_sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务已停止", file=_sys.stderr)
            server.server_close()
        return 0

    # --benchmark 模式
    if args.benchmark:
        import time

        from aetherforge.triage.monitor import BENCHMARK_SAMPLES

        gateway = "http://127.0.0.1:9000/v1/chat/completions"
        key = "sk-omlx-admin"

        prompt_tpl = """你是信息分诊助手。判断: 丢弃 / 沉淀 / 提醒 三选一。
标准: 丢弃=营销/广告/促销  沉淀=技术/知识/笔记同步  提醒=会议/账单/告警/待办
示例: 【淘宝】5折→丢弃  【GitHub】新PR→沉淀  【日历】开会→提醒  【读书笔记】同步→沉淀
信息: {text}
只输出一个词:"""

        def call_model(model, text, needs_off=True):
            import urllib.request

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt_tpl.format(text=text)}],
                "max_tokens": 20,
                "temperature": 0,
            }
            if needs_off:
                payload["extra_body"] = {"reasoning_effort": "none"}
            data = _json.dumps(payload).encode()
            req = urllib.request.Request(
                gateway, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310  (internal gateway call)
                    d = _json.loads(resp.read())
                content = d["choices"][0]["message"]["content"].strip()
                for v in ("丢弃", "沉淀", "提醒"):
                    if v in content:
                        return v, time.time() - t0
                return "未知", time.time() - t0
            except Exception:
                return "错误", time.time() - t0

        print(f"分诊 Benchmark: {len(BENCHMARK_SAMPLES)} 条", file=_sys.stderr)
        models = [("mid-local", True), ("mini-9b", True), ("deepseek-chat", False)]
        for model, needs_off in models:
            correct = 0
            lats = []
            errs = 0
            for text, expected in BENCHMARK_SAMPLES:
                verdict, lat = call_model(model, text, needs_off)
                lats.append(lat)
                if verdict == expected:
                    correct += 1
                elif verdict in ("错误", "未知"):
                    errs += 1
            acc = correct / len(BENCHMARK_SAMPLES)
            avg = sum(lats) / len(lats)
            status = "✅" if acc >= 0.8 and avg < 2.0 and errs == 0 else "❌"
            print(f"  {status} {model}: {correct}/{len(BENCHMARK_SAMPLES)} ({acc:.0%})  avg={avg:.2f}s  err={errs}")
        return 0

    # --monitor 模式
    if args.monitor:
        from aetherforge.triage.monitor import TriageMonitor
        from aetherforge.triage.router import TriageRouter

        router = TriageRouter()
        monitor = TriageMonitor(router)
        result = monitor.run_check()
        alert_str = f" ⚠️ {result.alert_reason}" if result.alert else ""
        print(f"准确率: {result.accuracy:.0%} ({result.correct}/{result.total_samples})")
        print(f"延迟: avg={result.avg_latency:.2f}s  p95={result.p95_latency:.2f}s")
        print(f"错误: {result.errors}")
        print(f"状态: {'🔴 告警' if result.alert else '✅ 正常'}{alert_str}")
        return 0

    # --batch 模式
    if args.batch:
        from pathlib import Path

        from aetherforge.triage.router import TriageRouter

        router = _make_router()
        texts = [line.strip() for line in Path(args.batch).read_text().splitlines() if line.strip()]
        results = []
        for text in texts:
            if args.consensus:
                r = router.consensus_triage(text)
                results.append(
                    {"text": text[:40], "verdict": r.verdict, "status": r.status, "latency": round(r.latency, 3)}
                )
            else:
                r = router.triage_one(text)
                results.append({"text": text[:40], "verdict": r.verdict, "latency": round(r.latency, 3)})
        if args.json:
            print(_json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(f"  [{r['verdict']}] {r.get('status', '单模型')} {r['latency']}s  {r['text']}")
        return 0

    # 单条分诊
    if args.text:
        router = _make_router()
        if args.consensus:
            result = router.consensus_triage(args.text)
            output = {
                "verdict": result.verdict,
                "votes": result.votes,
                "agreement": result.agreement,
                "status": result.status,
                "latency": round(result.latency, 3),
            }
        else:
            result = router.triage_one(args.text)
            output = {
                "verdict": result.verdict,
                "model": result.model,
                "latency": round(result.latency, 3),
                "error": result.error,
            }
        if args.json:
            print(_json.dumps(output, ensure_ascii=False, indent=2))
        else:
            if args.consensus:
                print(f"[{result.verdict}] {result.status} {result.latency:.2f}s")  # type: ignore[reportAttributeAccessIssue]
                print(f"  投票: {result.votes}")  # type: ignore[reportAttributeAccessIssue]
            else:
                print(f"[{result.verdict}] {result.latency:.2f}s ({result.model})")  # type: ignore[reportAttributeAccessIssue]
        return 0

    parser.print_help()
    return 1


def cmd_route(argv: list[str]) -> int:
    """RouteScheduler 演示 — 三级路由 (模型→Provider→节点). TASK-02788FE2.

    真实 registry 接 gateway/mesh 留后续; 此处用 demo 数据演示 select 4 步.
    设计: ARCHITECTURE-v2 line 88-134.
    """
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: aetherforge route <select|policies> [args]")
        print("  select [model]   演示三级路由 (demo registry)")
        print("  policies         列 routing_policy yaml")
        return 0

    from pathlib import Path

    from aetherforge.route import (
        Model,
        Node,
        Provider,
        RouteRequest,
        RouteScheduler,
        RoutingPolicy,
    )

    class _DemoModels:
        def get_models(self, _mid: str) -> list[Model]:
            return [Model(id="gpt-4o", providers=("deepseek", "openai"), cost_per_1k_input=0.001, speed_tps=50.0)]

    class _DemoProviders:
        def get_providers(self) -> list[Provider]:
            return [Provider(id="deepseek", quota_pct=100.0), Provider(id="openai", quota_pct=84.0)]

    class _DemoNodes:
        def get_nodes(self, provider: str) -> list[Node]:
            return [Node(id=f"{provider}-cloud", provider=provider)]

    cmd = argv[0]
    if cmd == "policies":
        pdir = Path(__file__).resolve().parent / "route" / "policies"
        for p in sorted(pdir.glob("*.yaml")):
            print(f"  {p.stem}")
        return 0

    if cmd == "select":
        model_id = argv[1] if len(argv) > 1 else "gpt-4o"
        sched = RouteScheduler(
            models=_DemoModels(),  # type: ignore[reportArgumentType]
            providers=_DemoProviders(),
            nodes=_DemoNodes(),  # type: ignore[reportArgumentType]
            policy=RoutingPolicy.balanced(),
        )
        route = sched.select(RouteRequest(model_id=model_id))
        print(
            f"🎯 Route: provider={route.provider} model={route.model} "
            f"node={route.node} cost=${route.cost_per_1k}/1k score={route.score}"
        )
        print(f"   policy=balanced ({route.reason})")
        return 0

    print(f"Unknown route subcommand: {cmd}")
    return 1


def main(argv: list[str] | None = None) -> int:
    import sys as _sys

    print("⚠️ AetherForge 独立 CLI 已弃用，请使用 cockpit 替代", file=_sys.stderr)
    if argv is None:
        argv = _sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: aetherforge {gateway,infer,mesh,swarm,route,triage} [subcommand_args]")
        print("\nCommands:")
        print("  gateway   LLM Gateway (List models, generate, MCP, serve)")
        print("  infer     BOS JSON stdin → AetherForge running data plane")
        print("  mesh      Compute Mesh (List nodes, status, topology-scan, health)")
        print("  swarm     Swarm Engine (Run multi-agent workflows)")
        print("  route     RouteScheduler (三级路由 select / policies)")
        print("  triage    分诊 (单条/共识/批量/benchmark/监控/服务)")
        return 0 if argv and argv[0] in ("-h", "--help") else 1

    domain = argv[0]
    sub_args = argv[1:]

    if domain == "gateway":
        return cmd_gateway(sub_args)
    elif domain == "infer":
        return cmd_infer(sub_args)
    elif domain == "mesh":
        return cmd_mesh(sub_args)
    elif domain == "swarm":
        return cmd_swarm(sub_args)
    elif domain == "route":
        return cmd_route(sub_args)
    elif domain == "triage":
        return cmd_triage(sub_args)
    else:
        print(f"Unknown domain: {domain}")
        print("Usage: aetherforge {gateway,infer,mesh,swarm,route} [subcommand_args]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
