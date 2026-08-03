#!/usr/bin/env python3
"""llm-gateway CLI — unified LLM access from the command line.

Usage:
    llm-gateway list                     List available models
    llm-gateway generate <prompt>        Generate from prompt
    llm-gateway generate -m deepseek "Explain..."
    llm-gateway mcp                      Start MCP server
    llm-gateway serve --port 9090        Start HTTP server
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .detection import create_provider, detect_backends
from .provider import LLMRequest
from .registry import ModelRegistry
from .scheduler import ModelScheduler
from .ssot_loader import load_ssot_models

# ── Module-level QuotaEngine singleton (stays running across CLI calls) ──────
_QUOTA_ENGINE: Any = None


def _get_quota_status() -> dict[str, dict]:
    """Lazy-init QuotaEngine, keep it running in background."""
    global _QUOTA_ENGINE
    if _QUOTA_ENGINE is None:
        try:
            from .quota_engine import QuotaEngine

            _QUOTA_ENGINE = QuotaEngine()
            _QUOTA_ENGINE.start()
            _QUOTA_ENGINE.wait_ready(timeout=8)  # Wait for first data batch
        except Exception:
            return {}
    try:
        all_status = _QUOTA_ENGINE.get_all_status()
        return {
            p: {"pct": s.quota_pct, "source": s.quota_source or "local", "available": s.available}
            for p, s in all_status.items()
        }
    except Exception:
        return {}


def cmd_list(
    use_ssot: bool = False, show_quota: bool = False, show_cost: bool = False, show_group: bool = False
) -> int:
    if show_quota:
        return _cmd_list_quota()
    if show_cost:
        return _cmd_list_cost()
    if use_ssot:
        from llm_gateway.paths import M1_COMPUTE_ENGINE_DIR, M1_MODEL_DIR

        m1_dir = M1_COMPUTE_ENGINE_DIR
        m1_model_dir = M1_MODEL_DIR
        if m1_dir.exists():
            import asyncio

            reg = ModelRegistry()
            load_ssot_models(reg, str(m1_dir), str(m1_model_dir) if m1_model_dir.exists() else None)
            asyncio.run(reg.refresh())
            sched = ModelScheduler(reg)
            loaded = sched.load_quota_rates()
            models = reg.list_models()
            # Load quota info (lazy init, cached across calls)
            quota_info: dict[str, dict] = _get_quota_status()

            if show_group:
                from collections import defaultdict

                groups: dict[str, list] = defaultdict(list)
                for m in models:
                    engine = m.id.split("/")[0] if "/" in m.id else "unknown"
                    groups[engine].append(m)
                print(f"L0 M1 compute_engine ({len(models)} models, {len(groups)} engines):")
                for engine_name in sorted(groups):
                    emodels = groups[engine_name]
                    print(f"  ┌─ {engine_name} ({len(emodels)} models)")
                    for m in emodels[:5]:  # Show first 5
                        cost = m.cost_per_1k_tokens
                        c_in = cost.get("input", "?")
                        c_out = cost.get("output", "?")
                        print(f"  │  🟢 {m.id.split('/')[-1]:45s} in=${c_in} out=${c_out}")
                    if len(emodels) > 5:
                        print(f"  │  ... and {len(emodels) - 5} more")
                    print()
            else:
                print(f"L0 M1 compute_engine ({len(models)} models, {loaded} with real prices):")
                for m in models:
                    cost = m.cost_per_1k_tokens
                    c_in = cost.get("input", "?")  # type: ignore[reportOptionalMemberAccess]
                    c_out = cost.get("output", "?")  # type: ignore[reportOptionalMemberAccess]
                    # Extract provider from model id (format: "ENG-XX/model-name")
                    prov_key = m.id.split("/")[0] if "/" in m.id else ""
                    # Map compute engine name → quota provider name
                    if prov_key == "ENG-CC-SWITCH":
                        model_name = m.id.split("/")[-1].lower() if "/" in m.id else ""
                        if model_name.startswith("claude"):
                            q_prov = "anthropic"
                        elif model_name.startswith("gpt") or model_name.startswith("o1"):
                            q_prov = "openai"
                        elif model_name.startswith("deepseek"):
                            q_prov = "deepseek"
                        elif model_name.startswith("gemini"):
                            q_prov = "gemini"
                        elif "minimax" in model_name:
                            q_prov = "minimax"
                        else:
                            q_prov = ""
                        q = quota_info.get(q_prov, {}) if q_prov else {}
                    else:
                        q = quota_info.get(prov_key, {})
                        if not q:
                            name_parts = prov_key.replace("ENG-", "").lower().split("-")
                            for part in name_parts:
                                if part in quota_info:
                                    q = quota_info[part]
                                    break
                    q_str = ""
                    if q:
                        pct = q.get("pct", 100)
                        if pct < 10:
                            q_str = f"  🔴 quota={pct:.0f}%"
                        elif pct < 50:
                            q_str = f"  🟡 quota={pct:.0f}%"
                        else:
                            q_str = f"  🟢 quota={pct:.0f}%"
                    print(f"  🟢 {m.id:50s} in=${c_in} out=${c_out}{q_str}")
            return 0

    providers = detect_backends()
    if not providers:
        print("No LLM backends available.")
        return 1
    print(f"Found {len(providers)} provider(s):")
    for p in providers:
        models = p.available_models()
        status = "🟢" if p.is_available() else "🔴"
        print(f"  {status} {p.provider_name}: {', '.join(models[:3])}")
    return 0


def cmd_generate(
    prompt: str, model: str | None, provider_name: str | None, strategy: str = "balanced", use_ssot: bool = False
) -> int:
    if use_ssot:
        import asyncio

        from llm_gateway.paths import M1_COMPUTE_ENGINE_DIR, M1_MODEL_DIR

        m1_dir = M1_COMPUTE_ENGINE_DIR
        m1_model_dir = M1_MODEL_DIR
        if not m1_dir.exists():
            print("M1 compute_engine dir not found.", file=sys.stderr)
            return 1
        from .registry import ModelRegistry
        from .ssot_loader import load_ssot_models

        reg = ModelRegistry()
        load_ssot_models(reg, str(m1_dir), str(m1_model_dir) if m1_model_dir.exists() else None)
        asyncio.run(reg.refresh())
        if not model:
            models = reg.list_models()
            if not models:
                print("No models available.", file=sys.stderr)
                return 1
            model = models[0].id
        # Find model regardless of provider prefix
        md = reg.get(model) or reg.get(f"ENG-CC-SWITCH/{model}")
        if not md:
            # Fuzzy match: find any model whose id contains the given name
            all_m = reg.list_models()
            matches = [m for m in all_m if model.lower() in m.id.lower()]
            if matches:
                md = matches[0]
        if not md:
            print(f"Model '{model}' not found in SSOT registry.", file=sys.stderr)
            return 1
        from .route_scheduler import RouteScheduler
        from .types import ChatOptions

        # Route-aware generation: use RouteScheduler for provider selection
        sched = RouteScheduler()
        route = sched.select(task=prompt, model=md.name, strategy=strategy)
        if route:
            print(
                f"[{route.provider}/{route.model}] cost=${route.cost_per_1k_input:.4f}/1K  quota={route.quota_pct:.0f}%  strategy={strategy}",
                file=sys.stderr,
            )

        opts = ChatOptions()
        result = asyncio.run(reg.chat(md.id, [{"role": "user", "content": prompt}], opts))
        if result:
            print(result.content)
            if result.usage:
                u = result.usage
                print(
                    f"\n[{result.model}] {u.get('prompt_tokens', 0)} in / {u.get('completion_tokens', 0)} out",
                    file=sys.stderr,
                )
            return 0
        print("No response.", file=sys.stderr)
        return 1

    if provider_name:
        providers = [create_provider(provider_name)]
    else:
        providers = detect_backends()
    if not providers:
        print("No LLM backends available.", file=sys.stderr)
        return 1
    provider = providers[0]
    if not provider.is_available():
        print(f"Provider {provider.provider_name} not available.", file=sys.stderr)
        return 1
    req = LLMRequest(prompt=prompt, model=model or provider.default_model)  # type: ignore[reportAttributeAccessIssue]
    try:
        resp = provider.complete(req)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(resp.content)
    if resp.input_tokens:
        print(f"\n[{resp.model}] {resp.input_tokens} in / {resp.output_tokens} out", file=sys.stderr)
    return 0


def cmd_mcp() -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0


def _cmd_list_quota() -> int:
    """显示所有 Provider 的配额状态。"""
    from .quota_engine import QuotaEngine

    qe = QuotaEngine()
    qe.start()
    qe.wait_ready(timeout=10)
    summary = qe.get_summary()
    print(f"{'Provider':18s} {'Key':6s} {'Online':7s} {'Quota':10s} {'Source':8s}")
    print("-" * 55)
    for p in summary["providers"]:
        key = "✅" if p["has_key"] else "—"
        online = "✅" if p.get("online") else ("—" if p["has_key"] else "  ")
        q = f"{p['quota_pct']:.0f}%" if p["quota_pct"] is not None else "—"
        src = p["quota_source"] if p["quota_source"] else "—"
        print(f"  {p['provider']:16s} {key:6s} {online:7s} {q:10s} {src:8s}")
    print(f"\ncodexbar: {'✅' if summary['codexbar_available'] else '❌'}")
    print(f"{summary['available']}/{summary['total']} available")
    qe.stop()
    return 0


def _cmd_list_cost() -> int:
    """显示所有模型的定价。"""
    from .pricing import PricingRegistry

    pricing = PricingRegistry()
    all_prices = pricing.list_all()
    print(f"{'Model':30s} {'Provider':12s} {'Cost In':10s} {'Cost Out':10s} {'Context':8s}")
    print("-" * 70)
    for mp in all_prices:
        ci = f"${mp.cost_per_1k_input:.4f}" if mp.cost_per_1k_input >= 0 else "?"
        co = f"${mp.cost_per_1k_output:.4f}" if mp.cost_per_1k_output >= 0 else "?"
        print(f"{mp.model_id:30s} {mp.provider:12s} {ci:10s} {co:10s} {mp.context_window:>8d}")
    print(f"\nTotal: {len(all_prices)} models")
    return 0


def cmd_serve(port: int) -> int:
    """Start a simple HTTP server for LLM generation."""
    from .http_server import serve

    serve(port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-gateway", description="Unified LLM Gateway CLI")
    sub = parser.add_subparsers(dest="cmd")

    list_p = sub.add_parser("list", help="List available models")
    list_p.add_argument("--ssot", action="store_true", help="从 L0 M1 节点加载")
    list_p.add_argument("--quota", "-q", action="store_true", help="显示配额大盘")
    list_p.add_argument("--cost", "-c", action="store_true", help="显示模型定价")
    list_p.add_argument("--group", "-g", action="store_true", help="按 Engine 分组展示")

    gen = sub.add_parser("generate", help="Generate LLM response")
    gen.add_argument("prompt")
    gen.add_argument("--model", "-m", help="Model name")
    gen.add_argument("--provider", "-p", help="Provider name (ollama, openai, ...)")
    gen.add_argument("--ssot", action="store_true", help="Use SSOT registry (L0 M1 models)")
    gen.add_argument(
        "--strategy",
        "-s",
        default="balanced",
        choices=["balanced", "cost_first", "speed_first", "quota_first"],
        help="Routing strategy",
    )

    mcp_p = sub.add_parser("mcp", help="Start MCP server (stdio)")
    mcp_p.add_argument("--ssot", action="store_true", help="从 L0 M1 节点加载模型")

    srv = sub.add_parser("serve", help="Start HTTP server")
    srv.add_argument("--port", "-p", type=int, default=int(os.environ.get("LLM_GATEWAY_PORT", "9290")))

    args = parser.parse_args(argv)
    if args.cmd == "list":
        return cmd_list(
            use_ssot=getattr(args, "ssot", False),
            show_quota=getattr(args, "quota", False),
            show_cost=getattr(args, "cost", False),
            show_group=getattr(args, "group", False),
        )
    elif args.cmd == "generate":
        return cmd_generate(
            args.prompt,
            args.model,
            args.provider,
            strategy=getattr(args, "strategy", "balanced"),
            use_ssot=getattr(args, "ssot", False),
        )
    elif args.cmd == "mcp":
        return cmd_mcp()
    elif args.cmd == "serve":
        return cmd_serve(args.port)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
