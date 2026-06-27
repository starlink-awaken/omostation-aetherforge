#!/usr/bin/env python3
"""m1-sync — M1 SSOT model definition synchronizer.

Reads from:
- brew models CLI (models list --json <provider>)
- cc-switch DB (pricing, model IDs)

Writes to:
- M1 model/*.yaml (merged model definitions)

Usage:
    m1-sync [--dry-run] [--providers anthropic,openai,deepseek]
"""

import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── Paths ──────────────────────────────────────────────────────────────────
from aetherforge._paths import M1_MODEL_DIR

CC_SWITCH_DB = Path.home() / ".cc-switch" / "cc-switch.db"

# ── Provider map ───────────────────────────────────────────────────────────
# brew provider id → cc-switch provider identity → M1 compute_engine ref
PROVIDERS = {
    "anthropic": {"engine": "ENG-ANTHROPIC-CLOUD"},
    "openai":    {"engine": "ENG-OPENROUTER-CLOUD"},
    "deepseek":  {"engine": "ENG-DEEPSEEK-CLOUD"},
    "google":    {"engine": "ENG-CC-SWITCH"},
    "zhipuai":   {"engine": "ENG-CC-SWITCH"},
    "kimi-for-coding": {"engine": "ENG-CC-SWITCH"},
    "minimax":   {"engine": "ENG-CC-SWITCH"},
}


def fetch_brew_models(provider: str) -> list[dict]:
    """Fetch model list from brew models CLI."""
    try:
        result = subprocess.run(
            ["models", "list", "--json", provider],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  ⚠️  brew models {provider} failed: {result.stderr[:100]}")
            return []
        return json.loads(result.stdout)
    except Exception as e:
        print(f"  ⚠️  brew models {provider} error: {e}")
        return []


def fetch_cc_switch_pricing() -> dict:
    """Fetch pricing from cc-switch DB. Returns {model_id: {input, output}}."""
    pricing = {}
    if not CC_SWITCH_DB.exists():
        return pricing
    try:
        conn = sqlite3.connect(str(CC_SWITCH_DB))
        cur = conn.cursor()
        cur.execute(
            "SELECT model_id, input_cost_per_million, output_cost_per_million "
            "FROM model_pricing"
        )
        for row in cur.fetchall():
            model_id, inp, out = row
            pricing[model_id] = {
                "cost_per_1k_input": round(float(inp) / 1000, 6),
                "cost_per_1k_output": round(float(out) / 1000, 6),
            }
        conn.close()
    except Exception as e:
        print(f"  ⚠️  cc-switch DB error: {e}")
    return pricing


def normalize_capabilities(model: dict) -> list[str]:
    """Derive capabilities from brew model data."""
    caps = ["chat"]
    modalities = (model.get("modalities") or "").lower()
    if "image" in modalities or "vision" in modalities:
        caps.append("vision")
    if model.get("tool_call"):
        caps.append("tools")
    if model.get("reasoning"):
        caps.append("reasoning")
    return caps


def parse_context(ctx_str: str | None) -> int:
    """Parse context string like '200k', '1M', '128K' to integer."""
    if not ctx_str:
        return 128000
    ctx_str = str(ctx_str).upper().replace(",", "").strip()
    if ctx_str.endswith("M"):
        return int(float(ctx_str[:-1]) * 1_000_000)
    elif ctx_str.endswith("K"):
        return int(float(ctx_str[:-1]) * 1_000)
    return int(ctx_str) if ctx_str.isdigit() else 128000


def sync_provider(provider: str, dry_run: bool = False) -> tuple[int, int]:
    """Sync a single provider's model definitions.

    Returns (brew_count, written_count).
    """
    brew_models = fetch_brew_models(provider)
    if not brew_models:
        return 0, 0

    cc_pricing = fetch_cc_switch_pricing()
    engine = PROVIDERS[provider]["engine"]

    models_out = []
    for bm in brew_models:
        mid = bm.get("id")
        if not mid:
            continue

        # Merge pricing from cc-switch or use brew data
        cost = cc_pricing.get(mid, {})
        inp_cost = cost.get("cost_per_1k_input", None)
        out_cost = cost.get("cost_per_1k_output", None)

        # Fall back to brew pricing (also per-million, convert to per-1K)
        if inp_cost is None and bm.get("input_cost") is not None:
            inp_cost = round(float(bm["input_cost"]) / 1000, 6)
        if out_cost is None and bm.get("output_cost") is not None:
            out_cost = round(float(bm["output_cost"]) / 1000, 6)

        model_entry = {
            "model_id": mid,
            "display_name": bm.get("name", mid),
            "cost_per_1k_input": inp_cost if inp_cost else 0,
            "cost_per_1k_output": out_cost if out_cost else 0,
            "context_window": parse_context(bm.get("context")),
            "capabilities": normalize_capabilities(bm),
        }
        models_out.append(model_entry)

    # Write YAML with governance metadata
    data = {
        "type": "ModelDefinition",
        "status": "active",
        "engine_ref": engine,
        "model_driven_refs": {
            "source": "brew+cc-switch",
            "synced_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "governance": {
            "owner": "aetherforge-gateway",
            "steward": "m1-sync",
            "freshness_policy": "weekly",
        },
        "models": models_out,
    }

    fname = f"MODEL-BREW-{provider.upper()}.yaml"
    path = M1_MODEL_DIR / fname

    if dry_run:
        print(f"  📄 Would write {fname} ({len(models_out)} models)")
    else:
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"  ✅ {fname} ({len(models_out)} models)")

    return len(brew_models), len(models_out)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M1 SSOT model definition synchronizer")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument(
        "--providers",
        default=",".join(PROVIDERS.keys()),
        help="Comma-separated providers (default: all)",
    )
    args = parser.parse_args()

    M1_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    total_brew = 0
    total_written = 0

    print("🔁 Syncing M1 model definitions from brew + cc-switch")
    print(f"   Providers: {', '.join(providers)}")
    print(f"   Output:    {M1_MODEL_DIR}")
    print(f"   {'[DRY RUN]' if args.dry_run else ''}")
    print()

    for provider in providers:
        print(f"  {provider}...")
        brew_cnt, written_cnt = sync_provider(provider, dry_run=args.dry_run)
        total_brew += brew_cnt
        total_written += written_cnt

    print()
    print(f"✨ Done: {total_brew} models from brew, {total_written} written to M1 YAML")


if __name__ == "__main__":
    main()
