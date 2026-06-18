"""Budget Policy Enforcement — AetherForge Unified Gateway.

Provides:
- BudgetExhausted exception
- check_budget_limit() for pre-call projection
- get_remaining_budget() for cross-session tracking
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .quota_engine import QuotaEngine
from .pricing import PricingRegistry

_log = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """Raised when an LLM call would exceed the configured budget."""

    def __init__(self, message: str, spent: float, cap: float, task_id: str | None = None):
        super().__init__(message)
        self.spent = spent
        self.cap = cap
        self.task_id = task_id


def get_remaining_budget() -> float | None:
    """Read the remaining budget from the QuotaEngine.
    
    Returns USD value or None if no budget is configured.
    """
    try:
        qe = QuotaEngine()
        # We don't start the thread, just read the cache
        summary = qe.get_summary()
        # In the new engine, it might be named differently or aggregated
        # Fallback to a high value if unlimited
        return summary.get("effective_remaining_budget_usd")
    except Exception as e:
        _log.debug("failed_to_get_remaining_budget: %s", e)
        return None


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost using the new PricingRegistry."""
    registry = PricingRegistry()
    price = registry.get_price(model_id)
    if not price:
        return 0.0
    
    i_cost = (input_tokens / 1000.0) * price.cost_per_1k_input
    o_cost = (output_tokens / 1000.0) * price.cost_per_1k_output
    return i_cost + o_cost


def _register_budget_debt(task_id: str, model_id: str, budget_usd: float, estimated_cost_usd: float) -> str:
    """Register a budget-rejection debt through the OMO ingress broker."""
    import re
    import sys
    from datetime import UTC, datetime

    ws_root = Path(os.environ.get("WORKSPACE") or (Path.home() / "Workspace")).resolve()
    omo_src = ws_root / "projects" / "omo" / "src"
    if not omo_src.exists():
        return ""

    if str(omo_src) not in sys.path:
        sys.path.insert(0, str(omo_src))

    try:
        from omo.omo_ingress import upsert_debt_item
    except Exception as exc:
        _log.debug("failed_to_import_omo_ingress: %s", exc)
        return ""

    suffix = re.sub(r"[^A-Za-z0-9]+", "-", task_id).strip("-").upper()[:48] or "UNNAMED"
    debt_id = f"DEBT-OPC-P4-BUDGET-{suffix}"
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_ref = f"aetherforge:budget:{task_id}"
    payload = {
        "id": debt_id,
        "title": "OPC P4 budget policy rejected an LLM execution path",
        "description": (
            f"AetherForge Gateway blocked task `{task_id}` because\n"
            f"  estimated cost {estimated_cost_usd:.6f} USD exceeded budget {budget_usd:.6f} USD.\n"
            f"  model={model_id}."
        ),
        "severity": "medium",
        "source": "aetherforge-gateway",
        "registered_at": now_iso,
        "first_seen_at": now_iso,
        "last_seen_at": now_iso,
        "occurrence_count": 1,
        "status": "open",
        "lifecycle_state": "identified",
        "task_id": task_id,
        "prerequisite_for": task_id,
        "remediation": "Increase budget or select a cheaper model.",
    }

    try:
        upsert_debt_item(
            ws_root / ".omo",
            debt_data=payload,
            ingress_plane="projects/aetherforge",
            source_ref=source_ref,
            now=now_iso,
        )
        return str(ws_root / ".omo" / "debt" / "items" / f"{debt_id}.yaml")
    except Exception as exc:
        _log.debug("failed_to_register_budget_debt: %s", exc)
        return ""


def check_budget_limit(
    *,
    model_id: str,
    input_tokens: int,
    max_output_tokens: int = 512,
    task_id: str | None = None,
    local_budget_limit: float | None = None,
) -> None:
    """Pre-call budget check."""
    projected_cost = estimate_cost(model_id, input_tokens, max_output_tokens)
    
    limit_hit = False
    active_limit = 0.0
    
    # 1. Local limit check
    if local_budget_limit is not None:
        if projected_cost > local_budget_limit:
            limit_hit = True
            active_limit = local_budget_limit
            
    # 2. Global limit check
    if not limit_hit:
        remaining = get_remaining_budget()
        if remaining is not None:
            if projected_cost > remaining:
                limit_hit = True
                active_limit = remaining
                
    if limit_hit:
        if task_id:
            _register_budget_debt(task_id, model_id, active_limit, projected_cost)
            
        raise BudgetExhausted(
            f"Projected cost ${projected_cost:.6f} exceeds budget ${active_limit:.6f} (model={model_id})",
            spent=projected_cost,
            cap=active_limit,
            task_id=task_id
        )
            
    _log.debug("budget_check_passed: projected_cost=%s", projected_cost)
