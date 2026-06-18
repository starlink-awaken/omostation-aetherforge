from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNTIME_HOME = Path(os.environ.get("RUNTIME_HOME", str(Path.home() / "runtime")))
LEDGER_LOG = RUNTIME_HOME / "data" / "llm_quota_ledger.jsonl"
SUMMARY_PATH = RUNTIME_HOME / "data" / "llm_quota_summary.json"


def append_quota_ledger_event(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
    ledger_log: Path | None = None,
) -> Path:
    record = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "estimated_cost_usd": round(float(estimated_cost_usd), 6),
    }
    target = ledger_log or LEDGER_LOG
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # ── Phase 15: Atomic Append via fcntl ──
    import fcntl
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with open(target, "ab") as fh:
        try:
            # Exclusive lock, blocking
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            
    return target


def summarize_quota_ledger(
    quota_payload: object,
    *,
    ledger_log: Path | None = None,
) -> dict[str, Any]:
    target = ledger_log or LEDGER_LOG
    records: list[dict[str, Any]] = []
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))

    total_estimated_cost = round(sum(float(item.get("estimated_cost_usd", 0.0)) for item in records), 6)
    total_input_tokens = sum(int(item.get("input_tokens", 0)) for item in records)
    total_output_tokens = sum(int(item.get("output_tokens", 0)) for item in records)
    remaining_ratio = _extract_remaining_ratio(quota_payload)
    remaining_budget_usd = _extract_remaining_budget_usd(quota_payload)
    effective_remaining_budget_usd = (
        round(max(remaining_budget_usd - total_estimated_cost, 0.0), 6)
        if remaining_budget_usd is not None
        else None
    )
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry_count": len(records),
        "total_estimated_cost_usd": total_estimated_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "remaining_ratio": remaining_ratio,
        "remaining_budget_usd": remaining_budget_usd,
        "effective_remaining_budget_usd": effective_remaining_budget_usd,
        "quota_low": remaining_ratio is not None and remaining_ratio < 0.05,
        "ledger_log": str(target),
    }


def write_quota_summary(
    quota_payload: object,
    *,
    ledger_log: Path | None = None,
    summary_path: Path | None = None,
) -> Path:
    summary = summarize_quota_ledger(quota_payload, ledger_log=ledger_log)
    target = summary_path or SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _extract_remaining_ratio(payload: object) -> float | None:
    if isinstance(payload, dict):
        for key in ("remaining_ratio", "remaining_pct", "remaining_percentage", "ratio", "percent_remaining"):
            normalized = _normalize_ratio(payload.get(key))
            if normalized is not None:
                return normalized

        remaining = payload.get("remaining")
        limit = payload.get("limit") or payload.get("total") or payload.get("quota")
        if isinstance(remaining, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
            return max(0.0, min(1.0, float(remaining) / float(limit)))

        nested = [_extract_remaining_ratio(value) for value in payload.values()]
        values = [value for value in nested if value is not None]
        return min(values) if values else None

    if isinstance(payload, list):
        nested = [_extract_remaining_ratio(value) for value in payload]
        values = [value for value in nested if value is not None]
        return min(values) if values else None

    return None


def _extract_remaining_budget_usd(payload: object) -> float | None:
    if isinstance(payload, dict):
        for key in ("remaining_budget_usd", "remaining_usd", "usd_remaining", "credits_remaining_usd"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return round(float(value), 6)
        nested = [_extract_remaining_budget_usd(value) for value in payload.values()]
        values = [value for value in nested if value is not None]
        return min(values) if values else None

    if isinstance(payload, list):
        nested = [_extract_remaining_budget_usd(value) for value in payload]
        values = [value for value in nested if value is not None]
        return min(values) if values else None

    return None


def _normalize_ratio(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric < 0:
        return 0.0
    if numeric <= 1:
        return numeric
    if numeric <= 100:
        return numeric / 100.0
    return None
