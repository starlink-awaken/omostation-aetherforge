from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

AUDIT_DIR = Path(
    os.environ.get(
        "LLM_GATEWAY_AUDIT_DIR",
        str(Path(__file__).resolve().parents[2] / "audit"),
    )
)
AUDIT_LOG = AUDIT_DIR / "llm_calls.jsonl"


class LLMCallAuditRecord(BaseModel):
    ts: str = Field(..., description="UTC ISO8601 with Z suffix")
    task_id: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    route: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ts(self) -> LLMCallAuditRecord:
        if not self.ts.endswith("Z"):
            raise ValueError("ts must end with Z")
        return self


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_llm_audit(
    *,
    task_id: str,
    role: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_cost_usd: float,
    latency_ms: float,
    route: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    audit_log: Path | None = None,
) -> Path:
    record = LLMCallAuditRecord(
        ts=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        task_id=task_id,
        role=role,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=total_cost_usd,
        latency_ms=latency_ms,
        route=dict(route or {}),
        metadata=dict(metadata or {}),
    )
    target = audit_log or AUDIT_LOG
    _append_jsonl(target, record.model_dump())
    return target
