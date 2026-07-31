"""Triage tracker — 逐条记录分诊调用的模型/token/成本.

用于:
- 统一记账 (哪个模型花了多少)
- 准确率追踪 (verdict vs expected)
- 成本分析 (本地 vs 云端)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TriageRecord:
    """单条记账记录."""
    timestamp: str
    text_hash: str  # 不存原文, 只存 hash
    verdict: str
    model: str
    latency: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: Optional[str] = None


@dataclass
class TriageTracker:
    """分诊记账器."""
    log_path: Optional[Path] = None
    records: list[TriageRecord] = field(default_factory=list)

    def record(self, result: "TriageResult"):
        """记录一条分诊结果."""
        import hashlib
        rec = TriageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            text_hash="",  # 不存原文
            verdict=result.verdict,
            model=result.model,
            latency=result.latency,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            error=result.error,
        )
        self.records.append(rec)

        if self.log_path:
            self._append_log(rec)

    def summary(self) -> dict:
        """汇总统计."""
        if not self.records:
            return {"total": 0}

        by_model = {}
        total_cost = 0.0
        total_latency = 0.0
        errors = 0

        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {"count": 0, "cost": 0.0, "latency_sum": 0.0, "errors": 0}
            by_model[r.model]["count"] += 1
            by_model[r.model]["cost"] += r.cost_usd
            by_model[r.model]["latency_sum"] += r.latency
            if r.error:
                by_model[r.model]["errors"] += 1
                errors += 1
            total_cost += r.cost_usd
            total_latency += r.latency

        return {
            "total": len(self.records),
            "total_cost_usd": round(total_cost, 6),
            "avg_latency": round(total_latency / len(self.records), 3),
            "errors": errors,
            "by_model": {
                m: {
                    "count": s["count"],
                    "cost_usd": round(s["cost"], 6),
                    "avg_latency": round(s["latency_sum"] / s["count"], 3),
                    "errors": s["errors"],
                }
                for m, s in by_model.items()
            },
        }

    def _append_log(self, rec: TriageRecord):
        """追加到 JSONL 日志."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "ts": rec.timestamp,
                "verdict": rec.verdict,
                "model": rec.model,
                "lat": round(rec.latency, 3),
                "tok_in": rec.tokens_in,
                "tok_out": rec.tokens_out,
                "cost": rec.cost_usd,
                "err": rec.error,
            }) + "\n")
