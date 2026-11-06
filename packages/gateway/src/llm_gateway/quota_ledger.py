"""X2 Atomic Budget Ledger —— append-only 配额分类账。

实现 `.omo/standards/x2-budget-integrity-standard.md`:

* §3「实时扣减」: 调用成功后通过 :func:`append_quota_ledger_event` 把真实
  Token 消耗写入 ``llm_quota_ledger.jsonl``; 所有 JSONL 写入都经
  ``fcntl.flock`` 取排他锁, 保证多进程并发时数据完整。
* §4「审计追踪」: 分类账每条含 时间戳 / 模型 ID / 输入输出 Token 数 / 折算 USD;
  :func:`get_remaining_budget` **实时聚合分类账历史**, 不读任何缓存快照。

分类账路径与 ecos 侧 ``ecos.workflow.validator.X2BudgetDeducer`` 共用同一条
(``~/.omo/state/llm_quota_ledger.jsonl``), 两边必须保持一致 —— 那是跨仓共享的
运行态 SSOT, 不是某一侧的私有文件。
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

LEDGER_PATH = Path.home() / ".omo" / "state" / "llm_quota_ledger.jsonl"

# 与 ecos X2BudgetDeducer._DEFAULT_TOKEN_BUDGET 同值: 无账本时的默认余额
DEFAULT_TOKEN_BUDGET = 100_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_quota_ledger_event(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    usd_cost: float,
    **extra: Any,
) -> dict[str, Any]:
    """追加一条真实消耗记录 (§3)。写期间持 ``LOCK_EX``。

    返回写下的那条 entry。写失败只记 warning 不抛 —— 记账失败不影响请求
    (与 gateway 既有「记账缺失 ≠ 拒绝服务」的口径一致)。
    """
    entry = {
        "timestamp": _now(),
        "event": "quota_consumed",
        "model_id": model_id,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd_cost": float(usd_cost),
    }
    entry.update(extra)
    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        _log.warning("append_quota_ledger_event failed: %s", exc)
    return entry


def read_quota_ledger() -> list[dict[str, Any]]:
    """读全量分类账; 坏行跳过 (append-only JSONL 的惯例)。"""
    if not LEDGER_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(LEDGER_PATH, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def get_remaining_budget() -> float | None:
    """§4: 实时聚合分类账历史, 返回剩余 Token 预算。

    口径: 默认额度 − 所有 ``quota_consumed`` 记录的 (input+output) 之和。
    无消费记录时返回默认额度; 账本不可读时返回 ``None`` (= 未知, 不是 0)。
    """
    try:
        consumed = sum(
            int(e.get("input_tokens") or 0) + int(e.get("output_tokens") or 0)
            for e in read_quota_ledger()
            if e.get("event") == "quota_consumed"
        )
    except OSError:
        return None
    return float(DEFAULT_TOKEN_BUDGET - consumed)
