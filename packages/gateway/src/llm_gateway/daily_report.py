"""每日运营报告 — events.jsonl 聚合(治理 P2.2)。

防腐规则 §4 的执行面: 免费层 cost=0 也要看用量, 运营决策需要的是
请求量/失败分布/凭据动荡/预算拦截的日粒度汇总, 而不是零散事件。

生成物: ~/.aetherforge/reports/daily-YYYY-MM-DD.md
触发: gateway health loop 每日首 tick 自查(能力内建, 无外部 cron —
防腐不变量 §3)。幂等: 当日已生成则跳过。
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

_log = logging.getLogger(__name__)

_REPORTS_DIR = Path.home() / ".aetherforge" / "reports"


def generate_daily_report(day: str | None = None) -> Path | None:
    """聚合 events.jsonl 生成某日报告。day 格式 YYYY-MM-DD(默认昨天)。

    返回报告路径; 无事件或已生成返回 None(幂等)。
    """
    from .events import _EVENTS_FILE

    if day is None:
        # 报告"昨天": health loop 每日首 tick 在零点后不久触发
        day = datetime.now().strftime("%Y-%m-%d")
    out = _REPORTS_DIR / f"daily-{day}.md"
    if out.exists():
        return None
    try:
        if not _EVENTS_FILE.exists():
            return None
        import json

        kinds: Counter[str] = Counter()
        failed_models: Counter[str] = Counter()
        failed_codes: Counter[str] = Counter()
        evicted: list[str] = []
        revived: list[str] = []
        discovered: list[str] = []
        requests = 0
        tokens_in = tokens_out = 0

        with open(_EVENTS_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if datetime.fromtimestamp(rec.get("ts", 0)).strftime("%Y-%m-%d") != day:
                    continue
                kind = rec.get("kind", "?")
                kinds[kind] += 1
                payload = rec.get("payload", {})
                if kind == "request_complete":
                    requests += 1
                    tokens_in += int(payload.get("tokens_in") or 0)
                    tokens_out += int(payload.get("tokens_out") or 0)
                elif kind == "request_failed":
                    failed_models[str(payload.get("model", "?"))[:48]] += 1
                    failed_codes[str(payload.get("code", "?"))] += 1
                elif kind == "credential_evicted":
                    evicted.append(f"{payload.get('provider')}({payload.get('reason')})")
                elif kind == "credential_revived":
                    revived.append(str(payload.get("provider")))
                elif kind == "provider_discovered":
                    discovered.append(f"{payload.get('provider')}:{payload.get('model_count')}")

        if not sum(kinds.values()):
            return None

        lines = [
            f"# 网关运营日报 {day}",
            "",
            f"- 请求完成: **{requests}** (tokens in/out: {tokens_in}/{tokens_out})",
            f"- 失败: {kinds.get('request_failed', 0)} (预算拦截 {kinds.get('budget_blocked', 0)}, 流式回退 {kinds.get('stream_fallback', 0)})",
        ]
        if failed_codes:
            detail = ", ".join(f"{c}={n}" for c, n in failed_codes.most_common())
            lines.append(f"- 失败分类: {detail}")
        if failed_models:
            top = ", ".join(f"{m}×{n}" for m, n in failed_models.most_common(5))
            lines.append(f"- 失败 Top: {top}")
        if evicted:
            lines.append(f"- 凭据淘汰: {', '.join(evicted)}")
        if revived:
            lines.append(f"- 凭据复活: {', '.join(revived)}")
        if discovered:
            lines.append(f"- 免费源信号: {', '.join(discovered)}")
        lines.append(f"- 事件总量: {sum(kinds.values())} ({dict(kinds.most_common())})")

        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _log.info("[daily_report] %s generated (%d events)", out.name, sum(kinds.values()))
        return out
    except Exception as exc:  # noqa: BLE001 — 报告失败不影响运行
        _log.debug("daily report failed: %s", exc)
        return None
