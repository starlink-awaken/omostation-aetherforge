"""Triage monitor — 分诊准确率持续监控.

功能:
- 定期跑 benchmark 检查准确率漂移
- 告警机制: 准确率 <80% 或延迟 >2s 时触发
- 历史趋势追踪
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .router import TriageRouter, ConsensusResult


@dataclass
class MonitorConfig:
    """监控配置."""
    check_interval: int = 3600  # 检查间隔 (秒), 默认 1 小时
    accuracy_threshold: float = 0.8  # 准确率告警阈值
    latency_threshold: float = 2.0  # 延迟告警阈值 (秒)
    log_path: Optional[Path] = None  # 监控日志路径


@dataclass
class MonitorResult:
    """单次监控结果."""
    timestamp: str
    accuracy: float
    avg_latency: float
    p95_latency: float
    total_samples: int
    correct: int
    errors: int
    alert: bool  # 是否触发告警
    alert_reason: Optional[str] = None


# 标准 benchmark 样本
BENCHMARK_SAMPLES = [
    ("【限时优惠】购买 Pro 会员立享 50% 折扣，仅限今日", "丢弃"),
    ("恭喜您中奖了！点击链接领取 100 万现金大奖", "丢弃"),
    ("今日星座运势：天秤座事业运爆棚，爱情有惊喜", "丢弃"),
    ("订阅我们的每周通讯，获取行业最新资讯", "丢弃"),
    ("广告：新店开业全场 8 折，满减优惠多多", "丢弃"),
    ("您的包裹已发货，物流单号 SF1234567890，预计明日送达", "丢弃"),
    ("Python asyncio 协程最佳实践：事件循环与任务调度详解", "沉淀"),
    ("RFC 9535: JSON-LD 1.1 序列化规范全文", "沉淀"),
    ("PostgreSQL 索引优化深度分析：B-tree vs Hash vs GIN", "沉淀"),
    ("分布式系统 CAP 定理与 BASE 理论详解", "沉淀"),
    ("Rust 所有权机制入门：borrow checker 工作原理", "沉淀"),
    ("机器学习模型评估指标综述：精确率/召回率/F1/AUC", "沉淀"),
    ("Git rebase vs merge 工作流对比与最佳实践", "沉淀"),
    ("Kubernetes Operator 模式设计与 CRD 实践", "沉淀"),
    ("会议提醒：下午 3 点产品评审会，会议室 A", "提醒"),
    ("账单提醒：信用卡还款日明天截止，请及时还款", "提醒"),
    ("紧急告警：生产服务器 CPU 使用率 100%，需立即处理", "提醒"),
    ("PR #615 等待你的 code review，已阻塞合并", "提醒"),
    ("航班变更通知：明天的 CA1234 航班提前 2 小时起飞", "提醒"),
    ("合同签约截止日期：本周五前需完成盖章", "提醒"),
]


class TriageMonitor:
    """分诊准确率监控器."""

    def __init__(
        self,
        router: TriageRouter,
        config: Optional[MonitorConfig] = None,
    ):
        self.router = router
        self.config = config or MonitorConfig()
        self.history: list[MonitorResult] = []

    def run_check(self, samples: Optional[list[tuple[str, str]]] = None) -> MonitorResult:
        """运行一次准确率检查."""
        if samples is None:
            samples = BENCHMARK_SAMPLES

        correct = 0
        lats = []
        errors = 0

        for text, expected in samples:
            result = self.router.consensus_triage(text)
            lats.append(result.latency)
            if result.verdict == expected:
                correct += 1
            elif result.verdict in ("错误", "未知"):
                errors += 1

        acc = correct / len(samples)
        avg_lat = sum(lats) / len(lats)
        p95_lat = sorted(lats)[int(len(lats) * 0.95)]

        # 判断是否告警
        alert = False
        alert_reason = None
        if acc < self.config.accuracy_threshold:
            alert = True
            alert_reason = f"准确率 {acc:.0%} < {self.config.accuracy_threshold:.0%}"
        elif avg_lat > self.config.latency_threshold:
            alert = True
            alert_reason = f"延迟 {avg_lat:.2f}s > {self.config.latency_threshold}s"

        result = MonitorResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            accuracy=acc,
            avg_latency=avg_lat,
            p95_latency=p95_lat,
            total_samples=len(samples),
            correct=correct,
            errors=errors,
            alert=alert,
            alert_reason=alert_reason,
        )

        self.history.append(result)

        if self.config.log_path:
            self._log_result(result)

        return result

    def get_trend(self, n: int = 10) -> dict:
        """获取最近 n 次检查的趋势."""
        recent = self.history[-n:] if self.history else []
        if not recent:
            return {"count": 0}

        accs = [r.accuracy for r in recent]
        lats = [r.avg_latency for r in recent]

        return {
            "count": len(recent),
            "accuracy": {
                "current": accs[-1],
                "min": min(accs),
                "max": max(accs),
                "avg": sum(accs) / len(accs),
            },
            "latency": {
                "current": lats[-1],
                "min": min(lats),
                "max": max(lats),
                "avg": sum(lats) / len(lats),
            },
            "alerts": sum(1 for r in recent if r.alert),
        }

    def _log_result(self, result: MonitorResult):
        """记录结果到日志."""
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.log_path, "a") as f:
            f.write(json.dumps({
                "ts": result.timestamp,
                "acc": result.accuracy,
                "lat_avg": result.avg_latency,
                "lat_p95": result.p95_latency,
                "samples": result.total_samples,
                "correct": result.correct,
                "errors": result.errors,
                "alert": result.alert,
                "reason": result.alert_reason,
            }) + "\n")
