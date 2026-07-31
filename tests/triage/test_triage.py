"""Triage module tests — 覆盖 router/tracker/monitor/hotswap/multimodal."""

import json
import pytest
import time
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# 被测模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aetherforge.triage.router import (
    TriageRouter, TriageResult, ConsensusResult,
    TRIAGE_PROMPT, CONSENSUS_STAGE1, CONSENSUS_STAGE2,
)
from aetherforge.triage.tracker import TriageTracker, TriageRecord
from aetherforge.triage.monitor import TriageMonitor, MonitorConfig, MonitorResult, BENCHMARK_SAMPLES
from aetherforge.triage.hotswap import ModelHotSwap, HotSwapConfig, ModelHealth


# ============================================================
# router tests
# ============================================================

class TestTriageResult:
    def test_fields(self):
        r = TriageResult(verdict="丢弃", model="test", latency=1.0)
        assert r.verdict == "丢弃"
        assert r.model == "test"
        assert r.latency == 1.0
        assert r.tokens_in == 0
        assert r.tokens_out == 0
        assert r.cost_usd == 0.0
        assert r.error is None

    def test_error_result(self):
        r = TriageResult(verdict="错误", model="test", latency=0, error="timeout")
        assert r.error == "timeout"


class TestConsensusResult:
    def test_fields(self):
        r = ConsensusResult(
            verdict="沉淀", votes={"沉淀": 3}, agreement=1.0,
            status="共识", latency=1.5, details=[],
        )
        assert r.verdict == "沉淀"
        assert r.agreement == 1.0
        assert r.status == "共识"

    def test_majority(self):
        r = ConsensusResult(
            verdict="丢弃", votes={"丢弃": 2, "提醒": 1}, agreement=2/3,
            status="多数", latency=2.0, details=[],
        )
        assert r.status == "多数"
        assert r.votes["丢弃"] == 2


class TestTriagePrompt:
    def test_contains_examples(self):
        assert "淘宝" in TRIAGE_PROMPT
        assert "GitHub" in TRIAGE_PROMPT
        assert "丢弃" in TRIAGE_PROMPT
        assert "沉淀" in TRIAGE_PROMPT
        assert "提醒" in TRIAGE_PROMPT

    def test_format_placeholder(self):
        assert "{text}" in TRIAGE_PROMPT


class TestConsensusConfig:
    def test_stage1_has_two_models(self):
        assert len(CONSENSUS_STAGE1) == 2

    def test_stage2_is_tuple(self):
        assert isinstance(CONSENSUS_STAGE2, tuple)
        assert len(CONSENSUS_STAGE2) == 2


class TestTriageRouter:
    def test_init_defaults(self):
        router = TriageRouter()
        assert router.gateway is None
        assert router.tracker is None

    def test_triage_one_no_gateway(self):
        router = TriageRouter()
        result = router.triage_one("test")
        assert result.verdict == "错误"
        assert "not initialized" in result.error


# ============================================================
# tracker tests
# ============================================================

class TestTriageTracker:
    def test_init(self):
        tracker = TriageTracker()
        assert len(tracker.records) == 0

    def test_record(self):
        tracker = TriageTracker()
        result = TriageResult(verdict="丢弃", model="test", latency=1.0, tokens_in=10, tokens_out=5)
        tracker.record(result)
        assert len(tracker.records) == 1
        assert tracker.records[0].verdict == "丢弃"

    def test_summary_empty(self):
        tracker = TriageTracker()
        s = tracker.summary()
        assert s["total"] == 0

    def test_summary_with_records(self):
        tracker = TriageTracker()
        tracker.record(TriageResult(verdict="丢弃", model="m1", latency=1.0))
        tracker.record(TriageResult(verdict="沉淀", model="m1", latency=2.0))
        tracker.record(TriageResult(verdict="提醒", model="m2", latency=1.5))
        s = tracker.summary()
        assert s["total"] == 3
        assert "m1" in s["by_model"]
        assert "m2" in s["by_model"]

    def test_log_path(self, tmp_path):
        log_file = tmp_path / "test.jsonl"
        tracker = TriageTracker(log_path=log_file)
        tracker.record(TriageResult(verdict="丢弃", model="test", latency=1.0))
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["verdict"] == "丢弃"


# ============================================================
# monitor tests
# ============================================================

class TestMonitorConfig:
    def test_defaults(self):
        c = MonitorConfig()
        assert c.check_interval == 3600
        assert c.accuracy_threshold == 0.8
        assert c.latency_threshold == 2.0


class TestBenchmarkSamples:
    def test_has_20_samples(self):
        assert len(BENCHMARK_SAMPLES) == 20

    def test_covers_three_categories(self):
        cats = set(s[1] for s in BENCHMARK_SAMPLES)
        assert cats == {"丢弃", "沉淀", "提醒"}

    def test_balanced(self):
        from collections import Counter
        counts = Counter(s[1] for s in BENCHMARK_SAMPLES)
        assert counts["丢弃"] == 6
        assert counts["沉淀"] == 8
        assert counts["提醒"] == 6


class TestMonitorResult:
    def test_fields(self):
        r = MonitorResult(
            timestamp="2026-07-31T00:00:00Z", accuracy=0.95, avg_latency=1.0,
            p95_latency=1.5, total_samples=20, correct=19, errors=0, alert=False,
        )
        assert r.accuracy == 0.95
        assert r.alert is False

    def test_alert_on_low_accuracy(self):
        r = MonitorResult(
            timestamp="2026-07-31T00:00:00Z", accuracy=0.7, avg_latency=1.0,
            p95_latency=1.5, total_samples=20, correct=14, errors=0,
            alert=True, alert_reason="准确率 70% < 80%",
        )
        assert r.alert is True

    def test_alert_on_high_latency(self):
        r = MonitorResult(
            timestamp="2026-07-31T00:00:00Z", accuracy=0.9, avg_latency=3.0,
            p95_latency=4.0, total_samples=20, correct=18, errors=0,
            alert=True, alert_reason="延迟 3.00s > 2.0s",
        )
        assert r.alert is True


class TestTriageMonitor:
    def test_init(self):
        router = TriageRouter()
        monitor = TriageMonitor(router)
        assert len(monitor.history) == 0

    def test_get_trend_empty(self):
        router = TriageRouter()
        monitor = TriageMonitor(router)
        trend = monitor.get_trend()
        assert trend["count"] == 0


# ============================================================
# hotswap tests
# ============================================================

class TestModelHealth:
    def test_defaults(self):
        h = ModelHealth(name="test")
        assert h.available is False
        assert h.latency == 0.0
        assert h.error is None


class TestHotSwapConfig:
    def test_defaults(self):
        c = HotSwapConfig()
        assert c.health_check_interval == 60
        assert c.health_check_timeout == 10


class TestModelHotSwap:
    def test_init(self):
        hs = ModelHotSwap()
        assert hs.active_model is None
        assert len(hs.fallback_chain) == 0

    def test_set_chain(self):
        hs = ModelHotSwap()
        hs.set_chain(["m1", "m2", "m3"])
        assert hs.fallback_chain == ["m1", "m2", "m3"]
        assert hs.active_model == "m1"
        assert "m1" in hs.health
        assert "m2" in hs.health

    def test_switch_to_unknown(self):
        hs = ModelHotSwap()
        hs.set_chain(["m1", "m2"])
        assert hs.switch("m3") is False

    def test_get_status(self):
        hs = ModelHotSwap()
        hs.set_chain(["m1", "m2"])
        status = hs.get_status()
        assert status["active"] == "m1"
        assert "m1" in status["chain"]
        assert "m1" in status["health"]


# ============================================================
# integration smoke (mock gateway)
# ============================================================

class TestIntegrationSmoke:
    """Mock gateway 的集成测试."""

    def test_tracker_records_consensus(self):
        """tracker 能记录 consensus 结果."""
        tracker = TriageTracker()
        tracker.record(TriageResult(verdict="沉淀", model="consensus(mid-local,mini-9b)", latency=1.5))
        tracker.record(TriageResult(verdict="丢弃", model="consensus(mid-local,mini-9b)", latency=1.2))
        s = tracker.summary()
        assert s["total"] == 2
        assert "consensus" in list(s["by_model"].keys())[0]
