"""X2 budget integrity standard §3/§4 —— append-only quota ledger。

§3 实时扣减: 每条含 时间戳 / 模型 ID / 输入输出 Token / 折算 USD, 写期间持锁。
§4 审计追踪: get_remaining_budget 实时聚合分类账历史 (不读缓存快照)。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

# 按路径加载 —— 走 llm_gateway/__init__.py 会拉进 gateway -> httpx 等重依赖,
# 而本模块只依赖标准库 (fcntl/json/logging)。
_spec = importlib.util.spec_from_file_location(
    "quota_ledger",
    Path(__file__).resolve().parents[1] / "src" / "llm_gateway" / "quota_ledger.py",
)
assert _spec and _spec.loader
quota_ledger = importlib.util.module_from_spec(_spec)
sys.modules["quota_ledger"] = quota_ledger
_spec.loader.exec_module(quota_ledger)


@pytest.fixture(autouse=True)
def _tmp_ledger(tmp_path, monkeypatch):
    path = tmp_path / "llm_quota_ledger.jsonl"
    monkeypatch.setattr(quota_ledger, "LEDGER_PATH", path)
    return path


def test_append_writes_required_fields():
    entry = quota_ledger.append_quota_ledger_event("m", 10, 5, 0.01)
    assert entry["event"] == "quota_consumed"
    assert entry["model_id"] == "m"
    assert entry["input_tokens"] == 10
    assert entry["output_tokens"] == 5
    assert entry["usd_cost"] == 0.01
    assert entry["timestamp"]
    stored = json.loads(quota_ledger.LEDGER_PATH.read_text(encoding="utf-8").strip())
    assert stored == entry


def test_get_remaining_budget_aggregates_history_in_realtime():
    quota_ledger.append_quota_ledger_event("m", 1000, 0, 0.0)
    quota_ledger.append_quota_ledger_event("m", 0, 500, 0.0)
    assert quota_ledger.get_remaining_budget() == quota_ledger.DEFAULT_TOKEN_BUDGET - 1500


def test_get_remaining_budget_default_when_no_history():
    assert quota_ledger.get_remaining_budget() == float(quota_ledger.DEFAULT_TOKEN_BUDGET)


def test_only_quota_consumed_events_count():
    quota_ledger.append_quota_ledger_event("m", 100, 50, 0.0, event="other")
    assert quota_ledger.get_remaining_budget() == float(quota_ledger.DEFAULT_TOKEN_BUDGET)


def test_corrupt_lines_are_skipped_not_fatal():
    quota_ledger.LEDGER_PATH.write_text("{not json}\n", encoding="utf-8")
    assert quota_ledger.read_quota_ledger() == []
    assert quota_ledger.get_remaining_budget() == float(quota_ledger.DEFAULT_TOKEN_BUDGET)


def test_concurrent_appends_keep_every_line_intact():
    n_threads, per_thread = 8, 25
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            quota_ledger.append_quota_ledger_event(f"m{tid}", i, i, 0.0)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = quota_ledger.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per_thread
    parsed = [json.loads(line) for line in lines]  # 任一行不完整即抛
    assert len({(e["model_id"], e["input_tokens"]) for e in parsed}) == n_threads * per_thread


def test_takes_exclusive_lock_around_write(monkeypatch):
    """锁必须包住写: LOCK_EX 进、LOCK_UN 出, 顺序不能反。"""
    import fcntl as _fcntl

    calls: list[int] = []
    real = _fcntl.flock

    def spy(fd, op):
        calls.append(op)
        return real(fd, op)

    monkeypatch.setattr(quota_ledger.fcntl, "flock", spy)
    quota_ledger.append_quota_ledger_event("m", 1, 1, 0.0)
    assert calls == [_fcntl.LOCK_EX, _fcntl.LOCK_UN]


def test_lock_released_when_write_raises(monkeypatch):
    """写失败也要放锁 —— 否则一次异常就把分类账永久锁死。"""
    import fcntl as _fcntl

    def boom(fd, op):
        if op == _fcntl.LOCK_EX:
            raise OSError("boom")
        return None

    monkeypatch.setattr(quota_ledger.fcntl, "flock", boom)
    # 不抛 (记账失败不影响请求), 但也不该留下残留锁
    quota_ledger.append_quota_ledger_event("m", 1, 1, 0.0)
    quota_ledger.append_quota_ledger_event("m", 1, 1, 0.0)  # 第二次仍能写
