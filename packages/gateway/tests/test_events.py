"""events.py 结构化事件流测试 — 治理观测 SSOT 的行为契约。

背景(2026-08-24): 事件 kind 是稳定契约(消费方按 kind 过滤), 坏行必须
静默跳过(append-only 文件被截断/手编时不能炸掉 tail), emit 任何失败
不得外溢(观测代价不转嫁业务路径)。
"""

from __future__ import annotations

from pathlib import Path

from llm_gateway import events


def _redirect(tmp_path: Path, monkeypatch) -> Path:
    f = tmp_path / "events.jsonl"
    monkeypatch.setattr(events, "_EVENTS_FILE", f)
    return f


def test_emit_then_tail_roundtrip(tmp_path: Path, monkeypatch) -> None:
    f = _redirect(tmp_path, monkeypatch)
    events.emit("request_complete", {"model": "m1", "tokens_in": 5})
    events.emit("budget_blocked", {"provider": "deepseek"})
    events.emit("request_complete", {"model": "m2"})

    got = events.tail_events(10)
    assert [r["kind"] for r in got] == ["request_complete", "budget_blocked", "request_complete"]
    assert got[0]["v"] == 1 and "ts" in got[0]
    assert got[0]["payload"]["model"] == "m2"  # 最近的在前
    assert f.exists()


def test_tail_filters_by_kind(tmp_path: Path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    events.emit("credential_evicted", {"provider": "kimi_for_coding", "reason": "request_401"})
    events.emit("credential_revived", {"provider": "siliconflow"})
    events.emit("credential_evicted", {"provider": "deepseek", "reason": "probe_401_403"})

    evicted = events.tail_events(10, kind="credential_evicted")
    assert [e["payload"]["provider"] for e in evicted] == ["deepseek", "kimi_for_coding"]


def test_tail_silent_on_missing_file_and_bad_lines(tmp_path: Path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    assert events.tail_events(5) == []  # 文件不存在 → 空表不炸

    (tmp_path / "events.jsonl").write_text("not json\n{broken\n")
    assert events.tail_events(5) == []  # 坏行静默跳过


def test_emit_never_raises_on_unwritable_path(tmp_path: Path, monkeypatch) -> None:
    # 指向一个"目录当文件写"的非法路径 → emit 必须静默
    monkeypatch.setattr(events, "_EVENTS_FILE", tmp_path)  # 目录本身
    events.emit("anything", {"x": 1})  # 不抛即通过
