"""结构化事件流 — 网关治理观测的 SSOT。

背景(2026-08-24 治理复盘): 网关此前只有自由文本日志(gateway.py 内 30+ 处
_log.error/warning), 关键治理动作(凭据淘汰/预算拦截/流式回退)发生时只留下
人读的散装文本 —— 事后排查要 grep 文本猜语义, 无法按事件类型过滤统计,
"可追溯/可排查"两项成熟度为零。

本模块提供 append-only JSONL 事件流:
- 每事件一行 JSON: {"v": schema_version, "kind": ..., "ts": epoch,
  "payload": {...}}
- fire-and-forget: 任何 IO/序列化异常静默吞掉(记 debug 日志), 绝不
  影响请求路径 —— 观测的代价不能转嫁给业务
- 文件锁追加写入(与 provider.py 的 llm_cost.jsonl 同 fcntl 模式),
  多进程并发安全
- 滚动截断: 单文件超上限时保留尾部重写, 防止无界增长(默认 10MB)

事件 kind 清单(稳定契约, 消费方按 kind 过滤):
- request_complete    一次云端生成完成(model/latency/tokens/cost)
- budget_blocked      预算拦截触发(provider/limit/spend)
- credential_evicted  key 被判死出局(provider/reason)
- credential_revived  死 key 复验复活(provider)
- stream_fallback     真流式回退聚合(model/reason)
- provider_unhealthy  连续失败熔断(model/failures)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_MAX_EVENT_BYTES = 10 * 1024 * 1024

_EVENTS_FILE = Path.home() / ".aetherforge" / "events.jsonl"


def _truncate_if_needed() -> None:
    """单文件超限则保尾重写(粗粒度治理, 事件价值密度在近期)。"""
    try:
        if not _EVENTS_FILE.exists() or _EVENTS_FILE.stat().st_size < _MAX_EVENT_BYTES:
            return
        with open(_EVENTS_FILE, "rb") as f:
            f.seek(max(0, _EVENTS_FILE.stat().st_size - _MAX_EVENT_BYTES // 2))
            f.readline()  # 丢弃半行, 对齐行首
            tail = f.read()
        tmp = _EVENTS_FILE.with_suffix(".jsonl.tmp")
        tmp.write_bytes(tail)
        os.replace(tmp, _EVENTS_FILE)
    except Exception as exc:
        _log.debug("events truncate failed: %s", exc)


def emit(kind: str, payload: dict[str, Any] | None = None) -> None:
    """追加一条结构化事件。任何失败静默(观测不得影响业务路径)。"""
    try:
        record = {"v": _SCHEMA_VERSION, "kind": kind, "ts": time.time(), "payload": payload or {}}
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        _EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        import fcntl

        fd = os.open(str(_EVENTS_FILE), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        _truncate_if_needed()
    except Exception as exc:
        _log.debug("event emit failed (kind=%s): %s", kind, exc)


def tail_events(n: int = 20, kind: str | None = None) -> list[dict[str, Any]]:
    """读最近 n 条事件(可选按 kind 过滤)。文件不存在返回空表。"""
    try:
        if not _EVENTS_FILE.exists():
            return []
        with open(_EVENTS_FILE) as f:
            lines = f.readlines()
        out: list[dict[str, Any]] = []
        for line in reversed(lines):
            if len(out) >= n:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind and rec.get("kind") != kind:
                continue
            out.append(rec)
        return out
    except Exception as exc:
        _log.debug("tail_events failed: %s", exc)
        return []
