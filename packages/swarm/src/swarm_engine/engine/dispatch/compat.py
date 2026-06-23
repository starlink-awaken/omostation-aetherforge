# engine/dispatch/compat.py — SHIM（向后兼容重定向）
# Canonical 版本已迁移至 swarm_engine.dispatch_compat
# 保留本文件仅为避免 import 路径回归，实质内容统一由 dispatch_compat.py 维护。
from __future__ import annotations

from swarm_engine.dispatch_compat import *  # noqa: F401, F403
from swarm_engine.dispatch_compat import ExecutionCompatHelper  # noqa: F401

__all__ = ["ExecutionCompatHelper"]
