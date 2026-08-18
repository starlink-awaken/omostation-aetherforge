"""AetherForge Swarm RPC — fail-closed shim (Y1Q4-T6-01 归并处置).

swarm_engine 包已于 2026-08-18 删除 (Y1Q4-T6-01 零外部真实消费者)。
本入口保留 BOS 路由契约 (bos://capability/swarm/run) 但 fail-closed.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

SWARM_DISABLED_MSG = (
    "swarm_engine removed (Y1Q4-T6-01); "
    "swarm capability disabled by design"
)


def run_swarm_workflow(goal: str, **kwargs: Any) -> dict[str, Any]:
    """BOS RPC 入口: bos://capability/swarm/run — fail-closed."""
    _log.warning("[Swarm RPC] %s (goal=%r)", SWARM_DISABLED_MSG, goal[:80] if goal else goal)
    return {
        "status": "failed",
        "error": SWARM_DISABLED_MSG,
        "goal": goal,
    }
