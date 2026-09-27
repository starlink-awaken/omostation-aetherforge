"""aetherforge.scheduler — 四级认知阶梯分级投机推理调度器.

L0 (1B~3B 专用 NPU 毫秒意图与护栏拦截, 首字 < 5ms)
→ L1 (7B~14B 骨干任务与代码补丁编写)
→ L2 (32B~70B+ 重型仲裁与高难逻辑)
→ L3 (fallback 到云端/外部 API)

BET-Y1Q4-T6-28 核心交付.
"""

from __future__ import annotations

from .hierarchy import (
    CognitiveHierarchy,
    CognitiveTier,
    HierarchyConfig,
    TierDecision,
    TierRoute,
)

__all__ = [
    "CognitiveHierarchy",
    "CognitiveTier",
    "HierarchyConfig",
    "TierDecision",
    "TierRoute",
]
