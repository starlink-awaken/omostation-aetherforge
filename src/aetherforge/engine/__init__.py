"""aetherforge.engine — 投机解码引擎.

实现基于草稿模型的投机解码 (Speculative Decoding):
- 小模型快速生成草稿 token
- 大模型并行验证
- 接受/拒绝机制确保输出质量

BET-Y1Q4-T6-28 核心交付.
"""

from __future__ import annotations

from .speculative import (
    SpeculativeEngine,
    SpeculativeConfig,
    SpeculativeResult,
    VerificationStatus,
)

__all__ = [
    "SpeculativeEngine",
    "SpeculativeConfig",
    "SpeculativeResult",
    "VerificationStatus",
]
