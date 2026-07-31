"""aetherforge triage — 信息分诊模块 (FUNC-01 S2)

通过 omlx 网关统一调用分诊模型, 支持:
- 本地模型: mid-local, mini-9b, coder-fast
- 云端模型: deepseek-chat
- 自动 fallback: 本地失败 → 云端
- 统一记账: tracker.record() 逐条记录模型/token/成本
"""

from .router import TriageRouter, TriageResult, ConsensusResult
from .tracker import TriageTracker

__all__ = ["TriageRouter", "TriageResult", "ConsensusResult", "TriageTracker"]
