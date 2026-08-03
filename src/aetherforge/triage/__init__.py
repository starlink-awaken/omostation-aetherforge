"""aetherforge triage — 信息分诊模块 (FUNC-01 S2)

通过 omlx 网关统一调用分诊模型, 支持:
- 本地模型: mid-local, mini-9b, coder-fast
- 云端模型: deepseek-chat
- 自动 fallback: 本地失败 → 云端
- 两级共识: stage1 双模型并行 + stage2 分歧复核
- 统一记账: tracker.record() 逐条记录模型/token/成本
- 持续监控: monitor 定期 benchmark 检查漂移
- 热切换: hotswap 运行时切换模型
- 独立 HTTP 服务: server 提供 REST API
- 多模态: multimodal 支持图片/URL 分诊
"""

from .hotswap import HotSwapConfig, ModelHotSwap
from .monitor import MonitorConfig, TriageMonitor
from .multimodal import MultiModalTriage
from .router import ConsensusResult, TriageResult, TriageRouter
from .tracker import TriageTracker

__all__ = [
    "TriageRouter",
    "TriageResult",
    "ConsensusResult",
    "TriageTracker",
    "TriageMonitor",
    "MonitorConfig",
    "ModelHotSwap",
    "HotSwapConfig",
    "MultiModalTriage",
]
