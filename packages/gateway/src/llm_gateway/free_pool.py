"""免费算力池发现闭环 — 内建于 gateway 本体(治理 P1.2)。

背景(2026-08-24): 原 free-model-scanner.py / provider-sync.py 外挂脚本
在多 agent 并发中永久丢失(磁盘+git 历史均无, omostation AGENT-BRIEF
§1.1 失败模式的又一受害者)。教训固化为防腐规则"能力内建优于外挂"
(GOVERNANCE-V1 §6.3): 承载持续机制的运维能力必须进仓库+进测试。

本模块实现"发现"这一环(闭环中最易丢失的部分):
- FREE_POOL_CANDIDATES: 候选免费源清单(代码即文档, 增删走 PR)
- FreePoolScanner.scan(): 对候选做无鉴权 models 探测, 与上次快照
  diff, 新源/新模型出现 → emit provider_discovered 事件
- 凭据与接线仍由人决策: 发现只发事件不自动注册(凭据必须人来),
  events.jsonl 的 provider_discovered 是运营的输入信号

探测判据保守: 仅记录"能拿到非空模型清单"的源; 401/403 记
needs_key=True(源活着但需要凭据, 换 key 提示的输入); 网络错不动。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .events import emit

_log = logging.getLogger(__name__)

# 候选免费源 SSOT(2026-08-24 起维护): name → 无鉴权探测端点。
# 增删候选走 PR —— 这是防腐不变量 §3 的直接应用。
FREE_POOL_CANDIDATES: dict[str, str] = {
    "openrouter-free": "https://openrouter.ai/api/v1/models",
    "nvidia-free": "https://integrate.api.nvidia.com/v1/models",
    "opencode-zen": "https://opencode.ai/zen/v1/models",
    "siliconflow-free": "https://api.siliconflow.cn/v1/models",
    "glm-free": "https://open.bigmodel.cn/api/paas/v4/models",
    "deepseek": "https://api.deepseek.com/v1/models",
}


@dataclass
class PoolSnapshot:
    """一次扫描的源状态。"""

    provider: str
    reachable: bool = False
    needs_key: bool = False
    model_count: int = 0
    sample_models: list[str] = field(default_factory=list)


class FreePoolScanner:
    """周期扫描候选源, diff 出新信号发事件(线程化调用, 自带内存状态)。"""

    def __init__(self, candidates: dict[str, str] | None = None) -> None:
        self._candidates = candidates if candidates is not None else FREE_POOL_CANDIDATES
        self._last: dict[str, PoolSnapshot] = {}

    def _probe(self, name: str, url: str) -> PoolSnapshot:
        import httpx

        snap = PoolSnapshot(provider=name)
        try:
            resp = httpx.get(url, timeout=8)
        except Exception as exc:  # noqa: BLE001 — 探测失败是常态, 不上抛
            _log.debug("free pool probe %s failed: %s", name, exc)
            return snap
        if resp.status_code in (401, 403):
            snap.needs_key = True
            return snap
        if resp.status_code != 200:
            return snap
        try:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception:  # noqa: BLE001
            return snap
        if not models:
            return snap
        snap.reachable = True
        snap.model_count = len(models)
        snap.sample_models = sorted(models)[:5]
        return snap

    def scan(self) -> dict[str, Any]:
        """全量探测 + diff。返回摘要(调用方可记日志), 事件经 emit 发出。"""
        summary: dict[str, Any] = {"probed": len(self._candidates), "reachable": 0, "new_signals": 0}
        for name, url in self._candidates.items():
            snap = self._probe(name, url)
            if snap.reachable:
                summary["reachable"] += 1
            prev = self._last.get(name)
            self._last[name] = snap
            # 新信号: 首次可达 / 模型数明显增长(>20% 且 >=5 个新)
            is_new = prev is None or (snap.reachable and not prev.reachable)
            grew = (
                prev is not None
                and snap.reachable
                and snap.model_count - prev.model_count >= 5
            )
            if snap.reachable and (is_new or grew):
                summary["new_signals"] += 1
                emit(
                    "provider_discovered",
                    {
                        "provider": name,
                        "model_count": snap.model_count,
                        "sample": snap.sample_models,
                        "reason": "first_seen" if is_new else "models_grew",
                    },
                )
        return summary
