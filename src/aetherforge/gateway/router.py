"""aetherforge.gateway.router — 网关层三层路由接口 (BET-Y1Q4-T3-03).

同步 omlxc TieredRouter 语义到 AetherForge 网关 (独立包, 不 import omlxc —
语义镜像 + provider 占位, 决策规则与 omlxc/speculative_router.py 单源对齐).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA = "aetherforge.gateway.router.v1"

# 与 omlxc.dataplane.speculative_router.TIERS 语义对齐 (单源镜像, 变更须双侧)
GATEWAY_TIERS = {
    "light": {"role": "意图分类+槽位 (<5ms)", "providers": ["edge-1.5b", "edge-3b"]},
    "mid": {"role": "格式校验+草稿初筛", "providers": ["local-8b", "local-14b"]},
    "heavy": {"role": "深度拟稿+政策推演", "providers": ["sovereign-27b", "cluster-70b"]},
}

_HEAVY_SIGNALS = ("架构设计", "长远愿景", "博弈推演", "复杂重构", "红蓝对抗", "立项方案", "政策推演", "战略规划", "深度分析")
_MID_SIGNALS = ("校验", "初筛", "审阅", "格式", "摘要", "翻译", "改写", "多段", "报告", "公文生成")
_LIGHT_ACTIONS = ("查", "看看", "几点", "天气", "提醒", "记录", "备注", "打开", "关闭", "搜索", "设置")


@dataclass(frozen=True, slots=True)
class GatewayRoute:
    tier: str
    provider: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "provider": self.provider, "reason": self.reason, "schema": SCHEMA}


def route_request(prompt: str, domain: str = "general") -> GatewayRoute:
    """网关入口路由 — 决策规则与 omlxc TieredRouter 单源对齐 (<1ms 语义)."""
    text = prompt.strip()
    if any(k in text for k in _HEAVY_SIGNALS) or len(text) > 600 or domain in ("strategy", "policy-deep"):
        return GatewayRoute("heavy", GATEWAY_TIERS["heavy"]["providers"][0], "深度推演信号 → 主力大模型")
    if any(k in text for k in _MID_SIGNALS) or len(text) > 120:
        return GatewayRoute("mid", GATEWAY_TIERS["mid"]["providers"][0], "校验/初筛 → 中层")
    if any(a in text for a in _LIGHT_ACTIONS) or len(text) <= 60:
        return GatewayRoute("light", GATEWAY_TIERS["light"]["providers"][0], "简单指令 → 端侧秒结")
    return GatewayRoute("light", GATEWAY_TIERS["light"]["providers"][0], "短常规 → 轻端兜底")
