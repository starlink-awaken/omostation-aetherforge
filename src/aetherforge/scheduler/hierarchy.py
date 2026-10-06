"""四级认知阶梯分级调度器 — BET-Y1Q4-T6-28.

实现 L0 → L1 → L2 → L3 四级投机调度架构:
- L0: 超轻量意图识别 + 护栏拦截 (首字 < 5ms)
- L1: 骨干任务执行 (代码/常规推理)
- L2: 重型仲裁 (高难逻辑/长上下文)
- L3: 云端 fallback

设计: docs/superpowers/specs/ (BET-Y1Q4-T6-28).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class CognitiveTier(IntEnum):
    """认知阶梯层级 — 数值越大能力越强、成本越高."""

    L0_FAST_INTERCEPT = 0   # 1B~3B: 意图/护栏/毫秒拦截
    L1_B_EXECUTOR = 1       # 7B~14B: 骨干任务执行
    L2_HEAVY_ARBITER = 2    # 32B~70B+: 重型仲裁
    L3_CLOUD_FALLBACK = 3   # 云端/外部 API


@dataclass(frozen=True)
class TierRoute:
    """单个层级的路由目标."""

    tier: CognitiveTier
    model_id: str
    endpoint: str
    max_latency_ms: float
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass
class TierDecision:
    """调度决策结果."""

    tier: CognitiveTier
    model_id: str
    endpoint: str
    confidence: float  # 0.0 ~ 1.0
    latency_ms: float
    reason: str
    degraded: bool = False  # 是否因置信度不足而降级


@dataclass
class HierarchyConfig:
    """四级认知阶梯配置."""

    # 各层路由表
    l0_routes: list[TierRoute] = field(default_factory=list)
    l1_routes: list[TierRoute] = field(default_factory=list)
    l2_routes: list[TierRoute] = field(default_factory=list)
    l3_routes: list[TierRoute] = field(default_factory=list)

    # 置信度阈值 — 低于则降级到上一层
    l0_confidence_threshold: float = 0.85
    l1_confidence_threshold: float = 0.70
    l2_confidence_threshold: float = 0.60

    # L0 快速拦截的最大延迟
    l0_max_latency_ms: float = 5.0

    # 降级链: 当前层失败时回退到哪一层
    degradation_chain: dict[CognitiveTier, CognitiveTier] = field(default_factory=lambda: {
        CognitiveTier.L0_FAST_INTERCEPT: CognitiveTier.L1_B_EXECUTOR,
        CognitiveTier.L1_B_EXECUTOR: CognitiveTier.L2_HEAVY_ARBITER,
        CognitiveTier.L2_HEAVY_ARBITER: CognitiveTier.L3_CLOUD_FALLBACK,
    })


class CognitiveHierarchy:
    """四级认知阶梯分级调度器.

    根据请求复杂度、置信度、延迟要求自动选择最优认知层级.
    支持级联降级: 当低层置信度不足时自动升级到高层.
    """

    def __init__(self, config: HierarchyConfig) -> None:
        self._config = config
        self._route_map: dict[CognitiveTier, list[TierRoute]] = {
            CognitiveTier.L0_FAST_INTERCEPT: config.l0_routes,
            CognitiveTier.L1_B_EXECUTOR: config.l1_routes,
            CognitiveTier.L2_HEAVY_ARBITER: config.l2_routes,
            CognitiveTier.L3_CLOUD_FALLBACK: config.l3_routes,
        }

    @property
    def config(self) -> HierarchyConfig:
        return self._config

    def routes_for_tier(self, tier: CognitiveTier) -> list[TierRoute]:
        """返回指定层级的所有路由."""
        return self._route_map.get(tier, [])

    def select_tier(
        self,
        intent_complexity: float = 0.5,
        required_capabilities: set[str] | None = None,
        max_latency_ms: float | None = None,
        _visited: set[CognitiveTier] | None = None,
    ) -> TierDecision:
        """根据请求特征选择最优认知层级.

        Args:
            intent_complexity: 意图复杂度 0.0~1.0 (越高越需要高层)
            required_capabilities: 所需能力集合
            max_latency_ms: 最大可接受延迟
            _visited: 内部防递归集合

        Returns:
            TierDecision 包含最终选中的层级和路由信息
        """
        start = time.monotonic()
        required = required_capabilities or set()
        visited = _visited or set()

        # 确定起始层级
        if intent_complexity < 0.3 and self._config.l0_routes:
            start_tier = CognitiveTier.L0_FAST_INTERCEPT
        elif intent_complexity < 0.7 and self._config.l1_routes:
            start_tier = CognitiveTier.L1_B_EXECUTOR
        elif self._config.l2_routes:
            start_tier = CognitiveTier.L2_HEAVY_ARBITER
        elif self._config.l3_routes:
            start_tier = CognitiveTier.L3_CLOUD_FALLBACK
        else:
            # 无可用路由 — 返回 L0 占位
            elapsed = (time.monotonic() - start) * 1000
            return TierDecision(
                tier=CognitiveTier.L0_FAST_INTERCEPT,
                model_id="none",
                endpoint="none",
                confidence=0.0,
                latency_ms=elapsed,
                reason="no routes configured",
            )

        # 防递归: 如果已经访问过该层级, 不再升级
        visited.add(start_tier)

        # 检查能力匹配
        routes = self._route_map.get(start_tier, [])
        capable = [r for r in routes if required.issubset(r.capabilities)] if required else routes

        if not capable:
            # 当前层无法满足能力需求 — 向上升级
            return self._upgrade_tier(
                start_tier, required, intent_complexity, start, visited,
                reason="capability mismatch"
            )

        # 检查延迟约束
        if max_latency_ms is not None:
            latency_ok = [r for r in capable if r.max_latency_ms <= max_latency_ms]
            if not latency_ok:
                return self._upgrade_tier(
                    start_tier, required, intent_complexity, start, visited,
                    reason=f"latency > {max_latency_ms}ms"
                )
            capable = latency_ok

        # 选择最优路由 (最低延迟)
        best = min(capable, key=lambda r: r.max_latency_ms)
        elapsed = (time.monotonic() - start) * 1000

        # 计算置信度 (基于能力匹配度和层级适配度)
        confidence = self._compute_confidence(best, intent_complexity, required)

        # 检查是否需要降级
        threshold = self._threshold_for_tier(start_tier)
        if confidence < threshold and start_tier != CognitiveTier.L3_CLOUD_FALLBACK:
            return self._upgrade_tier(
                start_tier, required, intent_complexity, start, visited,
                reason=f"confidence {confidence:.2f} < threshold {threshold}"
            )

        return TierDecision(
            tier=start_tier,
            model_id=best.model_id,
            endpoint=best.endpoint,
            confidence=confidence,
            latency_ms=elapsed,
            reason=f"selected {start_tier.name}",
        )

    def _upgrade_tier(
        self,
        current: CognitiveTier,
        required: set[str],
        complexity: float,
        start_time: float,
        visited: set[CognitiveTier],
        reason: str,
    ) -> TierDecision:
        """向上升级层级 — 直接尝试 fallback, 不再重新从 complexity 确定起始层."""
        fallback = self._config.degradation_chain.get(current)

        # 防递归: 如果升级目标已访问过或不存在, 停在此层
        if fallback is None or fallback == current or fallback in visited:
            elapsed = (time.monotonic() - start_time) * 1000
            routes = self._route_map.get(current, [])
            best = routes[0] if routes else None
            if best:
                return TierDecision(
                    tier=current,
                    model_id=best.model_id,
                    endpoint=best.endpoint,
                    confidence=0.5,
                    latency_ms=elapsed,
                    reason=f"{reason}, no further upgrade",
                    degraded=True,
                )
            return TierDecision(
                tier=CognitiveTier.L3_CLOUD_FALLBACK,
                model_id="none",
                endpoint="none",
                confidence=0.0,
                latency_ms=elapsed,
                reason=f"{reason}, no routes",
                degraded=True,
            )

        # 直接尝试 fallback 层级 (不再重新从 complexity 确定)
        visited.add(fallback)
        routes = self._route_map.get(fallback, [])
        capable = [r for r in routes if required.issubset(r.capabilities)] if required else routes

        if not capable:
            # fallback 也无法满足, 继续向上
            return self._upgrade_tier(
                fallback, required, complexity, start_time, visited,
                reason=f"{reason}, capability mismatch at {fallback.name}"
            )

        best = min(capable, key=lambda r: r.max_latency_ms)
        elapsed = (time.monotonic() - start_time) * 1000
        confidence = self._compute_confidence(best, complexity, required)

        threshold = self._threshold_for_tier(fallback)
        if confidence < threshold and fallback != CognitiveTier.L3_CLOUD_FALLBACK:
            return self._upgrade_tier(
                fallback, required, complexity, start_time, visited,
                reason=f"{reason}, confidence {confidence:.2f} < {threshold}"
            )

        return TierDecision(
            tier=fallback,
            model_id=best.model_id,
            endpoint=best.endpoint,
            confidence=confidence,
            latency_ms=elapsed,
            reason=f"upgraded from {current.name} to {fallback.name}: {reason}",
            degraded=True,
        )

    def _compute_confidence(
        self,
        route: TierRoute,
        complexity: float,
        required: set[str],
    ) -> float:
        """计算路由置信度."""
        # 能力匹配度
        if required:
            cap_match = len(required & route.capabilities) / len(required)
        else:
            cap_match = 1.0

        # 层级适配度 (复杂度与层级的匹配)
        tier_optimal = {
            CognitiveTier.L0_FAST_INTERCEPT: 0.15,
            CognitiveTier.L1_B_EXECUTOR: 0.5,
            CognitiveTier.L2_HEAVY_ARBITER: 0.85,
            CognitiveTier.L3_CLOUD_FALLBACK: 1.0,
        }
        optimal = tier_optimal.get(route.tier, 0.5)
        tier_fit = 1.0 - abs(complexity - optimal)

        return min(1.0, (cap_match * 0.6 + tier_fit * 0.4))

    def _threshold_for_tier(self, tier: CognitiveTier) -> float:
        """获取指定层级的置信度阈值."""
        thresholds = {
            CognitiveTier.L0_FAST_INTERCEPT: self._config.l0_confidence_threshold,
            CognitiveTier.L1_B_EXECUTOR: self._config.l1_confidence_threshold,
            CognitiveTier.L2_HEAVY_ARBITER: self._config.l2_confidence_threshold,
        }
        return thresholds.get(tier, 0.0)

    def get_degradation_target(self, tier: CognitiveTier) -> CognitiveTier | None:
        """获取指定层级的降级目标."""
        return self._config.degradation_chain.get(tier)
