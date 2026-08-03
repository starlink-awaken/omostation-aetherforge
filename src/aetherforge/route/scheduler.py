"""RouteScheduler — 三级路由调度引擎 (ARCHITECTURE-v2 §2.2 line 88-134).

select(request) → Route 四步:
  1. Model Match: 请求模型在哪些 Provider 上有
  2. Provider Filter: 配额够? 在线? 速率允许?
  3. Strategy Score: cost/speed/quota/affinity 综合评分 (policy weights)
  4. Node Bind: 绑定最优计算节点

FallbackManager: select 失败时自动降级 (放宽 filter / 备选 model / degraded route).

TASK-02788FE2 治本实现. 设计源: projects/aetherforge/ARCHITECTURE-v2.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

# ── 数据模型 (dataclass 抽象, 真实集成 gateway/mesh 留后续) ────────────────────


@dataclass(frozen=True)
class Model:
    """模型定义 (M1 model 命名空间)."""

    id: str
    providers: tuple[str, ...]  # 哪些 provider 提供这模型
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    speed_tps: float = 0.0  # tokens/sec (越高越快)


@dataclass(frozen=True)
class Provider:
    """Provider 状态 (M1 quota_definition + availability_check)."""

    id: str
    online: bool = True
    quota_pct: float = 100.0  # 剩余配额 0-100 (越高越充裕)
    rate_tpm: int = 0  # 速率限制 tokens/min (0=不限)


@dataclass(frozen=True)
class Node:
    """计算节点 (M1 compute_node)."""

    id: str
    provider: str
    healthy: bool = True


@dataclass(frozen=True)
class Route:
    """路由结果 (select 返回)."""

    provider: str
    model: str
    node: str
    cost_per_1k: float
    score: float
    degraded: bool = False  # FallbackManager 降级标记
    reason: str = ""


@dataclass(frozen=True)
class RouteRequest:
    """路由请求."""

    model_id: str
    min_quota_pct: float = 10.0  # 最低配额门槛
    max_cost_per_1k: float | None = None  # 成本上限 (None=不限)


@dataclass(frozen=True)
class RoutingPolicy:
    """路由策略 (M1 routing_policy, weights 综合评分)."""

    id: str
    strategy: str  # balanced / cost-first / speed-first / quota-first
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "cost": 0.35,
            "speed": 0.25,
            "quota": 0.25,
            "affinity": 0.15,
        }
    )
    constraints: dict[str, object] = field(default_factory=dict)

    @classmethod
    def balanced(cls) -> RoutingPolicy:
        """默认均衡策略."""
        return cls(id="RP-BALANCED", strategy="balanced")

    @classmethod
    def from_yaml(cls, path: Path) -> RoutingPolicy:
        """从 M1 routing_policy yaml 加载."""
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            id=data.get("id", path.stem),
            strategy=data.get("strategy", "balanced"),
            weights=data.get("weights", cls.balanced().weights),
            constraints=data.get("constraints", {}),
        )


@dataclass(frozen=True)
class ScoredCandidate:
    """评分后的候选 (provider + model + 综合 score)."""

    provider: Provider
    model: Model
    score: float


# ── 评分 (RouterPipeline: CostScore / SpeedScore / QuotaScore) ─────────────────


def _normalize(values: list[float]) -> list[float]:
    """归一化到 0-1 (越高越好; cost 反转: 越低成本分越高)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def score_candidates(
    candidates: list[tuple[Provider, Model]],
    policy: RoutingPolicy,
) -> list[ScoredCandidate]:
    """综合评分: cost(低优) + speed(高优) + quota(高优) + affinity(暂 1.0).

    返回按 score 降序排列. ARCHITECTURE-v2 line 123-126.
    """
    w = policy.weights
    n = len(candidates)
    if n == 0:
        return []
    # cost 越低越好 → 反转归一化 (1 - norm)
    costs = [1.0 - x for x in _normalize([m.cost_per_1k_input for _, m in candidates])]
    speeds = _normalize([m.speed_tps for _, m in candidates])
    quotas = _normalize([p.quota_pct for p, _ in candidates])
    affinities = [1.0] * n  # affinity 暂平权 (拓扑亲和留后续 TopologyManager)

    scored: list[ScoredCandidate] = []
    for i, (prov, model) in enumerate(candidates):
        s = (
            w.get("cost", 0) * costs[i]
            + w.get("speed", 0) * speeds[i]
            + w.get("quota", 0) * quotas[i]
            + w.get("affinity", 0) * affinities[i]
        )
        scored.append(ScoredCandidate(provider=prov, model=model, score=round(s, 4)))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


# ── RouteScheduler ────────────────────────────────────────────────────────────


class ModelRegistry(Protocol):
    """ModelManager 接口 (真实实现可接 gateway)."""

    def get_models(self, model_id: str) -> list[Model]: ...


class ProviderRegistry(Protocol):
    """ProviderManager 接口."""

    def get_providers(self) -> list[Provider]: ...


class NodeRegistry(Protocol):
    """TopologyManager 接口 (compute_node 绑定)."""

    def get_nodes(self, provider: str) -> list[Node]: ...


@dataclass
class RouteScheduler:
    """统一调度引擎: 模型 → Provider → 节点 三级路由."""

    models: ModelRegistry
    providers: ProviderRegistry
    nodes: NodeRegistry
    policy: RoutingPolicy = field(default_factory=RoutingPolicy.balanced)

    def select(self, request: RouteRequest) -> Route:
        """四步路由: match → filter → score → bind. 失败 raise LookupError."""
        # 1. Model Match: 请求模型在哪些 Provider
        known_models = {m.id: m for m in self.models.get_models(request.model_id)}
        target = known_models.get(request.model_id)
        if target is None:
            raise LookupError(f"model not found: {request.model_id}")

        # 2. Provider Filter: 在线 + 配额够 + (可选)成本上限
        prov_map = {p.id: p for p in self.providers.get_providers()}
        candidates: list[tuple[Provider, Model]] = []
        for pid in target.providers:
            p = prov_map.get(pid)
            if p is None or not p.online:
                continue
            if p.quota_pct < request.min_quota_pct:
                continue
            if request.max_cost_per_1k is not None and target.cost_per_1k_input > request.max_cost_per_1k:
                continue
            candidates.append((p, target))

        if not candidates:
            raise LookupError(f"no available provider for {request.model_id}")

        # 3. Strategy Score: policy weights 综合评分
        scored = score_candidates(candidates, self.policy)
        winner = scored[0]

        # 4. Node Bind: 绑定最优计算节点
        nodes = [n for n in self.nodes.get_nodes(winner.provider.id) if n.healthy]
        if not nodes:
            raise LookupError(f"no healthy node on {winner.provider.id}")
        node = nodes[0]

        return Route(
            provider=winner.provider.id,
            model=winner.model.id,
            node=node.id,
            cost_per_1k=winner.model.cost_per_1k_input,
            score=winner.score,
            reason=f"{self.policy.strategy} score={winner.score}",
        )


# ── FallbackManager (自动降级/回退) ────────────────────────────────────────────


@dataclass
class FallbackManager:
    """select 失败时逐级降级: 放宽 min_quota → 放宽 cost → degraded route.

    ARCHITECTURE-v2 line 12-13 (FallbackManager 自动降级/回退).
    """

    scheduler: RouteScheduler

    def select_with_fallback(self, request: RouteRequest) -> Route:
        """逐级降级尝试. 全失败 raise LookupError."""
        # Level 0: 原始约束
        try:
            return self.scheduler.select(request)
        except LookupError:
            pass

        # Level 1: 放宽配额门槛 (min_quota_pct → 0)
        if request.min_quota_pct > 0:
            r1 = RouteRequest(
                model_id=request.model_id,
                min_quota_pct=0.0,
                max_cost_per_1k=request.max_cost_per_1k,
            )
            try:
                route = self.scheduler.select(r1)
                return Route(
                    provider=route.provider,
                    model=route.model,
                    node=route.node,
                    cost_per_1k=route.cost_per_1k,
                    score=route.score,
                    degraded=True,
                    reason="fallback: 放宽配额门槛",
                )
            except LookupError:
                pass

        # Level 2: 放宽成本上限 (移除)
        if request.max_cost_per_1k is not None:
            r2 = RouteRequest(model_id=request.model_id, min_quota_pct=0.0, max_cost_per_1k=None)
            try:
                route = self.scheduler.select(r2)
                return Route(
                    provider=route.provider,
                    model=route.model,
                    node=route.node,
                    cost_per_1k=route.cost_per_1k,
                    score=route.score,
                    degraded=True,
                    reason="fallback: 放宽成本上限",
                )
            except LookupError:
                pass

        # Level 3: 备选 model 需 ModelRegistry 暴露 enumerate 接口 (YAGNI, 留后续)
        raise LookupError(
            f"all fallback levels exhausted for {request.model_id}; "
            f"no provider/node available even with relaxed constraints"
        )
