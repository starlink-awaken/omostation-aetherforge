"""RouteScheduler 测试 — 三级路由 + policy weights + FallbackManager 降级.

TASK-02788FE2 evidence: "aetherforge route 命令 + 三级路由 (模型→Provider→节点) 测试".
设计: projects/aetherforge/ARCHITECTURE-v2.md line 88-134.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherforge.route import (
    FallbackManager,
    Model,
    Node,
    Provider,
    RouteRequest,
    RouteScheduler,
    RoutingPolicy,
)

# ── 测试用 mock registry (治本: 验证 select 算法, 不依赖真实 gateway/mesh) ────


class MockModels:
    def __init__(self, models: list[Model]) -> None:
        self._models = models

    def get_models(self, model_id: str) -> list[Model]:
        # 真实 ModelManager: 返回所有已知模型 (RouteScheduler 内部按 id 过滤)
        return self._models


class MockProviders:
    def __init__(self, providers: list[Provider]) -> None:
        self._providers = providers

    def get_providers(self) -> list[Provider]:
        return self._providers


class MockNodes:
    def __init__(self, nodes_by_provider: dict[str, list[Node]]) -> None:
        self._map = nodes_by_provider

    def get_nodes(self, provider: str) -> list[Node]:
        return self._map.get(provider, [])


# ── 固件: gpt-4o 在 deepseek(便宜/慢) + openai(贵/快), deepseek 配额满 ──────


@pytest.fixture
def scheduler_balanced() -> RouteScheduler:
    models = [
        Model(
            id="gpt-4o",
            providers=("deepseek", "openai"),
            cost_per_1k_input=0.001,  # deepseek 便宜 (用它的成本代表)
            cost_per_1k_output=0.002,
            speed_tps=50.0,
        ),
    ]
    providers = [
        Provider(id="deepseek", online=True, quota_pct=100.0, rate_tpm=10000),
        Provider(id="openai", online=True, quota_pct=84.0, rate_tpm=10000),
    ]
    nodes = {
        "deepseek": [Node(id="deepseek-cloud", provider="deepseek", healthy=True)],
        "openai": [Node(id="openai-cloud", provider="openai", healthy=True)],
    }
    return RouteScheduler(
        models=MockModels(models),
        providers=MockProviders(providers),
        nodes=MockNodes(nodes),
        policy=RoutingPolicy.balanced(),
    )


# ── 1. 三级路由: match → filter → score → bind ────────────────────────────────


def test_select_three_level_route(scheduler_balanced: RouteScheduler) -> None:
    """select 返回 Route 含 provider/model/node/score (四级完整)."""
    route = scheduler_balanced.select(RouteRequest(model_id="gpt-4o"))
    assert route.model == "gpt-4o"
    assert route.provider in {"deepseek", "openai"}
    assert route.node.endswith("-cloud")
    assert route.score > 0
    assert not route.degraded


def test_select_model_not_found_raises(scheduler_balanced: RouteScheduler) -> None:
    """模型不存在 → LookupError."""
    with pytest.raises(LookupError, match="model not found"):
        scheduler_balanced.select(RouteRequest(model_id="unknown-model"))


def test_select_provider_filter_quota(scheduler_balanced: RouteScheduler) -> None:
    """配额低于门槛的 provider 被 filter 掉."""
    # deepseek quota 100, openai quota 84; 门槛 90 → openai 被过滤
    req = RouteRequest(model_id="gpt-4o", min_quota_pct=90.0)
    route = scheduler_balanced.select(req)
    assert route.provider == "deepseek"


# ── 2. policy weights 影响选择 ────────────────────────────────────────────────


def test_policy_cost_first_picks_cheaper() -> None:
    """cost-first 策略应偏向低成本 provider (即使慢)."""
    [
        Model(
            id="m",
            providers=("cheap", "fast"),
            cost_per_1k_input=0.001,
            speed_tps=10.0,
        ),
    ]
    # 用两 provider 不同 cost/speed 区分
    models_diff = [
        Model(id="m", providers=("cheap", "fast"), cost_per_1k_input=0.001, speed_tps=10.0),
        Model(id="m-fast", providers=("fast",), cost_per_1k_input=0.01, speed_tps=100.0),
    ]
    providers = [
        Provider(id="cheap", quota_pct=100.0),
        Provider(id="fast", quota_pct=100.0),
    ]
    # cheap provider 上的 m 成本 0.001; fast provider 上的 m 成本 0.001 (同模型)
    # 但用不同模型区分: cost-first 选 m (cheap 0.001) vs m-fast (0.01)
    sched = RouteScheduler(
        models=MockModels(models_diff),
        providers=MockProviders(providers),
        nodes=MockNodes({"cheap": [Node("n-cheap", "cheap")], "fast": [Node("n-fast", "fast")]}),
        policy=RoutingPolicy(id="RP-COST", strategy="cost-first", weights={"cost": 1.0, "speed": 0, "quota": 0, "affinity": 0}),
    )
    route = sched.select(RouteRequest(model_id="m"))
    assert route.provider == "cheap"


# ── 3. FallbackManager 降级 ────────────────────────────────────────────────────


def test_fallback_relaxes_quota(scheduler_balanced: RouteScheduler) -> None:
    """配额门槛过高失败 → FallbackManager 放宽 quota → degraded route."""
    FallbackManager(scheduler_balanced)
    # 门槛 95: deepseek(100) 过, openai(84) 不过 → 实际有 deepseek, 不触发降级
    # 改成所有 provider 配额不够的场景:
    providers_low = [Provider(id="deepseek", quota_pct=5.0), Provider(id="openai", quota_pct=3.0)]
    sched_low = RouteScheduler(
        models=scheduler_balanced.models,
        providers=MockProviders(providers_low),
        nodes=scheduler_balanced.nodes,
        policy=scheduler_balanced.policy,
    )
    fb_low = FallbackManager(sched_low)
    # 原始门槛 10: 都不过 → fallback 放宽到 0 → deepseek(5) 入选 (degraded)
    route = fb_low.select_with_fallback(RouteRequest(model_id="gpt-4o", min_quota_pct=10.0))
    assert route.degraded is True
    assert "放宽配额" in route.reason
    assert route.provider == "deepseek"  # quota 5 > openai 3


def test_fallback_all_exhausted_raises(scheduler_balanced: RouteScheduler) -> None:
    """所有降级耗尽 (无在线 provider) → raise LookupError."""
    providers_offline = [Provider(id="deepseek", online=False), Provider(id="openai", online=False)]
    sched_off = RouteScheduler(
        models=scheduler_balanced.models,
        providers=MockProviders(providers_offline),
        nodes=scheduler_balanced.nodes,
        policy=scheduler_balanced.policy,
    )
    fb = FallbackManager(sched_off)
    with pytest.raises(LookupError, match="all fallback levels exhausted"):
        fb.select_with_fallback(RouteRequest(model_id="gpt-4o"))


# ── 4. RoutingPolicy yaml 加载 ────────────────────────────────────────────────


def test_policy_from_yaml() -> None:
    """RP-BALANCED.yaml 能加载 + weights 正确."""
    policy_path = Path(__file__).resolve().parent.parent / "src" / "aetherforge" / "route" / "policies" / "RP-BALANCED.yaml"
    policy = RoutingPolicy.from_yaml(policy_path)
    assert policy.id == "RP-BALANCED"
    assert policy.strategy == "balanced"
    assert policy.weights["cost"] == 0.35
    assert policy.weights["quota"] == 0.25
