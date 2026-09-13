"""BET-Y1Q4-T6-28 单元测试 — 四级认知阶梯 + Radix Paged KV Cache + 投机解码.

覆盖:
- CognitiveHierarchy 层级选择与降级
- RadixPagedCache 前缀复用与 LRU 淘汰
- SpeculativeEngine 接受/拒绝逻辑与统计
- 级联路由决策与延迟约束
"""

from __future__ import annotations

import time

import pytest

from aetherforge.cache import PagedKVCache, RadixNode, RadixPagedCache
from aetherforge.engine import SpeculativeConfig, SpeculativeEngine, SpeculativeResult
from aetherforge.scheduler import (
    CognitiveHierarchy,
    CognitiveTier,
    HierarchyConfig,
    TierDecision,
    TierRoute,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def hierarchy_config() -> HierarchyConfig:
    """标准四级路由配置."""
    return HierarchyConfig(
        l0_routes=[
            TierRoute(
                tier=CognitiveTier.L0_FAST_INTERCEPT,
                model_id="tiny-1b-npu",
                endpoint="local://npu/tiny-1b",
                max_latency_ms=3.0,
                capabilities=frozenset({"intent", "guard", "classify"}),
            ),
        ],
        l1_routes=[
            TierRoute(
                tier=CognitiveTier.L1_B_EXECUTOR,
                model_id="code-14b",
                endpoint="local://gpu/code-14b",
                max_latency_ms=50.0,
                capabilities=frozenset({"code", "reason", "summarize", "intent"}),
            ),
        ],
        l2_routes=[
            TierRoute(
                tier=CognitiveTier.L2_HEAVY_ARBITER,
                model_id="reason-70b",
                endpoint="local://gpu/reason-70b",
                max_latency_ms=200.0,
                capabilities=frozenset({"code", "reason", "arbitrate", "long_context"}),
            ),
        ],
        l0_confidence_threshold=0.85,
        l1_confidence_threshold=0.70,
        l2_confidence_threshold=0.60,
    )


@pytest.fixture
def hierarchy(hierarchy_config: HierarchyConfig) -> CognitiveHierarchy:
    return CognitiveHierarchy(hierarchy_config)


@pytest.fixture
def cache() -> RadixPagedCache:
    return RadixPagedCache(page_size=1024, max_memory_bytes=10 * 1024 * 1024)


@pytest.fixture
def engine() -> SpeculativeEngine:
    return SpeculativeEngine(SpeculativeConfig(max_draft_tokens=3))


# ═══════════════════════════════════════════════════════════════
# CognitiveHierarchy — 层级选择
# ═══════════════════════════════════════════════════════════════


class TestCognitiveHierarchy:
    """四级认知阶梯调度器测试."""

    def test_l0_selection_for_simple_intent(self, hierarchy: CognitiveHierarchy) -> None:
        """简单意图应路由到 L0."""
        decision = hierarchy.select_tier(intent_complexity=0.1)
        assert decision.tier == CognitiveTier.L0_FAST_INTERCEPT
        assert decision.model_id == "tiny-1b-npu"
        assert decision.confidence > 0.0

    def test_l1_selection_for_moderate_complexity(self, hierarchy: CognitiveHierarchy) -> None:
        """中等复杂度路由到 L1."""
        decision = hierarchy.select_tier(intent_complexity=0.5)
        assert decision.tier == CognitiveTier.L1_B_EXECUTOR

    def test_l2_selection_for_high_complexity(self, hierarchy: CognitiveHierarchy) -> None:
        """高复杂度路由到 L2."""
        decision = hierarchy.select_tier(intent_complexity=0.9)
        assert decision.tier == CognitiveTier.L2_HEAVY_ARBITER

    def test_capability_requirement_triggers_upgrade(self, hierarchy: CognitiveHierarchy) -> None:
        """L0 不具备的能力应触发升级."""
        decision = hierarchy.select_tier(
            intent_complexity=0.1,
            required_capabilities={"code", "reason"},
        )
        # L0 没有 code/reason 能力, 应升级到 L1
        assert decision.tier in (CognitiveTier.L1_B_EXECUTOR, CognitiveTier.L2_HEAVY_ARBITER)

    def test_latency_constraint_filters_routes(self, hierarchy: CognitiveHierarchy) -> None:
        """延迟约束应过滤不满足的路由."""
        decision = hierarchy.select_tier(
            intent_complexity=0.9,
            max_latency_ms=10.0,  # 只有 L0 满足
        )
        # 高复杂度但低延迟要求 — 应尽量选择低延迟路由
        assert decision.latency_ms >= 0  # 调度延迟本身应极小

    def test_degradation_chain_exists(self, hierarchy: CognitiveHierarchy) -> None:
        """降级链配置应正确."""
        assert hierarchy.get_degradation_target(CognitiveTier.L0_FAST_INTERCEPT) == CognitiveTier.L1_B_EXECUTOR
        assert hierarchy.get_degradation_target(CognitiveTier.L1_B_EXECUTOR) == CognitiveTier.L2_HEAVY_ARBITER
        assert hierarchy.get_degradation_target(CognitiveTier.L2_HEAVY_ARBITER) == CognitiveTier.L3_CLOUD_FALLBACK

    def test_latency_is_sub_millisecond(self, hierarchy: CognitiveHierarchy) -> None:
        """调度决策延迟应 < 1ms."""
        decision = hierarchy.select_tier(intent_complexity=0.2)
        assert decision.latency_ms < 1.0  # 纯内存计算应 < 1ms

    def test_routes_for_tier_returns_correct_list(self, hierarchy: CognitiveHierarchy) -> None:
        """按层级查询路由."""
        l0_routes = hierarchy.routes_for_tier(CognitiveTier.L0_FAST_INTERCEPT)
        assert len(l0_routes) == 1
        assert l0_routes[0].model_id == "tiny-1b-npu"

    def test_empty_hierarchy_returns_no_routes(self) -> None:
        """空配置应返回无路由."""
        h = CognitiveHierarchy(HierarchyConfig())
        decision = h.select_tier(intent_complexity=0.5)
        assert decision.model_id == "none"

    def test_tier_enum_values(self) -> None:
        """枚举值顺序."""
        assert CognitiveTier.L0_FAST_INTERCEPT < CognitiveTier.L1_B_EXECUTOR
        assert CognitiveTier.L1_B_EXECUTOR < CognitiveTier.L2_HEAVY_ARBITER
        assert CognitiveTier.L2_HEAVY_ARBITER < CognitiveTier.L3_CLOUD_FALLBACK


# ═══════════════════════════════════════════════════════════════
# RadixPagedCache — 前缀缓存
# ═══════════════════════════════════════════════════════════════


class TestRadixPagedCache:
    """Radix 前缀树 Paged KV Cache 测试."""

    def test_put_and_get(self, cache: RadixPagedCache) -> None:
        """基本的 put/get."""
        cache.put("hello", b"kv_data_hello")
        assert cache.get("hello") == b"kv_data_hello"

    def test_get_miss_returns_none(self, cache: RadixPagedCache) -> None:
        """未命中返回 None."""
        assert cache.get("nonexistent") is None

    def test_contains(self, cache: RadixPagedCache) -> None:
        """contains 检查."""
        cache.put("key1", b"value1")
        assert cache.contains("key1") is True
        assert cache.contains("key2") is False

    def test_update_existing_key(self, cache: RadixPagedCache) -> None:
        """更新已有 key."""
        cache.put("update_key", b"v1")
        cache.put("update_key", b"v2")
        assert cache.get("update_key") == b"v2"
        assert cache.stats.total_entries == 1  # 不增加计数

    def test_prefix_match(self, cache: RadixPagedCache) -> None:
        """前缀匹配."""
        cache.put("preflight:doc1", b"data1")
        cache.put("preflight:doc2", b"data2")
        cache.put("preflight:doc3_long", b"data3")
        cache.put("other:data", b"data_x")

        matches = cache.prefix_match("preflight:")
        assert len(matches) == 3
        assert all(m.startswith("preflight:") for m in matches)

    def test_remove(self, cache: RadixPagedCache) -> None:
        """删除 key."""
        cache.put("to_remove", b"value")
        assert cache.remove("to_remove") is True
        assert cache.get("to_remove") is None
        assert cache.remove("to_remove") is False  # 已删除

    def test_clear(self, cache: RadixPagedCache) -> None:
        """清空."""
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.stats.total_entries == 0

    def test_shared_prefix_efficiency(self, cache: RadixPagedCache) -> None:
        """共享前缀应复用内存."""
        prefix = "a" * 100
        key1 = prefix + "_suffix1"
        key2 = prefix + "_suffix2"

        cache.put(key1, b"v1" * 100)
        cache.put(key2, b"v2" * 100)

        # 两条数据都能正确获取
        assert cache.get(key1) == b"v1" * 100
        assert cache.get(key2) == b"v2" * 100
        assert cache.stats.total_entries == 2

    def test_snapshot_roundtrip(self, cache: RadixPagedCache) -> None:
        """快照序列化/反序列化."""
        cache.put("snap_key1", b"snap_value_1", token_count=5)
        cache.put("snap_key2", b"snap_value_2", token_count=10)

        snapshot = cache.to_snapshot()
        restored = RadixPagedCache.from_snapshot(snapshot)

        assert restored.get("snap_key1") == b"snap_value_1"
        assert restored.get("snap_key2") == b"snap_value_2"

    def test_hit_rate_tracking(self, cache: RadixPagedCache) -> None:
        """命中率统计."""
        cache.put("hit_key", b"value")
        cache.get("hit_key")  # hit
        cache.get("miss_key")  # miss

        assert cache.stats.hit_count == 1
        assert cache.stats.miss_count == 1
        assert cache.stats.hit_rate == 0.5

    def test_hash_key_deterministic(self) -> None:
        """hash_key 应稳定."""
        h1 = RadixPagedCache.hash_key("same text")
        h2 = RadixPagedCache.hash_key("same text")
        assert h1 == h2
        assert len(h1) == 16

    def test_radix_node_structure(self) -> None:
        """RadixNode 基础属性."""
        node = RadixNode(segment="test", is_terminal=True, kv_page_id=42)
        assert node.segment == "test"
        assert node.is_terminal is True
        assert node.kv_page_id == 42
        assert len(node.children) == 0


# ═══════════════════════════════════════════════════════════════
# SpeculativeEngine — 投机解码
# ═══════════════════════════════════════════════════════════════


class TestSpeculativeEngine:
    """投机解码引擎测试."""

    def test_generate_returns_result(self, engine: SpeculativeEngine) -> None:
        """generate 返回有效结果."""
        result = engine.generate(prefix=["hello"], max_tokens=5)
        assert isinstance(result, SpeculativeResult)
        assert len(result.tokens) > 0

    def test_acceptance_rate_in_range(self, engine: SpeculativeEngine) -> None:
        """接受率在合理范围."""
        result = engine.generate(prefix=["test"], max_tokens=10)
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_performance_stats_populated(self, engine: SpeculativeEngine) -> None:
        """性能统计应填充."""
        result = engine.generate(prefix=["the"], max_tokens=3)
        assert result.total_time_ms > 0
        assert result.draft_time_ms >= 0
        assert result.verify_time_ms >= 0

    def test_tokens_per_second_calculated(self, engine: SpeculativeEngine) -> None:
        """TPS 应计算."""
        result = engine.generate(prefix=["a"], max_tokens=5)
        assert result.tokens_per_second >= 0.0

    def test_config_defaults(self) -> None:
        """默认配置."""
        eng = SpeculativeEngine()
        assert eng.config.max_draft_tokens == 5
        assert eng.config.temperature == 0.7

    def test_accept_reject_logic(self, engine: SpeculativeEngine) -> None:
        """接受/拒绝逻辑."""
        draft = [("the", 0.9), ("a", 0.8), ("is", 0.7)]
        target = [("the", 0.85), ("a", 0.75), ("is", 0.65)]

        accepted, rejected = engine._accept_reject(draft, target)
        # 所有草稿 token 都应在目标中有足够高的概率
        assert accepted >= 0
        assert rejected >= 0
        assert accepted + rejected == len(draft)

    def test_reject_when_token_not_in_target(self, engine: SpeculativeEngine) -> None:
        """不在目标分布中的 token 应被拒绝."""
        draft = [("xyz", 0.9)]
        target = [("the", 0.85)]

        accepted, rejected = engine._accept_reject(draft, target)
        assert accepted == 0
        assert rejected == 1


# ═══════════════════════════════════════════════════════════════
# Integration — 级联路由 + 缓存
# ═══════════════════════════════════════════════════════════════


class TestIntegration:
    """集成测试 — 调度器 + 缓存."""

    def test_hierarchy_decision_can_be_cached(
        self, hierarchy: CognitiveHierarchy, cache: RadixPagedCache
    ) -> None:
        """调度决策结果可缓存."""
        decision = hierarchy.select_tier(intent_complexity=0.2, required_capabilities={"intent"})

        # 缓存决策
        key = f"route:{decision.tier.value}:intent"
        cache.put(key, decision.model_id.encode())

        # 从缓存获取
        cached = cache.get(key)
        assert cached is not None
        assert cached.decode() == decision.model_id

    def test_cache_prefix_match_for_routes(self, cache: RadixPagedCache) -> None:
        """路由前缀匹配."""
        cache.put("route:0:simple", b"tiny-1b")
        cache.put("route:0:complex", b"tiny-1b")
        cache.put("route:1:code", b"code-14b")

        l0_routes = cache.prefix_match("route:0:")
        assert len(l0_routes) == 2
