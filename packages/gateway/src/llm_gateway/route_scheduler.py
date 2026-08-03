"""RouteScheduler — 三层路由引擎 (Provider → Model → Node).

集成了 QuotaEngine 的配额感知、PricingRegistry 的定价、
M1 model→provider 映射、和策略评分。

用法::

    from llm_gateway.route_scheduler import RouteScheduler, RouteStrategies

    scheduler = RouteScheduler()
    route = scheduler.select("写代码", model="gpt-4o")
    print(f"Route: {route.provider}/{route.model} ${route.cost}/1K")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aetherforge._paths import M1_MODEL_DIR as _M1_MODEL_DIR
from aetherforge._paths import M1_ROUTING_POLICY_DIR

from .pricing import PricingRegistry
from .quota_engine import QuotaEngine

_log = logging.getLogger(__name__)


@dataclass
class Route:
    """一次路由决策的结果。"""

    provider: str = ""
    model: str = ""
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    score: float = 0.0
    strategy: str = "balanced"
    quota_pct: float = 100.0
    quota_source: str = ""
    node_id: str = ""


class RouteStrategies:
    """路由策略权重配置。"""

    BALANCED = {"cost": 0.35, "quota": 0.35, "speed": 0.30}
    COST_FIRST = {"cost": 0.70, "quota": 0.20, "speed": 0.10}
    SPEED_FIRST = {"cost": 0.10, "quota": 0.10, "speed": 0.80}
    QUOTA_FIRST = {"cost": 0.10, "quota": 0.70, "speed": 0.20}

    @classmethod
    def get(cls, name: str) -> dict[str, float]:
        return getattr(cls, name.upper(), cls.BALANCED)


# 已知 Provider 的参考延迟 (ms)
_REF_LATENCY: dict[str, float] = {
    "deepseek": 800,
    "openai": 500,
    "anthropic": 600,
    "gemini": 900,
    "minimax": 1200,
    "kimi": 700,
    "openrouter": 1500,
    "siliconflow": 1000,
    "nvidia": 2000,
    "ollama": 200,
    "hitl": 5000,
}

# Quota provider → compute_engine name mapping
_QUOTA_TO_ENGINE: dict[str, str] = {
    "anthropic": "ENG-ANTHROPIC-CLOUD",
    "deepseek": "ENG-DEEPSEEK-CLOUD",
    "openrouter": "ENG-OPENROUTER-CLOUD",
    "openai": "ENG-OPENROUTER-CLOUD",
}


def _load_model_provider_map() -> dict[str, list[str]]:
    """Load model→engine mappings from M1 MODEL-BREW-*.yaml files.

    Returns {model_id_lower: [engine_ref, ...]}
    """
    mapping: dict[str, list[str]] = {}
    if not _M1_MODEL_DIR.is_dir():
        return mapping

    try:
        import yaml
    except ImportError:
        return mapping

    for yaml_file in sorted(_M1_MODEL_DIR.glob("MODEL-BREW-*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data or "models" not in data:
                continue
            engine_ref = data.get("engine_ref", "")
            if not engine_ref:
                continue
            for model in data["models"]:
                mid = model.get("model_id", "").lower()
                if mid:
                    mapping.setdefault(mid, []).append(engine_ref)
                    # Also index short name (e.g. "claude-sonnet-4-5" from "claude-sonnet-4-5-20250929")
                    parts = mid.split("-")
                    for i in range(2, len(parts)):
                        short = "-".join(parts[:i])
                        if len(short) > 5:
                            mapping.setdefault(short, []).append(engine_ref)
        except Exception as e:
            _log.debug("Failed to load model map from %s: %s", yaml_file, e)

    return mapping


class RouteScheduler:
    """三层路由: Model Match → Provider Filter → Score → Route.

    集成:
      - M1 模型→Provider 映射 (只筛选能提供该模型的 Engine)
      - QuotaEngine (真实可用性 + 实时配额)
      - PricingRegistry (模型定价)
      - 策略评分 (成本/速度/配额)
      - L0 MOF 动态策略加载与硬性约束约束过滤 (RoutingPolicy)
    """

    def __init__(self) -> None:
        self._quota = QuotaEngine()
        self._quota.start()
        self._quota.wait_ready(timeout=8)
        self._pricing = PricingRegistry()
        self._model_map: dict[str, list[str]] = _load_model_provider_map()
        if self._model_map:
            _log.info("RouteScheduler loaded %d model→engine mappings", len(self._model_map))
        self._policies: dict[str, dict] = {}
        self._load_routing_policies()

    def _load_routing_policies(self) -> None:
        """从 M1 routing_policy/ 目录动态加载策略配置。"""
        if not M1_ROUTING_POLICY_DIR.is_dir():
            return
        try:
            import yaml

            for yaml_file in M1_ROUTING_POLICY_DIR.glob("RP-*.yaml"):
                try:
                    with open(yaml_file, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if data and "strategy" in data:
                        strategy_name = data["strategy"].lower()
                        self._policies[strategy_name] = data
                except Exception as e:
                    _log.debug("Failed to load routing policy from %s: %s", yaml_file, e)
            if self._policies:
                _log.info("RouteScheduler loaded %d dynamic routing policies from M1", len(self._policies))
        except Exception as e:
            _log.warning("Failed to initialize routing policies: %s", e)

    # ── Public API ─────────────────────────────────────────────────────────

    def select(
        self,
        task: str = "",
        model: str = "",
        strategy: str = "balanced",
    ) -> Route | None:
        """选择最优路由。

        Args:
            task: 任务描述 (用于能力匹配)。
            model: 指定模型名称。RouteScheduler 会自动筛选能提供此模型的 Engine。
            strategy: 路由策略 (balanced / cost_first / speed_first / quota_first)。

        Returns:
            ``Route`` 或 ``None`` (无可用 Provider 时)。
        """
        # 1. 动态加载权重与约束
        policy_data = self._policies.get(strategy.lower()) or {}
        if "weights" in policy_data:
            w_conf = policy_data["weights"]
            weights = {
                "cost": w_conf.get("cost", 0.0),
                "quota": w_conf.get("quota", 0.0),
                "speed": w_conf.get("speed", 0.0),
            }
        else:
            weights = RouteStrategies.get(strategy)
        constraints = policy_data.get("constraints", {})

        # 2. 获取所有 Provider 状态
        all_status = self._quota.get_all_status()

        # 3. 模型感知过滤: 只保留能提供该模型的 Engine
        matching_engines = self._find_engines_for_model(model) if model else set()
        candidates = {}
        for pname, s in all_status.items():
            if not (s.available and s.has_key):
                continue
            if matching_engines:
                # Check if this quota provider maps to a matching engine
                engine = _QUOTA_TO_ENGINE.get(pname, pname)
                if engine not in matching_engines:
                    continue
            candidates[pname] = s

        # 4. 业务约束硬过滤 (例如 min_quota_pct, max_cost_per_1k)
        filtered_candidates = {}
        for provider, status in candidates.items():
            min_quota = constraints.get("min_quota_pct")
            if min_quota is not None and status.quota_pct < min_quota:
                continue

            cost_p = self._pricing.get_cost(model) if model else {"input": 0, "output": 0}
            cost_in = cost_p.get("input", 0.0)
            cost_out = cost_p.get("output", 0.0)

            max_in = constraints.get("max_cost_per_1k_input")
            if max_in is not None and cost_in > max_in:
                continue

            max_out = constraints.get("max_cost_per_1k_output")
            if max_out is not None and cost_out > max_out:
                continue

            filtered_candidates[provider] = status
        candidates = filtered_candidates

        if not candidates:
            _log.warning("RouteScheduler: no available providers for model=%s matching policies", model)
            return None

        # 5. Score each candidate
        best_score = -1.0
        best_route: Route | None = None

        for provider, status in candidates.items():
            cost_p = self._pricing.get_cost(model) if model else {"input": 0, "output": 0}
            max_cost = 0.1
            cost_in = cost_p.get("input", 0.01)
            cost_score = max(0, 1.0 - (cost_in / max_cost)) if cost_in > 0 else 1.0

            quota_score = status.quota_pct / 100.0 if status.quota_pct > 0 else 0.5

            latency = _REF_LATENCY.get(provider, 1000)
            speed_score = max(0, 1.0 - (latency / 10000))

            total = cost_score * weights["cost"] + quota_score * weights["quota"] + speed_score * weights["speed"]

            if total > best_score:
                best_score = total
                best_route = Route(
                    provider=provider,
                    model=model or self._pricing.get_price("", provider) or "",  # type: ignore[reportArgumentType]
                    cost_per_1k_input=cost_p.get("input", 0),
                    cost_per_1k_output=cost_p.get("output", 0),
                    score=round(total, 3),
                    strategy=strategy,
                    quota_pct=status.quota_pct,
                    quota_source=status.quota_source,
                    node_id=f"{provider}-cloud",
                )

        return best_route

    def select_all(
        self,
        task: str = "",
        model: str = "",
        strategy: str = "balanced",
    ) -> list[Route]:
        """返回所有可用 Provider 的评分排序结果。"""
        # 1. 动态加载权重与约束
        policy_data = self._policies.get(strategy.lower()) or {}
        if "weights" in policy_data:
            w_conf = policy_data["weights"]
            weights = {
                "cost": w_conf.get("cost", 0.0),
                "quota": w_conf.get("quota", 0.0),
                "speed": w_conf.get("speed", 0.0),
            }
        else:
            weights = RouteStrategies.get(strategy)
        constraints = policy_data.get("constraints", {})

        all_status = self._quota.get_all_status()

        # 2. 模型感知过滤: 只保留能提供该模型的 Engine
        matching_engines = self._find_engines_for_model(model) if model else set()
        candidates = {}
        for pname, s in all_status.items():
            if not (s.available and s.has_key):
                continue
            if matching_engines:
                engine = _QUOTA_TO_ENGINE.get(pname, pname)
                if engine not in matching_engines:
                    continue
            candidates[pname] = s

        # 3. 业务约束硬过滤
        filtered_candidates = {}
        for provider, status in candidates.items():
            min_quota = constraints.get("min_quota_pct")
            if min_quota is not None and status.quota_pct < min_quota:
                continue

            cost_p = self._pricing.get_cost(model) if model else {"input": 0, "output": 0}
            cost_in = cost_p.get("input", 0.0)
            cost_out = cost_p.get("output", 0.0)

            max_in = constraints.get("max_cost_per_1k_input")
            if max_in is not None and cost_in > max_in:
                continue

            max_out = constraints.get("max_cost_per_1k_output")
            if max_out is not None and cost_out > max_out:
                continue

            filtered_candidates[provider] = status
        candidates = filtered_candidates

        # 4. 排序各个可用 Provider
        routes = []
        for provider, status in candidates.items():
            cost_p = self._pricing.get_cost(model) if model else {"input": 0, "output": 0}
            cost_in = cost_p.get("input", 0.01)
            cost_score = max(0, 1.0 - (cost_in / 0.1)) if cost_in > 0 else 1.0
            quota_score = status.quota_pct / 100.0
            latency = _REF_LATENCY.get(provider, 1000)
            speed_score = max(0, 1.0 - (latency / 10000))
            total = cost_score * weights["cost"] + quota_score * weights["quota"] + speed_score * weights["speed"]

            routes.append(
                Route(
                    provider=provider,
                    model=model or "",
                    cost_per_1k_input=cost_p.get("input", 0),
                    cost_per_1k_output=cost_p.get("output", 0),
                    score=round(total, 3),
                    strategy=strategy,
                    quota_pct=status.quota_pct,
                    quota_source=status.quota_source,
                )
            )

        return sorted(routes, key=lambda r: r.score, reverse=True)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _find_engines_for_model(self, model: str) -> set[str]:
        """Find compute engines that can serve this model.

        Searches:
        1. Exact match in M1 model→engine map
        2. Prefix match (e.g. "claude-sonnet-4-5" matches "claude-sonnet-4-5-20250929")
        3. Fuzzy match (model name appears in any known model ID)
        """
        ml = model.lower()
        engines: set[str] = set()

        # 1. Exact match
        if ml in self._model_map:
            engines.update(self._model_map[ml])

        # 2. Prefix match (quota provider maps)
        for engine in self._model_map.values():
            engines.update(engine)

        # 3. Filter by model name prefix
        matched: set[str] = set()
        for mid, engs in self._model_map.items():
            if ml in mid or mid.startswith(ml) or ml.startswith(mid):
                matched.update(engs)

        if matched:
            return matched

        # 4. If model map is empty, return empty (no filter)
        if not self._model_map:
            return set()

        return engines & matched if matched else engines
