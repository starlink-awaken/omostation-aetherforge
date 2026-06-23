"""Model Scheduler — dynamic model selection with load awareness."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from .policies import score_models
from .quota_ledger import write_quota_summary
from .registry import ModelRegistry
from .types import (
    DEFAULT_SCHEDULER_CONFIG,
    LoadInfo,
    ModelRequest,
    ModelRoutePolicy,
    ModelSelection,
    SchedulerConfig,
)

_log = logging.getLogger(__name__)


class ModelScheduler:
    """Dynamic model scheduler with load-aware, policy-driven selection.

    Selects the optimal model for a :class:`ModelRequest` using scoring
    strategies from the policies module.  Tracks active-request load and
    optionally auto-refreshes the registry's model list.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: SchedulerConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or DEFAULT_SCHEDULER_CONFIG
        self._load_map: dict[str, LoadInfo] = {}
        self._refresh_task: asyncio.Task[None] | None = None
        self._quota_low_mode = False
        self._last_quota_refresh_at = 0.0

    @classmethod
    def from_m1_dir(cls, m1_dir: str, config: SchedulerConfig | None = None) -> ModelScheduler:
        """Create a ModelScheduler initialized with SSOT models from an M1 directory.

        Note: The caller still needs to `await registry.refresh()` or `start_auto_refresh()`
        to discover the models.
        """
        from .ssot_loader import load_ssot_models

        registry = ModelRegistry()
        load_ssot_models(registry, m1_dir)
        return cls(registry, config)

    def load_quota_rates(self) -> int:
        """从 ~/.runtime/cache/quota_rates.json 加载真实价格。

        models list --json 采集的价格数据写入缓存后，
        此方法将 ModelDescriptor 的 cost_per_1k_tokens 更新为真实价格。
        """
        cache_path = Path.home() / ".runtime" / "cache" / "quota_rates.json"
        if not cache_path.exists():
            self._quota_low_mode = False
            return 0

        try:
            with open(cache_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception):
            self._quota_low_mode = False
            return 0

        rates = data.get("rates", {})
        updated = 0
        for model in self._registry.get_all():
            model_id_short = model.id.split("/")[-1]
            if model_id_short in rates:
                r = rates[model_id_short]
                if r.get("input") is not None:
                    model.cost_per_1k_tokens["input"] = r["input"]
                    model.cost_per_1k_tokens["output"] = r.get("output", r["input"])
                    updated += 1
        quota_payload = data.get("quota")
        self._quota_low_mode = self._is_quota_low(quota_payload)
        self._last_quota_refresh_at = time.time()
        write_quota_summary(quota_payload)
        return updated

    def _is_quota_low(self, payload: object) -> bool:
        ratio = self._extract_remaining_ratio(payload)
        return ratio is not None and ratio < self._config.quota_low_remaining_ratio

    def _extract_remaining_ratio(self, payload: object) -> float | None:
        if isinstance(payload, dict):
            for key in ("remaining_ratio", "remaining_pct", "remaining_percentage", "ratio", "percent_remaining"):
                value = payload.get(key)
                normalized = self._normalize_ratio(value)
                if normalized is not None:
                    return normalized

            remaining = payload.get("remaining")
            limit = payload.get("limit") or payload.get("total") or payload.get("quota")
            if isinstance(remaining, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
                return max(0.0, min(1.0, float(remaining) / float(limit)))

            ratios = [self._extract_remaining_ratio(value) for value in payload.values()]
            ratios = [value for value in ratios if value is not None]
            return min(ratios) if ratios else None

        if isinstance(payload, list):
            ratios = [self._extract_remaining_ratio(item) for item in payload]
            ratios = [value for value in ratios if value is not None]
            return min(ratios) if ratios else None

        return None

    @staticmethod
    def _normalize_ratio(value: object) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if numeric < 0:
            return 0.0
        if numeric <= 1:
            return numeric
        if numeric <= 100:
            return numeric / 100.0
        return None

    async def select_model(
        self,
        request: ModelRequest,
        policy: ModelRoutePolicy | None = None,
    ) -> ModelSelection | None:
        """Select the best model for *request* using optional *policy*.

        Algorithm:
            1. Filter by availability + required capabilities
            2. Apply budget-aware filtering (Phase 4)
            3. Priority sort if *policy.priority* is set
            4. Score by strategy (delegated to ``policies.score_models``)
            5. Apply load penalty
            6. Return highest-scoring model
        """
        if self._quota_low_mode:
            self.load_quota_rates()

        all_models = self._registry.get_all()
        merged_policy = policy or ModelRoutePolicy(strategy=self._config.default_policy)

        # ── Phase 4: Budget-aware strategy adjustment ──
        effective_strategy = merged_policy.strategy
        if self._quota_low_mode and effective_strategy == "balanced":
            _log.info("[Scheduler] Quota low: pivoting from 'balanced' to 'cost-first'")
            effective_strategy = "cost-first"

        candidates = [
            m for m in all_models if m.is_available and all(c in m.capabilities for c in request.required_capabilities)
        ]

        # ── Phase 4: Local budget limit filtering ──
        if merged_policy.budget_limit_usd is not None:
            from .registry_data_loader import estimate_model_cost
            # Assume 1000 input + 512 output as a standard probe
            candidates = [
                m for m in candidates
                if estimate_model_cost(m.id, 1000, 512) <= merged_policy.budget_limit_usd
            ]

        if not candidates:
            return None

        # Priority-based short-circuit
        if merged_policy.priority:
            priority_map = {mid: i for i, mid in enumerate(merged_policy.priority)}
            # Still apply budget filter even for priority models
            p_candidates = [c for c in candidates if c.id in priority_map]
            if p_candidates:
                p_candidates.sort(key=lambda m: priority_map.get(m.id, 999))
                best = p_candidates[0]
                return ModelSelection(
                    model=best,
                    provider_name=best.provider,
                    confidence=1.0,
                    reasoning=f"Matched priority order: {best.id}",
                )

        scored = score_models(candidates, request, merged_policy, self._load_map)
        if not scored:
            return None

        best = scored[0]  # type: ignore[assignment]
        self._record_load(best.model.id)  # type: ignore[attr-defined]

        return ModelSelection(
            model=best.model,  # type: ignore[attr-defined]
            provider_name=best.model.provider,  # type: ignore[attr-defined]
            confidence=max(0.0, min(1.0, best.score - best.load_penalty)),  # type: ignore[attr-defined]
            reasoning=(
                f"Scored {best.score:.2f} (penalty: {best.load_penalty:.2f}): "  # type: ignore[attr-defined]
                f"{effective_strategy}"
            ),
        )

    def _record_load(self, model_id: str) -> None:
        existing = self._load_map.get(model_id)
        self._load_map[model_id] = LoadInfo(
            model_id=model_id,
            active_requests=(existing.active_requests if existing else 0) + 1,
            avg_latency_ms=existing.avg_latency_ms if existing else 0.0,
            last_checked=time.time() * 1000,
        )

    def release_load(self, model_id: str) -> None:
        """Decrement the active-request count for *model_id*."""
        load = self._load_map.get(model_id)
        if load:
            load.active_requests = max(0, load.active_requests - 1)

    def start_auto_refresh(self, interval_ms: int = 30_000) -> Callable[[], None]:
        """Start periodic model discovery refresh.

        Returns a ``dispose`` callable to stop the refresh loop.
        """
        self.stop_auto_refresh()

        async def _loop() -> None:
            while True:
                await asyncio.sleep(interval_ms / 1000)
                try:
                    await self._registry.refresh()
                    if (
                        time.time() - self._last_quota_refresh_at
                        >= self._config.quota_cache_refresh_interval_ms / 1000
                    ):
                        self.load_quota_rates()
                except Exception as exc:
                    _log.warning("[ModelScheduler] auto-refresh failed: %s", exc)

        self._refresh_task = asyncio.create_task(_loop())

        def dispose() -> None:
            self.stop_auto_refresh()

        return dispose

    def stop_auto_refresh(self) -> None:
        """Cancel the auto-refresh task if running."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

    def get_all_loads(self) -> list[LoadInfo]:
        """Return load info for all tracked models."""
        return list(self._load_map.values())
