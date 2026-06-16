from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .providers.base import BaseLLMProvider
from .registry import ModelRegistry
from .types import ChatOptions, ChatResult, ModelDescriptor, ModelRequest, ModelRoutePolicy, ModelSelection
from .scheduler import ModelScheduler


REGISTRY_DATA_DIR = Path(__file__).resolve().parent / "registry_data"
MODELS_PATH = REGISTRY_DATA_DIR / "models.yaml"
ROLE_ROUTES_PATH = REGISTRY_DATA_DIR / "role_routes.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_registry_models(models_path: Path | None = None) -> list[dict[str, Any]]:
    payload = _load_yaml(models_path or MODELS_PATH)
    return list(payload.get("models", []))


def get_registry_model_record(model_id: str, models_path: Path | None = None) -> dict[str, Any] | None:
    target = model_id if "/" in model_id else None
    for record in get_registry_models(models_path):
        record_id = f"{record['provider']}/{record['model_id']}"
        if target == record_id or model_id == record["model_id"]:
            return record
    return None


def estimate_model_cost(model_id: str, input_tokens: int, output_tokens: int, models_path: Path | None = None) -> float:
    record = get_registry_model_record(model_id, models_path=models_path)
    if not record:
        return 0.0
    cost = record.get("cost", {})
    input_rate = float(cost.get("input_per_1k", 0.0))
    output_rate = float(cost.get("output_per_1k", 0.0))
    return round((input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate, 6)


class StaticRegistryProvider(BaseLLMProvider):
    def __init__(self, provider_name: str, models: list[dict[str, Any]]) -> None:
        self._name = provider_name
        self._models = list(models)

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return "static"

    async def discover(self) -> list[ModelDescriptor]:
        descriptors: list[ModelDescriptor] = []
        for record in self._models:
            descriptors.append(
                ModelDescriptor(
                    id=f"{record['provider']}/{record['model_id']}",
                    name=record["model_id"],
                    provider=record["provider"],
                    capabilities=list(record.get("capabilities", [])),
                    context_window=int(record.get("context", 4096)),
                    is_available=True,
                    cost_per_1k_tokens={
                        "input": float(record.get("cost", {}).get("input_per_1k", 0.0)),
                        "output": float(record.get("cost", {}).get("output_per_1k", 0.0)),
                    },
                    avg_latency_ms=_latency_to_ms(record.get("latency")),
                    metadata={
                        "source": "registry_data",
                        "latency_label": record.get("latency"),
                    },
                )
            )
        return descriptors

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        raise RuntimeError(
            f"Static registry provider {self._name} only supports discovery, not live chat."
        )


def _latency_to_ms(label: str | None) -> float:
    mapping = {
        "fast": 900.0,
        "medium": 2500.0,
        "slow": 6000.0,
    }
    return mapping.get(label or "", 2500.0)


def load_registry_data(registry: ModelRegistry, models_path: Path | None = None) -> int:
    models = get_registry_models(models_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in models:
        grouped.setdefault(record["provider"], []).append(record)

    for provider_name, provider_models in grouped.items():
        registry.register(StaticRegistryProvider(provider_name, provider_models))
    return len(models)


def load_role_routes(role_routes_path: Path | None = None) -> dict[str, dict[str, Any]]:
    payload = _load_yaml(role_routes_path or ROLE_ROUTES_PATH)
    return payload.get("routes", {})


def _safe_run(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def build_static_registry() -> tuple[ModelRegistry, int]:
    registry = ModelRegistry()
    seeded = load_registry_data(registry)
    _safe_run(registry.refresh())
    return registry, seeded


def _policy_name(preferred_mode: str | None) -> str:
    mapping = {
        "balanced": "balanced",
        "cost": "cost-first",
        "reasoning": "capability-first",
        "long_context": "capability-first",
    }
    return mapping.get(preferred_mode or "", "balanced")


def route_role_request(
    role: str,
    required_capabilities: list[str] | None = None,
    registry: ModelRegistry | None = None,
    routes: dict[str, dict[str, Any]] | None = None,
) -> ModelSelection | None:
    registry = registry or build_static_registry()[0]
    routes = routes or load_role_routes()
    route = routes.get(role)
    if not route:
        return None

    scheduler = ModelScheduler(registry)
    policy = ModelRoutePolicy(
        strategy=_policy_name(route.get("preferred_mode")),
        priority=list(route.get("primary_models", [])),
        fallback_chain=list(route.get("fallback_models", [])),
    )
    request = ModelRequest(
        task=role,
        required_capabilities=list(required_capabilities or []),
        policy=policy,
    )
    import asyncio

    return asyncio.run(scheduler.select_model(request, policy=policy))
