"""PricingRegistry — 模型定价注册与查询。

数据来源优先级:
  1. L0 M1 YAML (model/pricing 命名空间)
  2. 内置默认值 (覆盖主流模型)
  3. 用户自定义 (aetherforge.yaml / credentials DB)

用法::

    from llm_gateway.pricing import PricingRegistry

    pricing = PricingRegistry()
    cost = pricing.get_cost("gpt-4o")  # → {"input": 0.0025, "output": 0.01}
    models = pricing.search(capability="vision")  # → 支持 vision 的模型列表
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aetherforge._paths import M1_MODEL_DIR

_log = logging.getLogger(__name__)


@dataclass
class ModelPrice:
    """Pricing and capability info for a single model."""

    model_id: str = ""
    provider: str = ""
    display_name: str = ""
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    context_window: int = 4096
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_per_1k(self) -> dict[str, float]:
        return {"input": self.cost_per_1k_input, "output": self.cost_per_1k_output}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name or self.model_id,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "context_window": self.context_window,
            "capabilities": self.capabilities,
        }


# ── Built-in default pricing (covers models NOT in M1 YAML) ─────────────────

_DEFAULT_PRICING: list[dict[str, Any]] = [
    # Ollama (local models — discovered at runtime, hardcoded for pricing only)
    {"model_id": "llama3", "provider": "ollama", "cost_in": 0.0, "cost_out": 0.0, "ctx": 8192, "caps": ["chat"]},
    {"model_id": "llama3.1", "provider": "ollama", "cost_in": 0.0, "cost_out": 0.0, "ctx": 131072, "caps": ["chat"]},
    {
        "model_id": "qwen3.5:9b",
        "provider": "ollama",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "ctx": 262144,
        "caps": ["chat", "tools", "thinking"],
    },
    {
        "model_id": "qwen3.5:4b",
        "provider": "ollama",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "ctx": 262144,
        "caps": ["chat", "vision", "tools", "thinking"],
    },
    # HITL (human-in-the-loop — special, never in M1)
    {
        "model_id": "human-expert",
        "provider": "hitl",
        "cost_in": 999.0,
        "cost_out": 999.0,
        "ctx": 999999,
        "caps": ["chat", "human"],
    },
]


class PricingRegistry:
    """Central registry for model pricing data.

    Thread-safe. Loads from multiple sources with priority.
    """

    def __init__(self) -> None:
        self._prices: dict[str, ModelPrice] = {}  # key: f"{provider}/{model_id}"
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_defaults()
        self._load_m1_yamls()
        self._loaded = True

    def _load_defaults(self) -> None:
        """Load built-in default pricing."""
        for entry in _DEFAULT_PRICING:
            mp = ModelPrice(
                model_id=entry["model_id"],
                provider=entry["provider"],
                cost_per_1k_input=entry["cost_in"],
                cost_per_1k_output=entry["cost_out"],
                context_window=entry.get("ctx", 4096),
                capabilities=entry.get("caps", []),
                display_name=entry.get("display_name", entry["model_id"]),
            )
            key = f"{mp.provider}/{mp.model_id}"
            self._prices[key] = mp

    def _load_m1_yamls(self) -> None:
        """Load pricing from L0 M1 model/ YAMLs if available."""
        if not M1_MODEL_DIR.is_dir():
            return
        import yaml

        # Load MODEL-PRICING-*.yaml (legacy format)
        for yaml_file in M1_MODEL_DIR.glob("MODEL-PRICING-*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                entries = data if isinstance(data, list) else [data]
                for entry in entries:
                    self._load_pricing_entry(entry)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Failed to load pricing YAML %s: %s", yaml_file, exc)

        # Load MODEL-BREW-*.yaml (new format with engine_ref + models[])
        for yaml_file in sorted(M1_MODEL_DIR.glob("MODEL-BREW-*.yaml")):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if not data or "models" not in data:
                    continue
                # Derive provider from engine_ref or filename
                engine_ref = data.get("engine_ref", "")
                if engine_ref and engine_ref != "ENG-CC-SWITCH":
                    # Clean engine names: "ENG-ANTHROPIC-CLOUD" → "anthropic"
                    provider = engine_ref.replace("ENG-", "").split("-")[0].lower()
                elif engine_ref == "ENG-CC-SWITCH":
                    # CC-SWITCH is a proxy hosting multiple providers;
                    # use filename as the actual provider name
                    provider = yaml_file.stem.replace("MODEL-BREW-", "").lower()
                else:
                    provider = ""
                for model in data["models"]:
                    entry = {
                        "model_id": model.get("model_id", ""),
                        "provider": provider,
                        "cost_per_1k_input": model.get("cost_per_1k_input", 0),
                        "cost_per_1k_output": model.get("cost_per_1k_output", 0),
                        "context_window": model.get("context_window", 128000),
                        "capabilities": model.get("capabilities", ["chat"]),
                        "display_name": model.get("display_name", ""),
                    }
                    self._load_pricing_entry(entry)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Failed to load model YAML %s: %s", yaml_file, exc)

    def _load_pricing_entry(self, entry: dict) -> None:
        """Load a single pricing entry into the registry."""
        mp = ModelPrice(
            model_id=entry.get("model_id", ""),
            provider=entry.get("provider", ""),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", entry.get("cost_in", 0))),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", entry.get("cost_out", 0))),
            context_window=int(entry.get("context_window", entry.get("ctx", 4096))),
            capabilities=entry.get("capabilities", entry.get("caps", [])),
            display_name=entry.get("display_name", ""),
            metadata=entry,
        )
        if mp.model_id and mp.provider:
            key = f"{mp.provider}/{mp.model_id}"
            self._prices[key] = mp

    # ── Public API ───────────────────────────────────────────────────────────

    def get_price(self, model_id: str, provider: str = "") -> ModelPrice | None:
        """Get pricing for a specific model.

        Args:
            model_id: Model identifier (e.g. ``"gpt-4o"``).
            provider: Optional provider filter.

        Returns:
            ``ModelPrice`` or ``None`` if not found.
        """
        self._ensure_loaded()
        # Try exact match first
        if provider:
            key = f"{provider}/{model_id}"
            if key in self._prices:
                return self._prices[key]
        # Fallback: search by model_id only
        for key, mp in self._prices.items():
            if mp.model_id == model_id or key.endswith(f"/{model_id}"):
                return mp
        return None

    def get_cost(self, model_id: str, provider: str = "") -> dict[str, float]:
        """Get cost per 1K tokens for a model.

        Returns ``{"input": 0.0, "output": 0.0}`` if not found.
        """
        mp = self.get_price(model_id, provider)
        if mp:
            return mp.cost_per_1k
        return {"input": 0.0, "output": 0.0}

    def search(self, capability: str = "", provider: str = "") -> list[ModelPrice]:
        """Search models by capability and/or provider."""
        self._ensure_loaded()
        results = []
        for mp in self._prices.values():
            if provider and mp.provider != provider:
                continue
            if capability and capability not in mp.capabilities:
                continue
            results.append(mp)
        return sorted(results, key=lambda x: x.cost_per_1k_input)

    def list_all(self) -> list[ModelPrice]:
        """List all known models with pricing."""
        self._ensure_loaded()
        return sorted(self._prices.values(), key=lambda x: (x.provider, x.model_id))

    def register(self, mp: ModelPrice) -> None:
        """Register (or override) a model's pricing."""
        key = f"{mp.provider}/{mp.model_id}"
        self._prices[key] = mp

    def get_stats(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "total_models": len(self._prices),
            "providers": list({mp.provider for mp in self._prices.values()}),
        }
