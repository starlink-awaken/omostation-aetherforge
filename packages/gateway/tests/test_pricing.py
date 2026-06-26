"""Tests for llm_gateway.pricing — PricingRegistry and ModelPrice."""

from __future__ import annotations


class TestModelPrice:
    def test_default_model_price(self):
        from llm_gateway.pricing import ModelPrice

        mp = ModelPrice()
        assert mp.model_id == ""
        assert mp.provider == ""
        assert mp.cost_per_1k_input == 0.0
        assert mp.cost_per_1k_output == 0.0
        assert mp.context_window == 4096
        assert mp.capabilities == []

    def test_model_price_with_data(self):
        from llm_gateway.pricing import ModelPrice

        mp = ModelPrice(
            model_id="gpt-4o",
            provider="openai",
            display_name="GPT-4o",
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            context_window=128000,
            capabilities=["chat", "vision"],
        )
        assert mp.model_id == "gpt-4o"
        assert mp.provider == "openai"
        assert mp.cost_per_1k == {"input": 0.0025, "output": 0.01}

    def test_to_dict(self):
        from llm_gateway.pricing import ModelPrice

        mp = ModelPrice(model_id="gpt-4o", provider="openai", cost_per_1k_input=0.0025, cost_per_1k_output=0.01)
        d = mp.to_dict()
        assert d["model_id"] == "gpt-4o"
        assert d["provider"] == "openai"
        assert d["cost_per_1k_input"] == 0.0025
        assert d["cost_per_1k_output"] == 0.01


class TestPricingRegistry:
    def test_default_pricing_loaded(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        assert reg._loaded is False

        all_models = reg.list_all()
        assert len(all_models) > 0
        assert reg._loaded is True

    def test_get_price_exact_match(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        price = reg.get_price("llama3")
        assert price is not None
        assert price.model_id == "llama3"
        assert price.provider == "ollama"

    def test_get_price_not_found(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        price = reg.get_price("nonexistent-model-xyz")
        assert price is None

    def test_get_cost(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        cost = reg.get_cost("llama3")
        assert cost == {"input": 0.0, "output": 0.0}

    def test_get_cost_not_found(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        cost = reg.get_cost("nonexistent")
        assert cost == {"input": 0.0, "output": 0.0}

    def test_search_by_capability(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        results = reg.search(capability="vision")
        assert len(results) > 0
        for r in results:
            assert "vision" in r.capabilities

    def test_register_override(self):
        from llm_gateway.pricing import ModelPrice, PricingRegistry

        reg = PricingRegistry()
        mp = ModelPrice(model_id="custom-model", provider="custom", cost_per_1k_input=0.001, cost_per_1k_output=0.002)
        reg.register(mp)

        price = reg.get_price("custom-model")
        assert price is not None
        assert price.cost_per_1k_input == 0.001

    def test_get_stats(self):
        from llm_gateway.pricing import PricingRegistry

        reg = PricingRegistry()
        stats = reg.get_stats()
        assert stats["total_models"] > 0
        assert "ollama" in stats["providers"]
