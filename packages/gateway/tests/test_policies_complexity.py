"""Tests for ComplexityFilter and ComplexityScore in llm_gateway.policies."""

from __future__ import annotations

from llm_gateway.policies import ComplexityFilter, ComplexityScore
from llm_gateway.types import ModelDescriptor, ModelRequest


def _cheap_model() -> ModelDescriptor:
    return ModelDescriptor(
        id="local-small",
        provider="omlx",
        cost_per_1k_tokens={"input": 0.001, "output": 0.002},
        is_available=True,
    )


def _expensive_model() -> ModelDescriptor:
    return ModelDescriptor(
        id="cloud-pro",
        provider="deepseek",
        cost_per_1k_tokens={"input": 0.05, "output": 0.10},
        is_available=True,
    )


def _free_model() -> ModelDescriptor:
    return ModelDescriptor(
        id="local-free",
        provider="omlx",
        cost_per_1k_tokens=None,
        is_available=True,
    )


class TestComplexityFilter:
    def test_simple_filters_expensive(self):
        f = ComplexityFilter()
        req = ModelRequest(complexity_hint="simple")
        result = f.filter([_cheap_model(), _expensive_model()], req)
        assert len(result) == 1
        assert result[0].id == "local-small"

    def test_simple_keeps_free(self):
        f = ComplexityFilter()
        req = ModelRequest(complexity_hint="simple")
        result = f.filter([_free_model(), _expensive_model()], req)
        assert len(result) == 1
        assert result[0].id == "local-free"

    def test_medium_no_filter(self):
        f = ComplexityFilter()
        req = ModelRequest(complexity_hint="medium")
        models = [_cheap_model(), _expensive_model()]
        result = f.filter(models, req)
        assert len(result) == 2

    def test_complex_no_filter(self):
        f = ComplexityFilter()
        req = ModelRequest(complexity_hint="complex")
        models = [_cheap_model(), _expensive_model()]
        result = f.filter(models, req)
        assert len(result) == 2

    def test_none_hint_no_filter(self):
        f = ComplexityFilter()
        req = ModelRequest()
        models = [_cheap_model(), _expensive_model()]
        result = f.filter(models, req)
        assert len(result) == 2

    def test_custom_cost_ceil(self):
        f = ComplexityFilter(cost_ceil=0.20)
        req = ModelRequest(complexity_hint="simple")
        result = f.filter([_cheap_model(), _expensive_model()], req)
        assert len(result) == 2


class TestComplexityScore:
    def test_simple_biases_cost(self):
        s = ComplexityScore()
        req = ModelRequest(complexity_hint="simple")
        cheap = _cheap_model()
        expensive = _expensive_model()
        assert s.score(cheap, req) > s.score(expensive, req)

    def test_complex_biases_capability(self):
        s = ComplexityScore()
        capable = ModelDescriptor(
            id="cap",
            capabilities=["vision", "function-calling"],
            is_available=True,
        )
        basic = ModelDescriptor(
            id="basic",
            capabilities=[],
            is_available=True,
        )
        req = ModelRequest(
            complexity_hint="complex",
            required_capabilities=["vision", "function-calling"],
        )
        assert s.score(capable, req) > s.score(basic, req)

    def test_medium_balanced(self):
        s = ComplexityScore()
        req = ModelRequest(complexity_hint="medium")
        model = _cheap_model()
        score = s.score(model, req)
        assert 0.0 <= score <= 1.0

    def test_none_hint_uses_balanced_weights(self):
        s = ComplexityScore()
        req = ModelRequest()
        model = _cheap_model()
        score = s.score(model, req)
        assert 0.0 <= score <= 1.0

    def test_score_bounded(self):
        s = ComplexityScore()
        for hint in [None, "simple", "medium", "complex"]:
            req = ModelRequest(complexity_hint=hint)
            for m in [_cheap_model(), _expensive_model(), _free_model()]:
                assert 0.0 <= s.score(m, req) <= 1.0
