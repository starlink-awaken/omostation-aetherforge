"""Tests for llm_gateway.budget — BudgetExhaustedError, estimate_cost, check_budget_limit."""

from __future__ import annotations

import pytest
from llm_gateway.budget import BudgetExhaustedError, estimate_cost


class TestBudgetExhaustedError:
    def test_basic_error(self):
        err = BudgetExhaustedError("Budget exceeded", spent=5.0, cap=2.0)
        assert str(err) == "Budget exceeded"
        assert err.spent == 5.0
        assert err.cap == 2.0
        assert err.task_id is None

    def test_with_task_id(self):
        err = BudgetExhaustedError("Budget exceeded", spent=5.0, cap=2.0, task_id="task-123")
        assert err.task_id == "task-123"


class TestEstimateCost:
    def test_estimate_cost_known_model(self):
        # human-expert is in built-in defaults: cost_per_1k_input=999, cost_per_1k_output=999
        cost = estimate_cost("human-expert", input_tokens=1000, output_tokens=500)
        # input:  1000/1000 * 999 = 999.0
        # output:  500/1000 * 999 = 499.5
        assert cost == pytest.approx(1498.5)

    def test_estimate_cost_unknown_model(self):
        cost = estimate_cost("nonexistent-model-xyz", input_tokens=1000, output_tokens=500)
        assert cost == 0.0


class TestCheckBudgetLimit:
    def test_budget_not_exceeded(self):
        from llm_gateway.budget import check_budget_limit

        # llama3 is free (ollama), cost is 0, so never exceeds budget
        check_budget_limit(model_id="llama3", input_tokens=100, local_budget_limit=0.01)

    def test_budget_exceeded_raises(self):
        from llm_gateway.budget import BudgetExhaustedError, check_budget_limit

        # human-expert model costs 999 per 1k tokens, so 1000 tokens = 999
        # which exceeds any reasonable local budget
        with pytest.raises(BudgetExhaustedError) as exc_info:
            check_budget_limit(
                model_id="human-expert",
                input_tokens=1000,
                local_budget_limit=0.01,
            )
        assert exc_info.value.cap == 0.01
        assert exc_info.value.spent > 0.01
