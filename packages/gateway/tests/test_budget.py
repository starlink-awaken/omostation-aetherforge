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

    def test_estimate_cost_resolves_fully_qualified_engine_id(self):
        """网关内部传的是 "ENG-XXX-CLOUD/model" 全限定形式, 此前不拆分/不带
        provider 精确匹配, PricingRegistry(按"{短provider}/{裸model}"建索引)
        恒对不上 —— deepseek 等真实计费 provider 的 estimate_cost 一直恒 0,
        budgets 表月限配置形同虚设(2026-08-23 实测发现)。"""
        cost = estimate_cost("ENG-DEEPSEEK-CLOUD/deepseek-chat", input_tokens=1000, output_tokens=500)
        # deepseek-chat: cost_per_1k_input=0.00014, cost_per_1k_output=0.00028
        assert cost == pytest.approx(1000 / 1000 * 0.00014 + 500 / 1000 * 0.00028)

    def test_estimate_cost_free_tier_engine_stays_zero(self):
        """免费层/扁平订阅引擎(longcat/opencode-go 等 cost_multiplier=0, SSOT
        明确标注)本来就该是 0 成本 —— 修复不能把"真实无定价数据"和"设计上
        免费"这两种情况混为一谈, 免费引擎不该被误判出非零成本。"""
        cost = estimate_cost("ENG-LONGCAT-CLOUD/LongCat-2.0", input_tokens=1000, output_tokens=500)
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
