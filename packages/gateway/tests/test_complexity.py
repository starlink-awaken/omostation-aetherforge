"""Tests for llm_gateway.complexity — TaskComplexityScorer."""

from __future__ import annotations

from llm_gateway.complexity import TaskComplexityScorer


class TestTaskComplexityScorer:
    def setup_method(self):
        self.scorer = TaskComplexityScorer()

    def test_simple_chat(self):
        result = self.scorer.estimate(prompt="hello", task="chat")
        assert result.level == "simple"
        assert result.score < 0.34

    def test_simple_greeting(self):
        result = self.scorer.estimate(prompt="hi there!", task="greeting")
        assert result.level == "simple"

    def test_simple_short_prompt(self):
        result = self.scorer.estimate(prompt="ok", task="chat")
        assert result.level == "simple"

    def test_complex_code_review(self):
        long_prompt = "x" * 10000  # ~2500 tokens
        result = self.scorer.estimate(
            prompt=long_prompt,
            task="code-review",
            required_capabilities=["function-calling"],
        )
        assert result.level == "complex"
        assert result.score >= 0.67

    def test_complex_analyze_long(self):
        long_prompt = "analyze this code " * 600  # ~2400 tokens
        result = self.scorer.estimate(prompt=long_prompt, task="analyze")
        assert result.level == "complex"

    def test_medium_code_gen(self):
        prompt = "def fibonacci(n):\n" * 60  # ~300 tokens
        result = self.scorer.estimate(prompt=prompt, task="code-gen")
        assert result.level == "medium"

    def test_medium_summarize(self):
        prompt = "Please summarize this document. " * 20
        result = self.scorer.estimate(prompt=prompt, task="summarize")
        assert result.level in ("simple", "medium")

    def test_unknown_task_defaults_medium(self):
        result = self.scorer.estimate(prompt="do something", task="unknown-task")
        assert result.level in ("simple", "medium")

    def test_empty_prompt_and_task(self):
        result = self.scorer.estimate()
        assert result.level in ("simple", "medium")
        assert 0.0 <= result.score <= 1.0

    def test_high_complexity_caps(self):
        result = self.scorer.estimate(
            prompt="describe this image",
            task="chat",
            required_capabilities=["vision"],
        )
        assert "caps=high" in result.signals[0] or any("caps=high" in s for s in result.signals)

    def test_signals_populated(self):
        result = self.scorer.estimate(prompt="hi", task="chat")
        assert len(result.signals) > 0

    def test_score_bounded(self):
        for prompt in ["", "hi", "x" * 20000]:
            for task in ["", "chat", "analyze"]:
                result = self.scorer.estimate(prompt=prompt, task=task)
                assert 0.0 <= result.score <= 1.0

    def test_production_triage_is_simple(self):
        result = self.scorer.estimate(prompt="route this to the right handler", task="triage")
        assert result.level == "simple"

    def test_production_rpc_is_simple(self):
        result = self.scorer.estimate(prompt="ping", task="rpc")
        assert result.level == "simple"

    def test_production_mcp_is_medium(self):
        prompt = "use the search tool to find information about the project status and recent changes. " * 15
        result = self.scorer.estimate(prompt=prompt, task="mcp")
        assert result.level == "medium"
