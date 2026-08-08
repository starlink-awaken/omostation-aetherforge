"""Task complexity estimation for dynamic model routing.

Estimates how complex a task is based on three signal dimensions:
  1. Length — token count approximation (len/4)
  2. Task type — keyword matching on the task field
  3. Capability demand — required_capabilities signals

The output drives model selection: simple tasks use cheap local models,
complex tasks get routed to premium cloud models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIMPLE_TOKEN_THRESHOLD = 200
COMPLEX_TOKEN_THRESHOLD = 2000

_SIMPLE_TASKS = frozenset({
    "chat", "greeting", "format", "translate", "complete",
    "classify", "label", "echo", "ping", "list",
})

_MEDIUM_TASKS = frozenset({
    "code-gen", "summarize", "extract", "rewrite", "explain",
    "convert", "parse", "generate", "compose", "draft",
})

_COMPLEX_TASKS = frozenset({
    "analyze", "review", "architect", "debug", "refactor",
    "design", "plan", "audit", "diagnose", "optimize",
    "code-review", "security-review",
})

_HIGH_COMPLEXITY_CAPS = frozenset({
    "vision", "function-calling", "code-interpreter",
    "file-search", "audio", "video",
})


@dataclass
class TaskComplexity:
    level: str = "medium"
    score: float = 0.5
    signals: list[str] = field(default_factory=list)


class TaskComplexityScorer:
    def estimate(
        self,
        prompt: str = "",
        task: str = "",
        required_capabilities: list[str] | None = None,
    ) -> TaskComplexity:
        signals: list[str] = []
        sub_scores: list[float] = []

        length_score = self._length_score(prompt, signals)
        sub_scores.append(length_score)

        task_score = self._task_score(task, signals)
        sub_scores.append(task_score)

        cap_score = self._capability_score(
            required_capabilities or [], signals,
        )
        sub_scores.append(cap_score)

        raw = sum(sub_scores) / len(sub_scores) if sub_scores else 0.5
        score = max(0.0, min(1.0, raw))

        if score < 0.34:
            level = "simple"
        elif score < 0.67:
            level = "medium"
        else:
            level = "complex"

        return TaskComplexity(level=level, score=round(score, 3), signals=signals)

    @staticmethod
    def _length_score(prompt: str, signals: list[str]) -> float:
        if not prompt:
            return 0.3
        approx_tokens = len(prompt) // 4
        if approx_tokens < SIMPLE_TOKEN_THRESHOLD:
            signals.append(f"length=short({approx_tokens}tok)")
            return 0.15
        elif approx_tokens < COMPLEX_TOKEN_THRESHOLD:
            signals.append(f"length=medium({approx_tokens}tok)")
            return 0.5
        else:
            signals.append(f"length=long({approx_tokens}tok)")
            return 0.9

    @staticmethod
    def _task_score(task: str, signals: list[str]) -> float:
        t = task.lower().strip()
        if not t:
            return 0.5
        if t in _SIMPLE_TASKS:
            signals.append(f"task=simple({t})")
            return 0.15
        if t in _MEDIUM_TASKS:
            signals.append(f"task=medium({t})")
            return 0.5
        if t in _COMPLEX_TASKS:
            signals.append(f"task=complex({t})")
            return 0.9
        signals.append(f"task=unknown({t})")
        return 0.5

    @staticmethod
    def _capability_score(
        caps: list[str], signals: list[str],
    ) -> float:
        if not caps:
            return 0.4
        high_demand = [c for c in caps if c in _HIGH_COMPLEXITY_CAPS]
        if high_demand:
            signals.append(f"caps=high({','.join(high_demand)})")
            return 0.85
        signals.append(f"caps=standard({len(caps)})")
        return 0.5
