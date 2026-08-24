from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CloudErrorCode(StrEnum):
    """云端 provider 错误分类(治理 P1: 结构化错误码替代 RuntimeError 字符串)。

    与 OmlxcErrorCode(本地语义) 平行; /stats 的 error_breakdown 按此聚合。
    """

    AUTH = "cloud_auth"             # 401/403 凭据失效(死 key)
    RATE_LIMIT = "cloud_rate_limit" # 429 限流
    BUDGET = "cloud_budget"         # 预算拦截触发
    EMPTY = "cloud_empty"           # 空回复(预算耗尽思考段等)
    TIMEOUT = "cloud_timeout"       # 上游超时
    UPSTREAM = "cloud_upstream"     # 其他上游错误


@dataclass
class ProviderConfig:
    """Provider connection configuration."""

    base_url: str = ""
    api_key: str = ""
    timeout: int = 60


@dataclass
class ChatOptions:
    """Options for a chat completion call."""

    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # 透传给下游的额外参数(如 {"reasoning_effort": "none"} 关 thinking)。
    # 各家支持的键不一样, 所以不在这里枚举, 由调用方按目标引擎给。
    extra: dict[str, Any] | None = None


@dataclass
class ChatResult:
    """Result of a chat completion call."""

    id: str = ""
    model: str = ""
    content: str = ""
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None
    # OpenAI 协议形状的工具调用, 空 tuple 表示无。
    tool_calls: tuple[dict[str, Any], ...] = ()


@dataclass
class StreamChunk:
    """A single streaming chunk from a chat completion call."""

    id: str = ""
    model: str = ""
    content: str = ""
    finish_reason: str | None = None
    # 2026-08-23: 流结束块可带 token 用量(provider 层 detailed 流透传),
    # 默认 None 向后兼容 —— 此前真流式路径完全不带 usage, 成本记账失真。
    usage: dict[str, int] | None = None
    # 聚合完成的工具调用(OpenAI 协议形状), provider 层块级聚合后吐出。
    tool_calls: tuple[dict[str, Any], ...] = ()


@dataclass
class ModelDescriptor:
    """Describes a model discovered from a provider."""

    id: str = ""
    name: str = ""
    provider: str = ""
    capabilities: list[str] = field(default_factory=list)
    context_window: int = 4096
    is_available: bool = True
    cost_per_1k_tokens: dict[str, float] | None = None
    avg_latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackRule:
    """A single fallback rule with strategy and timeout."""

    model: str = ""
    """Model name to fall back to."""
    strategy: str = "balanced"
    """Scoring strategy for this fallback level."""
    timeout_ms: int = 30_000
    """Max wait before triggering the next fallback."""
    cooldown_ms: int = 10_000
    """Min time before retrying this fallback after a failure."""


@dataclass
class ModelRoutePolicy:
    """Routing policy for model selection."""

    strategy: str = "balanced"
    priority: list[str] = field(default_factory=list)
    fallback_chain: list[FallbackRule] = field(default_factory=list)
    """Multi-level fallback chain. Each entry: model + strategy + timeout."""


@dataclass
class ModelRequest:
    """A request to the model scheduler."""

    task: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    preferred_provider: str | None = None
    policy: ModelRoutePolicy | None = None
    complexity_hint: str | None = None
    """Complexity level from TaskComplexityScorer: 'simple', 'medium', or 'complex'."""


@dataclass
class ModelSelection:
    """Result of model selection."""

    model: ModelDescriptor
    provider_name: str
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class LoadInfo:
    """Load information for a model."""

    model_id: str = ""
    active_requests: int = 0
    avg_latency_ms: float = 0.0
    last_checked: float = 0.0


@dataclass
class SchedulerConfig:
    """Configuration for the model scheduler."""

    health_check_interval_ms: int = 30_000
    load_window_size: int = 10
    default_policy: str = "balanced"
    cost_weight: float = 0.3
    speed_weight: float = 0.3
    capability_weight: float = 0.4


DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()
