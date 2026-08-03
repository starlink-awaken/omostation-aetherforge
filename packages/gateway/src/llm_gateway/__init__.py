"""LLM Gateway v0.5 — unified LLM provider abstraction layer + ModelGateway.

Stable Public API
=================

Core types (always available):
  * :class:`LLMProvider` — abstract base class for providers
  * :class:`LLMRequest` — request dataclass
  * :class:`LLMResponse` — response dataclass (includes token tracking)

Provider discovery:
  * :func:`detect_backends` — auto-detect available LLM providers
  * :func:`create_provider` — create a specific provider by name

Unified Gateway (v0.5 新增):
  * :class:`ModelGateway` — 统一模型网关 (唯一入口)
  * :class:`GatewayRequest` — 网关请求
  * :class:`GatewayResponse` — 网关响应
  * :class:`GatewayConfig` — 网关配置

Error types:
  * :exc:`LLMError` — base exception
  * :exc:`LLMRetryExhaustedError` — retry exhausted

Tool types:
  * :class:`ToolSchema` — tool definition schema
  * :class:`ToolCall` — tool call in request
  * :class:`ToolResult` — tool result in response

Backward Compatibility
======================

For minerva/ssot/ontoderive consumers, use:
  * :mod:`llm_gateway.compat` — legacy provider aliases

Version: 0.5.0
"""

import builtins  # noqa: F401

from .detection import create_provider, detect_backends
from .gateway import (
    GatewayConfig,
    GatewayRequest,
    GatewayResponse,
    ModelGateway,
    get_gateway,
    is_sensitive,
    reset_gateway,
    run_async,
    strip_thinking,
)
from .provider import (
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRetryExhaustedError,
    MockLLMProvider,
    NoneProvider,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from .providers.anthropic_provider import AnthropicProvider
from .providers.deepseek_provider import DeepSeekProvider
from .providers.gemini_provider import GeminiProvider
from .providers.hitl_provider import HitlLLMProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .ssot_loader import load_ssot_models

__version__ = "0.5.0"

__all__ = (
    # ── Stable public API ──
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMError",
    "LLMRetryExhaustedError",
    "ToolSchema",
    "ToolCall",
    "ToolResult",
    "create_provider",
    "detect_backends",
    "load_ssot_models",
    # ── ModelGateway (v0.5) ──
    "ModelGateway",
    "GatewayRequest",
    "GatewayResponse",
    "GatewayConfig",
    "get_gateway",
    "reset_gateway",
    "run_async",
    # ── K1 SSOT ──
    "is_sensitive",
    "strip_thinking",
    # ── Concrete providers ──
    "AnthropicProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "HitlLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    # ── Testing ──
    "MockLLMProvider",
    "NoneProvider",
)
