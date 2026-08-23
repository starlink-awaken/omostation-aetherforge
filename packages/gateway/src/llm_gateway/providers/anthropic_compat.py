"""AnthropicCompatProvider — 通用 Anthropic API 兼容 Provider。

用于通过 Anthropic 兼容接口访问的第三方 Provider:
  - MiniMax (api.minimaxi.com/anthropic)
  - Zhipu GLM (open.bigmodel.cn/api/anthropic)
  - Kimi (api.kimi.com/coding)
  - 其他 Anthropic 兼容服务

用法::
    from llm_gateway.providers.anthropic_compat import AnthropicCompatProvider
    p = AnthropicCompatProvider(
        name="minimax",
        api_key="sk-xxx",
        base_url="https://api.minimaxi.com/anthropic",
        default_model="MiniMax-M2.7",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from ..provider import LLMProvider, LLMRequest, LLMResponse

_log = logging.getLogger(__name__)


class AnthropicCompatProvider(LLMProvider):
    """通用 Anthropic 兼容 API Provider。

    通过 ``name`` 区分不同的第三方服务。
    """

    def __init__(
        self,
        name: str = "anthropic-compat",
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "claude-3-haiku-20240307",
    ) -> None:
        super().__init__()
        self._name = name
        self._api_key = api_key or ""
        self._base_url = base_url or ""
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._default_model

    def available_models(self) -> list[str]:
        # 2026-08-23: 有 base_url 的网关查询真实清单(longcat/火山方舟实测
        # /v1/models 200, 此前静态只回 default_model 一个 —— 没有 MODEL-BREW
        # yaml 的 anthropic 系引擎因此只能"看见"一个模型)。带 status 字段的
        # 条目(火山方舟)过滤掉 Shutdown 等非 active 状态。查询失败回退旧行为。
        if self._base_url:
            try:
                import httpx

                resp = httpx.get(
                    f"{self._base_url.rstrip('/')}/models",
                    headers=self._auth_headers(),
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = [
                        m["id"]
                        for m in data.get("data", [])
                        if isinstance(m, dict) and m.get("id") and m.get("status") in (None, "", "active")
                    ]
                    if models:
                        return models
            except Exception:
                pass
        return [self._default_model]

    def is_available(self) -> bool:
        return bool(self._api_key) and bool(self._base_url)

    def _auth_headers(self) -> dict[str, str]:
        # 双头发送: 标准 Anthropic 风格 x-api-key + Bearer。longcat/火山方舟
        # 实测只认 Bearer(x-api-key 401), 官方及多数兼容网关两者皆收 ——
        # 多发一个头对任何一家都无害, 少发则直接 401。
        return {
            "x-api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **self._auth_headers(),
        }

    def _build_body(self, request: LLMRequest) -> dict[str, Any]:
        messages = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        body: dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if request.temperature:
            body["temperature"] = request.temperature
        return body

    def _parse_response(self, data: dict[str, Any], model: str) -> LLMResponse:
        content = ""
        input_tokens = 0
        output_tokens = 0

        content_blocks = data.get("content", [])
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    content += block.get("text", "")
        else:
            content = str(content_blocks)

        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        # Anthropic stop_reason -> OpenAI finish_reason 映射: 透传 "end_turn"
        # 这类 Anthropic 词汇会穿透到 /v1/chat/completions 响应体, 严格按
        # OpenAI 协议判 finish_reason 的客户端(如某些 agent 框架只在
        # "stop"/"length" 上继续)会当成未知值。
        stop_reason = data.get("stop_reason", "stop")
        finish_reason = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }.get(stop_reason, "stop")
        return LLMResponse(
            content=content or data.get("content", {}).get("text", ""),
            provider=self._name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        import httpx

        body = self._build_body(request)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self._base_url}/messages",
                    headers=self._build_headers(),
                    json=body,
                )
                resp.raise_for_status()
                return self._parse_response(resp.json(), request.model or self._default_model)
        except Exception as e:
            _log.error("%s generate failed: %s", self._name, e)
            raise

    def complete(self, request: LLMRequest) -> LLMResponse:
        import httpx

        body = self._build_body(request)
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{self._base_url}/messages",
                    headers=self._build_headers(),
                    json=body,
                )
                resp.raise_for_status()
                return self._parse_response(resp.json(), request.model or self._default_model)
        except Exception as e:
            _log.error("%s complete failed: %s", self._name, e)
            raise
