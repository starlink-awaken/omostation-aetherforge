"""OpenAI provider — GPT-4o/turbo/3.5 via openai SDK (optional dependency).

Environment variables:
    OPENAI_API_KEY:  OpenAI API key.

The provider is unavailable when:
- ``api_key`` is empty or ``"MOCK_KEY"``
- The ``openai`` package is not installed
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from ..provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    _with_llm_retry,
)

_log = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI ChatCompletions provider via the ``openai`` SDK (optional dependency)."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def available_models(self) -> list[str]:
        # 本地端点（localhost/127.0.0.1）和 Tailscale（100.x.x.x）：查询真实模型列表
        if self.base_url and (
            any(host in self.base_url for host in ["localhost", "127.0.0.1"]) or self.base_url.startswith("http://100.")
        ):
            try:
                import httpx

                resp = httpx.get(
                    f"{self.base_url.rstrip('/')}/models",
                    headers={"Authorization": "Bearer ignore"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", []) if "id" in m]
                    if models:
                        return models
            except Exception:  # noqa: BLE001
                pass
            return [self.default_model]
        return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o",
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_model: str = default_model
        self.base_url: str | None = base_url
        self._client: Any | None = None
        self._async_client: Any | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        # 本地端点（localhost/127.0.0.1）和 Tailscale（100.x.x.x）不需要 api_key
        if (
            self.base_url
            and any(
                host in self.base_url
                for host in ["localhost", "127.0.0.1"]
                # Tailscale IP range: 100.x.x.x
            )
            or (self.base_url and self.base_url.startswith("http://100."))
        ):
            return True
        if not self._api_key or self._api_key == "MOCK_KEY":
            return False
        try:
            import openai  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            _log.debug("openai SDK not installed — OpenAIProvider unavailable")
            return False

    # ------------------------------------------------------------------
    # Clients (lazy)
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            key = (
                self._api_key or "not-needed"
                if self.base_url
                and (
                    "localhost" in self.base_url
                    or "127.0.0.1" in self.base_url
                    or self.base_url.startswith("http://100.")
                )
                else self._api_key
            )
            self._client = openai.OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            import openai

            key = (
                self._api_key or "not-needed"
                if self.base_url
                and (
                    "localhost" in self.base_url
                    or "127.0.0.1" in self.base_url
                    or self.base_url.startswith("http://100.")
                )
                else self._api_key
            )
            self._async_client = openai.AsyncOpenAI(api_key=key, base_url=self.base_url)
        return self._async_client

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            client = self._get_async_client()
            messages: list[dict] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.extend(request.context)
            messages.append({"role": "user", "content": request.prompt})

            model = request.model or self.default_model
            resp = await client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop_sequences or None,
            )
            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                provider=self.provider_name,
                model=model,
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                finish_reason=choice.finish_reason or "stop",
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("OpenAIProvider.generate failed: %s", exc)
            raise

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            client = self._get_client()
            messages: list[dict] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.extend(request.context)
            messages.append({"role": "user", "content": request.prompt})

            model = request.model or self.default_model
            resp = _with_llm_retry(
                lambda: client.chat.completions.create(  # type: ignore[attr-defined]
                    model=model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop_sequences or None,
                ),
                max_retries=3,
                base_delay=1.0,
            )
            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                provider=self.provider_name,
                model=model,
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                finish_reason=choice.finish_reason or "stop",
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("OpenAIProvider.complete failed: %s", exc)
            raise

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        try:
            client = self._get_async_client()
            messages: list[dict] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.extend(request.context)
            messages.append({"role": "user", "content": request.prompt})

            model = request.model or self.default_model
            stream = await client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:  # noqa: BLE001
            _log.error("OpenAIProvider.stream_generate failed: %s", exc)
            raise
