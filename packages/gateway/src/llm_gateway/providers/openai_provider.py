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
    LLMStreamEvent,
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
        # 2026-08-23: 公网 OpenAI 兼容聚合网关(openrouter/opencode-go/siliconflow
        # 等)此前无差别拿到硬编码 gpt 静态清单 —— 对这些网关是错的模型名
        # (opencode-go 真实清单是 glm/kimi/minimax 系, 29 个)。有真实 api_key
        # 的远端现在用真实 key 查真实清单; 静态清单只作为查询失败时的回退。
        if self.base_url:
            headers = {"Authorization": "Bearer ignore"}
            if self._api_key and self._api_key != "MOCK_KEY":
                headers["Authorization"] = f"Bearer {self._api_key}"
            try:
                import httpx

                resp = httpx.get(
                    f"{self.base_url.rstrip('/')}/models",
                    headers=headers,
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", []) if "id" in m]
                    if models:
                        return models
            except Exception:
                pass
            if any(host in self.base_url for host in ["localhost", "127.0.0.1"]) or self.base_url.startswith(
                "http://100."
            ):
                return [self.default_model]
            return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
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
            # 本地 unix socket(omlxc UDS: unix://omlxc/api/v1)同样无需 api_key —
            # 2026-08-25: 此前 unix:// 不在白名单, ENG-OMLX-LOCAL 被判"无凭据
            # 不可用" → discover 准入闸静默归零 23 个本地模型(总闸下一层病根)
            or (self.base_url and self.base_url.startswith("unix://"))
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
            import httpx

            if self.base_url and self.base_url.startswith("unix://"):
                # 本地 unix socket(omlxc UDS: unix://omlxc/api/v1) — OpenAI SDK
                # 不认 unix:// base_url, 用 httpx UDS transport + 虚拟 http base_url
                # (2026-08-25: 本地直连最后一公里; socket 路径复用 default_omlxc_socket;
                # 实测 UDS 通道 18 模型全通, chat 达 daemon 业务层 "no capacity")
                from llm_gateway.omlxc_client import default_omlxc_socket

                socket_path = default_omlxc_socket()
                http_client = httpx.Client(
                    base_url="http://omlxc/api/v1", transport=httpx.HTTPTransport(uds=str(socket_path))
                )
                self._client = openai.OpenAI(
                    api_key="not-needed", base_url="http://omlxc/api/v1", http_client=http_client
                )
                return self._client

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
            import httpx

            if self.base_url and self.base_url.startswith("unix://"):
                from llm_gateway.omlxc_client import default_omlxc_socket

                socket_path = default_omlxc_socket()
                async_http_client = httpx.AsyncClient(
                    base_url="http://omlxc/api/v1",
                    transport=httpx.AsyncHTTPTransport(uds=str(socket_path)),
                )
                self._async_client = openai.AsyncOpenAI(
                    api_key="not-needed", base_url="http://omlxc/api/v1", http_client=async_http_client
                )
                return self._async_client

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
                # extra_body 原样进 JSON body —— LM Studio 的 reasoning_effort
                # 这类非标准字段走这里, SDK 不认识也不会拦。
                **({"extra_body": request.extra} if request.extra else {}),
            )
            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                provider=self.provider_name,
                model=model,
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                finish_reason=choice.finish_reason or "stop",
                tool_calls=tuple(
                    {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {
                            "name": (tc.function.name if tc.function else "") or "",
                            "arguments": (tc.function.arguments if tc.function else None) or "{}",
                        },
                    }
                    for tc in (choice.message.tool_calls or [])
                ),
            )
        except Exception as exc:
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
        except Exception as exc:
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
                **({"extra_body": request.extra} if request.extra else {}),
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            _log.error("OpenAIProvider.stream_generate failed: %s", exc)
            raise

    async def stream_generate_detailed(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """带 usage/finish_reason 的真流式(openai SDK stream_options)。

        OpenAI 系引擎(openrouter/siliconflow/opencode-go 等)此前流式走基类
        默认包装(真流式但无元数据) —— usage 记账与终止块 finish 全缺。
        兼容性防御: 部分 OpenAI 兼容网关不认 stream_options(400), 检测到
        后降级为不带该参数重试(usage 缺失可容忍, 流式内容不丢)。
        """
        try:
            client = self._get_async_client()
            messages: list[dict] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.extend(request.context)
            messages.append({"role": "user", "content": request.prompt})

            model = request.model or self.default_model
            common = dict(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                # extra(tools 等)经 extra_body 顶层合并进请求体, 与 generate 对齐
                **({"extra_body": request.extra} if request.extra else {}),
            )
            try:
                stream = await client.chat.completions.create(  # type: ignore[attr-defined]
                    **common, stream_options={"include_usage": True}
                )
            except Exception as exc:
                if type(exc).__name__ == "BadRequestError" or "stream_options" in str(exc):
                    _log.debug("stream_options unsupported by gateway, streaming without usage")
                    stream = await client.chat.completions.create(**common)  # type: ignore[attr-defined]
                else:
                    raise
            finish_reason: str | None = None
            pending_calls: dict[int, dict[str, Any]] = {}
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    # include_usage 的最后块: choices 为空, usage 齐全
                    yield LLMStreamEvent(
                        finish_reason="tool_calls" if pending_calls else (finish_reason or "stop"),
                        usage={
                            "prompt_tokens": usage.prompt_tokens or 0,
                            "completion_tokens": usage.completion_tokens or 0,
                            "total_tokens": usage.total_tokens or 0,
                        },
                        tool_calls=tuple(pending_calls[i] for i in sorted(pending_calls)),
                    )
                    continue
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta and delta.content:
                        yield LLMStreamEvent(text=delta.content)
                    if delta and getattr(delta, "tool_calls", None):
                        # OpenAI 流式工具调用分片: 按 index 聚合, id/name 只在
                        # 首片带, arguments 增量拼接, 流尾统一吐完整调用。
                        for tc in delta.tool_calls:
                            idx = tc.index or 0
                            entry = pending_calls.setdefault(
                                idx,
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                            )
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function and tc.function.name:
                                entry["function"]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments
            # 降级路径兜底: 网关不认 stream_options(无 usage 终块)时, 聚合中的
            # 工具调用在流尾统一吐出, 不因降级丢失。
            if pending_calls:
                yield LLMStreamEvent(
                    finish_reason="tool_calls",
                    tool_calls=tuple(pending_calls[i] for i in sorted(pending_calls)),
                )
        except Exception as exc:
            _log.error("OpenAIProvider.stream_generate_detailed failed: %s", exc)
            raise
