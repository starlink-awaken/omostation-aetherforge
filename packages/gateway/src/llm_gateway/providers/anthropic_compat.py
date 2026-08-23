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
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..provider import LLMProvider, LLMRequest, LLMResponse

_log = logging.getLogger(__name__)




def _append_user(messages: list[dict[str, Any]], content: object, as_blocks: bool = False) -> None:
    """追加 user turn; 与上一条 user 相邻时合并(部分 Anthropic 兼容网关
    不接受相邻同 role 消息)。"""
    if messages and messages[-1]["role"] == "user":
        prev = messages[-1]["content"]
        blocks = prev if isinstance(prev, list) else ([{"type": "text", "text": prev}] if prev else [])
        if as_blocks and isinstance(content, list):
            blocks.extend(content)
        else:
            import json as _json

            text = content if isinstance(content, str) else _json.dumps(content, ensure_ascii=False)
            if text:
                blocks.append({"type": "text", "text": text})
        messages[-1]["content"] = blocks
        return
    if as_blocks and isinstance(content, list):
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": content})


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

    def _convert_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """OpenAI 格式对话历史(context) → Anthropic messages 协议转换。

        2026-08-23 前的严重缺陷: _build_body 只用 prompt 构造单条 user 消息,
        request.context 整个丢弃 —— Anthropic 系引擎的多轮对话全部失忆,
        且 assistant 的 tool_calls / role=tool 的工具结果无法回传, agent
        链路第二轮就断。本方法做完整转换:
          - assistant.tool_calls → content 里的 tool_use blocks
          - role=tool → 紧随 user turn 里的 tool_result blocks(Anthropic
            协议要求; 连续多条 tool 结果合并进同一 user turn)
          - 连续 user 消息合并(Anthropic 部分兼容网关不接受同 role 相邻)
        """
        import json

        messages: list[dict[str, Any]] = []

        def _text(value: object) -> str:
            if isinstance(value, str):
                return value
            if value is None:
                return ""
            return json.dumps(value, ensure_ascii=False)

        pending_results: list[dict[str, Any]] = []

        def _flush_results(next_role: str | None, next_content: object = None) -> None:
            """把积压的 tool_result 合成一个 user turn(或并入下一条 user)。"""
            nonlocal pending_results
            if not pending_results:
                return
            if next_role == "user":
                blocks = list(pending_results)
                text = _text(next_content)
                if text:
                    blocks.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": blocks})
                pending_results = []
                # 标记: 消费方需跳过原消息(已并入)
                _flush_results.consumed_next = True
            else:
                messages.append({"role": "user", "content": pending_results})
                pending_results = []

        _flush_results.consumed_next = False

        for raw in request.context:
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role") or "user")
            content = raw.get("content")
            _flush_results.consumed_next = False
            if role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(raw.get("tool_call_id") or raw.get("id") or ""),
                        "content": _text(content),
                    }
                )
                continue
            _flush_results(role, content)
            if _flush_results.consumed_next:
                continue
            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                text = _text(content)
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in raw.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tc.get("id") or ""),
                            "name": str(fn.get("name") or ""),
                            "input": args,
                        }
                    )
                messages.append({"role": "assistant", "content": blocks or text})
            else:
                _append_user(messages, _text(content))
        # 尾部残留 tool_result
        if pending_results:
            _append_user(messages, pending_results, as_blocks=True)
        # 本轮 prompt(与尾部 user 合并, 避免相邻同 role)
        _append_user(messages, request.prompt or " ")
        return messages

    def _build_body(self, request: LLMRequest) -> dict[str, Any]:
        messages = self._convert_messages(request)

        body: dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            # Anthropic 协议的 system 是顶层参数, 不是 messages 里的角色
            body["system"] = request.system_prompt
        if request.temperature:
            body["temperature"] = request.temperature
        # extra 里的 tools/tool_choice(OpenAI 协议形状)转换为 Anthropic 格式
        # —— 此前 extra 被整个丢弃, 工具定义到不了引擎, agent 链路断裂。
        if request.extra:
            tools = request.extra.get("tools")
            if isinstance(tools, list) and tools:
                body["tools"] = [
                    {
                        "name": str((t.get("function") or {}).get("name") or ""),
                        "description": str((t.get("function") or {}).get("description") or ""),
                        "input_schema": (t.get("function") or {}).get("parameters") or {"type": "object"},
                    }
                    for t in tools
                    if isinstance(t, dict)
                ]
            choice = request.extra.get("tool_choice")
            if choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
                body["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
        return body

    def _parse_response(self, data: dict[str, Any], model: str) -> LLMResponse:
        content = ""
        input_tokens = 0
        output_tokens = 0

        tool_calls: list[Mapping[str, Any]] = []
        content_blocks = data.get("content", [])
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    # Anthropic tool_use block -> OpenAI 协议形状; input 是已
                    # 解析的对象, arguments 序列化回字符串(OpenAI 协议约定)。
                    import json

                    tool_calls.append(
                        {
                            "id": str(block.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
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
            # 2026-08-23 修复: 原 fallback 是 data.get("content", {}).get("text", "")
            # —— Anthropic 响应的 content 是 block 数组(list), 一旦拼接结果为空
            # (空 content 响应, 如限流/异常路径), 对 list 调 .get 直接
            # AttributeError 把 provider 炸掉, 掩盖真实响应状态。
            content=content,
            provider=self._name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            tool_calls=tuple(tool_calls),
        )

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        """真流式: Anthropic Messages SSE (content_block_delta 逐 token 透传)。

        此前继承基类默认(非流式 generate 聚合后一次性吐), 走 Anthropic 协议
        的全部引擎(deepseek/longcat/火山/minimax/zhipu 等)流式 TTFT 等于
        整个生成时长。SSE 解析只认 text_delta, 其余事件(message_start/
        message_delta/usage)按需后续扩展。
        """
        import json

        import httpx

        body = self._build_body(request)
        body["stream"] = True
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/messages",
                headers=self._build_headers(),
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict) or event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta")
                    if isinstance(delta, dict) and delta.get("type") == "text_delta" and delta.get("text"):
                        yield str(delta["text"])

    async def stream_generate_detailed(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """带 usage/finish_reason 的真流式(Anthropic Messages SSE 全解析)。

        message_start 的 input_tokens + message_delta 的 output_tokens/stop_reason
        聚合在流的结束块上 —— 成本记账(CostTracker)与 OpenAI 流式协议的
        usage 字段都靠它, 此前真流式路径完全不带 usage, 记账失真。
        """
        import json

        import httpx

        from ..provider import LLMStreamEvent

        body = self._build_body(request)
        body["stream"] = True
        input_tokens = 0
        output_tokens = 0
        finish_reason: str | None = None
        pending_tools: dict[int, dict[str, Any]] = {}
        tool_calls: list[Mapping[str, Any]] = []
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/messages",
                headers=self._build_headers(),
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    kind = event.get("type")
                    if kind == "content_block_delta":
                        delta = event.get("delta")
                        if isinstance(delta, dict) and delta.get("type") == "text_delta" and delta.get("text"):
                            yield LLMStreamEvent(text=str(delta["text"]))
                        elif (
                            isinstance(delta, dict)
                            and delta.get("type") == "input_json_delta"
                            and event.get("index") in pending_tools
                        ):
                            pending_tools[int(event["index"])]["json"] += str(delta.get("partial_json") or "")
                    elif kind == "content_block_start":
                        block = event.get("content_block")
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            # 工具调用块开始: 记 id/name, arguments 分片在后续
                            # input_json_delta 里增量到达, content_block_stop 时聚合完成。
                            pending_tools[int(event.get("index") or 0)] = {
                                "id": str(block.get("id") or ""),
                                "name": str(block.get("name") or ""),
                                "json": "",
                            }
                    elif kind == "content_block_stop":
                        idx = event.get("index")
                        if idx in pending_tools:
                            spec = pending_tools.pop(int(idx))
                            tool_calls.append(
                                {
                                    "id": spec["id"],
                                    "type": "function",
                                    "function": {"name": spec["name"], "arguments": spec["json"] or "{}"},
                                }
                            )
                    elif kind == "message_start":
                        message = event.get("message")
                        if isinstance(message, dict):
                            usage = message.get("usage")
                            if isinstance(usage, dict):
                                input_tokens = int(usage.get("input_tokens") or 0)
                    elif kind == "message_delta":
                        delta = event.get("delta")
                        if isinstance(delta, dict) and delta.get("stop_reason"):
                            finish_reason = {
                                "end_turn": "stop",
                                "stop_sequence": "stop",
                                "max_tokens": "length",
                                "tool_use": "tool_calls",
                            }.get(str(delta["stop_reason"]), "stop")
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            output_tokens = int(usage.get("output_tokens") or 0)
                            # longcat 等兼容实现 message_start.usage 为空, 真实
                            # input_tokens 在 message_delta 里一并给出(实测)。
                            if usage.get("input_tokens"):
                                input_tokens = int(usage["input_tokens"])
        # 异常容错: content_block_stop 未到的残留 pending 一并补齐
        for spec in pending_tools.values():
            tool_calls.append(
                {
                    "id": spec["id"],
                    "type": "function",
                    "function": {"name": spec["name"], "arguments": spec["json"] or "{}"},
                }
            )
        yield LLMStreamEvent(
            finish_reason="tool_calls" if tool_calls else (finish_reason or "stop"),
            usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                   "total_tokens": input_tokens + output_tokens},
            tool_calls=tuple(tool_calls),
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
