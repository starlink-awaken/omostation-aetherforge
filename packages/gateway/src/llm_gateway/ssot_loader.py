"""SSOT M1 model loader — reads compute_engine + model definitions + credentials.

Architecture:
    M1 SSOT has three layers:
    1. compute_engine/ — "where" (endpoint, protocol)
    2. model/ — "what" (model_id, pricing, capabilities)
    3. CredentialsManager — "how" (API keys, base_url)

    SSOTProviderAdapter merges all three into a unified provider.
"""

import asyncio
import glob
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import yaml

from .provider import LLMRequest
from .providers.anthropic_compat import AnthropicCompatProvider
from .providers.base import BaseLLMProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .registry import ModelRegistry
from .types import ChatOptions, ChatResult, ModelDescriptor, StreamChunk

_log = logging.getLogger(__name__)


def _load_model_defs(m1_model_dir: str) -> dict[str, list[dict]]:
    """Load M1 model/ YAMLs, group by engine_ref.

    Returns {engine_ref: [model_dict, ...]}
    """
    engine_models: dict[str, list[dict]] = {}
    pattern = os.path.join(m1_model_dir, "MODEL-BREW-*.yaml")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                continue
            models = data.get("models", [])
            engine_ref = data.get("engine_ref", "")
            if not engine_ref or not models:
                continue
            engine_models.setdefault(engine_ref, []).extend(models)
        except Exception as e:
            _log.warning("Failed to load model defs from %s: %s", filepath, e)
    return engine_models


def _get_credentials_for(provider_name: str) -> dict | None:
    """Look up credentials from CredentialsManager by provider name."""
    try:
        from .credentials import CredentialsManager

        cm = CredentialsManager()

        # Find the actual provider name in credentials table
        def _find_key(prov: str) -> str | None:
            """Try direct and aliased provider names."""
            for name in [prov, _PROVIDER_ALIASES.get(prov, prov)]:
                key = cm.get_key(name)
                if key:
                    return key
            return None

        api_key = _find_key(provider_name)
        if not api_key:
            return None

        # 2026-08-23 修复两处: (1) 原来取的是 note 字段(复制粘贴错误), base_url
        # 实际拿到 "from cc-switch: xxx" 这种字符串, YAML 里没写 base_url 的引擎
        # 会静默拿到一个非法 URL; (2) list_keys 用的是未解析别名的原始名, 别名
        # 命中 key 时这里反而取不到行。两处都改成与 _find_key 相同的解析顺序。
        base_url = ""
        for name in [provider_name, _PROVIDER_ALIASES.get(provider_name, provider_name)]:
            rows = cm.list_keys(name)
            if rows:
                base_url = str(rows[0].get("base_url") or "")
                break

        return {"api_key": api_key, "base_url": base_url}
    except Exception as e:
        _log.debug("Credentials lookup failed for %s: %s", provider_name, e)
    return None


# Map compute_engine name tokens to credential provider names
_PROVIDER_ALIASES: dict[str, str] = {
    "anthropic": "claude",
    "deepseek": "deepseek",
    "openai": "openai",
    "minimax": "minimax",
    "gemini": "gemini",
    "openrouter": "openrouter",
    "zhipu": "zhipu_glm",
    "siliconflow": "siliconflow",
    "nvidia": "nvidia",
    "longcat": "longcat",
    "kimi": "kimi_for_coding",
    # 2026-08-23: ENG-OPENCODE-GO 引擎名解析出的 token 是 "opencode",
    # credentials.db 里的 provider 名是 "opencode-go"(cc-switch 同步写入)
    "opencode": "opencode-go",
    # 2026-08-23: ENG-VOLCANO-CLOUD 引擎 token 是 "volcano",
    # credentials.db 里的 provider 名是中文 "火山agentplan"(cc-switch 同步写入)
    "volcano": "火山agentplan",
}


class SSOTProviderAdapter(BaseLLMProvider):
    """Adapter to map an L0 M1 compute_engine config + model defs to a BaseLLMProvider."""

    def __init__(
        self,
        m1_config: dict,
        model_defs: list[dict] | None = None,
    ):
        self._config = m1_config
        self._name = m1_config.get("id", "unknown")
        self._type = m1_config.get("engine_type", "unknown")
        self.base_url = m1_config.get("base_url")
        self.cost_multiplier = float(m1_config.get("cost_multiplier", 1.0))
        self.protocols = m1_config.get("supported_protocols", [])
        self.request_defaults = dict(m1_config.get("request_defaults") or {})
        self._model_defs = model_defs or []
        self._credentials: dict | None = None
        self._underlying = None

        # Try to inject credentials from CredentialsManager
        self._cred_token = self._name.replace("ENG-", "").split("-")[0].lower()
        cred = _get_credentials_for(self._cred_token)
        if cred:
            self._credentials = cred
            # Override base_url from credentials if not set in config
            if not self.base_url and cred.get("base_url"):
                self.base_url = cred["base_url"]

        # Create underlying provider
        kwargs: dict[str, Any] = {"base_url": self.base_url}
        if self._credentials and self._credentials.get("api_key"):
            kwargs["api_key"] = self._credentials["api_key"]

        if "openai" in self.protocols:
            self._underlying = OpenAIProvider(**kwargs)
            self._provider_type = "openai"
        elif "anthropic" in self.protocols:
            self._underlying = AnthropicCompatProvider(**kwargs)
            self._provider_type = "anthropic"
        elif self._type == "local_daemon" or "ollama" in self.protocols:
            self._underlying = OllamaProvider(base_url=self.base_url)  # type: ignore[reportArgumentType]
            self._provider_type = "ollama"
        else:
            self._provider_type = "unknown"

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return self._provider_type

    async def discover(self) -> list[ModelDescriptor]:
        """Discover models: use M1 model defs first, fall back to underlying API."""
        # 准入闸: 凭据无效(如单 key 过期被复验判死)的引擎不注册任何模型
        # —— 静态 model_defs 会让死 key 引擎的模型永远留在 registry 被
        # scheduler 反复选中(实测 k2p5: 过期 key 每轮 401 白烧一次)。
        if self._underlying is None or not self._underlying.is_available():
            return []
        if self._model_defs:
            descriptors = []
            for m in self._model_defs:
                cost = {
                    "input": m.get("cost_per_1k_input", self.cost_multiplier),
                    "output": m.get("cost_per_1k_output", self.cost_multiplier),
                }
                descriptors.append(
                    ModelDescriptor(
                        id=f"{self._name}/{m['model_id']}",
                        name=m["model_id"],
                        provider=self._name,
                        capabilities=m.get("capabilities", ["chat"]),
                        cost_per_1k_tokens=cost,
                    )
                )
            return descriptors

        # Fall back to underlying provider API
        if not self._underlying:
            return []
        # available_models 是同步 SDK 调用。直接在 async discover 里跑会把整个
        # event loop 卡死，registry.gather/timeout 形同虚设；放到线程后各节点才
        # 能真正并发发现，离线 Y7000P 也不会拖住 MBP 首次请求。
        model_names = await asyncio.to_thread(self._underlying.available_models)
        cost = {"input": self.cost_multiplier, "output": self.cost_multiplier}
        return [
            ModelDescriptor(
                id=f"{self._name}/{m}",
                name=m,
                provider=self._name,
                capabilities=["chat"],
                cost_per_1k_tokens=cost,
            )
            for m in model_names
        ]

    def _build_request(self, model: str, messages: list[dict[str, Any]], options: ChatOptions | None) -> LLMRequest:
        real_model = model.split("/", 1)[-1] if "/" in model else model

        context = list(messages)
        prompt = ""
        if context and context[-1].get("role") == "user":
            prompt = context.pop()["content"]

        sys_prompts = [msg["content"] for msg in context if msg.get("role") == "system"]
        sys_prompt = "\n".join(sys_prompts)
        context = [msg for msg in context if msg.get("role") != "system"]

        req = LLMRequest(
            prompt=prompt or " ",
            system_prompt=sys_prompt,
            model=real_model,
            context=context,
            extra=dict(self.request_defaults),
        )
        if options:
            if options.temperature is not None:
                req.temperature = options.temperature
            if options.max_tokens is not None:
                req.max_tokens = options.max_tokens
            if options.extra:
                if req.extra is None:
                    req.extra = {}
                req.extra.update(options.extra)
        return req

    def _sync_credentials(self) -> None:
        """惰性凭据同步: 每次调用前重取活 key, 变化时热更新底层 provider。

        背景(2026-08-23 实证): siliconflow 双 key 一活一死, provider 实例
        常驻 + __init__ 只取一次 key —— 死 key 被锁进实例, db 标记也不生效,
        get_key 五五开选中死 key 导致请求随机 401。多 key 轮转的意义就在
        get_key 每次都读库, provider 却把 key 缓存死了, 两者根本没对上。
        """
        cred = _get_credentials_for(self._cred_token)
        if not cred or not cred.get("api_key"):
            # 凭据彻底出局(全部 key 被判死): 清空底层 key 让 is_available
            # 转 False, discover 停止注册该引擎的模型(死引擎退出调度视野)。
            if self._underlying is not None and getattr(self._underlying, "_api_key", ""):
                self._underlying._api_key = ""
                for attr in ("_client", "_async_client"):
                    if hasattr(self._underlying, attr):
                        setattr(self._underlying, attr, None)
            return
        if cred["api_key"] == (self._credentials or {}).get("api_key"):
            return
        self._credentials = cred
        underlying = self._underlying
        if underlying is None:
            return
        underlying._api_key = cred["api_key"]
        # OpenAIProvider 缓存了以旧 key 构造的 SDK client, 必须一并重置
        for attr in ("_client", "_async_client"):
            if hasattr(underlying, attr):
                setattr(underlying, attr, None)

    def _report_auth_failure(self) -> None:
        """请求驱动淘汰: 真实调用 401 时把当前 key 标死。

        主动探测(reverify)有盲区: 部分端点(实测 kimi coding)对无效 key
        的 /models 不返回 401, 死 key 判不出来; 真实生成请求的 401 是
        最权威信号。标死后 get_key 出局, 下次 sync 引擎退出调度。
        """
        try:
            from .credentials import CredentialsManager

            key = (self._credentials or {}).get("api_key")
            if key:
                CredentialsManager().mark_key_active(self._cred_token, key, active=False)
                _log.warning("credential disabled by auth failure: %s", self._name)
        except Exception:
            return

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        name = type(exc).__name__
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return True
        text = str(exc)
        return "401" in text and ("auth" in text.lower() or "token" in text.lower() or "api key" in text.lower())

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        if not self._underlying:
            raise RuntimeError(f"Provider {self.name} has no underlying implementation.")

        self._sync_credentials()
        req = self._build_request(model, messages, options)
        try:
            resp = await self._underlying.generate(req)
        except Exception as exc:
            if self._is_auth_error(exc):
                self._report_auth_failure()
            raise

        return ChatResult(
            id="",
            model=model,
            content=resp.content,
            finish_reason=resp.finish_reason,
            usage={
                "prompt_tokens": resp.input_tokens,
                "completion_tokens": resp.output_tokens,
            },
            tool_calls=tuple(dict(tc) for tc in resp.tool_calls),
        )

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self._underlying:
            raise RuntimeError(f"Provider {self.name} has no underlying implementation.")

        self._sync_credentials()
        req = self._build_request(model, messages, options)

        # 优先带元数据的流(usage/finish_reason); underlying 未覆盖 detailed
        # 时基类默认包装 stream_generate, 行为不变。
        detailed = getattr(self._underlying, "stream_generate_detailed", None)
        if detailed is not None:
            async for event in detailed(req):
                yield StreamChunk(
                    model=model,
                    content=event.text,
                    finish_reason=event.finish_reason,
                    usage=event.usage,
                    tool_calls=tuple(dict(tc) for tc in event.tool_calls),
                )
            return

        async for chunk_text in self._underlying.stream_generate(req):
            yield StreamChunk(
                model=model,
                content=chunk_text,
            )


def load_ssot_models(
    registry: ModelRegistry,
    m1_compute_dir: str,
    m1_model_dir: str | None = None,
) -> None:
    """Load L0 M1 models from YAML and register them into the given ModelRegistry.

    Reads:
    - compute_engine/ YAMLs for provider endpoints
    - model/ YAMLs (MODEL-BREW-*.yaml) for model definitions
    - CredentialsManager for API keys

    Args:
        registry: ModelRegistry to register providers into.
        m1_compute_dir: Path to M1 compute_engine/ directory.
        m1_model_dir: Path to M1 model/ directory. If None, skips model defs.
    """
    # Pre-load model definitions
    engine_models: dict[str, list[dict]] = {}
    if m1_model_dir and os.path.isdir(m1_model_dir):
        engine_models = _load_model_defs(m1_model_dir)
        _log.info(
            "Loaded model definitions for %d compute engines from %s",
            len(engine_models),
            m1_model_dir,
        )

    # Load compute_engine configs
    pattern = os.path.join(m1_compute_dir, "*.yaml")
    count = 0
    for filepath in glob.glob(pattern):
        try:
            with open(filepath) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            _log.warning("Failed to load yaml %s: %s", filepath, e)
            continue

        if not config or not isinstance(config, dict):
            continue

        if config.get("type") in ("compute_engine", "ComputeEngine") and config.get("status") == "active":
            engine_id = config.get("id", "unknown")
            model_defs = engine_models.get(engine_id, [])
            provider = SSOTProviderAdapter(config, model_defs=model_defs)
            registry.register(provider)
            count += 1

    _log.info("Registered %d compute engines from %s", count, m1_compute_dir)
