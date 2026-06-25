"""SSOT M1 model loader — reads compute_engine + model definitions + credentials.

Architecture:
    M1 SSOT has three layers:
    1. compute_engine/ — "where" (endpoint, protocol)
    2. model/ — "what" (model_id, pricing, capabilities)
    3. CredentialsManager — "how" (API keys, base_url)

    SSOTProviderAdapter merges all three into a unified provider.
"""

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

        keys = cm.list_keys(provider_name)
        base_url = keys[0].get("note", "") if keys else ""

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
        self._model_defs = model_defs or []
        self._credentials: dict | None = None
        self._underlying = None

        # Try to inject credentials from CredentialsManager
        cred = _get_credentials_for(self._name.replace("ENG-", "").split("-")[0].lower())
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
            self._underlying = OllamaProvider(base_url=self.base_url)
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
        model_names = self._underlying.available_models()
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

    def _build_request(self, model: str, messages: list[dict[str, Any]],
                       options: ChatOptions | None) -> LLMRequest:
        real_model = model.split("/")[-1] if "/" in model else model

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
        )
        if options:
            if options.temperature is not None:
                req.temperature = options.temperature
            if options.max_tokens is not None:
                req.max_tokens = options.max_tokens
        return req

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        if not self._underlying:
            raise RuntimeError(
                f"Provider {self.name} has no underlying implementation."
            )

        req = self._build_request(model, messages, options)
        resp = await self._underlying.generate(req)

        return ChatResult(
            id="",
            model=model,
            content=resp.content,
            finish_reason=resp.finish_reason,
            usage={
                "prompt_tokens": resp.input_tokens,
                "completion_tokens": resp.output_tokens,
            },
        )

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self._underlying:
            raise RuntimeError(
                f"Provider {self.name} has no underlying implementation."
            )

        req = self._build_request(model, messages, options)

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
