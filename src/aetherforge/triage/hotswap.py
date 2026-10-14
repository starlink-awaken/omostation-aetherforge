"""Hot-swap — 运行时切换分诊模型.

功能:
- 运行时切换主模型 (无需重启)
- 自动降级: 主模型不可用时自动 fallback
- 健康检查: 定期探活模型
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from aetherforge.endpoint import chat_url, gateway_key


@dataclass
class ModelHealth:
    """模型健康状态."""

    name: str
    available: bool = False
    latency: float = 0.0
    last_check: float = 0.0
    error: str | None = None


@dataclass
class HotSwapConfig:
    """热切换配置."""

    gateway_url: str = field(default_factory=chat_url)
    api_key: str = field(default_factory=gateway_key)
    health_check_interval: int = 60  # 健康检查间隔 (秒)
    health_check_timeout: int = 10  # 健康检查超时 (秒)


class ModelHotSwap:
    """模型热切换管理器."""

    def __init__(self, config: HotSwapConfig | None = None):
        self.config = config or HotSwapConfig()
        self.health: dict[str, ModelHealth] = {}
        self.active_model: str | None = None
        self.fallback_chain: list[str] = []

    def set_chain(self, chain: list[str]):
        """设置 fallback 链."""
        self.fallback_chain = chain
        for model in chain:
            if model not in self.health:
                self.health[model] = ModelHealth(name=model)
        if chain and not self.active_model:
            self.active_model = chain[0]

    def switch(self, model: str) -> bool:
        """切换到指定模型."""
        if model not in self.fallback_chain:
            return False

        # 检查模型是否可用
        health = self.check_health(model)
        if health.available:
            self.active_model = model
            return True
        return False

    def check_health(self, model: str) -> ModelHealth:
        """检查模型健康状态."""
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
                "temperature": 0,
                "extra_body": {"reasoning_effort": "none"},
            }
        ).encode()

        req = urllib.request.Request(  # noqa: S310  (internal gateway call)
            self.config.gateway_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.config.health_check_timeout) as resp:  # noqa: S310  (internal gateway call)
                d = json.loads(resp.read())
            latency = time.time() - t0
            content = d["choices"][0]["message"]["content"].strip()

            health = ModelHealth(
                name=model,
                available=bool(content),
                latency=latency,
                last_check=time.time(),
            )
        except Exception as e:
            health = ModelHealth(
                name=model,
                available=False,
                latency=time.time() - t0,
                last_check=time.time(),
                error=str(e)[:50],
            )

        self.health[model] = health
        return health

    def check_all(self) -> dict[str, ModelHealth]:
        """检查所有模型健康状态."""
        for model in self.fallback_chain:
            self.check_health(model)
        return self.health

    def get_active(self) -> tuple[str, ModelHealth]:
        """获取当前活跃模型."""
        if not self.active_model:
            raise RuntimeError("No active model set")

        health = self.health.get(self.active_model)
        if health and health.available:
            return self.active_model, health

        # 自动降级
        for model in self.fallback_chain:
            h = self.check_health(model)
            if h.available:
                self.active_model = model
                return model, h

        raise RuntimeError("No available model")

    def get_status(self) -> dict:
        """获取当前状态."""
        return {
            "active": self.active_model,
            "chain": self.fallback_chain,
            "health": {
                name: {
                    "available": h.available,
                    "latency": round(h.latency, 3),
                    "error": h.error,
                }
                for name, h in self.health.items()
            },
        }
