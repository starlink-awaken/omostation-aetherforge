"""按目标模型的并发闸(背压): 高负载下排队等槽, 而不是并发全砸成 504/502。"""

from __future__ import annotations

import asyncio

import pytest
from llm_gateway.gateway import ModelGateway


def _bare_gateway() -> ModelGateway:
    gw = ModelGateway.__new__(ModelGateway)
    gw._model_slots = {}
    return gw


@pytest.mark.asyncio
async def test_slot_limits_concurrency_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHERFORGE_MAX_CONCURRENT_PER_MODEL", "2")
    gw = _bare_gateway()
    sem = gw._slot("heavy-model")
    assert sem is not None
    await sem.acquire()
    await sem.acquire()
    assert gw._slot("heavy-model") is sem  # 同模型复用同一闸
    with pytest.raises(TimeoutError):  # 第三次获取必须等待(并发上限生效)
        await asyncio.wait_for(sem.acquire(), timeout=0.05)


def test_slot_disabled_and_isolated_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHERFORGE_MAX_CONCURRENT_PER_MODEL", "0")
    gw = _bare_gateway()
    assert gw._slot("any") is None  # 0 = 关闭
    monkeypatch.setenv("AETHERFORGE_MAX_CONCURRENT_PER_MODEL", "3")
    assert gw._slot("a") is not gw._slot("b")  # 不同模型互不挤占
