from __future__ import annotations
import logging
import json
from collections.abc import AsyncIterator
from ..provider import LLMProvider, LLMRequest, LLMResponse

_log = logging.getLogger(__name__)

class MockProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"
    def available_models(self) -> list[str]:
        return ["mock-model"]
    def is_available(self) -> bool:
        return True
    async def generate(self, request: LLMRequest) -> LLMResponse:
        return self.complete(request)
    def complete(self, request: LLMRequest) -> LLMResponse:
        _log.info("[MOCK] Generating mock structured response")
        # 返回合法的 JSON 数组，使得 C2G 能够解析
        content = json.dumps([
            {
                "title": "AetherForge 网关的 Mock 测试任务",
                "description": "这是由 aetherforge 的 Mock Provider 生成的测试任务，证明大模型网关链路完全贯通。",
                "task_type": "feature",
                "risk_level": "L0", "cognitive_cartridge": "OPENSPEC-V1",
                "deliverables": ["测试报告"],
                "evidence_required": ["日志中包含 Mock Provider 响应"],
                "test_plan": ["无"]
            }
        ])
        return LLMResponse(content=content, model="mock-model", provider="mock")
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        resp = await self.generate(request)
        yield resp.content
