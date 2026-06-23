import pytest
from llm_gateway.providers.mock_provider import MockProvider
from llm_gateway.provider import LLMRequest

@pytest.mark.asyncio
async def test_mock_provider_basic():
    provider = MockProvider()
    assert provider.provider_name == "mock"
    assert "mock-model" in provider.available_models()

    req = LLMRequest(
        model="mock-model",
        prompt="hello"
    )
    
    resp = await provider.generate(req)
    assert resp.model == "mock-model"
    assert "MOCK" in resp.content or "Mock" in resp.content
    assert resp.input_tokens >= 0
    assert resp.output_tokens >= 0

def test_mock_provider_health():
    provider = MockProvider()
    assert provider.is_available() is True
    assert provider.health_check() == ""
