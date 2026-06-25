from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llm_gateway.provider import LLMRequest
from llm_gateway.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(api_key="test-key", default_model="claude-3-haiku-20240307")


@pytest.mark.asyncio
@patch.object(AnthropicProvider, "_get_async_client")
async def test_anthropic_generate(mock_get_client, anthropic_provider):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world!")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 15
    mock_client.messages.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    req = LLMRequest(model="claude-3-haiku-20240307", prompt="Hi")

    resp = await anthropic_provider.generate(req)
    assert resp.model == "claude-3-haiku-20240307"
    assert resp.content == "Hello world!"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 15

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["model"] == "claude-3-haiku-20240307"


@patch.object(AnthropicProvider, "is_available", return_value=True)
def test_anthropic_health(mock_is_available, anthropic_provider):
    assert anthropic_provider.health_check() == ""
