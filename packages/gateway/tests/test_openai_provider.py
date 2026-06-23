import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from llm_gateway.providers.openai_provider import OpenAIProvider
from llm_gateway.provider import LLMRequest

@pytest.fixture
def openai_provider():
    return OpenAIProvider(
        api_key="test-key",
        default_model="gpt-4o"
    )

@pytest.mark.asyncio
@patch.object(OpenAIProvider, "_get_async_client")
async def test_openai_generate(mock_get_client, openai_provider):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello world!"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10
    
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    req = LLMRequest(
        model="gpt-4o",
        prompt="Hi"
    )
    
    resp = await openai_provider.generate(req)
    assert resp.model == "gpt-4o"
    assert resp.content == "Hello world!"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 10
    
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "gpt-4o"

@patch.object(OpenAIProvider, "is_available", return_value=True)
def test_openai_health(mock_is_available, openai_provider):
    assert openai_provider.health_check() == ""
