from unittest.mock import MagicMock, patch

import pytest
from llm_gateway.provider import LLMRequest
from llm_gateway.providers.ollama_provider import OllamaProvider


@pytest.fixture
def ollama_provider():
    return OllamaProvider(base_url="http://localhost:11434", default_model="llama3")


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_ollama_generate(mock_post, ollama_provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {"role": "assistant", "content": "Hello world!"},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 10,
    }
    mock_post.return_value = mock_response

    req = LLMRequest(model="llama3", prompt="Hi")

    resp = await ollama_provider.generate(req)
    assert resp.model == "llama3"
    assert resp.content == "Hello world!"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 10

    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["json"]["model"] == "llama3"


@patch.object(OllamaProvider, "is_available", return_value=True)
def test_ollama_health(mock_is_available, ollama_provider):
    assert ollama_provider.health_check() == ""
