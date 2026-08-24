from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llm_gateway.provider import LLMRequest
from llm_gateway.providers.openai_provider import OpenAIProvider


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="test-key", default_model="gpt-4o")


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

    req = LLMRequest(model="gpt-4o", prompt="Hi")

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


def _chunk(content=None, finish=None, usage=None):
    """构造 SDK 流式 chunk 的模拟对象。"""
    from types import SimpleNamespace

    if usage is not None:
        return SimpleNamespace(choices=[], usage=usage)
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
@patch.object(OpenAIProvider, "_get_async_client")
async def test_stream_detailed_yields_text_then_usage(mock_get_client, openai_provider):
    mock_client = AsyncMock()

    async def fake_create(**kwargs):
        async def gen():
            yield _chunk(content="你")
            yield _chunk(content="好")
            yield _chunk(finish="stop")
            yield _chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8))

        return gen()

    mock_client.chat.completions.create.side_effect = fake_create
    mock_get_client.return_value = mock_client


    events = [e async for e in openai_provider.stream_generate_detailed(LLMRequest(model="gpt-4o", prompt="hi"))]
    assert [(e.text, e.finish_reason, e.usage) for e in events] == [
        ("你", None, None),
        ("好", None, None),
        ("", "stop", {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}),
    ]
    # 首选路径必须带 include_usage
    first_kwargs = mock_client.chat.completions.create.call_args_list[0][1]
    assert first_kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
@patch.object(OpenAIProvider, "_get_async_client")
async def test_stream_detailed_degrades_when_stream_options_rejected(mock_get_client, openai_provider):
    """不认 stream_options 的网关: 400 后降级重试, 内容流不丢。"""
    mock_client = AsyncMock()
    from openai import BadRequestError

    request_obj = MagicMock()
    request_obj.status_code = 400
    http_response = MagicMock()
    http_response.status_code = 400
    http_response.headers = {}
    http_response.request = request_obj

    calls = {"n": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1 and "stream_options" in kwargs:
            raise BadRequestError(
                message="Unknown parameter: stream_options", response=http_response, body=None
            )

        async def gen():
            yield _chunk(content="ok")

        return gen()

    mock_client.chat.completions.create.side_effect = fake_create
    mock_get_client.return_value = mock_client

    events = [e async for e in openai_provider.stream_generate_detailed(LLMRequest(model="gpt-4o", prompt="hi"))]
    assert [e.text for e in events] == ["ok"]
    assert calls["n"] == 2  # 第一次 400 被拒, 第二次不带 stream_options 成功
