"""Tests for llm_gateway.tracing module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


def test_get_langfuse_client_not_configured(monkeypatch):
    from llm_gateway.tracing import get_langfuse_client

    # Ensure environment does not have Langfuse variables
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_API_KEY", raising=False)

    # Global client cache reset
    import llm_gateway.tracing as tracing
    tracing._client = None

    client = get_langfuse_client()
    assert client is None


def test_get_langfuse_client_configured(monkeypatch):
    # Mock Langfuse constructor
    mock_langfuse = MagicMock()
    import llm_gateway.tracing as tracing
    monkeypatch.setattr(tracing, "Langfuse", mock_langfuse)
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    tracing._client = None

    # Inject mock keys
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test_pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test_sec")

    client = tracing.get_langfuse_client()
    assert client is not None
    mock_langfuse.assert_called_once_with(public_key="test_pub", secret_key="test_sec", host="http://localhost:3000")


def test_trace_llm_call_context_manager(monkeypatch):
    mock_generation = MagicMock()
    mock_trace = MagicMock()
    mock_trace.generation.return_value = mock_generation
    mock_client = MagicMock()
    mock_client.trace.return_value = mock_trace

    import llm_gateway.tracing as tracing
    tracing._client = mock_client
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)

    messages = [{"role": "user", "content": "hello"}]
    with tracing.trace_llm_call("engine/model-name", messages) as gen:
        assert gen is mock_generation
        mock_client.trace.assert_called_once_with(
            name="aetherforge-gateway-call",
            metadata={"model_id": "engine/model-name", "strategy": "balanced"}
        )
        mock_trace.generation.assert_called_once_with(
            name="chat-completion",
            model="model-name",
            input="user: hello",
            metadata={"messages": messages}
        )
