"""Unit tests for AetherForge MCP fabric tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from aetherforge.mcp_server import (
    forge_fabric_inspect,
    forge_fabric_vram,
    forge_fabric_warm,
)


def test_forge_fabric_inspect() -> None:
    fake_data = {
        "schema_version": "1",
        "data": {
            "thermal_pressure": "nominal",
            "supported_tiers": ["fast", "standard", "reasoning"],
        },
    }
    mock_proc = MagicMock(returncode=0, stdout=json.dumps(fake_data), stderr="")

    with patch("subprocess.run", return_value=mock_proc):
        res = forge_fabric_inspect()
        assert res["schema_version"] == "1"
        assert res["data"]["thermal_pressure"] == "nominal"


def test_forge_fabric_warm() -> None:
    fake_data = {
        "schema_version": "1",
        "data": {
            "model_id": "coding",
            "warmed_count": 3,
        },
    }
    mock_proc = MagicMock(returncode=0, stdout=json.dumps(fake_data), stderr="")

    with patch("subprocess.run", return_value=mock_proc):
        res = forge_fabric_warm("coding")
        assert res["data"]["warmed_count"] == 3


def test_forge_fabric_vram() -> None:
    fake_data = {
        "schema_version": "1",
        "data": {
            "model_id": "coding",
            "context_tokens": 32768,
            "kv_cache_mb": 8448.0,
        },
    }
    mock_proc = MagicMock(returncode=0, stdout=json.dumps(fake_data), stderr="")

    with patch("subprocess.run", return_value=mock_proc):
        res = forge_fabric_vram("coding", 32768)
        assert res["data"]["kv_cache_mb"] == 8448.0
