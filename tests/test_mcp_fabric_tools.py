"""Unit tests for AetherForge MCP fabric tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from aetherforge.mcp_server import (
    forge_fabric_compact,
    forge_fabric_inspect,
    forge_fabric_vram,
    forge_fabric_warm,
    forge_swarm_run,
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


def test_forge_fabric_compact() -> None:
    fake_data = {
        "schema_version": "1",
        "data": {
            "model_id": "coding",
            "compaction_advised": True,
            "compression_ratio": 0.35,
            "pruned_tokens": 10240,
        },
    }
    mock_proc = MagicMock(returncode=0, stdout=json.dumps(fake_data), stderr="")

    with patch("subprocess.run", return_value=mock_proc):
        res = forge_fabric_compact("coding", 32768, 4096.0)
        assert res["data"]["compaction_advised"] is True
        assert res["data"]["compression_ratio"] == 0.35


def test_forge_swarm_run() -> None:
    with patch("aetherforge.swarm.rpc.run_swarm_workflow") as mock_rpc:
        mock_rpc.return_value = {
            "status": "success",
            "goal": "Build an AST triage agent",
            "result": "Completed plan and code review",
            "steps": [{"name": "任务规划", "status": "ok"}, {"name": "任务执行", "status": "ok"}],
        }
        res = forge_swarm_run("Build an AST triage agent")
        assert res["status"] == "success"
        assert len(res["steps"]) == 2

