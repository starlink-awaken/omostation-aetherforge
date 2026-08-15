"""GET /v1/compute observe surface — inventory warnings only, /health unchanged."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web
from llm_gateway import openai_proxy as proxy
from llm_gateway.gateway import (
    GatewayConfig,
    ModelGateway,
    _sanitize_inventory_warnings,
)
from llm_gateway.omlxc_client import OmlxcError, OmlxcErrorCode, OmlxcHealth


class _FakeOmlxc:
    def __init__(self, result: OmlxcHealth | OmlxcError) -> None:
        self.result = result

    async def health(self, *, timeout: float = 2.0) -> OmlxcHealth:
        if isinstance(self.result, OmlxcError):
            raise self.result
        return self.result


class _FakeRegistry:
    def list_models(self) -> list[object]:
        return [object()]


class _FakeScheduler:
    async def select_model(self, _request: object) -> None:
        return None


def _gateway(client: _FakeOmlxc, *, mode: str = "active") -> ModelGateway:
    config = GatewayConfig(
        omlxc_mode=mode,
        model_ports={},
        model_sizes={},
        lmstudio_fallback={},
        ollama_fallback={},
        background_tasks_enabled=False,
    )
    return ModelGateway(_FakeRegistry(), _FakeScheduler(), config, omlxc_client=client)  # type: ignore[arg-type]


class TestSanitizeInventoryWarnings:
    def test_keeps_inventory_drop_and_drops_names(self) -> None:
        cleaned = _sanitize_inventory_warnings(
            [
                {
                    "code": "inventory_drop",
                    "node_id": "mbp",
                    "backend_id": "omlx-app",
                    "baseline": 23,
                    "current": 6,
                    "models": ["coding", "coding-fast"],
                    "missing": ["vision"],
                }
            ]
        )
        assert cleaned == [
            {
                "code": "inventory_drop",
                "node_id": "mbp",
                "backend_id": "omlx-app",
                "baseline": 23,
                "current": 6,
            }
        ]

    def test_drops_unknown_codes_and_malformed_rows(self) -> None:
        cleaned = _sanitize_inventory_warnings(
            [
                {"code": "something_else", "node_id": "mbp", "backend_id": "x", "baseline": 1, "current": 0},
                {"code": "inventory_drop", "node_id": "", "backend_id": "x", "baseline": 1, "current": 0},
                {"code": "inventory_drop", "node_id": "mbp", "backend_id": "x", "baseline": True, "current": 0},
                "nope",
            ]
        )
        assert cleaned == []


@pytest.mark.asyncio
async def test_observe_forwards_inventory_drop() -> None:
    client = _FakeOmlxc(
        OmlxcHealth(
            status="ready",
            warnings=(
                {
                    "code": "inventory_drop",
                    "node_id": "mbp-m5-max-128g",
                    "backend_id": "mbp-m5-max-128g-omlx-app",
                    "baseline": 23,
                    "current": 0,
                    "model_names": ["coding"],
                },
            ),
        )
    )
    payload = await _gateway(client, mode="shadow").observe_omlxc_compute()
    assert payload["omlxc"] == "ok"
    assert payload["omlxc_mode"] == "shadow"
    assert payload["warnings"] == [
        {
            "code": "inventory_drop",
            "node_id": "mbp-m5-max-128g",
            "backend_id": "mbp-m5-max-128g-omlx-app",
            "baseline": 23,
            "current": 0,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (OmlxcError(OmlxcErrorCode.UNAVAILABLE), "unavailable"),
        (OmlxcError(OmlxcErrorCode.TIMEOUT), "timeout"),
        (OmlxcError(OmlxcErrorCode.INVALID), "invalid"),
    ],
)
async def test_observe_degrades_without_raising(error: OmlxcError, expected: str) -> None:
    payload = await _gateway(_FakeOmlxc(error), mode="legacy").observe_omlxc_compute()
    assert payload == {"omlxc": expected, "omlxc_mode": "legacy", "warnings": []}


class TestHealthUnchanged:
    def test_health_body_is_identity_only(self) -> None:
        response = asyncio.run(proxy.handle_health(SimpleNamespace()))
        assert response.status == 200
        assert response.body == b'{"status": "ok", "service": "aetherforge-openai-proxy"}'


class TestComputeAuth:
    @staticmethod
    def _run(api_key, path="/v1/compute", header=None):
        app = {proxy.API_KEY: api_key}
        req = SimpleNamespace(app=app, path=path, headers=header or {})

        async def handler(_):
            return web.json_response({"ok": True})

        return asyncio.run(proxy.auth_middleware(req, handler))

    def test_compute_requires_the_same_key_as_models(self) -> None:
        assert self._run("secret").status == 401
        assert self._run("secret", path="/v1/models").status == 401

    def test_health_stays_open(self) -> None:
        assert self._run("secret", path="/health").status == 200
        assert self._run("secret", path="/").status == 200

    def test_compute_accepts_bearer(self) -> None:
        assert self._run("secret", header={"Authorization": "Bearer secret"}).status == 200


@pytest.mark.asyncio
async def test_handle_compute_returns_200_when_daemon_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Down:
        async def observe_omlxc_compute(self) -> dict[str, object]:
            return {"omlxc": "unavailable", "omlxc_mode": "active", "warnings": []}

    monkeypatch.setattr(proxy, "get_gateway", lambda: _Down())
    response = await proxy.handle_compute(SimpleNamespace())
    assert response.status == 200
    assert response.body == b'{"omlxc": "unavailable", "omlxc_mode": "active", "warnings": []}'
