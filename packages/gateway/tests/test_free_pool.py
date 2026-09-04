"""free_pool 回归与刷新能力测试 (2026-08-24)。

背景: FreePoolScanner 顶部漏 import pathlib.Path, 自诞生(同日凌晨)起
实例化即 NameError —— gateway 的 try/except 把它吞成 debug 级日志,
"发现闭环"从未真正转过一圈, 无人察觉。本文件把这条回归钉死, 并覆盖
openrouter free 清单刷新的过滤/diff/写盘逻辑。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from llm_gateway import paths
from llm_gateway.free_pool import FreePoolScanner, _is_free_chat_model, refresh_openrouter_free


class TestScannerRegression:
    def test_scanner_instantiates_without_name_error(self):
        """Path import 回归: 此前 NameError: name 'Path' is not defined。"""
        scanner = FreePoolScanner()
        assert scanner._STATE_FILE.name == "free_pool_last_seen.json"


class TestFreeChatModelFilter:
    def test_free_text_model_passes(self):
        model = {
            "id": "z-ai/glm-5.2:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        }
        assert _is_free_chat_model(model) is True

    def test_paid_model_rejected(self):
        model = {
            "id": "openai/gpt-5",
            "pricing": {"prompt": "0.001", "completion": "0.002"},
            "architecture": {"output_modalities": ["text"]},
        }
        assert _is_free_chat_model(model) is False

    def test_half_free_rejected(self):
        """prompt 免费但 completion 收费(试用型)不算真免费。"""
        model = {
            "id": "x/trial",
            "pricing": {"prompt": "0", "completion": "0.001"},
            "architecture": {"output_modalities": ["text"]},
        }
        assert _is_free_chat_model(model) is False

    def test_non_text_modality_rejected(self):
        """lyria 音乐生成(out 含 audio)这类名义免费但非 chat 的条目要排除。"""
        model = {
            "id": "google/lyria-3-pro-preview",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text", "audio"]},
        }
        assert _is_free_chat_model(model) is False

    def test_content_safety_classifier_rejected(self):
        """nemotron content-safety 是审核分类器, 纯 text 模态但非通用 chat。"""
        model = {
            "id": "nvidia/nemotron-3.5-content-safety:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        }
        assert _is_free_chat_model(model) is False

    def test_official_free_router_kept(self):
        """openrouter/free 官方免费路由, 对 free 池有直接价值, 必须保留。"""
        model = {
            "id": "openrouter/free",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        }
        assert _is_free_chat_model(model) is True

    def test_missing_pricing_defaults_paid(self):
        assert _is_free_chat_model({"id": "x"}) is False


class TestRefreshOpenrouterFree:
    def _fake_response(self, models: list[dict]):
        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"data": models}

        return _Resp()

    def _make_yaml(self, tmp_path: Path, model_ids: list[str]) -> Path:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        doc = {
            "type": "ModelDefinition",
            "status": "active",
            "engine_ref": "ENG-OPENROUTER-CLOUD",
            "model_driven_refs": {"source": "openrouter_api_free_tier", "synced_at": "2026-08-09T00:00:00Z"},
            "governance": {"owner": "aetherforge-gateway", "steward": "user-provided-key"},
            "models": [
                {
                    "model_id": mid,
                    "display_name": f"{mid} (Free)",
                    "cost_per_1k_input": 0,
                    "cost_per_1k_output": 0,
                    "context_window": 131072,
                    "capabilities": ["chat"],
                }
                for mid in model_ids
            ],
        }
        path = model_dir / "MODEL-BREW-OPENROUTER-FREE.yaml"
        path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
        return model_dir

    def test_dry_run_reports_drift_without_writing(self, tmp_path, monkeypatch):
        model_dir = self._make_yaml(tmp_path, ["old/static:free"])
        monkeypatch.setattr(paths, "M1_MODEL_DIR", model_dir)

        remote = [
            {
                "id": "old/static:free",
                "name": "Old Static",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
                "context_length": 65536,
            },
            {
                "id": "new/hot:free",
                "name": "New Hot",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
            },
            {  # 名义免费但非 chat, 不应进入 remote 清单
                "id": "google/lyria:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["audio"]},
            },
        ]
        monkeypatch.setattr("httpx.get", lambda *a, **k: self._fake_response(remote))

        result = refresh_openrouter_free(write=False)

        assert result["ok"] is True
        assert result["remote_free"] == 2
        assert result["static"] == 1
        assert result["added"] == ["new/hot:free"]
        assert result["removed"] == []
        assert result["written"] is False
        # dry-run 不落盘: yaml 原样
        doc = yaml.safe_load((model_dir / "MODEL-BREW-OPENROUTER-FREE.yaml").read_text())
        assert [m["model_id"] for m in doc["models"]] == ["old/static:free"]

    def test_write_updates_yaml_and_steward(self, tmp_path, monkeypatch):
        model_dir = self._make_yaml(tmp_path, ["old/static:free", "gone/model:free"])
        monkeypatch.setattr(paths, "M1_MODEL_DIR", model_dir)

        remote = [
            {
                "id": "old/static:free",
                "name": "Old Static",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
                "context_length": 65536,
            },
        ]
        monkeypatch.setattr("httpx.get", lambda *a, **k: self._fake_response(remote))

        result = refresh_openrouter_free(write=True)

        assert result["written"] is True
        assert result["removed"] == ["gone/model:free"]
        doc = yaml.safe_load((model_dir / "MODEL-BREW-OPENROUTER-FREE.yaml").read_text())
        assert [m["model_id"] for m in doc["models"]] == ["old/static:free"]
        assert doc["governance"]["steward"] == "free-pool-refresh"
        assert doc["model_driven_refs"]["synced_at"] != "2026-08-09T00:00:00Z"
        # 引擎归属与治理结构保留
        assert doc["engine_ref"] == "ENG-OPENROUTER-CLOUD"
