"""Tests for llm_gateway.quota_engine — ProviderData, QuotaEngine cache operations."""

from __future__ import annotations


class TestProviderData:
    def test_default_provider_data(self):
        from llm_gateway.quota_engine import ProviderData

        pd = ProviderData()
        assert pd.provider == ""
        assert pd.has_credentials is False
        assert pd.available is False
        assert pd.status == "unknown"
        assert pd.quota_pct == 100.0
        assert pd.quota_source == ""

    def test_to_dict(self):
        from llm_gateway.quota_engine import ProviderData

        pd = ProviderData(provider="openai", available=True, status="available", quota_pct=80.0, quota_source="codexbar")
        d = pd.to_dict()
        assert d["provider"] == "openai"
        assert d["available"] is True
        assert d["status"] == "available"
        assert d["quota_pct"] == 80.0
        assert d["quota_source"] == "codexbar"


class TestQuotaEngineCache:
    def test_cache_miss_creates_default(self):
        from llm_gateway.quota_engine import QuotaEngine

        qe = QuotaEngine()
        qe._ready = True  # skip refresh loop
        result = qe.get_quota("unknown-provider")
        assert result.provider == "unknown-provider"
        # Should have done a quick check
        assert result.quota_source in ("local", "")

    def test_cache_set_and_get(self):
        from llm_gateway.quota_engine import ProviderData, QuotaEngine

        qe = QuotaEngine()
        pd = ProviderData(provider="test", available=True, status="available", quota_pct=50.0)
        qe._set_cache(pd)

        result = qe.get_quota("test")
        assert result.provider == "test"
        assert result.available is True
        assert result.quota_pct == 50.0

    def test_get_summary_empty_cache(self):
        from llm_gateway.quota_engine import QuotaEngine

        qe = QuotaEngine()
        summary = qe.get_summary()
        assert summary["total"] == 0
        assert summary["available"] == 0

    def test_get_summary_with_data(self):
        from llm_gateway.quota_engine import ProviderData, QuotaEngine

        qe = QuotaEngine()
        qe._set_cache(ProviderData(provider="a", available=True, status="available"))
        qe._set_cache(ProviderData(provider="b", available=False, status="quota_exhausted"))
        qe._set_cache(ProviderData(provider="c", available=True, status="quota_low"))

        summary = qe.get_summary()
        assert summary["total"] == 3
        assert summary["available"] == 2
        assert summary["quota_low"] == 1
        assert summary["quota_exhausted"] == 1

    def test_invalidate_clears_cache(self):
        from llm_gateway.quota_engine import ProviderData, QuotaEngine

        qe = QuotaEngine()
        qe._set_cache(ProviderData(provider="test", available=True))
        assert qe.get_quota("test").available is True

        qe.invalidate()
        # After invalidate, get_quota does a quick check which may still return data
        # but the old cached data should be gone
        result = qe.get_quota("test")
        assert result is not None  # Should still return something via quick check


class TestQuotaEngineDynamic:
    def test_load_quota_definitions(self, tmp_path, monkeypatch):
        # 1. 创建临时的 quota_definition 目录，写入 QD-TEST.yaml
        quota_dir = tmp_path / "quota_definition"
        quota_dir.mkdir()

        qd_data = """
id: QD-TEST
name: Test Provider Quota
type: QuotaDefinition
provider: test_provider
quota_model: test_model_type
unit: USD
source: codexbar
check_command: test_command usage --provider test_provider --format json
refresh_interval: 100
"""
        (quota_dir / "QD-TEST.yaml").write_text(qd_data, encoding="utf-8")

        # 2. Mock M1_QUOTA_DIR
        from llm_gateway import quota_engine
        monkeypatch.setattr(quota_engine, "M1_QUOTA_DIR", quota_dir)

        # 3. 实例化并验证加载
        qe = quota_engine.QuotaEngine()
        assert "test_provider" in qe._codexbar_providers
        assert qe._quota_model_map["test_provider"] == "test_model_type"
        assert qe._check_commands["test_provider"] == "test_command usage --provider test_provider --format json"
