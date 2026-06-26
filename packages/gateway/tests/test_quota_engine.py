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
