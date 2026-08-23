"""CredentialsManager.add_key() 去重回归测试。

背景(2026-08-23): import_from_cc_switch() 在每次 gateway 启动时都会对
cc-switch 数据库里的每个 provider 无条件调用 add_key(), 而 add_key() 原本
是纯 INSERT, 没有任何"已存在就跳过"的判断 —— 两个月的重启历史里,
credentials.db 里 kimi_for_coding 这一个 provider 积累了 748 条完全相同
的重复记录, 是排查一个不相关的路由问题时顺带发现的。
"""

from __future__ import annotations

from pathlib import Path

from llm_gateway.credentials import CredentialsManager, _get_connection


def _manager(tmp_path: Path) -> CredentialsManager:
    return CredentialsManager(db_path=tmp_path / "credentials.db")


def test_add_key_is_idempotent_on_provider_and_api_key(tmp_path: Path) -> None:
    cm = _manager(tmp_path)
    for _ in range(5):
        cm.add_key("test-provider", "sk-same-key", base_url="https://example.com", note="repeat call")

    with _get_connection(tmp_path / "credentials.db") as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM credentials WHERE provider = ? AND api_key = ?",
            ("test-provider", "sk-same-key"),
        ).fetchone()
    assert rows[0] == 1


def test_add_key_repeated_call_updates_metadata_not_inserts(tmp_path: Path) -> None:
    cm = _manager(tmp_path)
    cm.add_key("test-provider", "sk-same-key", base_url="https://old.example.com", note="first")
    cm.add_key("test-provider", "sk-same-key", base_url="https://new.example.com", note="second")

    with _get_connection(tmp_path / "credentials.db") as conn:
        row = conn.execute(
            "SELECT base_url, note FROM credentials WHERE provider = ? AND api_key = ?",
            ("test-provider", "sk-same-key"),
        ).fetchone()
    assert tuple(row) == ("https://new.example.com", "second")


def test_add_key_supports_multiple_distinct_keys_per_provider(tmp_path: Path) -> None:
    cm = _manager(tmp_path)
    cm.add_key("test-provider", "sk-key-a", weight=70)
    cm.add_key("test-provider", "sk-key-b", weight=30)

    with _get_connection(tmp_path / "credentials.db") as conn:
        rows = conn.execute(
            "SELECT api_key FROM credentials WHERE provider = ? ORDER BY api_key",
            ("test-provider",),
        ).fetchall()
    assert [r[0] for r in rows] == ["sk-key-a", "sk-key-b"]


def test_sync_credentials_hot_swaps_key_and_resets_cached_clients(monkeypatch) -> None:
    """惰性凭据同步: key 变化时热更新 underlying 并重置 SDK client 缓存。

    背景(2026-08-23 实证): siliconflow 双 key 一活一死, provider 常驻 +
    __init__ 只取一次 key, 死 key 被锁进实例, db 标记不生效。
    """
    from types import SimpleNamespace

    from llm_gateway import ssot_loader
    from llm_gateway.ssot_loader import SSOTProviderAdapter

    keys = {"current": "old-key"}
    monkeypatch.setattr(
        ssot_loader, "_get_credentials_for", lambda token: {"api_key": keys["current"], "base_url": "https://x/v1"}
    )
    adapter = SSOTProviderAdapter({"id": "ENG-TEST-CLOUD", "supported_protocols": ["openai"]})
    assert adapter._underlying._api_key == "old-key"
    adapter._underlying._async_client = object()  # 模拟已缓存的 SDK client

    keys["current"] = "new-key"  # db 侧换 key(如死 key 出局后 get_key 改选)
    adapter._sync_credentials()

    assert adapter._underlying._api_key == "new-key"
    assert adapter._underlying._async_client is None  # 旧 key 的 client 已重置
