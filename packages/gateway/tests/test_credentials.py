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
