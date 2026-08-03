"""compute_mesh.pool.db_pool — SQLite 连接池

解决 object_store.py / cost_db.py 中每次调用都 open/close SQLite 连接的性能问题。

用法：
    from compute_mesh.pool.db_pool import get_connection

    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT ...")
        conn.commit()

特性：
- 每个 db_path 维护一个长连接（check_same_thread=False，适合多线程）
- 在 fork/close 时自动重建失效连接
- 上下文管理器自动 commit / rollback
- 线程安全（RLock 保护连接字典）
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_log = logging.getLogger(__name__)

# 全局连接缓存：{db_path_str → sqlite3.Connection}
_connections: dict[str, sqlite3.Connection] = {}
_lock = threading.RLock()


def _open_connection(db_path: str) -> sqlite3.Connection:
    """Open a new SQLite connection with sensible defaults."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,  # allow multi-thread access with external locking
        timeout=10.0,  # wait up to 10s for write lock
    )
    conn.execute("PRAGMA journal_mode=WAL")  # write-ahead log for concurrency
    conn.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _get_or_create(db_path: str) -> sqlite3.Connection:
    """Return a cached connection, creating or repairing it as needed."""
    with _lock:
        conn = _connections.get(db_path)

        # Ping the connection; if it's broken, recreate
        if conn is not None:
            try:
                conn.execute("SELECT 1")
            except sqlite3.Error:
                _log.warning("Cached SQLite connection broken for %s, reconnecting", db_path)
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                conn = None

        if conn is None:
            conn = _open_connection(db_path)
            _connections[db_path] = conn
            _log.debug("Opened SQLite connection: %s", db_path)

        return conn


@contextmanager
def get_connection(db_path: str | Path) -> Generator[sqlite3.Connection]:
    """Context manager that yields a pooled SQLite connection.

    Commits on success, rolls back on exception.

    Example::

        with get_connection(db_path) as conn:
            conn.execute("INSERT INTO ... VALUES (?)", (value,))
            # auto-committed on exit
    """
    path_str = str(db_path)
    conn = _get_or_create(path_str)

    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error as rb_err:
            _log.warning("Rollback failed for %s: %s", path_str, rb_err)
        raise


def close_all() -> None:
    """Close all pooled connections (call at process shutdown)."""
    with _lock:
        for path, conn in list(_connections.items()):
            try:
                conn.close()
                _log.debug("Closed SQLite connection: %s", path)
            except sqlite3.Error as exc:
                _log.warning("Error closing connection %s: %s", path, exc)
        _connections.clear()
