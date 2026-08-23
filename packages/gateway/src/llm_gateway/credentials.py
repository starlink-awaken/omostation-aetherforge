"""CredentialsManager — SQLite 凭据 + 配额 + 约束管理 (原 cc-switch + codexbar 遗产).

替代环境变量管理 API Key，提供:
  - 凭据安全存储 (SQLite + 可选的加密)
  - 多 Key 轮转 (一个 Provider 多个 Key)
  - 月度预算约束
  - 配额查询与预警

用法::

    from llm_gateway.credentials import CredentialsManager

    cm = CredentialsManager()
    cm.add_key("openai", "sk-xxx")
    cm.add_key("openai", "sk-yyy", weight=30)  # 30% 流量

    key = cm.get_key("openai")  # 按权重返回
    print(cm.get_quota("openai"))  # 配额状态

CLI::

    aetherforge credentials add openai --key sk-xxx
    aetherforge credentials list
    aetherforge credentials quota openai
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .pricing import PricingRegistry

_log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".aetherforge" / "credentials.db"


@contextmanager
def _get_connection(db_path: str | Path) -> Generator[sqlite3.Connection]:
    """Context manager: WAL + NORMAL + auto-commit/rollback. Per-call close."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


@dataclass
class CredentialEntry:
    """A single API key / credential entry."""

    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    weight: int = 100  # 流量权重 (用于多 Key 轮转)
    is_active: bool = True
    note: str = ""
    created_at: float = 0.0


@dataclass
class BudgetConstraint:
    """Monthly budget constraint for a provider."""

    provider: str = ""
    monthly_limit: float = 0.0  # 月预算上限 ($)
    action: str = "warn"  # block | warn | log
    current_month_spend: float = 0.0
    month: str = ""  # "YYYY-MM"


# ── CodexBar integration ──────────────────────────────────────────────────

_CODEXBAR_PATH: str | None = None


def _find_codexbar() -> str | None:
    """Locate the codexbar binary (used by CodexBarProvider)."""
    global _CODEXBAR_PATH
    if _CODEXBAR_PATH is not None:
        return _CODEXBAR_PATH
    for path in os.environ.get("PATH", "").split(":"):
        candidate = os.path.join(path, "codexbar")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            _CODEXBAR_PATH = candidate
            return candidate
    _CODEXBAR_PATH = ""
    return None


def codexbar_available() -> bool:
    """Check if codexbar CLI is installed."""
    return _find_codexbar() is not None


_PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "claude",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "azure": "azure-openai",
    "bedrock": "bedrock",
    "ollama": "ollama",
    "vertex": "vertexai",
}


def fetch_codexbar_quota(provider: str) -> dict[str, Any]:
    """Fetch real-time quota from codexbar CLI.

    Returns dict with ``limit``, ``used``, ``remaining`` or
    ``{"status": "unavailable"}``.

    This is the direct replacement for the old ``CodexBarCache``
    in SharedBrain B-OS.
    """
    codexbar = _find_codexbar()
    if not codexbar:
        return {"status": "unavailable", "reason": "codexbar not installed"}

    mapped = _PROVIDER_MAP.get(provider, provider)
    import json
    import subprocess

    try:
        result = subprocess.run(
            [codexbar, "usage", "--format", "json", "--provider", mapped],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"status": "unavailable", "reason": f"codexbar exit {result.returncode}"}

        data = json.loads(result.stdout)
        # codexbar returns a list: [{"source":"api","provider":"deepseek","usage":{...}}]
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            usage = entry.get("usage", {})
            primary = usage.get("primary", {})
            used_pct = primary.get("usedPercent", 0)
            reset_desc = primary.get("resetDescription", "")
            return {
                "status": "available",
                "source": "codexbar",
                "provider": entry.get("provider", provider),
                "limit": 100,
                "used": used_pct,
                "remaining": 100 - used_pct,
                "usage_pct": used_pct,
                "reset_description": reset_desc,
                "raw": data,
            }
        if isinstance(data, dict):
            return {
                "status": "available",
                "source": "codexbar",
                "limit": data.get("limit", data.get("total", 0)),
                "used": data.get("used", data.get("consumed", 0)),
                "remaining": data.get("remaining", data.get("remaining_credits", 0)),
                "raw": data,
            }
        return {"status": "available", "source": "codexbar", "raw": data}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        return {"status": "unavailable", "reason": str(e)}


# ── CC-Switch DB integration ────────────────────────────────────────────────


# Default cc-switch DB path
_DEFAULT_CC_SWITCH_DB = str(Path.home() / "SharedConf" / "CC_Switch" / "cc-switch.db")


def _find_cc_switch_db() -> str | None:
    """Find cc-switch DB across possible locations — prefer largest (most complete)."""
    candidates = [
        Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "SharedConf" / "CC_Switch" / "cc-switch.db",
        Path.home() / ".cc-switch" / "cc-switch.db",
        Path(_DEFAULT_CC_SWITCH_DB),
    ]
    env_path = os.environ.get("BOS_CC_SWITCH_DB", "")
    if env_path:
        candidates.insert(0, Path(env_path))
    # Prefer the largest file (iCloud sync > local copy)
    found = [(str(p), p.stat().st_size) for p in candidates if p.exists() and p.stat().st_size > 0]
    if not found:
        return None
    found.sort(key=lambda x: -x[1])  # largest first
    return found[0][0]


def import_from_cc_switch(db_path: str | None = None) -> int:
    """Import credentials from cc-switch SQLite database.

    Reads the ``providers`` table from cc-switch's DB and imports
    any credentials with API keys into CredentialsManager.

    Also imports model pricing data from ``model_pricing`` table.

    Args:
        db_path: Path to cc-switch SQLite DB. Falls back to
                 ``BOS_CC_SWITCH_DB`` env var, then default path
                 ``~/SharedConf/CC_Switch/cc-switch.db``.

    Returns:
        Number of credentials imported.
    """
    global _cc_switch_importing
    try:
        _cc_switch_importing  # type: ignore[reportUnboundVariable]
    except NameError:
        _cc_switch_importing = False
    if _cc_switch_importing:
        return 0
    _cc_switch_importing = True
    try:
        return _import_cc_switch_impl(db_path)
    finally:
        _cc_switch_importing = False


def _import_cc_switch_impl(db_path: str | None = None) -> int:
    """Internal implementation of cc-switch import.

    Extracts ALL credential types from cc-switch providers (not just ANTHROPIC/OPENAI),
    imports model_pricing data, and handles iCloud-synced DBs.
    """
    path = db_path or _find_cc_switch_db()
    if not path:
        _log.info("cc-switch DB not found in any known location")
        return 0

    count = 0
    try:
        with _get_connection(path) as conn:
            rows = conn.execute(
                "SELECT name, settings_config, website_url FROM providers"
            ).fetchall()

        cm = CredentialsManager()
        for name, settings_json, _website_url in rows:
            if not settings_json:
                continue
            try:
                settings = json.loads(settings_json)
                env = settings.get("env", {})

                # Extract API key from any env var matching key patterns
                auth_token = ""
                base_url = ""
                for k, v in env.items():
                    ku = k.upper()
                    if not auth_token and ("KEY" in ku or "TOKEN" in ku or "AUTH" in ku) and "TIMEOUT" not in ku:
                        auth_token = v
                    if not base_url and ("BASE_URL" in ku or "API_BASE" in ku):
                        base_url = v

                # Also check non-env credential fields
                if not auth_token and "apiKey" in settings:
                    auth_token = settings["apiKey"]
                if not base_url and "apiBaseUrl" in settings:
                    base_url = settings["apiBaseUrl"]

                if auth_token:
                    provider_key = name.lower().replace(" ", "_").replace("-", "_").split("/")[0]
                    cm.add_key(provider_key, auth_token, base_url=base_url, note=f"from cc-switch: {name}")
                    count += 1
            except (json.JSONDecodeError, Exception) as exc:
                _log.debug("cc-switch import error for %s: %s", name, exc)
                continue

        # Import model pricing data
        try:
            from .pricing import ModelPrice
            with _get_connection(path) as conn:
                pricing_rows = conn.execute(
                    "SELECT model_id, display_name, input_cost_per_million, output_cost_per_million "
                    "FROM model_pricing"
                ).fetchall()
            pr = PricingRegistry()
            for model_id, _display_name, in_cost, out_cost in pricing_rows:
                pr.register(ModelPrice(
                    model_id=model_id,
                    display_name=_display_name or model_id,
                    cost_per_1k_input=float(in_cost or 0) / 1000.0,
                    cost_per_1k_output=float(out_cost or 0) / 1000.0,
                ))
            _log.info("cc-switch import: %d model prices synced", len(pricing_rows))
        except (sqlite3.Error, Exception) as exc:
            _log.debug("cc-switch pricing import skipped: %s", exc)

        _log.info("cc-switch import: %d credentials from %s", count, path)
    except (sqlite3.Error, OSError) as e:
        _log.warning("cc-switch import failed: %s", e)
    return count


class CredentialsManager:
    """SQLite-backed credential and quota manager.

    Thread-safe. Replaces environment variable management for API keys.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._pricing = PricingRegistry()
        self._init_db()
        self._migrate_env_vars()
        # cc-switch import is NOT automatic — use import_from_cc_switch() explicitly

    def _init_db(self) -> None:
        """Initialize schema."""
        with _get_connection(self._db_path) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    base_url TEXT DEFAULT '',
                    weight INTEGER DEFAULT 100,
                    is_active INTEGER DEFAULT 1,
                    note TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_cred_provider
                ON credentials(provider)
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    provider TEXT PRIMARY KEY,
                    monthly_limit REAL NOT NULL DEFAULT 0.0,
                    action TEXT NOT NULL DEFAULT 'warn',
                    month TEXT NOT NULL DEFAULT '',
                    month_spend REAL NOT NULL DEFAULT 0.0
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    cost REAL NOT NULL DEFAULT 0.0,
                    timestamp REAL NOT NULL
                )
            """)

    def _migrate_env_vars(self) -> None:
        """Auto-import credentials from environment variables on first run."""
        with _get_connection(self._db_path) as conn:
            c = conn.cursor()
            existing = c.execute("SELECT COUNT(*) FROM credentials").fetchone()[0]
            if existing > 0:
                return

            env_map = {
                "openai": ("OPENAI_API_KEY", ""),
                "anthropic": ("ANTHROPIC_API_KEY", ""),
                "gemini": ("GOOGLE_API_KEY", ""),
                "deepseek": ("DEEPSEEK_API_KEY", ""),
                "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
            }
            now = time.time()
            for provider, (key_env, url_env) in env_map.items():
                api_key = os.environ.get(key_env, "")
                if api_key:
                    base_url = os.environ.get(url_env, "") if url_env else ""
                    c.execute(
                        """INSERT INTO credentials
                           (provider, api_key, base_url, weight, is_active, created_at)
                           VALUES (?, ?, ?, 100, 1, ?)""",
                        (provider, api_key, base_url, now),
                    )
                    _log.info("Migrated %s credential from env", provider)

    # ── Credential CRUD ──────────────────────────────────────────────────────

    def add_key(
        self,
        provider: str,
        api_key: str,
        base_url: str = "",
        weight: int = 100,
        note: str = "",
    ) -> None:
        """Add an API key for a provider.

        Idempotent on (provider, api_key): re-adding the same pair updates its
        metadata in place instead of inserting a duplicate row. 2026-08-23:
        import_from_cc_switch() calls this unconditionally on every gateway
        start with no de-dup, which had produced 748 duplicate rows for a
        single kimi_for_coding key over ~2 months of restarts before this was
        caught by chance while debugging an unrelated routing issue.
        """
        with self._lock:
            with _get_connection(self._db_path) as conn:
                existing = conn.execute(
                    "SELECT id FROM credentials WHERE provider = ? AND api_key = ?",
                    (provider, api_key),
                ).fetchone()
                if existing is not None:
                    conn.execute(
                        "UPDATE credentials SET base_url = ?, weight = ?, note = ?, is_active = 1 WHERE id = ?",
                        (base_url, weight, note, existing[0]),
                    )
                    return
                conn.execute(
                    """INSERT INTO credentials
                       (provider, api_key, base_url, weight, is_active, note, created_at)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (provider, api_key, base_url, weight, note, time.time()),
                )

    def remove_key(self, provider: str, api_key: str) -> bool:
        """Remove a specific key."""
        with self._lock:
            with _get_connection(self._db_path) as conn:
                conn.execute("DELETE FROM credentials WHERE provider = ? AND api_key = ?", (provider, api_key))
                return conn.total_changes > 0

    def mark_key_active(self, provider: str, api_key: str, active: bool) -> bool:
        """标记某个 key 的存活状态(401 验证死/复验活)。

        get_key 只在 is_active=1 的行里选, 死 key 标 0 后自动出局。
        返回是否有行被更新。
        """
        with self._lock:
            with _get_connection(self._db_path) as conn:
                conn.execute(
                    "UPDATE credentials SET is_active = ? WHERE provider = ? AND api_key = ?",
                    (1 if active else 0, provider, api_key),
                )
                return conn.total_changes > 0

    def reverify_provider_keys(self, timeout: float = 8.0) -> dict[str, int]:
        """对所有多 key provider 逐 key 验证(GET base_url/models), 标活/死。

        判定保守: 仅 401/403 判死(明确凭据失效), 200 且清单非空判活,
        其余(404/超时/网络错)不动 —— 网关没实现 models 端点或暂时网络
        问题不该淘汰可能好的 key。单 key provider 的 key 死活由真实请求
        的 401 反馈处理(此处验也会因无对照而无收益)。
        返回 {provider: dead_count}。
        """
        import httpx

        dead: dict[str, int] = {}
        with _get_connection(self._db_path) as conn:
            # 2026-08-23: 从"仅多 key provider"扩展为全量 —— 单 key 过期(如
            # kimi)同样 401 判死; 死 key 出局后 get_key 返回 None, provider
            # is_available 转 False, discover 不再注册该引擎模型(见
            # SSOTProviderAdapter.discover 准入), scheduler 不再选中死模型。
            rows = conn.execute(
                "SELECT DISTINCT provider, api_key, base_url FROM credentials WHERE is_active = 1"
            ).fetchall()
        for provider, api_key, base_url in rows:
            if not base_url:
                continue
            url = f"{str(base_url).rstrip('/')}/models"
            try:
                resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
            except Exception:
                continue
            if resp.status_code in (401, 403):
                if self.mark_key_active(str(provider), str(api_key), active=False):
                    dead[str(provider)] = dead.get(str(provider), 0) + 1
                    _log.warning("credential disabled: %s key failed auth (401/403)", provider)
            elif resp.status_code == 200:
                # 曾被标死的 key 复验通过则复活(自动恢复)
                self.mark_key_active(str(provider), str(api_key), active=True)
        return dead

    def budget_blocked(self, provider: str) -> bool:
        """该凭据名是否已超月预算且动作为 block(请求路径轻量查询, 不走 codexbar)。

        codexbar 遗产的"预算拦截"此前从未接线: budgets 表有配置(如
        deepseek 月限 $50/block)但无任何请求路径读取它 —— 超支无保护。
        """
        with self._lock:
            with _get_connection(self._db_path) as conn:
                row = conn.execute(
                    "SELECT monthly_limit, month_spend FROM budgets "
                    "WHERE provider = ? AND action = 'block'",
                    (provider,),
                ).fetchone()
        if not row:
            return False
        limit, spend = row
        return bool(limit) and (spend or 0.0) >= float(limit)

    def get_key(self, provider: str) -> str | None:
        """Get an API key for *provider*, with weighted random selection.

        Supports multi-key rotation: keys with higher ``weight`` are
        more likely to be returned.
        """
        with self._lock:
            with _get_connection(self._db_path) as conn:
                rows = conn.execute(
                    """SELECT * FROM credentials
                       WHERE provider = ? AND is_active = 1
                       ORDER BY weight DESC""",
                    (provider,),
                ).fetchall()

        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]["api_key"]

        # Weighted random selection (non-cryptographic use OK)
        import random

        total_weight = sum(r["weight"] for r in rows)
        r = random.randint(0, total_weight - 1)  # noqa: S311 (weighted load balance)
        for row in rows:
            r -= row["weight"]
            if r < 0:
                return row["api_key"]
        return rows[-1]["api_key"]

    def list_keys(self, provider: str = "") -> list[dict[str, Any]]:
        """List all stored credentials."""
        with self._lock:
            with _get_connection(self._db_path) as conn:
                if provider:
                    rows = conn.execute("SELECT * FROM credentials WHERE provider = ?", (provider,)).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM credentials").fetchall()
        return [
            {
                "provider": r["provider"],
                "key_preview": r["api_key"][:8] + "..." if len(r["api_key"]) > 8 else "***",
                "weight": r["weight"],
                "active": bool(r["is_active"]),
                "note": r["note"] if r["note"] else "",
            }
            for r in rows
        ]

    # ── Budget / Quota ───────────────────────────────────────────────────────

    def set_budget(self, provider: str, monthly_limit: float, action: str = "warn") -> None:
        """Set monthly budget for a provider.

        Args:
            provider: Provider name.
            monthly_limit: Monthly spending limit in USD.
            action: What to do when exceeded — ``block``, ``warn``, or ``log``.
        """
        with self._lock:
            with _get_connection(self._db_path) as conn:
                current_month = datetime.now().strftime("%Y-%m")
                conn.execute(
                    """INSERT OR REPLACE INTO budgets
                       (provider, monthly_limit, action, month, month_spend)
                       VALUES (?, ?, ?, COALESCE(
                           (SELECT month FROM budgets WHERE provider = ?), ?), 0)""",
                    (provider, monthly_limit, action, provider, current_month),
                )

    def record_usage(
        self, provider: str, cost: float, model: str = "", tokens_input: int = 0, tokens_output: int = 0
    ) -> None:
        """Record a usage event and update budget tracking."""
        with self._lock:
            with _get_connection(self._db_path) as conn:
                # Usage log
                conn.execute(
                    """INSERT INTO usage_log
                       (provider, model, tokens_input, tokens_output, cost, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (provider, model, tokens_input, tokens_output, cost, time.time()),
                )
                # Update monthly spend
                current_month = datetime.now().strftime("%Y-%m")
                conn.execute(
                    "UPDATE budgets SET month_spend = month_spend + ?, month = ? WHERE provider = ? AND month = ?",
                    (cost, current_month, provider, current_month),
                )

    def get_quota(self, provider: str, use_codexbar: bool = True) -> dict[str, Any]:
        """Get quota status for a provider.

        Tries codexbar CLI first (real-time quota), falls back to
        local budget tracking.

        Args:
            provider: Provider name.
            use_codexbar: If True (default), try codexbar first.

        Returns:
            Dict with ``provider``, ``limit``, ``used``, ``remaining``,
            ``usage_pct``, and ``source`` (``"codexbar"`` or ``"local"``).
        """
        # Try codexbar first
        if use_codexbar and codexbar_available():
            cq = fetch_codexbar_quota(provider)
            if cq.get("status") == "available":
                limit = cq.get("limit", 0) or 0
                used = cq.get("used", 0) or 0
                remaining = cq.get("remaining", 0) or max(0, limit - used)
                return {
                    "provider": provider,
                    "source": "codexbar",
                    "limit": limit,
                    "used": used,
                    "remaining": remaining,
                    "usage_pct": round((used / limit * 100) if limit > 0 else 0, 1),
                    "reset_description": cq.get("reset_description", ""),
                }

        # Fallback to local budget tracking
        with self._lock:
            with _get_connection(self._db_path) as conn:
                current_month = datetime.now().strftime("%Y-%m")
                row = conn.execute(
                    "SELECT monthly_limit, action, month_spend FROM budgets WHERE provider = ?",
                    (provider,),
                ).fetchone()

        if not row:
            return {"provider": provider, "source": "local", "status": "unlimited"}

        limit, action, spend = row
        spend = spend or 0.0
        remaining = max(0.0, limit - spend)
        return {
            "provider": provider,
            "source": "local",
            "monthly_limit": limit,
            "spend": round(spend, 4),
            "remaining": round(remaining, 4),
            "usage_pct": round((spend / limit * 100) if limit > 0 else 0, 1),
            "action": action,
            "month": current_month,
        }

    def check_constraint(self, provider: str, estimated_cost: float = 0.0) -> dict[str, Any]:
        """Check if a request would violate budget constraints.

        Returns:
            Dict with ``allowed`` (bool), ``reason`` (str), and
            ``quota`` (dict).
        """
        quota = self.get_quota(provider)
        if quota.get("status") == "unlimited":
            return {"allowed": True, "reason": "unlimited", "quota": quota}

        would_exceed = (quota["spend"] + estimated_cost) > quota["monthly_limit"]

        if not would_exceed:
            return {"allowed": True, "reason": "within_budget", "quota": quota}

        action = quota.get("action", "warn")
        if action == "block":
            return {
                "allowed": False,
                "reason": f"Monthly budget ${quota['monthly_limit']:.2f} exceeded",
                "quota": quota,
            }
        elif action == "warn":
            return {"allowed": True, "reason": f"Warning: {quota['usage_pct']:.0f}% budget used", "quota": quota}
        else:
            return {"allowed": True, "reason": "budget_exceeded_logged", "quota": quota}

    # ── CLI-friendly ─────────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all credentials and quotas."""
        keys = self.list_keys()
        providers = list(set(k["provider"] for k in keys))

        quotas = {}
        for p in providers:
            q = self.get_quota(p)
            if q.get("status") != "unlimited":
                quotas[p] = q

        return {
            "total_keys": len(keys),
            "providers": providers,
            "keys": keys,
            "quotas": quotas,
        }
