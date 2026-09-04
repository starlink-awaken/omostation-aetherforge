"""免费算力池发现闭环 — 内建于 gateway 本体(治理 P1.2)。

背景(2026-08-24): 原 free-model-scanner.py / provider-sync.py 外挂脚本
在多 agent 并发中永久丢失(磁盘+git 历史均无, omostation AGENT-BRIEF
§1.1 失败模式的又一受害者)。教训固化为防腐规则"能力内建优于外挂"
(GOVERNANCE-V1 §6.3): 承载持续机制的运维能力必须进仓库+进测试。

本模块实现"发现"这一环(闭环中最易丢失的部分):
- FREE_POOL_CANDIDATES: 候选免费源清单(代码即文档, 增删走 PR)
- FreePoolScanner.scan(): 对候选做无鉴权 models 探测, 与上次快照
  diff, 新源/新模型出现 → emit provider_discovered 事件
- 凭据与接线仍由人决策: 发现只发事件不自动注册(凭据必须人来),
  events.jsonl 的 provider_discovered 是运营的输入信号

探测判据保守: 仅记录"能拿到非空模型清单"的源; 401/403 记
needs_key=True(源活着但需要凭据, 换 key 提示的输入); 网络错不动。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import emit

_log = logging.getLogger(__name__)

# 候选免费源 SSOT(2026-08-24 起维护): name → 无鉴权探测端点。
# 增删候选走 PR —— 这是防腐不变量 §3 的直接应用。
FREE_POOL_CANDIDATES: dict[str, str] = {
    "openrouter-free": "https://openrouter.ai/api/v1/models",
    "nvidia-free": "https://integrate.api.nvidia.com/v1/models",
    "opencode-zen": "https://opencode.ai/zen/v1/models",
    "siliconflow-free": "https://api.siliconflow.cn/v1/models",
    "glm-free": "https://open.bigmodel.cn/api/paas/v4/models",
    "deepseek": "https://api.deepseek.com/v1/models",
}


@dataclass
class PoolSnapshot:
    """一次扫描的源状态。"""

    provider: str
    reachable: bool = False
    needs_key: bool = False
    model_count: int = 0
    sample_models: list[str] = field(default_factory=list)


class FreePoolScanner:
    """周期扫描候选源, diff 出新信号发事件(线程化调用, 自带内存状态)。"""

    _STATE_FILE = Path.home() / ".aetherforge" / "state" / "free_pool_last_seen.json"

    def __init__(self, candidates: dict[str, str] | None = None) -> None:
        self._candidates = candidates if candidates is not None else FREE_POOL_CANDIDATES
        self._last: dict[str, PoolSnapshot] = {}
        self._seen: set[str] = self._load_seen()

    @classmethod
    def _load_seen(cls) -> set[str]:
        """持久化 last-seen(2026-08-24): 内存态在重启后清零导致
        first_seen 事件重复(当日日报 openrouter-free 出现两遍)。"""
        import json

        try:
            return set(json.loads(cls._STATE_FILE.read_text()))
        except Exception:
            return set()

    def _save_seen(self) -> None:
        import json

        try:
            self._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._STATE_FILE.write_text(json.dumps(sorted(self._seen)))
        except Exception as exc:
            _log.debug("free pool state save failed: %s", exc)

    def _probe(self, name: str, url: str) -> PoolSnapshot:
        import httpx

        snap = PoolSnapshot(provider=name)
        try:
            resp = httpx.get(url, timeout=8)
        except Exception as exc:
            _log.debug("free pool probe %s failed: %s", name, exc)
            return snap
        if resp.status_code in (401, 403):
            snap.needs_key = True
            return snap
        if resp.status_code != 200:
            return snap
        try:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            return snap
        if not models:
            return snap
        snap.reachable = True
        snap.model_count = len(models)
        snap.sample_models = sorted(models)[:5]
        return snap

    def scan(self) -> dict[str, Any]:
        """全量探测 + diff。返回摘要(调用方可记日志), 事件经 emit 发出。"""
        summary: dict[str, Any] = {"probed": len(self._candidates), "reachable": 0, "new_signals": 0}
        for name, url in self._candidates.items():
            snap = self._probe(name, url)
            if snap.reachable:
                summary["reachable"] += 1
            prev = self._last.get(name)
            self._last[name] = snap
            # 新信号: 首次可达 / 模型数明显增长(>20% 且 >=5 个新)
            first_ever = name not in self._seen
            is_new = first_ever or (prev is None or (snap.reachable and not prev.reachable))
            grew = (
                prev is not None
                and snap.reachable
                and snap.model_count - prev.model_count >= 5
            )
            if snap.reachable:
                self._seen.add(name)
            if snap.reachable and (is_new or grew):
                summary["new_signals"] += 1
                emit(
                    "provider_discovered",
                    {
                        "provider": name,
                        "model_count": snap.model_count,
                        "sample": snap.sample_models,
                        "reason": "first_seen" if is_new else "models_grew",
                    },
                )
        # 无条件落盘(2026-08-24): 此前只在有新信号时写, 无信号轮次不触碰
        # 状态文件 → full-status 的心跳检查判不出"scan 活着"。幂等数据每轮
        # 重写, mtime 即心跳。
        self._save_seen()
        return summary


# ---------------------------------------------------------------------------
# openrouter free 清单刷新 (2026-08-24)
#
# 背景: MODEL-BREW-OPENROUTER-FREE.yaml 是 2026-08-09 手写的静态快照,
# 之后无刷新机制 —— 实测当日漂移 5→19 个真免费 chat 模型(z-ai/glm-5.2:free
# 等上线半个月无人知晓)。另外 FreePoolScanner 顶部曾漏 import Path 导致
# scan 自诞生起静默 NameError(gateway 的 except 吞成 debug 日志), 本轮
# 一并修复并补测试。能力内建: 拉 API → 过滤 → diff → dry-run/--write。
# ---------------------------------------------------------------------------

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _is_free_chat_model(model: dict) -> bool:
    """真免费(prompt+completion 均 0)且纯文本输出 —— 排除 lyria 音乐
    (out 含 audio)、content-safety 审核(分类器非 chat)这类名义免费但
    非通用对话的条目。openrouter/free(官方免费 router)保留: 它把请求
    自动路由到免费池, 对 free 池有直接价值。"""
    pricing = model.get("pricing") or {}
    try:
        prompt_free = float(pricing.get("prompt", "1") or 1) == 0
        completion_free = float(pricing.get("completion", "1") or 1) == 0
    except (TypeError, ValueError):
        return False
    if not (prompt_free and completion_free):
        return False
    modalities = (model.get("architecture") or {}).get("output_modalities") or ["text"]
    if modalities != ["text"]:
        return False
    model_id = model.get("id") or ""
    return "content-safety" not in model_id


def refresh_openrouter_free(*, write: bool = False) -> dict[str, Any]:
    """拉 openrouter 实时免费清单, 与 MODEL-BREW-OPENROUTER-FREE.yaml diff。

    write=False: 只报告漂移(默认)。write=True: 更新 yaml 的 models 段 +
    synced_at, steward 标记 free-pool-refresh 供追溯。
    外网访问走 httpx 默认代理环境变量(本机实测需 http_proxy 指向 7890)。
    """
    import httpx

    from .paths import M1_MODEL_DIR

    yaml_path = M1_MODEL_DIR / "MODEL-BREW-OPENROUTER-FREE.yaml"
    try:
        resp = httpx.get(OPENROUTER_MODELS_URL, timeout=30)
        resp.raise_for_status()
        remote_models = [m for m in resp.json().get("data", []) if _is_free_chat_model(m)]
    except Exception as exc:
        return {"ok": False, "error": f"openrouter models 拉取失败: {exc}"}

    remote_ids = sorted(m["id"] for m in remote_models)
    static_ids: set[str] = set()
    doc: dict[str, Any] | None = None
    if yaml_path.exists():
        import yaml

        doc = yaml.safe_load(yaml_path.read_text()) or {}
        static_ids = {m.get("model_id", "") for m in doc.get("models", []) if m.get("model_id")}

    added = [i for i in remote_ids if i not in static_ids]
    removed = sorted(static_ids - set(remote_ids))

    result: dict[str, Any] = {
        "ok": True,
        "remote_free": len(remote_ids),
        "static": len(static_ids),
        "added": added,
        "removed": removed,
        "written": False,
    }
    emit("free_pool_drift", {"provider": "openrouter-free", "added": added, "removed": removed})

    if write and doc is not None:
        from datetime import UTC, datetime

        import yaml

        by_id = {m["id"]: m for m in remote_models}
        doc["models"] = [
            {
                "model_id": mid,
                "display_name": ((by_id[mid].get("name") or mid).removesuffix(":free")) + " (Free)",
                "cost_per_1k_input": 0,
                "cost_per_1k_output": 0,
                "context_window": by_id[mid].get("context_length") or 131072,
                "capabilities": ["chat"],
            }
            for mid in remote_ids
        ]
        doc.setdefault("model_driven_refs", {})
        doc["model_driven_refs"]["synced_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        doc["model_driven_refs"]["source"] = "openrouter_api_free_tier"
        doc.setdefault("governance", {})["steward"] = "free-pool-refresh"
        yaml_path.write_text(
            yaml.safe_dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )
        result["written"] = True

    return result
