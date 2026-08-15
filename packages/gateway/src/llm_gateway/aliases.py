"""模型别名解析 — 消费者用的名字 → 网关能路由的模型名。

**为什么需要这一层**

消费者(工具、脚本、其他项目)用的是 ``coding`` / ``reasoner`` / ``embed``
这类**意图名**, 而不是 ``mlx-community/Qwen3-...-8bit`` 这类模型 ID。
LiteLLM 一直在提供这层映射; 网关要接管它, 别名就是前置而非可选项 ——
否则消费者切过来的第一件事就是全部改名。

**为什么放在这里**

别名是**调度决策**(哪个意图对应哪个模型), 按分层它属于 llm_gateway,
并由 openai_proxy(HTTP)与 aetherforge.bridge(库)共用同一份 ——
两个入口必须给出相同的路由结果, 否则会出现"库里能路由、HTTP 里不同"
的双份真相。

**不放在 omlxc**: 那是模型生命周期管理面, 让它做名字路由会使它变成
半个网关。

配置来源(按优先级, 先命中先用):
  1. $AETHERFORGE_ALIASES 指定的 YAML
  2. ~/.config/aetherforge/aliases.yaml   (用户覆盖)
  3. <包内>/aliases.yaml                  (仓内默认, 随代码走)
  4. 内置 DEFAULT_ALIASES                 (兜底, 保证无配置也能跑)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# 兜底别名。仅覆盖本机 omlx 后端能直接服务的意图名。
# 远端节点(mac-mini / Y7000P)的别名不在此 —— 网关本身没有这些节点的
# 地址, 跨机路由属 compute_mesh 的职责, 见 aliases.yaml 的 unsupported 段。
DEFAULT_ALIASES: dict[str, str] = {
    "coder": "coding",
    "coder-fast": "coding-fast",
    "reasoner": "reasoning",
    "reasoner-lite": "reasoning-lite",
    "embed": "embedding",
    "deepseek-v4-flash": "qwen-3.5-9b-flash",
    "deepseek-v4-pro": "qwen-3.5-9b-pro",
}


def _candidate_paths() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("AETHERFORGE_ALIASES")
    if env:
        out.append(Path(env).expanduser())
    out.append(Path.home() / ".config" / "aetherforge" / "aliases.yaml")
    out.append(Path(__file__).resolve().parent / "aliases.yaml")
    return out


def load_aliases() -> dict[str, str]:
    """加载别名表。任一来源解析失败都不致命 —— 退到下一个, 最后用内置表。"""
    for p in _candidate_paths():
        if not p.is_file():
            continue
        try:
            import yaml

            data: Any = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            _log.warning("alias config unreadable, skipping %s: %s", p, exc)
            continue
        raw = data.get("aliases") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            _log.warning("alias config has no 'aliases' mapping: %s", p)
            continue
        table = {str(k): str(v) for k, v in raw.items() if k and v}
        _log.info("loaded %d aliases from %s", len(table), p)
        return table
    return dict(DEFAULT_ALIASES)


def resolve(name: str, table: dict[str, str] | None = None, *, _depth: int = 0) -> str:
    """把别名解析成目标名。

    支持别名指向别名(如 chat → coder → coding), 但**限制链长**并检测环 ——
    配置写错时应当退化为原样返回, 而不是无限递归把网关拖死。
    """
    if not name:
        return name
    tbl = DEFAULT_ALIASES if table is None else table
    seen: set[str] = set()
    cur = name
    for _ in range(8):
        nxt = tbl.get(cur)
        if nxt is None or nxt == cur:
            return cur
        if nxt in seen:
            _log.warning("alias cycle detected at %r, keeping %r", nxt, cur)
            return cur
        seen.add(cur)
        cur = nxt
    _log.warning("alias chain too long for %r, keeping %r", name, cur)
    return cur
