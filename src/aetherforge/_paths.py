"""aetherforge._paths — 工作区路径常量统一出口

所有子包（gateway、mesh、swarm）从此处引入 M1 路径常量，
不再各自硬编码。

环境变量覆盖（适用于 CI / 自定义部署路径）：
  AETHERFORGE_M1_DIR           — 覆盖整个 M1 根目录
  AETHERFORGE_M1_COMPUTE_DIR  — 仅覆盖 compute_engine 目录
  LLM_GATEWAY_M1_DIR          — 向后兼容别名（同 AETHERFORGE_M1_COMPUTE_DIR）
"""

from __future__ import annotations

import os
from pathlib import Path

# ── M1 根目录 ─────────────────────────────────────────────────────────────────
_M1_ROOT_OVERRIDE = os.environ.get("AETHERFORGE_M1_DIR", "")

if _M1_ROOT_OVERRIDE:
    M1_ROOT_DIR: Path = Path(_M1_ROOT_OVERRIDE)
else:
    M1_ROOT_DIR = Path.home() / "Workspace" / "projects" / "ecos" / "src" / "ecos" / "ssot" / "mof" / "m1"

# ── M1 子目录（按需添加） ──────────────────────────────────────────────────────
_COMPUTE_OVERRIDE = (
    os.environ.get("AETHERFORGE_M1_COMPUTE_DIR")
    or os.environ.get("LLM_GATEWAY_M1_DIR")  # backward compat
    or ""
)

if _COMPUTE_OVERRIDE:
    M1_COMPUTE_ENGINE_DIR: Path = Path(_COMPUTE_OVERRIDE)
else:
    M1_COMPUTE_ENGINE_DIR = M1_ROOT_DIR / "compute_engine"

M1_COMPUTE_NODE_DIR: Path = M1_ROOT_DIR / "compute_node"
M1_HARDWARE_ASSET_DIR: Path = M1_ROOT_DIR / "hardware_asset"
M1_NETWORK_ZONE_DIR: Path = M1_ROOT_DIR / "network_zone"
M1_MODEL_DIR: Path = M1_ROOT_DIR / "model"
M1_QUOTA_DIR: Path = M1_ROOT_DIR / "quota_definition"
M1_ROUTING_POLICY_DIR: Path = M1_ROOT_DIR / "routing_policy"

__all__ = [
    "M1_ROOT_DIR",
    "M1_COMPUTE_ENGINE_DIR",
    "M1_COMPUTE_NODE_DIR",
    "M1_HARDWARE_ASSET_DIR",
    "M1_NETWORK_ZONE_DIR",
    "M1_MODEL_DIR",
    "M1_QUOTA_DIR",
    "M1_ROUTING_POLICY_DIR",
]
