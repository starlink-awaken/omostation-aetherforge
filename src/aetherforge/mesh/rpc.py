"""AetherForge Mesh — BOS RPC 入口 (internal 同进程直调)。

暴露: bos://capability/compute/mesh-status
返回算力网格各节点的健康与在线状态 (含 omlx / tailnet 节点)。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# 补齐 internal 同进程调用时子包的 sys.path
_af_dir = Path(__file__).resolve().parents[3]
_p = str(_af_dir / "packages" / "mesh" / "src")
if _p not in sys.path:
    sys.path.insert(0, _p)

from compute_mesh.pool import ComputePool

_log = logging.getLogger(__name__)


def run_mesh_status(**kwargs: Any) -> dict[str, Any]:
    """BOS RPC 入口: bos://capability/compute/mesh-status.

    主动探活所有节点并返回在线/离线摘要 (供 cockpit/agora 展示算力大盘)。
    """
    try:
        pool = ComputePool()
        pool.scan()
        results = pool.health_check_all()
        online = sum(1 for v in results.values() if v)
        return {
            "status": "success",
            "total": len(results),
            "online": online,
            "offline": len(results) - online,
            "nodes": [{"id": nid, "online": bool(alive)} for nid, alive in results.items()],
        }
    except Exception as e:
        _log.warning("[Compute RPC] mesh-status failed: %s", e)
        return {"status": "failed", "error": str(e)}
