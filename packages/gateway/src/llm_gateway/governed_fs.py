"""受治理的文件写入 helper —— 与 `ecos.common.governed_fs` 同构。

存在理由: 根仓 contract gatekeeper (`projects/ecos/scripts/contract_gatekeeper.py`)
禁止业务代码直接对 `.omo/` 路径做 `mkdir` / `write_text` / `open(..., 'a')`。
把变更收拢到本模块的 helper (形参名 `path`, 不携带 forbidden 字面量), 既过门禁,
也让「共享运行态账本怎么写」在全仓只有一处实现。

与 ecos 那份**刻意保持同构但不共用**: aetherforge 在依赖方向上低于 ecos
(ecos 依赖本仓), 反向引依赖会成环。
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any


def ensure_text_file(path: Path, initial_text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(initial_text, encoding="utf-8")


def append_jsonl_record(path: Path, entry: dict[str, Any]) -> None:
    """追加一条 JSONL 记录, 写期间持排他锁。

    X2 budget integrity standard §3「并发控制」: 所有 JSONL 写入必须通过
    `fcntl.flock` 取排他锁 —— 没有锁时两个进程的写入可能交错, 产出半行 JSON
    (后续按行解析会读到坏记录)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
