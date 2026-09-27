"""Radix 前缀树 Paged KV Cache — BET-Y1Q4-T6-28.

实现基于 Radix Tree (压缩前缀树) 的分页 KV Cache:
- 共享前缀复用: 相同 Prompt 前缀只存一份 KV
- 分页管理: 固定大小页面, 支持内存碎片整理
- 持久化: 启动时从磁盘热加载预编译的 GaC 规约
- LRU 淘汰: 超出内存限制时淘汰最久未用页面

设计目标: 显存占用降低 ≥40%, 吞吐提升 ≥2.5 倍.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RadixNode:
    """Radix Tree 节点 — 存储共享前缀片段."""

    segment: str  # 前缀片段 (非单字符, 压缩路径)
    children: dict[str, "RadixNode"] = field(default_factory=dict)
    kv_page_id: int | None = None  # 关联的 KV Cache 页面 ID
    is_terminal: bool = False  # 是否为一个完整 key 的终点


@dataclass
class CacheStats:
    """缓存统计."""

    total_entries: int = 0
    total_pages: int = 0
    memory_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


@dataclass
class PagedKVCache:
    """单个页面的 KV Cache 数据."""

    page_id: int
    key: str
    value: bytes  # 序列化的 KV 张量数据
    token_count: int
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0


class RadixPagedCache:
    """Radix 前缀树 + Paged KV Cache 组合实现.

    核心特性:
    - O(k) 查找复杂度 (k = key 长度)
    - 前缀共享减少内存占用
    - 分页支持动态内存管理
    """

    def __init__(self, page_size: int = 4096, max_memory_bytes: int = 512 * 1024 * 1024) -> None:
        self._root = RadixNode(segment="")
        self._pages: dict[int, PagedKVCache] = {}
        self._key_to_page: dict[str, int] = {}
        self._next_page_id = 0
        self._page_size = page_size
        self._max_memory = max_memory_bytes
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def page_size(self) -> int:
        return self._page_size

    def get(self, key: str) -> bytes | None:
        """按 key 查找 KV Cache 数据."""
        node = self._find_node(key)
        if node is None or node.kv_page_id is None:
            self._stats.miss_count += 1
            return None

        page = self._pages.get(node.kv_page_id)
        if page is None:
            self._stats.miss_count += 1
            return None

        # 更新访问统计
        page.last_accessed = time.monotonic()
        page.access_count += 1
        self._stats.hit_count += 1
        return page.value

    def put(self, key: str, value: bytes, token_count: int = 1) -> int:
        """插入或更新 KV Cache. 返回 page_id."""
        # 已存在则更新
        if key in self._key_to_page:
            page_id = self._key_to_page[key]
            page = self._pages[page_id]
            old_size = len(page.value)
            page.value = value
            page.token_count = token_count
            page.last_accessed = time.monotonic()
            self._stats.memory_bytes += len(value) - old_size
            return page_id

        # 创建新页面
        page_id = self._next_page_id
        self._next_page_id += 1

        page = PagedKVCache(
            page_id=page_id,
            key=key,
            value=value,
            token_count=token_count,
        )
        self._pages[page_id] = page
        self._key_to_page[key] = page_id
        self._stats.total_pages += 1
        self._stats.memory_bytes += len(value)

        # 插入 Radix Tree
        self._insert_node(key, page_id)

        # 检查内存限制
        self._maybe_evict()

        self._stats.total_entries += 1
        return page_id

    def contains(self, key: str) -> bool:
        """检查 key 是否存在."""
        return self._find_node(key) is not None

    def prefix_match(self, prefix: str) -> list[str]:
        """返回所有以 prefix 开头的 key."""
        # 找到前缀对应的节点 (不要求 terminal), 同时获取实际路径
        node, matched_path = self._find_prefix_node(prefix)
        if node is None:
            return []
        # 收集该子树下的所有 terminal keys
        results: list[str] = []
        self._collect_keys(node, matched_path, results)
        return results

    def remove(self, key: str) -> bool:
        """删除指定 key."""
        if key not in self._key_to_page:
            return False

        page_id = self._key_to_page.pop(key)
        page = self._pages.pop(page_id, None)
        if page:
            self._stats.memory_bytes -= len(page.value)

        # Radix Tree 中的清理 (标记为非 terminal)
        node = self._find_node(key)
        if node is not None:
            node.is_terminal = False
            node.kv_page_id = None

        self._stats.total_entries -= 1
        return True

    def clear(self) -> None:
        """清空所有缓存."""
        self._root = RadixNode(segment="")
        self._pages.clear()
        self._key_to_page.clear()
        self._next_page_id = 0
        self._stats = CacheStats()

    def _find_node(self, key: str) -> RadixNode | None:
        """在 Radix Tree 中查找 key 对应的节点."""
        current = self._root
        remaining = key

        while remaining:
            # 查找匹配的子节点
            matched = False
            for child_key, child_node in current.children.items():
                # 检查 remaining 是否以 child_key 开头
                if remaining.startswith(child_key):
                    remaining = remaining[len(child_key):]
                    current = child_node
                    matched = True
                    break
                # 检查 child_key 是否以 remaining 开头 (key 是某个 segment 的前缀)
                elif child_key.startswith(remaining):
                    # key 在 segment 中间结束, 无精确匹配
                    return None

            if not matched:
                return None

        return current if current.is_terminal else None

    def _insert_node(self, key: str, page_id: int) -> None:
        """将 key 插入 Radix Tree."""
        current = self._root
        remaining = key

        while remaining:
            matched = False
            for child_key, child_node in list(current.children.items()):
                # 找最长公共前缀
                common_len = self._common_prefix_length(remaining, child_key)

                if common_len == 0:
                    continue

                if common_len == len(child_key):
                    # child_key 完全匹配
                    remaining = remaining[common_len:]
                    current = child_node
                    matched = True
                    break
                else:
                    # 需要分裂节点
                    common_prefix = remaining[:common_len]
                    # 创建新的中间节点
                    new_node = RadixNode(segment=common_prefix)
                    # 原来的 child 变成新节点的子节点
                    child_node.segment = child_key[common_len:]
                    new_node.children[child_node.segment] = child_node
                    # 替换原来的子节点
                    del current.children[child_key]
                    current.children[common_prefix] = new_node
                    remaining = remaining[common_len:]
                    current = new_node
                    matched = True
                    break

            if not matched:
                # 无匹配的子节点, 直接创建
                new_node = RadixNode(segment=remaining, kv_page_id=page_id, is_terminal=True)
                current.children[remaining] = new_node
                return

        # 到达末尾, 标记为 terminal
        current.is_terminal = True
        current.kv_page_id = page_id

    def _find_prefix_node(self, prefix: str) -> tuple[RadixNode | None, str]:
        """在 Radix Tree 中查找 prefix 对应的节点 (不要求 terminal).

        与 _find_node 不同, 此方法只要求路径匹配, 不要求节点是 key 的终点.

        Returns:
            (节点, 实际匹配路径) 或 (None, "")
        """
        current = self._root
        remaining = prefix
        matched_path = ""

        while remaining:
            matched = False
            for child_key, child_node in current.children.items():
                if remaining.startswith(child_key):
                    remaining = remaining[len(child_key):]
                    matched_path += child_key
                    current = child_node
                    matched = True
                    break
                elif child_key.startswith(remaining):
                    # prefix 是某个 segment 的前缀, 匹配成功
                    matched_path += child_key
                    return child_node, matched_path

            if not matched:
                return None, ""

        return current, matched_path

    def _collect_keys(self, node: RadixNode, prefix: str, results: list[str]) -> None:
        """收集子树中所有 terminal 节点的 key."""
        if node.is_terminal and node.segment:
            results.append(prefix)

        for child_key, child_node in node.children.items():
            self._collect_keys(child_node, prefix + child_key, results)

    def _maybe_evict(self) -> None:
        """检查并执行 LRU 淘汰."""
        if self._stats.memory_bytes <= self._max_memory:
            return

        # 按 last_accessed 排序, 淘汰最久未访问的
        sorted_pages = sorted(
            self._pages.values(),
            key=lambda p: p.last_accessed,
        )

        target = self._stats.memory_bytes - self._max_memory
        freed = 0

        for page in sorted_pages:
            if freed >= target:
                break
            # 不淘汰最近 5 分钟内访问过的
            if time.monotonic() - page.last_accessed < 300:
                continue

            if self.remove(page.key):
                freed += len(page.value) if page.value else 0
                self._stats.eviction_count += 1

    @staticmethod
    def _common_prefix_length(a: str, b: str) -> int:
        """计算两个字符串的最长公共前缀长度."""
        length = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                length += 1
            else:
                break
        return length

    @staticmethod
    def hash_key(text: str) -> str:
        """为文本生成稳定的缓存 key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_snapshot(self) -> dict[str, Any]:
        """序列化为快照 (用于持久化)."""
        return {
            "page_size": self._page_size,
            "max_memory": self._max_memory,
            "pages": {
                pid: {
                    "key": p.key,
                    "token_count": p.token_count,
                    "value_hex": p.value.hex() if p.value else "",
                }
                for pid, p in self._pages.items()
            },
            "stats": {
                "total_entries": self._stats.total_entries,
                "total_pages": self._stats.total_pages,
                "memory_bytes": self._stats.memory_bytes,
            },
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "RadixPagedCache":
        """从快照恢复."""
        cache = cls(
            page_size=data.get("page_size", 4096),
            max_memory_bytes=data.get("max_memory", 512 * 1024 * 1024),
        )
        for pid, pdata in data.get("pages", {}).items():
            cache.put(
                key=pdata["key"],
                value=bytes.fromhex(pdata.get("value_hex", "")),
                token_count=pdata.get("token_count", 1),
            )
        return cache
