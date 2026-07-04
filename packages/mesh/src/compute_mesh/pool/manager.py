"""ComputePool — resource aggregation, health monitoring, and load management.

The pool sits on top of the topology layer, taking discovered nodes
and providing a unified interface for health-checking, load tracking,
and node selection.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ..topology import ComputeNode, NodeRegistry, NodeStatus, TopologyScanner

_log = logging.getLogger(__name__)

# Default health-check timeout in seconds
_DEFAULT_HEALTH_TIMEOUT = 5.0

# Type for pool event listeners
PoolListener = Callable[[str, ComputeNode], None]


def _bus_publish_node_state(node: ComputeNode) -> None:
    """Publish a mesh:node:status_changed event onto bus-foundation.

    Graceful degradation: if bus-foundation is not installed or unavailable,
    the error is logged at DEBUG level and the call silently returns.
    """
    try:
        from bus_foundation import publish as _publish  # type: ignore[import]
        from bus_foundation.envelope import BusEnvelope  # type: ignore[import]

        env = BusEnvelope(
            topic="mesh:node:status_changed",
            source_uri="compute_mesh.pool.manager",
            payload={
                "node_id": node.node_id,
                "status": str(node.status),
                "base_url": getattr(node, "base_url", ""),
                "network_zone": getattr(node, "network_zone", ""),
                "load_factor": getattr(node, "load_factor", 0.0),
            },
        )
        _publish(env)
        _log.debug("bus:mesh:node:status_changed published: node=%s status=%s", node.node_id, node.status)
    except ImportError:
        _log.debug("bus-foundation not available, skipping mesh event publish")
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to publish mesh node state event: %s", exc)


class ComputePool:
    """Aggregates compute nodes from topology, monitors health and load.

    The pool is the main interface for higher layers (scheduler, API)
    to interact with mesh compute resources.

    Usage::

        pool = ComputePool()
        pool.scan()  # discover nodes via topology scanner
        pool.health_check_all()  # probe all nodes
        online = pool.get_online()
    """

    def __init__(
        self,
        registry: NodeRegistry | None = None,
        scanner: TopologyScanner | None = None,
    ) -> None:
        self._registry = registry or NodeRegistry()
        self._scanner = scanner or TopologyScanner(self._registry)
        self._lock = threading.RLock()
        self._listeners: list[PoolListener] = []
        self._health_history: dict[str, list[dict[str, Any]]] = {}
        self._max_history = 100

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def registry(self) -> NodeRegistry:
        return self._registry

    @property
    def scanner(self) -> TopologyScanner:
        return self._scanner

    @property
    def node_count(self) -> int:
        return self._registry.count()

    # ── Discovery ────────────────────────────────────────────────────────────

    def scan(self) -> list[ComputeNode]:
        """Run full topology discovery and return discovered nodes."""
        return self._scanner.scan_all()

    # ── Health checks ────────────────────────────────────────────────────────

    def health_check_node(self, node_id: str) -> bool:
        """Probe a single node's health.

        Performs a TCP-level port check for local daemons, or a simple
        availability check. Returns ``True`` if the node is reachable.

        Updates the node's status and ``last_seen`` timestamp.
        """
        node = self._registry.get(node_id)
        if node is None:
            return False

        is_alive = self._probe_node(node)
        now = time.time()

        with self._lock:
            entry = self._registry.get(node_id)
            if entry is None:
                return False
            entry.last_seen = now
            old_status = entry.status
            entry.status = NodeStatus.ONLINE if is_alive else NodeStatus.OFFLINE
            if old_status != entry.status:
                self._notify("status_change", entry)
            self._record_health(node_id, is_alive)

        return is_alive

    def health_check_all(self) -> dict[str, bool]:
        """Probe all registered nodes. Returns ``{node_id: is_alive}``."""
        import concurrent.futures

        results: dict[str, bool] = {}
        nodes = self._registry.get_all()
        if not nodes:
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(nodes), 20)) as executor:
            future_to_node = {executor.submit(self.health_check_node, node.node_id): node.node_id for node in nodes}
            for future in concurrent.futures.as_completed(future_to_node):
                node_id = future_to_node[future]
                try:
                    results[node_id] = future.result()
                except Exception:  # noqa: BLE001
                    _log.exception("Health check failed for node %s", node_id)
                    results[node_id] = False

        return results

    def wakeup_node(self, node_id: str) -> bool:
        """尝试唤醒离线的物理节点 (Wake-on-LAN)。"""
        node_upper = node_id.upper()
        target_node = None
        if "MACMINI" in node_upper or "MAC-MINI" in node_upper:
            target_node = "mac-mini-M4"
        elif "Y7000P" in node_upper:
            target_node = "Y7000P-4070"

        if not target_node:
            _log.warning("No physical hardware mapping defined for node %s", node_id)
            return False

        import yaml
        try:
            # 动态查找 workspace root
            cur = Path(__file__).resolve()
            workspace_root = None
            for parent in cur.parents:
                if (parent / "docs" / "project-registry.yaml").is_file():
                    workspace_root = parent
                    break
            
            if not workspace_root:
                _log.error("Could not locate workspace root for project-registry.yaml")
                return False

            reg_path = workspace_root / "docs" / "project-registry.yaml"
            data = yaml.safe_load(reg_path.read_text(encoding="utf-8")) or {}
            nodes = data.get("compute_nodes", {})
            cfg = nodes.get(target_node)
            if not cfg or "mac" not in cfg:
                _log.warning("MAC address not registered for node %s in project-registry.yaml", target_node)
                return False

            mac = cfg["mac"]
            lan_ip = cfg.get("lan_ip", "255.255.255.255")

            _log.info("Sending Magic Packet to wake up %s (MAC=%s, LAN_IP=%s)", node_id, mac, lan_ip)
            
            import socket
            clean_mac = mac.replace(":", "").replace("-", "").replace(".", "")
            mac_bytes = bytes.fromhex(clean_mac)
            packet = b'\xff' * 6 + mac_bytes * 16
            
            for port in [9, 7]:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    s.sendto(packet, ("255.255.255.255", port))
                    if lan_ip != "255.255.255.255":
                        s.sendto(packet, (lan_ip, port))
            return True
        except Exception as e:
            _log.exception("Wakeup failed for node %s: %s", node_id, e)
            return False

    def _probe_node(self, node: ComputeNode) -> bool:
        """Low-level node probe. Returns True if reachable."""
        from urllib.parse import urlparse

        if not node.base_url:
            # No URL = assume configured but not yet reachable
            return False

        try:
            parsed = urlparse(node.base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(_DEFAULT_HEALTH_TIMEOUT)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:  # noqa: BLE001
            return False

    def _record_health(self, node_id: str, is_alive: bool) -> None:
        """Record a health check result in the history buffer."""
        if node_id not in self._health_history:
            self._health_history[node_id] = []
        history = self._health_history[node_id]
        history.append({"ts": time.time(), "alive": is_alive})
        # Trim to max history
        if len(history) > self._max_history:
            self._health_history[node_id] = history[-self._max_history :]

    def get_health_history(self, node_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent health check history for a node."""
        history = self._health_history.get(node_id, [])
        return history[-limit:]

    # ── Load tracking ────────────────────────────────────────────────────────

    def assign_request(self, node_id: str) -> bool:
        """Increment active request count for a node.

        Returns ``False`` if the node is at max concurrency.
        """
        node = self._registry.get(node_id)
        if node is None:
            return False
        with self._lock:
            node = self._registry.get(node_id)
            if node is None:
                return False
            if node.active_requests >= node.max_concurrency:
                return False
            node.active_requests += 1
        return True

    def release_request(self, node_id: str) -> bool:
        """Decrement active request count for a node."""
        node = self._registry.get(node_id)
        if node is None:
            return False
        with self._lock:
            node = self._registry.get(node_id)
            if node is None:
                return False
            node.active_requests = max(0, node.active_requests - 1)
        return True

    # ── Selection helpers ────────────────────────────────────────────────────

    def get_online(self) -> list[ComputeNode]:
        """Return all nodes currently marked ONLINE."""
        return self._registry.get_online()

    def get_best_node(self, preferred_zone: str = "") -> ComputeNode | None:
        """Pick the best available node (lowest load, online, highest priority).

        Args:
            preferred_zone: If set, prefer nodes in this network zone.

        Returns:
            The best node, or ``None`` if no online nodes exist.
        """
        candidates = self.get_online()
        if not candidates:
            return None

        # Sort by: zone match → priority → load factor → cost
        def sort_key(n: ComputeNode) -> tuple:
            zone_match = 0 if preferred_zone and n.network_zone == preferred_zone else 1
            return (zone_match, n.priority, n.load_factor, n.effective_cost)

        candidates.sort(key=sort_key)
        return candidates[0]

    # ── Listeners ────────────────────────────────────────────────────────────

    def add_listener(self, listener: PoolListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: PoolListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, event: str, node: ComputeNode) -> None:
        # ── 1. 通知本地 PoolListeners ──
        for listener in self._listeners:
            try:
                listener(event, node)
            except Exception:  # noqa: BLE001
                _log.exception("Pool listener failed for event %s", event)

        # ── 2. 向 bus-foundation 发布 mesh 状态变更事件（R3 闭环） ──
        if event == "status_change":
            _bus_publish_node_state(node)

    # ── Status report ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return a full status snapshot of the pool."""
        online = self.get_online()
        all_nodes = self._registry.get_all()
        return {
            "total_nodes": len(all_nodes),
            "online_nodes": len(online),
            "offline_nodes": len(all_nodes) - len(online),
            "nodes": [n.to_dict() for n in all_nodes],
        }

    def get_summary(self) -> dict[str, Any]:
        """Return a compact summary string."""
        online = self.get_online()
        return {
            "total": self.node_count,
            "online": len(online),
            "offline": self.node_count - len(online),
            "zones": list({n.network_zone for n in self._registry.get_all()}),
        }

    # ── Auto-scaling ─────────────────────────────────────────────────────────

    def auto_scale_workers(
        self,
        worker_registry,
        *,
        min_workers: int = 1,
        max_workers: int = 20,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        workers_per_node: int = 2,
    ) -> dict[str, Any]:
        """Dynamically adjust the number of workers based on load.

        Called periodically (e.g. every 30s) to scale workers up or down.

        Args:
            worker_registry: The worker registry to manage.
            min_workers: Minimum total workers across all nodes.
            max_workers: Maximum total workers across all nodes.
            scale_up_threshold: Fraction of busy workers that triggers scale-up.
            scale_down_threshold: Fraction of busy workers that triggers scale-down.
            workers_per_node: Target workers per online node.

        Returns:
            Dict with ``added``, ``removed``, ``total`` counts.
        """
        from ..worker import TaskDispatcher

        dispatcher = TaskDispatcher(self, worker_registry)
        total_workers = worker_registry.count()
        stats = worker_registry.get_stats()
        busy_ratio = stats["busy"] / max(1, total_workers)

        result: dict[str, Any] = {"added": 0, "removed": 0, "total": total_workers, "reason": "stable"}

        # Scale up: if busy ratio exceeds threshold or too few workers
        if busy_ratio > scale_up_threshold or total_workers < min_workers:
            if total_workers < max_workers:
                new_workers = dispatcher.provision_all(workers_per_node=workers_per_node)
                result["added"] = len(new_workers)
                result["total"] = worker_registry.count()
                result["reason"] = f"scale_up (busy={busy_ratio:.0%})"
            else:
                result["reason"] = "at_max_capacity"

        # Scale down: if busy ratio is below threshold and we have excess
        elif busy_ratio < scale_down_threshold and total_workers > min_workers:
            idle = worker_registry.get_idle()
            to_remove = max(0, len(idle) - min_workers)
            if to_remove > 0:
                for w in idle[:to_remove]:
                    worker_registry.unregister(w.worker_id)
                result["removed"] = to_remove
                result["total"] = worker_registry.count()
                result["reason"] = f"scale_down (busy={busy_ratio:.0%})"

        return result

    def get_load_report(self) -> dict[str, Any]:
        """Return a detailed load report for capacity planning."""
        nodes = self._registry.get_all()
        total_capacity = sum(n.max_concurrency for n in nodes if n.is_online)
        total_load = sum(n.active_requests for n in nodes if n.is_online)
        return {
            "total_nodes": len(nodes),
            "online_nodes": len(self.get_online()),
            "total_capacity": total_capacity,
            "total_load": total_load,
            "utilization": round(total_load / max(1, total_capacity), 3),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "load": n.active_requests,
                    "capacity": n.max_concurrency,
                    "utilization": round(n.active_requests / max(1, n.max_concurrency), 3),
                }
                for n in nodes
            ],
        }


# Import socket at module level for _probe_node
import socket
