"""Mesh core mechanics: NodeRegistry lifecycle, MeshScheduler queue.

Task F68B791B: AUDIT 1.x thin-test remediation for compute_mesh core.
Covers:
- NodeRegistry: register/unregister, filter, listener notification
- MeshScheduler: enqueue/dequeue, queue stats, max-size overflow
"""

from __future__ import annotations


def _make_node(node_id: str, *, online: bool = True, zone: str = "z1"):
    """Helper to build a ComputeNode for registry/scheduler tests."""
    from compute_mesh.topology.node import ComputeNode, NodeEngineType, NodeStatus, TopologyLabels

    return ComputeNode(
        node_id=node_id,
        name=node_id,
        engine_type=NodeEngineType.LOCAL_DAEMON,
        base_url="http://localhost:11434",
        topology=TopologyLabels(zone=zone),
        status=NodeStatus.ONLINE if online else NodeStatus.OFFLINE,
        protocols=["openai"],
    )


class TestNodeRegistry:
    def test_register_then_get(self):
        from compute_mesh.topology import NodeRegistry

        registry = NodeRegistry()
        node = _make_node("n1")
        assert registry.register(node) is True
        assert registry.count() == 1
        fetched = registry.get("n1")
        assert fetched is not None
        assert fetched.node_id == "n1"

    def test_unregister_removes_node(self):
        from compute_mesh.topology import NodeRegistry

        registry = NodeRegistry()
        registry.register(_make_node("n1"))
        registry.register(_make_node("n2"))
        assert registry.count() == 2
        assert registry.unregister("n1") is True
        assert registry.count() == 1
        assert registry.get("n1") is None

    def test_filter_by_status(self):
        from compute_mesh.topology import NodeRegistry
        from compute_mesh.topology.node import NodeStatus

        registry = NodeRegistry()
        registry.register(_make_node("n1", online=True))
        registry.register(_make_node("n2", online=False))
        registry.register(_make_node("n3", online=True))
        online = registry.filter(status=NodeStatus.ONLINE)
        assert {n.node_id for n in online} == {"n1", "n3"}

    def test_listener_receives_register_unregister(self):
        from compute_mesh.topology import NodeRegistry

        registry = NodeRegistry()
        events: list[tuple[str, str]] = []

        def listener(event: str, node) -> None:
            events.append((event, node.node_id))

        registry.add_listener(listener)
        registry.register(_make_node("n1"))
        registry.unregister("n1")
        assert ("registered", "n1") in events
        assert ("unregistered", "n1") in events


class TestMeshSchedulerQueue:
    def _make_scheduler(self, max_queue_size: int = 5):
        from compute_mesh.pool import ComputePool
        from compute_mesh.scheduler import MeshScheduler

        pool = ComputePool()
        return MeshScheduler(pool, gateway_scheduler=None, max_queue_size=max_queue_size)

    def _make_request(self, task: str = "t1"):
        from llm_gateway.types import ModelRequest

        return ModelRequest(task=task, required_capabilities=[])

    def test_enqueue_then_stats_reports_size(self):
        sched = self._make_scheduler(max_queue_size=5)
        ok = sched.enqueue_request(self._make_request("t1"))
        assert ok is True
        stats = sched.get_queue_stats()
        assert stats["queued"] == 1
        assert stats["max_size"] == 5

    def test_enqueue_returns_false_when_full(self):
        sched = self._make_scheduler(max_queue_size=2)
        assert sched.enqueue_request(self._make_request("a")) is True
        assert sched.enqueue_request(self._make_request("b")) is True
        assert sched.enqueue_request(self._make_request("c")) is False
        assert sched.get_queue_stats()["queued"] == 2

    def test_dequeue_ready_empty_pool_returns_empty(self):
        sched = self._make_scheduler()
        sched.enqueue_request(self._make_request("t1"))
        # No online nodes → no ready requests, item stays queued.
        ready = sched.dequeue_ready()
        assert ready == []
        assert sched.get_queue_stats()["queued"] == 1

    def test_dequeue_ready_drains_when_node_has_capacity(self):
        from compute_mesh.pool import ComputePool
        from compute_mesh.scheduler import MeshScheduler

        pool = ComputePool()
        node = _make_node("n1", online=True)
        pool.registry.register(node)
        sched = MeshScheduler(pool, gateway_scheduler=None, max_queue_size=5)
        sched.enqueue_request(self._make_request("t1"))
        ready = sched.dequeue_ready()
        assert len(ready) == 1
        assert ready[0][0].task == "t1"
        assert sched.get_queue_stats()["queued"] == 0
