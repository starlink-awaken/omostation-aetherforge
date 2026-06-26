"""Tests for compute_mesh.scheduler — MeshScheduler queue and status operations."""

from __future__ import annotations


class TestMeshSchedulerQueue:
    def test_enqueue_below_limit(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.scheduler.mesh_scheduler import MeshScheduler

        pool = ComputePool()
        sched = MeshScheduler(pool, max_queue_size=10)
        from llm_gateway.types import ModelRequest

        assert sched.enqueue_request(ModelRequest(task="test")) is True
        assert sched.get_queue_stats()["queued"] == 1

    def test_enqueue_at_limit(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.scheduler.mesh_scheduler import MeshScheduler

        pool = ComputePool()
        sched = MeshScheduler(pool, max_queue_size=2)
        from llm_gateway.types import ModelRequest

        assert sched.enqueue_request(ModelRequest(task="t1")) is True
        assert sched.enqueue_request(ModelRequest(task="t2")) is True
        assert sched.enqueue_request(ModelRequest(task="t3")) is False

    def test_dequeue_ready_no_online_nodes(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.scheduler.mesh_scheduler import MeshScheduler

        pool = ComputePool()
        sched = MeshScheduler(pool, max_queue_size=10)
        from llm_gateway.types import ModelRequest

        sched.enqueue_request(ModelRequest(task="test"))
        ready = sched.dequeue_ready()
        assert len(ready) == 0  # no online nodes with capacity
        assert sched.get_queue_stats()["queued"] == 1

    def test_dequeue_ready_with_online_nodes(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.scheduler.mesh_scheduler import MeshScheduler
        from compute_mesh.topology.node import ComputeNode, NodeStatus

        pool = ComputePool()
        node = ComputeNode(node_id="n1", max_concurrency=4, active_requests=0)
        node.status = NodeStatus.ONLINE
        pool.registry.register(node)

        sched = MeshScheduler(pool, max_queue_size=10)
        from llm_gateway.types import ModelRequest

        sched.enqueue_request(ModelRequest(task="test"))
        ready = sched.dequeue_ready()
        assert len(ready) == 1
        assert sched.get_queue_stats()["queued"] == 0

    def test_get_scheduler_status(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.scheduler.mesh_scheduler import MeshScheduler

        pool = ComputePool()
        sched = MeshScheduler(pool, max_queue_size=10)

        status = sched.get_scheduler_status()
        assert "provider_node_map" in status
        assert "online_nodes" in status
        assert "queue" in status
        assert status["queue"]["max_size"] == 10
