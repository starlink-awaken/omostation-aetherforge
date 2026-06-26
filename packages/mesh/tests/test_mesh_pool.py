"""Tests for compute_mesh.pool — ComputePool basic operations."""

from __future__ import annotations


class TestComputePoolBasic:
    def test_init_default(self):
        from compute_mesh.pool.manager import ComputePool

        pool = ComputePool()
        assert pool.node_count == 0
        assert pool.registry is not None
        assert pool.scanner is not None

    def test_register_node(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.topology.node import ComputeNode, NodeStatus

        pool = ComputePool()
        node = ComputeNode(node_id="n1")
        node.status = NodeStatus.ONLINE
        pool.registry.register(node)

        assert pool.node_count == 1
        online = pool.get_online()
        assert len(online) == 1
        assert online[0].node_id == "n1"

    def test_assign_and_release_request(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.topology.node import ComputeNode

        pool = ComputePool()
        node = ComputeNode(node_id="n1", max_concurrency=2)
        pool.registry.register(node)

        assert pool.assign_request("n1") is True
        assert node.active_requests == 1

        assert pool.assign_request("n1") is True
        assert node.active_requests == 2

        # At max concurrency
        assert pool.assign_request("n1") is False

        assert pool.release_request("n1") is True
        assert node.active_requests == 1

    def test_assign_request_nonexistent_node(self):
        from compute_mesh.pool.manager import ComputePool

        pool = ComputePool()
        assert pool.assign_request("nonexistent") is False
        assert pool.release_request("nonexistent") is False

    def test_get_best_node(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.topology.node import ComputeNode, NodeStatus

        pool = ComputePool()

        # Low priority, high load
        n1 = ComputeNode(node_id="n1", priority=5, max_concurrency=4, active_requests=3)
        n1.status = NodeStatus.ONLINE
        n1.cost_per_1k_tokens = {"input": 0.01, "output": 0.02}

        # High priority, low load
        n2 = ComputeNode(node_id="n2", priority=1, max_concurrency=4, active_requests=0)
        n2.status = NodeStatus.ONLINE
        n2.cost_per_1k_tokens = {"input": 0.001, "output": 0.001}

        pool.registry.register(n1)
        pool.registry.register(n2)

        best = pool.get_best_node()
        assert best is not None
        assert best.node_id == "n2"  # higher priority, lower load

    def test_get_status(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.topology.node import ComputeNode, NodeStatus

        pool = ComputePool()
        n1 = ComputeNode(node_id="n1")
        n1.status = NodeStatus.ONLINE
        n2 = ComputeNode(node_id="n2")
        n2.status = NodeStatus.OFFLINE
        pool.registry.register(n1)
        pool.registry.register(n2)

        status = pool.get_status()
        assert status["total_nodes"] == 2
        assert status["online_nodes"] == 1
        assert status["offline_nodes"] == 1

    def test_get_summary(self):
        from compute_mesh.pool.manager import ComputePool
        from compute_mesh.topology.node import ComputeNode, NodeStatus

        pool = ComputePool()
        n1 = ComputeNode(node_id="n1")
        n1.status = NodeStatus.ONLINE
        n1.network_zone = "local"
        pool.registry.register(n1)

        summary = pool.get_summary()
        assert summary["total"] == 1
        assert summary["online"] == 1
        assert "local" in summary["zones"]
