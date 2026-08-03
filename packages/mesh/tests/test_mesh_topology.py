"""Tests for compute_mesh.topology — TopologyLabels, ComputeNode, NodeRegistry."""

from __future__ import annotations


class TestTopologyLabels:
    def test_default_labels(self):
        from compute_mesh.topology.node import TopologyLabels

        labels = TopologyLabels()
        assert labels.region == ""
        assert labels.zone == ""
        assert labels.rack == ""
        assert labels.host == ""

    def test_matches_exact(self):
        from compute_mesh.topology.node import TopologyLabels

        a = TopologyLabels(region="us-east-1", zone="us-east-1a")
        b = TopologyLabels(region="us-east-1", zone="us-east-1a")
        assert a.matches(b) is True

    def test_matches_partial(self):
        from compute_mesh.topology.node import TopologyLabels

        a = TopologyLabels(region="us-east-1", zone="us-east-1a")
        b = TopologyLabels(region="us-east-1")  # only region specified
        assert a.matches(b) is True

    def test_matches_different_region(self):
        from compute_mesh.topology.node import TopologyLabels

        a = TopologyLabels(region="us-east-1")
        b = TopologyLabels(region="eu-west-1")
        assert a.matches(b) is False

    def test_affinity_score_exact_match(self):
        from compute_mesh.topology.node import TopologyLabels

        a = TopologyLabels(region="us-east-1", zone="us-east-1a", rack="r1", host="h1")
        b = TopologyLabels(region="us-east-1", zone="us-east-1a", rack="r1", host="h1")
        assert a.affinity_score(b) == 1.0

    def test_affinity_score_partial_match(self):
        from compute_mesh.topology.node import TopologyLabels

        a = TopologyLabels(region="us-east-1", zone="us-east-1a", rack="r1", host="h1")
        b = TopologyLabels(region="us-east-1", zone="us-east-1a")
        assert a.affinity_score(b) == 0.5

    def test_affinity_score_no_match(self):
        from compute_mesh.topology.node import TopologyLabels

        a = TopologyLabels(region="us-east-1", zone="us-east-1a")
        b = TopologyLabels(region="eu-west-1", zone="eu-west-1a")
        assert a.affinity_score(b) == 0.0

    def test_to_dict(self):
        from compute_mesh.topology.node import TopologyLabels

        labels = TopologyLabels(region="us-east-1", zone="us-east-1a", rack="r1", host="h1")
        d = labels.to_dict()
        assert d == {"region": "us-east-1", "zone": "us-east-1a", "rack": "r1", "host": "h1"}


class TestComputeNode:
    def test_default_node(self):
        from compute_mesh.topology.node import ComputeNode

        node = ComputeNode(node_id="n1")
        assert node.node_id == "n1"
        assert node.name == "n1"
        assert node.status.value == "unknown"
        assert node.network_zone == "local"

    def test_node_properties(self):
        from compute_mesh.topology.node import ComputeNode, NodeStatus

        node = ComputeNode(node_id="n1", max_concurrency=4, active_requests=2)
        assert node.load_factor == 0.5
        assert node.is_online is False

        node.status = NodeStatus.ONLINE
        assert node.is_online is True

    def test_effective_cost(self):
        from compute_mesh.topology.node import ComputeNode

        node = ComputeNode(node_id="n1", cost_per_1k_tokens={"input": 0.01, "output": 0.02})
        assert node.effective_cost == 0.03

    def test_to_dict(self):
        from compute_mesh.topology.node import ComputeNode

        node = ComputeNode(node_id="n1", name="Test Node", base_url="http://localhost:11434")
        d = node.to_dict()
        assert d["node_id"] == "n1"
        assert d["name"] == "Test Node"
        assert d["base_url"] == "http://localhost:11434"
        assert d["status"] == "unknown"
        assert "load_factor" in d


class TestNodeRegistry:
    def test_register_and_get(self):
        from compute_mesh.topology.node import ComputeNode
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        node = ComputeNode(node_id="n1")
        assert reg.register(node) is True  # new
        assert reg.count() == 1
        assert reg.get("n1") is node

    def test_register_update(self):
        from compute_mesh.topology.node import ComputeNode
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        node1 = ComputeNode(node_id="n1", name="First")
        reg.register(node1)
        node2 = ComputeNode(node_id="n1", name="Second")
        assert reg.register(node2) is False  # update, not new
        assert reg.get("n1").name == "Second"  # type: ignore[reportOptionalMemberAccess]

    def test_unregister(self):
        from compute_mesh.topology.node import ComputeNode
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        reg.register(ComputeNode(node_id="n1"))
        assert reg.unregister("n1") is True
        assert reg.count() == 0
        assert reg.unregister("n1") is False  # already gone

    def test_filter(self):
        from compute_mesh.topology.node import ComputeNode, NodeStatus
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        n1 = ComputeNode(node_id="n1")
        n1.status = NodeStatus.ONLINE
        n2 = ComputeNode(node_id="n2")
        n2.status = NodeStatus.OFFLINE
        reg.register(n1)
        reg.register(n2)

        online = reg.filter(status=NodeStatus.ONLINE)
        assert len(online) == 1
        assert online[0].node_id == "n1"

    def test_get_online(self):
        from compute_mesh.topology.node import ComputeNode, NodeStatus
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        n1 = ComputeNode(node_id="n1")
        n1.status = NodeStatus.ONLINE
        n2 = ComputeNode(node_id="n2")
        reg.register(n1)
        reg.register(n2)

        assert len(reg.get_online()) == 1

    def test_merge_and_clear(self):
        from compute_mesh.topology.node import ComputeNode
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        nodes = [ComputeNode(node_id=f"n{i}") for i in range(3)]
        assert reg.merge(nodes) == 3
        assert reg.count() == 3

        reg.clear()
        assert reg.count() == 0

    def test_to_dict(self):
        from compute_mesh.topology.node import ComputeNode
        from compute_mesh.topology.registry import NodeRegistry

        reg = NodeRegistry()
        reg.register(ComputeNode(node_id="n1"))
        d = reg.to_dict()
        assert d["count"] == 1
        assert "n1" in d["nodes"]
