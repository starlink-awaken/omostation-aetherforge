"""Tests for compute_mesh.worker — MeshWorker and WorkerRegistry."""

from __future__ import annotations

import time


class TestMeshWorker:
    def test_default_worker(self):
        from compute_mesh.worker.worker import MeshWorker

        w = MeshWorker(worker_id="w1", node_id="n1")
        assert w.worker_id == "w1"
        assert w.node_id == "n1"
        assert w.is_idle is True
        assert w.is_busy is False
        assert w.tasks_completed == 0
        assert w.tasks_failed == 0
        assert w.created_at > 0
        assert w.last_heartbeat > 0

    def test_success_rate(self):
        from compute_mesh.worker.worker import MeshWorker

        w = MeshWorker(worker_id="w1", node_id="n1")
        assert w.success_rate == 1.0  # no tasks yet, default to 1.0

        w.tasks_completed = 7
        w.tasks_failed = 3
        assert w.success_rate == 0.7

    def test_to_dict(self):
        from compute_mesh.worker.worker import MeshWorker

        w = MeshWorker(worker_id="w1", node_id="n1")
        d = w.to_dict()
        assert d["worker_id"] == "w1"
        assert d["node_id"] == "n1"
        assert d["status"] == "idle"
        assert d["success_rate"] == 1.0
        assert d["uptime_seconds"] >= 0


class TestWorkerRegistry:
    def test_register_and_get(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker

        reg = WorkerRegistry()
        w = MeshWorker(worker_id="w1", node_id="n1")
        assert reg.register(w) is True
        assert reg.count() == 1
        assert reg.get("w1") is w

    def test_unregister(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker

        reg = WorkerRegistry()
        reg.register(MeshWorker(worker_id="w1", node_id="n1"))
        assert reg.unregister("w1") is True
        assert reg.count() == 0

    def test_heartbeat(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker, WorkerStatus

        reg = WorkerRegistry()
        w = MeshWorker(worker_id="w1", node_id="n1")
        old_heartbeat = w.last_heartbeat
        reg.register(w)

        time.sleep(0.01)
        assert reg.heartbeat("w1") is True
        assert w.last_heartbeat > old_heartbeat

        # Heartbeat should recover from ERROR state
        w.status = WorkerStatus.ERROR
        reg.heartbeat("w1")
        assert w.status == WorkerStatus.IDLE

    def test_heartbeat_not_found(self):
        from compute_mesh.worker.registry import WorkerRegistry

        reg = WorkerRegistry()
        assert reg.heartbeat("nonexistent") is False

    def test_check_stale(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker, WorkerStatus

        reg = WorkerRegistry(heartbeat_timeout=0.001)
        w = MeshWorker(worker_id="w1", node_id="n1")
        w.last_heartbeat = 0  # very old heartbeat
        reg.register(w)

        stale = reg.check_stale()
        assert "w1" in stale
        assert w.status == WorkerStatus.ERROR

    def test_set_busy_idle_error(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker, WorkerStatus

        reg = WorkerRegistry()
        w = MeshWorker(worker_id="w1", node_id="n1")
        reg.register(w)

        assert reg.set_busy("w1", task_id="task-1") is True
        assert w.status == WorkerStatus.BUSY
        assert w.current_task == "task-1"

        assert reg.set_idle("w1") is True
        assert w.status == WorkerStatus.IDLE
        assert w.current_task == ""

        assert reg.set_error("w1", reason="oom") is True
        assert w.status == WorkerStatus.ERROR
        assert w.metadata["last_error"] == "oom"

    def test_filter_and_get_idle(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker, WorkerStatus

        reg = WorkerRegistry()
        w1 = MeshWorker(worker_id="w1", node_id="n1")
        w2 = MeshWorker(worker_id="w2", node_id="n1")
        w2.status = WorkerStatus.BUSY
        reg.register(w1)
        reg.register(w2)

        idle = reg.get_idle()
        assert len(idle) == 1
        assert idle[0].worker_id == "w1"

    def test_get_by_node(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker

        reg = WorkerRegistry()
        reg.register(MeshWorker(worker_id="w1", node_id="n1"))
        reg.register(MeshWorker(worker_id="w2", node_id="n1"))
        reg.register(MeshWorker(worker_id="w3", node_id="n2"))

        assert len(reg.get_by_node("n1")) == 2
        assert len(reg.get_by_node("n2")) == 1

    def test_get_stats(self):
        from compute_mesh.worker.registry import WorkerRegistry
        from compute_mesh.worker.worker import MeshWorker, WorkerStatus

        reg = WorkerRegistry()
        w1 = MeshWorker(worker_id="w1", node_id="n1", tasks_completed=5, tasks_failed=1)
        w2 = MeshWorker(worker_id="w2", node_id="n1")
        w2.status = WorkerStatus.BUSY
        reg.register(w1)
        reg.register(w2)

        stats = reg.get_stats()
        assert stats["total"] == 2
        assert stats["idle"] == 1
        assert stats["busy"] == 1
        assert stats["total_completed"] == 5
        assert stats["total_failed"] == 1
