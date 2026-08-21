"""Supplemental tests for recently-added modules.

Covers: GroupChat, GraphWorkflow, StepCallbacks, ObjectStore, new providers.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "gateway", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "mesh", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "swarm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

passed = 0
failed = 0


def test(name):
    def dec(fn):
        def wrapper():
            global passed, failed
            try:
                fn()
                passed += 1
                print(f"  ✅ {name}")
            except Exception as e:
                failed += 1
                import traceback

                print(f"  ❌ {name}: {e}\n{traceback.format_exc()}")

        return wrapper

    return dec


test.__test__ = False  # type: ignore[reportFunctionMemberAccess]


# ══════════════════════════════════════════════════════════════════════════
# StepCallbacks (vs CrewAI)
# ══════════════════════════════════════════════════════════════════════════


@test("StepCallbacks: 6 hooks + decorator")
def test_callbacks():
    from compute_mesh.worker.callbacks import StepCallbacks

    cb = StepCallbacks()
    events = []

    @cb.on_task_start
    def s(wid, task):
        events.append(f"start:{wid}")

    @cb.on_task_complete
    def c(wid, res):
        events.append(f"complete:{wid}")

    @cb.on_task_fail
    def f(wid, err):
        events.append(f"fail:{wid}")

    @cb.on_worker_claim
    def cl(wid):
        events.append(f"claim:{wid}")

    @cb.on_worker_release
    def rl(wid):
        events.append(f"release:{wid}")

    @cb.on_retry
    def rt(wid, n, err):
        events.append(f"retry:{wid}:{n}")

    cb.fire_task_start("w1", "task")
    cb.fire_task_complete("w1", {"ok": True})
    cb.fire_task_fail("w2", "err")
    cb.fire_worker_claim("w3")
    cb.fire_worker_release("w3")
    cb.fire_retry("w4", 2, "timeout")

    assert len(events) == 6
    assert events == ["start:w1", "complete:w1", "fail:w2", "claim:w3", "release:w3", "retry:w4:2"]
    assert len(cb.on_task_start) == 1
    assert len(cb.on_task_complete) == 1


@test("StepCallbacks: add/remove/clear")
def test_callbacks_management():
    from compute_mesh.worker.callbacks import StepCallbacks

    cb = StepCallbacks()

    def h1(wid, task):
        pass

    def h2(wid, task):
        pass

    cb.on_task_start.add(h1)
    cb.on_task_start.add(h2)
    assert len(cb.on_task_start) == 2

    cb.on_task_start.remove(h1)
    assert len(cb.on_task_start) == 1

    cb.on_task_start.clear()
    assert len(cb.on_task_start) == 0


# NOTE: the previous GroupChat / GraphWorkflow (swarm_engine) sections were
# removed (Y1Q4-T6-01): swarm_engine was deleted and replaced with a
# fail-closed shim (src/aetherforge/swarm), so those APIs no longer exist
# and their tests cannot run.


# ══════════════════════════════════════════════════════════════════════════
# ObjectStore (vs Ray)
# ══════════════════════════════════════════════════════════════════════════


@test("ObjectStore: put/get/delete")
def test_objectstore_basic():
    from compute_mesh.worker.object_store import ObjectStore

    store = ObjectStore(db_path=None)
    oid = store.put({"msg": "hello", "num": 42})
    data = store.get(oid)
    assert data["msg"] == "hello"  # type: ignore[reportOptionalSubscript]
    assert data["num"] == 42  # type: ignore[reportOptionalSubscript]
    assert store.exists(oid) is True
    store.delete(oid)
    assert store.exists(oid) is False
    assert store.get(oid) is None


@test("ObjectStore: TTL expiry")
def test_objectstore_ttl():
    import time

    from compute_mesh.worker.object_store import ObjectStore

    store = ObjectStore(db_path=None)
    oid = store.put({"temp": True}, ttl=0.05)
    assert store.get(oid) is not None
    time.sleep(0.1)
    assert store.get(oid) is None


@test("ObjectStore: SQLite persistence")
def test_objectstore_persist():
    import os
    import tempfile

    from compute_mesh.worker.object_store import ObjectStore

    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_objs.db")
    store = ObjectStore(db_path=db_path)
    oid = store.put({"persistent": True})
    assert store.get(oid)["persistent"] is True  # type: ignore[reportOptionalSubscript]

    # New instance should load from DB
    store2 = ObjectStore(db_path=db_path)
    assert store2.get(oid)["persistent"] is True  # type: ignore[reportOptionalSubscript]

    store2.delete(oid)
    assert store2.get(oid) is None


@test("ObjectStore: put_many/get_many")
def test_objectstore_bulk():
    from compute_mesh.worker.object_store import ObjectStore

    store = ObjectStore(db_path=None)
    refs = store.put_many({"a": 1, "b": 2, "c": 3})
    assert len(refs) == 3
    assert "a" in refs
    results = store.get_many(list(refs.values()))
    assert len(results) == 3


@test("ObjectStore: stats")
def test_objectstore_stats():
    from compute_mesh.worker.object_store import ObjectStore

    store = ObjectStore(db_path=None)
    store.put({"x": "y"})
    stats = store.get_stats()
    assert stats["total_objects"] == 1
    assert stats["total_size_bytes"] > 0


# ══════════════════════════════════════════════════════════════════════════
# New Providers
# ══════════════════════════════════════════════════════════════════════════


@test("Providers: 9 registered in detection")
def test_providers_9():
    from llm_gateway.detection import _PROVIDER_REGISTRY

    assert len(_PROVIDER_REGISTRY) == 9
    assert "azure" in _PROVIDER_REGISTRY
    assert "bedrock" in _PROVIDER_REGISTRY
    assert "vertex" in _PROVIDER_REGISTRY


@test("Providers: all classes importable")
def test_providers_import():
    from llm_gateway.providers import (
        AzureOpenAIProvider,
        BedrockProvider,
        VertexAIProvider,
    )

    assert AzureOpenAIProvider
    assert BedrockProvider
    assert VertexAIProvider


@test("Providers: detection priority includes new")
def test_providers_priority():
    from llm_gateway.detection import detect_backends

    # Should not crash, returns available (hitl/ollama)
    available = detect_backends()
    assert isinstance(available, list)


@test("Providers: L0 M1 includes new engines")
def test_providers_l0():
    from aetherforge._paths import M1_COMPUTE_ENGINE_DIR

    m1_dir = M1_COMPUTE_ENGINE_DIR
    if m1_dir.exists():
        files = list(m1_dir.glob("*.yaml"))
        names = [f.stem for f in files]
        assert "ENG-AZURE-OPENAI" in names, f"Missing from {names}"
        assert "ENG-BEDROCK" in names
        assert "ENG-VERTEX-AI" in names
        print(f"    L0 M1: {len(files)} engine nodes ({len(names)})")
    else:
        print("    L0 M1 dir not found (skip)")


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════


def run_all():
    global passed, failed
    print("=" * 60)
    print("  AetherForge Supplemental Tests")
    print("=" * 60)

    import inspect

    test_fns = []
    for name, fn in inspect.getmembers(sys.modules[__name__]):
        if name.startswith("test_") and callable(fn):
            test_fns.append(fn)

    for fn in sorted(test_fns, key=lambda f: f.__name__):
        fn()

    total = passed + failed
    print()
    print(f"  Supplemental: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
