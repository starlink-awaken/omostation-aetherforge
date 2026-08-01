from __future__ import annotations

from swarm_engine.graph_workflow import GraphWorkflow
from swarm_engine.workflow_checkpoint import WorkflowCheckpointStore


def test_graph_workflow_resumes_from_durable_checkpoint(tmp_path) -> None:
    store = WorkflowCheckpointStore(tmp_path / "swarm-checkpoints.jsonl")
    calls: list[str] = []

    first = GraphWorkflow()

    @first.node("prepare")
    def prepare(_state: dict) -> dict:
        calls.append("prepare")
        return {"prepared": True}

    @first.node("execute")
    def execute(_state: dict) -> dict:
        calls.append("execute")
        raise RuntimeError("temporary failure")

    first.add_edge("prepare", "execute")
    first.set_entry("prepare")
    failed = first.run({}, workflow_run_id="swarm-resume", checkpoint_store=store)
    assert failed["_errors"]

    second = GraphWorkflow()

    @second.node("prepare")
    def prepare_again(_state: dict) -> dict:
        calls.append("prepare-again")
        return {"prepared": True}

    @second.node("execute")
    def execute_again(_state: dict) -> dict:
        calls.append("execute-again")
        return {"output": "recovered"}

    second.add_edge("prepare", "execute")
    second.set_entry("prepare")
    recovered = second.run(
        {}, workflow_run_id="swarm-resume", checkpoint_store=store
    )

    assert recovered["output"] == "recovered"
    assert calls == ["prepare", "execute", "execute-again"]
