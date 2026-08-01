from __future__ import annotations

from swarm_engine.graph_workflow import GraphWorkflow


def test_graph_workflow_emits_mesh_lifecycle() -> None:
    workflow = GraphWorkflow()

    @workflow.node("plan")
    def plan(state: dict) -> dict:
        return {"plan": "ok"}

    workflow.set_entry("plan")
    events: list[dict] = []

    state = workflow.run({}, workflow_run_id="swarm-run-1", event_sink=events.append)

    assert state["_errors"] == []
    assert [event["event_type"] for event in events] == [
        "WorkflowRequested",
        "WorkflowAdmitted",
        "StepDispatched",
        "StepStarted",
        "StepHeartbeat",
        "CheckpointSaved",
        "WorkflowSucceeded",
    ]
    assert len({event["idempotency_key"] for event in events}) == len(events)


def test_graph_workflow_emits_failure_event() -> None:
    workflow = GraphWorkflow()

    @workflow.node("broken")
    def broken(_state: dict) -> dict:
        raise RuntimeError("boom")

    workflow.set_entry("broken")
    events: list[dict] = []

    state = workflow.run({}, workflow_run_id="swarm-run-2", event_sink=events.append)

    assert state["_errors"]
    assert [event["event_type"] for event in events][-2:] == [
        "StepFailed",
        "WorkflowFailed",
    ]
