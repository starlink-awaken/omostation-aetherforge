from __future__ import annotations

from datetime import UTC, datetime, timedelta

from swarm_engine.graph_workflow import GraphWorkflow
from swarm_engine.workflow_admission import admission_proof


def _grant(run_id: str, nodes: list[str]) -> dict:
    grant = {
        "admission_id": f"adm-{run_id}",
        "status": "admitted",
        "workflow_run_id": run_id,
        "trace_id": run_id,
        "backend": "aetherforge",
        "step_run_ids": [f"{run_id}:{node}" for node in nodes],
        "capabilities": ["execute"],
        "policy_digest": "policy-test",
        "issued_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    grant["proof"] = admission_proof(grant)
    return grant


def test_graph_workflow_emits_mesh_lifecycle() -> None:
    workflow = GraphWorkflow()

    @workflow.node("plan")
    def plan(state: dict) -> dict:
        return {"plan": "ok"}

    workflow.set_entry("plan")
    events: list[dict] = []

    state = workflow.run(
        {},
        workflow_run_id="swarm-run-1",
        event_sink=events.append,
        admission=_grant("swarm-run-1", ["plan"]),
    )

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

    state = workflow.run(
        {},
        workflow_run_id="swarm-run-2",
        event_sink=events.append,
        admission=_grant("swarm-run-2", ["broken"]),
    )

    assert state["_errors"]
    assert [event["event_type"] for event in events][-2:] == [
        "StepFailed",
        "WorkflowFailed",
    ]


def test_mesh_tracked_graph_rejects_missing_admission() -> None:
    workflow = GraphWorkflow()
    workflow.add_node("plan", lambda _state: {"plan": "ok"})
    workflow.set_entry("plan")
    state = workflow.run({}, workflow_run_id="swarm-no-admission")
    assert state["_errors"][0]["error_code"] == "WORKFLOW_ADMISSION_REQUIRED"
