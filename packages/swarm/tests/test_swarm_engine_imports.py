"""Import smoke tests for newly merged swarm-engine modules.

These tests only cover copied modules that are self-contained enough to load
without unresolved legacy imports. Modules with known stale references to
`organs.*`, `nucleus.*`, etc. are intentionally omitted and documented in
MERGE-NOTES.md.
"""

from __future__ import annotations

import importlib

import pytest

MERGED_MODULES = [
    # Top-level modules
    "swarm_engine.association_engine",
    "swarm_engine.burndown_engine",
    "swarm_engine.hypothesis_pipeline",
    "swarm_engine.ils_defaults",
    "swarm_engine.ils_plugins",
    "swarm_engine.ils_types",
    "swarm_engine.mapping_engine",
    "swarm_engine.mapping_worker_abstraction",
    "swarm_engine.mapping_worker_registry",
    "swarm_engine.mutation_validator",
    "swarm_engine.perception_manager",
    "swarm_engine.perception_validation",
    "swarm_engine.possession_multi_session",
    "swarm_engine.primordial_toolkit",
    "swarm_engine.redis_message_broker",
    "swarm_engine.refinement_daemon",
    "swarm_engine.result_aggregator",
    "swarm_engine.result_summarizer",
    "swarm_engine.rl_optimizer",
    "swarm_engine.semantic_index",
    "swarm_engine.structural_merger",
    "swarm_engine.synapse_anthropic",
    "swarm_engine.synapse_github",
    "swarm_engine.synapse_hub",
    "swarm_engine.synapse_ollama",
    "swarm_engine.worker_dispatcher",
    "swarm_engine.worker_node",
    # Dispatch compatibility shim
    "swarm_engine.dispatch.compat",
    # Engine subpackage modules
    "swarm_engine.engine.a2a_protocol",
    "swarm_engine.engine.archetype_distiller",
    "swarm_engine.engine.compute_harvester",
    "swarm_engine.engine.compute_pool",
    "swarm_engine.engine.compute_pool_shard",
    "swarm_engine.engine.dispatch.compat",
    "swarm_engine.engine.ego_dispatch_loop",
    "swarm_engine.engine.hifi_query",
    "swarm_engine.engine.knowledge_enhancement_mixin",
    "swarm_engine.engine.lifecycle.cluster",
    "swarm_engine.engine.local_worker",
    "swarm_engine.engine.ooda_loop",
    "swarm_engine.engine.possession_multi_session",
    "swarm_engine.engine.swarm_emergence",
    "swarm_engine.engine.swarm_optimizer",
    "swarm_engine.execution_scheduler",
    # Newly reconciled legacy modules
    "swarm_engine.hatcher_core",
    "swarm_engine.ils_engine",
    "swarm_engine.nks_task_planner",
    "swarm_engine.universal_worker",
    "swarm_engine.vision_metabolizer",
]


@pytest.mark.parametrize("module_name", MERGED_MODULES)
def test_merged_module_imports(module_name: str) -> None:
    """Each merged module must import without raising."""
    module = importlib.import_module(module_name)
    assert module is not None
