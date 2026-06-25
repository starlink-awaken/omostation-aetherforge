"""AetherForge Swarm — compatibility shim exposing swarm_engine via aetherforge.swarm."""

from swarm_engine import __version__
from swarm_engine.graph_workflow import GraphWorkflow

__all__ = ["__version__", "GraphWorkflow"]
