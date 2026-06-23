"""AetherForge Swarm — compatibility shim exposing swarm_engine via aetherforge.swarm."""

from swarm_engine import __version__
from swarm_engine.cli import main as cli
from swarm_engine.graph_workflow import GraphWorkflow

__all__ = ["cli", "__version__", "GraphWorkflow"]

