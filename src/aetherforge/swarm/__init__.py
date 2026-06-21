"""AetherForge Swarm — compatibility shim exposing swarm_engine via aetherforge.swarm."""

from swarm_engine import __version__
from swarm_engine.cli import main as cli

__all__ = ["cli", "__version__"]
