"""AetherForge Swarm — compatibility shim exposing swarm_engine via aetherforge.swarm."""

from swarm_engine.cli import main as cli
from swarm_engine import __version__

__all__ = ["cli", "__version__"]
