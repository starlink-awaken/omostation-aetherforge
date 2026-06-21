"""AetherForge Gateway — compatibility shim exposing llm_gateway via aetherforge.gateway."""

from llm_gateway import __version__
from llm_gateway.cli import main as cli

__all__ = ["cli", "__version__"]
