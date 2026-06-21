"""AetherForge Mesh — compatibility shim exposing compute_mesh via aetherforge.mesh."""

from compute_mesh import __version__
from compute_mesh.api.cli import main as cli

__all__ = ["cli", "__version__"]
