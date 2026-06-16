"""AetherForge Mesh — compatibility shim exposing compute_mesh via aetherforge.mesh."""

from compute_mesh.api.cli import main as cli
from compute_mesh import __version__

__all__ = ["cli", "__version__"]
