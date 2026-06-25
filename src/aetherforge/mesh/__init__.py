"""AetherForge Mesh — compatibility shim exposing compute_mesh via aetherforge.mesh."""

from compute_mesh import __version__
from compute_mesh.api.cli import main as cli
from compute_mesh.api.mcp_server import (
    mesh_cost_report,
    mesh_generate,
    mesh_health_check,
    mesh_list_nodes,
    mesh_status,
)
from compute_mesh.pool import CostTracker
from compute_mesh.topology import NodeRegistry
from compute_mesh.topology.network_scanner import NetworkScanner
from compute_mesh.topology.scanner import detect_cloud_nodes, load_static_nodes, probe_local_daemons

__all__ = [
    "cli",
    "__version__",
    "NetworkScanner",
    "detect_cloud_nodes",
    "load_static_nodes",
    "probe_local_daemons",
    "CostTracker",
    "NodeRegistry",
    "mesh_cost_report",
    "mesh_generate",
    "mesh_health_check",
    "mesh_list_nodes",
    "mesh_status",
]
