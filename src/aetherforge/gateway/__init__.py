"""AetherForge Gateway — compatibility shim exposing llm_gateway via aetherforge.gateway."""

from llm_gateway import __version__
from llm_gateway.cli import main as cli
from llm_gateway.detection import create_provider
from llm_gateway.quota_engine import QuotaEngine
from llm_gateway.mcp_server import llm_generate

__all__ = ["cli", "__version__", "create_provider", "QuotaEngine", "llm_generate"]


