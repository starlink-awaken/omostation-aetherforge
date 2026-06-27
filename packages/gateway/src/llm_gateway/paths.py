"""AetherForge Gateway — path configuration.

Re-export from canonical aetherforge._paths SSOT so any env-var override
(AETHERFORGE_M1_DIR / AETHERFORGE_M1_COMPUTE_DIR) is honored uniformly
across the gateway package without per-file duplication.

Backward-compat note: legacy callers may still set LLM_GATEWAY_M1_DIR,
which aetherforge._paths honors as a fallback alias for AETHERFORGE_M1_COMPUTE_DIR.
"""

from aetherforge._paths import M1_COMPUTE_ENGINE_DIR, M1_MODEL_DIR

__all__ = ["M1_COMPUTE_ENGINE_DIR", "M1_MODEL_DIR"]
