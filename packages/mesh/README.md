# AetherForge Mesh [LIBRARY-ONLY]

> **Note:** This package is **[LIBRARY-ONLY]**. 
> It does not expose any standalone HTTP servers, daemon processes, or CLI entrypoints in the final 5+4+1+1 architecture. All compute mesh topology discovery, resource pooling, and worker management capabilities are accessed via the Agora BOS URI (`bos://capability/mesh/*`) or directly integrated into the L1 Runtime scheduler.

## Overview

AetherForge Mesh handles compute grid topology, tracking nodes, and assigning workloads across distributed hardware backends.
