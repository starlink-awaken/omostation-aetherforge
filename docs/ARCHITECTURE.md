# AetherForge Architecture

> Architecture overview for **AetherForge**. For the full workspace architecture, see [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).

## Responsibilities

AetherForge is part of the eCOS v6 workspace. See [`../README.md`](../README.md) for a one-line description and [`../CAPABILITY-MAP.md`](../CAPABILITY-MAP.md) for capability mapping.

## Key Surfaces

- `packages/gateway/` — LLM Gateway (ModelGateway, ModelScheduler, CredentialsManager, cc-switch sync, SSOT loader)
- `packages/mesh/` — Compute Mesh (topology discovery, node scheduler, worker pool)
- `packages/swarm/` — Swarm Engine (GraphWorkflow, Hatcher, IntelligentAgent, risk/trust/pi, digital brain workers)
- `src/aetherforge/` — Core (CLI, MCP server, triage router, route scheduler)

## Architecture (2026-08-09 Refactoring)

### Model Routing (3-tier)

| Tier | Source | Models | Routing |
|------|--------|--------|---------|
| Local | omlx (MLX on Apple Silicon) | 23 | Direct port (8081-8194) via `_generate_via_omlx_router` |
| Cloud Paid | DeepSeek, OpenRouter, Anthropic | 74 | Registry → SSOTProviderAdapter → direct API |
| Cloud Free/Forward | GLM, Kimi, MiniMax, Nvidia, etc. | 47 | cc-switch → CredentialsManager → SSOTProviderAdapter |

### Key Components

- **ModelGateway** — Unified LLM entry point with K1 sensitive flow detection, direct port routing for local omlx, and registry-based routing for cloud providers
- **IntelligentAgent** — 5-layer decision chain: Risk Gate → MOS Memory → LLM → Decision Record → Trust Feedback
- **GraphWorkflow** — DAG-based workflow engine with checkpoint/resume, admission grants, retry policy
- **Hatcher** — Agent factory spawning CLI subprocesses, internal threads, or external agent CLIs
- **RiskEngine** — 5-level dynamic safety gate (L0 auto → L4 forbidden) with trust accumulation
- **cc-switch Adapter** — Syncs 15 cloud providers + 57 model prices from cc-switch iCloud DB

## Design Notes

- Runtime facts (counts, ports, health) are intentionally not maintained here. Use the workspace registries and project source as the truth.
- For boundaries and call chains, read [`../BOUNDARY.md`](../BOUNDARY.md) and [`../CALLCHAIN.md`](../CALLCHAIN.md).
- For developer rules, read [`../AGENTS.md`](../AGENTS.md).

## Component Overview

```mermaid
graph TD
    User([User / Agent])
    N0[Gateway]
    N1[Mesh]
    N2[Swarm]
    Core[Core]
    N0 --> N1
    N1 --> N2
    N2 --> Core
    User --> Core
```

- Arrows show typical interaction flow, not strict call direction.
- See [`../CALLCHAIN.md`](../CALLCHAIN.md) for detailed call chains.
