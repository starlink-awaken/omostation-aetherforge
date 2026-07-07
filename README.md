# AetherForge

    > X · 能力与算力框架，承载 gateway/mesh/swarm 能力
    > Metadata SSOT: [`../../docs/project-registry.yaml`](../../docs/project-registry.yaml)

    ## What It Owns

    能力与算力框架，承载 gateway/mesh/swarm 能力.

    ## Quick Start

    ```bash
    uv sync
uv run pytest
uv run ruff check "packages/" "src/"
    ```

    ## Key Surfaces

    - `packages/gateway/`
- `packages/mesh/`
- `packages/swarm/`
- `src/aetherforge/`

    ## Documentation

    - Developer guide: [`AGENTS.md`](AGENTS.md)
    - AI context loader: [`CLAUDE.md`](CLAUDE.md) when present
    - Workspace architecture: [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)
    - Layer placement: [`../../LAYER-INDEX.md`](../../LAYER-INDEX.md)

    ## SSOT Rules

    Runtime facts, counts, ports, health, and generated inventories are intentionally not maintained here. Use the workspace registries and project source as the truth.
## Project Governance

- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
- [Code of Conduct](CODE_OF_CONDUCT.md)
