# AetherForge

🌐 [简体中文](README.zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributing](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/security-policy-blue.svg)](SECURITY.md)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://docs.astral.sh/uv/)

    > X · 能力与算力框架，承载 gateway/mesh/swarm 能力
    > Metadata SSOT: [`../../docs/project-registry.yaml`](../../docs/project-registry.yaml)

    ## What It Owns

    能力与算力框架，承载 gateway/mesh/swarm 能力.

    ## Installation

```bash
# Clone the workspace recursively
git clone --recursive https://github.com/starlink-awaken/omostation.git
cd omostation/projects/aetherforge

# Install dependencies with uv
uv sync
```

Requires Python 3.13+ (see `pyproject.toml`).

## Quick Start

    ```bash
    uv sync
uv run pytest
uv run ruff check "packages/" "src/"
    ```

## Local Compute Hub

AetherForge exposes one OpenAI-compatible facade over the local compute fleet. The
default request policy stays local and uses the governed order:

1. oMLX App on the MacBook Pro.
2. LM Link / LM Studio engines registered in ComputeEngine SSOT.
3. Ollama engines registered in ComputeEngine SSOT.

Cloud providers are eligible only when the caller explicitly selects
`routing_mode=hybrid` or `routing_mode=cloud`. Provider request defaults disable
reasoning/thinking unless the caller supplies a supported override.

The BOS/Agora adapter accepts either a normal JSON request or the Agora
`{"kwargs": ...}` envelope on stdin:

```bash
printf '%s\n' '{"model":"mythos-fast","messages":[{"role":"user","content":"Reply OK"}]}' \
  | uv run python -m aetherforge.cli infer
```

The executable resolves the running authenticated AetherForge facade; it does not
start a second gateway or bypass Agora/ComputeEngine SSOT.

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

- [Maintainers](MAINTAINERS.md)
- [Acknowledgments](ACKNOWLEDGMENTS.md)

- [Development](docs/DEVELOPMENT.md)
- [Release Process](RELEASE.md)

- [Governance](GOVERNANCE.md)
- [Support](SUPPORT.md)

- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributors](CONTRIBUTORS.md)
## Getting Help

- [FAQ](docs/FAQ.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [API / Usage Reference](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
