---
type: derived
source: projects/aetherforge
owner: governance-team
last_updated: 2026-09-03
---

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

AetherForge exposes one authenticated OpenAI-compatible facade over the local
compute fleet. It owns logical aliases, K1 constraints, and cloud policy. In
`active` mode it sends the resolved local model ID through the private `omlxcd`
Unix socket; omlxc exclusively owns physical placement, capacity, residency, and
local backend fallback.

Cloud providers are eligible only when the caller explicitly selects
`routing_mode=hybrid` or `routing_mode=cloud`. Provider request defaults disable
reasoning/thinking unless the caller supplies a supported override.

Migration is controlled by `AETHERFORGE_OMLXC_MODE=legacy|shadow|active` and
defaults to `legacy`. `shadow` performs one bounded route-plan comparison while
the single real inference remains on the legacy path. `OMLXC_SOCKET` overrides
the platform-default socket. The same application and gateway singleton serve
the canonical `9290` facade and optional transition listener `4000`.

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

---

## 项目状态 (动态数据)

| 指标 | 权威读源 | 说明 |
|------|----------|------|
| 版本 | `pyproject.toml` → `[project.version]` | 以 pyproject.toml 为准 |
| 测试数 | `pytest --collect-only -q` | 动态计数 |
| 代码行数 | `find src -name "*.py" \| wc -l` | 以实际文件为准 |
| 源文件数 | `find src -name "*.py" \| wc -l` | 以实际文件为准 |
| 测试文件数 | `find tests -name "*.py" \| wc -l` | 以实际文件为准 |

> **doc-ssot 契约**: 上表中的所有数字均为易变事实, 不在本文件硬编码. 运行权威读源命令获取实时值.
> 模板: `.omo/standards/readme-template.md` | 检测: `bin/gac/check-readme-hardcoded.py`

