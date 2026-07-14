# CLAUDE.md — AetherForge AI Context

> Session loader for AI work inside `aetherforge`.
> Keep durable engineering rules in [`AGENTS.md`](AGENTS.md) and volatile facts in SSOT files.

## Load First

1. [`AGENTS.md`](AGENTS.md)
2. [`README.md`](README.md)
3. The source files and tests directly related to the task
4. Workspace context in [`../../CLAUDE.md`](../../CLAUDE.md) when the task crosses project boundaries

## Project Role

- Layer: X
- Responsibility: 能力与算力框架，承载 gateway/mesh/swarm 能力
- Stack: Python / uv / pytest

## Commands

```bash
uv sync
uv run pytest
uv run ruff check "packages/" "src/"
```

## Safe Editing Rules

- 归档能力合并关系以 docs/project-registry.yaml 的 archived 段为准。
- 跨层能力暴露应走 BOS/Agora，不直接绕入口。
- Do not commit, push, reset, or bump submodule pointers unless the user explicitly asks.
- Preserve unrelated dirty changes in this repository.
- Keep Markdown pointed at SSOT files instead of copying generated facts.

## Closeout

```bash
git status --short
uv run --with "pyyaml" python "../../bin/ssot/doc-ssot-lint.py" --json
```

Report the checks you actually ran and any pre-existing dirty state that remains.
