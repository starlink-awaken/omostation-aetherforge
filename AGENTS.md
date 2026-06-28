# AGENTS.md — AetherForge

    > Scope: project-local developer guide for `aetherforge`.
    > Workspace rules live in [`../../AGENTS.md`](../../AGENTS.md); project metadata lives in [`../../docs/project-registry.yaml`](../../docs/project-registry.yaml).

    ## Role

    - Layer: X
    - Stack: Python / uv / pytest
    - Responsibility: 能力与算力框架，承载 gateway/mesh/swarm 能力

    Do not copy volatile facts such as test counts, tool counts, service counts, ports, or current health into this file.

    ## Before Editing

    1. Read this file and [`CLAUDE.md`](CLAUDE.md) when it exists.
    2. Check `git status --short` inside this project and at the workspace root.
    3. Read the specific source or tests you are about to change.
    4. Prefer project-local commands and targeted tests.

    ## Commands

    ```bash
    uv sync
uv run pytest
uv run ruff check "packages/" "src/"
    ```

    ## Key Files

    - `packages/gateway/`
- `packages/mesh/`
- `packages/swarm/`
- `src/aetherforge/`

    ## Gotchas

    - `归档能力合并关系以 docs/project-registry.yaml 的 archived 段为准。`
- `跨层能力暴露应走 BOS/Agora，不直接绕入口。`

    ## Verification

    - Documentation-only changes: run `uv run --with "pyyaml" python "../../bin/doc-ssot-lint.py" --json` from this project or from the workspace root.
    - Code changes: run the narrowest relevant project test first, then broaden if shared contracts changed.
    - Cross-layer behavior: verify the caller and the callee, not just the touched module.

    ## SSOT Pointers

    - Workspace architecture: [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)
    - Layer index: [`../../LAYER-INDEX.md`](../../LAYER-INDEX.md)
    - Project metadata: [`../../docs/project-registry.yaml`](../../docs/project-registry.yaml)
    - Runtime state: [`../../.omo/state/system.yaml`](../../.omo/state/system.yaml)
