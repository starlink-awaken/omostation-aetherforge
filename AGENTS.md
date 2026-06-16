# AGENTS.md — AetherForge

> eCOS v5 蜂群基础设施核心引擎 · LLM 网关路由 + Agent 编排

## Quick Commands

```bash
cd projects/aetherforge
uv run pytest tests/ -q
uv run ruff check src/
uv run ruff format src/ --check
```

## Architecture

AetherForge 是 L0 横切面蜂群引擎：

```
L3 cockpit         ── 统一入口
L2 kairon          ── 知识引擎
I0 agora           ── MCP Hub
L0 aetherforge     ── 蜂群引擎（本仓）
L0 ecos            ── 协议层
```

### 核心模块

| 模块 | 职责 |
|:-----|:------|
| `gateway/` | LLM 路由、负载均衡、重试 |
| `swarm/` | Agent 蜂群调度、任务分发 |
| `mcp/` | FastMCP 工具注册 (stdio) |

## Key Dependencies

- **aetherforge-gateway** — LLM 网关（原 llm-gateway，已并入本仓）
- **aetherforge-mesh** — 算力网格（原 compute-mesh，已并入本仓）
- **aetherforge-swarm** — 蜂群引擎（原 swarm-engine + aetherforge-swarm-ext，已并入本仓）
- **httpx** — LLM 网关 HTTP 客户端
- **fastmcp** — MCP Server

## Testing

```bash
make test                               # 全量 (packages/*)
uv run pytest -k "keyword" -q           # 按关键字
```
