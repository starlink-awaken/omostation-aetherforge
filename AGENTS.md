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

- **aetherforge-swarm-ext** — 依赖扩展模块 (ils/perception/planning/legacy)
- **httpx** — LLM 网关 HTTP 客户端
- **fastmcp** — MCP Server

## Testing

```bash
uv run pytest tests/ -q               # 全量
uv run pytest tests/ -k "keyword" -q  # 按关键字
```
