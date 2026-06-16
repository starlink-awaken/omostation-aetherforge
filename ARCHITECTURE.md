# aetherforge — Architecture

> **Layer**: X 横切框架  
> **Role**: 算力网格 + LLM 网关 + 群体智能引擎  
> **Stack**: Python 3.10+, uv workspace, hatchling, fastmcp  
> **Health**: Active — gateway ~90%, mesh ~15%, swarm ~80%
>
> 系统全景参见：[`docs/ARCHITECTURE-DIAGRAM.md`](../docs/ARCHITECTURE-DIAGRAM.md)

---

## 1. 内部架构

```mermaid

graph TB
    CLI[aetherforge CLI]
    MCP[aetherforge MCP]
    GW[llm_gateway]
    Mesh[compute_mesh]
    Swarm[swarm_engine]
    Providers[LLM Providers]
    Nodes[Compute Nodes]

    CLI --> GW
    CLI --> Mesh
    CLI --> Swarm
    MCP --> GW
    GW --> Providers
    Mesh --> Nodes
    Swarm --> Mesh

```

## 2. 入口

| Type | Entry | Port / Notes |
|:--|:--|:--|
| CLI | `aetherforge` | gateway/mesh/swarm subcommands |
| MCP stdio | `aetherforge-mcp` | forge_generate, forge_mesh_status, ... |

## 3. 核心模块

| Module | Responsibility |
|:--|:--|
| `src/aetherforge/cli.py` | Unified CLI dispatcher |
| `src/aetherforge/mcp_server.py` | Aggregated MCP server |
| `packages/gateway/src/llm_gateway/` | LLM provider routing / fallback |
| `packages/mesh/src/compute_mesh/` | Compute node discovery / scheduler |
| `packages/swarm/src/swarm_engine/` | Swarm orchestration |

## 4. 测试

```bash
cd projects/aetherforge && make test
```
