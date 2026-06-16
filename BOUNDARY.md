# aetherforge — System Boundary

> 本文档描述 aetherforge 与 eCOS 系统其他部分的边界：暴露的接口、依赖的上游、影响的下游。
>
> 架构演进对比参见：[`docs/ARCHITECTURE-EVOLUTION.md`](../docs/ARCHITECTURE-EVOLUTION.md)

---

## 1. 暴露接口

### BOS URI

- `bos://capability/forge/*`

### 入口

- **CLI**: `aetherforge` gateway/mesh/swarm subcommands
- **MCP stdio**: `aetherforge-mcp` forge_generate, forge_mesh_status, ...

## 2. 上游依赖

- agora (I0)
- bus-foundation (X)

## 3. 下游影响

- runtime
- kairon
- swarm-engine

## 4. 配置 / SSOT

- 项目源码：`projects/aetherforge/`
- 入口定义：`projects/aetherforge/pyproject.toml` 或 `package.json`
- 测试：`cd projects/aetherforge && make test`
