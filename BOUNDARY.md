# aetherforge — System Boundary

> 本文档描述 aetherforge 与 eCOS 系统其他部分的边界：暴露的接口、依赖的上游、影响的下游。
>
> 系统全景参见：[`../../docs/PANORAMA.md`](../../docs/PANORAMA.md)

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
- cockpit / agora-dashboard

## 4. 配置 / SSOT

- 项目源码：`projects/aetherforge/`
- 入口定义：`projects/aetherforge/pyproject.toml`
- 测试：`cd projects/aetherforge && make test`

## 5. 归档说明

- `projects/compute-mesh` 已于 2026-06-16 从工作区子模块中移除并归档至 `_archived/compute-mesh/`。
  其 mesh-specific 代码（拓扑、调度、Worker、API）已并入 `projects/aetherforge/packages/mesh/src/compute_mesh/`，`provider/` 层与 `aetherforge-gateway` 合并，不再独立维护。
- `projects/swarm-engine` 已于 2026-06-16 归档至 `_archived/swarm-engine/`；缺失的 `swarm_engine` 模块已并入 `projects/aetherforge/packages/swarm/src/swarm_engine/`，不再独立维护。
- `projects/aetherforge-swarm-ext` 已于 2026-06-16 归档至 `_archived/aetherforge-swarm-ext/`；14 个唯一扩展模块已并入 `projects/aetherforge/packages/swarm/src/swarm_engine/ext/`，其余模块已由 `swarm-engine` 合并覆盖，不再独立维护。

## 架构演进与项目边界索引

参见工作区架构演进与项目边界：[`../../docs/ARCHITECTURE-EVOLUTION.md`](../../docs/ARCHITECTURE-EVOLUTION.md)
