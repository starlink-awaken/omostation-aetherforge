---
type: ssot
owner: governance-team
last_updated: 2026-09-03
---

# llm-gateway → aetherforge/packages/gateway/ 合并清单

> 状态：Phase 1 物理迁移完成（2026-06-16）  
> 下一步：逐项消化 `_legacy/` 中的能力到主 `llm_gateway/` 包

## 1. 背景

`projects/llm-gateway/` 已作为 `aetherforge/packages/gateway/` 的源码上游运行一段时间，但两者已出现显著分叉：

- **llm-gateway 独有**：`audit.py`、`budget.py`、`bus_adapter.py`、`quota_ledger.py`、`registry_data_loader.py`、`registry_data/`
- **aetherforge-gateway 独有**：`credentials*.py`、`metrics.py`、`paths.py`、`plugins.py`、`pricing.py`、`quota_engine.py`、`rate_limiter.py`、`route_scheduler.py`、更多 providers（azure/bedrock/mock/vertex）
- **同名文件均有差异**：`__init__.py`、`cli.py`、`detection.py`、`http_server.py`、`mcp_server.py`、`policies.py`、`provider.py`、`providers/*.py`、`scheduler.py`、`types.py`

## 2. 本次完成项

- [x] 将 `llm-gateway/src/llm_gateway/` 全部源码复制到 `aetherforge/packages/gateway/src/llm_gateway/_legacy/`
- [x] 清理 `__pycache__` 和 `.pyc`
- [x] 保留主 `llm_gateway/` 包不变，避免破坏现有 `aetherforge gateway` CLI 与测试
- [x] 更新 `pyproject.toml`：添加 `aetherforge-llm-gateway-legacy` 兼容入口

## 3. 待消化能力清单

| 文件 | 来源 | 目标 | 优先级 | 说明 |
|:--|:--|:--|:--:|:--|
| `audit.py` | llm-gateway | `metrics.py` / 新 `audit.py` | P1 | LLM 调用审计，与 aetherforge metrics 互补 |
| `budget.py` | llm-gateway | `pricing.py` / `quota_engine.py` | P1 | 预算限制，需与 quota_engine 统一 |
| `bus_adapter.py` | llm-gateway | `plugins.py` / 新 adapter | P2 | bus_foundation 适配，需确认是否仍使用 |
| `quota_ledger.py` | llm-gateway | `quota_engine.py` | P1 | 配额记账，可能重复 |
| `registry_data_loader.py` + `registry_data/` | llm-gateway | `registry.py` / `ssot_loader.py` | P1 | P4 静态 registry 加载，MCP `list --registry-data` 依赖 |
| `provider.py` 中的 `finalize_llm_response` / `infer_compute_route` | llm-gateway | `provider.py` | P1 | 响应后处理与算力路由 |
| 各 provider 的 `finalize_llm_response` 调用 | llm-gateway | `providers/*.py` | P1 | 需合并到 aetherforge providers |
| `cli.py` 中的 `list --registry-data` 等命令 | llm-gateway | `cli.py` | P1 | 原 llm-gateway CLI 功能 |

## 4. 合并原则

1. **不破坏 `aetherforge gateway` CLI**：主包保持稳定；
2. **逐步消化 `_legacy/`**：每个能力单独评审、迁移、测试；
3. **优先处理 P1**：audit、budget、quota、registry_data_loader 是原 llm-gateway 的核心差异化能力；
4. **删除重复**：`quota_ledger.py` 与 `quota_engine.py` 需统一为单一实现；
5. **保留历史**：`_legacy/` 在完全消化前不删除，便于 diff 追溯。

## 5. 验收标准

- [ ] `_legacy/` 中能力全部迁移到主包或明确废弃；
- [ ] `aetherforge/packages/gateway/src/llm_gateway/_legacy/` 目录可删除；
- [ ] `aetherforge gateway` CLI 覆盖原 `llm-gateway` 全部命令；
- [ ] `projects/llm-gateway/` 标记 archived。
