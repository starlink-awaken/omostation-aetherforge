# _legacy/ — llm-gateway v0.4 (deprecated)

> **Status**: DEPRECATED since 2026-06-16 (P43 R5 closure)
> **Replacement**: [`../`](../) — active llm-gateway v1.0+
> **Removal**: Planned for aetherforge v2.0 (Q4 2026)

## 历史背景

2026-06-16 之前, llm-gateway 是独立项目 `projects/llm-gateway/`, 自带 v0.4 三位一体
架构 (provider / quota / scheduler).

P43 R5 完成架构合并 (commit `31381ed` in projects/aetherforge), llm-gateway v0.4
代码全部迁移至本目录 (`_legacy/`), active llm-gateway v1.0+ 在 `../` 实现.

## 治理面 SSOT

- **X1 策略**: `X1-ARCH-MERGE-LLMGATEWAY-20260616` (active) — 架构变更审计
- **X4 一致性**: `X4-CONS-LLMGATEWAY-ARCHIVED` (active) — 禁止活跃项目形式
- **L0 约束**: 31 条, 含 `X4-CONS-LLMGATEWAY-ARCHIVED` 引用

## 工具链配置

**ruff --exclude** (`.omo/_truth/x1-governance-policies.yaml`):
```toml
[tool.ruff]
exclude = ["packages/gateway/src/llm_gateway/_legacy"]
```

**X2 freshness audit** (`scripts/omo/x2_freshness_audit.py`):
```python
exclude_args.append("--exclude=packages/gateway/src/llm_gateway/_legacy")
```

**pyproject optional**:
```toml
[project.optional-dependencies]
legacy = ["click", "rich", "tabulate", "gitpython"]
```

## 文件清单 (20+)

- `audit.py` — v0.4 audit CLI (已弃用, 迁移到 aetherforge audit)
- `budget.py` — v0.4 budget (active version in `../budget.py` 重写)
- `bus_adapter.py` — v0.4 bus adapter (active version in `../runtime_bus_adapter.py`)
- `circuit_breaker.py` — v0.4 circuit breaker
- `cli.py` — v0.4 CLI (active: aetherforge CLI)
- ... (详见 `__init__.py` 完整列表)

## 迁移路径

| v0.4 (legacy) | v1.0+ (active) |
|----------------|------------------|
| `llm_gateway.budget.BudgetExhausted` | `llm_gateway.budget.BudgetExhaustedError` |
| `llm_gateway.cli.main` | `aetherforge.cli.gateway` |
| `llm_gateway.providers.AnthropicProvider` | `llm_gateway.providers.anthropic_provider.AnthropicProvider` |
| `llm_gateway.scheduler.CACHE_PATH` | `llm_gateway.scheduler.cache_path` |

**完整迁移指南**: `projects/aetherforge/MIGRATION.md` (待补充)

## 何时删除

条件 1: aetherforge v2.0 release
条件 2: v0.4 API 调用方全部迁移 (通过 grep -r "from llm_gateway._legacy" 验证 = 0)
条件 3: ruff --exclude 移除 (即 X2 freshness audit 0 errors 跨 _legacy/)

**最早删除时间**: 2026-09-16 (合并后 3 个月, 给生态充分迁移时间)