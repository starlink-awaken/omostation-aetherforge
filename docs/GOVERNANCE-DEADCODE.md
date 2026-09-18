---
type: ssot
owner: governance-team
last_updated: 2026-09-03
last-reviewed: 2026-09-18
---

# 死代码台账 — 零调用方法审计（2026-08-24）

> 治理附录 · 审计命令与判据见文末 · 处置原则: 接线或删除, 不留"建了没接"

## 背景

"方法存在但零接线"在本仓库已实证三例(record_usage / estimate_cost 格式错配 /
record_error), 均为功能写了但从未生效。本次系统审计(AST 收集 + 全仓双模式
grep 引用检测, 排除框架覆写/property/无括号注册)清查全部 185 个 public 方法。

## 审计结果: 23 个真嫌疑, 5 个整模块未接线

### 整模块级(建了没接的功能层)

| 模块 | 未接线 API | 判断 |
|---|---|---|
| `rate_limiter.py` | async_acquire / remove_limit / set_default_limits / set_rate_limiter(接线口) | 限流器整体未生效 —— 429 防护全靠上游, 网关层无主动限流 |
| `plugins.py` | create_plugin_provider / list_plugins / unregister_plugin | 插件体系未接线 |
| `policies.py` | list_policies / score_models | 策略引擎未接线(路由打分走 scheduler 内置) |
| `mcp_server.py` | gateway_generate / gateway_health / get_prompt | MCP 服务面未接线 |
| `scheduler.py` 生命周期 | dispose / from_m1_dir / get_all_loads / start_auto_refresh | auto_refresh 未启用 = scheduler 数据可能静态 |

### 零散方法

| 方法 | 文件 | 备注 |
|---|---|---|
| export_jsonl | metrics.py | 观测导出无人触发(数据锁内存, /stats 已部分弥补) |
| check_constraint | credentials.py | |
| get_providers / register_many / set_metrics_collector | registry.py | |
| select_all | route_scheduler.py | |
| invalidate | quota_engine.py | 预算缓存失效未接线 |

## 处置建议(按 Y1"系统变小"原则)

1. **一个月内无接线计划的, 删除**(plugins/policies/route_scheduler 的未接 API)
   —— 每个死模块都是阅读负担与伪能力
2. **有真实价值的, 排期接线**:
   - rate_limiter(429 防护, 与 reverify 打通)
   - export_jsonl(观测持久化, 与日报打通)
   - start_auto_refresh(调度数据新鲜度)
3. 新增代码的防腐规约: **PR 必须包含调用点或明确标注"暂不接线+计划"**,
   杜绝第四例 record_error

## 审计方法(可复现)

```bash
# AST 收集 public def → 对每名双模式 grep:
#   name(  调用     name[^(:\w]  无括号引用
# 排除: 框架覆写(do_GET 等) / @property / 测试文件 / def 行自身
```
