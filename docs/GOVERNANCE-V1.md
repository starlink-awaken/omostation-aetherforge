---
type: ssot
owner: governance-team
last_updated: 2026-09-03
---

# AetherForge 网关治理架构 v1 — 成熟度模型、差距与落地路线

> 状态: active · 创建: 2026-08-24 · 所有者: gateway 维护者
> 触发: 2026-08-24 治理 goal("可进化/可迭代/可观测/可追溯/可排查/可优化, 成熟度 90%")
> 本文档是治理 SSOT: 差距以实测为准, 打分标准显式, 每一项 P1/P2 都有验收命令。

## 1. 范围与背景

AetherForge 网关(22 引擎 / 369+ 模型)在 2026-08-22~24 的深度重构后达到
"全链路可用"(流式/工具/多轮/凭据自治/预算记账), 但复盘暴露出**治理维度
的系统性缺口**: 关键治理动作(凭据淘汰/预算拦截/流式回退)只有人读的散装
文本日志, 观测数据锁在进程内存无出口, 免费池运营工具在多 agent 并发中
永久丢失。本文档定义六维成熟度模型, 量化差距, 并给出分阶段落地路线。

## 2. 六维成熟度模型(打分标准显式)

| 维度 | 判定标准(可验证) | v1 前 | P0 后 | 90% 目标态 |
|---|---|---|---|---|
| **可观测** | 核心指标有 HTTP 出口, 无需登机器 | 20% (report() 锁内存) | **80%** (/stats 聚合五源) | 90% (+错误率/分位延迟面板) |
| **可追溯** | 关键治理动作留结构化事件, 按 kind 可过滤 | 10% (30+ 处自由文本) | **75%** (6 类事件 JSONL) | 90% (+请求级 trace_id 贯穿) |
| **可排查** | 一次故障 ≤3 步定位(端点→事件→日志) | 25% (要 ssh+sqlite+grep) | **70%** (/stats 一屏) | 90% (+错误分类码全量) |
| **可进化** | 新 provider/免费源接入 ≤1 处配置 | 50% (SSOT 引擎 yaml) | 50% | 90% (P1: 发现闭环内建) |
| **防腐** | 协议/事件契约有测试锁定, 无隐式漂移 | 60% (423 测试含协议) | **70%** (kind 契约 +4) | 90% (P1: tool_calls 场景矩阵) |
| **可持续运营** | 免费池发现/凭据轮换/预算巡检自动化 | 35% (reverify 5min 已挂) | 35% | 90% (P2: 发现闭环+巡检报表) |

**"90% 成熟度"的严格定义: 六维全部 ≥90% 且各有验收命令。** P0 后加权
约 63% — 如实声明: 本轮把观测三兄弟(可观测/可追溯/可排查)从地面拉到
70-80%, 进化/运营两维的方案在 §5, 落地在后续 PR。

## 3. 差距清单(2026-08-24 实测)

| # | 差距 | 证据 | 处置 |
|---|---|---|---|
| G1 | 结构化日志为零 | gateway.py 0 处 json.dumps/structlog, 30 处散装 _log | **P0 已修**: events.py |
| G2 | 指标无出口 | 端点清单无 /stats, report() 无人调用展示 | **P0 已修**: /stats |
| G3 | 云端错误无分类 | OmlxcErrorCode 仅 6 个本地语义码; auth/rate_limit/budget/empty/fallback 全靠 RuntimeError 字符串 | P1 |
| G4 | 免费池运营工具丢失 | free-model-scanner.py / provider-sync.py 磁盘+git 历史均无(多 agent 清理受害, 同 omostation AGENT-BRIEF §1.1 历史模式); sync 功能已由 import_from_cc_switch 内建覆盖, 丢失的是"发现新免费源" | P1: 发现闭环内建 gateway |
| G5 | tool_calls 复杂场景未压测 | 只验过最简闭环(一问一工具一结果) | P1: 场景矩阵测试 |
| G6 | 凭据非标失效盲区 | reverify 只认 401/403; "200+body 错误"型失效无识别 | P2 |
| G7 | ecos CI 持续 failure | mof-enforce 硬编码 $HOME 路径, 单仓库 checkout 缺兄弟项目 | 独立线, 方案见 §5.4 |

## 4. 目标架构(分层与治理机制)

```
┌─ 接入层 openai_proxy ── /v1/* /health /ready /stats ◄─ P0 新增
├─ 路由层 gateway/scheduler ── 真流式优先+聚合回退, 复杂度分流
├─ 凭据治理层 credentials ── 加权轮选/reverify 判死复活/请求驱动 401 淘汰
├─ 财务治理层 budget+pricing ── 预算拦截/PricingRegistry 计价/usage_log
├─ 观测层 events+metrics ◄─ P0 新增 ── JSONL 事件流(6 kind)/指标聚合
└─ 运营层(P1) ── 免费源发现闭环/凭据巡检报表/预算月报
```

**各层治理机制(已生效部分)**:
- 凭据生命周期状态机: active → (probe 401/403 或 request 401) → dead →
  (复验 200) → active。每次跃迁 emit 事件(P0 起), 状态可从 /stats 读。
- 财务闭环: 请求前 budget_blocked 拦截 → 成功后 estimate_cost 计价 →
  usage_log/month_spend 累计 → 超限自动拦截。免费层引擎 cost=0 是 SSOT
  设计意图(SSOT yaml cost_multiplier=0), 不误计。
- 防腐契约: 事件 kind 是稳定契约(test_events.py 锁定); 协议形状
  (OpenAI tool_calls/tool_use 映射)有专项测试; PR 纪律+sync-and-deploy.sh
  固化部署链。

## 5. 落地路线

### P0(本轮, 已完成)
1. events.py 事件流 + 5 挂点(request_complete/budget_blocked/
   credential_evicted×2/stream_fallback) + 契约测试×4
2. /stats 端点(metrics/凭据健康/预算/registry/近 20 事件五源聚合,
   Bearer 保护)
验收: `curl -H "Authorization: Bearer $KEY" http://127.0.0.1:4000/stats`

### P1(下一轮)
1. **CloudErrorCode 分类**: auth/rate_limit/budget/empty/fallback 五码
   进 GatewayResponse.error_code, /stats 按码聚合错误率
2. **免费源发现闭环内建**: health loop 周期性对候选 provider 列表
   (ENV_SSOT 配置)做 models 探测, 新源出现→emit provider_discovered
   事件→人工/自动接线。**不再做外挂脚本**(G4 教训: 外挂易丢失)
3. **tool_calls 场景矩阵**: 并行多工具/工具+文本混合/超大 arguments/
   畸形 JSON 分片 四场景契约测试
4. ecos mof-enforce 兼容修复: 单仓库环境(CI)下降级 skip 多仓库扫描项,
   `--json` 输出 skipped 原因 — 属 ecos 仓 PR, 独立走

### P2(规划)
1. 凭据非标失效探测(200+body 错误特征库, 每源白名单式学习)
2. 运营报表: events.jsonl 聚合日报(请求量/成本/淘汰/拦截), launchd 定时
3. trace_id 贯穿(request_complete 事件带 proxy 请求 ID, 与 access log 对齐)

## 6. 长期防腐规则(治理的"不变量")

1. **观测先于功能**: 新治理机制必须同时 emit 事件, 否则不予合并
2. **kind 只增不改**: 事件 kind 一旦发布即冻结, 变更走新 kind + deprecate
3. **能力内建优于外挂**: 任何"运维脚本"若承载持续机制, 必须进仓库+进
   测试, 否则视为未交付(G4 的制度化的教训)
4. **免费/付费一视同仁计量**: 免费层 cost=0 也必须有 usage_log 记录
   (用量≠成本, 运营决策要看用量)
5. 既有纪律延续: PR 纪律(不直推 main)/sync-and-deploy 部署链/
   端到端真实验证(不满足于测试绿)
