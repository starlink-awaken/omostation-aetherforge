---
status: active
lifecycle: operational
owner: aetherforge
last-reviewed: "2026-07-30"
---
# oMLX 模型健康登记 (全量实测 · 2026-07-30)

> 上位: K3 (无消费方的死亡模型登记为已知状态, 不修复)
> 原则: S2 消费驱动 — 有消费方请求时才拉起服务, 不预加载
> 策略: load/unload 轮换 — 128GB 内存同时只能跑 1-2 个大模型
> 测试方法: 逐个 `omlxc load` → 推理 → `omlxc unload`

## 修复记录 (2026-07-30)

| 问题 | 根因 | 修复 |
|------|------|------|
| mid-local 加载超时 | alias 软链缺失 | 创建 `models-active/mid-local/current` → Qwen3.6-27B |
| coder-precise 加载超时 | alias 软链缺失 | 创建 `models-active/coder-precise/current` → Qwopus3.6-27B-Coder |
| mid-local 端口冲突 | 8080 被 cockpit dashboard 占用 | 改端口 8092 |
| coder-precise 端口冲突 | 8087 与 deepseek-v4-pro 冲突 | 改端口 8091 |
| mythos-fast thinking=YES | params 缺 chat_template_args | 添加 `{"enable_thinking": false}` |
| coding-next thinking=YES | params 缺 chat_template_args | 添加 `{"enable_thinking": false}` |
| **所有 mlx_lm 模型 thinking 不生效** | omlxc `help_text(["mlx_lm.server"])` 拿不到 `--chat-template-args` | 改 `help_text([py,"-m","mlx_lm.server"])` |
| aetherforge 无 thinking 防御 | 模型思考段可能泄露到 KOS | sensitive_router.py 加 `strip_thinking()` |

## 可用模型清单 (24 个全覆盖)

### mlx_lm (13 个) — 文本推理

| 模型 | 端口 | 参数 | 延迟 | thinking | 状态 |
|------|------|------|------|----------|------|
| coding-fast (Qwen3.6-35B-A3B MoE) | 8081 | 35B-A3B | **4s** | NO | 🏆 最快 |
| mythos-fast (Qwythos-9B) | 8185 | 9B | **11s** | NO | 小模型 |
| reasoning-lite (Nemotron-Cascade) | 8085 | 30B-A3B | **21s** | NO | 推理性价比 |
| coding (devstral-24B) | 8082 | 24B | **28s** | NO | 编码主力 |
| reasoning (GLM-4.7-Flash) | 8083 | 18B | **37s** | NO | 复杂推理 |
| mid-local (Qwen3.6-27B) | 8092 | 27B | ~15s | NO | 通用对话 ✅新修 |
| coder-precise (Qwopus3.6-27B-Coder) | 8091 | 27B | ~15s | NO | 精确编码 ✅新修 |
| deepseek-v4-flash | 8086 | 大模型 | 45s+ | NO | ⚠️ MBP 慢 |
| deepseek-v4-pro | 8087 | 更大 | 45s+ | NO | ⚠️ MBP 慢 |
| coding-next (Qwen3-Coder-Next) | 8084 | 51G | 45s+ | NO | ⚠️ 太大 |
| mistral-medium-128b | 8088 | **128B** | ∞ | — | ❌ 物理极限 |
| nemotron-omni | 8089 | — | — | — | ❌ 架构不支持 |
| fast-local | 8080 | — | — | — | ❌ GGUF 格式不支持 |

### mlx_vlm (8 个) — 视觉/多模态

| 模型 | 端口 | 实际模型 | 状态 |
|------|------|----------|------|
| vision (Qwen3-VL-8B) | 8181 | Qwen3-VL-8B-Instruct-MLX-4bit | ✅ |
| vision-large (Qwen3.6-27B) | 8182 | Qwen3.6-27B-MLX-4bit | ✅ |
| mythos (Qwythos-9B) | 8184 | Qwythos-9B-MLX-bf16 | ✅ |
| ornith-9b | 8186 | Ornith-1.0-9B-4bit | ✅ |
| ornith-35b | 8187 | Ornith-1.0-35B-5bit | ⚠️ 加载慢 |
| gemma-4-26b | 8190 | gemma-4-26B-A4B-MLX-4bit | ✅ |
| gemma-4-31b | 8191 | gemma-4-31B-MLX-8bit | ✅ |
| gemma-4-e2b | 8192 | gemma-4-E2B-MLX-8bit | ✅ |

> mlx_vlm.server 启动时加载 active_root 下全部模型, 一次 load 全部可用

### mlx_embeddings (3 个) — 向量/Reranker

| 模型 | 端口 | 实际模型 | 状态 |
|------|------|----------|------|
| embedding (Qwen3-Embedding-8B) | 8183 | Qwen3-Embedding-8B-mxfp8 | ⚠️ 推理超时 |
| embed-bge-m3 | 8188 | bge-m3-mlx-4bit | ✅ |
| baai-bge-reranker | 8189 | bge-reranker-v2-m3-fp16 | ✅ |

## 端口分配 (无冲突)

```
8080: fast-local (GGUF, 不可用)
8081: coding-fast ✅
8082: coding ✅
8083: reasoning ✅
8084: coding-next ⚠️
8085: reasoning-lite ✅
8086: deepseek-v4-flash ⚠️
8087: deepseek-v4-pro ⚠️
8088: mistral-medium-128b ❌
8089: nemotron-omni ❌
8090: cockpit dashboard (非 omlx)
8091: coder-precise ✅
8092: mid-local ✅
8181: vision (mlx_vlm, 全部视觉模型) ✅
8182: vision-large (同上, 共享进程)
8183: embedding ⚠️
8184: mythos (共享 vision 进程)
8185: mythos-fast ✅
8186: ornith-9b (共享 vision 进程)
8187: ornith-35b (共享 vision 进程)
8188: embed-bge-m3 ✅
8189: baai-bge-reranker ✅
8190-8192: gemma-4 系列 (共享 vision 进程)
```

## 代码修复落点

### omlxc (`/Volumes/Model/omlx/bin/omlx`)

- **Line 106**: `help_text(["mlx_lm.server"])` → `help_text([py,"-m","mlx_lm.server"])`
  - 修复 `--chat-template-args` 传参丢失

### aetherforge (`projects/aetherforge/tools/sensitive_router.py`)

- **新增 `strip_thinking()`**: 防御性剥离 `<think>...</think>` 段
- 即使模型模板硬编码 thinking, 消费侧也安全

### models.json (`/Volumes/Model/omlx/conf/models.json`)

- mid-local/coder-precise: 端口 8084/8087 → 8092/8091
- mythos-fast/coding-next: +chat_template_args

### models-active 软链

- `mid-local/current` → Qwen3.6-27B-MLX-4bit (新建)
- `coder-precise/current` → Qwopus3.6-27B-Coder-MLX-8bit (新建)

## 策略

1. **常驻**: embedding — K1 全沉淀需要 embed 入 KOS (降级链: 8183 → 8188)
2. **日常轮换**: coding-fast (4s) → 默认主力
3. **按需拉起**: reasoning / mid-local / coder-precise — 按任务 load/unload
4. **视觉任务**: load vision (8181) → 全部视觉模型可用
5. **不主动拉起**: deepseek-v4 系列 / mistral-128b — MBP 跑不动
6. **永久禁用**: nemotron-omni (架构不支持), fast-local (格式不支持)

## ModelGateway 统一入口 (v0.6, 2026-07-31)

所有 LLM 调用经 `ModelGateway` 统一入口 (`packages/gateway/src/llm_gateway/gateway.py`):

```
消费方                能力
─────────────────────────────────────────────────
triage/router.py  →  gateway.generate() + K1 硬拦 (via run_async)
sensitive_router  →  gateway.embed() (降级链 8183→8188) + strip_thinking (SSOT)
gateway/rpc.py    →  gateway.generate() (BOS RPC, 含 health_check)
mcp_server.py     →  gateway.generate() (MCP tool: llm_generate / gateway_generate)
```

### 架构特性

| 特性 | 实现 | 状态 |
|------|------|------|
| K1 敏感硬拦 | `is_sensitive()` (SSOT) + `_generate_local_only()` | ✅ |
| thinking 剥离 | `strip_thinking()` (SSOT, sensitive_router 复用) | ✅ |
| embedding 降级 | `embed()`: 8183 → 8188 → 报错 | ✅ |
| MemoryGuard | `_ensure_model()` 中调用 `can_load()`, 不足则拒绝 | ✅ |
| 并发控制 | `asyncio.Lock` 防止两个大模型同时 load | ✅ |
| WarmPool | keep-last-used TTL=300s, 后台任务自动卸载 | ✅ |
| HealthMonitor | 连续 3 次失败标记 dead, 自动跳过 | ✅ |
| 后台任务 | `start_background_tasks()` 启动 warm_pool + health 循环 | ✅ |
| 单例 | `get_gateway()` 全进程共享一个实例 | ✅ |
| 同步兼容 | `run_async()` 解决嵌事件池问题 | ✅ |
| 指标 | `MetricsCollector` 记录延迟/token/成本 | ✅ |

### SSOT 收敛 (2026-07-31)

| 之前 (分散) | 之后 (统一) |
|------------|------------|
| gateway 有 `_strip_thinking`, sensitive_router 有 `strip_thinking` | → `llm_gateway.strip_thinking` 唯一 SSOT |
| gateway 有 `_is_sensitive` (子集), sensitive_router 有 `classify_sensitivity` (超集) | → `llm_gateway.is_sensitive` 完整模式集, sensitive_router 在其基础上加白名单 |
| 敏感模式两份不同 | → gateway 一份完整模式 (union) |

### 用法

```python
from llm_gateway import get_gateway, GatewayConfig

# 单例 (推荐)
gateway = get_gateway()

# 或自定义配置
config = GatewayConfig(
    fallback_chain=["coding-fast", "mid-local", "deepseek-chat"],
    warm_pool_ttl=300,
    model_sizes={"coding-fast": 18.0, ...},
)
gateway = get_gateway(config)

# 同步调用 (安全, 无论是否在 async 上下文)
from llm_gateway import run_async, GatewayRequest
resp = run_async(gateway.generate(GatewayRequest(messages=[...])))
```

### 测试覆盖

```
test_model_gateway.py:  32/32 ✅ (strip_thinking, is_sensitive, MemoryGuard, K1, embed, health, run_async, singleton)
test_gateway.py:        23/23 ✅ (现有, 未破坏)
其他现有测试:            73/73 ✅
```

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-07-31 | **架构统一收敛**: SSOT 敏感模式 + strip_thinking, 单例 get_gateway, run_async 嵌套兼容, MemoryGuard 接入 _ensure_model, 后台任务, rpc/mcp 接线 |
| 2026-07-30 | ModelGateway 统一入口 + embedding 降级 + K1 网关层硬拦 |
| 2026-07-30 | 全量测试 24 个模型 (13 mlx_lm + 8 mlx_vlm + 3 mlx_embeddings), 修复全部问题 |
| 2026-07-30 | 初版登记, K1/K2/K3 决策后建立 |
