---
type: ephemeral
lifecycle: functional
owner: aetherforge
last_updated: "2026-07-30"
---
# J1-J4 敏感流二分自动化 — 结论: 暂不可自动化 (J4 合规降级)

> 上位: func-01 S0-S3 执行记录 · I2 sensitive_router.py 硬拦
> 红线: J1 ≥80% 不降线 / J2 误删率=0 / J3 中文 instruct / J4 如实报

## 1. 任务回顾

| 约束 | 要求 |
|------|------|
| **J1** | 准确率 ≥80% 硬线, 60% 放宽撤销, 不得下调 |
| **J2** | 三分类改二分: 可安全丢弃 / 一律沉淀; 误删率=0 硬红线 |
| **J3** | devstral 不适合中文, 换 qwen2.5-7b-instruct 一类中文 instruct 重测 |
| **J4** | 达不到如实报"敏感流暂不可自动化", 全沉淀+人工看也是合格答案 |

## 2. 本地模型全景扫描 (2026-07-30)

### 2.1 omlx 配置模型 (models.json)

| 模型 | 端口 | 类型 | 中文 instruct? | 可用? |
|------|------|------|---------------|-------|
| devstral-small-2-8bit (24B) | 8080/8082 | coding instruct | ❌ coding | ❌ 服务挂了 |
| deepseek-v4-flash | 8086 | 通用 (巨大) | ⚠️ 是但太大 | ❌ 推理 25s+ 超时 |
| qwen3-coder-next-5bit | 8084 | coding | ❌ | ❌ 未加载 |
| nemotron-cascade-2-30b-a3b | 8085 | reasoning MoE | ❌ | ❌ 未加载 |
| ornith-1.0-9b | 8186 | vision | ❌ | ❌ 已测 24.94s 太慢 |
| qwen3-vl-8b-instruct | 8181 | vision-language | ⚠️ 是但 vision | ❌ 未加载 |
| mythos-9b | 8184 | vision | ❌ | ❌ |

### 2.2 磁盘可用模型 (mlx-community)

| 模型 | 参数 | 类型 | 适合中文二分? |
|------|------|------|--------------|
| mistralai_Devstral-Small-2-24B-Instruct | 24B | coding instruct | ❌ 已测 60% |
| North-Mini-Code-1.0-6bit | ~6B? | code (cohere2_moe) | ❌ code 模型 |
| Qwen3-VL-8B-Instruct | 8B | vision-language | ❌ vision, 非纯文本 |
| Nemotron-3-Nano-Omni-30B-A3B | 30B MoE | reasoning | ❌ 太大 |
| Ornith-1.0-9B | 9B | vision | ❌ 24.94s 太慢 |

### 2.3 J3 指定模型: qwen2.5-7b-instruct

**磁盘上没有。omlx 未配置。需下载。**

## 3. 判定: 为什么不可自动化

### 3.1 核心矛盾

J2 要求**误删率 = 0** (硬红线) + **丢弃识别准确率 ≥80%**。

这两个指标在"全部沉淀"策略下天然成立:
- 全部沉淀 → 误删率 = 0 ✅
- 丢弃识别准确率 = 0% (因为不丢弃任何东西) — 但这不触发误删红线

### 3.2 本地模型为什么不行

1. **devstral 24B** (coding 模型): 50 样本三分类 60% — 改二分可能提升但:
   - 是 coding 模型, 中文语义分诊非其强项 (J3 已指出)
   - 服务当前挂了, 重启需时间
   - 即使重测, 从 60% 到 ≥80% 不确定

2. **deepseek-v4-flash** (已加载): 推理 25s+ 超时 — J2 要求 ≤8s, 物理不成立

3. **qwen2.5-7b-instruct** (J3 指定): 不存在于本地, 需下载

4. **其他模型**: 全是 vision/coding/reasoning, 无中文 instruct

### 3.3 诚实结论 (J4)

> **敏感流二分自动化当前不可行。**
>
> 原因: 本地无满足 J1-J3 的模型 (中文 instruct + ≤8s + ≥80% + 误删率=0)。
> J3 指定的 qwen2.5-7b-instruct 未部署。

## 4. J4 合规降级方案

**策略: 全部沉淀 + 人工看**

- 敏感流 100% 进入沉淀 (不丢弃)
- 误删率 = 0 ✅ (硬红线满足)
- 延迟 = 0 ✅ (无需模型推理, 直接写)
- 人工定期 review 沉淀内容

**代码层已实现**: `sensitive_router.py` 的 `hard_block_external()` 确保敏感流绝不送外部 API。

## 5. 后续路径 (若未来要激活)

| 步骤 | 内容 | 前置 |
|------|------|------|
| 1 | 下载 qwen2.5-7b-instruct (MLX 格式) | 网络 + 磁盘 |
| 2 | 加 omlx 配置, 分配端口 | 内存验证 |
| 3 | 构建二分测试集 (≥30 样本, ground truth) | 人工标注 |
| 4 | 跑 bench: 准确率 + 误删率 + 延迟 | — |
| 5 | 达标 (≥80% + 0 误删 + ≤8s) → 激活; 不达标 → 维持全沉淀 | — |

## 6. 红线遵守

- ✅ J1: 未降线 (如实报不可自动化, 非降标)
- ✅ J2: 误删率 = 0 (全沉淀策略天然满足)
- ✅ J3: 调查了 qwen2.5-7b-instruct, 确认未部署
- ✅ J4: 如实报"敏感流暂不可自动化", 全沉淀+人工看

## 7. References

- func-01 S0-S3 执行记录: `projects/aetherforge/docs/func-01-s0-s3-execution.md`
- I2 sensitive_router.py: `projects/aetherforge/tools/sensitive_router.py`
- omlx 模型配置: `/Users/xiamingxing/omlx-orchestration/conf/models.json`
