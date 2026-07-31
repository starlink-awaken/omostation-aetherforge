# FUNC-01 S0: 本地模型清单 + 分诊选型

> 功能线文档 (非治理标准). 上位: func-01-info-triage-pipeline.md S0.
> 红线: 撑不住如实降级 DeepSeek, 不为"用上本地"牺牲质量.

## 本地模型实际清单 (>13, 跨 LMStudio/omlx/ollama)

### LMStudio 在线 (31 模型, 主力)
分诊候选 (小快, ≤30B 或 flash):
- **google/gemma-4-e2b** (2B, reasoning model, 最小)
- google/gemma-4-26b-a4b-qat (26B)
- google/gemma-4-31b-it-mlx (31B)
- ornith-1.0-9b (9B)
- ornith-1.0-35b-xl-mlx (35B, 大)
- qwen3.6-27b (27B)
- qwythos-9b-claude-mythos-5-1m (9B, 多版本)
- deepseek-v4-flash-mtp-mlx (flash)
- deepseek-v4-pro-mtp-mlx (pro, 大)
- nemotron-3-nano-omni-30b (30B)
- (kokoro-82m = TTS, 不支持 chat)

### omlx models-active (4 类, 软链接到 LMStudio)
- coding: devstral-small-2 / deepseek-v4-flash / deepseek-v4-pro
- reasoning: glm-4.7-flash-8bit / mistral-medium-128b
- retrieval: bge-reranker / bge-m3 / qwen3-embedding (embedding, 非 chat)
- vision: gemma-4-26b / 31b / e2b

### ollama (2 模型)
- gemma4:31b-mlx
- north-mini-code-1.0:mlx-nvfp4

## 分诊候选 (S0 实测对象)

分诊 = 海量/简单/重复/判据明确 (三选一: 丢弃/沉淀/提醒).
需小快模型. reasoning model 准确但慢 (gemma-4-e2b 14s/条).

候选 (按小快排序):
1. **google/gemma-4-e2b** (2B reasoning, 准确但 14s)
2. ornith-1.0-9b (9B)
3. deepseek-v4-flash-mtp-mlx (flash)
4. qwen3.5-9b-deepseek-v4-flash-mtp (9B)
5. qwythos-9b (9B)
6. nemotron-3-nano-omni (30B, 大)

## 实测 (func-01-s0-triage-bench.py)

20 真实样本 (丢弃6/沉淀8/提醒6), max_tokens=500 (reasoning 需足够), LMStudio API.

**初步发现** (gemma-4-e2b 单样本):
- content: '沉淀' ✅ (技术教程→沉淀, 正确)
- reasoning_content: thinking (282 reasoning_tokens)
- 延迟 14.6s (reasoning 占大头)
- 是 reasoning model (max_tokens <300 会截断 content 空)

**待实测完填**: 各候选准确率 + 延迟 + 选型建议.

## 选型标准
- 准确率 ≥80% (够用)
- 延迟最低 (最便宜, 本地无金钱成本 = 算力 = 延迟)
- 若都 <80% → 降级 DeepSeek (deepseek-chat $0.0005/$0.0015, 准确高)

## S0 实测结果 (2026-07-29, 20 样本三档)

| 模型 | 准确率 | 延迟 | 错误 | 来源 |
|------|--------|------|------|------|
| ornith-1.0-9b | **16/20 (80%)** | 24.94s | 4 | 本地 LMStudio |
| qwen3.5-9b-deepseek-v4-flash | 12/20 (60%) | 12.64s | 8 | 本地 LMStudio |
| qwythos-9b-claude-mythos | 0/20 (0%) | 7.17s | 20 | 本地 (API 失败) |
| **deepseek-chat** | **16/20 (80%)** | **1.36s** | **0** | 云 DeepSeek API |

## 选型结论: deepseek-chat (降级, 实测优于本地)

**决定**: 分诊主模型 = **deepseek-chat** (云).

**理由** (红线"撑不住如实降级, 不为用本地牺牲质量"):
- **准确率同 80%** (本地 ornith = DeepSeek, 不牺牲质量)
- **延迟 1.36s vs 24.94s** (快 18 倍, 海量实用: 1000 条 23min vs 6.9h)
- **0 错误** (ornith 4 错误, 更稳定)
- **成本 $0.000054/条** (1000 条 $0.054, 极便宜)
- 本地 ornith 准确够但**延迟撑不住海量** → 效率撑不住 = 降级合理

**本地备用**: ornith-1.0-9b (80%, 离线/网络断时 fallback). gemma-4-e2b 单样本准确 (2B, 未全测, 候选).

**aetherforge 集成** (S2): 分诊调用走 aetherforge gateway → DeepSeekProvider, 用 tracker.record() 逐条记账 (模型/token/成本).

## G1' 全量 Benchmark (2026-07-30, 走 omlx 网关 :4000, 不直连 LMStudio/Ollama)

**G5 已落地**: 分诊调用统一走 LiteLLM 网关 :4000 (OpenAI 兼容), 云端 DeepSeek 已挂到同一网关.

| 排名 | 模型 | 准确率 | 延迟avg | 延迟p95 | 错误 | 网关别名 | 判据 |
|------|------|--------|---------|---------|------|----------|------|
| 🥇 | **mid-local (Qwen3.6-27B)** | **20/20 (100%)** | **1.38s** | 1.73s | **0** | mid-local (LMStudio JIT) | ✅ PASS |
| 🥈 | **mini-9b (qwen3.5:9b)** | **20/20 (100%)** | **1.46s** | 2.66s | **0** | mini-9b (mac-mini Ollama) | ✅ PASS |
| 🥉 | **coder-fast (Qwen3.6-35B)** | **20/20 (100%)** | **1.63s** | 12.12s | **0** | coder-fast (MBP omlx) | ✅ PASS |
| 4 | **deepseek-chat (云端)** | **19/20 (95%)** | **0.79s** | 1.04s | **0** | deepseek-chat | ✅ PASS |
| ❌ | fast-local (Qwen3.5-9B) | 0/20 (0%) | 3.42s | - | 20 | fast-local (JIT) | ❌ thinking |
| ❌ | coder-precise (Qwopus3.6-27B) | 0/20 (0%) | 16.33s | - | 20 | coder-precise (JIT) | ❌ 超时 |

**排除 (G1' 红线)**: reasoner / mythos / ornith — 推理模型不参与分诊评测.

**修复记录**:
1. mac-mini qwen3.5:9b chat template `{{ .Prompt }}` → ChatML 格式
2. thinking 模式: `drop_params: false` + `extra_body.reasoning_effort=none` 绕过 LiteLLM 验证
3. gemma4:e4b template → `<start_of_turn>`, 但 2B 太小 (25%)
4. DeepSeek API key 硬编码 (环境变量传递问题)
5. SearXNG 端口 8080→8880 (避免与 omlx 冲突)
6. coder-precise/fast-local thinking 问题: LMStudio JIT 不支持 `reasoning_effort` 透传

**结论**:
- **分诊主力**: mid-local (Qwen3.6-27B) — 100%, 1.38s, LMStudio JIT 按需加载
- **常驻备选**: mini-9b (mac-mini) — 100%, 1.46s, 零冷启
- **云端兜底**: deepseek-chat — 95%, 0.79s, 最快
- **编码模型**: coder-fast — 100%, 1.63s, 但 p95 波动大 (12s)

**G5 落地状态**: ✅ 网关 :4000 — 27 别名 + 3 通配, 云端 DeepSeek 已挂入, 统一记账到位.

**网关别名清单**:
- 常驻: coder, coder-fast, embed, deepseek-v4-flash (MBP omlx)
- JIT: mid-local, fast-local, coder-precise (LMStudio 按需)
- mac-mini: mini-9b, mini-chat, rerank, embed-bge, ocr, vision-lite
- 云端: deepseek-chat, deepseek-reasoner

## 治理线冻结声明
本文档是**功能线选型记录**, 非治理标准/门禁. 治理线本月冻结 (func-01 §红线).
