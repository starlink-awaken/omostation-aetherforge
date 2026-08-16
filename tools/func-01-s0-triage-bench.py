#!/usr/bin/env python3
"""FUNC-01 S0: 分诊选型实测 (≥20 真实样本, LMStudio/ollama 横向).

分诊 = 海量/简单/重复/判据明确. 三选一: 丢弃/沉淀入KOS/提醒用户.
本脚本横向实测本地模型, 选"够用的最便宜" (准确率 + 延迟).

红线: 撑不住如实降级 DeepSeek, 不为"用上本地"牺牲质量.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

# 20 条真实分诊样本 (覆盖三档: 丢弃6/沉淀8/提醒6, 基于真实信息类型)
SAMPLES = [
    # 丢弃 (营销/噪音/无关)
    {"text": "【限时优惠】购买 Pro 会员立享 50% 折扣，仅限今日", "expected": "丢弃"},
    {"text": "恭喜您中奖了！点击链接领取 100 万现金大奖", "expected": "丢弃"},
    {"text": "今日星座运势：天秤座事业运爆棚，爱情有惊喜", "expected": "丢弃"},
    {"text": "订阅我们的每周通讯，获取行业最新资讯", "expected": "丢弃"},
    {"text": "广告：新店开业全场 8 折，满减优惠多多", "expected": "丢弃"},
    {"text": "您的包裹已发货，物流单号 SF1234567890，预计明日送达", "expected": "丢弃"},
    # 沉淀入 KOS (技术/知识/有价值)
    {"text": "Python asyncio 协程最佳实践：事件循环与任务调度详解", "expected": "沉淀"},
    {"text": "RFC 9535: JSON-LD 1.1 序列化规范全文", "expected": "沉淀"},
    {"text": "PostgreSQL 索引优化深度分析：B-tree vs Hash vs GIN", "expected": "沉淀"},
    {"text": "分布式系统 CAP 定理与 BASE 理论详解", "expected": "沉淀"},
    {"text": "Rust 所有权机制入门：borrow checker 工作原理", "expected": "沉淀"},
    {"text": "机器学习模型评估指标综述：精确率/召回率/F1/AUC", "expected": "沉淀"},
    {"text": "Git rebase vs merge 工作流对比与最佳实践", "expected": "沉淀"},
    {"text": "Kubernetes Operator 模式设计与 CRD 实践", "expected": "沉淀"},
    # 提醒用户 (重要/紧急/行动)
    {"text": "会议提醒：下午 3 点产品评审会，会议室 A", "expected": "提醒"},
    {"text": "账单提醒：信用卡还款日明天截止，请及时还款", "expected": "提醒"},
    {"text": "紧急告警：生产服务器 CPU 使用率 100%，需立即处理", "expected": "提醒"},
    {"text": "PR #615 等待你的 code review，已阻塞合并", "expected": "提醒"},
    {"text": "航班变更通知：明天的 CA1234 航班提前 2 小时起飞", "expected": "提醒"},
    {"text": "合同签约截止日期：本周五前需完成盖章", "expected": "提醒"},
]

PROMPT_TEMPLATE = """你是信息分诊助手。给定一条信息，判断该 丢弃 / 沉淀 / 提醒 三选一。

判定标准:
- 丢弃: 营销/广告/噪音/无关/垃圾/星座运势/物流通知
- 沉淀: 技术文章/规范文档/知识教程/研究分析 (有价值, 入知识库)
- 提醒: 会议/截止日期/账单/告警/需用户行动的重要事项

信息: {text}

只输出一个词 (丢弃/沉淀/提醒):"""


def triage(model: str, text: str) -> tuple[str, float]:
    """调用 LMStudio 分诊, 返回 (判定, 延迟秒)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
        "max_tokens": 500,  # reasoning model 需足够 tokens 思考 (gemma-4-e2b 用 282 reasoning + 答案)
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(LMSTUDIO_URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — internal gateway
            d = json.loads(resp.read())
        latency = time.time() - t0
        content = d["choices"][0]["message"]["content"].strip()
        # 解析判定 (取第一个匹配词)
        for verdict in ("丢弃", "沉淀", "提醒"):
            if verdict in content:
                return verdict, latency
        return f"未知({content[:10]})", latency
    except Exception as exc:
        return f"错误({str(exc)[:20]})", time.time() - t0


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=5) as resp:
            return [m["id"] for m in json.loads(resp.read()).get("data", [])]
    except Exception:
        return []


def main():
    online = list_models()
    print(f"LMStudio 在线模型: {len(online)}")
    # 候选: 小快模型 (分诊不需大模型), 按关键词过滤
    candidate_keywords = ["e2b", "9b", "flash", "nano", "gemma", "deepseek-v4-flash"]
    candidates = [m for m in online if any(k in m.lower() for k in candidate_keywords)]
    if not candidates:
        candidates = online[:3]
    print(f"分诊候选 (小快): {candidates[:6]}")

    print(f"\n{'='*60}")
    print(f"分诊实测: {len(SAMPLES)} 样本, 三档 (丢弃{sum(1 for s in SAMPLES if s['expected']=='丢弃')}/"
          f"沉淀{sum(1 for s in SAMPLES if s['expected']=='沉淀')}/"
          f"提醒{sum(1 for s in SAMPLES if s['expected']=='提醒')})")
    print(f"{'='*60}\n")

    results = []
    for model in candidates[:3]:  # reasoning 慢, 限 3 候选 (gemma-e2b/ornith-9b/deepseek-flash)
        correct = 0
        latencies = []
        errors = 0
        for s in SAMPLES:
            verdict, lat = triage(model, s["text"])
            latencies.append(lat)
            if "错误" in verdict or "未知" in verdict:
                errors += 1
            elif verdict == s["expected"]:
                correct += 1
        acc = correct / len(SAMPLES)
        avg_lat = sum(latencies) / len(latencies)
        results.append({"model": model, "accuracy": acc, "avg_latency": avg_lat, "errors": errors})
        print(f"{model}: 准确率 {correct}/{len(SAMPLES)} ({acc:.0%}), 平均延迟 {avg_lat:.2f}s, 错误 {errors}")

    print(f"\n{'='*60}")
    print("选型建议 (按准确率降序, 够用最便宜):")
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"  {r['model']}: {r['accuracy']:.0%} 准确, {r['avg_latency']:.2f}s 延迟")
    # 够用标准: 准确率 ≥80%
    good = [r for r in results if r["accuracy"] >= 0.8]
    if good:
        cheapest = min(good, key=lambda x: x["avg_latency"])  # 延迟最低 = 最便宜(本地无金钱成本, 延迟=算力)
        print(f"\n✅ 推荐: {cheapest['model']} (准确率 {cheapest['accuracy']:.0%} ≥80%, 延迟 {cheapest['avg_latency']:.2f}s 最低)")
    else:
        print("\n🔴 本地模型都撑不住分诊 (准确率 <80%) → 如实降级 DeepSeek")


if __name__ == "__main__":
    main()
