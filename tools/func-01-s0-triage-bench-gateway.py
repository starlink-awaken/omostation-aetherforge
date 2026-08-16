#!/usr/bin/env python3
"""FUNC-01 S0: 分诊选型实测 — 走 omlx 统一网关 (LiteLLM :4000).

与 func-01-s0-triage-bench.py 同 20 样本, 但所有调用走网关:
  - 本地模型: fast / mini-9b / deepseek-v4-flash 等别名
  - 云端 DeepSeek: deepseek-chat (同一网关, 统一记账)
  - 通配: macmini-ollama/qwen3.5:9b

判据: 准确率 ≥80% 且延迟 <2s.
红线: reasoner/mythos/ornith 不参与分诊评测.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

GATEWAY_URL = "http://127.0.0.1:9000/v1/chat/completions"
GATEWAY_KEY = os.environ.get("OMLX_GATEWAY_KEY", "sk-omlx-admin")

# 20 条真实分诊样本 (覆盖三档: 丢弃6/沉淀8/提醒6)
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

# G1' 红线: reasoning 模型不参与分诊评测
EXCLUDED_KEYWORDS = ["reasoner", "mythos", "ornith"]


def triage_gateway(model: str, text: str) -> tuple[str, float]:
    """通过 omlx 网关分诊, 返回 (判定, 延迟秒)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
        "max_tokens": 100,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        GATEWAY_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GATEWAY_KEY}",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — internal gateway
            d = json.loads(resp.read())
        latency = time.time() - t0
        content = d["choices"][0]["message"]["content"].strip()
        for verdict in ("丢弃", "沉淀", "提醒"):
            if verdict in content:
                return verdict, latency
        return f"未知({content[:20]})", latency
    except Exception as exc:
        return f"错误({str(exc)[:30]})", time.time() - t0


def list_gateway_models() -> list[str]:
    """列出网关所有模型别名."""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9000/v1/models",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — internal gateway
            return [m["id"] for m in json.loads(resp.read()).get("data", [])]
    except Exception:
        return []


def is_excluded(model: str) -> bool:
    """检查模型是否在 G1' 红线排除列表."""
    return any(kw in model.lower() for kw in EXCLUDED_KEYWORDS)


def main():
    # G1' 目标模型 (必须走网关, 不直连)
    target_models = [
        "fast",                          # Y7000P LMStudio qwen3.5-9b
        "mini-9b",                       # mac-mini Ollama qwen3.5:9b
        "macmini-ollama/qwen3.5:9b",    # 常驻零冷启
        "deepseek-chat",                 # 云端 DeepSeek (对比基线)
        "deepseek-v4-flash",             # MBP 本地 DeepSeek flash
    ]

    # 过滤 reasoning 模型
    target_models = [m for m in target_models if not is_excluded(m)]

    all_models = list_gateway_models()
    print(f"网关在线模型: {len(all_models)}")
    print(f"G1' 测试目标: {target_models}")
    print(f"排除(reasoning): {[m for m in all_models if is_excluded(m)]}")
    print()

    # 检查目标是否在网关
    for m in target_models:
        base = m.split("/")[0] if "/" in m else m
        available = base in all_models or m in all_models
        status = "✅ 网关可达" if available else "❌ 网关无此别名"
        print(f"  {m}: {status}")

    print(f"\n{'='*60}")
    print(f"分诊实测: {len(SAMPLES)} 样本, 三档 (丢弃{sum(1 for s in SAMPLES if s['expected']=='丢弃')}/"
          f"沉淀{sum(1 for s in SAMPLES if s['expected']=='沉淀')}/"
          f"提醒{sum(1 for s in SAMPLES if s['expected']=='提醒')})")
    print("判据: 准确率 ≥80% 且延迟 <2s")
    print(f"{'='*60}\n")

    results = []
    for model in target_models:
        print(f"测试 {model} ...", end=" ", flush=True)
        correct = 0
        latencies = []
        errors = 0
        error_details = []
        for i, s in enumerate(SAMPLES):
            verdict, lat = triage_gateway(model, s["text"])
            latencies.append(lat)
            if "错误" in verdict or "未知" in verdict:
                errors += 1
                error_details.append(f"  样本{i+1}: {verdict}")
            elif verdict == s["expected"]:
                correct += 1
        acc = correct / len(SAMPLES)
        avg_lat = sum(latencies) / len(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        results.append({
            "model": model,
            "accuracy": acc,
            "avg_latency": avg_lat,
            "p95_latency": p95_lat,
            "errors": errors,
        })
        print(f"准确率 {correct}/{len(SAMPLES)} ({acc:.0%}), 延迟 avg={avg_lat:.2f}s p95={p95_lat:.2f}s, 错误 {errors}")
        if error_details:
            for ed in error_details[:3]:
                print(ed)

    print(f"\n{'='*60}")
    print("选型结果 (G1' 判据: 准确率 ≥80% 且延迟 <2s):")
    print(f"{'模型':<35} {'准确率':>8} {'延迟avg':>10} {'延迟p95':>10} {'状态':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in sorted(results, key=lambda x: (-x["accuracy"], x["avg_latency"])):
        acc_ok = r["accuracy"] >= 0.8
        lat_ok = r["avg_latency"] < 2.0
        status = "✅ PASS" if (acc_ok and lat_ok and r["errors"] == 0) else "❌ FAIL"
        if not acc_ok:
            status = "❌ 准确率不足"
        elif not lat_ok:
            status = "❌ 延迟过高"
        elif r["errors"] > 0:
            status = f"⚠️ {r['errors']}错误"
        print(f"{r['model']:<35} {r['accuracy']:>7.0%} {r['avg_latency']:>9.2f}s {r['p95_latency']:>9.2f}s {status:>8}")

    # 推荐
    passed = [r for r in results if r["accuracy"] >= 0.8 and r["avg_latency"] < 2.0 and r["errors"] == 0]
    print()
    if passed:
        best = min(passed, key=lambda x: x["avg_latency"])
        print(f"✅ G1' 推荐: {best['model']} (准确率 {best['accuracy']:.0%}, 延迟 {best['avg_latency']:.2f}s)")
    else:
        print("🔴 G1' 无本地模型同时满足准确率≥80%且延迟<2s → 降级 DeepSeek 云端")
        ds = [r for r in results if "deepseek-chat" in r["model"]]
        if ds:
            print(f"   DeepSeek 云端: {ds[0]['accuracy']:.0%} 准确, {ds[0]['avg_latency']:.2f}s 延迟")


if __name__ == "__main__":
    main()
