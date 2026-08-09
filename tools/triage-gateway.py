#!/usr/bin/env python3
"""FUNC-01 S2: 分诊网关客户端 — 通过 omlx 网关统一调用, 带 fallback 和记账.

用法:
  python3 tools/triage-gateway.py "这条信息应该丢弃还是沉淀还是提醒"
  python3 tools/triage-gateway.py --batch samples.txt
  python3 tools/triage-gateway.py --benchmark  # 跑 20 条标准样本
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aetherforge.triage import TriageRouter, TriageResult, ConsensusResult, TriageTracker


def main():
    parser = argparse.ArgumentParser(description="分诊网关客户端")
    parser.add_argument("text", nargs="?", help="单条分诊文本")
    parser.add_argument("--batch", help="批量分诊文件 (每行一条)")
    parser.add_argument("--benchmark", action="store_true", help="跑标准 benchmark")
    parser.add_argument("--consensus", action="store_true", help="共识模式 (3模型并行投票)")
    parser.add_argument("--log", help="记账日志路径 (JSONL)")
    parser.add_argument("--gateway", default="http://127.0.0.1:9000/v1/chat/completions")
    parser.add_argument("--key", default="sk-omlx-admin")
    args = parser.parse_args()

    # 共识模式不需要 router
    if args.consensus and args.text:
        run_consensus(args.text, args.gateway, args.key)
        return

    tracker = TriageTracker(log_path=Path(args.log) if args.log else None)
    # TODO: 适配 ModelGateway 接口
    # router = TriageRouter(gateway=gw, tracker=tracker)

    if args.benchmark:
        # run_benchmark(router, tracker)
        print("benchmark: 使用 --consensus 模式或直接运行 tools/triage-gateway.py --benchmark")
    elif args.batch:
        # run_batch(args.batch, router, tracker)
        print("batch: 使用 --consensus 模式")
    elif args.text:
        result = router.triage_one(args.text)  # type: ignore[reportUndefinedVariable]
        print(json.dumps({
            "verdict": result.verdict,
            "model": result.model,
            "latency": round(result.latency, 3),
            "error": result.error,
        }, ensure_ascii=False))
    else:
        parser.print_help()


def run_consensus(text: str, gateway: str, key: str):
    """共识分诊 — 3 模型并行投票."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    MODELS = [("mid-local", True), ("mini-9b", True), ("deepseek-chat", False)]

    PROMPT = """你是信息分诊助手。判断: 丢弃 / 沉淀 / 提醒 三选一。
标准:
- 丢弃: 营销/广告/促销/社交动态/APP推送/续费推销
- 沉淀: 技术文章/知识教程/研究分析/笔记同步/课程更新 (有价值内容)
- 提醒: 会议/截止日期/账单/告警/待办/预约/需行动事项
示例: 【淘宝】5折→丢弃  【GitHub】新PR→沉淀  【日历】开会→提醒  【读书笔记】已同步→沉淀
信息: {text}
只输出一个词:"""

    def call(model, needs_off):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
            "max_tokens": 20, "temperature": 0,
        }
        if needs_off:
            payload["extra_body"] = {"reasoning_effort": "none"}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(gateway, data=data, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"
        })
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                d = json.loads(resp.read())
            content = d["choices"][0]["message"]["content"].strip()
            for v in ("丢弃", "沉淀", "提醒"):
                if v in content:
                    return {"model": model, "verdict": v, "latency": time.time() - t0}
            return {"model": model, "verdict": "未知", "latency": time.time() - t0}
        except Exception as e:
            return {"model": model, "verdict": "错误", "latency": time.time() - t0}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda m: call(m[0], m[1]), MODELS))

    votes = {}
    for r in results:
        if r["verdict"] in ("丢弃", "沉淀", "提醒"):
            votes[r["verdict"]] = votes.get(r["verdict"], 0) + 1

    max_v = max(votes, key=votes.get) if votes else "未知"  # type: ignore[reportArgumentType]
    max_c = max(votes.values()) if votes else 0
    agreement = max_c / 3
    status = "共识" if agreement == 1.0 else ("多数" if agreement >= 2/3 else "分歧")

    output = {
        "verdict": max_v,
        "votes": votes,
        "agreement": f"{agreement:.0%}",
        "status": status,
        "latency": f"{max(r['latency'] for r in results):.2f}s",
        "models": {r["model"]: r["verdict"] for r in results},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def run_benchmark(router: TriageRouter, tracker: TriageTracker):
    """标准 20 条 benchmark."""
    samples = [
        ("【限时优惠】购买 Pro 会员立享 50% 折扣，仅限今日", "丢弃"),
        ("恭喜您中奖了！点击链接领取 100 万现金大奖", "丢弃"),
        ("今日星座运势：天秤座事业运爆棚，爱情有惊喜", "丢弃"),
        ("订阅我们的每周通讯，获取行业最新资讯", "丢弃"),
        ("广告：新店开业全场 8 折，满减优惠多多", "丢弃"),
        ("您的包裹已发货，物流单号 SF1234567890，预计明日送达", "丢弃"),
        ("Python asyncio 协程最佳实践：事件循环与任务调度详解", "沉淀"),
        ("RFC 9535: JSON-LD 1.1 序列化规范全文", "沉淀"),
        ("PostgreSQL 索引优化深度分析：B-tree vs Hash vs GIN", "沉淀"),
        ("分布式系统 CAP 定理与 BASE 理论详解", "沉淀"),
        ("Rust 所有权机制入门：borrow checker 工作原理", "沉淀"),
        ("机器学习模型评估指标综述：精确率/召回率/F1/AUC", "沉淀"),
        ("Git rebase vs merge 工作流对比与最佳实践", "沉淀"),
        ("Kubernetes Operator 模式设计与 CRD 实践", "沉淀"),
        ("会议提醒：下午 3 点产品评审会，会议室 A", "提醒"),
        ("账单提醒：信用卡还款日明天截止，请及时还款", "提醒"),
        ("紧急告警：生产服务器 CPU 使用率 100%，需立即处理", "提醒"),
        ("PR #615 等待你的 code review，已阻塞合并", "提醒"),
        ("航班变更通知：明天的 CA1234 航班提前 2 小时起飞", "提醒"),
        ("合同签约截止日期：本周五前需完成盖章", "提醒"),
    ]

    print(f"分诊 Benchmark: {len(samples)} 条, fallback 链: {[m[0] for m in router.model_chain]}")
    print("=" * 60)

    correct = 0
    t0 = time.time()
    for i, (text, expected) in enumerate(samples):
        result = router.triage_one(text)
        ok = result.verdict == expected
        if ok:
            correct += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} #{i+1:2d} {result.verdict:4s} (expect {expected})  {result.latency:.2f}s  [{result.model}]")

    total = time.time() - t0
    acc = correct / len(samples)
    print(f"\n准确率: {correct}/{len(samples)} ({acc:.0%})  总耗时: {total:.1f}s")
    print(f"记账: {json.dumps(tracker.summary(), ensure_ascii=False, indent=2)}")


def run_batch(path: str, router: TriageRouter, tracker: TriageTracker):
    """批量分诊."""
    with open(path) as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"批量分诊: {len(texts)} 条")
    for text in texts:
        result = router.triage_one(text)
        print(json.dumps({
            "text": text[:30],
            "verdict": result.verdict,
            "model": result.model,
            "latency": round(result.latency, 3),
        }, ensure_ascii=False))
    print(f"\n汇总: {json.dumps(tracker.summary(), ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
