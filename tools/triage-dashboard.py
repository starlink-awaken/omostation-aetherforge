#!/usr/bin/env python3
"""分诊结果可视化 — 基于 tracker 数据构建 dashboard.

用法:
  python3 tools/triage-dashboard.py --log /tmp/triage-log.jsonl
  python3 tools/triage-dashboard.py --log /tmp/triage-log.jsonl --output dashboard.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_log(path: Path) -> list[dict]:
    """加载 JSONL 日志."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze(records: list[dict]) -> dict:
    """分析记录."""
    if not records:
        return {"total": 0}

    by_model = defaultdict(lambda: {"count": 0, "cost": 0.0, "latency_sum": 0.0, "errors": 0})
    by_verdict = defaultdict(int)
    total_cost = 0.0
    total_latency = 0.0
    errors = 0

    for r in records:
        model = r.get("model", "unknown")
        verdict = r.get("verdict", "unknown")

        by_model[model]["count"] += 1
        by_model[model]["cost"] += r.get("cost", 0)
        by_model[model]["latency_sum"] += r.get("lat", 0)
        if r.get("err"):
            by_model[model]["errors"] += 1
            errors += 1

        by_verdict[verdict] += 1
        total_cost += r.get("cost", 0)
        total_latency += r.get("lat", 0)

    total = len(records)

    return {
        "total": total,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency": round(total_latency / total, 3) if total else 0,
        "errors": errors,
        "by_verdict": dict(by_verdict),
        "by_model": {
            m: {
                "count": s["count"],
                "cost_usd": round(s["cost"], 6),
                "avg_latency": round(s["latency_sum"] / s["count"], 3) if s["count"] else 0,
                "errors": s["errors"],
            }
            for m, s in by_model.items()
        },
    }


def generate_html(stats: dict, records: list[dict]) -> str:
    """生成 HTML dashboard."""
    verdicts = stats.get("by_verdict", {})
    models = stats.get("by_model", {})

    # 颜色映射
    colors = {"丢弃": "#ef4444", "沉淀": "#3b82f6", "提醒": "#f59e0b", "错误": "#6b7280"}

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>分诊 Dashboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
.metric .value {{ font-size: 2em; font-weight: bold; }}
.metric .label {{ color: #666; font-size: 0.9em; }}
.bar {{ height: 30px; border-radius: 4px; margin: 5px 0; display: flex; align-items: center; padding: 0 10px; color: white; font-size: 0.9em; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>分诊 Dashboard</h1>

<div class="card">
  <div class="metric">
    <div class="value">{stats['total']}</div>
    <div class="label">总处理量</div>
  </div>
  <div class="metric">
    <div class="value">{stats['avg_latency']:.2f}s</div>
    <div class="label">平均延迟</div>
  </div>
  <div class="metric">
    <div class="value">${stats['total_cost_usd']:.4f}</div>
    <div class="label">总成本</div>
  </div>
  <div class="metric">
    <div class="value">{stats['errors']}</div>
    <div class="label">错误数</div>
  </div>
</div>

<div class="card">
  <h2>分类分布</h2>
"""

    for verdict, count in sorted(verdicts.items(), key=lambda x: -x[1]):
        pct = count / stats['total'] * 100 if stats['total'] else 0
        color = colors.get(verdict, "#6b7280")
        html += f'  <div class="bar" style="width: {pct}%; background: {color};">{verdict}: {count} ({pct:.0f}%)</div>\n'

    html += """</div>

<div class="card">
  <h2>模型统计</h2>
  <table>
    <tr><th>模型</th><th>调用次数</th><th>平均延迟</th><th>成本</th><th>错误</th></tr>
"""

    for model, data in sorted(models.items(), key=lambda x: -x[1]["count"]):
        html += f'    <tr><td>{model}</td><td>{data["count"]}</td><td>{data["avg_latency"]:.3f}s</td><td>${data["cost_usd"]:.6f}</td><td>{data["errors"]}</td></tr>\n'

    html += """  </table>
</div>

<div class="card">
  <h2>最近记录</h2>
  <table>
    <tr><th>时间</th><th>判定</th><th>模型</th><th>延迟</th></tr>
"""

    for r in records[-20:]:
        ts = r.get("ts", "")[:19]
        html += f'    <tr><td>{ts}</td><td>{r.get("verdict", "")}</td><td>{r.get("model", "")}</td><td>{r.get("lat", 0):.3f}s</td></tr>\n'

    html += """  </table>
</div>

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="分诊 Dashboard")
    parser.add_argument("--log", required=True, help="JSONL 日志路径")
    parser.add_argument("--output", help="输出 HTML 路径")
    args = parser.parse_args()

    records = load_log(Path(args.log))
    stats = analyze(records)

    if args.output:
        html = generate_html(stats, records)
        Path(args.output).write_text(html)
        print(f"Dashboard 已生成: {args.output}")
    else:
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
