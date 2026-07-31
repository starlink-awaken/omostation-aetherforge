#!/bin/bash
# gen-dashboard.sh — 生成分诊 dashboard HTML
#
# 用法:
#   bash tools/gen-dashboard.sh [日志路径] [输出路径]

LOG_PATH="${1:-/tmp/triage-log.jsonl}"
OUTPUT="${2:-/tmp/triage-dashboard.html}"

if [ ! -f "$LOG_PATH" ]; then
    echo "⚠️ 日志文件不存在: $LOG_PATH"
    echo "   先运行分诊生成日志:"
    echo "   python3 tools/triage-gateway.py --consensus 'test' --log $LOG_PATH"
    exit 1
fi

python3 tools/triage-dashboard.py --log "$LOG_PATH" --output "$OUTPUT"
echo "✅ Dashboard 已生成: $OUTPUT"
echo "   用浏览器打开查看"
