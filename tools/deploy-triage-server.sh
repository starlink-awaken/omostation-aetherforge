#!/bin/bash
# deploy-triage-server.sh — 部署分诊 HTTP 服务
#
# 用法:
#   bash tools/deploy-triage-server.sh           # 前台运行
#   bash tools/deploy-triage-server.sh --daemon   # 后台运行
#   bash tools/deploy-triage-server.sh --stop      # 停止服务

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="/tmp/triage-server.pid"
LOG_FILE="/tmp/triage-server.log"
PORT=${TRIAGE_PORT:-8095}

case "${1:-}" in
    --stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                rm -f "$PID_FILE"
                echo "✅ 分诊服务已停止 (PID: $PID)"
            else
                rm -f "$PID_FILE"
                echo "⚠️ PID 文件存在但进程不在"
            fi
        else
            echo "⚠️ 分诊服务未运行"
        fi
        exit 0
        ;;
    --daemon)
        # 先停止已有实例
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            kill "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
        fi

        echo "🚀 启动分诊服务 (后台模式, 端口 $PORT)..."
        cd "$PROJECT_DIR"
        PYTHONPATH=src nohup python3 -m aetherforge.triage.server --port "$PORT" > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 2

        if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "✅ 分诊服务已启动"
            echo "   PID: $(cat "$PID_FILE")"
            echo "   端口: $PORT"
            echo "   日志: $LOG_FILE"
            echo ""
            echo "测试:"
            echo "   curl -X POST http://localhost:$PORT/triage -d '{\"text\":\"test\"}'"
            echo "   curl -X POST http://localhost:$PORT/triage/consensus -d '{\"text\":\"test\"}'"
            echo "   curl http://localhost:$PORT/health"
        else
            echo "❌ 启动失败，查看日志: $LOG_FILE"
            exit 1
        fi
        ;;
    *)
        echo "🚀 启动分诊服务 (前台模式, 端口 $PORT)..."
        cd "$PROJECT_DIR"
        PYTHONPATH=src exec python3 -m aetherforge.triage.server --port "$PORT"
        ;;
esac
