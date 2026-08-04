"""Triage HTTP server — 独立分诊 REST API.

端点:
- POST /triage — 单条分诊
- POST /triage/batch — 批量分诊
- POST /triage/consensus — 共识分诊
- GET /health — 健康检查
- GET /status — 状态查询

端口: 由 TRIAGE_PORT 环境变量注入 (默认 8095, 注册于 protocols/port-registry.yaml).

启动:
  python -m aetherforge.triage.server --port 8095
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# 添加 src 到路径
_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

from aetherforge.triage.monitor import TriageMonitor
from aetherforge.triage.router import TriageRouter
from aetherforge.triage.tracker import TriageTracker


class TriageHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器."""

    router: TriageRouter
    tracker: TriageTracker
    monitor: TriageMonitor

    def do_POST(self):
        if self.path == "/triage":
            self._handle_single()
        elif self.path == "/triage/batch":
            self._handle_batch()
        elif self.path == "/triage/consensus":
            self._handle_consensus()
        else:
            self._respond(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/status":
            self._handle_status()
        else:
            self._respond(404, {"error": "Not found"})

    def _handle_single(self):
        """单条分诊."""
        body = self._read_body()
        if not body:
            return

        text = body.get("text", "")
        if not text:
            self._respond(400, {"error": "Missing 'text' field"})
            return

        result = self.router.triage_one(text)
        self._respond(
            200,
            {
                "verdict": result.verdict,
                "model": result.model,
                "latency": round(result.latency, 3),
                "error": result.error,
            },
        )

    def _handle_batch(self):
        """批量分诊."""
        body = self._read_body()
        if not body:
            return

        texts = body.get("texts", [])
        if not texts:
            self._respond(400, {"error": "Missing 'texts' field"})
            return

        results = self.router.triage_batch(texts)
        self._respond(
            200,
            {
                "results": [
                    {
                        "verdict": r.verdict,
                        "model": r.model,
                        "latency": round(r.latency, 3),
                        "error": r.error,
                    }
                    for r in results
                ]
            },
        )

    def _handle_consensus(self):
        """共识分诊."""
        body = self._read_body()
        if not body:
            return

        text = body.get("text", "")
        if not text:
            self._respond(400, {"error": "Missing 'text' field"})
            return

        result = self.router.consensus_triage(text)
        self._respond(
            200,
            {
                "verdict": result.verdict,
                "votes": result.votes,
                "agreement": result.agreement,
                "status": result.status,
                "latency": round(result.latency, 3),
                "cost_usd": result.cost_usd,
                "models": {d.model: {"verdict": d.verdict, "latency": round(d.latency, 3)} for d in result.details},
            },
        )

    def _handle_status(self):
        """状态查询."""
        trend = self.monitor.get_trend()
        self._respond(
            200,
            {
                "tracker": self.tracker.summary(),
                "monitor": trend,
            },
        )

    def _read_body(self) -> dict[str, Any] | None:
        """读取请求体."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._respond(400, {"error": "Empty body"})
                return None
            body = self.rfile.read(content_length)
            return json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return None

    def _respond(self, code: int, data: dict):
        """发送响应."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        """禁用默认日志."""
        pass


def create_server(
    port: int | None = None,
    gateway_url: str = "http://100.96.126.35:4000/v1/chat/completions",
    api_key: str = "sk-omlx-admin",
) -> HTTPServer:
    """创建 HTTP 服务器."""
    if port is None:
        port = int(os.environ.get("TRIAGE_PORT", "8095"))
    tracker = TriageTracker()
    router = TriageRouter(tracker=tracker)
    monitor = TriageMonitor(router)

    TriageHandler.router = router
    TriageHandler.tracker = tracker
    TriageHandler.monitor = monitor

    return HTTPServer(("0.0.0.0", port), TriageHandler)  # noqa: S104  (server binds all interfaces by design)


def main():
    parser = argparse.ArgumentParser(description="分诊 HTTP 服务")
    parser.add_argument("--port", type=int, default=None, help="监听端口 (默认: TRIAGE_PORT 环境变量或 8095)")
    parser.add_argument("--gateway", default="http://100.96.126.35:4000/v1/chat/completions")
    parser.add_argument("--key", default="sk-omlx-admin")
    args = parser.parse_args()

    server = create_server(args.port, args.gateway, args.key)
    print(f"分诊服务启动: http://0.0.0.0:{args.port}")
    print("  POST /triage — 单条分诊")
    print("  POST /triage/batch — 批量分诊")
    print("  POST /triage/consensus — 共识分诊")
    print("  GET /health — 健康检查")
    print("  GET /status — 状态查询")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
