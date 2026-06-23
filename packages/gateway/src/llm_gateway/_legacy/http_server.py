"""HTTP server for llm-gateway — standalone testable module.

Provides a lightweight HTTP server (stdlib only) that exposes
:func:`detect_backends` and :func:`create_provider` as REST endpoints.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .detection import detect_backends
from .provider import LLMRequest


class LLMGatewayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for LLM generation."""

    def do_POST(self) -> None:
        if self.path != "/v1/generate":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"Invalid JSON"}')
            return

        prompt: str = body.get("prompt", "")
        model: str | None = body.get("model")
        budget_usd: float | None = body.get("budget_usd")
        task_id: str | None = body.get("task_id")

        providers = detect_backends()
        if not providers:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"No LLM backends available"}')
            return

        provider = providers[0]
        model_id = model or provider.default_model

        # ── Phase 4: Pre-call Budget Check ──
        from .budget import BudgetExhausted, check_budget_limit
        try:
            # Estimate tokens
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                input_tokens = len(enc.encode(prompt))
            except ImportError:
                input_tokens = (len(prompt) + 3) // 4

            check_budget_limit(
                model_id=model_id,
                input_tokens=input_tokens,
                max_output_tokens=body.get("max_output_tokens", 512),
                task_id=task_id,
                local_budget_limit=budget_usd
            )
        except BudgetExhausted as e:
            self.send_response(402)  # Payment Required
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"BUDGET_EXHAUSTED: {str(e)}", "status": "blocked"}).encode())
            return
        except Exception as e:
            print(f"[llm-gateway] Budget check failed: {e}")

        if not provider.is_available():
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"Provider not available"}')
            return

        req = LLMRequest(prompt=prompt, model=model or None)
        try:
            resp = provider.complete(req)
            result: dict[str, Any] = {
                "content": resp.content,
                "provider": resp.provider,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            }
            for key in ("latency_ms", "tokens_per_second", "node_id", "node_label", "route_type"):
                if key in resp.metadata:
                    result[key] = resp.metadata[key]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            providers = detect_backends()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            result = {
                "status": "ok",
                "providers": len(providers),
            }
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[llm-gateway] {args[0]} {args[1]} {args[2]}\n")


def serve(port: int = 9290) -> None:
    """Start the llm-gateway HTTP server.

    Endpoints:
        POST /v1/generate  — generate LLM response
        GET  /v1/health    — health check
    """
    server = HTTPServer(("0.0.0.0", port), LLMGatewayHandler)  # noqa: S104
    print(f"llm-gateway HTTP server: http://localhost:{port}")
    print("  POST /v1/generate  — generate LLM response")
    print("  GET  /v1/health     — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
