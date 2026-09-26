"""aetherforge 包内调用门面的唯一连接点(地址 + 密钥)。

与 kairon `kos.llm_gateway`、主仓 bin 脚本同一解析顺序:
  地址  LLM_GATEWAY_URL → AETHERFORGE_URL → OMLX_URL → http://127.0.0.1:4000
  密钥  LLM_GATEWAY_KEY → AETHERFORGE_API_KEY → OMLX_API_KEY → Keychain(aetherforge-gateway)

此前 triage / mcp_server / cli benchmark 写死 :9000(omlx-autostart, 已下线)与假 key
"sk-omlx-admin", sensitive_router 写死 mbp 旧 tailnet IP 与已下线的 8183/8188 端口。
"""

from __future__ import annotations

import functools
import os
import subprocess


def gateway_url() -> str:
    for name in ("LLM_GATEWAY_URL", "AETHERFORGE_URL", "OMLX_URL"):
        if os.environ.get(name):
            return os.environ[name].rstrip("/")
    return "http://127.0.0.1:4000"


def chat_url() -> str:
    return f"{gateway_url()}/v1/chat/completions"


@functools.lru_cache(maxsize=1)
def _keychain_key() -> str:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "aetherforge-gateway", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def gateway_key() -> str:
    for name in ("LLM_GATEWAY_KEY", "AETHERFORGE_API_KEY", "OMLX_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    return _keychain_key()
