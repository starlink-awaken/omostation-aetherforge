"""ModelGateway — 统一模型网关, 唯一入口.

架构职责 (单一职责, 不外包):
  1. K1 敏感流硬拦 (网关层, 不依赖消费者自觉)
  2. 模型选择 + fallback 链 (本地 → 云端)
  3. ensure_model (自动 load/unload + MemoryGuard)
  4. 推理 + 超时 + 重试 + strip_thinking (兜底)
  5. 记账 + 指标 (MetricsCollector)
  6. WarmPool (keep-last-used, 减少冷启动)
  7. HealthMonitor (自动健康检查)

设计原则:
  - KISS: 一个类管所有, 不搞微服务拆分
  - YAGNI: 只实现当前需要的 (不搞多租户/配额/灰度)
  - DRY: strip_thinking / sensitive_check 在网关层统一, 不散落消费者

消费方:
  - triage/router.py → gateway.generate()
  - sensitive_router.py → gateway.embed() + gateway.generate()
  - gateway/rpc.py → gateway.generate()
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .complexity import TaskComplexityScorer
from .metrics import MetricsCollector
from .omlxc_client import (
    OmlxcCatalogModel,
    OmlxcClient,
    OmlxcError,
    OmlxcErrorCode,
    OmlxcStreamChunk,
)
from .paths import M1_COMPUTE_ENGINE_DIR, M1_MODEL_DIR
from .registry import ModelRegistry
from .scheduler import ModelScheduler
from .ssot_loader import load_ssot_models
from .types import ChatOptions

_log = logging.getLogger(__name__)

# ============================================================
# Thinking 剥离 (SSOT: 全项目唯一源头)
# ============================================================
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """剥离模型输出的 thinking/reasoning 段, 只保留最终回答.

    部分 MLX 推理模型(如 Qwen3.6)的 chat template 硬编码了 <think> 标签,
    即使传 enable_thinking=False 也会在 content 中输出思考段.
    本函数在网关出口做最终兜底.

    这是全项目的 SSOT — sensitive_router.py 等消费方直接 import 复用.
    """
    if not text:
        return text
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_INVENTORY_DROP_CODE = "inventory_drop"


def _sanitize_inventory_warnings(warnings: object) -> list[dict[str, object]]:
    """Forward only inventory_drop and its four identity/count fields."""
    if not isinstance(warnings, (list, tuple)):
        return []
    out: list[dict[str, object]] = []
    for item in warnings:
        if not isinstance(item, Mapping):
            continue
        if item.get("code") != _INVENTORY_DROP_CODE:
            continue
        node_id = item.get("node_id")
        backend_id = item.get("backend_id")
        baseline = item.get("baseline")
        current = item.get("current")
        if not isinstance(node_id, str) or not node_id:
            continue
        if not isinstance(backend_id, str) or not backend_id:
            continue
        if not isinstance(baseline, int) or isinstance(baseline, bool) or baseline < 0:
            continue
        if not isinstance(current, int) or isinstance(current, bool) or current < 0:
            continue
        out.append({
            "code": _INVENTORY_DROP_CODE,
            "node_id": node_id,
            "backend_id": backend_id,
            "baseline": baseline,
            "current": current,
        })
    return out


# ============================================================
# K1: 敏感流判断 (SSOT: 全项目统一标准)
# ============================================================
# 完整敏感模式 = 网关原有 + sensitive_router 更全的模式
# 以 sensitive_router 的列表为基准 (覆盖 OA/邮箱/企业协作/内部 IP)
_SENSITIVE_PATTERNS = [
    # 工作 OA / 公文系统
    r"oa\.",
    r"office\.",
    r"erp\.",
    r"crm\.",
    r"hr\.",
    r"gov\.cn",
    r"gov\.com",
    r"政务",
    r"公文",
    # 邮件
    r"mail\.",
    r"email\.",
    r"imap\.",
    r"smtp\.",
    r"mail\.google\.com",
    r"outlook\.",
    r"office365\.",
    # 工作协作
    r"slack\.",
    r"teams\.",
    r"zoom\.",
    r"meet\.",
    r"feishu\.",
    r"lark\.",
    r"wecom\.",
    r"dingtalk\.",
    # 代码/文档 (企业内部)
    r"gitlab\.",
    r"github\.com/starlink",
    r"bitbucket\.",
    r"confluence\.",
    r"notion\.so",
    r"wiki\.",
    r"docs\.google\.com",  # Google 文档 (工作文档/行动卡)
    # 企业邮箱
    r"163\.com",
    r"126\.com",
    r"yeah\.net",  # 网易系
    r"qq\.com",
    r"vip\.qq\.com",  # QQ 邮箱
    r"sina\.com\.cn",
    r"sina\.cn",  # 新浪邮箱
    r"sohu\.com",  # 搜狐邮箱
    # 内网 IP (协议无关: http://10.x / 10.0.0.1 都匹配)
    r"(?:^|://)10\.\d+\.\d+\.\d+",
    r"(?:^|://)192\.168\.\d+\.\d+",
    r"(?:^|://)127\.\d+\.\d+\.\d+",
    r"(?:^|://)172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+",
    # OA 协同软件
    r"seeyon",
    r"致远",
    r"泛微",
    r"蓝凌",
    r"通达",
    # 财务/法务
    r"finance\.",
    r"legal\.",
    r"contract\.",
    r"invoice\.",
]

_SENSITIVE_KEYWORDS = re.compile(
    r"(通知|会议|纪要|请示|报告|批复|函|意见|方案|总结|汇报|批示|通报|决定|公告"
    r"|待办|截止|到期|逾期|审批|操作确认|Confirm access"
    r"|工资|薪资|合同|协议|发票|报销|财务|法务|人事|入职|离职"
    r"|密码|密钥|token|credential|secret|私密|机密|内部"
    r"|OA|ERP|CRM|HR|公文|政务)"
)


def is_sensitive(title: str, url: str) -> bool:
    """判信息敏感度. True = 敏感 (禁止送外部 API).

    这是全项目的 SSOT — sensitive_router 的 classify_sensitivity 内部也调用此函数.
    比 sensitive_router 的 classify_sensitivity 更严格 (无公开域名白名单, 默认敏感).
    """
    if not title and not url:
        return True  # 空 = 保守判敏感

    for pat in _SENSITIVE_PATTERNS:
        if re.search(pat, url) or re.search(pat, url.lower()):
            return True

    if title and _SENSITIVE_KEYWORDS.search(title):
        return True

    return False


# 内部兼容: _strip_thinking / _is_sensitive 是 strip_thinking / is_sensitive 的别名
_strip_thinking = strip_thinking
_is_sensitive = is_sensitive


# ============================================================
# 数据结构
# ============================================================
@dataclass
class GatewayRequest:
    """网关统一请求."""

    messages: list[dict[str, Any]]
    model: str = ""  # 优先模型 (bare name or full id)
    task: str = ""  # triage / chat / embed
    timeout: float = 30.0

    # OpenAI 兼容参数。此前 openai_proxy 从请求体读了 temperature/max_tokens
    # 却无处可传(GatewayRequest 无此字段), 结果被静默丢弃 —— 门面要接管
    # LiteLLM, 丢这两个参数是功能回退。None 表示不指定, 由下游取默认。
    temperature: float | None = None
    max_tokens: int | None = None
    # 下游扩展参数：oMLX App / LM Studio / Ollama 的关 thinking 参数均走这里。
    extra: dict[str, Any] = field(default_factory=dict)
    # local(默认): 只用本地三机算力；hybrid: 允许云端参与 fallback；cloud: 只用云端。
    routing_mode: str = "local"

    # K1 敏感检查上下文 (可选, 传了才检查)
    content_title: str = ""
    content_url: str = ""


@dataclass
class GatewayResponse:
    """网关统一响应."""

    content: str
    model: str  # 实际使用的模型
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    provider: str = ""
    error: str = ""
    stripped_thinking: bool = False  # 是否剥离了 thinking 段
    finish_reason: str = "stop"
    error_code: OmlxcErrorCode | None = None
    tool_calls: tuple[Mapping[str, object], ...] = ()


# ============================================================
# omlx 动态对接。root 由 OMLX_ROOT 决定, 与 omlx CLI 同一约定。
#
# 2026-08-10: 默认从 /Volumes/Model/omlx 迁到 ~/omlx —— launchd 服务受 macOS
# TCC 限制**无法执行外置卷上的二进制**(Operation not permitted), 导致本网关
# 作为 launchd 服务运行时调不到 omlx, 本地模型全部路由失败。
# 症状很隐蔽: 服务起得来、健康检查绿, 只有翻日志才看到 "not loadable"。
# 模型权重仍在 /Volumes/Model/LMStudio, 未迁移。
# 避免硬编码端口漂移 —— omlx 改端口/加模型后自动同步
# ============================================================
OMLX_ROOT = os.environ.get("OMLX_ROOT", os.path.expanduser("~/omlx"))
OMLX_CONF = os.path.join(OMLX_ROOT, "conf", "models.json")


# 网关别名 → omlx 本地 key(上层习惯用别名, omlx 后端用 key)
# 同一模型可能同时挂在多个引擎下(LM Link 把三机合成同一池, 每个端点都报一遍)。
# 选谁必须可复现, 否则同样的请求今天走 mac-mini 明天走 Y7000P, 排查时对不上。
# 顺序: 本机 omlx > 本机 LM Studio > mac-mini > Y7000P > 其余 > 云端。
_ENGINE_PREFERENCE = (
    "ENG-OMLX-LOCAL",
    "ENG-LMSTUDIO-MACBOOKPRO",
    "ENG-LMSTUDIO-MACMINI",
    "ENG-LMSTUDIO-Y7000P",
    "ENG-OLLAMA-MACBOOKPRO",
    "ENG-OLLAMA-MACMINI",
    "ENG-OLLAMA-Y7000P",
)


class _StreamUnsupported(Exception):
    """Registry 真流式不适用(本地专属端口模型/未注册名/自指端点), 调用方回退聚合。"""



def _id_tail(model_id: str) -> str:
    """去掉 `ENG-XXX/` 引擎前缀, 留下模型自己的名字。

    注意模型名本身可能含斜杠(qwen/qwen3.5-9b), 所以只剥第一段, 且只在
    第一段确实是引擎 id 时才剥。
    """
    head, sep, tail = model_id.partition("/")
    return tail if sep and head.startswith("ENG-") else model_id


def _engine_rank(model_id: str) -> tuple[int, str]:
    head = model_id.partition("/")[0]
    try:
        return (_ENGINE_PREFERENCE.index(head), model_id)
    except ValueError:
        return (len(_ENGINE_PREFERENCE), model_id)


def _load_aliases() -> dict[str, str]:
    """加载别名表(配置优先, 失败回退内置)。"""
    from .aliases import load_aliases

    return load_aliases()


# 兼容保留: 早期硬编码别名。新增别名请改 aliases.yaml, 不要动这里。
OMLX_ALIAS_MAP: dict[str, str] = {
    "coder": "coding-next",
    "coder-fast": "coding-fast",
    "coder-next": "coding-next",
    "reasoner": "reasoning",
    "reasoner-lite": "reasoning-lite",
    "embed": "embedding",
    "fast": "mythos-fast",
    "mid": "qwen-3.8-27b",
    "general": "qwen-3.8-27b",
    "vision-mid": "vision-large",
    "deepseek-v4-flash": "qwen-3.5-9b-flash",
    "deepseek-v4-pro": "qwen-3.5-9b-pro",
}


def _load_omlx_ports() -> dict[str, int]:
    """从 omlx models.json 读 model_name → port(含别名)。失败则回退硬编码。"""
    fallback = {
        "coding-fast": 8081,
        "coding": 8082,
        "reasoning": 8083,
        "reasoning-lite": 8085,
        "mythos-fast": 8185,
    }
    if os.environ.get("AETHERFORGE_OMLXC_MODE", "legacy").lower() != "legacy":
        return fallback
    try:
        with open(OMLX_CONF) as f:
            conf = json.load(f)
        ports = {k: m["port"] for k, m in conf.get("models", {}).items() if m.get("port")}
        # 别名也指向同一端口, 这样 model="coder" 也能解析
        for alias, key in OMLX_ALIAS_MAP.items():
            if key in ports:
                ports[alias] = ports[key]
        return ports or fallback
    except Exception:
        return fallback


def _load_omlx_sizes() -> dict[str, float]:
    """加载 omlxc 实测模型体积(GB)，供 MemoryGuard 做准入判断。

    ``~/omlx/conf/models.json`` 是模型命名和体积的 SSOT。回退值只用于
    控制面暂不可读时；它们宁可向上取整，也不能低估到放过 OOM 风险。
    """
    fallback = {
        "coding-fast": 28.0,
        "coding": 24.0,
        "coding-next": 52.0,
        "reasoning": 30.0,
        "reasoning-lite": 18.0,
        "embedding": 8.0,
        "vision": 6.0,
        "vision-large": 16.0,
        "mid-local": 16.0,
        "qwen-3.8-27b": 22.0,
        "coder-precise": 28.0,
        "mythos-fast": 5.0,
        "mythos": 18.0,
        "mistral-medium-128b": 74.0,
        # Canonical catalog IDs. deepseek-v4-* remain only as size aliases
        # for leftover client names; they are not App catalog keys.
        "qwen-3.5-9b-pro": 14.0,
        "qwen-3.5-9b-flash": 4.0,
        "deepseek-v4-pro": 14.0,
        "deepseek-v4-flash": 4.0,
    }
    out = dict(fallback)
    if os.environ.get("AETHERFORGE_OMLXC_MODE", "legacy").lower() != "legacy":
        return out
    try:
        with open(OMLX_CONF) as f:
            conf = json.load(f)
        for key, model in (conf.get("models") or {}).items():
            raw_size = model.get("size_gb") if isinstance(model, dict) else None
            if (
                isinstance(raw_size, (int, float))
                and not isinstance(raw_size, bool)
                and math.isfinite(float(raw_size))
                and float(raw_size) > 0
            ):
                out[str(key)] = float(raw_size)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        _log.warning("读取 omlxc 模型体积失败，MemoryGuard 使用安全回退值", exc_info=True)

    for alias, key in OMLX_ALIAS_MAP.items():
        if key in out:
            out[alias] = out[key]
    return out


def _load_lmstudio_fallback() -> dict[str, str]:
    """omlx 本机 key → LM Studio 池里的等价模型 id。

    SSOT 是 omlx 的 models.json(`fallback` 段), 与 omlxc 共用一份 ——
    网关和 CLI 各自硬编码一份映射迟早会漂移。
    读不到就返回空: 没有兜底比兜到错模型上强。
    """
    if os.environ.get("AETHERFORGE_OMLXC_MODE", "legacy").lower() != "legacy":
        return {}
    try:
        with open(OMLX_CONF) as f:
            conf = json.load(f)
    except Exception:
        return {}
    fb = conf.get("fallback") or {}
    out: dict[str, str] = {k: v for k, v in fb.items() if isinstance(v, str)}
    # 别名也享受同一张表(coder 和 coding 应指向同一个兜底)
    for alias, key in OMLX_ALIAS_MAP.items():
        if key in out:
            out.setdefault(alias, out[key])
    return out


def _load_ollama_fallback() -> dict[str, str]:
    """本机逻辑 key → Ollama 第二兜底模型。"""
    if os.environ.get("AETHERFORGE_OMLXC_MODE", "legacy").lower() != "legacy":
        return {}
    try:
        with open(OMLX_CONF) as f:
            conf = json.load(f)
    except Exception:
        return {}
    fb = conf.get("fallback_ollama") or {}
    out: dict[str, str] = {k: v for k, v in fb.items() if not k.startswith("_") and isinstance(v, str)}
    for alias, key in OMLX_ALIAS_MAP.items():
        if key in out:
            out.setdefault(alias, out[key])
    return out


@dataclass
class GatewayConfig:
    """网关配置."""

    # legacy = 当前回滚路径；shadow = legacy 推理 + 只读 plan；active = omlxcd 推理。
    omlxc_mode: str = field(default_factory=lambda: os.environ.get("AETHERFORGE_OMLXC_MODE", "legacy").lower())

    # omlx CLI 路径
    omlx_bin: str = field(default_factory=lambda: os.path.join(OMLX_ROOT, "bin", "omlx"))
    # 本地模型基础 URL
    # omlx 后端只绑 loopback(:4000 网关才绑 tailscale IP), 直连必须用 127.0.0.1
    local_base_url: str = "http://127.0.0.1"
    # 模型端口映射 (model_name → port)
    model_ports: dict[str, int] = field(default_factory=_load_omlx_ports)
    # 模型大小 (GB) — MemoryGuard 用 (未知大小的模型跳过检查)
    model_sizes: dict[str, float] = field(default_factory=_load_omlx_sizes)
    # 本机 omlx 后端起不来时改用 LM Studio 池里的哪个模型
    lmstudio_fallback: dict[str, str] = field(default_factory=_load_lmstudio_fallback)
    # LM Link 也失败时的 Ollama 第二兜底。
    ollama_fallback: dict[str, str] = field(default_factory=_load_ollama_fallback)
    # app = oMLX App 单进程承载(默认); legacy = 每模型独立端口, 仅供回滚。
    local_backend: str = field(default_factory=lambda: os.environ.get("AETHERFORGE_LOCAL_BACKEND", "app"))
    # 冷加载一个几十 GB 的 MLX 模型远不止 30s。原默认值会让"其实在加载中"
    # 被判成失败, 于是回退到别的模型 —— 表现为随机地拿到不是自己要的模型。
    load_ready_timeout: float = 300.0
    # 加载后再发一发 max_tokens=1 的真生成, 确认后端不是"端口活着但卡死"。
    # 关掉只在极端在意加载延迟时才有意义 —— 代价是卡死后端会静默吞请求。
    readiness_probe_enabled: bool = True
    # thinking 模型在小 max_tokens 下会把额度全花在思考段, content 留空。
    # 命中时补到这个额度重试一次。设 0 关闭(那就只能拿到空回复的失败)。
    thinking_retry_budget: int = 1024
    # 关 thinking 的参数。实测只有 reasoning_effort=none 对 LM Studio 生效;
    # chat_template_kwargs.enable_thinking / thinking、reasoning_effort=low、
    # 消息里加 /no_think 前缀 —— 四种写法都不管用(2026-08-10, qwen3.5-9b)。
    # 设成 None 可整体关掉这条补救。
    no_think_param: dict[str, object] | None = field(default_factory=lambda: {"reasoning_effort": "none"})
    # 本网关自己的 OpenAI 门面端点。SSOT 的 ENG-OMLX-LOCAL 现指向门面
    # (原先指向 LiteLLM :4000), 于是 registry 回退路径有可能打回自己 ——
    # 一个请求在"直连端口失败 → 回退 registry → 门面 → 本网关"之间成环。
    # 这里显式声明自身端点, 供 _is_self_endpoint 识别并跳过。
    self_facade_ports: tuple[int, ...] = (9290,)
    # 别名表: 消费者意图名 → 可路由模型名。配置驱动, 见 aliases.py/aliases.yaml。
    # openai_proxy(HTTP) 与 aetherforge.bridge(库) 共用同一个 ModelGateway,
    # 因此天然共用这一份 —— 两个入口的路由结果必须一致。
    aliases: dict[str, str] = field(default_factory=_load_aliases)
    # fallback 链 (按优先级)
    fallback_chain: list[str] = field(default_factory=lambda: ["coding", "reasoning", "mythos-fast"])
    # 按复杂度分流: level → 定制 fallback 链 (未配置的 level 回退 fallback_chain)
    complexity_chains: dict[str, list[str]] = field(
        default_factory=lambda: {
            "simple": ["mythos-fast", "coding-fast", "coding"],
            "complex": ["reasoning", "coding", "mythos-fast"],
        }
    )
    # MemoryGuard: 预留内存倍数
    memory_safety_factor: float = 1.2
    # 是否启用内存检查
    memory_check_enabled: bool = True
    # WarmPool: keep-last-used TTL (秒)
    warm_pool_ttl: int = 300
    # 健康检查间隔 (秒)
    health_check_interval: int = 60
    # 首次/周期模型发现的单 provider 上限。
    # 2026-08-23: 原为 2.0s(本地 loopback/tailnet 足够, 但实测云端 provider
    # 的真实网络往返经常超时被静默跳过 —— 结合上面两个 404/引擎前缀修复,
    # 这是"云端资源紧张"表象的最后一块拼图: discover 数据没进 registry,
    # 前面两个修复再对也无济于事)。调到 8.0s: 实测 15s 能完整 discover 全部
    # 197 个模型, 8s 留足真实余量; 只在首次/registry 为空时触发一次(见
    # _ensure_registry_ready, 成功后 _registry_ready=True 不再重跑), 代价
    # 只是首次触发多等几秒, 不是持续性开销。
    registry_discover_timeout: float = 8.0
    # 是否启用后台任务 (warm pool sweep + health check)
    background_tasks_enabled: bool = True

    def __post_init__(self) -> None:
        if self.omlxc_mode not in {"legacy", "shadow", "active"}:
            raise ValueError("AETHERFORGE_OMLXC_MODE must be one of: legacy, shadow, active")


# ============================================================
# MemoryGuard — 加载前检查内存
# ============================================================
class MemoryGuard:
    """加载模型前检查可用内存, 防止 OOM/swap thrash."""

    def __init__(self, safety_factor: float = 1.2):
        self._safety_factor = safety_factor

    def can_load(self, model_size_gb: float) -> bool:
        """检查是否有足够内存加载模型."""
        free_gb = self._get_free_memory_gb()
        required = model_size_gb * self._safety_factor
        if free_gb < required:
            _log.warning(
                "[MemoryGuard] 内存不足: 需要 %.1fGB, 可用 %.1fGB (模型 %.1fGB × %.1f)",
                required,
                free_gb,
                model_size_gb,
                self._safety_factor,
            )
            return False
        return True

    def _get_free_memory_gb(self) -> float:
        """获取可用内存 (GB). macOS: vm_stat + sysctl."""
        try:
            import subprocess

            # 用 vm_stat 获取 page size 和 free pages
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return 64.0  # 无法获取时假设充足

            lines = result.stdout.strip().split("\n")
            page_size = 4096  # 默认 4KB
            free_pages = 0
            inactive_pages = 0

            for line in lines:
                if "page size of" in line:
                    import re

                    m = re.search(r"page size of (\d+) bytes", line)
                    if m:
                        page_size = int(m.group(1))
                elif "Pages free" in line:
                    free_pages = int(line.split(":")[-1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    inactive_pages = int(line.split(":")[-1].strip().rstrip("."))

            free_bytes = (free_pages + inactive_pages) * page_size
            return free_bytes / (1024**3)
        except Exception:
            return 64.0  # 无法获取时假设充足


# ============================================================
# ModelGateway — 统一入口
# ============================================================
class ModelGateway:
    """统一模型网关 — 所有 LLM 调用的唯一入口.

    用法:
        gateway = ModelGateway.create()
        resp = await gateway.generate(GatewayRequest(
            messages=[{"role": "user", "content": "hello"}],
            model="coding-fast",
        ))
        print(resp.content)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        scheduler: ModelScheduler,
        config: GatewayConfig | None = None,
        metrics: MetricsCollector | None = None,
        omlxc_client: OmlxcClient | None = None,
    ):
        self._registry = registry
        self._scheduler = scheduler
        self._config = config or GatewayConfig()
        self._metrics = metrics or MetricsCollector()
        self._memory_guard = MemoryGuard(self._config.memory_safety_factor)
        self._complexity_scorer = TaskComplexityScorer()
        self._omlxc = omlxc_client or OmlxcClient()

        # 已加载模型集合 (model_name → load_time)
        self._loaded_models: dict[str, float] = {}
        # 实测需要关 thinking 才出正文的 model_id。只活在本进程内 ——
        # 模型换了模板重启一次就自然纠正, 不必持久化。
        self._needs_no_think: set[str] = set()
        # 加载锁 (防止并发 load 两个大模型)
        self._load_lock = asyncio.Lock()
        # warm pool: model_name → last_used_time
        self._last_used: dict[str, float] = {}
        # health status: model_name → consecutive_failures
        self._health_failures: dict[str, int] = {}
        # 后台任务
        self._bg_tasks: list[asyncio.Task] = []

    @classmethod
    def create(
        cls,
        m1_engine_dir: str = "",
        m1_model_dir: str = "",
        config: GatewayConfig | None = None,
    ) -> ModelGateway:
        """工厂方法: 从 SSOT 创建网关."""
        engine_dir = m1_engine_dir or str(M1_COMPUTE_ENGINE_DIR)
        model_dir = m1_model_dir or (str(M1_MODEL_DIR) if M1_MODEL_DIR.exists() else "")

        registry = ModelRegistry()
        if os.path.isdir(engine_dir):
            load_ssot_models(registry, engine_dir, model_dir or None)

        scheduler = ModelScheduler(registry)
        registry.set_scheduler(scheduler)

        return cls(registry, scheduler, config)

    # ==========================================================
    # 核心: generate
    # ==========================================================
    async def _ensure_registry_ready(self) -> None:
        """registry 尚未发现模型时补一次 discover(幂等, 只跑一次)。"""
        if getattr(self, "_registry_ready", False):
            return
        try:
            if not self._registry.list_models():
                discovered = await self._registry.refresh(self._config.registry_discover_timeout)
                # 2026-08-23: discover 返回 0 个模型和返回几百个模型此前日志级别
                # 完全一样(静默), 一次真实故障(启动脚本里 AETHERFORGE_M1_DIR
                # 指向一个已被清理的 worktree, SSOT 路径 exists=False, discover
                # 立即返回 0 个模型且不报错)排查了很久才定位到 —— 这类"配置生效
                # 但数据是空的"情况必须比日常的 provider discover 失败更显眼。
                if not discovered:
                    _log.error(
                        "[ModelGateway] registry discover returned 0 models — "
                        "check M1_MODEL_DIR/M1_COMPUTE_ENGINE_DIR resolve to a real, "
                        "populated SSOT path (see AETHERFORGE_M1_DIR / "
                        "AETHERFORGE_M1_COMPUTE_DIR env overrides); cloud providers "
                        "will be silently unreachable until this is fixed"
                    )
            self._registry_ready = True  # only set on success
        except Exception as e:
            _log.warning("[ModelGateway] registry refresh failed: %s", e)
            # Do NOT set _registry_ready — allow retry on next call

    async def list_omlxc_models(self) -> tuple[OmlxcCatalogModel, ...]:
        """Return only the logical models executable by the active UDS data plane."""
        if self._config.omlxc_mode != "active":
            raise OmlxcError(OmlxcErrorCode.INVALID)
        return await self._omlxc.list_models()

    async def observe_omlxc_compute(self) -> dict[str, object]:
        """Read-only inventory observe. Orthogonal to routing mode and /health."""
        mode = self._config.omlxc_mode
        try:
            report = await self._omlxc.health()
        except OmlxcError as error:
            mapped = {
                OmlxcErrorCode.TIMEOUT: "timeout",
                OmlxcErrorCode.UNAVAILABLE: "unavailable",
                OmlxcErrorCode.INVALID: "invalid",
            }.get(error.code, "unavailable")
            return {"omlxc": mapped, "omlxc_mode": mode, "warnings": []}
        except Exception:
            return {"omlxc": "unavailable", "omlxc_mode": mode, "warnings": []}
        return {
            "omlxc": "ok",
            "omlxc_mode": mode,
            "warnings": _sanitize_inventory_warnings(report.warnings),
        }

    async def generate(self, request: GatewayRequest) -> GatewayResponse:
        """带端到端 deadline 的统一入口。

        timeout 覆盖发现、选路、加载、重试和全部 fallback，而不是每一跳都重新
        获得一份完整预算。这个区别决定故障时是 30 秒返回，还是挂几分钟。
        """
        t0 = time.time()
        try:
            async with asyncio.timeout(max(0.1, request.timeout)):
                if self._config.omlxc_mode == "legacy":
                    return await self._generate_legacy(request)
                if self._config.omlxc_mode == "shadow":
                    return await self._generate_shadow(request)
                return await self._generate_active(request)
        except TimeoutError:
            return GatewayResponse(
                content="",
                model="",
                latency_ms=(time.time() - t0) * 1000,
                error=f"Gateway deadline exceeded ({request.timeout:.1f}s)",
                finish_reason="error",
                error_code=OmlxcErrorCode.TIMEOUT,
            )

    async def _generate_shadow(self, request: GatewayRequest) -> GatewayResponse:
        """Compare only the local route plan; inference remains exactly legacy."""
        plan = None
        logical = request.model or self._business_model(request)
        resolved = self.resolve_alias(logical)
        if request.routing_mode != "cloud":
            try:
                plan = await self._omlxc.route_plan(
                    resolved,
                    profile="interactive",
                    capabilities={"chat"},
                    thinking=False,
                    timeout=min(request.timeout, 2.0),
                )
            except OmlxcError as error:
                _log.warning(
                    "omlxc shadow route failed logical=%s resolved=%s code=%s",
                    logical,
                    resolved,
                    error.code.value,
                )
        result = await self._generate_legacy(request)
        if plan is not None:
            _log.info(
                "omlxc shadow route request_id=%s logical=%s resolved=%s "
                "legacy_provider=%s legacy_model=%s placement=%s explanation=%s",
                plan.request_id,
                logical,
                resolved,
                result.provider,
                result.model,
                plan.selected,
                plan.explanation[:120],
            )
        return result

    async def _generate_active(self, request: GatewayRequest) -> GatewayResponse:
        """Execute the approved local phase only through omlxcd."""
        if request.routing_mode not in {"local", "hybrid", "cloud"}:
            return GatewayResponse(
                content="",
                model="",
                latency_ms=0,
                error=f"Invalid routing_mode: {request.routing_mode}",
                finish_reason="error",
                error_code=OmlxcErrorCode.INVALID,
            )
        sensitive = bool(
            (request.content_title or request.content_url) and _is_sensitive(request.content_title, request.content_url)
        )
        if request.routing_mode == "cloud" and not sensitive:
            return await self._generate_legacy(request)
        logical = request.model or self._business_model(request)
        resolved = self.resolve_alias(logical)
        t0 = time.time()
        agent_fields = self._agent_fields(request.extra)
        try:
            result = await self._omlxc.chat(
                model=resolved,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=request.timeout,
                profile="interactive",
                thinking=False,
                **agent_fields,
            )
        except OmlxcError as error:
            if sensitive:
                return GatewayResponse(
                    content="",
                    model=logical,
                    latency_ms=(time.time() - t0) * 1000,
                    error="[K1] local inference unavailable",
                    finish_reason="error",
                    error_code=error.code,
                )
            if request.routing_mode == "hybrid" and error.cloud_fallback_allowed:
                return await self._generate_legacy(replace(request, routing_mode="cloud"))
            return GatewayResponse(
                content="",
                model=logical,
                latency_ms=(time.time() - t0) * 1000,
                error="local inference failed",
                finish_reason="error",
                error_code=error.code,
            )
        return GatewayResponse(
            content=strip_thinking(result.content),
            model=logical,
            latency_ms=(time.time() - t0) * 1000,
            tokens_in=result.usage.prompt_tokens,
            tokens_out=result.usage.completion_tokens,
            provider=f"omlxc:{result.backend or 'local'}",
            finish_reason=result.finish_reason,
            tool_calls=tuple(result.tool_calls),
        )

    @staticmethod
    def _agent_fields(extra: Mapping[str, Any]) -> dict[str, Any]:
        return {key: extra[key] for key in ("tools", "tool_choice") if key in extra and extra[key] is not None}

    def _business_model(self, request: GatewayRequest) -> str:
        prompt = self._extract_prompt(request.messages)
        complexity = self._complexity_scorer.estimate(prompt=prompt, task=request.task)
        chain = self._config.complexity_chains.get(complexity.level, self._config.fallback_chain)
        return chain[0] if chain else "coding"

    async def _generate_legacy(self, request: GatewayRequest) -> GatewayResponse:
        """统一生成入口.

        流程:
          1. K1 敏感检查 (硬拦)
          2. 复杂度评估 → 选择 fallback 链 (小任务用小模型, 大任务用大模型)
          3. ensure_model (load + MemoryGuard)
          4. 推理 + 超时
          5. strip_thinking (兜底)
          6. 记账
        """
        t0 = time.time()
        if request.routing_mode not in {"local", "hybrid", "cloud"}:
            return GatewayResponse(
                content="",
                model="",
                latency_ms=0,
                error=f"Invalid routing_mode: {request.routing_mode}",
                finish_reason="error",
            )

        # 0. 惰性发现: registry 为空时先 discover(SSOT model_defs 不走网络, 很快)
        await self._ensure_registry_ready()

        # 1. K1 敏感硬拦
        if request.content_title or request.content_url:
            if _is_sensitive(request.content_title, request.content_url):
                # 敏感流: 只允许本地模型, 禁止云端
                return await self._generate_local_only(request)

        # 2. 复杂度评估 → 选择 fallback 链 (小任务用小模型, 大任务用大模型)
        prompt = self._extract_prompt(request.messages)
        complexity = self._complexity_scorer.estimate(
            prompt=prompt,
            task=request.task,
        )
        chain = self._config.complexity_chains.get(
            complexity.level,
            self._config.fallback_chain,
        )
        _log.debug(
            "[ModelGateway] complexity=%s (%.3f) signals=%s chain=%s",
            complexity.level,
            complexity.score,
            complexity.signals,
            chain,
        )

        # 3. Scheduler 选路 — 与复杂度链是互补的两层:
        #    复杂度决定"用哪条链"(该配多大的模型), scheduler 决定"链里先试谁"
        #    (当下哪个节点最健康/最省)。两者都失效时仍回落到复杂度链本身。
        full_chain = [request.model] if request.model else []
        try:
            from .types import ModelRequest

            sched_req = ModelRequest(task=request.task or "chat")
            selection = await self._scheduler.select_model(sched_req)
            if selection and selection.model.name:
                sched_model = selection.model.name
                sched_id = selection.model.id or sched_model
                allowed = self._routing_allows_model(sched_id, request.routing_mode)
                if allowed and sched_model not in full_chain:
                    full_chain.append(sched_model)
                _log.info(
                    "[ModelGateway] scheduler selected: %s (%.2f) — %s",
                    sched_model,
                    selection.confidence,
                    selection.reasoning[:60],
                )
        except Exception as e:
            _log.debug("[ModelGateway] scheduler selection skipped: %s", e)

        # 4. 尝试 fallback 链
        full_chain.extend(chain)

        # 去重并执行本地/云边界。默认 local，不再因为 scheduler 恰好偏爱某个
        # 云模型就把本地任务送出去。
        filtered_chain: list[str] = []
        for model_name in full_chain:
            if model_name in filtered_chain:
                continue
            if self._routing_allows_model(model_name, request.routing_mode):
                filtered_chain.append(model_name)
        full_chain = filtered_chain

        last_error = ""
        for model_name in full_chain:
            if self._health_failures.get(model_name, 0) >= 3:
                continue  # 跳过已标记不健康的模型

            try:
                resp = await self._try_generate(model_name, request)
                self._last_used[model_name] = time.time()
                self._health_failures.pop(model_name, None)  # 成功重置
                self._metrics.record_generation(
                    model=model_name,
                    latency_ms=(time.time() - t0) * 1000,
                    cost=resp.cost_usd,
                    tokens=resp.tokens_in + resp.tokens_out,
                )
                return resp
            except Exception as e:
                last_error = str(e)[:100]
                self._health_failures[model_name] = self._health_failures.get(model_name, 0) + 1
                _log.warning("[ModelGateway] %s failed: %s", model_name, last_error)
                continue

        return GatewayResponse(
            content="",
            model="",
            latency_ms=(time.time() - t0) * 1000,
            error=f"All models failed. Last: {last_error}",
        )

    async def generate_stream(self, request: GatewayRequest) -> AsyncGenerator[OmlxcStreamChunk]:
        """Stream without buffering the active local UDS response."""
        sensitive = bool(
            (request.content_title or request.content_url) and _is_sensitive(request.content_title, request.content_url)
        )
        effective = replace(request, routing_mode="local") if sensitive else request

        if self._config.omlxc_mode == "legacy":
            relay = self._relay_stream(self._generate_legacy_stream(effective))
            try:
                async for chunk in relay:
                    yield chunk
            finally:
                await relay.aclose()
            return

        logical = effective.model or self._business_model(effective)
        resolved = self.resolve_alias(logical)
        if self._config.omlxc_mode == "shadow":
            if effective.routing_mode != "cloud":
                try:
                    await self._omlxc.route_plan(
                        resolved,
                        profile="interactive",
                        capabilities={"chat", "streaming"},
                        thinking=False,
                        timeout=min(effective.timeout, 2.0),
                    )
                except OmlxcError as error:
                    _log.warning(
                        "omlxc shadow stream route failed logical=%s resolved=%s code=%s",
                        logical,
                        resolved,
                        error.code.value,
                    )
            relay = self._relay_stream(self._generate_legacy_stream(effective))
            try:
                async for chunk in relay:
                    yield chunk
            finally:
                await relay.aclose()
            return

        if effective.routing_mode == "cloud":
            relay = self._relay_stream(self._generate_legacy_stream(effective))
            try:
                async for chunk in relay:
                    yield chunk
            finally:
                await relay.aclose()
            return

        emitted = False
        source = self._omlxc.stream_chat(
            model=resolved,
            messages=effective.messages,
            temperature=effective.temperature,
            max_tokens=effective.max_tokens,
            timeout=effective.timeout,
            profile="interactive",
            thinking=False,
            **self._agent_fields(effective.extra),
        )
        try:
            async for chunk in source:
                emitted = emitted or bool(chunk.content) or bool(chunk.tool_calls)
                yield replace(chunk, model=logical)
        except OmlxcError as error:
            if sensitive:
                raise OmlxcError(error.code, emitted_content=emitted) from error
            if effective.routing_mode == "hybrid" and not emitted and error.cloud_fallback_allowed:
                relay = self._relay_stream(self._generate_legacy_stream(replace(effective, routing_mode="cloud")))
                try:
                    async for chunk in relay:
                        yield chunk
                finally:
                    await relay.aclose()
                return
            raise
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    async def _generate_legacy_stream(self, request: GatewayRequest) -> AsyncGenerator[OmlxcStreamChunk]:
        """Legacy stream boundary; 优先 registry 真流式, 回退到聚合单块。

        真流式只覆盖 registry 路径(云端/LM Link 池引擎): provider 层
        AnthropicCompat/OpenAI 均已支持 SSE 逐块输出。本地 omlx 专属端口
        模型、registry 未命中的名字、以及首块之前的任何失败, 全部回退到
        与旧实现语义一致的聚合路径(自带完整 fallback 链)。已吐出内容后
        的失败无法回退, 如实上抛 —— 与 active 模式 omlxc 流式行为对齐。
        """
        stream = self._try_registry_stream(request)
        emitted = False
        try:
            async for chunk in stream:
                emitted = True
                yield chunk
            return
        except _StreamUnsupported:
            pass
        except Exception:
            if emitted:
                raise
            _log.info("[ModelGateway] true-stream unavailable before first chunk, aggregating")
        finally:
            await stream.aclose()

        response = await self._generate_legacy(request)
        if response.error and not response.content:
            raise OmlxcError(OmlxcErrorCode.UNAVAILABLE)
        yield OmlxcStreamChunk(
            content=response.content,
            model=response.model,
            finish_reason=response.finish_reason,
            usage={
                "prompt_tokens": response.tokens_in,
                "completion_tokens": response.tokens_out,
                "total_tokens": response.tokens_in + response.tokens_out,
            },
        )

    async def _try_registry_stream(self, request: GatewayRequest) -> AsyncGenerator[OmlxcStreamChunk]:
        """Registry 引擎的真流式(异步生成器: 所有不支持判断在首个 __anext__ 冒出)。"""
        logical = request.model or self._business_model(request)
        resolved = self.resolve_alias(logical)
        if resolved in self._config.model_ports:
            # 本地 omlx 专属端口走 ensure+直连(A 分支), 不属于 registry 流式。
            raise _StreamUnsupported(resolved)
        model_id = self._resolve_model_id(resolved)
        if not model_id or self._provider_is_self(model_id):
            raise _StreamUnsupported(resolved)
        source = self._registry.chat_stream(
            model_id,
            request.messages,
            ChatOptions(temperature=request.temperature, max_tokens=request.max_tokens, stream=True),
        )
        finish_emitted = False
        async for chunk in source:
            if not chunk.content and chunk.finish_reason is None:
                continue
            yield OmlxcStreamChunk(content=chunk.content, model=logical, finish_reason=chunk.finish_reason)
            if chunk.finish_reason:
                finish_emitted = True
        if not finish_emitted:
            # provider 基类默认实现(一-shot)不带 finish_reason, 补上终止块。
            yield OmlxcStreamChunk(model=logical, finish_reason="stop")

    @staticmethod
    async def _relay_stream(
        source: AsyncIterator[OmlxcStreamChunk],
    ) -> AsyncGenerator[OmlxcStreamChunk]:
        try:
            async for chunk in source:
                yield chunk
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    async def _generate_local_only(self, request: GatewayRequest) -> GatewayResponse:
        """敏感流: 只用本地模型, 不 fallback 到云端."""
        t0 = time.time()

        # 只尝试本地模型
        local_models = [m for m in self._config.fallback_chain if m != "deepseek-chat"]
        chain = [request.model] if request.model and request.model != "deepseek-chat" else []
        chain.extend(local_models)
        chain = [m for m in dict.fromkeys(chain) if self._routing_allows_model(m, "local")]

        for model_name in chain:
            try:
                resp = await self._try_generate(model_name, request)
                self._last_used[model_name] = time.time()
                return resp
            except Exception as e:
                _log.warning("[ModelGateway] local-only %s failed: %s", model_name, e)
                continue

        return GatewayResponse(
            content="",
            model="",
            latency_ms=(time.time() - t0) * 1000,
            error="[K1] 敏感流无可用本地模型",
        )

    async def _try_generate(self, model_name: str, request: GatewayRequest) -> GatewayResponse:
        """尝试用指定模型生成.

        路由分两类, 互不串门:

        A. 本机 omlx 模型(解析后落在 model_ports 里)
           端口通 → 直连;  端口不通 → 先 `omlxc load` 拉起再直连;
           拉不起来 → 按 models.json 的 fallback 映射落 LM Studio;
           都不行 → 抛错, 交给上层 fallback 链。
           **绝不**掉进 registry 的模糊匹配 —— 它们有专属端口, 顶包出来的
           是完全不同的模型(实测 reasoning → ...-reasoning-distilled)。

        B. 其余(云端 / LM Link 池)
           registry + provider chain。
        """
        t0 = time.time()
        local_key = self.resolve_alias(model_name)

        # ── A. 本机 omlx ────────────────────────────────────
        if local_key in self._config.model_ports:
            if self._config.local_backend == "app":
                app_id = self._resolve_model_id(local_key, ("ENG-OMLX-LOCAL",))
                if app_id and not self._provider_is_self(app_id):
                    try:
                        return await self._generate_via_registry(app_id, model_name, request, t0)
                    except Exception as e:
                        _log.warning("[ModelGateway] oMLX App %s 失败: %s", local_key, e)
                else:
                    _log.warning("[ModelGateway] oMLX App 模型不可解析或端点自指: %s", local_key)
            else:
                port = self._config.model_ports[local_key]
                reachable = await self._port_reachable(port)
                if not reachable:
                    # legacy 回滚模式才拉独立 server；App 模式严禁偷偷复活旧架构。
                    self._loaded_models.pop(local_key, None)
                    reachable = await self._ensure_model(local_key)
                if reachable:
                    try:
                        return await self._generate_via_omlx_router(local_key, request, t0, display_name=model_name)
                    except Exception as e:
                        _log.warning("[ModelGateway] legacy omlx 直连 %s 失败: %s", local_key, e)
                        if isinstance(e, TimeoutError) or "timeout" in str(e).lower():
                            await self._recycle_backend(local_key)

            fb = self._lmstudio_fallback(local_key)
            if fb:
                _log.warning("[ModelGateway] %s → LM Link 兜底 %s", local_key, fb)
                fb_id = self._resolve_model_id(fb, ("ENG-LMSTUDIO-",))
                if fb_id and not self._provider_is_self(fb_id):
                    try:
                        return await self._generate_via_registry(fb_id, model_name, request, t0)
                    except Exception as e:
                        _log.warning("[ModelGateway] LM Link 兜底 %s 失败: %s", fb, e)

            ollama = self._ollama_fallback(local_key)
            if ollama:
                _log.warning("[ModelGateway] %s → Ollama 二级兜底 %s", local_key, ollama)
                ollama_id = self._resolve_model_id(ollama, ("ENG-OLLAMA-",))
                if ollama_id and not self._provider_is_self(ollama_id):
                    return await self._generate_via_registry(ollama_id, model_name, request, t0)

            raise RuntimeError(
                f"{model_name}: oMLX App/legacy 本地后端不可用"
                + (f", LM Link 兜底 {fb} 不可用" if fb else ", 无 LM Link 兜底")
                + (f", Ollama 兜底 {ollama} 不可用" if ollama else ", 无 Ollama 兜底")
            )

        # ── B. registry + provider chain ────────────────────
        model_id = self._resolve_model_id(model_name)
        if model_id and self._provider_is_self(model_id):
            # 该模型的 provider 端点就是本网关的门面 —— 走下去会成环
            # (直连失败 → registry → 门面 → 本网关 → 直连失败 → ...)。
            # 如实报错比在环里耗尽超时好排查得多。
            raise RuntimeError(
                f"{model_name}: provider endpoint points back at this gateway's "
                f"own facade; local model must be served by its direct port"
            )
        if model_id:
            return await self._generate_via_registry(model_id, model_name, request, t0)

        raise RuntimeError(f"Model {model_name} not in registry")

    async def _generate_via_registry(
        self,
        model_id: str,
        display_name: str,
        request: GatewayRequest,
        t0: float,
    ) -> GatewayResponse:
        """经 registry/provider 链生成。display_name 是消费者原本要的名字。"""

        async def _call(max_tokens: int | None, extra: dict | None = None):
            merged_extra = dict(request.extra)
            if extra:
                merged_extra.update(extra)
            return await self._registry.chat(
                model_id,
                request.messages,
                ChatOptions(
                    temperature=request.temperature,
                    max_tokens=max_tokens,
                    extra=merged_extra or None,
                ),
            )

        # 之前已经证实过这个模型不关 thinking 就不出正文 —— 直接带上,
        # 省掉那次注定烧满预算的首发(实测 triage 热态 14s → 1s)。
        preset = (
            dict(self._config.no_think_param)
            if model_id in self._needs_no_think and self._config.no_think_param
            else None
        )
        result = await _call(request.max_tokens, preset)
        if not result:
            raise RuntimeError(f"No response from {display_name}")

        content = result.content or ""
        stripped = _strip_thinking(content)

        # 预算耗尽在思考段: finish_reason=length 且剥离后没正文。
        # 这不是模型不行, 是给的额度不够 —— 补足再来一次, 只补一次。
        if not stripped.strip() and result.finish_reason == "length":
            # 第一手: 直接把 thinking 关掉。实测 qwen/qwen3.5-9b 从
            # 8.2s/64token 空回复变成 0.6s/2token 正常回答, 比抬预算划算得多。
            if self._config.no_think_param:
                _log.info(
                    "[ModelGateway] %s 预算耗尽在思考段(max_tokens=%s), 关 thinking 重试",
                    display_name,
                    request.max_tokens,
                )
                try:
                    retry = await _call(request.max_tokens, dict(self._config.no_think_param))
                except Exception as e:
                    _log.info("[ModelGateway] %s 不接受关 thinking 参数: %s", display_name, e)
                    retry = None
                if retry and _strip_thinking(retry.content or "").strip():
                    result = retry
                    content = retry.content or ""
                    stripped = _strip_thinking(content)
                    self._needs_no_think.add(model_id)  # 记住, 下次直接带上

            # 第二手: 下游不认这个参数(或认了仍不出正文)时才抬预算。
            budget = self._config.thinking_retry_budget
            if not stripped.strip() and budget and (request.max_tokens or 0) < budget:
                _log.info("[ModelGateway] %s 仍无正文, 预算提到 %d 再试一次", display_name, budget)
                result = await _call(budget)
                content = (result.content or "") if result else ""
                stripped = _strip_thinking(content)

        was_stripped = stripped != content
        # 到这儿还空, 就是真没回答。返回空的 200 会让上层以为成功, 必须当失败,
        # 好让 fallback 链继续往下走。
        if not stripped.strip():
            raise RuntimeError(
                f"{display_name} via {model_id}: 回复为空"
                + (
                    f"(finish={result.finish_reason}, 补到 {self._config.thinking_retry_budget} token 仍无正文)"
                    if result and result.finish_reason == "length"
                    else "(thinking 段剥离后无正文)"
                    if was_stripped
                    else ""
                )
            )
        if result is None:
            raise RuntimeError(f"{display_name} via {model_id}: empty provider result")

        usage = result.usage or {}
        provider = self._registry.get_provider(model_id)
        provider_name = provider.name if provider else ""

        return GatewayResponse(
            content=stripped,
            model=display_name,
            latency_ms=(time.time() - t0) * 1000,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            provider=provider_name,
            stripped_thinking=was_stripped,
            finish_reason=result.finish_reason or "stop",
        )

    def _lmstudio_fallback(self, local_key: str) -> str | None:
        """本机 omlx 后端起不来时的 LM Studio 兜底模型名。

        映射来自 omlx models.json 的 `fallback` 段 —— 与 omlxc 同一 SSOT,
        避免网关和 CLI 各持一份互相漂移。
        """
        return self._config.lmstudio_fallback.get(local_key)

    def _ollama_fallback(self, local_key: str) -> str | None:
        """LM Link 之后的 Ollama 兜底，映射同样来自 models.json。"""
        return self._config.ollama_fallback.get(local_key)

    async def _generate_via_omlx_router(
        self,
        model_name: str,
        request: GatewayRequest,
        t0: float,
        display_name: str | None = None,
    ) -> GatewayResponse:
        """Generate via direct omlx port — bypasses dead :4000 proxy.

        Connects directly to the model's port (from model_ports config).
        omlx servers require the full model path in the 'model' field,
        so we fetch it from GET /v1/models first (cached per model).
        """
        import aiohttp

        # Ensure model is loaded on its port
        ready = await self._ensure_model(model_name)
        if not ready:
            raise RuntimeError(f"Model {model_name} not loadable")

        port = self._config.model_ports[model_name]
        base = self._config.local_base_url

        real_model_id = await self._omlx_real_model_id(base, port, model_name)

        url = f"{base}:{port}/v1/chat/completions"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        "model": real_model_id,
                        "messages": request.messages,
                        # None 时不下发, 保持模型自身默认(而不是硬塞一个值)
                        **({"temperature": request.temperature} if request.temperature is not None else {}),
                        **({"max_tokens": request.max_tokens} if request.max_tokens is not None else {}),
                        **request.extra,
                    },
                    timeout=aiohttp.ClientTimeout(total=request.timeout),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"omlx :{port} {model_name} HTTP {resp.status}: {body[:100]}")
                    data = await resp.json()
        except TimeoutError:
            raise RuntimeError(f"omlx :{port} {model_name} timeout ({request.timeout}s)")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"omlx :{port} {model_name} connection error: {e}")

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        stripped = _strip_thinking(content)
        was_stripped = stripped != content
        if not stripped.strip():
            raise RuntimeError(f"{display_name or model_name}: legacy 后端回复为空")

        usage = data.get("usage", {})
        latency_ms = (time.time() - t0) * 1000

        return GatewayResponse(
            content=stripped,
            # 回消费者问的那个名字(coder), 而不是内部后端键(coding)
            model=display_name or model_name,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            provider="ENG-OMLX-LOCAL",
            stripped_thinking=was_stripped,
            finish_reason=data.get("choices", [{}])[0].get("finish_reason") or "stop",
        )

    async def _port_reachable(self, port: int, timeout: float = 0.5) -> bool:
        """Quick TCP reachability check — avoids slow timeouts on unreachable ports."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._config.local_base_url}:{port}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ==========================================================
    # 模型管理
    # ==========================================================
    async def _ensure_model(self, model_name: str) -> bool:
        """确保模型已加载 (自动 load + MemoryGuard)."""
        if model_name not in self._config.model_ports:
            _log.warning("[ModelGateway] %s 无端口配置", model_name)
            return False

        if model_name in self._loaded_models:
            # 只信端口, 不信缓存 —— autopilot 会按 idle TTL 在网关背后卸载模型,
            # 缓存说"已加载"而端口早断, 是上一轮间歇性失败的帮凶之一。
            if await self._port_reachable(self._config.model_ports[model_name]):
                return True
            _log.info("[ModelGateway] %s 缓存说已加载但端口已断, 重新拉起", model_name)
            self._loaded_models.pop(model_name, None)

        if model_name not in self._config.model_ports:
            _log.warning("[ModelGateway] %s 无端口配置", model_name)
            return False

        # MemoryGuard: 加载前检查内存
        if self._config.memory_check_enabled and model_name in self._config.model_sizes:
            size_gb = self._config.model_sizes[model_name]
            if not self._memory_guard.can_load(size_gb):
                _log.error(
                    "[ModelGateway] %s 内存不足 (需要 %.1fGB), 跳过加载",
                    model_name,
                    size_gb * self._config.memory_safety_factor,
                )
                return False

        async with self._load_lock:
            # double-check after lock
            if model_name in self._loaded_models:
                return True

            port = self._config.model_ports[model_name]
            base_url = self._config.local_base_url

            try:
                # 用 omlxc load
                proc = await asyncio.create_subprocess_exec(
                    self._config.omlx_bin,
                    "load",
                    self.resolve_alias(model_name),  # 别名→本地 key(走统一解析点)
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # `omlxc load` 只是后台拉起进程就返回, 不等模型加载完,
                # 所以这里 30s 足够; 真正的等待在下面的 _wait_healthy。
                _, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=30,
                )

                if proc.returncode != 0:
                    _log.warning(
                        "[ModelGateway] omlxc load %s failed: %s",
                        model_name,
                        stderr.decode()[:200],
                    )
                    return False

                # 等待端口应答(存活)
                if not await self._wait_healthy(model_name, base_url, port, timeout=self._config.load_ready_timeout):
                    _log.warning("[ModelGateway] %s load timeout", model_name)
                    return False

                # 再证明它真能生成(就绪)。/v1/models 只能说明进程活着 ——
                # 卡死的 mlx_lm.server 照样答 200。这一发同时把权重预热进内存,
                # 所以不是白花的开销: 用户的第一个请求本来也要付这份冷加载。
                if self._config.readiness_probe_enabled and not await self._probe_generation(
                    model_name, base_url, port, timeout=self._config.load_ready_timeout
                ):
                    _log.error("[ModelGateway] %s 端口应答但生成不了(疑似卡死), 回收后端", model_name)
                    await self._recycle_backend(model_name)
                    return False

                self._loaded_models[model_name] = time.time()
                _log.info("[ModelGateway] %s loaded on port %d", model_name, port)
                return True

            except TimeoutError:
                _log.warning("[ModelGateway] %s load timed out", model_name)
                return False
            except Exception as e:
                _log.error("[ModelGateway] %s load error: %s", model_name, e)
                return False

    async def _probe_generation(self, model_name: str, base_url: str, port: int, timeout: float) -> bool:
        """发一发最小生成, 确认后端真能干活(不只是端口应答)。

        用 max_tokens=1, 内容随便; 拿到 200 且有 choices 即算就绪。
        """
        import aiohttp

        real_id = await self._omlx_real_model_id(base_url, port, model_name)
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{base_url}:{port}/v1/chat/completions",
                    json={
                        "model": real_id,
                        "messages": [{"role": "user", "content": "ok"}],
                        "max_tokens": 1,
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as r:
                    if r.status != 200:
                        _log.warning("[ModelGateway] %s 就绪探针 HTTP %s", model_name, r.status)
                        return False
                    return bool((await r.json()).get("choices"))
        except Exception as e:
            _log.warning("[ModelGateway] %s 就绪探针失败: %s", model_name, e)
            return False

    async def _recycle_backend(self, model_name: str) -> None:
        """回收卡死的本机后端 —— 它不会自愈, 留着只会让后续请求继续超时。

        只 stop 不 start: 下一次请求走 _ensure_model 自然会拉起干净的进程,
        避免在这里和 autopilot 抢着起同一个后端。
        """
        self._loaded_models.pop(model_name, None)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._config.omlx_bin,
                "stop",
                model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            _log.info("[ModelGateway] 已回收后端 %s", model_name)
        except Exception as e:
            _log.warning("[ModelGateway] 回收 %s 失败: %s", model_name, e)

    async def _omlx_real_model_id(self, base_url: str, port: int, model_name: str) -> str:
        """omlx 后端要的是权重全路径, 不是友好名。取一次缓存起来。"""
        import aiohttp

        cache_key = f"_omlx_mid_{model_name}"
        cached = getattr(self, cache_key, None)
        if cached:
            return cached
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{base_url}:{port}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    mid = (await r.json()).get("data", [{}])[0].get("id", model_name)
        except Exception:
            mid = model_name  # best-effort
        setattr(self, cache_key, mid)
        return mid

    async def _wait_healthy(
        self,
        model_name: str,
        base_url: str,
        port: int,
        timeout: float = 30.0,
    ) -> bool:
        """等待模型服务健康."""
        import aiohttp

        url = f"{base_url}:{port}/v1/models"
        deadline = time.time() + timeout

        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                await asyncio.sleep(1.0)

        return False

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型."""
        if model_name not in self._loaded_models:
            return True

        try:
            proc = await asyncio.create_subprocess_exec(
                self._config.omlx_bin,
                "unload",
                model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            self._loaded_models.pop(model_name, None)
            _log.info("[ModelGateway] %s unloaded", model_name)
            return True
        except Exception as e:
            _log.error("[ModelGateway] %s unload error: %s", model_name, e)
            return False

    async def warm_pool_sweep(self) -> None:
        """清理过期模型 (keep-last-used TTL)."""
        now = time.time()
        expired = [
            m for m, t in self._last_used.items() if now - t > self._config.warm_pool_ttl and m in self._loaded_models
        ]
        for model_name in expired:
            _log.info("[ModelGateway] warm pool expired: %s", model_name)
            await self.unload_model(model_name)

    # ==========================================================
    # 后台任务 (warm pool + health monitor)
    # ==========================================================
    async def start_background_tasks(self) -> None:
        """启动后台任务 (warm pool sweep + health check)."""
        if not self._config.background_tasks_enabled:
            return

        async def _warm_loop() -> None:
            while True:
                await asyncio.sleep(self._config.warm_pool_ttl / 5)  # TTL/5 频率检查
                try:
                    await self.warm_pool_sweep()
                except Exception:
                    _log.exception("[ModelGateway] warm pool sweep error")

        async def _health_loop() -> None:
            while True:
                await asyncio.sleep(self._config.health_check_interval)
                try:
                    await self._registry.refresh(self._config.registry_discover_timeout)
                    await self.health()
                except Exception:
                    _log.exception("[ModelGateway] health check error")

        loop = asyncio.get_event_loop()
        self._bg_tasks.append(loop.create_task(_warm_loop()))
        self._bg_tasks.append(loop.create_task(_health_loop()))
        _log.info("[ModelGateway] background tasks started (warm_pool + health)")

    async def stop_background_tasks(self) -> None:
        """停止后台任务."""
        for task in self._bg_tasks:
            task.cancel()
        self._bg_tasks.clear()
        close = getattr(self._omlxc, "aclose", None)
        if close is not None:
            await close()
        _log.info("[ModelGateway] background tasks stopped")

    # ==========================================================
    # Embedding (降级链)
    # ==========================================================
    async def embed(
        self,
        texts: list[str],
        model: str = "embedding",
        timeout: float = 30.0,
        *,
        routing_mode: str = "local",
        content_title: str = "",
        content_url: str = "",
    ) -> list[list[float]]:
        """Embed with the same legacy/shadow/active ownership as chat."""
        if routing_mode not in {"local", "hybrid", "cloud"}:
            raise ValueError(f"Invalid routing_mode: {routing_mode}")
        if self._config.omlxc_mode == "legacy":
            return await self._embed_legacy(texts, model, timeout, routing_mode=routing_mode)
        sensitive = bool((content_title or content_url) and _is_sensitive(content_title, content_url))
        if self._config.omlxc_mode == "shadow":
            if routing_mode != "cloud":
                resolved = self.resolve_alias(model)
                try:
                    await self._omlxc.route_plan(
                        resolved,
                        capabilities={"embedding"},
                        thinking=False,
                        timeout=min(timeout, 2.0),
                    )
                except OmlxcError as error:
                    _log.warning(
                        "omlxc shadow embedding route failed logical=%s resolved=%s code=%s",
                        model,
                        resolved,
                        error.code.value,
                    )
            return await self._embed_legacy(texts, model, timeout, routing_mode=routing_mode)
        if routing_mode == "cloud" and not sensitive:
            return await self._embed_legacy(texts, model, timeout, routing_mode="cloud")

        resolved = self.resolve_alias(model)
        try:
            return await self._omlxc.embed(model=resolved, inputs=texts, timeout=timeout, profile="interactive")
        except OmlxcError as error:
            if sensitive:
                raise RuntimeError("[K1] local embedding unavailable") from error
            if routing_mode == "hybrid" and error.cloud_fallback_allowed:
                return await self._embed_legacy(texts, model, timeout, routing_mode="cloud")
            raise

    async def _embed_legacy(
        self,
        texts: list[str],
        model: str = "embedding",
        timeout: float = 30.0,
        *,
        routing_mode: str = "local",
    ) -> list[list[float]]:
        """Embedding 降级链。

        App 模式复用 ComputeEngine SSOT：oMLX App → LM Link → Ollama；legacy
        回滚模式保留旧 8183/8188 行为。
        """
        import aiohttp

        if routing_mode == "cloud" or self._config.local_backend == "app":
            await self._ensure_registry_ready()
            local_key = self.resolve_alias(model)
            candidates: list[str] = []
            if routing_mode == "cloud":
                candidates.extend(
                    item.id
                    for item in self._registry.list_models()
                    if not item.id.partition("/")[0].startswith(("ENG-OMLX-", "ENG-LMSTUDIO-", "ENG-OLLAMA-"))
                    and _id_tail(item.id) == local_key
                )
            else:
                app_id = self._resolve_model_id(local_key, ("ENG-OMLX-LOCAL",))
                if app_id:
                    candidates.append(app_id)
                lm = self._lmstudio_fallback(local_key)
                if lm:
                    lm_id = self._resolve_model_id(lm, ("ENG-LMSTUDIO-",))
                    if lm_id:
                        candidates.append(lm_id)
                ollama = self._ollama_fallback(local_key)
                if ollama:
                    ollama_id = self._resolve_model_id(ollama, ("ENG-OLLAMA-",))
                    if ollama_id:
                        candidates.append(ollama_id)

            last_error = "no embedding provider"
            async with asyncio.timeout(max(0.1, timeout)):
                for model_id in dict.fromkeys(candidates):
                    if self._provider_is_self(model_id):
                        continue
                    provider = self._registry.get_provider(model_id)
                    base = str(getattr(provider, "base_url", "") or "").rstrip("/")
                    if not base:
                        continue
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"{base}/embeddings",
                                json={"input": texts, "model": _id_tail(model_id)},
                            ) as resp:
                                if resp.status != 200:
                                    last_error = f"{model_id} HTTP {resp.status}"
                                    continue
                                data = await resp.json()
                        embeddings = [item["embedding"] for item in data.get("data", [])]
                        if len(embeddings) == len(texts):
                            return embeddings
                        last_error = f"{model_id} returned {len(embeddings)}/{len(texts)} vectors"
                    except Exception as e:
                        last_error = f"{model_id}: {e}"
                        _log.warning("[ModelGateway] embed %s failed: %s", model_id, e)
            raise RuntimeError(f"All embedding providers failed. Last: {last_error}")

        chain = [
            ("embedding-8183", f"{self._config.local_base_url}:8183"),
            ("embedding-8188", f"{self._config.local_base_url}:8188"),
        ]

        for provider_name, base_url in chain:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/v1/embeddings",
                        json={"input": texts, "model": provider_name},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            embeddings = [item["embedding"] for item in data.get("data", [])]
                            if embeddings:
                                self._metrics.record_latency(provider_name, 0)
                                return embeddings
            except Exception as e:
                _log.warning("[ModelGateway] embed %s failed: %s", provider_name, e)
                continue

        raise RuntimeError("All embedding providers failed (8183, 8188)")

    # ==========================================================
    # 健康检查
    # ==========================================================
    async def health(self) -> dict[str, Any]:
        """全模型健康状态."""
        import aiohttp

        result = {}
        for model_name, port in self._config.model_ports.items():
            url = f"{self._config.local_base_url}:{port}/v1/models"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        result[model_name] = {
                            "status": "healthy" if resp.status == 200 else "unhealthy",
                            "port": port,
                            "loaded": model_name in self._loaded_models,
                            "consecutive_failures": self._health_failures.get(model_name, 0),
                        }
            except Exception:
                result[model_name] = {
                    "status": "unreachable",
                    "port": port,
                    "loaded": model_name in self._loaded_models,
                    "consecutive_failures": self._health_failures.get(model_name, 0),
                }
        return result

    # ==========================================================
    # 内部工具
    # ==========================================================
    @staticmethod
    def _extract_prompt(messages: list[dict[str, Any]]) -> str:
        """提取最后一条 user 消息作为复杂度评估输入."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def _provider_is_self(self, model_id: str) -> bool:
        """该模型的 provider 是否指向本网关自己的门面。"""
        try:
            provider = self._registry.get_provider(model_id)
        except Exception:  # 取不到 provider 就不做判定
            return False
        base = getattr(provider, "base_url", "") or ""
        return self._is_self_endpoint(str(base))

    def _is_self_endpoint(self, base_url: str) -> bool:
        """该 endpoint 是不是本网关自己的门面。

        指向自己时必须跳过, 否则回退路径会成环。宁可让这次调用失败并如实
        报错, 也不要在环里耗尽超时 —— 后者排查起来要难得多。
        """
        if not base_url:
            return False
        try:
            from urllib.parse import urlparse

            u = urlparse(base_url)
        except Exception:  # 解析不了就不当作自引用
            return False
        host = (u.hostname or "").lower()
        # S104 在此为误报: 这是把 0.0.0.0 归入"指向本机"的**比较**,
        # 不是把服务绑到所有网卡。门面若以 0.0.0.0 暴露, 自引用同样成立。
        loopback = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}  # noqa: S104
        if host not in loopback:
            return False
        return (u.port or 0) in self._config.self_facade_ports

    def resolve_alias(self, name: str) -> str:
        """把消费者的意图名展开成可路由的模型名。

        单一解析点 —— HTTP 门面与库入口都经由 ModelGateway, 因此都会命中这里。
        """
        from .aliases import resolve

        return resolve(name, self._config.aliases)

    def _routing_allows_model(self, model_name: str, mode: str) -> bool:
        """执行本地/云边界；未知模型在 local 模式按保守策略拒绝。"""
        if mode == "hybrid":
            return True
        resolved = self.resolve_alias(model_name)
        if resolved in self._config.model_ports:
            is_local = True
        else:
            model_id = self._resolve_model_id(resolved)
            engine = (model_id or resolved).partition("/")[0]
            is_local = engine.startswith(("ENG-OMLX-", "ENG-LMSTUDIO-", "ENG-OLLAMA-"))
        return is_local if mode == "local" else not is_local

    def _resolve_model_id(
        self,
        model_name: str,
        engine_prefixes: tuple[str, ...] | None = None,
    ) -> str | None:
        """解析模型名到 registry ID.

        2026-08-10 实测缺陷: 末尾曾是 `model_name.lower() in m.id.lower()` 的
        裸子串匹配, 会把意图名撞到任意名字里含该子串的模型上, 且悄无声息:
            reasoning → ENG-LMSTUDIO-MACMINI/qwen3-4b-...-reasoning-distilled
            embedding → ENG-CC-SWITCH/gemini-embedding-001   (本地请求上了云!)
        顶包出来的模型往往能返回 200, 错误一路沉到"回答变奇怪"才被发现。

        现在只认精确匹配(整名, 或去掉 ENGINE 前缀后的整名)。同一模型在
        多引擎上重复出现时(LM Link 池)按 _ENGINE_PREFERENCE 定序, 保证
        可复现 —— 原来的 matches[0] 取决于 registry 遍历顺序。
        """
        model_name = self.resolve_alias(model_name)
        reg = self._registry

        def _allowed(model_id: str) -> bool:
            return not engine_prefixes or model_id.partition("/")[0].startswith(engine_prefixes)

        if reg.get(model_name) and _allowed(model_name):
            return model_name
        # 2026-08-23: 曾经是手写静态列表(_ENGINE_PREFERENCE + 几个硬编码云端
        # 前缀), SSOT 新增引擎时必须记得同步更新, 否则那个引擎下的所有模型都
        # 会"not in registry"而不是走到下面的 tail 匹配兜底 —— 一次真实故障
        # 就是这么产生的(10 个云端引擎当时全部缺失)。改为动态枚举 registry 里
        # 实际注册的全部引擎: 本地引擎仍按 _ENGINE_PREFERENCE 定义的顺序优先
        # (保持既有的确定性排序语义), 其余(含云端/CC-SWITCH/未来任何新引擎)
        # 按 provider_names() 的字母序补在后面, 不需要再手工维护第二份清单。
        direct_engines = _ENGINE_PREFERENCE + tuple(
            name for name in reg.provider_names() if name not in _ENGINE_PREFERENCE
        )
        for engine in direct_engines:
            candidate = f"{engine}/{model_name}"
            if _allowed(candidate) and reg.get(candidate):
                return candidate

        # 本机 omlx key 有专属端口, 该走 ensure + 直连;
        # 让它落到别的引擎上就是顶包, 直接拒绝。
        if model_name in self._config.model_ports and not engine_prefixes:
            return None

        matches = [m.id for m in reg.list_models() if _id_tail(m.id) == model_name and _allowed(m.id)]
        if not matches:
            return None
        matches.sort(key=_engine_rank)
        return matches[0]

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def loaded_models(self) -> dict[str, float]:
        return dict(self._loaded_models)


# ============================================================
# 同步调用辅助 (解决 asyncio.run() 嵌套事件池问题)
# ============================================================
def run_async(coro):
    """安全地运行 async coroutine, 无论调用方是否在事件循环内.

    解决 asyncio.run() 在已有事件循环中调用会报
    "asyncio.run() cannot be called from a running event loop" 的问题.

    策略:
      - 无运行中的事件循环 → asyncio.run()
      - 有运行中的事件循环 → 在新线程中运行
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # 无事件循环, 直接用 asyncio.run
        return asyncio.run(coro)
    else:
        # 已有事件循环, 在新线程中运行
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(asyncio.run, coro)
            return future.result()


# ============================================================
# 单例 (全进程共享一个 gateway 实例)
# ============================================================
_gateway_singleton: ModelGateway | None = None


def get_gateway(config: GatewayConfig | None = None) -> ModelGateway:
    """获取全进程单例 gateway.

    首次调用时创建, 后续调用返回同一实例.
    所有消费方 (triage, sensitive_router, rpc) 共享一个 gateway,
    避免重复加载模型和端口冲突.

    用法:
        from llm_gateway import get_gateway
        gateway = get_gateway()
    """
    global _gateway_singleton
    if _gateway_singleton is None:
        _gateway_singleton = ModelGateway.create(config=config)
    return _gateway_singleton


def reset_gateway() -> None:
    """重置单例 (测试用)."""
    global _gateway_singleton
    _gateway_singleton = None
