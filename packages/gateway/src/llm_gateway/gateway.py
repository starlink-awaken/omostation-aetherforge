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
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .complexity import TaskComplexityScorer
from .metrics import MetricsCollector
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


# ============================================================
# omlx 动态对接 (SSOT: /Volumes/Model/omlx/conf/models.json)
# 避免硬编码端口漂移 —— omlx 改端口/加模型后自动同步
# ============================================================
OMLX_CONF = "/Volumes/Model/omlx/conf/models.json"


# 网关别名 → omlx 本地 key(上层习惯用别名, omlx 后端用 key)
def _load_aliases() -> dict[str, str]:
    """加载别名表(配置优先, 失败回退内置)。"""
    from .aliases import load_aliases

    return load_aliases()


# 兼容保留: 早期硬编码别名。新增别名请改 aliases.yaml, 不要动这里。
OMLX_ALIAS_MAP: dict[str, str] = {
    "coder": "coding",
    "coder-fast": "coding-fast",
    "reasoner": "reasoning",
    "reasoner-lite": "reasoning-lite",
    "embed": "embedding",
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
    """粗略模型大小(GB) — MemoryGuard 用。按 alias 名启发式, 未知则不设(跳过检查)。"""
    hints = {
        "coding-fast": 18.0,
        "coding": 13.0,
        "reasoning": 10.0,
        "reasoning-lite": 18.0,
        "mid-local": 15.0,
        "coder-precise": 28.0,
        "mythos-fast": 5.0,
        "mythos": 18.0,
        "mistral-medium-128b": 64.0,
        "deepseek-v4-pro": 40.0,
        "deepseek-v4-flash": 20.0,
    }
    out = dict(hints)
    for alias, key in OMLX_ALIAS_MAP.items():
        if key in hints:
            out[alias] = hints[key]
    return out


@dataclass
class GatewayConfig:
    """网关配置."""

    # omlx CLI 路径
    omlx_bin: str = "/Volumes/Model/omlx/bin/omlx"
    # 本地模型基础 URL
    # omlx 后端只绑 loopback(:4000 网关才绑 tailscale IP), 直连必须用 127.0.0.1
    local_base_url: str = "http://127.0.0.1"
    # 模型端口映射 (model_name → port)
    model_ports: dict[str, int] = field(default_factory=_load_omlx_ports)
    # 模型大小 (GB) — MemoryGuard 用 (未知大小的模型跳过检查)
    model_sizes: dict[str, float] = field(default_factory=_load_omlx_sizes)
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
    # 是否启用后台任务 (warm pool sweep + health check)
    background_tasks_enabled: bool = True


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
    ):
        self._registry = registry
        self._scheduler = scheduler
        self._config = config or GatewayConfig()
        self._metrics = metrics or MetricsCollector()
        self._memory_guard = MemoryGuard(self._config.memory_safety_factor)
        self._complexity_scorer = TaskComplexityScorer()

        # 已加载模型集合 (model_name → load_time)
        self._loaded_models: dict[str, float] = {}
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
                await self._registry.refresh()
            self._registry_ready = True  # only set on success
        except Exception as e:
            _log.warning("[ModelGateway] registry refresh failed: %s", e)
            # Do NOT set _registry_ready — allow retry on next call

    async def generate(self, request: GatewayRequest) -> GatewayResponse:
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
                if sched_model not in full_chain:
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

    async def _generate_local_only(self, request: GatewayRequest) -> GatewayResponse:
        """敏感流: 只用本地模型, 不 fallback 到云端."""
        t0 = time.time()

        # 只尝试本地模型
        local_models = [m for m in self._config.fallback_chain if m != "deepseek-chat"]
        chain = [request.model] if request.model and request.model != "deepseek-chat" else []
        chain.extend(local_models)

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

        Strategy:
          - Local omlx models (in model_ports): try direct port first, fall back to registry
          - Cloud models: registry + provider chain only
        """
        t0 = time.time()

        # Local omlx models: try direct port routing (quick check first)
        if model_name in self._config.model_ports:
            port = self._config.model_ports[model_name]
            if await self._port_reachable(port):
                try:
                    return await self._generate_via_omlx_router(model_name, request, t0)
                except Exception as e:
                    _log.debug("[ModelGateway] direct port %s failed, trying registry: %s", model_name, e)

        # Registry + provider chain (cloud models + fallback for local)
        model_id = self._resolve_model_id(model_name)
        if model_id:
            result = await asyncio.wait_for(
                self._registry.chat(
                    model_id,
                    request.messages,
                    ChatOptions(
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ),
                ),
                timeout=request.timeout,
            )
            if not result:
                raise RuntimeError(f"No response from {model_name}")

            content = result.content or ""
            stripped = _strip_thinking(content)
            was_stripped = stripped != content
            usage = result.usage or {}
            provider = self._registry.get_provider(model_id)
            provider_name = provider.name if provider else ""

            return GatewayResponse(
                content=stripped,
                model=model_name,
                latency_ms=(time.time() - t0) * 1000,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                provider=provider_name,
                stripped_thinking=was_stripped,
            )

        # Registry can't resolve — try omlx :9000 router for local models
        if model_name in self._config.model_ports:
            return await self._generate_via_omlx_router(model_name, request, t0)

        raise RuntimeError(f"Model {model_name} not in registry")

    async def _generate_via_omlx_router(self, model_name: str, request: GatewayRequest, t0: float) -> GatewayResponse:
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

        # Cache the real model ID (omlx needs full HF path, not friendly name)
        cache_key = f"_omlx_mid_{model_name}"
        real_model_id = getattr(self, cache_key, None)
        if not real_model_id:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{base}:{port}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        models_data = await resp.json()
                        real_model_id = models_data.get("data", [{}])[0].get("id", model_name)
                        setattr(self, cache_key, real_model_id)
            except Exception:
                real_model_id = model_name  # best-effort

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

        usage = data.get("usage", {})
        latency_ms = (time.time() - t0) * 1000

        return GatewayResponse(
            content=stripped,
            model=model_name,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            provider="ENG-OMLX-LOCAL",
            stripped_thinking=was_stripped,
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
        if model_name in self._loaded_models:
            return True

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
                _, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=120,
                )

                if proc.returncode != 0:
                    _log.warning(
                        "[ModelGateway] omlxc load %s failed: %s",
                        model_name,
                        stderr.decode()[:200],
                    )
                    return False

                # 等待服务就绪
                if await self._wait_healthy(model_name, base_url, port):
                    self._loaded_models[model_name] = time.time()
                    _log.info("[ModelGateway] %s loaded on port %d", model_name, port)
                    return True
                else:
                    _log.warning("[ModelGateway] %s load timeout", model_name)
                    return False

            except TimeoutError:
                _log.warning("[ModelGateway] %s load timed out", model_name)
                return False
            except Exception as e:
                _log.error("[ModelGateway] %s load error: %s", model_name, e)
                return False

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
        _log.info("[ModelGateway] background tasks stopped")

    # ==========================================================
    # Embedding (降级链)
    # ==========================================================
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding 降级链: 8183 → 8188 → 错误."""
        import aiohttp

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

    def resolve_alias(self, name: str) -> str:
        """把消费者的意图名展开成可路由的模型名。

        单一解析点 —— HTTP 门面与库入口都经由 ModelGateway, 因此都会命中这里。
        """
        from .aliases import resolve

        return resolve(name, self._config.aliases)

    def _resolve_model_id(self, model_name: str) -> str | None:
        """解析模型名到 registry ID."""
        model_name = self.resolve_alias(model_name)
        reg = self._registry
        if reg.get(model_name):
            return model_name
        for engine in ("ENG-OMLX-LOCAL", "ENG-CC-SWITCH"):
            if reg.get(f"{engine}/{model_name}"):
                return f"{engine}/{model_name}"
        matches = [m for m in reg.list_models() if model_name.lower() in m.id.lower()]
        return matches[0].id if matches else None

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
