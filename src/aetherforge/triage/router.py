"""Triage router — 通过 ModelGateway 统一分诊调用.

支持:
- 多模型 fallback: coding-fast → mid-local → deepseek-cloud (由 gateway 自动处理)
- thinking 剥离 (网关层兜底)
- K1 敏感流硬拦 (网关层)
- 统一记账 (TriageTracker)

架构变更 (2026-07-30):
- 旧: 直接 HTTP 调 omlx gateway (硬编码 IP/端口)
- 新: 通过 ModelGateway 类 (自动 load/unload + fallback + MemoryGuard)
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# 补齐 aetherforge / gateway 包路径 (同 rpc.py)
_af_dir = Path(__file__).resolve().parents[3]
# aetherforge 包 (for aetherforge._paths)
_src_p = str(_af_dir / "src")
if _src_p not in sys.path:
    sys.path.insert(0, _src_p)
# gateway / mesh 子包
for _sub in ("gateway", "mesh"):
    _p = str(_af_dir / "packages" / _sub / "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_gateway.gateway import GatewayRequest, ModelGateway, run_async

if TYPE_CHECKING:
    from .tracker import TriageTracker

TRIAGE_PROMPT = """你是信息分诊助手。给定一条信息，判断该 丢弃 / 沉淀 / 提醒 三选一。

判定标准:
- 丢弃: 营销/广告/促销优惠/社交动态/APP推送/星座运势/物流通知/续费推销
  特征: "限时优惠""点击领取""猜你喜欢""新视频""热搜""新人红包""续费享优惠"
- 沉淀: 技术文章/规范文档/知识教程/研究分析/学习资料/笔记同步/课程更新
  特征: 包含具体知识内容(技术/架构/论文/读书笔记), 即使是"同步通知"只要内容有价值就选沉淀
  注意: "已同步到笔记""新课上线""读书笔记"这类知识类通知属于沉淀, 不是噪音
- 提醒: 会议/截止日期/账单/告警/待办/预约/审批/需用户行动的事项
  特征: 有明确时间、地点、行动要求, 不包含优惠促销内容

示例:
【淘宝】双十一全场5折 → 丢弃
【GitHub】新PR: feat: add auth module → 沉淀
【日历】明天10点开会 → 提醒
【微信读书】《置身事内》读书笔记已同步 → 沉淀
【极客时间】新课上线：架构师训练营第12讲 → 沉淀
【WPS笔记】笔记「架构设计」已同步到云端 → 沉淀
【京东】PLUS会员到期，续费享优惠 → 丢弃
【银行】信用卡账单￥3000，还款日明天 → 提醒

信息: {text}
只输出一个词 (丢弃/沉淀/提醒):"""

# fallback 链: 空 = 使用 gateway 默认 fallback_chain (coding-fast → mid-local → deepseek-chat)
# 不指定模型名, 让 gateway 自动选择最优
TRIAGE_CHAIN: list[str] = []

# 共识模型 (3 个独立模型并行校验, 2/3 投票)
CONSENSUS_MODELS = [
    ("mid-local", True),       # Qwen3.6-27B, 100%, 1.13s
    ("mini-9b", True),         # qwen3.5:9b, 100%, 1.42s
    ("deepseek-chat", False),  # 云端, 95%, 0.66s
]


@dataclass
class TriageResult:
    """单条分诊结果."""
    verdict: str  # 丢弃/沉淀/提醒
    model: str  # 实际使用的模型
    latency: float  # 延迟秒
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class ConsensusResult:
    """共识分诊结果 (多模型投票)."""
    verdict: str  # 最终判定
    votes: dict[str, int]  # 各判定票数 {"丢弃": 2, "沉淀": 1}
    agreement: float  # 一致率 (0-1)
    status: str  # "共识" / "多数" / "分歧"
    latency: float  # 并行延迟 (取最慢)
    details: list[TriageResult]  # 各模型详情
    cost_usd: float = 0.0


@dataclass
class TriageRouter:
    """分诊路由器 — 通过 ModelGateway 调用, 自动 fallback + 敏感硬拦."""
    gateway: Optional[ModelGateway] = None
    model_chain: list = field(default_factory=lambda: list(TRIAGE_CHAIN))
    tracker: Optional["TriageTracker"] = None

    def triage_one(self, text: str, title: str = "", url: str = "") -> TriageResult:
        """分诊单条信息, 自动 fallback.

        Args:
            text: 信息内容
            title: 标题 (用于 K1 敏感检查)
            url: URL (用于 K1 敏感检查)
        """
        if self.gateway is None:
            return TriageResult(
                verdict="错误", model="", latency=0,
                error="ModelGateway not initialized",
            )

        # 如果配置了 model_chain, 逐个尝试
        last_result = TriageResult(verdict="错误", model="", latency=0, error="no chain")
        if self.model_chain:
            for model in self.model_chain:
                last_result = self._call_model(text, model, title, url)
                if last_result.error is None and last_result.verdict in ("丢弃", "沉淀", "提醒"):
                    if self.tracker:
                        self.tracker.record(last_result)
                    return last_result
            # all failed
            if self.tracker:
                self.tracker.record(last_result)
            return last_result

        # 使用 gateway 默认 fallback
        last_result = self._call_with_gateway(text, title, url)
        if self.tracker:
            self.tracker.record(last_result)
        return last_result

    def triage_batch(self, texts: list[str]) -> list[TriageResult]:
        """批量分诊."""
        return [self.triage_one(t) for t in texts]

    def consensus_triage(
        self, text: str, title: str = "", url: str = "",
        models: Optional[list[tuple[str, bool]]] = None,
        min_agreement: float = 2 / 3,
    ) -> ConsensusResult:
        """多模型共识分诊 — 并行调用多个模型, 投票决定.

        Args:
            text: 信息内容
            title: 标题
            url: URL
            models: [(模型名, 是否需要reasoning_effort=none), ...], 默认 CONSENSUS_MODELS
            min_agreement: 最低一致率 (默认 2/3)
        """
        if models is None:
            models = CONSENSUS_MODELS

        # 并行调用
        details: list[TriageResult] = []
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {
                pool.submit(self._call_model, text, model, title, url): model
                for model, _ in models
            }
            for future in as_completed(futures):
                try:
                    details.append(future.result())
                except Exception as e:
                    model = futures[future]
                    details.append(TriageResult(
                        verdict="错误", model=model, latency=0, error=str(e)[:30]
                    ))

        # 投票
        votes: dict[str, int] = {}
        for r in details:
            if r.verdict in ("丢弃", "沉淀", "提醒"):
                votes[r.verdict] = votes.get(r.verdict, 0) + 1

        total = sum(votes.values())
        max_verdict = max(votes, key=votes.get) if votes else "未知"
        max_count = max(votes.values()) if votes else 0
        agreement = max_count / len(models) if models else 0

        if agreement >= 1.0:
            status = "共识"
        elif agreement >= min_agreement:
            status = "多数"
        else:
            status = "分歧"

        max_latency = max((r.latency for r in details), default=0)
        total_cost = sum(r.cost_usd for r in details)

        result = ConsensusResult(
            verdict=max_verdict,
            votes=votes,
            agreement=agreement,
            status=status,
            latency=max_latency,
            details=details,
            cost_usd=total_cost,
        )

        # 记账
        if self.tracker:
            self.tracker.record(TriageResult(
                verdict=max_verdict,
                model=f"consensus({','.join(m[0] for m in models)})",
                latency=max_latency,
                cost_usd=total_cost,
            ))

        return result

    def _call_model(self, text: str, model: str, title: str, url: str) -> TriageResult:
        """通过 gateway 调用指定模型."""
        return self._call_with_gateway(text, title, url, model)

    def _call_with_gateway(
        self, text: str, title: str, url: str, model: str = "",
    ) -> TriageResult:
        """通过 ModelGateway 调用."""
        if self.gateway is None:
            return TriageResult(verdict="错误", model=model, latency=0, error="gateway is None")

        try:
            req = GatewayRequest(
                messages=[{"role": "user", "content": TRIAGE_PROMPT.format(text=text)}],
                model=model,
                task="triage",
                content_title=title,
                content_url=url,
            )

            resp = run_async(self.gateway.generate(req))

            latency = resp.latency_ms / 1000

            if resp.error:
                return TriageResult(
                    verdict="错误",
                    model=resp.model or model,
                    latency=latency,
                    error=resp.error,
                )

            # 解析 verdict
            verdict = None
            for v in ("丢弃", "沉淀", "提醒"):
                if v in resp.content:
                    verdict = v
                    break

            if verdict is None:
                return TriageResult(
                    verdict=f"未知({resp.content[:20]})",
                    model=resp.model,
                    latency=latency,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    error=f"无法解析: {resp.content[:30]}",
                )

            return TriageResult(
                verdict=verdict,
                model=resp.model,
                latency=latency,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cost_usd=resp.cost_usd,
            )
        except PermissionError as e:
            # K1 硬拦
            return TriageResult(
                verdict="敏感",
                model="",
                latency=0,
                error=f"[K1] {e}",
            )
        except Exception as e:
            return TriageResult(
                verdict="错误",
                model=model,
                latency=0,
                error=str(e)[:50],
            )
