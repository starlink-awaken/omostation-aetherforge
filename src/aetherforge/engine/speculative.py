"""投机解码引擎 — BET-Y1Q4-T6-28.

实现基于草稿模型的投机解码 (Speculative Decoding):
1. 草稿模型 (小/快) 生成 γ 个候选 token
2. 目标模型 (大/慢) 并行验证
3. 接受匹配的 token, 拒绝后从第一个分歧点重采样

性能目标: 相比纯目标模型, 吞吐提升 ≥2.5 倍.

参考: Chen et al. 2023 "Accelerating Large Language Model Decoding with Speculative Decoding".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class VerificationStatus(Enum):
    """验证状态."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIAL = "partial"


@dataclass
class SpeculativeResult:
    """投机解码结果."""

    tokens: list[str]
    accepted_count: int
    rejected_count: int
    draft_time_ms: float
    verify_time_ms: float
    total_time_ms: float
    tokens_per_second: float

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_count + self.rejected_count
        return self.accepted_count / total if total > 0 else 0.0


@dataclass
class SpeculativeConfig:
    """投机解码配置."""

    max_draft_tokens: int = 5  # 每次草稿的最大 token 数
    temperature: float = 0.7
    # 接受阈值: 草稿概率 / 目标概率 的比值
    acceptance_threshold: float = 0.8
    # 早停: 连续拒绝多少次后停止
    early_stop_rejects: int = 2


class SpeculativeEngine:
    """投机解码引擎.

    协调草稿模型和目标模型实现加速推理.
    """

    def __init__(
        self,
        config: SpeculativeConfig | None = None,
        draft_model: Callable[[list[str], int], list[tuple[str, float]]] | None = None,
        target_model: Callable[[list[str]], list[tuple[str, float]]] | None = None,
    ) -> None:
        self._config = config or SpeculativeConfig()
        # 默认使用 stub 模型 (实际部署时注入真实模型)
        self._draft_model = draft_model or self._default_draft
        self._target_model = target_model or self._default_target

    @property
    def config(self) -> SpeculativeConfig:
        return self._config

    def generate(
        self,
        prefix: list[str],
        max_tokens: int = 100,
    ) -> SpeculativeResult:
        """使用投机解码生成 token.

        Args:
            prefix: 已有的 token 序列 (上下文)
            max_tokens: 最大生成 token 数

        Returns:
            SpeculativeResult 包含生成的 token 和性能统计
        """
        total_start = time.monotonic()
        generated: list[str] = []
        total_accepted = 0
        total_rejected = 0
        total_draft_time = 0.0
        total_verify_time = 0.0
        consecutive_rejects = 0

        while len(generated) < max_tokens:
            # 1. 草稿阶段
            draft_start = time.monotonic()
            draft_tokens = self._draft_model(
                prefix + generated,
                min(self._config.max_draft_tokens, max_tokens - len(generated)),
            )
            draft_end = time.monotonic()
            draft_ms = (draft_end - draft_start) * 1000
            total_draft_time += draft_ms

            if not draft_tokens:
                break

            # 2. 验证阶段 — 目标模型并行评估
            verify_start = time.monotonic()
            target_probs = self._target_model(prefix + generated)
            verify_end = time.monotonic()
            verify_ms = (verify_end - verify_start) * 1000
            total_verify_time += verify_ms

            # 3. 接受/拒绝决策
            accepted, rejected = self._accept_reject(draft_tokens, target_probs)
            total_accepted += accepted
            total_rejected += rejected

            if accepted > 0:
                # 添加被接受的 token
                for i, (tok, _) in enumerate(draft_tokens[:accepted]):
                    if len(generated) < max_tokens:
                        generated.append(tok)
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                if consecutive_rejects >= self._config.early_stop_rejects:
                    # 早停: 回退到目标模型直接生成剩余部分
                    fallback = self._target_fallback(prefix + generated, max_tokens - len(generated))
                    generated.extend(fallback)
                    break

            # 如果有拒绝, 从第一个分歧点重采样
            if rejected > 0 and len(generated) < max_tokens:
                resampled = self._resample(draft_tokens, target_probs, accepted)
                if resampled:
                    generated.append(resampled)
                    total_accepted += 1

        total_time = (time.monotonic() - total_start) * 1000
        total_tokens = len(generated)
        tps = total_tokens / (total_time / 1000) if total_time > 0 else 0.0

        return SpeculativeResult(
            tokens=generated,
            accepted_count=total_accepted,
            rejected_count=total_rejected,
            draft_time_ms=total_draft_time,
            verify_time_ms=total_verify_time,
            total_time_ms=total_time,
            tokens_per_second=tps,
        )

    def _accept_reject(
        self,
        draft: list[tuple[str, float]],
        target: list[tuple[str, float]],
    ) -> tuple[int, int]:
        """执行接受/拒绝逻辑.

        对每个草稿 token:
        - 如果草稿 token 在目标分布中概率足够高 → 接受
        - 否则 → 拒绝, 停止

        Returns:
            (accepted_count, rejected_count)
        """
        accepted = 0
        target_dict = {tok: prob for tok, prob in target}

        for tok, draft_prob in draft:
            target_prob = target_dict.get(tok, 0.0)

            if target_prob > 0 and draft_prob <= target_prob:
                # 草稿概率 ≤ 目标概率 → 直接接受
                accepted += 1
            elif target_prob > 0 and draft_prob > 0:
                # 按概率阈值决定是否接受
                ratio = target_prob / draft_prob
                if ratio >= self._config.acceptance_threshold:
                    accepted += 1
                else:
                    break
            else:
                # token 不在目标分布中
                break

        rejected = len(draft) - accepted
        return accepted, rejected

    def _resample(
        self,
        draft: list[tuple[str, float]],
        target: list[tuple[str, float]],
        accepted_count: int,
    ) -> str | None:
        """从第一个分歧点重采样.

        使用目标模型的分布重新采样一个 token.
        """
        if not target:
            return None
        # 简单策略: 返回目标模型中概率最高的 token
        return max(target, key=lambda x: x[1])[0] if target else None

    def _target_fallback(self, prefix: list[str], count: int) -> list[str]:
        """目标模型兜底生成."""
        result = self._target_model(prefix, count) if callable(self._target_model) else []
        return [tok for tok, _ in result[:count]] if result else []

    @staticmethod
    def _default_draft(prefix: list[str], count: int) -> list[tuple[str, float]]:
        """默认占位草稿模型 — 实际部署时替换."""
        stub_tokens = [("the", 0.9), ("a", 0.8), ("is", 0.7), ("of", 0.6), ("and", 0.5)]
        return stub_tokens[:count]

    @staticmethod
    def _default_target(prefix: list[str], count: int = 1) -> list[tuple[str, float]]:
        """默认占位目标模型 — 实际部署时替换."""
        stub_tokens = [("the", 0.85), ("a", 0.75), ("is", 0.65), ("of", 0.55), ("and", 0.45)]
        return stub_tokens[:count]
