"""Gemma-4 思考通道剥离 (2026-09-28: mini-chat 经门面拿到「思考文字<channel|>答案」)。"""

from __future__ import annotations

import pytest
from llm_gateway.gateway import strip_thinking


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # LM Studio 实际输出: 开头 <|channel> 已被吃掉
        ('The user wants to know the Chinese translation of "good morning".<channel|>早上好', "早上好"),
        ("用户要求一个早餐建议。<channel|>**推荐：** 燕麦粥配水果", "**推荐：** 燕麦粥配水果"),
        # 完整通道块
        ("<|channel>thought\n先想一下<channel|>北京是中国的首都。", "北京是中国的首都。"),
        # 无通道标记的普通回答不受影响
        ("北京是中国的首都。", "北京是中国的首都。"),
        ("<think>想</think>答案", "答案"),
    ],
)
def test_strip_gemma_channel(raw: str, expected: str) -> None:
    assert strip_thinking(raw) == expected
