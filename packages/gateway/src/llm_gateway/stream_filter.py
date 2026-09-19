"""StreamThinkingFilter — 流式思考链 (<think>...</think>) 拦截与结构化分流状态机.

架构职责:
  1. 解决模型流式输出时 <think> 标签直接混入 content 的代际断层问题;
  2. 支持跨 chunk 分裂标签处理 (如 chunk1: "<thi", chunk2: "nk>reasoning");
  3. 支持双模分流:
     - emit_reasoning=True: 思考段内容重定向至 chunk.reasoning_content (content 置空);
     - emit_reasoning=False: 思考段内容丢弃, 仅透传正常 content (保护 Agent 工具调用上下文);
  4. 对非思考文本零开销/快速透传.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .omlxc_client import OmlxcStreamChunk

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tag_length(value: str, tag: str) -> int:
    lowered = value.lower()
    for length in range(min(len(value), len(tag) - 1), 0, -1):
        if lowered[-length:] == tag[:length]:
            return length
    return 0


class StreamThinkingFilter:
    """Stateful stream filter for <think>...</think> blocks."""

    def __init__(self, *, emit_reasoning: bool = False) -> None:
        self.emit_reasoning = emit_reasoning
        self._buffer = ""
        self._in_think = False
        self._saw_reasoning = False

    @property
    def in_think(self) -> bool:
        return self._in_think

    @property
    def saw_reasoning(self) -> bool:
        return self._saw_reasoning

    def process_chunk(self, chunk: OmlxcStreamChunk) -> list[OmlxcStreamChunk]:
        """处理输入的单个 chunk，可能产生 0 个、1 个或多个分流后的 chunk。"""
        if not chunk.content:
            return [chunk]

        self._buffer += chunk.content
        out_chunks: list[OmlxcStreamChunk] = []

        while self._buffer:
            lowered = self._buffer.lower()

            if self._in_think:
                # 处于 <think> 内部，寻找 </think>
                close_at = lowered.find(_THINK_CLOSE)
                if close_at >= 0:
                    think_text = self._buffer[:close_at]
                    if think_text and self.emit_reasoning:
                        out_chunks.append(replace(chunk, content="", reasoning_content=think_text))
                    self._buffer = self._buffer[close_at + len(_THINK_CLOSE) :]
                    self._in_think = False
                    continue

                # 没找到闭合标签，检查尾部是否是 </think> 的前缀
                keep = _partial_tag_length(self._buffer, _THINK_CLOSE)
                if keep > 0:
                    think_text = self._buffer[:-keep]
                    self._buffer = self._buffer[-keep:]
                else:
                    think_text = self._buffer
                    self._buffer = ""

                if think_text and self.emit_reasoning:
                    out_chunks.append(replace(chunk, content="", reasoning_content=think_text))
                break

            else:
                # 处于思考块外部，寻找 <think>
                open_at = lowered.find(_THINK_OPEN)
                if open_at >= 0:
                    normal_text = self._buffer[:open_at]
                    if normal_text:
                        out_chunks.append(replace(chunk, content=normal_text, reasoning_content=""))
                    self._buffer = self._buffer[open_at + len(_THINK_OPEN) :]
                    self._in_think = True
                    self._saw_reasoning = True
                    continue

                # 没找到开放标签，检查尾部是否是 <think> 的前缀
                keep = _partial_tag_length(self._buffer, _THINK_OPEN)
                if keep > 0:
                    normal_text = self._buffer[:-keep]
                    self._buffer = self._buffer[-keep:]
                else:
                    normal_text = self._buffer
                    self._buffer = ""

                if normal_text:
                    out_chunks.append(replace(chunk, content=normal_text, reasoning_content=""))
                break

        return out_chunks

    def finish(self, last_chunk: OmlxcStreamChunk | None = None) -> list[OmlxcStreamChunk]:
        """流结束时冲刷残余 buffer。"""
        out: list[OmlxcStreamChunk] = []
        if self._buffer:
            remaining = self._buffer
            self._buffer = ""
            if self._in_think:
                if self.emit_reasoning and last_chunk is not None:
                    out.append(replace(last_chunk, content="", reasoning_content=remaining))
            else:
                if last_chunk is not None:
                    out.append(replace(last_chunk, content=remaining, reasoning_content=""))
        self._in_think = False
        return out
