"""Unit tests for StreamThinkingFilter."""

from __future__ import annotations

from llm_gateway.omlxc_client import OmlxcStreamChunk
from llm_gateway.stream_filter import StreamThinkingFilter


def test_stream_filter_pure_content():
    flt = StreamThinkingFilter(emit_reasoning=False)
    chunks = [
        OmlxcStreamChunk(content="Hello "),
        OmlxcStreamChunk(content="world!"),
    ]
    out = []
    for c in chunks:
        out.extend(flt.process_chunk(c))
    out.extend(flt.finish(chunks[-1]))

    result = "".join(c.content for c in out)
    assert result == "Hello world!"
    assert all(c.reasoning_content == "" for c in out)
    assert not flt.saw_reasoning


def test_stream_filter_strip_thinking():
    flt = StreamThinkingFilter(emit_reasoning=False)
    chunks = [
        OmlxcStreamChunk(content="Answer: "),
        OmlxcStreamChunk(content="<think>Step 1: calculate 1+1=2</think>"),
        OmlxcStreamChunk(content="It is 2."),
    ]
    out = []
    for c in chunks:
        out.extend(flt.process_chunk(c))
    out.extend(flt.finish(chunks[-1]))

    result = "".join(c.content for c in out)
    assert result == "Answer: It is 2."
    assert all(c.reasoning_content == "" for c in out)
    assert flt.saw_reasoning


def test_stream_filter_emit_reasoning():
    flt = StreamThinkingFilter(emit_reasoning=True)
    chunks = [
        OmlxcStreamChunk(content="<think>Deep reflection here</think>Final answer"),
    ]
    out = []
    for c in chunks:
        out.extend(flt.process_chunk(c))
    out.extend(flt.finish(chunks[-1]))

    content = "".join(c.content for c in out)
    reasoning = "".join(c.reasoning_content for c in out)
    assert content == "Final answer"
    assert reasoning == "Deep reflection here"
    assert flt.saw_reasoning


def test_stream_filter_split_across_chunks():
    flt = StreamThinkingFilter(emit_reasoning=True)
    chunks = [
        OmlxcStreamChunk(content="Greeting <thi"),
        OmlxcStreamChunk(content="nk>thinking "),
        OmlxcStreamChunk(content="about things</th"),
        OmlxcStreamChunk(content="ink> done!"),
    ]
    out = []
    for c in chunks:
        out.extend(flt.process_chunk(c))
    out.extend(flt.finish(chunks[-1]))

    content = "".join(c.content for c in out)
    reasoning = "".join(c.reasoning_content for c in out)
    assert content == "Greeting  done!"
    assert reasoning == "thinking about things"
    assert flt.saw_reasoning
