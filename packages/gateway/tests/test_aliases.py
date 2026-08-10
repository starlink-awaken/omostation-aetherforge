"""别名层测试.

重点不在"映射对不对"(那是配置), 而在三条约束:
  1. 两个入口(HTTP 门面 / 库 bridge)必须给出**相同**的解析结果
  2. 坏配置不能让网关起不来 —— 只能降级
  3. 别名链不能把网关拖死(环 / 过长)
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_gateway import aliases

# ── 1. 基本解析 ────────────────────────────────────────────


def test_builtin_defaults_resolve():
    assert aliases.resolve("coder") == "coding"
    assert aliases.resolve("reasoner") == "reasoning"
    assert aliases.resolve("embed") == "embedding"


def test_unknown_name_passes_through():
    """未登记的名字必须原样返回 —— 别名层不是白名单, 不能吞掉真实模型 ID。"""
    assert aliases.resolve("mlx-community/Qwen3-32B") == "mlx-community/Qwen3-32B"
    assert aliases.resolve("") == ""


def test_shipped_yaml_loads_and_covers_litellm_names():
    """仓内默认配置应覆盖迁移基准里的意图名。"""
    table = aliases.load_aliases()
    for name in ("coder", "reasoner", "embed", "triage", "mini-9b", "fast", "mid", "ocr", "rerank", "vision-lite"):
        assert name in table, f"缺别名: {name}"


def test_ollama_names_route_to_lm_link_equivalent():
    """triage/mini-9b 原指向 ollama 的 qwen3.5:9b(不在 LM Link 池内),
    已决策由池内等价模型承接 —— 不应再出现 ollama 风格的 ':' 标签。"""
    table = aliases.load_aliases()
    for name in ("triage", "mini-9b"):
        assert ":" not in table[name], f"{name} 仍指向 ollama 风格标签"


# ── 2. 坏配置只能降级, 不能让网关起不来 ──────────────────


def test_broken_config_falls_back(tmp_path, monkeypatch):
    bad = tmp_path / "aliases.yaml"
    bad.write_text("aliases: [not, a, mapping]\n", encoding="utf-8")
    monkeypatch.setenv("AETHERFORGE_ALIASES", str(bad))
    table = aliases.load_aliases()
    assert table  # 降级到下一来源, 不是空表也不是异常


def test_unparseable_config_falls_back(tmp_path, monkeypatch):
    bad = tmp_path / "aliases.yaml"
    bad.write_text("aliases: {this: [is, broken\n", encoding="utf-8")
    monkeypatch.setenv("AETHERFORGE_ALIASES", str(bad))
    assert aliases.load_aliases()


def test_missing_config_falls_back(monkeypatch):
    monkeypatch.setenv("AETHERFORGE_ALIASES", "/nonexistent/aliases.yaml")
    assert aliases.load_aliases() == aliases.load_aliases()


# ── 3. 链: 支持但有界 ──────────────────────────────────────


def test_alias_chain_resolves():
    table = {"chat": "coder", "coder": "coding"}
    assert aliases.resolve("chat", table) == "coding"


def test_alias_cycle_does_not_hang():
    table = {"a": "b", "b": "a"}
    assert aliases.resolve("a", table) in {"a", "b"}


def test_self_reference_terminates():
    assert aliases.resolve("x", {"x": "x"}) == "x"


def test_overlong_chain_terminates():
    table = {f"n{i}": f"n{i + 1}" for i in range(50)}
    assert aliases.resolve("n0", table).startswith("n")


# ── 4. 硬约束: 两个入口结果一致 ──────────────────────────


def test_single_resolution_point_shared_by_both_entries():
    """HTTP 门面与库入口都经由 ModelGateway.resolve_alias。

    这条约束若破了, 会出现"库里能路由、HTTP 里路由不同"的双份真相 ——
    设计文档把它列为硬约束, 故在此固化。
    """
    from llm_gateway.gateway import get_gateway

    gw = get_gateway()
    for name in ("coder", "reasoner", "triage", "unknown-model-xyz"):
        assert gw.resolve_alias(name) == aliases.resolve(name, gw._config.aliases)


def test_gateway_config_loads_alias_table():
    from llm_gateway.gateway import GatewayConfig

    cfg = GatewayConfig()
    assert cfg.aliases, "GatewayConfig 未加载别名表"
    assert cfg.aliases.get("coder") == "coding"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
