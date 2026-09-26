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
    assert aliases.resolve("coder") == "coding-next"
    assert aliases.resolve("reasoner") == "reasoning"
    assert aliases.resolve("embed") == "embedding"
    assert aliases.resolve("deepseek-v4-flash") == "qwen-3.5-9b-flash"
    assert aliases.resolve("deepseek-v4-pro") == "qwen-3.5-9b-pro"


def test_unknown_name_passes_through():
    """未登记的名字必须原样返回 —— 别名层不是白名单, 不能吞掉真实模型 ID。"""
    assert aliases.resolve("mlx-community/Qwen3-32B") == "mlx-community/Qwen3-32B"
    assert aliases.resolve("") == ""


def test_shipped_yaml_loads_and_covers_litellm_names():
    """仓内默认配置应覆盖迁移基准里的意图名。"""
    table = aliases.load_aliases()
    for name in (
        "coder",
        "reasoner",
        "embed",
        "triage",
        "mini-9b",
        "fast",
        "mid",
        "general",
        "vision-mid",
        "ocr",
        "rerank",
        "vision-lite",
        "qwen38-27b",
    ):
        assert name in table, f"缺别名: {name}"
    assert table["qwen38-27b"] == "qwen3.8-27b-splash"


def test_unservable_aliases_are_not_shipped():
    """deepseek-v4-* 在所有运行时都推理不了(2026-09-25 实测), 登记了就只会静默兜底
    到别的模型 —— 宁可未知名走 fallback 链, 也不能伪装成已支持。"""
    table = aliases.load_aliases()
    for name in ("deepseek-v4-flash", "deepseek-v4-pro"):
        assert name not in table


def test_ollama_names_route_through_logical_model_chain():
    """triage/mini-9b 不应绕开统一降级链直指 Ollama 标签。"""
    table = aliases.load_aliases()
    for name in ("triage", "mini-9b"):
        assert ":" not in table[name], f"{name} 仍指向 ollama 风格标签"


def test_common_intents_route_to_measured_tiers():
    """高频意图落到 2026-09-25 实测档: Splash(LM Studio) 为主力, oMLX 保留可点名/兜底。"""
    table = aliases.load_aliases()
    assert table["opus"] == "qwen3.8-27b-splash"
    assert table["sonnet"] == table["haiku"] == table["coder"] == "qwen3.6-35b-a3b-splash"
    assert table["triage"] == table["fast"] == "qwen3.6-35b-a3b-splash"
    assert table["mini-9b"] == "mythos-fast"
    assert table["mid"] == table["general"] == "qwen3.8-27b-splash"
    assert table["coder-next"] == "coding-next"
    assert table["vision"] == table["vision-mid"] == "qwen3.6-35b-a3b-splash"


def test_directly_named_omlx_keys_are_shadowed_to_working_tiers():
    """调用方直呼 reasoning/coding/mid-local(工作区百余处): oMLX 对应物推理报错/无投影/极慢,
    别名层必须接住, 否则网关 fallback_chain 里的同名项也跟着失效。"""
    table = aliases.load_aliases()
    assert table["reasoning"] == "qwen3.8-27b-splash"
    assert table["coding"] == table["mid-local"] == "qwen3.6-35b-a3b-splash"


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
    assert cfg.aliases.get("coder") == "qwen3.6-35b-a3b-splash"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 5. 自引用防护 ──────────────────────────────────────────


def test_self_endpoint_detection():
    """SSOT 的 ENG-OMLX-LOCAL 现指向本网关门面(原先是 LiteLLM :4000),
    回退路径必须能识别"打回自己"并拒绝, 否则会成环。"""
    from llm_gateway.gateway import get_gateway

    gw = get_gateway()
    for url in (
        "http://127.0.0.1:9290/v1",
        "http://localhost:9290/v1",
        "http://0.0.0.0:9290/v1",
    ):
        assert gw._is_self_endpoint(url), f"未识别为自引用: {url}"


def test_non_self_endpoints_not_flagged():
    """别的端点不能被误判 —— 误判会让正常的云端/远端路由失效。"""
    from llm_gateway.gateway import get_gateway

    gw = get_gateway()
    for url in (
        "http://127.0.0.1:1234/v1",  # LM Link
        "http://127.0.0.1:8082/v1",  # omlx 直连端口
        "https://api.deepseek.com/v1",  # 云端
        "http://100.99.210.78:1234/v1",  # 远端节点
        "",
        "not-a-url",
    ):
        assert not gw._is_self_endpoint(url), f"误判为自引用: {url}"


def test_media_aliases_point_at_served_backends():
    """语音别名落 oMLX 实测可用的音频模型; 中文 TTS 不能指 Kokoro(缺 misaki[zh])。"""
    table = aliases.load_aliases()
    assert table["tts"] == table["tts-zh"] == "tts-qwen3"
    assert table["tts-en"] == "tts-kokoro"
    assert table["asr"] == table["whisper"] == "asr-whisper"
