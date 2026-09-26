#!/usr/bin/env python3
"""sensitive_router: 敏感流硬拦截路由 + 全沉淀策略 (K1/K2, 代码层硬拦, 不靠约定).

K1 策略: 敏感流 → 全沉淀 + embed 索引入 KOS, 不再尝试 LLM 分类.
K2 策略: "提醒"改规则 — 发件人白名单(领导/上级单位) + 关键词(截止/报送/督办/反馈).
K3: omlx 死亡模型登记为已知状态, 不修.
K4: Gmail 接源待用户凭据, 当前唯一阻塞.

敏感数据不出机. 必须是硬的.

架构 (2026-07-31):
- strip_thinking / is_sensitive: import 自 llm_gateway.gateway (SSOT)
- classify_sensitivity: 本模块的公开域名白名单逻辑 (在 gateway.is_sensitive 基础上加白名单)
- route / is_reminder / hard_block_external: 本模块核心决策
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# 补齐 aetherforge / gateway 包路径
# __file__ = .../aetherforge/tools/sensitive_router.py
# parents[0]=tools, parents[1]=aetherforge ← 项目根
_af_dir = Path(__file__).resolve().parents[1]
# aetherforge 包 (for aetherforge._paths)
_src_p = str(_af_dir / "src")
if _src_p not in sys.path:
    sys.path.insert(0, _src_p)
# gateway 子包
_gw_p = str(_af_dir / "packages" / "gateway" / "src")
if _gw_p not in sys.path:
    sys.path.insert(0, _gw_p)

from llm_gateway.gateway import is_sensitive as _gateway_is_sensitive
from llm_gateway.gateway import run_async, strip_thinking

from aetherforge.endpoint import gateway_key, gateway_url

# 导出 strip_thinking 供外部直接使用
__all__ = ["strip_thinking", "classify_sensitivity", "is_reminder", "route", "hard_block_external", "embed_texts"]

# ============================================================
# 公开流域名 (允许外部 API)
# ============================================================
# 注意: docs.google.com (工作文档) 不在此列, 已加入敏感模式
# 注意: mail.google.com (Gmail) 已在敏感模式中
PUBLIC_DOMAINS = {
    "zhihu.com",
    "weibo.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "stackoverflow.com",
    "github.com",
    "docs.python.org",
    "developer.mozilla.org",
    "wikipedia.org",
    "baidu.com",
    "bing.com",
    "jd.com",
    "taobao.com",
    "tmall.com",
    "amazon.com",
    "bilibili.com",
    "douyin.com",
    "youtube.com",
    # Google 公开服务 (搜索/地图/翻译等, 非邮件/文档)
    "google.com",
    "google.com.hk",
    "google.cn",
    "googleapis.com",
    "gstatic.com",
    "notifications.googleapis.com",
}

# ============================================================
# K2: 提醒规则 (发件人白名单 + 关键词)
# ============================================================
# 发件人白名单: 领导/上级单位域名或地址关键词
REMINDER_SENDER_WHITELIST = re.compile(
    r"(卫健委|卫建|卫健委|区政府|市政府|区政府|县委|市委|省委"
    r"|领导|上级|主管|督导|巡视|巡查"
    r"|@gov\.cn|@.*\.gov\.cn)"
)

# 提醒关键词
REMINDER_KEYWORDS = re.compile(
    r"(截止|截至|限期|逾期|督办|催办|反馈|报送|上报|汇报|提交|办理"
    r"|紧急|加急|即刻|立即|尽快|务必|请予|请尽快|请按时"
    r"|deadline|due date|action required|please respond)"
)


def classify_sensitivity(title: str, url: str) -> str:
    """判信息敏感度: sensitive / public.

    sensitive → 全沉淀 + embed 入 KOS (K1 策略).
    public → 允许 DeepSeek 云.

    逻辑:
      1. 先调 gateway.is_sensitive (SSOT: 域名/IP/关键词)
      2. 再叠加公开域名白名单 (gateway 没有的维度)
    """
    if not title and not url:
        return "sensitive"  # 空 = 保守判敏感

    # 1. 网关层 SSOT 敏感判断
    if _gateway_is_sensitive(title, url):
        return "sensitive"

    # 2. 公开域名白名单 (精确匹配 or 父域名匹配)
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if domain in PUBLIC_DOMAINS:
        return "public"
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in PUBLIC_DOMAINS:
            return "public"

    # 3. 默认保守 = 敏感 (宁可错杀, 不可泄露)
    return "sensitive"


# OA 系统域名模式 (这些域名下的公文默认需要提醒)
# 注意: 内网 IP 用协议无关模式 (匹配 http://10.x 和裸 IP)
OA_SYSTEM_DOMAINS = re.compile(
    r"(seeyon|致远|泛微|蓝凌|通达"  # OA 软件名
    r"|(?:^|://)10\.\d+\.\d+\.\d+"  # 内网 A 类
    r"|(?:^|://)172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+"  # 内网 B 类
    r"|(?:^|://)192\.168\.\d+\.\d+"  # 内网 C 类
    r"|(?:^|://)127\.\d+\.\d+\.\d+)"  # localhost
)

# OA 公文类型关键词 (OA 系统域名 + 以下任一关键词 = 提醒)
OA_DOC_KEYWORDS = re.compile(
    r"(通知|请示|报告|批复|函|意见|通报|纪要|决定|公告|议案"
    r"|报送|上报|汇报|提交|办理|整改|督办|批示|转发)"
)


def is_reminder(title: str, sender: str = "", url: str = "") -> bool:
    """K2: 规则判断是否为"提醒"(需人工关注).

    策略:
    1. OA 系统域名 + 公文类型关键词 → 提醒 (覆盖 OA 待办通知)
    2. 发件人白名单(领导/上级) → 提醒
    3. 行动关键词(截止/报送/督办/反馈/紧急) → 提醒
    不用 LLM, 纯规则.
    """
    # 1. OA 系统域名 + 公文类型
    if url and OA_SYSTEM_DOMAINS.search(url):
        if title and OA_DOC_KEYWORDS.search(title):
            return True

    # 2. 发件人匹配
    if sender and REMINDER_SENDER_WHITELIST.search(sender):
        return True

    # 3. 行动关键词匹配 (全域名适用)
    if title and REMINDER_KEYWORDS.search(title):
        return True

    return False


def route(title: str, url: str, sender: str = "") -> dict:
    """路由决策 + 硬拦 + 全沉淀策略 (K1/K2).

    返回:
    - mode: "archive" (全沉淀) / "cloud" (公开流)
    - reminder: bool (是否提醒)
    - reason: str
    """
    sensitivity = classify_sensitivity(title, url)

    if sensitivity == "sensitive":
        # K1: 敏感流 → 全沉淀 + embed 入 KOS, 不送外部 API
        return {
            "mode": "archive",
            "reminder": is_reminder(title, sender, url),
            "action": "全沉淀 + embed 索引入 KOS",
            "reason": f"敏感流硬拦: 标题含敏感词或域名匹配 (domain={urlparse(url).netloc})",
        }

    # 公开流 → 允许 DeepSeek 云
    return {
        "mode": "cloud",
        "reminder": False,
        "action": "DeepSeek 云处理",
        "reason": f"公开流: 域名白名单或无明显敏感特征 (domain={urlparse(url).netloc})",
    }


def hard_block_external(title: str, url: str) -> None:
    """硬拦: 敏感流调用外部 API 时抛异常, 不靠约定."""
    sensitivity = classify_sensitivity(title, url)
    if sensitivity == "sensitive":
        raise PermissionError(f"[硬拦] 敏感流禁止送外部 API: {title[:40]} ({urlparse(url).netloc})")


# ============================================================
# Gateway 集成 (K1 全沉淀 → embed 入 KOS)
# ============================================================
def embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 ModelGateway embedding (降级链: 8183 → 8188).

    用于 K1 全沉淀策略: 敏感流内容 embed 后索引入 KOS.

    Args:
        texts: 待嵌入文本列表

    Returns:
        向量列表, 每个向量是 float list

    Raises:
        RuntimeError: 所有 embedding 提供者都失败
    """
    try:
        from llm_gateway.gateway import ModelGateway, get_gateway  # noqa: F401 — 探测可用性

        gateway = get_gateway()
        return run_async(gateway.embed(texts))
    except ImportError:
        # fallback: 直接调 omlxc API
        return _embed_via_omlc(texts)


def _embed_via_omlc(texts: list[str]) -> list[list[float]]:
    """直接调 omlx embedding API (无 gateway 时的 fallback)."""
    import json
    import urllib.request

    # 降级链: 8183 → 8188
    ports = ["gateway"]  # 旧 8183/8188 单模型端口已下线, 统一经门面 /v1/embeddings
    for port in ports:
        try:
            payload = json.dumps(
                {
                    "input": texts,
                    "model": "embedding",
                }
            ).encode()
            req = urllib.request.Request(  # noqa: S310 — 内部门面地址(aetherforge.endpoint, http)
                f"{gateway_url()}/v1/embeddings",
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {gateway_key()}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — internal omlx API
                data = json.loads(resp.read())
                embeddings = [item["embedding"] for item in data.get("data", [])]
                if embeddings:
                    return embeddings
        except Exception:  # noqa: S112 — probe next port
            continue

    raise RuntimeError("embedding 经门面失败")


if __name__ == "__main__":
    # 自测
    test_cases = [
        # (标题, URL, 发件人)
        ("会议提醒下午3点", "https://oa.company.com/meeting/123", "卫健委办公室"),
        ("Python asyncio教程", "https://docs.python.org/3/library/asyncio.html", ""),
        ("京东商品广告", "https://item.jd.com/12345.html", ""),
        ("房山区卫健委通知: 报送总结截止周五", "https://mail.google.com/mail/u/0/#inbox", "房山区卫健委"),
        ("GitHub PR review", "https://github.com/starlink-awaken/omostation/pull/615", ""),
        ("知乎首页", "https://www.zhihu.com/hot", ""),
        ("督办反馈: 请尽快提交方案", "https://gov.cn/notice/456", "市政府督查室"),
        ("B站视频推荐", "https://www.bilibili.com/video/BV123", ""),
    ]
    print("=== K1/K2 路由 + 提醒自测 ===")
    for title, url, sender in test_cases:
        r = route(title, url, sender)
        print(f"  [{r['mode']:6s}] reminder={r['reminder']!s:5s} | {title[:30]}")
        print(f"           action: {r['action']}")
        print(f"           reason: {r['reason'][:60]}")
