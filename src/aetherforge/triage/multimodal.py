"""Multi-modal triage — 多模态分诊扩展.

支持:
- 文本分诊 (已有)
- 图片分诊 (通过视觉模型)
- URL 分诊 (抓取内容后分诊)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from .router import TriageResult, TriageRouter


@dataclass
class MultiModalTriage:
    """多模态分诊器."""

    router: TriageRouter
    gateway_url: str = "http://100.96.126.35:4000/v1/chat/completions"
    api_key: str = "sk-omlx-admin"

    def triage_url(self, url: str, title: str = "") -> TriageResult:
        """URL 分诊 — 抓取内容后分诊."""
        try:
            # 简单抓取
            req = urllib.request.Request(url, headers={"User-Agent": "aetherforge-triage/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="ignore")[:2000]

            # 提取标题
            if not title:
                import re

                title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else url

            # 提取文本
            import re

            text = re.sub(r"<[^>]+>", " ", content)
            text = re.sub(r"\s+", " ", text).strip()[:500]

            return self.router.triage_one(text, title=title, url=url)
        except Exception as e:
            return TriageResult(
                verdict="错误",
                model="",
                latency=0,
                error=f"URL 抓取失败: {str(e)[:50]}",
            )

    def triage_image_url(self, image_url: str, description: str = "") -> TriageResult:
        """图片 URL 分诊 — 通过视觉模型分析."""
        # 使用 vision 模型分析图片
        prompt = f"""分析这张图片，判断其内容类型:
- 丢弃: 广告/营销/无关图片
- 沉淀: 技术图表/架构图/知识图谱/有价值截图
- 提醒: 包含时间/日期/待办/告警的截图

图片描述: {description if description else "无"}
URL: {image_url}

只输出一个词 (丢弃/沉淀/提醒):"""

        payload = json.dumps(
            {
                "model": "vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": 20,
                "temperature": 0,
            }
        ).encode()

        req = urllib.request.Request(
            self.gateway_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        import time

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read())
            latency = time.time() - t0
            content = d["choices"][0]["message"]["content"].strip()

            verdict = None
            for v in ("丢弃", "沉淀", "提醒"):
                if v in content:
                    verdict = v
                    break

            return TriageResult(
                verdict=verdict or f"未知({content[:15]})",
                model="vision",
                latency=latency,
            )
        except Exception as e:
            return TriageResult(
                verdict="错误",
                model="vision",
                latency=time.time() - t0,
                error=str(e)[:50],
            )
