"""Deterministic in-process source used for demos, tests and air-gapped runs.

It emits a realistic mixture of articles: relevant hits with different
sentiment, one cross-source duplicate (to exercise the de-duplicator) and a
couple of off-topic items (to exercise the relevance filter).
"""

from __future__ import annotations

from datetime import timedelta

from ..models import RawArticle, SkillRequest, utcnow
from .base import NewsSource

_POSITIVE = [
    (
        "{query}行业迎来新一轮增长，多家机构上调预期",
        "多家研究机构发布报告，看好{query}相关产业的中长期增长空间，并上调未来两年的市场预期。分析师指出，下游需求持续回暖是主要驱动因素。",
        "{query}的产业化进程明显提速。报告显示，头部企业的订单量同比增长显著，产业链上下游均受益。多位专家认为，随着政策支持与资本投入加码，{query}有望成为新的增长引擎。",
    ),
    (
        "{query}技术取得突破，落地场景进一步拓展",
        "研究团队宣布在{query}关键技术上取得阶段性突破，相关方案已在多个行业完成验证。",
        "该团队表示，新方案在效率与成本上均有明显改善，已有企业开始小规模试用。行业观察人士认为，这将加快{query}的商业化节奏。",
    ),
    (
        "{query}投资热度上升，一级市场融资活跃",
        "统计显示，近期{query}领域的一级市场融资事件数量与金额均出现回升。",
        "多家早期基金表示，正在加大对{query}方向的配置力度。创业者反馈，客户付费意愿较去年明显改善。",
    ),
]

_NEGATIVE = [
    (
        "{query}面临监管收紧，合规成本上升",
        "监管机构就{query}相关问题发布新的指引，要求企业加强信息披露与风险控制。",
        "业内人士表示，新规将抬高中小企业的合规成本，部分项目可能因此推迟上线。也有观点认为，长期看有助于行业健康发展。",
    ),
    (
        "{query}项目出现延期，供应链压力显现",
        "受关键元器件交付延迟影响，多个{query}项目进度不及预期。",
        "供应链人士透露，部分环节的交付周期已延长数周。企业正在寻找替代供应商，但短期内难以完全缓解。",
    ),
]

_NEUTRAL = [
    (
        "关于{query}的行业观察：机遇与挑战并存",
        "本文梳理了{query}当前的发展现状、主要玩家与未来的不确定性。",
        "从供给端看，参与者数量快速增加；从需求端看，付费意愿仍在培育之中。文章认为，{query}的竞争格局尚未定型。",
    ),
    (
        "{query}相关标准启动制定，行业进入规范化阶段",
        "相关标准化组织宣布成立工作组，启动{query}领域标准的起草工作。",
        "工作组将围绕术语、接口与安全要求展开讨论，预计明年形成征求意见稿。企业可通过公开渠道反馈意见。",
    ),
]

_NOISE = [
    (
        "本地天气：周末将迎来降温与降雨",
        "气象部门提醒市民注意保暖，出行携带雨具。",
        "受冷空气影响，本周末气温将明显下降，部分地区有小到中雨。",
    ),
    (
        "体育赛事综述：主队加时赛险胜对手",
        "在一场焦点战中，主队通过加时赛以微弱优势取胜。",
        "双方在常规时间战平，加时赛中主队依靠关键球锁定胜局，全场观众气氛热烈。",
    ),
]


class MockSource(NewsSource):
    """Offline source with fully deterministic output."""

    type = "mock"

    async def _fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
        now = utcnow()
        articles: list[RawArticle] = []
        templates = [
            *[(title, summary, content, "positive") for title, summary, content in _POSITIVE],
            *[(title, summary, content, "negative") for title, summary, content in _NEGATIVE],
            *[(title, summary, content, "neutral") for title, summary, content in _NEUTRAL],
            *[(title, summary, content, "neutral") for title, summary, content in _NOISE],
        ]

        for index, (title, summary, content, _sentiment) in enumerate(templates):
            if len(articles) >= max(limit, 8):
                break
            article = self._make_article(
                title=title.format(query=request.query),
                url=f"https://mock.news/{self.name}/{index}",
                published_at=now - timedelta(hours=index * 3 + 1),
                summary=summary.format(query=request.query),
                content=content.format(query=request.query),
                language=request.language,
                author="news-agent",
                source_name=f"{self.name}-{index % 3}",
            )
            if article and self._within_window(article, request):
                articles.append(article)

        # a cross-source duplicate of the first article (different url, slightly
        # different title) so the de-duplication logic is always exercised
        if articles:
            original = articles[0]
            duplicate = RawArticle(
                id="mock-duplicate",
                title=original.title + "（转载）",
                url="https://mirror.news/duplicate/1",
                source="mock-mirror",
                published_at=original.published_at,
                summary=original.summary,
                content=original.content,
                language=original.language,
            )
            articles.append(duplicate)
        return articles
