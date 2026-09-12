"""内容质量过滤：博彩/体育下注 SEO 站群、促销页与内容农场。

线上真实案例：查询「科技新闻」时，Google News 的搜索结果里混入了

    「科技新闻简报：m88JDB电子引领体育数字化革新- 体坛网」
    「正规买球万博APP科技新闻摘要合集：AI革新、融资合作与全球扩张」

这类页面把查询词原样塞进标题，词面相关性打分很高（0.88），LLM 精排也会
把它们当成"确实在讲科技新闻"。它们的问题不在关键词匹配，而在**页面性质**，
所以需要独立的质量维度：

* ``SPAM_BRAND_TERMS``：赌博/博彩品牌词，命中即判定为垃圾（1.0）；
* ``SPAM_TERMS``：博彩、下注、返佣等语义词，累计计分（单词命中不致命，
  避免误伤"澳门博彩业营收"这类正当报道）；
* ``SPAM_HOST_PATTERNS``：域名特征（站群域名往往一眼可辨）；
* 促销词与站群关键词共现时加成。

只依赖标题/摘要/URL，确定性、零成本，可与 LLM 精排叠加使用。
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import RawArticle
from .text_utils import clean_text

#: 命中即判定为垃圾的品牌/站群词（不区分大小写）。
SPAM_BRAND_TERMS: tuple[str, ...] = (
    "m88",
    "sbobet",
    "1xbet",
    "bet365",
    "betway",
    "dafabet",
    "188bet",
    "12bet",
    "bbin",
    "dafa",
    "万博",
    "九游会",
    "威尼斯人",
    "新葡京",
    "永利皇宫",
    "澳门银河",
    "太阳城集团",
    "pk10",
    "六合彩",
    "时时彩",
    "快乐8",
    "幸运飞艇",
    "北京赛车",
    "千炮捕鱼",
)

#: 博彩/下注语义词：单个命中只加分，累计到阈值才判定为垃圾。
SPAM_TERMS: tuple[str, ...] = (
    "买球",
    "赌球",
    "赌场",
    "赌博",
    "博彩",
    "体育下注",
    "下注",
    "投注",
    "赔率",
    "盘口",
    "让球",
    "真人娱乐",
    "电子游艺",
    "娱乐城",
    "老虎机",
    "捕鱼",
    "棋牌",
    "彩票",
    "开户返水",
    "返水",
    "注册送",
    "首存",
    "casino",
    "gambling",
    "betting",
    "sportsbook",
    "lottery",
    "jackpot",
    "roulette",
    "wager",
    "poker",
)

#: 促销/导流词：与博彩词共现时显著提高可疑度。
PROMO_TERMS: tuple[str, ...] = (
    "app下载",
    "立即下载",
    "官网",
    "登录",
    "注册",
    "代理",
    "招商",
    "优惠",
    "彩金",
    "礼包",
    "免费领取",
    "点击进入",
)

#: 域名特征（站群/博彩站点）。
SPAM_HOST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"m88",
        r"sbobet",
        r"1xbet",
        r"bet\d{3,}",
        r"betway",
        r"dafabet",
        r"casino",
        r"gambling",
        r"lottery",
        r"poker",
        r"bocai",
        r"caipiao",
    )
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:  # pragma: no cover - defensive
        return ""


def spam_score(article: RawArticle) -> float:
    """0.0（正常）～1.0（确定是垃圾页面）的可疑度。"""
    host = _host(article.url)
    if host and any(pattern.search(host) for pattern in SPAM_HOST_PATTERNS):
        return 1.0

    text = clean_text(f"{article.title} {article.summary or ''}").lower()
    if not text:
        return 0.0

    if any(term in text for term in SPAM_BRAND_TERMS):
        return 1.0

    score = 0.0
    hits = sum(1 for term in SPAM_TERMS if term in text)
    if hits:
        # 单词命中不致命：正当报道也会提到"博彩/赌场"（监管、营收等）
        score += min(0.75, 0.25 * hits)
    promo_hits = sum(1 for term in PROMO_TERMS if term in text)
    if promo_hits and hits:
        score += min(0.3, 0.15 * promo_hits)
    return round(min(1.0, score), 4)


def is_spam(article: RawArticle, *, threshold: float = 0.6) -> bool:
    return spam_score(article) >= threshold


def filter_spam(
    articles: list[RawArticle], *, threshold: float = 0.6
) -> tuple[list[RawArticle], list[RawArticle]]:
    """Split ``articles`` into ``(kept, dropped)`` by the spam score."""
    kept: list[RawArticle] = []
    dropped: list[RawArticle] = []
    for article in articles:
        (dropped if is_spam(article, threshold=threshold) else kept).append(article)
    return kept, dropped


__all__ = ["filter_spam", "is_spam", "spam_score"]
