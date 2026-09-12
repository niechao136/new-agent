"""Topic level aggregation for the ``analyze_trend`` skill."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from .models import AnalyzedArticle, TrendInsight
from .text_utils import tokenize

Sentiment = Literal["positive", "neutral", "negative"]


def _sentiment_label(scores: list[float]) -> Sentiment:
    if not scores:
        return "neutral"
    average = sum(scores) / len(scores)
    if average >= 0.15:
        return "positive"
    if average <= -0.15:
        return "negative"
    return "neutral"


def build_trends(
    articles: Iterable[AnalyzedArticle], *, top_n: int = 8, min_mentions: int = 2
) -> list[TrendInsight]:
    """Group analysed articles by entity and produce ranked trend insights."""
    articles = list(articles)
    if not articles:
        return []

    buckets: dict[str, list[AnalyzedArticle]] = defaultdict(list)
    for article in articles:
        for entity in _entities(article):
            buckets[entity].append(article)

    insights: list[TrendInsight] = []
    for topic, grouped in buckets.items():
        if len(grouped) < min_mentions:
            continue
        scores = [article.sentiment_score for article in grouped]
        ordered = sorted(grouped, key=lambda item: item.relevance, reverse=True)
        insights.append(
            TrendInsight(
                topic=topic,
                mentions=len(grouped),
                sentiment=_sentiment_label(scores),
                average_sentiment=round(sum(scores) / len(scores), 3),
                keywords=_topic_keywords(grouped, topic),
                representative_urls=[article.url for article in ordered[:3]],
            )
        )

    if not insights:
        # nothing repeated -> fall back to the individually most salient entity
        flat: dict[str, list[AnalyzedArticle]] = defaultdict(list)
        for article in articles:
            for entity in _entities(article):
                flat[entity].append(article)
        for topic, grouped in sorted(flat.items(), key=lambda item: -len(item[1]))[:top_n]:
            scores = [article.sentiment_score for article in grouped]
            ordered = sorted(grouped, key=lambda item: item.relevance, reverse=True)
            insights.append(
                TrendInsight(
                    topic=topic,
                    mentions=len(grouped),
                    sentiment=_sentiment_label(scores),
                    average_sentiment=round(sum(scores) / len(scores), 3),
                    keywords=_topic_keywords(grouped, topic),
                    representative_urls=[article.url for article in ordered[:3]],
                )
            )

    insights.sort(key=lambda insight: (-insight.mentions, -abs(insight.average_sentiment)))
    return insights[:top_n]


def _entities(article: AnalyzedArticle) -> list[str]:
    entities = [entity.strip() for entity in article.entities if entity and len(entity.strip()) >= 2]
    if entities:
        return list(dict.fromkeys(entities))[:5]
    # fall back to the longest salient tokens of the title
    tokens = [token for token in tokenize(article.title) if len(token) >= 2]
    return list(dict.fromkeys(tokens))[:3]


def _topic_keywords(articles: list[AnalyzedArticle], topic: str, top_n: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for article in articles:
        for token in tokenize(f"{article.title} {article.summary or ''}"):
            if token == topic:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:top_n]]
