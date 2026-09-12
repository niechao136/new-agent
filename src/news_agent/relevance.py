"""Relevance scoring / noise filtering (TODO item 9, ``filter_node``).

The score is an interpretable linear combination of token overlaps so that it
can run without an embedding model.  Embedding based re-ranking can be plugged
in by replacing :func:`rank_articles`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from .models import RawArticle, SkillRequest, utcnow
from .text_utils import clamp, normalize_for_compare, token_set


def relevance_score(
    article: RawArticle, query: str, *, reference_time: datetime | None = None
) -> float:
    """Score in ``[0, 1]`` describing how well ``article`` matches ``query``."""
    query_tokens = token_set(query)
    if not query_tokens:
        return 0.5

    title_tokens = token_set(article.title)
    body_tokens = token_set(f"{article.summary or ''}")
    if article.content:
        body_tokens |= token_set(article.content[:2000])

    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
    body_overlap = len(query_tokens & body_tokens) / len(query_tokens)
    score = 0.65 * title_overlap + 0.35 * body_overlap

    # No topical overlap at all -> irrelevant, regardless of how fresh it is.
    # (Otherwise the recency bonus alone could push off-topic items over the
    # relevance threshold.)
    if score <= 0:
        return 0.0

    # whole-phrase bonus (works well for CJK queries)
    query_norm = normalize_for_compare(query)
    if query_norm and len(query_norm) >= 2:
        if query_norm in normalize_for_compare(article.title):
            score += 0.35
        elif query_norm in normalize_for_compare(f"{article.summary or ''}{article.content or ''}"):
            score += 0.15
    else:
        # latin queries: all tokens present in the title
        if query_tokens and query_tokens <= title_tokens:
            score += 0.2

    # recency bonus -- news decay fast
    now = reference_time or utcnow()
    if article.published_at is not None:
        age = now - article.published_at
        if age <= timedelta(hours=24):
            score += 0.1
        elif age <= timedelta(days=7):
            score += 0.05
        elif age > timedelta(days=30):
            score -= 0.1

    return round(clamp(score), 4)


def rank_articles(
    articles: Iterable[RawArticle], query: str, *, reference_time: datetime | None = None
) -> list[tuple[RawArticle, float]]:
    scored = [
        (article, relevance_score(article, query, reference_time=reference_time))
        for article in articles
    ]
    scored.sort(
        key=lambda item: (
            item[1],
            item[0].published_at.timestamp() if item[0].published_at else 0.0,
        ),
        reverse=True,
    )
    return scored


def select_relevant(
    articles: list[RawArticle],
    request: SkillRequest,
    *,
    threshold: float,
    limit: int,
) -> tuple[list[RawArticle], int, int]:
    """Return ``(selected, dropped, below_threshold_kept)``.

    Articles scoring below ``threshold`` are dropped, but when that leaves
    fewer than three results the best scoring leftovers are topped up (and the
    caller is told about it through ``below_threshold_kept``) -- degrading to a
    slightly noisy answer beats returning nothing (TODO item 3).
    """
    if not articles:
        return [], 0, 0

    scored = rank_articles(articles, request.query)
    above = [(article, score) for article, score in scored if score >= threshold]
    below = [(article, score) for article, score in scored if score < threshold]

    selected = above
    topped_up = 0
    if len(selected) < 3 and below:
        missing = 3 - len(selected)
        topped_up = min(missing, len(below))
        selected = selected + below[:topped_up]

    selected = selected[:limit]
    dropped = len(articles) - len(selected)
    return [article for article, _ in selected], dropped, topped_up
