"""Relevance scoring / noise filtering (``filter_node``).

The score is an interpretable linear combination of token overlaps so that it
can run without an embedding model.  On top of the naive ``|Q∩T| / |Q|`` overlap
it adds three cheap but effective refinements:

1. **Contiguity constraint** — a CJK bigram crossing a word boundary (query
   「特斯拉新闻」 → 「拉新」) matches an unrelated article by accident.  The score
   is therefore scaled by the longest *contiguous* character run shared with the
   term: a 2-of-5 character accident is discounted, an entity match keeps its
   weight and an exact phrase is untouched.
2. **Multi-keyword matching** — the score is the best match over the query and
   its expansion keywords (translations / aliases); this is what makes the
   international RSS feeds usable for a Chinese query.
3. **Source weight & diversity** — ``SourceConfig.weight`` orders results of
   equal relevance, and no single source may dominate the final answer.

Deliberately absent: corpus statistics (IDF/BM25).  With a candidate pool of a
few dozen articles the document frequencies are dominated by whatever happened
to be fetched, which made rare-but-accidental bigrams *outrank* real entity
matches.  The score here depends only on the article and the query, so the same
article always gets the same score.

Threshold decisions always use the raw lexical score; weights only affect
ordering, so tuning a source weight never silently changes recall.

Embedding or LLM based re-ranking can be plugged in on top: see
``news_agent.rerank`` (used by the filter node) or replace :func:`rank_articles`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta

from .models import RawArticle, SkillRequest, utcnow
from .text_utils import clamp, normalize_for_compare, token_set

#: Shortest normalised term (in characters) that gets the contiguity constraint.
_MIN_CONTIGUITY_CHARS = 4
#: Score multiplier for a term with zero contiguous overlap (0 → 0.7, 1 → 1.0).
_CONTIGUITY_FLOOR = 0.7
#: Below this topical score the recency bonus is scaled down: freshness must not
#: rescue an article that barely matches the query.
_RECENCY_GATE = 0.3


def longest_common_run(left: str, right: str) -> int:
    """Length of the longest contiguous substring shared by ``left`` and ``right``."""
    if not left or not right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    best = 0
    for right_char in right:
        current = [0] * (len(left) + 1)
        for index, left_char in enumerate(left, start=1):
            if left_char == right_char:
                current[index] = previous[index - 1] + 1
                if current[index] > best:
                    best = current[index]
        previous = current
    return best


def _contiguity(term_norm: str, text_norm: str) -> float:
    if not term_norm or not text_norm:
        return 0.0
    return min(1.0, longest_common_run(term_norm, text_norm) / len(term_norm))


def _coverage(term_tokens: set[str], target: set[str]) -> float:
    """Share of the term's tokens present in ``target``."""
    hit = term_tokens & target
    if not hit:
        return 0.0
    return len(hit) / len(term_tokens)


def _term_score(article: RawArticle, term: str) -> float:
    """Topical score of a single term (query or one expansion keyword)."""
    term_tokens = token_set(term)
    if not term_tokens:
        return 0.0

    title_tokens = token_set(article.title)
    body_tokens = token_set(article.summary or "")
    if article.content:
        body_tokens |= token_set(article.content[:2000])

    score = 0.65 * _coverage(term_tokens, title_tokens) + 0.35 * _coverage(
        term_tokens, body_tokens
    )
    if score <= 0:
        return 0.0

    term_norm = normalize_for_compare(term)
    title_norm = normalize_for_compare(article.title)
    body_norm = normalize_for_compare(f"{article.summary or ''}{article.content or ''}")

    # Contiguity constraint: discount accidental (word-boundary) bigram hits.
    if len(term_norm) >= _MIN_CONTIGUITY_CHARS:
        contiguity = max(_contiguity(term_norm, title_norm), _contiguity(term_norm, body_norm))
        score *= _CONTIGUITY_FLOOR + (1.0 - _CONTIGUITY_FLOOR) * contiguity

    # whole-phrase bonus (works well for CJK queries)
    if term_norm and len(term_norm) >= 2:
        if term_norm in title_norm:
            score += 0.35
        elif term_norm in body_norm:
            score += 0.15
    elif term_tokens <= title_tokens:
        # latin queries: all tokens present in the title
        score += 0.2

    return score


def _recency_bonus(article: RawArticle, reference_time: datetime | None, topical: float) -> float:
    """Freshness bonus, scaled down for weak topical matches."""
    if article.published_at is None:
        return 0.0
    now = reference_time or utcnow()
    age = now - article.published_at
    if age <= timedelta(hours=24):
        bonus = 0.1
    elif age <= timedelta(days=7):
        bonus = 0.05
    elif age > timedelta(days=30):
        bonus = -0.1
    else:
        return 0.0
    if bonus > 0 and topical < _RECENCY_GATE:
        bonus *= max(0.0, topical / _RECENCY_GATE)
    return bonus


def _article_score(
    article: RawArticle,
    terms: Sequence[str],
    *,
    reference_time: datetime | None = None,
) -> float:
    """Best score over ``terms`` (query + expansion keywords)."""
    topical = 0.0
    for term in terms:
        if not term:
            continue
        topical = max(topical, _term_score(article, term))
    if topical <= 0:
        return 0.0
    return round(
        clamp(topical + _recency_bonus(article, reference_time, topical)), 4
    )


def _terms(query: str, keywords: Sequence[str] | None) -> list[str]:
    terms = [query]
    for keyword in keywords or []:
        if keyword and keyword not in terms:
            terms.append(str(keyword).strip())
    return [term for term in terms if term]


def relevance_score(
    article: RawArticle,
    query: str,
    *,
    reference_time: datetime | None = None,
    keywords: Sequence[str] | None = None,
) -> float:
    """Score in ``[0, 1]`` describing how well ``article`` matches ``query``.

    ``keywords`` are alternative phrasings (translations/aliases); the article is
    scored against each and the best match wins.
    """
    return _article_score(
        article, _terms(query, keywords), reference_time=reference_time
    )


def rank_articles(
    articles: Iterable[RawArticle],
    query: str,
    *,
    reference_time: datetime | None = None,
    keywords: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[RawArticle, float]]:
    """Return ``[(article, score)]`` sorted by (weighted score, score, time) desc.

    The returned score is the raw lexical score (threshold semantics), while the
    ordering also takes the source ``weight`` into account.
    """
    terms = _terms(query, keywords)
    scored = [
        (article, _article_score(article, terms, reference_time=reference_time))
        for article in articles
    ]

    def _weight(article: RawArticle) -> float:
        if not weights:
            return 1.0
        return float(weights.get(article.source, 1.0))

    scored.sort(
        key=lambda item: (
            item[1] * _weight(item[0]),
            item[1],
            item[0].published_at.timestamp() if item[0].published_at else 0.0,
        ),
        reverse=True,
    )
    return scored


def diversify(
    scored: Sequence[tuple[RawArticle, float]],
    *,
    per_source_cap: int,
) -> list[tuple[RawArticle, float]]:
    """Cap how many articles a single source contributes, keeping the order.

    Overflow items are not discarded: they are appended after the capped ones, so
    they still fill the result when there is not enough material elsewhere.
    """
    if per_source_cap <= 0:
        return list(scored)
    kept: list[tuple[RawArticle, float]] = []
    deferred: list[tuple[RawArticle, float]] = []
    counts: dict[str, int] = {}
    for item in scored:
        source = item[0].source
        if counts.get(source, 0) < per_source_cap:
            counts[source] = counts.get(source, 0) + 1
            kept.append(item)
        else:
            deferred.append(item)
    return kept + deferred


def pick_relevant(
    scored: Sequence[tuple[RawArticle, float]],
    *,
    threshold: float,
    limit: int,
    per_source_cap: int | None = None,
) -> tuple[list[RawArticle], int, int]:
    """Select the final set from ``(article, score)`` pairs.

    Articles below ``threshold`` are dropped; when that leaves fewer than three
    results the best scoring leftovers are topped up **as long as they still show
    some lexical overlap** (``score >= threshold * 0.25``) — degrading to a
    slightly noisy but on-topic answer beats returning nothing, while pure noise
    (score 0) is never injected.

    @returns ``(selected, dropped, below_threshold_kept)``
    """
    if not scored:
        return [], 0, 0

    above = [item for item in scored if item[1] >= threshold]
    below = [item for item in scored if item[1] < threshold]

    selected = list(above)
    topped_up = 0
    if len(selected) < 3 and below:
        floor = max(1e-6, threshold * 0.25)
        usable = [item for item in below if item[1] >= floor]
        topped_up = min(3 - len(selected), len(usable))
        selected = selected + usable[:topped_up]

    if per_source_cap is not None:
        selected = diversify(selected, per_source_cap=per_source_cap)

    selected = selected[:limit]
    dropped = len(scored) - len(selected)
    return [article for article, _ in selected], dropped, topped_up


def select_relevant(
    articles: list[RawArticle],
    request: SkillRequest,
    *,
    threshold: float,
    limit: int,
) -> tuple[list[RawArticle], int, int]:
    """Backwards compatible wrapper around :func:`rank_articles` + :func:`pick_relevant`."""
    scored = rank_articles(articles, request.query, keywords=request.keywords)
    return pick_relevant(scored, threshold=threshold, limit=limit)


__all__ = [
    "diversify",
    "longest_common_run",
    "pick_relevant",
    "rank_articles",
    "relevance_score",
    "select_relevant",
]
