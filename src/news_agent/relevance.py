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

#: 栏目领域 -> 触发该领域的查询词（多语言）。
#: 用于「泛化查询」的栏目路由，例如「科技新闻」应当直接采信科技栏目。
TOPIC_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "tech": (
        "科技", "技术", "数码", "互联网", "人工智能", "芯片", "软件", "硬件", "信息技术",
        "tech", "technology", "software", "hardware", "internet", "ai", "gadget",
        "startup", "chip", "digital",
    ),
    "business": (
        "财经", "金融", "商业", "经济", "股市", "投资", "基金", "债券", "银行", "财报",
        "finance", "financial", "business", "economy", "economic", "market", "markets",
        "stock", "stocks", "investment", "wall street", "trade",
    ),
    "world": (
        "国际", "世界", "时政", "全球", "外交", "地缘", "美国", "欧洲", "中东", "亚太",
        "world", "international", "global", "politics", "diplomacy", "geopolitics",
    ),
    "sports": (
        "体育", "足球", "篮球", "赛事", "奥运", "网球", "世界杯", "欧冠",
        "sport", "sports", "football", "soccer", "basketball", "tennis", "olympic", "nba",
    ),
    "entertainment": (
        "娱乐", "影视", "电影", "明星", "综艺", "音乐", "电视剧",
        "entertainment", "movie", "movies", "film", "celebrity", "hollywood", "music",
    ),
    "health": (
        "健康", "医疗", "医药", "医学", "疾病", "疫苗", "养生",
        "health", "healthcare", "medical", "medicine", "drug", "disease", "vaccine",
    ),
}

#: 泛化新闻词（不含领域信息）：只有这些词构成的查询没有检索区分度。
GENERIC_NEWS_TERMS: tuple[str, ...] = (
    "新闻", "资讯", "报道", "消息", "热点", "动态", "最新", "要闻", "快讯", "简报",
    "摘要", "汇总", "总结", "新闻资讯",
    "news", "headline", "headlines", "latest", "update", "updates", "briefing",
    "daily", "weekly", "roundup", "digest", "report",
)

#: 泛化查询命中栏目后给予的基础分（高于默认阈值 0.2，低于强词面命中）。
TOPIC_ROUTING_BASE = 0.5


def _norm_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = (normalize_for_compare(term) for term in terms)
    return tuple(term for term in normalized if term)


_TOPIC_TERMS_NORM: dict[str, tuple[str, ...]] = {
    topic: _norm_terms(terms) for topic, terms in TOPIC_QUERY_TERMS.items()
}
_GENERIC_TERMS_NORM: tuple[str, ...] = _norm_terms(GENERIC_NEWS_TERMS)


def _strip_terms(text: str, terms: tuple[str, ...]) -> str:
    for term in sorted(terms, key=len, reverse=True):
        text = text.replace(term, "")
    return text


def matched_topics(query: str) -> set[str]:
    """领域标签集合：查询词命中了哪些栏目（如「科技新闻」→ ``{"tech"}``）。"""
    norm = normalize_for_compare(query)
    if not norm:
        return set()
    return {
        topic
        for topic, terms in _TOPIC_TERMS_NORM.items()
        if any(term in norm for term in terms)
    }


def is_broad_query(query: str, matched: set[str] | None = None) -> bool:
    """True 表示查询只是"某领域 + 新闻"这类泛化表达，没有具体检索对象。

    例如「科技新闻」「财经新闻」「ai chips」「sports news」为 True，
    「人形机器人」「固态电池」「英伟达财报」为 False。
    """
    norm = normalize_for_compare(query)
    if not norm:
        return False
    matched = matched_topics(query) if matched is None else matched
    residual = _strip_terms(norm, _GENERIC_TERMS_NORM)
    if not residual:
        return True  # 只说了「新闻/最新/资讯」
    if not matched:
        return False
    return any(
        len(_strip_terms(residual, _TOPIC_TERMS_NORM[topic])) <= 1 for topic in matched
    )


def topic_floor(query: str, source_topic: str | None, *, matched: set[str] | None = None) -> float:
    """栏目路由基础分：泛化查询 + 领域匹配的栏目来源才能拿到。"""
    if not source_topic:
        return 0.0
    matched = matched_topics(query) if matched is None else matched
    if not matched or source_topic not in matched:
        return 0.0
    if not is_broad_query(query, matched):
        return 0.0
    return TOPIC_ROUTING_BASE


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
    source_topics: Mapping[str, str] | None = None,
) -> list[tuple[RawArticle, float]]:
    """Return ``[(article, score)]`` sorted by (weighted score, score, time) desc.

    The returned score is the raw lexical score (threshold semantics), while the
    ordering also takes the source ``weight`` into account.

    ``source_topics`` maps a source name to its栏目领域（tech/business/...）。
    当 ``query`` 是泛化表达（「科技新闻」）时，命中领域的栏目来源会拿到
    :data:`TOPIC_ROUTING_BASE` 的基础分——这既让泛查询有合理结果，也解决了
    中文泛查询无法词面命中英文源的问题。
    """
    items = list(articles)
    terms = _terms(query, keywords)
    matched = matched_topics(query) if source_topics else set()

    def _floor(article: RawArticle) -> float:
        if not source_topics:
            return 0.0
        return topic_floor(query, source_topics.get(article.source), matched=matched)

    scored = [
        (
            article,
            max(
                _article_score(article, terms, reference_time=reference_time),
                _floor(article),
            ),
        )
        for article in items
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
    "TOPIC_QUERY_TERMS",
    "TOPIC_ROUTING_BASE",
    "diversify",
    "is_broad_query",
    "longest_common_run",
    "matched_topics",
    "pick_relevant",
    "rank_articles",
    "relevance_score",
    "select_relevant",
    "topic_floor",
]
