"""LangGraph state definition (TODO item 10)."""

from __future__ import annotations

from typing import Any, TypedDict

from ..models import AnalyzedArticle, NewsResult, RawArticle, SkillRequest


class NewsState(TypedDict, total=False):
    """State flowing through the news sub-graph.

    Every node returns a partial dict which LangGraph merges into the state.
    Progress/errors/timings deliberately live on the
    :class:`~news_agent.runtime.RunContext` instead of the state: they are
    observability data, not business data, and the context must stay reachable
    even if the graph is interrupted by a timeout.
    """

    # input
    request: SkillRequest

    # fetch_node
    raw_articles: list[RawArticle]

    # filter_node
    deduped_articles: list[RawArticle]
    filtered_articles: list[RawArticle]
    duplicates_removed: int

    # analyze_node
    analyzed_articles: list[AnalyzedArticle]

    # summarize_node
    summary: str

    # format_node
    result: NewsResult
    warnings: list[str]
    errors: list[str]
    timings: dict[str, float]
    counters: dict[str, int]
    extras: dict[str, Any]
