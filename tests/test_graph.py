"""End-to-end graph tests against the offline mock source (TODO item 19)."""

from __future__ import annotations

from news_agent.config import SourceConfig
from news_agent.models import ErrorCode, NewsResult, RawArticle, SkillMode, SkillRequest
from news_agent.runtime import RunContext


async def test_fetch_news_skill_skips_analysis(agent, request_factory):
    result = await agent.run(request_factory(skill="fetch_news", limit=8))

    assert result.mode is SkillMode.FETCH
    assert result.articles
    assert result.summary == ""
    assert all(article.sentiment == "neutral" for article in result.articles)
    assert all(not article.entities for article in result.articles)
    # relevance is still attached so callers can rank
    assert all(article.relevance >= 0 for article in result.articles)


async def test_summarize_news_end_to_end(agent, request_factory):
    result = await agent.run(request_factory(skill="summarize_news", limit=10))

    assert result.mode is SkillMode.SUMMARIZE
    assert result.articles, "mock source should always produce articles"
    assert result.summary
    assert result.degraded is False
    assert result.errors == []

    counts = result.counts
    assert counts["fetched"] > counts["after_dedup"], "cross-source duplicate must be removed"
    assert counts["duplicates_removed"] >= 1
    assert counts["selected"] >= counts["analyzed"] or counts["analyzed"] > 0

    # stages are timed for observability (TODO item 17)
    for stage in ("fetch", "filter", "analyze", "summarize", "format"):
        assert stage in result.timings_ms
    assert result.duration_ms >= 0
    assert result.metrics["analyzer"] == "heuristic"


async def test_analyze_trend_returns_trend_insights(agent, request_factory):
    result = await agent.run(request_factory(skill="analyze_trend", limit=10))

    assert result.trends
    top = result.trends[0]
    assert top.topic
    assert top.mentions >= 1
    assert top.representative_urls
    assert top.sentiment in {"positive", "neutral", "negative"}


async def test_offline_run_has_no_source_errors(agent, request_factory):
    result = await agent.run(request_factory(limit=6))
    assert not [error for error in result.errors if error.code is ErrorCode.SOURCE_UNAVAILABLE]
    assert result.counts["sources_used"] >= 1


async def test_second_run_serves_from_cache(agent, request_factory):
    first = await agent.run(request_factory(limit=6))
    assert first.articles

    ctx = RunContext()
    second = await agent.run(request_factory(limit=6), ctx=ctx)

    assert ctx.counters.get("fetch_cache_hit") == 1
    assert len(second.articles) == len(first.articles)
    assert "fetch" in {event.stage for event in ctx.events}
    assert any("缓存" in event.message for event in ctx.events)


async def test_results_are_sorted_by_relevance(agent, request_factory):
    result = await agent.run(request_factory(skill="analyze_trend", limit=10))
    scores = [article.relevance for article in result.articles]
    assert scores == sorted(scores, reverse=True)


async def test_empty_result_is_reported_not_raised(settings, request_factory):
    from news_agent.graph.agent import NewsAgent
    from news_agent.sources import HttpClient, NewsSource, SourceRegistry

    class EmptySource(NewsSource):
        """Real source whose backend simply returns nothing."""

        async def _fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
            return []

    http = HttpClient(settings)
    registry = SourceRegistry(
        [EmptySource(SourceConfig(name="empty", type="mock"), settings, http)],
        http=http,
        settings=settings,
    )
    instance = await NewsAgent.create(
        settings.model_copy(update={"cache_enabled": False}),
        registry=registry,
        cache=None,
    )
    try:
        result = await instance.run(request_factory())
    finally:
        await instance.aclose()

    assert result.articles == []
    assert any(error.code is ErrorCode.NO_RESULTS for error in result.errors)
    assert result.warnings
    assert result.degraded is True


def test_skill_request_cache_key_is_stable_and_ordered():
    left = SkillRequest(query="人形机器人", limit=5, sources=["b", "a"])
    right = SkillRequest(query="人形机器人", limit=5, sources=["a", "b"])
    other = SkillRequest(query="人形机器人", limit=6, sources=["a", "b"])
    assert left.cache_key() == right.cache_key()
    assert left.cache_key() != other.cache_key()


def test_news_result_ok_flag():
    assert NewsResult(query="q", summary="s").ok
    from news_agent.models import ErrorInfo

    broken = NewsResult(query="q", errors=[ErrorInfo(code=ErrorCode.INTERNAL_ERROR, message="x")])
    assert broken.ok is False
