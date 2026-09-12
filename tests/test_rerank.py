"""LLM 精排（rerank）测试：单次打分、融合、以及图集成。"""

from __future__ import annotations

import pytest

from news_agent.config import LLMSettings, Settings, SourceConfig
from news_agent.graph.agent import NewsAgent
from news_agent.models import RawArticle, SkillRequest
from news_agent.rerank import (
    LLMReranker,
    RerankError,
    _RerankOutput,
    blend_scores,
    build_reranker,
)


class _FakeStructured:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def ainvoke(self, messages):
        return self._behaviour()


class _FakeLLM:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def with_structured_output(self, schema, include_raw=True):
        return _FakeStructured(self._behaviour)


def _reranker_with(parsed=None, error: Exception | None = None) -> LLMReranker:
    reranker = LLMReranker(
        LLMSettings(base_url="http://localhost:8000/v1", api_key="test")
    )

    def _behaviour():
        if error is not None:
            raise error
        return {"parsed": parsed, "raw": None}

    reranker._llm = _FakeLLM(_behaviour)  # noqa: SLF001 - 测试桩注入
    return reranker


def test_blend_scores_uses_llm_weight():
    assert blend_scores(0.0, 1.0, weight=0.6) == pytest.approx(0.6)
    assert blend_scores(1.0, 0.0, weight=0.6) == pytest.approx(0.4)
    assert blend_scores(1.0, 1.0) == 1.0


def test_build_reranker_requires_llm_configuration():
    assert build_reranker(Settings(llm=LLMSettings(enabled=False))) is None
    enabled = Settings(
        llm=LLMSettings(base_url="http://localhost:8000/v1", api_key="test")
    )
    assert build_reranker(enabled) is not None
    assert build_reranker(enabled.model_copy(update={"rerank_enabled": False})) is None


@pytest.mark.asyncio
async def test_llm_reranker_returns_scores_by_id(article_factory):
    candidates = [article_factory(f"人形机器人快讯 {index}") for index in range(3)]
    reranker = _reranker_with(
        _RerankOutput(results=[{"index": 2, "relevance": 0.9}, {"index": 0, "relevance": 0.1}])
    )
    scores = await reranker.rerank("人形机器人", [], candidates)
    assert scores == {candidates[2].id: 0.9, candidates[0].id: 0.1}


@pytest.mark.asyncio
async def test_llm_reranker_ignores_out_of_range_indices(article_factory):
    candidates = [article_factory("人形机器人快讯")]
    reranker = _reranker_with(_RerankOutput(results=[{"index": 7, "relevance": 1.0}]))
    with pytest.raises(RerankError):
        await reranker.rerank("人形机器人", [], candidates)


@pytest.mark.asyncio
async def test_llm_reranker_wraps_provider_errors(article_factory):
    candidates = [article_factory("人形机器人快讯")]
    reranker = _reranker_with(error=RuntimeError("boom"))
    with pytest.raises(RerankError):
        await reranker.rerank("人形机器人", [], candidates)


# ---------------------------------------------------------------------------
# 图集成
# ---------------------------------------------------------------------------
class _FakeReranker:
    name = "fake"
    uses_llm = True

    def __init__(self, boost_last: bool = True, error: Exception | None = None) -> None:
        self.boost_last = boost_last
        self.error = error
        self.calls = 0

    async def rerank(self, query, keywords, candidates, *, ctx=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        scores = {article.id: 0.1 for article in candidates}
        if self.boost_last:
            scores[candidates[-1].id] = 1.0
        return scores


def _settings() -> Settings:
    return Settings(
        sources=[SourceConfig(name="mock", type="mock")],
        llm=LLMSettings(enabled=False, api_key=None, base_url=None),
        cache_enabled=False,
        log_level="WARNING",
        relevance_threshold=0.1,
    )


@pytest.mark.asyncio
async def test_graph_reranker_reorders_results(request_factory):
    reranker = _FakeReranker()
    instance = await NewsAgent.create(_settings(), reranker=reranker)
    try:
        result = await instance.run(request_factory(skill="fetch_news", limit=5))
    finally:
        await instance.aclose()

    assert reranker.calls == 1
    assert len(result.articles) > 1
    # 词面分几乎全为 1.0，精排把最后一篇提到最前面
    boosted = result.articles[0]
    assert boosted.relevance == pytest.approx(1.0)
    assert result.metrics.get("articles_reranked", 0) >= 1
    scores = [article.relevance for article in result.articles]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_rerank_can_promote_article_beyond_the_limit(request_factory):
    """精排前会多取候选，因此 LLM 能把词面排名靠后的文章提回结果里。"""
    reranker = _FakeReranker()
    instance = await NewsAgent.create(_settings(), reranker=reranker)
    try:
        result = await instance.run(request_factory(skill="fetch_news", limit=2))
    finally:
        await instance.aclose()

    assert len(result.articles) == 2
    # 被 boost 的是词面排序里最后（最旧）的一篇，精排后才可能进入前二
    assert result.articles[0].relevance == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_graph_rerank_failure_degrades_to_lexical_order(request_factory):
    reranker = _FakeReranker(error=RuntimeError("judge down"))
    instance = await NewsAgent.create(_settings(), reranker=reranker)
    try:
        result = await instance.run(request_factory(skill="fetch_news", limit=5))
    finally:
        await instance.aclose()

    assert reranker.calls == 1
    assert any("精排" in warning for warning in result.warnings)
    scores = [article.relevance for article in result.articles]
    assert scores == sorted(scores, reverse=True)
    assert result.counts["analyzed"] == len(result.articles)


@pytest.mark.asyncio
async def test_no_reranker_keeps_lexical_order(request_factory):
    instance = await NewsAgent.create(_settings(), reranker=None)
    try:
        assert instance.reranker is None
        result = await instance.run(request_factory(skill="fetch_news", limit=5))
    finally:
        await instance.aclose()
    assert result.metrics.get("articles_reranked", 0) == 0
