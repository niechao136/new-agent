"""Analyzer tests: map-reduce summarisation, batching and LLM degradation."""

from __future__ import annotations

from typing import Any

import pytest

from news_agent.analyzer import (
    FallbackAnalyzer,
    HeuristicAnalyzer,
    LLMAnalyzer,
    LLMError,
    build_analyzer,
    extract_entities,
    light_analyze,
)
from news_agent.config import LLMSettings, Settings
from news_agent.models import AnalyzedArticle, ErrorCode, SkillRequest
from news_agent.runtime import RunContext


def _analyzed(count: int, title: str = "人形机器人产业进展") -> list[AnalyzedArticle]:
    return [
        AnalyzedArticle(id=f"a{index}", title=f"{title} {index}", url=f"https://e.com/{index}", source="unit")
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# heuristics
# ---------------------------------------------------------------------------
async def test_heuristic_analyze_extracts_structure(article_factory, request_factory):
    article = article_factory(
        "人形机器人产业迎来新一轮增长",
        summary="多家机构上调预期，订单量显著增长，产业链持续回暖。",
        content="分析师指出，下游需求是第一驱动因素。",
    )
    results = await HeuristicAnalyzer().analyze(request_factory(), [article])

    assert len(results) == 1
    item = results[0]
    assert item.sentiment == "positive"
    assert item.sentiment_score > 0
    assert item.entities
    assert item.key_points
    assert item.summary


async def test_heuristic_analyze_detects_negative(article_factory, request_factory):
    article = article_factory(
        "人形机器人项目出现延期",
        summary="供应链压力显现，部分项目延期，合规成本上升，企业面临风险。",
    )
    results = await HeuristicAnalyzer().analyze(request_factory(), [article])
    assert results[0].sentiment == "negative"


def test_extract_entities_prefers_query_related_terms():
    entities = extract_entities(
        "人形机器人量产在即 Tesla Optimus 计划扩产", query="人形机器人"
    )
    assert entities
    assert len(entities) <= 5


def test_light_analyze_returns_metadata_only(article_factory):
    items = light_analyze([article_factory("标题", summary="摘要")])
    assert items[0].summary == "摘要"
    assert items[0].entities == []
    assert items[0].sentiment == "neutral"


# ---------------------------------------------------------------------------
# map-reduce summarisation (TODO item 11)
# ---------------------------------------------------------------------------
async def test_summarise_uses_map_reduce_chunking(monkeypatch):
    analyzer = LLMAnalyzer(
        LLMSettings(model="test-model", api_key="test", summary_chunk_size=5)
    )
    articles = _analyzed(12)
    chunk_sizes: list[int] = []
    reduced_payloads: list[str] = []

    async def fake_chunk(request, chunk, ctx=None):
        chunk_sizes.append(len(chunk))
        return f"partial-{len(chunk_sizes)}"

    async def fake_reduce(request, payload, ctx=None):
        reduced_payloads.append(payload)
        return "FINAL SUMMARY"

    monkeypatch.setattr(analyzer, "_summarize_chunk", fake_chunk)
    monkeypatch.setattr(analyzer, "_reduce", fake_reduce)

    summary = await analyzer.summarize(SkillRequest(query="人形机器人"), articles)

    assert chunk_sizes == [5, 5, 2]
    assert len(reduced_payloads) == 1
    assert "partial-1" in reduced_payloads[0]
    assert "partial-3" in reduced_payloads[0]
    assert summary == "FINAL SUMMARY"


async def test_summarise_degrades_when_reduce_fails(monkeypatch):
    analyzer = LLMAnalyzer(
        LLMSettings(model="test-model", api_key="test", summary_chunk_size=6)
    )
    articles = _analyzed(6)

    async def fake_chunk(request, chunk, ctx=None):
        return "partial text"

    async def fake_reduce(request, payload, ctx=None):
        raise RuntimeError("reduce exploded")

    monkeypatch.setattr(analyzer, "_summarize_chunk", fake_chunk)
    monkeypatch.setattr(analyzer, "_reduce", fake_reduce)

    ctx = RunContext()
    summary = await analyzer.summarize(SkillRequest(query="q"), articles, ctx)

    assert "partial text" in summary
    assert any("退化" in warning for warning in ctx.warnings)


async def test_summarise_raises_when_every_chunk_fails(monkeypatch):
    analyzer = LLMAnalyzer(LLMSettings(model="m", api_key="k"))

    async def boom(request, chunk, ctx=None):
        raise RuntimeError("nope")

    monkeypatch.setattr(analyzer, "_summarize_chunk", boom)

    ctx = RunContext()
    with pytest.raises(LLMError):
        await analyzer.summarize(SkillRequest(query="q"), _analyzed(4), ctx)
    assert any(error.code is ErrorCode.LLM_FAILED for error in ctx.errors)


# ---------------------------------------------------------------------------
# batching + concurrency + degradation
# ---------------------------------------------------------------------------
async def test_llm_analyze_batches_and_backfills_with_heuristics(monkeypatch, article_factory, request_factory):
    from news_agent.analyzer import _BatchAnalysis, _ItemAnalysis

    analyzer = LLMAnalyzer(
        LLMSettings(model="m", api_key="k", batch_size=2, concurrency=1)
    )
    articles = [
        article_factory(f"人形机器人新闻 {index}", summary="增长明显") for index in range(5)
    ]
    seen_batches: list[int] = []

    async def fake_call_batch(request, items, ctx=None):
        seen_batches.append(len(items))
        # only answer the first item of each batch -> forces heuristic backfill
        index = items[0][0]
        return _BatchAnalysis(
            results=[
                _ItemAnalysis(
                    index=index,
                    entities=["示例实体"],
                    events=["示例事件"],
                    sentiment="positive",
                    sentiment_score=0.8,
                    stance="看好",
                    key_points=["要点"],
                    summary="LLM 摘要",
                )
            ]
        )

    monkeypatch.setattr(analyzer, "_call_batch", fake_call_batch)

    ctx = RunContext()
    results = await analyzer.analyze(request_factory(), articles, ctx)

    assert len(results) == len(articles)
    assert sorted(seen_batches) == [1, 2, 2]
    llm_items = [item for item in results if item.entities == ["示例实体"]]
    assert len(llm_items) == 3
    assert any("启发式" in warning for warning in ctx.warnings)
    # every result keeps the article identity
    assert {item.id for item in results} == {article.id for article in articles}


async def test_fallback_analyzer_degrades_on_llm_failure(article_factory, request_factory):
    class BoomAnalyzer:
        name = "boom"
        uses_llm = True

        async def analyze(self, *args: Any, **kwargs: Any) -> list[AnalyzedArticle]:
            raise RuntimeError("llm is down")

        async def summarize(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("llm is down")

    analyzer = FallbackAnalyzer(BoomAnalyzer(), HeuristicAnalyzer())
    ctx = RunContext()
    articles = [
        article_factory("人形机器人产业迎来新一轮增长", summary="订单量显著增长，机构上调预期")
    ]

    results = await analyzer.analyze(request_factory(), articles, ctx)
    assert len(results) == 1
    assert results[0].sentiment == "positive"
    assert any(error.code is ErrorCode.LLM_FAILED for error in ctx.errors)

    summary = await analyzer.summarize(request_factory(), results, ctx)
    assert "人形机器人" in summary
    assert any("降级" in warning for warning in ctx.warnings)


def test_build_analyzer_switches_on_configuration():
    heuristic = build_analyzer(Settings(llm=LLMSettings(enabled=False)))
    assert isinstance(heuristic, HeuristicAnalyzer)

    hybrid = build_analyzer(Settings(llm=LLMSettings(enabled=True, api_key="k")))
    assert isinstance(hybrid, FallbackAnalyzer)

    disabled = build_analyzer(Settings(llm=LLMSettings(enabled=False, api_key="k")))
    assert isinstance(disabled, HeuristicAnalyzer)


async def test_unconfigured_llm_analyzer_raises_llm_error(monkeypatch, article_factory):
    """A configured-but-broken endpoint must raise, not crash the graph."""
    analyzer = LLMAnalyzer(LLMSettings(model="m", api_key="k", max_retries=0))

    class FailingChain:
        async def ainvoke(self, messages: Any) -> Any:
            raise RuntimeError("connection refused")

    monkeypatch.setattr(analyzer, "_structured", lambda method: FailingChain())

    article = article_factory("人形机器人进展", summary="增长明显")
    with pytest.raises(LLMError) as excinfo:
        await analyzer._call_batch(  # noqa: SLF001
            SkillRequest(query="q"), [(0, article)]
        )
    assert "connection refused" in str(excinfo.value) or "attempt" in str(excinfo.value)
