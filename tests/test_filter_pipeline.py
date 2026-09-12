"""筛选流水线的节点级行为：分数只算一次、结果按来源去偏、精排接线。"""

from __future__ import annotations

import pytest

from news_agent.models import SkillMode, SkillRequest


@pytest.mark.asyncio
async def test_filter_node_stores_relevance_scores(agent, article_factory):
    # 标题要足够不同，否则会被跨源去重合并成一篇
    titles = [
        "人形机器人产业观察",
        "人形机器人量产计划公布",
        "人形机器人技术路线之争",
        "人形机器人商业化提速",
        "人形机器人企业融资回暖",
        "人形机器人行业标准启动",
    ]
    articles = [article_factory(title) for title in titles]
    state = {
        "request": SkillRequest(query="人形机器人", limit=4),
        "raw_articles": articles,
    }
    out = await agent.nodes.filter_node(state, {"configurable": {}})

    scores = out["relevance_scores"]
    assert len(scores) == len(articles)
    assert all(score > 0 for score in scores.values())
    assert len(out["filtered_articles"]) <= 4


@pytest.mark.asyncio
async def test_analyze_node_reuses_filter_scores(agent, article_factory):
    """analyze 不再重算 relevance，直接复用 filter 阶段的分数。"""
    articles = [article_factory(f"人形机器人快讯 {index}") for index in range(3)]
    state = {
        "request": SkillRequest(query="人形机器人"),
        "filtered_articles": articles,
        "relevance_scores": {article.id: 0.42 for article in articles},
    }
    out = await agent.nodes.analyze_node(state, {"configurable": {}})

    assert out["analyzed_articles"]
    assert all(item.relevance == pytest.approx(0.42) for item in out["analyzed_articles"])


@pytest.mark.asyncio
async def test_format_node_reuses_scores_for_fetch_skill(agent, article_factory):
    article = article_factory("人形机器人快讯")
    state = {
        "request": SkillRequest(query="人形机器人", skill=SkillMode.FETCH),
        "filtered_articles": [article],
        "relevance_scores": {article.id: 0.77},
    }
    out = await agent.nodes.format_node(state, {"configurable": {}})

    assert out["result"].articles[0].relevance == pytest.approx(0.77)


@pytest.mark.asyncio
async def test_filter_node_prefers_higher_weight_source(agent, article_factory):
    """来源权重只影响排序：同分时高质量来源排在前面。"""
    low = article_factory("人形机器人产业观察", source="low")
    high = article_factory("人形机器人量产提速", source="high")
    agent.registry.weights = lambda: {"low": 0.6, "high": 1.2}  # type: ignore[method-assign]
    state = {
        "request": SkillRequest(query="人形机器人", limit=5),
        "raw_articles": [low, high],
    }
    out = await agent.nodes.filter_node(state, {"configurable": {}})

    assert [article.source for article in out["filtered_articles"]][0] == "high"


@pytest.mark.asyncio
async def test_per_source_cap_is_ceil_third(agent):
    assert agent.nodes._per_source_cap(9) == 3  # noqa: SLF001
    assert agent.nodes._per_source_cap(10) == 4  # noqa: SLF001
    assert agent.nodes._per_source_cap(1) == 3  # noqa: SLF001
