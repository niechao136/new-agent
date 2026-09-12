"""筛选流水线的节点级行为：分数只算一次、结果按来源去偏、精排接线。"""

from __future__ import annotations

import pytest

from news_agent.models import ErrorCode, SkillMode, SkillRequest
from news_agent.runtime import RunContext


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


# ---------------------------------------------------------------------------
# 垃圾内容过滤
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_filter_node_removes_gambling_spam(agent, article_factory):
    good = article_factory("人形机器人产业观察")
    spam = article_factory(
        "正规买球万博APP科技新闻摘要合集", summary="提供下注与投注参考。"
    )
    state = {
        "request": SkillRequest(query="人形机器人", limit=5),
        "raw_articles": [good, spam],
    }
    out = await agent.nodes.filter_node(state, {"configurable": {}})

    assert out["spam_removed"] == 1
    assert [article.id for article in out["filtered_articles"]] == [good.id]


@pytest.mark.asyncio
async def test_spam_filter_can_be_disabled(monkeypatch, article_factory):
    from news_agent.config import LLMSettings, Settings

    from news_agent.graph.agent import NewsAgent

    settings = Settings(
        sources=[],
        llm=LLMSettings(enabled=False),
        cache_enabled=False,
        log_level="WARNING",
        spam_filter_enabled=False,
        relevance_threshold=0.0,
    )
    instance = await NewsAgent.create(settings)
    try:
        spam = article_factory("正规买球万博APP科技新闻摘要合集", summary="提供下注与投注参考。")
        state = {
            "request": SkillRequest(query="科技新闻", limit=5),
            "raw_articles": [spam],
        }
        out = await instance.nodes.filter_node(state, {"configurable": {}})
        assert out["spam_removed"] == 0
    finally:
        await instance.aclose()


# ---------------------------------------------------------------------------
# 线上真实场景回归：查询「科技新闻」
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reported_broad_query_scenario(agent, article_factory):
    """复现线上「科技新闻」的结果：赌博 SEO 页必须被剔除，英文科技源要能入选。"""
    gambling_1 = article_factory(
        "科技新闻简报：m88JDB电子引领体育数字化革新- 体坛网",
        summary="文章简报报道了m88JDB电子引领体育数字化革新的相关科技新闻。",
        source="体坛加",
    )
    gambling_2 = article_factory(
        "正规买球万博APP科技新闻摘要合集：AI革新、融资合作与全球扩张",
        source="体坛加",
    )
    legit = article_factory(
        "AI赋能历史经典产业 给千年技艺插上科技翅膀", source="新华网"
    )
    english_tech = article_factory(
        "Apple unveils new MacBook lineup", source="techcrunch"
    )

    agent.registry.topics = lambda: {"techcrunch": "tech"}  # type: ignore[method-assign]
    agent.registry.weights = lambda: {}  # type: ignore[method-assign]
    state = {
        "request": SkillRequest(query="科技新闻", limit=10),
        "raw_articles": [gambling_1, gambling_2, legit, english_tech],
    }
    out = await agent.nodes.filter_node(state, {"configurable": {}})

    assert out["spam_removed"] == 2
    kept = {article.source for article in out["filtered_articles"]}
    assert "体坛加" not in kept
    assert "新华网" in kept
    assert "techcrunch" in kept


# ---------------------------------------------------------------------------
# 泛化查询的栏目路由（节点级）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_filter_node_routes_broad_query_by_topic(agent, article_factory):
    english_tech = article_factory(
        "Apple unveils new MacBook lineup", source="techcrunch"
    )
    english_sport = article_factory(
        "United win derby in stoppage time", source="bbc-sport"
    )
    agent.registry.topics = lambda: {  # type: ignore[method-assign]
        "techcrunch": "tech",
        "bbc-sport": "sports",
    }
    agent.registry.weights = lambda: {}  # type: ignore[method-assign]
    state = {
        "request": SkillRequest(query="科技新闻", limit=5),
        "raw_articles": [english_tech, english_sport],
    }
    out = await agent.nodes.filter_node(state, {"configurable": {}})

    assert [article.source for article in out["filtered_articles"]] == ["techcrunch"]


# ---------------------------------------------------------------------------
# 降级语义：只有结果真的受影响才算降级
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_minor_source_failures_do_not_degrade(agent, article_factory):
    article = article_factory("人形机器人产业观察")
    ctx = RunContext()
    ctx.add_error(ErrorCode.SOURCE_UNAVAILABLE, "s1 down", source="s1")
    ctx.add_error(ErrorCode.FETCH_TIMEOUT, "s2 timeout", source="s2")
    state = {
        "request": SkillRequest(query="人形机器人", skill=SkillMode.FETCH),
        "raw_articles": [article],
        "filtered_articles": [article],
        "sources_attempted": 20,
    }
    out = await agent.nodes.format_node(state, {"configurable": {"ctx": ctx}})

    assert out["result"].degraded is False
    assert any("新闻源" in warning for warning in out["result"].warnings)


@pytest.mark.asyncio
async def test_majority_source_failures_degrade(agent, article_factory):
    article = article_factory("人形机器人产业观察")
    ctx = RunContext()
    for index in range(3):
        ctx.add_error(ErrorCode.SOURCE_UNAVAILABLE, "down", source=f"s{index}")
    state = {
        "request": SkillRequest(query="人形机器人", skill=SkillMode.FETCH),
        "raw_articles": [article],
        "filtered_articles": [article],
        "sources_attempted": 4,
    }
    out = await agent.nodes.format_node(state, {"configurable": {"ctx": ctx}})

    assert out["result"].degraded is True


@pytest.mark.asyncio
async def test_empty_result_is_still_degraded(agent):
    ctx = RunContext()
    state = {
        "request": SkillRequest(query="人形机器人", skill=SkillMode.FETCH),
        "raw_articles": [],
        "filtered_articles": [],
        "sources_attempted": 10,
    }
    out = await agent.nodes.format_node(state, {"configurable": {"ctx": ctx}})
    assert out["result"].degraded is True
