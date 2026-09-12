"""Relevance scoring / filtering tests (filter_node)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from news_agent.models import SkillRequest, utcnow
from news_agent.relevance import (
    TOPIC_ROUTING_BASE,
    diversify,
    is_broad_query,
    longest_common_run,
    matched_topics,
    pick_relevant,
    rank_articles,
    relevance_score,
    select_diverse,
    select_relevant,
    topic_floor,
)


def test_score_prefers_on_topic_title(article_factory):
    query = "人形机器人"
    on_topic = article_factory("人形机器人量产在即", summary="多家公司公布量产计划")
    off_topic = article_factory("周末将迎来降温与降雨", summary="气象部门提醒注意保暖")
    assert relevance_score(on_topic, query) > relevance_score(off_topic, query)
    assert relevance_score(on_topic, query) > 0.6


def test_recency_bonus_applies(article_factory):
    query = "固态电池"
    fresh = article_factory("固态电池取得突破", published_at=utcnow() - timedelta(hours=3))
    old = article_factory("固态电池取得突破", published_at=utcnow() - timedelta(days=60))
    assert relevance_score(fresh, query) > relevance_score(old, query)


# ---------------------------------------------------------------------------
# 多关键词（跨语言/别名）匹配
# ---------------------------------------------------------------------------
def test_english_keyword_matches_chinese_query(article_factory):
    """中文查询 + 英文关键词应能召回英文报道（此前恒为 0 分）。"""
    query = "固态电池"
    english = article_factory(
        "Solid-state battery breakthrough announced",
        summary="A solid-state battery with higher energy density was unveiled.",
    )
    assert relevance_score(english, query) == 0.0
    assert relevance_score(english, query, keywords=["solid-state battery"]) > 0.6


def test_keywords_do_not_leak_to_unrelated_articles(article_factory):
    unrelated = article_factory("周末将迎来降温与降雨", summary="气象部门提醒注意保暖")
    assert relevance_score(unrelated, "固态电池", keywords=["solid-state battery"]) == 0.0


# ---------------------------------------------------------------------------
# 连续匹配约束（假 bigram 修复）
# ---------------------------------------------------------------------------
def test_fake_bigram_hit_is_not_enough(article_factory):
    """「特斯拉新闻」的跨界 bigram「拉新」不应让营销类文章通过阈值。"""
    query = "特斯拉新闻"
    corpus = [
        article_factory(f"特斯拉发布新款车型 {index}") for index in range(6)
    ] + [article_factory("增长黑客：产品拉新活动复盘")]
    scored = dict((article.title, score) for article, score in rank_articles(corpus, query))
    fake_hit = scored["增长黑客：产品拉新活动复盘"]
    real_hit = scored["特斯拉发布新款车型 0"]
    assert fake_hit < 0.2
    assert real_hit > fake_hit


def test_longest_common_run():
    assert longest_common_run("特斯拉新闻", "增长黑客产品拉新活动复盘") == 2
    assert longest_common_run("特斯拉新闻", "特斯拉发布新款车型") == 3
    assert longest_common_run("", "abc") == 0


def test_contiguity_discounts_boundary_matches(article_factory):
    """只有跨界 bigram 命中的文章，分数应显著低于短语命中的文章。"""
    query = "量子计算芯片"
    corpus = [article_factory(f"量子计算芯片研究进展 {index}") for index in range(5)]
    partial = article_factory("行业周报：芯片价格波动")
    scored = dict((a.title, s) for a, s in rank_articles([*corpus, partial], query))
    assert scored["行业周报：芯片价格波动"] < scored["量子计算芯片研究进展 0"] / 3


# ---------------------------------------------------------------------------
# 泛化查询的栏目路由
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    ["科技新闻", "财经新闻", "体育新闻", "人工智能", "AI芯片", "tech news", "sports news"],
)
def test_broad_queries_are_detected(query):
    assert is_broad_query(query)


@pytest.mark.parametrize(
    "query",
    ["人形机器人", "固态电池", "英伟达财报", "特斯拉新闻", "俄乌冲突"],
)
def test_specific_queries_are_not_broad(query):
    assert not is_broad_query(query)


def test_matched_topics():
    assert matched_topics("科技新闻") == {"tech"}
    assert matched_topics("财经新闻") == {"business"}
    assert matched_topics("人形机器人") == set()


def test_topic_floor_only_for_matching_broad_queries():
    assert topic_floor("科技新闻", "tech") == TOPIC_ROUTING_BASE
    assert topic_floor("科技新闻", "sports") == 0.0
    assert topic_floor("人形机器人", "tech") == 0.0


def test_topic_routing_rescues_english_articles_for_generic_query(article_factory):
    """「科技新闻」这类泛查询无法词面命中英文标题，靠栏目路由兜底。"""
    english_tech = article_factory(
        "Apple unveils new MacBook lineup", source="techcrunch"
    )
    english_sport = article_factory(
        "United win derby in stoppage time", source="bbc-sport"
    )
    topics = {"techcrunch": "tech", "bbc-sport": "sports"}

    assert relevance_score(english_tech, "科技新闻") == 0.0

    scored = dict(
        (article.source, score)
        for article, score in rank_articles(
            [english_tech, english_sport], "科技新闻", source_topics=topics
        )
    )
    assert scored["techcrunch"] == TOPIC_ROUTING_BASE
    assert scored["bbc-sport"] == 0.0


def test_topic_routing_does_not_apply_to_specific_queries(article_factory):
    english_tech = article_factory("Solid-state battery breakthrough", source="techcrunch")
    scored = rank_articles(
        [english_tech], "人形机器人", source_topics={"techcrunch": "tech"}
    )
    assert scored[0][1] == 0.0


def test_lexical_match_still_outranks_topic_floor(article_factory):
    on_topic = article_factory("科技新闻：某公司发布新一代芯片", source="ithome")
    routed = article_factory("Apple unveils new MacBook lineup", source="techcrunch")
    scored = dict(
        (article.source, score)
        for article, score in rank_articles(
            [routed, on_topic],
            "科技新闻",
            source_topics={"ithome": "tech", "techcrunch": "tech"},
        )
    )
    assert scored["ithome"] > TOPIC_ROUTING_BASE
    assert scored["techcrunch"] == TOPIC_ROUTING_BASE


# ---------------------------------------------------------------------------
# 来源权重与多样性
# ---------------------------------------------------------------------------
def test_source_weight_affects_ranking(article_factory):
    query = "人形机器人"
    low = article_factory("人形机器人量产在即", source="low")
    high = article_factory("人形机器人量产在即", source="high")
    ranked = rank_articles(
        [low, high], query, weights={"low": 0.6, "high": 1.2}
    )
    assert [article.source for article, _ in ranked] == ["high", "low"]


def test_diversify_defers_overflow_instead_of_dropping(article_factory):
    scored = [(article_factory(f"人形机器人快讯 {index}", source="same"), 0.9) for index in range(5)]
    scored.append((article_factory("人形机器人快讯 other", source="other"), 0.8))
    result = diversify(scored, per_source_cap=2)
    assert len(result) == len(scored)
    assert [article.source for article, _ in result][:3] == ["same", "same", "other"]


# ---------------------------------------------------------------------------
# pick_relevant / select_relevant
# ---------------------------------------------------------------------------
def test_select_relevant_drops_noise(article_factory):
    query = "人形机器人"
    articles = [
        article_factory(f"人形机器人行业观察 {index}") for index in range(4)
    ] + [article_factory("体育赛事综述：主队险胜"), article_factory("天气预报：周末降温")]

    selected, dropped, topped_up = select_relevant(
        articles, SkillRequest(query=query), threshold=0.3, limit=10
    )
    assert len(selected) == 4
    assert dropped == 2
    assert topped_up == 0


def test_select_relevant_tops_up_with_weak_but_on_topic_articles(article_factory):
    """低于阈值但有词面重叠的文章可以兜底，纯噪声不行。"""
    articles = [
        article_factory("行业观察：量子技术路线", summary="量子计算与传统计算路线之争"),
        article_factory("技术前沿：量子芯片展望", summary="量子计算的工程化仍待验证"),
    ]
    selected, dropped, topped_up = select_relevant(
        articles, SkillRequest(query="量子计算"), threshold=0.9, limit=10
    )
    assert topped_up == 2
    assert len(selected) == 2
    assert dropped == 0


def test_pure_noise_is_never_topped_up(article_factory):
    articles = [article_factory("毫不相关的新闻标题"), article_factory("另一条无关消息")]
    selected, dropped, topped_up = select_relevant(
        articles, SkillRequest(query="量子计算"), threshold=0.9, limit=10
    )
    assert selected == []
    assert topped_up == 0
    assert dropped == 2


def test_select_relevant_respects_limit(article_factory):
    articles = [article_factory(f"人形机器人快讯 {index}") for index in range(20)]
    selected, dropped, _ = select_relevant(
        articles, SkillRequest(query="人形机器人"), threshold=0.1, limit=5
    )
    assert len(selected) == 5
    assert dropped == 15


def test_select_diverse_respects_cap_and_fills_when_scarce(article_factory):
    dominant = [article_factory(f"人形机器人观察 {index}", source="a") for index in range(6)]
    others = [article_factory(f"人形机器人动态 {index}", source="b") for index in range(2)]
    picked = select_diverse([*dominant, *others], limit=6, per_source_cap=3)
    assert len(picked) == 6
    # 另一来源只有 2 篇，配额内凑不满 6 篇，因此用溢出项补 1 篇
    assert sum(1 for article in picked if article.source == "a") == 4
    assert sum(1 for article in picked if article.source == "b") == 2

    # 素材充足时严格生效
    plenty = [article_factory(f"人形机器人观察 {index}", source="a") for index in range(6)]
    plenty += [article_factory(f"人形机器人动态 {index}", source="b") for index in range(6)]
    strict = select_diverse(plenty, limit=6, per_source_cap=3)
    assert sum(1 for article in strict if article.source == "a") == 3
    assert sum(1 for article in strict if article.source == "b") == 3
    # 素材不足时用溢出项补满，而不是返回短列表
    only_dominant = [article_factory(f"人形机器人周报 {index}", source="a") for index in range(5)]
    assert len(select_diverse(only_dominant, limit=4, per_source_cap=2)) == 4
    assert select_diverse(only_dominant, limit=4, per_source_cap=None) == only_dominant[:4]


def test_pick_relevant_applies_source_cap(article_factory):
    scored = [
        (article_factory(f"人形机器人快讯 A{index}", source="a"), 0.9) for index in range(6)
    ] + [
        (article_factory(f"人形机器人快讯 B{index}", source="b"), 0.8) for index in range(4)
    ]
    selected, _, _ = pick_relevant(
        scored, threshold=0.1, limit=6, per_source_cap=3
    )
    sources = [article.source for article in selected]
    assert sources.count("a") == 3
    assert sources.count("b") == 3
