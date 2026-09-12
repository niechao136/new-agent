"""Relevance scoring / filtering tests (TODO item 9, filter_node)."""

from __future__ import annotations

from datetime import timedelta

from news_agent.models import SkillRequest, utcnow
from news_agent.relevance import relevance_score, select_relevant


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


def test_select_relevant_tops_up_when_below_threshold(article_factory):
    articles = [article_factory("毫不相关的新闻标题"), article_factory("另一条无关消息")]
    selected, dropped, topped_up = select_relevant(
        articles, SkillRequest(query="量子计算"), threshold=0.9, limit=10
    )
    assert topped_up == 2
    assert len(selected) == 2
    assert dropped == 0


def test_select_relevant_respects_limit(article_factory):
    articles = [article_factory(f"人形机器人快讯 {index}") for index in range(20)]
    selected, dropped, _ = select_relevant(
        articles, SkillRequest(query="人形机器人"), threshold=0.1, limit=5
    )
    assert len(selected) == 5
    assert dropped == 15
