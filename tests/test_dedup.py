"""De-duplication + text helper tests (TODO item 6)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from news_agent.dedup import Deduplicator, hamming_distance, simhash, title_similarity
from news_agent.models import utcnow
from news_agent.text_utils import normalize_url, tokenize


def test_normalize_url_strips_tracking_and_scheme_noise():
    left = "HTTPS://www.Example.com/news/1/?utm_source=x&id=7#frag"
    right = "https://example.com/news/1?id=7"
    assert normalize_url(left) == normalize_url(right)
    assert normalize_url("") == ""


def test_tokenize_is_cjk_aware():
    tokens = tokenize("人形机器人产业迎来增长")
    assert "人形" in tokens and "增长" in tokens
    assert "产业" in tokens
    assert tokenize("The quick brown fox") == ["quick", "brown", "fox"]


def test_simhash_is_stable_and_sensitive():
    a = "人形机器人产业迎来新一轮增长"
    b = "人形机器人产业迎来新一轮增长，机构上调预期"
    c = "本周末将迎来降温与降雨"
    assert simhash(a) == simhash(a)
    assert hamming_distance(simhash(a), simhash(b)) < hamming_distance(simhash(a), simhash(c))


def test_title_similarity_handles_containment():
    assert title_similarity("人形机器人增长", "人形机器人增长（转载）") > 0.9
    assert title_similarity("人形机器人增长", "周末天气降温降雨") < 0.5


def test_dedupe_collapses_url_title_and_fuzzy_duplicates(article_factory):
    articles = [
        article_factory("人形机器人产业迎来新一轮增长", url="https://a.com/1", source="a"),
        # exact URL duplicate (tracking params only)
        article_factory("另一个标题", url="https://a.com/1?utm_source=feed", source="b"),
        # exact normalised title duplicate
        article_factory(
            "人形机器人产业迎来新一轮增长",
            url="https://c.com/2",
            source="c",
            summary="摘要",
        ),
        # fuzzy duplicate (near identical title)
        article_factory(
            "人形机器人产业迎来新一轮增长！",
            url="https://d.com/3",
            source="d",
            content="正文内容",
        ),
        # unrelated
        article_factory("周末将迎来降温与降雨", url="https://e.com/4", source="e"),
    ]

    deduped = Deduplicator().dedupe(articles)

    assert len(deduped) == 2
    head = deduped[0]
    assert head.source == "a"
    # duplicate sources are tracked and the richest payload is merged in
    assert set(head.duplicate_sources) >= {"c", "d"}
    assert head.content == "正文内容"


def test_dedupe_keeps_earliest_publication_time(article_factory):
    fresh = utcnow() - timedelta(hours=1)
    older = utcnow() - timedelta(hours=9)
    articles = [
        article_factory("同一事件的报道", url="https://a.com/1", published_at=fresh),
        article_factory("同一事件的报道", url="https://b.com/2", published_at=older),
    ]
    deduped = Deduplicator().dedupe(articles)
    assert len(deduped) == 1
    assert deduped[0].published_at == older


@pytest.mark.parametrize("distance", [0, 1, 5])
def test_hamming_distance_math(distance: int):
    assert hamming_distance(0, (1 << distance) - 1 if distance < 64 else 0) == distance
