"""SQLite cache + incremental history tests (TODO item 8)."""

from __future__ import annotations

import pytest

from news_agent.cache import SqliteCache


@pytest.fixture
async def cache(tmp_path):
    instance = SqliteCache(tmp_path / "cache.sqlite3", ttl_s=60)
    await instance.init()
    try:
        yield instance
    finally:
        await instance.aclose()


async def test_article_cache_roundtrip(cache, article_factory):
    articles = [article_factory("标题一"), article_factory("标题二")]
    await cache.set_articles("k1", articles)

    restored = await cache.get_articles("k1")
    assert restored is not None
    assert [item.title for item in restored] == ["标题一", "标题二"]
    assert restored[0].url == articles[0].url

    assert await cache.get_articles("missing") is None


async def test_expired_entries_are_ignored(tmp_path, article_factory):
    cache = SqliteCache(tmp_path / "cache.sqlite3", ttl_s=60)
    await cache.init()
    try:
        await cache.set_articles("k1", [article_factory("会过期的标题")])
        assert await cache.get_articles("k1") is not None

        # a negative TTL makes every entry stale regardless of clock resolution
        cache.ttl_s = -1
        assert await cache.get_articles("k1") is None
        assert await cache.stats() == {
            "article_cache": 0,
            "result_cache": 0,
            "article_history": 0,
        }
    finally:
        await cache.aclose()


async def test_history_reports_only_new_articles(cache, article_factory):
    first = [article_factory("新闻 A"), article_factory("新闻 B")]
    new_ids = await cache.upsert_history(first)
    assert len(new_ids) == 2

    # second run: one known, one new
    second = [first[0], article_factory("新闻 C")]
    new_ids = await cache.upsert_history(second)
    assert len(new_ids) == 1
    assert new_ids[0] == second[1].id

    stats = await cache.stats()
    assert stats["article_history"] == 3


async def test_result_cache_roundtrip(cache):
    from news_agent.models import NewsResult, SkillMode

    result = NewsResult(query="测试", mode=SkillMode.SUMMARIZE, summary="摘要")
    await cache.set_result("rk", result)

    restored = await cache.get_result("rk")
    assert restored is not None
    assert restored.query == "测试"
    assert restored.summary == "摘要"


async def test_disabled_cache_is_a_noop(tmp_path, article_factory):
    cache = SqliteCache(tmp_path / "cache.sqlite3", enabled=False)
    await cache.init()
    await cache.set_articles("k", [article_factory("标题")])
    assert await cache.get_articles("k") is None
    assert cache.available is False
