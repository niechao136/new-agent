"""Source layer tests with mocked HTTP responses (TODO item 19)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import pytest

from news_agent.config import LLMSettings, Settings, SourceConfig
from news_agent.models import ErrorCode, SkillRequest, utcnow
from news_agent.sources import MockSource, RSSSource
from news_agent.sources.api import GNewsSource, NewsAPISource
from news_agent.sources.http import SourceError

RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Mock Feed</title>
    <item>
      <title>人形机器人量产在即 - 示例媒体</title>
      <link>https://example.com/news/1?utm_source=rss</link>
      <description>&lt;p&gt;多家公司公布&lt;b&gt;量产&lt;/b&gt;计划&lt;/p&gt;</description>
      <pubDate>{fresh}</pubDate>
      <source url="https://example.com">示例媒体</source>
    </item>
    <item>
      <title>无关的旧闻</title>
      <link>https://example.com/news/2</link>
      <description>很久以前的消息</description>
      <pubDate>{stale}</pubDate>
    </item>
  </channel>
</rss>
"""


class StubResponse:
    """Structurally satisfies :class:`news_agent.sources.http.HttpResponse`."""

    def __init__(
        self,
        *,
        content: bytes = b"",
        payload: Any = None,
        status_code: int = 200,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - not used
        return None


class StubHttp:
    """Structurally satisfies :class:`news_agent.sources.http.HttpClientProtocol`."""

    def __init__(self, response: StubResponse | Exception | None) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> StubResponse:
        self.calls.append((url, dict(params or {})))
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is None:  # pragma: no cover - sources that never fetch
            raise AssertionError("StubHttp was called without a configured response")
        return self.response

    async def aclose(self) -> None:
        return None


@pytest.fixture
def base_settings(tmp_path) -> Settings:
    return Settings(
        sources=[],
        llm=LLMSettings(enabled=False),
        cache_enabled=False,
        cache_path=str(tmp_path / "c.sqlite3"),
        log_level="WARNING",
    )


async def test_rss_source_parses_filters_and_cleans(base_settings, monkeypatch):
    from email.utils import format_datetime

    fresh = format_datetime(utcnow() - timedelta(hours=2))
    stale = format_datetime(utcnow() - timedelta(days=40))
    body = RSS_FEED.format(fresh=fresh, stale=stale).encode("utf-8")

    http = StubHttp(StubResponse(content=body))
    source = RSSSource(
        SourceConfig(
            name="rss",
            type="rss",
            feeds=["https://feed.example.com/search?q={query}&hl={hl}"],
        ),
        base_settings,
        http,
    )
    request = SkillRequest(query="人形机器人", since=utcnow() - timedelta(days=7))

    articles = await source.fetch(request, limit=5)

    assert len(articles) == 1
    article = articles[0]
    # publisher suffix stripped, html cleaned, tracking params kept in the raw url
    assert article.title == "人形机器人量产在即"
    assert article.source == "示例媒体"
    assert article.summary == "多家公司公布 量产 计划"
    assert "utm_source=rss" in article.url
    assert source.last_errors == []
    # the {query}/{hl} template was rendered
    assert "q=%E4%BA%BA%E5%BD%A2%E6%9C%BA%E5%99%A8%E4%BA%BA" in http.calls[0][0]


async def test_rss_source_reports_parse_errors(base_settings):
    source = RSSSource(
        SourceConfig(name="rss", type="rss", feeds=["https://feed.example.com/rss"]),
        base_settings,
        StubHttp(StubResponse(content=b"<html>not a feed</html>")),
    )
    source.max_attempts = 1

    articles = await source.fetch(SkillRequest(query="x"), limit=5)

    assert articles == []
    assert source.last_errors
    assert source.last_errors[0].source == "rss"


async def test_source_swallows_and_classifies_transport_errors(base_settings):
    source = RSSSource(
        SourceConfig(name="rss", type="rss", feeds=["https://feed.example.com/rss"]),
        base_settings,
        StubHttp(SourceError(ErrorCode.FETCH_TIMEOUT, "boom", retryable=False)),
    )
    source.max_attempts = 1

    articles = await source.fetch(SkillRequest(query="x"), limit=5)
    assert articles == []


async def test_newsapi_source_maps_payload(base_settings):
    payload = {
        "status": "ok",
        "articles": [
            {
                "title": "Solid state battery breakthrough",
                "url": "https://news.example.com/a",
                "publishedAt": utcnow().isoformat(),
                "description": "desc",
                "content": "content",
                "author": "reporter",
                "urlToImage": "https://img.example.com/1.png",
                "source": {"name": "Example News"},
            }
        ],
    }
    http = StubHttp(StubResponse(content=b"{}", payload=payload))
    source = NewsAPISource(
        SourceConfig(name="newsapi", type="newsapi", api_key="key"), base_settings, http
    )

    articles = await source.fetch(SkillRequest(query="battery", language="en"), limit=5)

    assert len(articles) == 1
    assert articles[0].source == "Example News"
    assert articles[0].image_url == "https://img.example.com/1.png"
    assert http.calls[0][1]["apiKey"] == "key"


async def test_newsapi_source_requires_key(base_settings):
    source = NewsAPISource(
        SourceConfig(name="newsapi", type="newsapi", api_key=None),
        base_settings,
        StubHttp(StubResponse(payload={"status": "error"})),
    )
    source.max_attempts = 1
    articles = await source.fetch(SkillRequest(query="x"), limit=5)
    assert articles == []
    assert source.last_errors[0].code.value == "invalid_request"


async def test_gnews_rate_limit_is_classified(base_settings):
    source = GNewsSource(
        SourceConfig(name="gnews", type="gnews", api_key="key"),
        base_settings,
        StubHttp(SourceError(ErrorCode.RATE_LIMITED, "429", retryable=False)),
    )
    source.max_attempts = 1
    await source.fetch(SkillRequest(query="x"), limit=5)
    assert source.last_errors[0].code.value == "rate_limited"


async def test_mock_source_includes_duplicate_and_noise(base_settings):
    source = MockSource(SourceConfig(name="mock", type="mock"), base_settings, StubHttp(None))
    articles = await source.fetch(SkillRequest(query="人形机器人", language="zh"), limit=8)

    assert len(articles) >= 6
    assert any("duplicate" in article.url for article in articles)
    assert any("天气" in article.title for article in articles)
    # deterministic
    again = await source.fetch(SkillRequest(query="人形机器人", language="zh"), limit=8)
    assert [item.id for item in again] == [item.id for item in articles]
