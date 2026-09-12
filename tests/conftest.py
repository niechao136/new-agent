"""Shared fixtures: an offline, deterministic configuration for the test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from news_agent.config import LLMSettings, Settings, SourceConfig
from news_agent.graph.agent import NewsAgent
from news_agent.models import RawArticle, SkillRequest, make_article_id

MOCK_SOURCE = "mock"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that never touch the network and never call an LLM."""
    return Settings(
        sources=[SourceConfig(name=MOCK_SOURCE, type="mock")],
        llm=LLMSettings(enabled=False, api_key=None, base_url=None),
        cache_enabled=True,
        cache_path=str(tmp_path / "cache.sqlite3"),
        cache_ttl_s=60,
        log_level="WARNING",
        task_timeout_s=60,
        relevance_threshold=0.1,
    )


@pytest.fixture
def offline_settings(settings: Settings) -> Settings:
    """Same as ``settings`` but without the cache."""
    return settings.model_copy(update={"cache_enabled": False})


@pytest.fixture
async def agent(settings: Settings) -> AsyncIterator[NewsAgent]:
    instance = await NewsAgent.create(settings)
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.fixture
def request_factory() -> Callable[..., SkillRequest]:
    def _make(query: str = "人形机器人", **kwargs: Any) -> SkillRequest:
        payload: dict[str, Any] = {"query": query, "limit": 10, "language": "zh"}
        payload.update(kwargs)
        return SkillRequest(**payload)

    return _make


@pytest.fixture
def article_factory() -> Callable[..., RawArticle]:
    def _make(
        title: str,
        *,
        url: str | None = None,
        source: str = "unit",
        summary: str | None = None,
        content: str | None = None,
        published_at: datetime | None = None,
    ) -> RawArticle:
        url = url or f"https://example.com/{abs(hash(title)) % 10**8}"
        return RawArticle(
            id=make_article_id(url, title),
            title=title,
            url=url,
            source=source,
            summary=summary,
            content=content,
            published_at=published_at,
        )

    return _make
