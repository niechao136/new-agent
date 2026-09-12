"""The ``NewsSource`` abstraction (TODO item 5).

Adding a new source = subclass :class:`NewsSource`, implement ``_fetch`` and
register it in :data:`news_agent.sources.registry.SOURCE_TYPES`.  Everything
else (retries, timeout, error mapping, metrics) is provided here.
"""

from __future__ import annotations

import abc
import asyncio
from datetime import datetime
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config import SourceConfig
from ..models import (
    ErrorCode,
    ErrorInfo,
    RawArticle,
    SkillRequest,
    ensure_aware,
    make_article_id,
    utcnow,
)
from ..text_utils import clean_text, normalize_url, strip_html
from .http import HttpClientProtocol, SourceError


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, SourceError):
        return exc.retryable
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError))


class NewsSource(abc.ABC):
    """Uniform interface implemented by every acquisition backend."""

    type: str = "base"

    def __init__(
        self, config: SourceConfig, settings: Any, http: HttpClientProtocol
    ) -> None:
        self.config = config
        self.settings = settings
        self.http = http
        self.name = config.name
        self.weight = config.weight
        self.topic = config.topic
        self.last_errors: list[ErrorInfo] = []
        self.max_attempts = 3
        self._timeout = config.timeout_s or settings.fetch_timeout_s

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
        """Fetch articles with retry + hard timeout, never raising."""
        self.last_errors = []
        if not self.config.enabled:
            return []
        started = utcnow()
        articles: list[RawArticle] = []
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential_jitter(initial=0.4, max=4.0, jitter=0.2),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    articles = await asyncio.wait_for(
                        self._fetch(request, limit), timeout=self._timeout
                    )
        except asyncio.TimeoutError:
            self.last_errors.append(
                ErrorInfo(
                    code=ErrorCode.FETCH_TIMEOUT,
                    message=f"source '{self.name}' timed out after {self._timeout:.0f}s",
                    source=self.name,
                    stage="fetch",
                    retryable=True,
                )
            )
            return []
        except SourceError as exc:
            self.last_errors.append(
                ErrorInfo(
                    code=exc.code,
                    message=f"source '{self.name}': {exc.message}",
                    source=self.name,
                    stage="fetch",
                    retryable=exc.retryable,
                )
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive
            self.last_errors.append(
                ErrorInfo(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"source '{self.name}' failed: {exc!r}",
                    source=self.name,
                    stage="fetch",
                    retryable=False,
                )
            )
            return []

        cleaned = [
            article for article in articles if article.title or article.url
        ]
        for article in cleaned:
            article.source = article.source or self.name
            article.fetched_at = started
        return cleaned

    @abc.abstractmethod
    async def _fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
        """Source specific implementation."""

    # ------------------------------------------------------------------
    # helpers for subclasses
    # ------------------------------------------------------------------
    def _make_article(
        self,
        *,
        title: str,
        url: str,
        published_at: datetime | None = None,
        summary: str | None = None,
        content: str | None = None,
        language: str | None = None,
        author: str | None = None,
        image_url: str | None = None,
        source_name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RawArticle | None:
        title = clean_text(strip_html(title))
        url = (url or "").strip()
        if not title or not url:
            return None
        return RawArticle(
            id=make_article_id(url, title),
            title=title,
            url=url,
            source=source_name or self.name,
            published_at=ensure_aware(published_at),
            summary=clean_text(strip_html(summary))[:1000] or None,
            content=clean_text(strip_html(content))[:6000] or None,
            language=language,
            author=clean_text(author) or None,
            image_url=image_url,
            extra=extra or {},
        )

    @staticmethod
    def _within_window(article: RawArticle, request: SkillRequest) -> bool:
        if article.published_at is None:
            return True
        if request.since and article.published_at < request.since:
            return False
        if request.until and article.published_at > request.until:
            return False
        return True

    @staticmethod
    def _dedupe_urls(articles: list[RawArticle]) -> list[RawArticle]:
        seen: set[str] = set()
        unique: list[RawArticle] = []
        for article in articles:
            key = normalize_url(article.url) or article.id
            if key in seen:
                continue
            seen.add(key)
            unique.append(article)
        return unique
