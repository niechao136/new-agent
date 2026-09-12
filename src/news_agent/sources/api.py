"""Structured news API sources: NewsAPI.org and GNews.io.

Both are optional and only enabled when an API key is provided through the
environment (``NEWS_AGENT_NEWSAPI_KEY`` / ``NEWS_AGENT_GNEWS_KEY``).
"""

from __future__ import annotations

from typing import Any

from ..models import ErrorCode, RawArticle, SkillRequest
from .base import NewsSource
from .http import SourceError

_LANG_MAP = {"zh": "zh", "zh-cn": "zh", "zh-tw": "zh", "en": "en"}


class NewsAPISource(NewsSource):
    type = "newsapi"

    @property
    def base_url(self) -> str:
        return self.config.base_url or "https://newsapi.org/v2/everything"

    async def _fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
        api_key = self.config.api_key
        if not api_key:
            raise SourceError(
                ErrorCode.INVALID_REQUEST,
                "NewsAPI key is not configured",
                retryable=False,
            )
        params: dict[str, Any] = {
            "q": request.query,
            "pageSize": max(1, min(100, limit)),
            "sortBy": "publishedAt",
            "apiKey": api_key,
        }
        if request.since:
            params["from"] = request.since.isoformat()
        if request.until:
            params["to"] = request.until.isoformat()
        if request.language:
            params["language"] = _LANG_MAP.get(request.language.lower(), "en")

        response = await self.http.get(self.base_url, params=params, timeout=self._timeout)
        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceError(
                ErrorCode.PARSE_ERROR, "NewsAPI returned malformed JSON", retryable=True
            ) from exc
        if payload.get("status") != "ok":
            code = str(payload.get("code", "")).lower()
            raise SourceError(
                ErrorCode.RATE_LIMITED if "rate" in code else ErrorCode.SOURCE_UNAVAILABLE,
                f"NewsAPI error: {payload.get('message') or payload.get('code')}",
                retryable=True,
            )

        articles: list[RawArticle] = []
        for item in payload.get("articles") or []:
            article = self._make_article(
                title=item.get("title") or "",
                url=item.get("url") or "",
                published_at=_parse_dt(item.get("publishedAt")),
                summary=item.get("description"),
                content=item.get("content"),
                language=request.language,
                author=item.get("author"),
                image_url=item.get("urlToImage"),
                source_name=(item.get("source") or {}).get("name") or self.name,
            )
            if article and self._within_window(article, request):
                articles.append(article)
        return self._dedupe_urls(articles)


class GNewsSource(NewsSource):
    type = "gnews"

    @property
    def base_url(self) -> str:
        return self.config.base_url or "https://gnews.io/api/v4/search"

    async def _fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
        api_key = self.config.api_key
        if not api_key:
            raise SourceError(
                ErrorCode.INVALID_REQUEST,
                "GNews key is not configured",
                retryable=False,
            )
        params: dict[str, Any] = {
            "q": request.query,
            "lang": _LANG_MAP.get((request.language or "en").lower(), "en"),
            "max": max(1, min(100, limit)),
            "sortby": "publishedAt",
            "apikey": api_key,
        }
        if request.since:
            params["from"] = request.since.isoformat()
        if request.until:
            params["to"] = request.until.isoformat()

        response = await self.http.get(self.base_url, params=params, timeout=self._timeout)
        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceError(
                ErrorCode.PARSE_ERROR, "GNews returned malformed JSON", retryable=True
            ) from exc
        if payload.get("errors"):
            raise SourceError(
                ErrorCode.SOURCE_UNAVAILABLE,
                f"GNews error: {payload['errors']}",
                retryable=True,
            )

        articles: list[RawArticle] = []
        for item in payload.get("articles") or []:
            article = self._make_article(
                title=item.get("title") or "",
                url=item.get("url") or "",
                published_at=_parse_dt(item.get("publishedAt")),
                summary=item.get("description"),
                content=item.get("content"),
                language=request.language,
                author=None,
                image_url=item.get("image"),
                source_name=(item.get("source") or {}).get("name") or self.name,
            )
            if article and self._within_window(article, request):
                articles.append(article)
        return self._dedupe_urls(articles)


def _parse_dt(value: Any):
    from datetime import datetime

    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover - defensive
        return None
