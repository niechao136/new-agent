"""RSS/Atom based source -- the zero-configuration default.

Feed URLs are templates; ``{query}``, ``{language}``, ``{hl}``, ``{gl}`` and
``{ceid}`` are rendered per request which lets a single feed entry behave like
a keyword search (Google News / Bing News RSS).
"""

from __future__ import annotations

import calendar
import datetime as dt
from typing import Any
from urllib.parse import quote_plus

import feedparser

from ..models import RawArticle, SkillRequest, ensure_aware, utcnow
from ..text_utils import clean_text, strip_html
from .base import NewsSource
from .http import SourceError
from ..models import ErrorCode

#: Language -> (hl, gl, ceid) mapping used by Google News RSS.
LOCALE_MAP: dict[str, tuple[str, str, str]] = {
    "zh": ("zh-CN", "CN", "CN:zh-Hans"),
    "zh-cn": ("zh-CN", "CN", "CN:zh-Hans"),
    "zh-tw": ("zh-TW", "TW", "TW:zh-Hant"),
    "en": ("en-US", "US", "US:en"),
    "en-us": ("en-US", "US", "US:en"),
    "ja": ("ja", "JP", "JP:ja"),
    "ko": ("ko", "KR", "KR:ko"),
    "fr": ("fr", "FR", "FR:fr"),
    "de": ("de", "DE", "DE:de"),
}


class RSSSource(NewsSource):
    type = "rss"

    def _render_feeds(self, request: SkillRequest) -> list[str]:
        locale = LOCALE_MAP.get((request.language or "en").lower(), LOCALE_MAP["en"])
        hl, gl, ceid = locale
        search_term = request.query
        # Google News RSS 支持 when:Xd 新鲜度操作符；窗口来自请求的时间
        # 解析（本周/最近N天）或默认窗口。超过 30 天或静态 feed 不附加。
        if request.since is not None:
            days = (utcnow() - request.since).days + 1
            if 0 < days <= 30:
                search_term = f"{request.query} when:{days}d"
        rendered: list[str] = []
        for template in self.config.feeds:
            if not template:
                continue
            term = search_term if "news.google.com" in template else request.query
            try:
                rendered.append(
                    template.format(
                        query=quote_plus(term),
                        language=request.language or "en",
                        hl=hl,
                        gl=gl,
                        ceid=ceid,
                    )
                )
            except (KeyError, IndexError):  # pragma: no cover - bad template
                rendered.append(template)
        return rendered

    async def _fetch(self, request: SkillRequest, limit: int) -> list[RawArticle]:
        feeds = self._render_feeds(request)
        if not feeds:
            return []
        per_feed = max(5, int(limit))
        articles: list[RawArticle] = []
        failures: list[str] = []
        for feed_url in feeds:
            try:
                articles.extend(await self._fetch_feed(feed_url, request, per_feed))
            except SourceError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                failures.append(f"{feed_url}: {exc!r}")
        if not articles and failures:
            raise SourceError(
                ErrorCode.PARSE_ERROR,
                "all feeds failed: " + "; ".join(failures[:3]),
                retryable=False,
            )
        return self._dedupe_urls(articles)

    async def _fetch_feed(
        self, feed_url: str, request: SkillRequest, limit: int
    ) -> list[RawArticle]:
        response = await self.http.get(
            feed_url,
            headers={"Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"},
            timeout=self._timeout,
        )
        parsed = feedparser.parse(response.content)
        has_feed_metadata = bool(parsed.feed) and bool(parsed.feed.get("title"))
        if not parsed.entries and not has_feed_metadata:
            raise SourceError(
                ErrorCode.PARSE_ERROR,
                f"feed {feed_url} could not be parsed",
                retryable=False,
            )
        status = getattr(parsed, "status", None)
        if status and status >= 400:
            raise SourceError(
                ErrorCode.SOURCE_UNAVAILABLE,
                f"feed {feed_url} returned HTTP {status}",
                retryable=status >= 500,
            )

        feed_title = clean_text(parsed.feed.get("title") if parsed.feed else "")
        articles: list[RawArticle] = []
        for entry in parsed.entries[: max(limit * 3, limit)]:
            article = self._entry_to_article(entry, feed_title, request)
            if article is None:
                continue
            if not self._within_window(article, request):
                continue
            articles.append(article)
            if len(articles) >= limit:
                break
        return articles

    def _entry_to_article(
        self, entry: Any, feed_title: str, request: SkillRequest
    ) -> RawArticle | None:
        link = clean_text(entry.get("link") or "")
        title = clean_text(entry.get("title") or "")
        source_name = self.name
        entry_source = entry.get("source")
        if isinstance(entry_source, dict):
            source_name = clean_text(entry_source.get("title") or "") or self.name
        elif entry_source:
            source_name = clean_text(str(entry_source)) or self.name

        # Google News titles look like "Headline - Publisher"; drop the suffix
        # when it duplicates the publisher we already know.
        if " - " in title and source_name and title.rstrip().endswith(source_name):
            title = title.rstrip()[: -len(source_name)].rstrip(" -–—").strip()

        summary = entry.get("summary") or entry.get("description") or ""
        content_html = ""
        if entry.get("content"):
            try:
                content_html = entry["content"][0].get("value", "")
            except (TypeError, IndexError, KeyError):  # pragma: no cover
                content_html = ""
        if not content_html:
            content_html = summary

        image_url = None
        for media in entry.get("media_content") or []:
            if isinstance(media, dict) and media.get("url"):
                image_url = media["url"]
                break
        if not image_url:
            for link_info in entry.get("links") or []:
                if isinstance(link_info, dict) and link_info.get("rel") == "enclosure":
                    if str(link_info.get("type", "")).startswith("image"):
                        image_url = link_info.get("href")
                        break

        return self._make_article(
            title=title,
            url=link,
            published_at=self._entry_date(entry),
            summary=strip_html(summary)[:1000] or None,
            content=strip_html(content_html)[:6000] or None,
            language=request.language,
            author=clean_text(entry.get("author") or "") or None,
            image_url=image_url,
            source_name=source_name,
            extra={"feed": feed_title} if feed_title else None,
        )

    @staticmethod
    def _entry_date(entry: Any) -> dt.datetime | None:
        for key in ("published_parsed", "updated_parsed", "created_parsed"):
            value = entry.get(key) if hasattr(entry, "get") else None
            if value:
                try:
                    return dt.datetime.fromtimestamp(
                        calendar.timegm(value), tz=dt.timezone.utc
                    )
                except (ValueError, OverflowError, TypeError):  # pragma: no cover
                    continue
        for key in ("published", "updated"):
            raw = entry.get(key) if hasattr(entry, "get") else None
            if raw:
                try:
                    from email.utils import parsedate_to_datetime

                    return ensure_aware(parsedate_to_datetime(raw))
                except (TypeError, ValueError):  # pragma: no cover
                    continue
        return None
