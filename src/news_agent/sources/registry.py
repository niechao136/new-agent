"""Source registry: builds concrete sources from configuration."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ..config import Settings
from ..models import ErrorCode, ErrorInfo, RawArticle, SkillRequest
from ..runtime import get_logger
from .api import GNewsSource, NewsAPISource
from .base import NewsSource
from .http import HttpClient, gather_limited
from .mock import MockSource
from .rss import RSSSource

SOURCE_TYPES: dict[str, type[NewsSource]] = {
    RSSSource.type: RSSSource,
    NewsAPISource.type: NewsAPISource,
    GNewsSource.type: GNewsSource,
    MockSource.type: MockSource,
}

log = get_logger("sources")

#: Called as ``progress(source_name, article_count)`` after each source finishes.
ProgressSink = Callable[[str, int], None]


class SourceRegistry:
    """Owns the configured :class:`NewsSource` instances and the HTTP client."""

    def __init__(
        self,
        sources: list[NewsSource],
        *,
        http: HttpClient,
        settings: Settings,
    ) -> None:
        self.settings = settings
        self.http = http
        self._sources = {source.name: source for source in sources}
        self.last_errors: list[ErrorInfo] = []

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(
        cls, settings: Settings, http: HttpClient | None = None
    ) -> "SourceRegistry":
        client = http or HttpClient(settings)
        sources: list[NewsSource] = []
        for config in settings.sources:
            if not config.enabled:
                continue
            source_cls = SOURCE_TYPES.get(config.type)
            if source_cls is None:
                log.warning("unknown source type %r for %r", config.type, config.name)
                continue
            sources.append(source_cls(config, settings, client))
        return cls(sources, http=client, settings=settings)

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self._sources)

    def select(self, names: Iterable[str] | None = None) -> list[NewsSource]:
        if not names:
            return list(self._sources.values())
        wanted = {str(name).strip() for name in names if str(name).strip()}
        selected = [self._sources[name] for name in self._sources if name in wanted]
        if not selected:
            log.warning("requested sources %s are unknown, using all", sorted(wanted))
            return list(self._sources.values())
        return selected

    def add(self, source: NewsSource) -> None:
        self._sources[source.name] = source

    # ------------------------------------------------------------------
    async def fetch_all(
        self,
        request: SkillRequest,
        *,
        per_source_limit: int,
        concurrency: int | None = None,
        progress: ProgressSink | None = None,
    ) -> tuple[list[RawArticle], list[ErrorInfo]]:
        """Fetch from every selected source concurrently (bounded)."""
        sources = self.select(request.sources)
        if not sources:
            return [], []

        async def _one(source: NewsSource) -> list[RawArticle]:
            try:
                articles = await source.fetch(request, per_source_limit)
            except Exception as exc:  # pragma: no cover - source.fetch never raises
                log.exception("source %s crashed", source.name)
                source.last_errors.append(
                    ErrorInfo(
                        code=ErrorCode.INTERNAL_ERROR,
                        message=f"source '{source.name}' crashed: {exc!r}",
                        source=source.name,
                        stage="fetch",
                    )
                )
                return []
            if progress is not None:
                progress(source.name, len(articles))
            return articles

        batches = await gather_limited(
            [_one(source) for source in sources],
            concurrency=concurrency or self.settings.source_concurrency,
        )

        articles: list[RawArticle] = []
        errors: list[ErrorInfo] = []
        for source, batch in zip(sources, batches):
            articles.extend(batch)
            errors.extend(source.last_errors)
        self.last_errors = errors
        return articles, errors

    async def aclose(self) -> None:
        await self.http.aclose()
