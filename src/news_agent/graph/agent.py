"""``NewsAgent``: the composable facade used by the CLI and the A2A executor."""

from __future__ import annotations

import asyncio
from typing import Any

from ..analyzer import Analyzer, HeuristicAnalyzer, build_analyzer
from ..cache import SqliteCache
from ..config import Settings, load_settings
from ..dedup import Deduplicator
from ..models import ErrorCode, NewsResult, SkillMode, SkillRequest, utcnow
from ..rerank import Reranker, build_reranker
from ..runtime import Metrics, RunContext, configure_logging, get_logger
from ..sources import SourceRegistry
from .builder import build_news_graph
from .nodes import GraphDeps, NewsGraphNodes

log = get_logger("agent")


class NewsAgent:
    """Ties the sources, the cache, the analyser and the LangGraph together."""

    def __init__(
        self,
        settings: Settings,
        *,
        registry: SourceRegistry,
        cache: SqliteCache | None,
        metrics: Metrics | None = None,
        analyzer: Analyzer | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.cache = cache
        self.metrics = metrics or Metrics()
        self.heuristic = HeuristicAnalyzer()
        self.analyzer: Analyzer = analyzer or build_analyzer(settings)
        self.reranker: Reranker | None = (
            reranker if reranker is not None else build_reranker(settings)
        )
        self.deduplicator = Deduplicator(
            title_ratio=settings.dedup_title_ratio,
            simhash_distance=settings.dedup_simhash_distance,
        )
        self.nodes = NewsGraphNodes(
            GraphDeps(
                settings=settings,
                registry=registry,
                analyzer=self.analyzer,
                heuristic=self.heuristic,
                cache=cache,
                metrics=self.metrics,
                deduplicator=self.deduplicator,
                reranker=self.reranker,
            )
        )
        self.graph = build_news_graph(self.nodes)

    # ------------------------------------------------------------------
    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        *,
        registry: SourceRegistry | None = None,
        cache: SqliteCache | None = None,
        analyzer: Analyzer | None = None,
        reranker: Reranker | None = None,
        metrics: Metrics | None = None,
    ) -> "NewsAgent":
        settings = settings or load_settings()
        configure_logging(settings.log_level)

        if cache is None and settings.cache_enabled:
            cache = SqliteCache(
                settings.cache_file(), ttl_s=settings.cache_ttl_s, enabled=True
            )
            await cache.init()
        if registry is None:
            registry = SourceRegistry.from_settings(settings)
        return cls(
            settings,
            registry=registry,
            cache=cache,
            metrics=metrics,
            analyzer=analyzer,
            reranker=reranker,
        )

    # ------------------------------------------------------------------
    async def run(
        self, request: SkillRequest, *, ctx: RunContext | None = None
    ) -> NewsResult:
        """Execute the sub-graph and return the final result.

        Never raises for expected failures: a degraded ``NewsResult`` with
        populated ``errors``/``warnings`` is returned instead.
        """
        context = ctx or RunContext(metrics=self.metrics)
        state: dict[str, Any] = {"request": request}
        config = {"configurable": {"ctx": context}, "recursion_limit": 25}
        try:
            final_state = await self.graph.ainvoke(state, config)
        except Exception as exc:  # noqa: BLE001 - convert to the degradation contract
            log.exception("graph execution failed")
            context.add_error(
                ErrorCode.INTERNAL_ERROR,
                f"graph execution failed: {exc!r}",
                stage="graph",
            )
            return self._fallback_result(request, context, exc)
        result = final_state.get("result") if isinstance(final_state, dict) else None
        if result is None:  # pragma: no cover - defensive
            return self._fallback_result(request, context, RuntimeError("no result produced"))
        return result

    # ------------------------------------------------------------------
    def _fallback_result(
        self, request: SkillRequest, ctx: RunContext, exc: BaseException
    ) -> NewsResult:
        if ctx.partial is not None:
            ctx.partial.warnings.append(f"任务异常终止：{exc!r}")
            return ctx.partial
        return NewsResult(
            query=request.query,
            mode=request.skill,
            language=request.language,
            generated_at=utcnow(),
            duration_ms=ctx.elapsed_ms(),
            degraded=True,
            counts={"fetched": 0, "analyzed": 0},
            summary="",
            warnings=list(ctx.warnings),
            errors=list(ctx.errors),
            timings_ms=dict(ctx.timings),
            metrics={"analyzer": getattr(self.analyzer, "name", "unknown")},
        )

    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        await self.registry.aclose()
        if self.cache is not None:
            await self.cache.aclose()

    async def __aenter__(self) -> "NewsAgent":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


async def run_once(
    query: str,
    *,
    skill: str | SkillMode = SkillMode.SUMMARIZE,
    limit: int = 10,
    language: str = "zh",
    settings: Settings | None = None,
    sources: list[str] | None = None,
    ctx: RunContext | None = None,
) -> NewsResult:
    """Convenience helper: build an agent, run one request, tear it down."""
    agent = await NewsAgent.create(settings)
    try:
        request = SkillRequest(
            skill=SkillMode.coerce(skill),
            query=query,
            limit=limit,
            language=language,
            sources=sources,
        )
        return await agent.run(request, ctx=ctx)
    finally:
        await agent.aclose()


def run_sync(query: str, **kwargs: Any) -> NewsResult:  # pragma: no cover - sync helper
    return asyncio.run(run_once(query, **kwargs))
