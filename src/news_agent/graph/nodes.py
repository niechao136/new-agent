"""Graph nodes: fetch -> filter -> analyze -> summarize -> format (TODO item 9).

Each node is defensive on purpose: a node never raises for an expected failure
(source down, LLM refusing, empty result).  Problems are recorded on the
:class:`~news_agent.runtime.RunContext` and the run keeps going with whatever
data is available, which is what the degradation contract promises to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..analyzer import Analyzer, HeuristicAnalyzer, light_analyze
from ..cache import SqliteCache
from ..config import Settings
from ..dedup import Deduplicator
from ..models import (
    ErrorCode,
    NewsResult,
    RawArticle,
    SkillMode,
    SkillRequest,
    utcnow,
)
from ..quality import filter_spam
from ..relevance import pick_relevant, rank_articles, relevance_score
from ..rerank import Reranker, blend_scores
from ..runtime import Metrics, RunContext, get_logger
from ..sources import SourceRegistry
from ..trends import build_trends
from .state import NewsState

log = get_logger("graph")


@dataclass
class GraphDeps:
    """Everything the nodes need, injected once."""

    settings: Settings
    registry: SourceRegistry
    analyzer: Analyzer
    heuristic: HeuristicAnalyzer
    cache: SqliteCache | None
    metrics: Metrics
    deduplicator: Deduplicator
    #: Optional LLM re-ranker; ``None`` keeps the pure lexical ordering.
    reranker: Reranker | None = None


def get_ctx(config: RunnableConfig | None) -> RunContext:
    """Retrieve (or lazily create) the run context from the LangGraph config."""
    configurable = (config or {}).get("configurable") or {}
    ctx = configurable.get("ctx")
    if ctx is None:
        ctx = RunContext()
        configurable["ctx"] = ctx
    return ctx


def require_request(state: NewsState) -> SkillRequest:
    """Read the (mandatory) ``request`` entry of the graph state.

    ``NewsState`` is ``total=False`` because every node returns a partial update,
    so the request is looked up through ``.get`` and validated explicitly.
    """
    request = state.get("request")
    if request is None:  # pragma: no cover - the graph is always seeded with it
        raise RuntimeError("NewsState is missing the required 'request' entry")
    return request


class NewsGraphNodes:
    """Callables registered as LangGraph nodes."""

    def __init__(self, deps: GraphDeps) -> None:
        self.deps = deps
        self.settings = deps.settings

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------
    async def fetch_node(self, state: NewsState, config: RunnableConfig) -> dict[str, Any]:
        ctx = get_ctx(config)
        request = require_request(state)
        articles: list[RawArticle] = []
        cached = False

        with ctx.timeit("fetch"):
            cache_key = request.cache_key()
            if self.deps.cache is not None:
                cached_articles = await self.deps.cache.get_articles(cache_key)
                if cached_articles:
                    articles = cached_articles
                    cached = True
                    ctx.count("fetch_cache_hit")
                    ctx.emit(
                        "fetch",
                        f"命中抓取缓存，复用 {len(articles)} 篇（未访问新闻源）",
                        cached=True,
                    )

            if not cached:
                sources = self.deps.registry.select(request.sources)
                if not sources:
                    ctx.add_error(
                        ErrorCode.SOURCE_UNAVAILABLE,
                        "没有可用的新闻源，请检查 NEWS_AGENT_* 配置",
                        stage="fetch",
                    )
                per_source = self._per_source_limit(request, len(sources))
                ctx.emit(
                    "fetch",
                    f"并发抓取 {len(sources)} 个新闻源（每源上限 {per_source} 篇）",
                )

                def _progress(name: str, count: int) -> None:
                    ctx.emit(
                        "fetch",
                        f"新闻源 {name} 返回 {count} 篇",
                        event="progress",
                        source=name,
                        count=count,
                    )

                articles, errors = await self.deps.registry.fetch_all(
                    request,
                    per_source_limit=per_source,
                    progress=_progress,
                )
                for error in errors:
                    ctx.add_error(
                        error.code,
                        error.message,
                        source=error.source,
                        stage=error.stage or "fetch",
                        retryable=error.retryable,
                    )
                ctx.count("articles_fetched", len(articles))

                if self.deps.cache is not None:
                    await self.deps.cache.set_articles(cache_key, articles)
                    new_ids = await self.deps.cache.upsert_history(articles)
                    if new_ids:
                        ctx.count("articles_new_since_last_run", len(new_ids))
                        ctx.emit(
                            "fetch",
                            f"其中 {len(new_ids)} 篇为增量新文章",
                            event="progress",
                            new_articles=len(new_ids),
                        )

        if not articles:
            ctx.add_error(
                ErrorCode.NO_RESULTS,
                f"所有新闻源均未返回与「{request.query}」相关的结果",
                stage="fetch",
                retryable=True,
            )
            ctx.add_warning("本次运行没有抓到任何新闻，返回空结果集")
        else:
            ctx.emit("fetch", f"抓取完成，共获得 {len(articles)} 篇原始文章")

        return {"raw_articles": articles, "sources_attempted": self._sources_attempted(request)}

    def _per_source_limit(self, request: SkillRequest, source_count: int) -> int:
        if source_count <= 0:
            return request.limit
        # over-fetch a little: relevance filtering and de-duplication will trim
        per_source = request.limit * self.settings.fetch_per_source_multiplier / source_count
        return int(max(5, per_source + 3))

    def _sources_attempted(self, request: SkillRequest) -> int:
        try:
            return len(self.deps.registry.select(request.sources))
        except Exception:  # pragma: no cover - defensive
            return len(self.deps.registry.names())

    # ------------------------------------------------------------------
    # filter
    # ------------------------------------------------------------------
    async def filter_node(self, state: NewsState, config: RunnableConfig) -> dict[str, Any]:
        ctx = get_ctx(config)
        request = require_request(state)
        raw = list(state.get("raw_articles") or [])

        with ctx.timeit("filter"):
            # 0) 内容质量：先剔除博彩/SEO 站群/促销页——它们把查询词原样塞进
            #    标题，词面分很高但完全不是新闻。
            spam_dropped: list[RawArticle] = []
            if self.settings.spam_filter_enabled:
                raw, spam_dropped = filter_spam(
                    raw, threshold=self.settings.spam_threshold
                )
                if spam_dropped:
                    ctx.count("articles_spam_removed", len(spam_dropped))
                    ctx.add_warning(
                        f"已过滤 {len(spam_dropped)} 篇垃圾内容（博彩/SEO 站群/促销页）"
                    )
                    ctx.emit(
                        "filter",
                        f"质量过滤：移除 {len(spam_dropped)} 篇垃圾内容",
                        event="progress",
                        removed=len(spam_dropped),
                    )

            deduped = self.deps.deduplicator.dedupe(raw)
            duplicates = len(raw) - len(deduped)
            ctx.count("articles_deduped", duplicates)
            ctx.emit(
                "filter",
                f"去重完成：{len(raw)} → {len(deduped)} 篇（移除 {duplicates} 篇重复报道）",
                duplicates_removed=duplicates,
            )

            threshold = (
                request.threshold
                if request.threshold is not None
                else self.settings.relevance_threshold
            )
            # 词面打分（连续匹配约束 + 多关键词）＋泛化查询的栏目路由，
            # 来源权重只影响排序。
            scored = rank_articles(
                deduped,
                request.query,
                keywords=request.keywords,
                weights=self.deps.registry.weights(),
                source_topics=self.deps.registry.topics(),
            )
            scores = {article.id: score for article, score in scored}
            # 有精排时先多取一些候选，让 LLM 有机会把临界的相关文章提到前面；
            # 精排后再按 request.limit 截断。
            retrieve_limit = request.limit
            if self.deps.reranker is not None:
                retrieve_limit = max(request.limit, self.settings.llm.rerank_max_candidates)
            selected, _, topped_up = pick_relevant(
                scored,
                threshold=threshold,
                limit=retrieve_limit,
                # 配额按"最终返回条数"计算：多取的候选只是为了给精排留余量，
                # 不应放宽单一来源的占比上限。
                per_source_cap=self._per_source_cap(request.limit),
            )
            if topped_up:
                ctx.add_warning(
                    f"{topped_up} 篇结果相关度低于阈值 {threshold}，为满足数量要求被保留"
                )
            if deduped and not selected:
                ctx.add_warning("候选文章均未通过相关度过滤")
            selected, scores = await self._maybe_rerank(request, selected, scores, ctx)
            if len(selected) > request.limit:
                selected = selected[: request.limit]
            dropped = len(scored) - len(selected)
            ctx.count("articles_selected", len(selected))
            ctx.emit(
                "filter",
                f"相关性过滤完成：保留 {len(selected)} 篇，丢弃 {dropped} 篇",
                dropped=dropped,
                threshold=threshold,
            )

        return {
            "deduped_articles": deduped,
            "filtered_articles": selected,
            "duplicates_removed": duplicates,
            "spam_removed": len(spam_dropped),
            "relevance_scores": scores,
        }

    def _per_source_cap(self, limit: int) -> int | None:
        """Max articles a single source may contribute (None = no cap)."""
        if not self.settings.source_diversity:
            return None
        return max(3, -(-max(1, limit) // 3))  # ceil(limit / 3)

    async def _maybe_rerank(
        self,
        request: SkillRequest,
        selected: list[RawArticle],
        scores: dict[str, float],
        ctx: RunContext,
    ) -> tuple[list[RawArticle], dict[str, float]]:
        """LLM 精排：只在有候选需要取舍时调用一次，失败则保持词面顺序。"""
        reranker = self.deps.reranker
        if reranker is None or len(selected) < 2:
            return selected, scores

        budget = max(2, self.settings.llm.rerank_max_candidates)
        candidates = selected[:budget]
        with ctx.timeit("rerank"):
            try:
                llm_scores = await reranker.rerank(
                    request.query, request.keywords, candidates, ctx=ctx
                )
            except Exception as exc:  # noqa: BLE001 - degrade to lexical ranking
                ctx.add_warning(f"LLM 精排失败，保持词面相关性排序（{exc}）")
                return selected, scores

        if not llm_scores:
            return selected, scores

        merged = dict(scores)
        for article in candidates:
            if article.id in llm_scores:
                merged[article.id] = round(
                    blend_scores(scores.get(article.id, 0.0), llm_scores[article.id]), 4
                )

        ordered = sorted(
            selected,
            key=lambda article: (
                merged.get(article.id, 0.0),
                scores.get(article.id, 0.0),
                article.published_at.timestamp() if article.published_at else 0.0,
            ),
            reverse=True,
        )
        ctx.count("articles_reranked", len(llm_scores))
        ctx.emit(
            "filter",
            f"LLM 精排完成：对 {len(llm_scores)} 篇候选重新排序",
            event="progress",
            reranked=len(llm_scores),
        )
        return ordered, merged

    def route_after_filter(self, state: NewsState) -> str:
        """``fetch_news`` stops here; the other skills continue to analysis."""
        request = require_request(state)
        if request.skill is SkillMode.FETCH:
            return "format"
        if not state.get("filtered_articles"):
            return "format"
        return "analyze"

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------
    async def analyze_node(self, state: NewsState, config: RunnableConfig) -> dict[str, Any]:
        ctx = get_ctx(config)
        request = require_request(state)
        articles = list(state.get("filtered_articles") or [])

        if not articles:
            return {"analyzed_articles": []}

        with ctx.timeit("analyze"):
            if request.skill is SkillMode.FETCH or not request.include_analyzed:
                analyzed = light_analyze(articles)
                ctx.emit("analyze", f"fetch_news：跳过 LLM，仅返回 {len(analyzed)} 篇结构化元数据")
            else:
                budget = max(1, self.settings.max_articles_for_llm)
                head, tail = articles[:budget], articles[budget:]
                ctx.emit("analyze", f"开始结构化分析 {len(head)} 篇文章（LLM≈{self.deps.analyzer.name}）")
                analyzed = await self.deps.analyzer.analyze(request, head, ctx)
                if tail:
                    ctx.add_warning(
                        f"超过 LLM 预算（{budget} 篇）的 {len(tail)} 篇文章使用启发式分析"
                    )
                    analyzed = analyzed + await self.deps.heuristic.analyze(request, tail, ctx)

            # attach the relevance score computed by the filter stage
            scores = dict(state.get("relevance_scores") or {})
            if not scores:
                scores = {article.id: relevance_score(article, request.query) for article in articles}
            for item in analyzed:
                item.relevance = scores.get(item.id, 0.0)
            analyzed.sort(
                key=lambda item: (
                    item.relevance,
                    item.published_at.timestamp() if item.published_at else 0.0,
                ),
                reverse=True,
            )
            ctx.count("articles_analyzed", len(analyzed))
            positive = sum(1 for item in analyzed if item.sentiment == "positive")
            negative = sum(1 for item in analyzed if item.sentiment == "negative")
            ctx.emit(
                "analyze",
                f"分析完成：正面 {positive} 篇 / 负面 {negative} 篇 / 其他 {len(analyzed) - positive - negative} 篇",
                positive=positive,
                negative=negative,
            )

        return {"analyzed_articles": analyzed}

    # ------------------------------------------------------------------
    # summarize
    # ------------------------------------------------------------------
    async def summarize_node(self, state: NewsState, config: RunnableConfig) -> dict[str, Any]:
        ctx = get_ctx(config)
        request = require_request(state)
        analyzed = list(state.get("analyzed_articles") or [])

        if request.skill is SkillMode.FETCH or not analyzed:
            return {"summary": ""}

        with ctx.timeit("summarize"):
            ctx.emit("summarize", f"生成话题级摘要（map-reduce，{len(analyzed)} 篇）")
            summary = await self.deps.analyzer.summarize(request, analyzed, ctx)
            ctx.emit("summarize", "摘要生成完成", length=len(summary))
        return {"summary": summary}

    # ------------------------------------------------------------------
    # format
    # ------------------------------------------------------------------
    async def format_node(self, state: NewsState, config: RunnableConfig) -> dict[str, Any]:
        ctx = get_ctx(config)
        request = require_request(state)
        started = ctx.started_at

        raw = list(state.get("raw_articles") or [])
        deduped = list(state.get("deduped_articles") or [])
        filtered = list(state.get("filtered_articles") or [])
        analyzed = list(state.get("analyzed_articles") or [])
        summary = state.get("summary") or ""

        with ctx.timeit("format"):
            if request.skill is SkillMode.FETCH and not analyzed and filtered:
                # fetch_news skips the analyze node; still return metadata rows
                analyzed = light_analyze(filtered)
                scores = dict(state.get("relevance_scores") or {})
                if not scores:
                    scores = {
                        article.id: relevance_score(article, request.query)
                        for article in filtered
                    }
                for item in analyzed:
                    item.relevance = scores.get(item.id, 0.0)

            trends = []
            if request.skill is SkillMode.TREND and analyzed:
                trends = build_trends(analyzed, top_n=self.settings.trends_top_n)

            if not summary and request.skill is not SkillMode.FETCH and analyzed:
                # last-resort degradation: a deterministic summary instead of nothing
                summary = await self.deps.heuristic.summarize(request, analyzed, ctx)
                ctx.add_warning("未获得 LLM 摘要，已退化为启发式摘要")

            # 降级判定：只有"结果本身受影响"才算降级。少数新闻源失败是常态
            # （RSS 下线、单站限流），不应把每次都标成降级，否则这个信号就失去意义。
            hard_failure = any(
                error.code
                in {
                    ErrorCode.NO_RESULTS,
                    ErrorCode.INTERNAL_ERROR,
                    ErrorCode.LLM_FAILED,
                    ErrorCode.LLM_TIMEOUT,
                    ErrorCode.CACHE_ERROR,
                    ErrorCode.CONTEXT_TOO_LONG,
                    ErrorCode.TASK_TIMEOUT,
                }
                for error in ctx.errors
            )
            source_failures = {
                error.source or error.code.value
                for error in ctx.errors
                if error.code
                in {
                    ErrorCode.SOURCE_UNAVAILABLE,
                    ErrorCode.FETCH_TIMEOUT,
                    ErrorCode.RATE_LIMITED,
                    ErrorCode.PARSE_ERROR,
                }
            }
            attempted = int(state.get("sources_attempted") or 0)
            partial_sources = bool(attempted) and len(source_failures) / attempted >= 0.5
            degraded = hard_failure or partial_sources or not analyzed
            if source_failures and not degraded:
                ctx.add_warning(
                    f"{len(source_failures)}/{attempted or '?'} 个新闻源本次未返回数据，"
                    "结果可能略有缺失"
                )

            counts = {
                "fetched": len(raw),
                "spam_removed": int(state.get("spam_removed") or 0),
                "duplicates_removed": int(state.get("duplicates_removed") or 0),
                "after_dedup": len(deduped),
                "sources_attempted": int(state.get("sources_attempted") or 0),
                "selected": len(filtered),
                "analyzed": len(analyzed),
                "sources_used": len({item.source for item in analyzed}),
                "errors": len(ctx.errors),
                "warnings": len(ctx.warnings),
            }
            metrics = {
                "analyzer": self.deps.analyzer.name,
                "llm_enabled": bool(getattr(self.deps.analyzer, "uses_llm", False)),
                **{key: value for key, value in ctx.counters.items()},
            }

            result = NewsResult(
                query=request.query,
                mode=request.skill,
                language=request.language,
                generated_at=utcnow(),
                duration_ms=int((utcnow() - started).total_seconds() * 1000),
                degraded=degraded,
                counts=counts,
                summary=summary,
                articles=analyzed,
                trends=trends,
                warnings=list(ctx.warnings),
                errors=list(ctx.errors),
                timings_ms={},
                metrics=metrics,
            )

        # the "format" stage timing is only known once the `with` block exits
        result.timings_ms = dict(ctx.timings)
        result.duration_ms = int((utcnow() - started).total_seconds() * 1000)
        ctx.set_partial(result)
        ctx.emit(
            "format",
            f"任务完成：{result.counts.get('analyzed', 0)} 篇 / 耗时 {result.duration_ms} ms"
            + ("（降级）" if result.degraded else ""),
            event="completed",
        )

        return {
            "result": result,
            "warnings": list(ctx.warnings),
            "errors": [error.code.value for error in ctx.errors],
            "timings": dict(ctx.timings),
            "counters": dict(ctx.counters),
        }
