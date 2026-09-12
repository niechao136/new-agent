"""``AgentExecutor`` implementation backed by the LangGraph sub-graph.

This is the glue between the A2A server (``a2a-sdk``) and the news agent:

``RequestContext`` → :class:`~news_agent.models.SkillRequest` → LangGraph run
→ ``NewsResult`` → ``TaskUpdater`` (status/artifact events) → task store.

Everything protocol related is delegated to the SDK:

* task creation / persistence     → ``DefaultRequestHandler`` + ``InMemoryTaskStore``
* state machine & terminal states → ``TaskUpdater`` (``submit``/``start_work``/…)
* streaming / SSE                 → SDK event queue
* request validation errors       → ``SkillRequestError`` → ``rejected`` task

Progress is streamed through *standard* ``TaskStatusUpdateEvent`` messages while
the task is ``working``; each carries ``metadata = {stage, kind, data}`` so
callers can render fine-grained progress without vendor-specific endpoints.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from a2a.helpers.proto_helpers import (
    get_data_parts,
    get_message_text,
    new_data_part,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import a2a_pb2
from a2a.utils.proto_utils import make_dict_serializable

from ..config import Settings
from ..graph.agent import NewsAgent
from ..intent import parse_query_intent
from ..models import ErrorCode, NewsResult, SkillMode, SkillRequest, utcnow
from ..runtime import Metrics, ProgressEvent, RunContext, get_logger
from ..text_utils import truncate
from ..time_window import parse_time_window

log = get_logger("a2a.executor")

_SKILL_PREFIX = re.compile(
    r"^\s*(fetch_news|summarize_news|analyze_trend|fetch|summarize|summarise|"
    r"analyze|analyse|trend|summary)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)

VALID_SKILL_STRINGS = {
    *(mode.value for mode in SkillMode),
    "fetch",
    "summarize",
    "summarise",
    "analyze",
    "analyse",
    "trend",
    "summary",
}

#: States after which no further event may be published.
TERMINAL_STATES = frozenset(
    {
        a2a_pb2.TASK_STATE_COMPLETED,
        a2a_pb2.TASK_STATE_FAILED,
        a2a_pb2.TASK_STATE_CANCELED,
        a2a_pb2.TASK_STATE_REJECTED,
    }
)


class SkillRequestError(ValueError):
    """The incoming A2A message could not be turned into a valid skill call."""


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise a metadata payload into a ``Struct``-compatible plain dict."""
    normalized = make_dict_serializable(payload)
    assert isinstance(normalized, dict)  # noqa: S101 - dict in, dict out
    return normalized


def _extract_payload(
    data: dict[str, Any] | None,
    text: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge the various request carriers into one payload dict with a ``query``."""
    payload: dict[str, Any] = dict(data or {})
    metadata = dict(metadata or {})

    if not payload and text.strip().startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed

    if not payload.get("query"):
        for key in ("query", "topic", "keywords", "q", "input"):
            if metadata.get(key):
                payload["query"] = metadata[key]
                break

    if text and not payload.get("query"):
        stripped = text.strip()
        match = _SKILL_PREFIX.match(stripped)
        if match:
            payload.setdefault("skill", match.group(1))
            payload["query"] = match.group(2).strip()
        else:
            payload["query"] = stripped

    return payload


def _merge_keywords(
    parsed: list[str] | None,
    payload: dict[str, Any],
    query: str,
) -> list[str]:
    """Combine LLM keywords with caller supplied ones (``keywords``/``expanded_keywords``)."""
    raw = payload.get("expanded_keywords") or payload.get("keywords")
    extra: list[str] = []
    if isinstance(raw, (list, tuple)):
        extra = [str(item).strip() for item in raw]
    elif isinstance(raw, str) and raw.strip() and raw.strip() != query:
        # list 形式一定是扩展词；字符串只有在与 query 不同时才算
        extra = [part.strip() for part in re.split(r"[,，;；\s]+", raw)]
    merged: list[str] = []
    for item in [*(parsed or []), *extra]:
        if item and item != query and item not in merged:
            merged.append(item)
    return merged


async def aparse_skill_request(
    *,
    data: dict[str, Any] | None = None,
    text: str = "",
    metadata: dict[str, Any] | None = None,
    context_id: str | None = None,
    settings: Settings,
) -> SkillRequest:
    """Async counterpart of :func:`parse_skill_request`.

    与同步版本唯一的差别：查询的主题/时间要素优先交给 LLM 解析
    （:func:`news_agent.intent.parse_query_intent`，LLM 不可用时自动回退
    正则解析）。调用方显式传入 ``since`` 时跳过解析。
    """
    payload = _extract_payload(data, text, metadata)
    metadata = dict(metadata or {})

    time_window: tuple[datetime | None, datetime | None, str] | None = None
    keywords: list[str] | None = None
    if payload.get("since") is None:
        query = payload.get("query") or metadata.get("query") or ""
        if isinstance(query, (list, tuple)):
            query = " ".join(str(item) for item in query)
        query = str(query).strip()[:300]
        if query:
            intent = await parse_query_intent(query, settings.llm)
            time_window = (intent.since, intent.until, intent.search_query)
            keywords = intent.keywords

    return parse_skill_request(
        data=payload,
        metadata=metadata,
        context_id=context_id,
        settings=settings,
        time_window=time_window,
        keywords=keywords,
    )


def parse_skill_request(
    *,
    data: dict[str, Any] | None = None,
    text: str = "",
    metadata: dict[str, Any] | None = None,
    context_id: str | None = None,
    settings: Settings,
    time_window: tuple[datetime | None, datetime | None, str] | None = None,
    keywords: list[str] | None = None,
) -> SkillRequest:
    """Normalise an A2A message payload into a validated :class:`SkillRequest`.

    Accepts the parameters either as a JSON object (``DataPart.data``), as plain
    text (``"<skill>: <query>"`` or just ``"<query>"``) or in the request
    ``metadata`` — whichever the calling agent finds convenient.

    ``time_window`` allows the caller (usually :func:`aparse_skill_request`)
    to inject a pre-parsed ``(since, until, cleaned_query)`` result, e.g. one
    produced by the LLM intent parser.  ``keywords`` carries retrieval keywords
    (translations / aliases) that are only used for relevance matching.
    """
    payload = _extract_payload(data, text, metadata)
    metadata = dict(metadata or {})

    raw_skill = (
        payload.get("skill")
        or payload.get("skillId")
        or payload.get("skill_id")
        or metadata.get("skill")
        or metadata.get("skillId")
        or settings.default_mode
    )
    if isinstance(raw_skill, str) and raw_skill.strip().lower() not in VALID_SKILL_STRINGS:
        raise SkillRequestError(
            f"unknown skill: {raw_skill!r} (valid: {', '.join(sorted(VALID_SKILL_STRINGS))})"
        )

    query = payload.get("query") or metadata.get("query") or ""
    if isinstance(query, (list, tuple)):
        query = " ".join(str(item) for item in query)
    query = str(query).strip()
    if not query:
        raise SkillRequestError(
            "missing required parameter 'query' — send it as a DataPart "
            '{"query": ...} or as plain text'
        )
    if len(query) > 300:
        query = query[:300]

    limit = payload.get("limit", settings.default_limit)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise SkillRequestError("'limit' must be an integer") from None
    limit = max(1, min(limit, settings.max_limit))

    threshold = payload.get("threshold")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise SkillRequestError("'threshold' must be a number") from None
        threshold = max(0.0, min(1.0, threshold))

    sources = payload.get("sources")
    if isinstance(sources, str):
        sources = [item.strip() for item in sources.split(",") if item.strip()]
    elif isinstance(sources, (list, tuple)):
        sources = [str(item).strip() for item in sources if str(item).strip()]
    else:
        sources = None

    # 时间窗口：调用方显式传入 since 优先；否则使用预解析结果（LLM 意图解析）
    # 或从查询中正则解析「本周/昨天/最近N天」等表达；
    # 都没有时走默认窗口（default_window_days）。
    since = payload.get("since")
    until = payload.get("until")
    search_query = query
    if since is None:
        if time_window is not None:
            parsed_since, parsed_until, cleaned = time_window
        else:
            parsed_since, parsed_until, cleaned = parse_time_window(query)
        search_query = cleaned or query
        if parsed_since is not None:
            since = parsed_since
            until = parsed_until if parsed_until is not None else until

    try:
        return SkillRequest(
            skill=SkillMode.coerce(raw_skill),
            query=search_query,
            keywords=_merge_keywords(keywords, payload, query),
            since=since,
            until=until,
            limit=limit,
            language=str(payload.get("language") or settings.default_language),
            sources=sources,
            threshold=threshold,
            include_analyzed=bool(payload.get("include_analyzed", True)),
            context_id=context_id,
        ).with_default_window(settings.default_window_days)
    except Exception as exc:  # noqa: BLE001 - pydantic validation
        raise SkillRequestError(f"invalid request: {exc}") from exc


class NewsAgentExecutor(AgentExecutor):
    """Runs news skills as A2A tasks."""

    def __init__(
        self,
        agent_provider: Callable[[], Awaitable[NewsAgent]],
        settings: Settings,
        *,
        metrics: Metrics | None = None,
    ) -> None:
        #: Async factory so the (heavy) ``NewsAgent`` can be created lazily on the
        #: first request, regardless of whether the ASGI server runs the lifespan.
        self._agent_provider = agent_provider
        self.settings = settings
        self.metrics = metrics or Metrics()
        self._running: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # AgentExecutor interface
    # ------------------------------------------------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or uuid.uuid4().hex
        context_id = context.context_id or uuid.uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)
        self.metrics.incr("a2a_tasks_submitted")

        # A2A 1.0 requires the agent to publish the Task itself before any
        # TaskStatusUpdateEvent / TaskArtifactUpdateEvent is accepted.
        await event_queue.enqueue_event(
            self._initial_task(task_id, context_id, context.message)
        )

        try:
            request = await self._build_request(context)
        except SkillRequestError as exc:
            log.info("task %s rejected: %s", task_id, exc)
            self.metrics.incr("a2a_tasks_rejected")
            await updater.reject(
                updater.new_agent_message(
                    [
                        new_text_part(f"请求无效：{exc}"),
                        new_data_part(
                            {
                                "error": "invalid_request",
                                "message": str(exc),
                                "validSkills": sorted(mode.value for mode in SkillMode),
                            }
                        ),
                    ]
                )
            )
            return

        try:
            agent = await self._agent_provider()
        except Exception as exc:  # noqa: BLE001 - report instead of crashing the task
            log.exception("could not initialise the news agent")
            self.metrics.incr("a2a_tasks_failed")
            await updater.failed(
                updater.new_agent_message(
                    [new_text_part(f"agent 初始化失败：{exc}")]
                )
            )
            return

        ctx = RunContext(task_id=task_id, metrics=self.metrics)

        async def _publish(event: ProgressEvent) -> None:
            await self._publish_progress(updater, event)

        ctx.subscribe(_publish)

        await updater.start_work(
            updater.new_agent_message(
                [new_text_part(f"开始抓取并分析「{request.query}」相关新闻")]
            )
        )

        current = asyncio.current_task()
        if current is not None:
            self._running[task_id] = current
        try:
            result = await self._run(agent, request, ctx)
        except asyncio.CancelledError:
            log.info("task %s cancelled", task_id)
            self.metrics.incr("a2a_tasks_canceled")
            raise
        finally:
            self._running.pop(task_id, None)

        await ctx.drain()
        await self._publish_result(updater, request, result)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or task_id
        updater = TaskUpdater(event_queue, task_id, context_id)

        handle = self._running.pop(task_id, None)
        if handle is not None and not handle.done():
            handle.cancel()
            log.info("cancelling in-flight task %s", task_id)

        try:
            await updater.cancel(
                updater.new_agent_message([new_text_part("任务已按调用方请求取消")])
            )
            self.metrics.incr("a2a_tasks_canceled")
        except RuntimeError:
            # already terminal - nothing to do
            log.info("task %s already finished, cancel ignored", task_id)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _initial_task(
        task_id: str, context_id: str, message: a2a_pb2.Message | None
    ) -> a2a_pb2.Task:
        """The ``Task`` the agent must publish before any other event."""
        task = a2a_pb2.Task(
            id=task_id,
            context_id=context_id,
            status=a2a_pb2.TaskStatus(state=a2a_pb2.TASK_STATE_SUBMITTED),
        )
        if message is not None:
            task.history.append(message)
        return task

    async def _build_request(self, context: RequestContext) -> SkillRequest:
        message = context.message
        text = get_message_text(message) if message is not None else ""
        data: dict[str, Any] = {}
        if message is not None:
            for chunk in get_data_parts(message.parts):
                if isinstance(chunk, dict):
                    data.update(chunk)
        return await aparse_skill_request(
            data=data,
            text=text,
            metadata=dict(context.metadata or {}),
            context_id=context.context_id,
            settings=self.settings,
        )

    async def _publish_progress(self, updater: TaskUpdater, event: ProgressEvent) -> None:
        try:
            await updater.update_status(
                a2a_pb2.TASK_STATE_WORKING,
                message=updater.new_agent_message(
                    [new_text_part(f"[{event.stage}] {event.message}")]
                ),
                metadata=_metadata(
                    {
                        "stage": event.stage,
                        "kind": event.event,
                        "data": event.data,
                    }
                ),
            )
        except RuntimeError:  # pragma: no cover - race with a terminal state
            log.debug("progress update dropped, task already terminal")

    async def _run(
        self, agent: NewsAgent, request: SkillRequest, ctx: RunContext
    ) -> NewsResult:
        timeout = max(5.0, float(self.settings.task_timeout_s))
        try:
            return await asyncio.wait_for(agent.run(request, ctx=ctx), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("task %s exceeded the %.0fs deadline", ctx.task_id, timeout)
            self.metrics.incr("a2a_task_timeouts")
            return self._timeout_result(request, ctx, timeout)
        except Exception as exc:  # noqa: BLE001 - never leak a raw exception
            log.exception("task %s crashed", ctx.task_id)
            ctx.add_error(
                ErrorCode.INTERNAL_ERROR, f"executor crashed: {exc!r}", stage="executor"
            )
            return self._fallback_result(request, ctx)

    def _timeout_result(
        self, request: SkillRequest, ctx: RunContext, timeout: float
    ) -> NewsResult:
        ctx.add_error(
            ErrorCode.TASK_TIMEOUT,
            f"任务超过最大执行时间 {timeout:.0f}s，返回已完成的阶段结果",
            stage="executor",
            retryable=True,
        )
        ctx.add_warning(f"任务超时（>{timeout:.0f}s），以下为部分结果")
        if ctx.partial is not None:
            partial = ctx.partial
            partial.degraded = True
            partial.warnings = list(ctx.warnings)
            partial.errors = list(ctx.errors)
            partial.duration_ms = ctx.elapsed_ms()
            return partial
        return self._fallback_result(request, ctx)

    @staticmethod
    def _fallback_result(request: SkillRequest, ctx: RunContext) -> NewsResult:
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
        )

    async def _publish_result(
        self, updater: TaskUpdater, request: SkillRequest, result: NewsResult
    ) -> None:
        payload = result.model_dump(mode="json")
        parts = [
            new_data_part(payload, media_type="application/json"),
            new_text_part(
                result.summary or "（无摘要）",
                media_type="text/plain",
            ),
        ]
        await updater.add_artifact(
            parts,
            artifact_id=f"{updater.task_id}-result",
            name="news-result",
            metadata=_metadata(
                {
                    "query": result.query,
                    "mode": result.mode.value,
                    "counts": result.counts,
                    "degraded": result.degraded,
                    "timings_ms": result.timings_ms,
                }
            ),
        )

        succeeded = bool(result.articles or result.summary)
        if succeeded:
            note = (
                f"完成：{result.counts.get('analyzed', 0)} 篇相关新闻，"
                f"耗时 {result.duration_ms} ms"
                + ("（降级）" if result.degraded else "")
            )
            self.metrics.incr("a2a_tasks_completed")
        else:
            note = "未获得任何新闻结果，请检查关键词或稍后重试"
            self.metrics.incr("a2a_tasks_failed")

        body = f"{note}\n\n{truncate(result.summary or note, 800)}"
        message = updater.new_agent_message(
            [
                new_text_part(body),
                new_data_part(
                    {
                        "counts": result.counts,
                        "degraded": result.degraded,
                        "errors": [error.model_dump(mode="json") for error in result.errors],
                        "warnings": list(result.warnings),
                    }
                ),
            ]
        )
        try:
            if succeeded:
                await updater.complete(message)
            else:
                await updater.failed(message)
        except RuntimeError:  # pragma: no cover - cancelled while finishing
            log.info("task %s finished before the result was published", updater.task_id)

    # ------------------------------------------------------------------
    def in_flight(self) -> int:
        return sum(1 for handle in self._running.values() if not handle.done())

    async def aclose(self) -> None:
        for handle in list(self._running.values()):
            if not handle.done():
                handle.cancel()


__all__ = [
    "NewsAgentExecutor",
    "SkillRequestError",
    "TERMINAL_STATES",
    "VALID_SKILL_STRINGS",
    "aparse_skill_request",
    "parse_skill_request",
]
