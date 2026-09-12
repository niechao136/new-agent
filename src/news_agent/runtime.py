"""Observability + run-scoped state.

Two pieces live here:

``Metrics``
    A tiny thread-safe counter/histogram registry exposed through the
    ``/metrics`` endpoint (node latency, fetch success rate, LLM tokens, ...).

``RunContext``
    Per-task scratch pad shared between the graph nodes and the A2A executor.
    It carries the progress event stream (used by SSE), the accumulated
    warnings/errors, per-stage timings and -- crucially -- the *latest partial
    result* so that a timed out task can still return something useful instead
    of hanging (TODO item 18).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import ErrorCode, ErrorInfo, NewsResult, utcnow

LOGGER_NAME = "news_agent"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger(LOGGER_NAME)
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class Metrics:
    """Process wide counters + latency histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self.started_at = utcnow()

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(float(value))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            histograms: dict[str, Any] = {}
            for name, values in self._histograms.items():
                if not values:
                    continue
                ordered = sorted(values)
                count = len(ordered)
                histograms[name] = {
                    "count": count,
                    "sum": round(sum(ordered), 2),
                    "avg": round(sum(ordered) / count, 2),
                    "p50": round(ordered[count // 2], 2),
                    "p95": round(ordered[min(count - 1, int(count * 0.95))], 2),
                    "max": round(ordered[-1], 2),
                }
        return {
            "started_at": self.started_at.isoformat(),
            "counters": counters,
            "histograms": histograms,
        }

    def reset(self) -> None:  # pragma: no cover - used by tests/tools
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


@dataclass
class ProgressEvent:
    """A single progress notification (streamed to SSE subscribers)."""

    event: str
    message: str
    stage: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
            "at": self.at.isoformat(),
        }


#: Progress subscribers may be sync or async (async ones are scheduled by
#: :meth:`RunContext.emit` and flushed by :meth:`RunContext.drain`).
ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


class RunContext:
    """Mutable state shared by graph nodes, executor and SSE subscribers."""

    def __init__(
        self,
        *,
        task_id: str | None = None,
        metrics: Metrics | None = None,
        log_level: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.metrics = metrics
        self.started_at = utcnow()
        self.stage: str = "submitted"
        self.events: list[ProgressEvent] = []
        self.timings: dict[str, float] = {}
        self.counters: dict[str, int] = defaultdict(int)
        self.warnings: list[str] = []
        self.errors: list[ErrorInfo] = []
        self.partial: NewsResult | None = None
        self.canceled = False
        self._callbacks: list[ProgressCallback] = []
        self._pending: set[asyncio.Task[None]] = set()
        self._lock = threading.Lock()
        if log_level:
            configure_logging(log_level)
        self.log = get_logger("run")

    # -- progress ---------------------------------------------------------
    def subscribe(self, callback: ProgressCallback) -> None:
        self._callbacks.append(callback)

    def unsubscribe(self, callback: ProgressCallback) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(
        self,
        stage: str,
        message: str,
        *,
        event: str = "status",
        **data: Any,
    ) -> ProgressEvent:
        """Publish a progress event.

        Subscribers may be sync callables or coroutine functions.  Async
        subscribers are scheduled on the running loop (never awaited inline) so
        that a graph node is never blocked by a slow transport; use
        :meth:`drain` before publishing a terminal event to guarantee ordering.
        """
        self.stage = stage
        payload = ProgressEvent(event=event, stage=stage, message=message, data=data)
        self.events.append(payload)
        self.log.info("[%s] %s %s", stage, message, data or "")
        for callback in list(self._callbacks):
            try:
                result = callback(payload)
            except Exception:  # pragma: no cover - subscribers must not break runs
                self.log.exception("progress subscriber failed")
                continue
            if inspect.isawaitable(result):
                self._schedule_awaitable(result)
        return payload

    def _schedule_awaitable(self, awaitable: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no loop, nothing to await
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            return
        task = loop.create_task(awaitable)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Wait until every async progress subscriber has been flushed."""
        while self._pending:
            pending = list(self._pending)
            await asyncio.gather(*pending, return_exceptions=True)

    # -- bookkeeping ------------------------------------------------------
    def count(self, name: str, value: int = 1) -> None:
        self.counters[name] += value
        if self.metrics is not None:
            self.metrics.incr(name, value)

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add_error(
        self,
        code: ErrorCode,
        message: str,
        *,
        source: str | None = None,
        stage: str | None = None,
        retryable: bool = False,
    ) -> ErrorInfo:
        error = ErrorInfo(
            code=code,
            message=message,
            source=source,
            stage=stage,
            retryable=retryable,
        )
        self.errors.append(error)
        self.log.warning("error[%s] %s (source=%s stage=%s)", code, message, source, stage)
        return error

    @contextmanager
    def timeit(self, stage: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timings[stage] = round(self.timings.get(stage, 0.0) + elapsed_ms, 2)
            if self.metrics is not None:
                self.metrics.observe(f"stage_ms.{stage}", elapsed_ms)

    def elapsed_ms(self) -> int:
        return int((utcnow() - self.started_at).total_seconds() * 1000)

    # -- partial results ---------------------------------------------------
    def set_partial(self, result: NewsResult) -> None:
        self.partial = result

    def note_llm_usage(self, usage: Any) -> None:
        """Record token usage coming from LangChain's ``usage_metadata``."""
        if not usage:
            return
        if isinstance(usage, dict):
            prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        else:  # pragma: no cover - pydantic usage object
            prompt = getattr(usage, "input_tokens", 0) or 0
            completion = getattr(usage, "output_tokens", 0) or 0
        self.count("llm_prompt_tokens", int(prompt))
        self.count("llm_completion_tokens", int(completion))
