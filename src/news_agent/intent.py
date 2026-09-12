"""自然语言查询意图解析（主题 + 时间要素）——LLM 优先，正则回退。

此前查询中的主题与时间要素（「本周/昨天/最近7天…」）由
:mod:`news_agent.time_window` 的硬编码正则解析，覆盖的表达有限，且中文
口语变化多端。现在优先把整句交给 LLM 做结构化解析：

* ``search_query`` —— 剥离时间词与口语填充词后的搜索关键词；
* ``time_type``    —— 标准化的时间枚举（today/this_week/past_days/date_range…）。

LLM 未配置或调用失败时，透明回退到原有的正则解析，保证离线可用。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

from .config import LLMSettings
from .models import utcnow
from .runtime import get_logger
from .time_window import parse_time_window

log = get_logger("intent")

INTENT_SYSTEM_PROMPT = (
    "You parse a user's news request into search keywords and a time window. "
    "Extract the essential topic keywords only (drop politeness phrases and "
    "time words from the keywords). Never invent dates. Always answer with "
    "the requested JSON structure."
)

INTENT_PROMPT = """Current date and time (UTC): {now}.

User request: {query}

Rules:
- `search_query`: concise search keywords for a news search engine. Remove
  politeness/filler words (帮我、请、汇总、please、summarise...) and any time
  words; keep the topic itself, in its original language.
- `time_type`: the time window expressed by the request, choose exactly one:
  "none" (no time constraint), "today", "yesterday", "this_week", "last_week",
  "this_month", "last_month", "this_year", "past_days" (the last N days),
  "date_range" (an explicit start/end date).
- `past_days`: only for "past_days", the number of days N (integer >= 1).
- `start_date` / `end_date`: only for "date_range", ISO dates YYYY-MM-DD
  (end date inclusive). null otherwise.
"""


class _LLMIntent(BaseModel):
    """Structured extraction of the user's query intent."""

    search_query: str = Field(
        description="Concise search keywords with politeness and time words removed."
    )
    time_type: Literal[
        "none",
        "today",
        "yesterday",
        "this_week",
        "last_week",
        "this_month",
        "last_month",
        "this_year",
        "past_days",
        "date_range",
    ] = Field(default="none", description="The time window expressed by the request.")
    past_days: int | None = Field(
        default=None, description="Number of days for time_type == 'past_days'."
    )
    start_date: str | None = Field(
        default=None, description="Inclusive ISO start date for time_type == 'date_range'."
    )
    end_date: str | None = Field(
        default=None, description="Inclusive ISO end date for time_type == 'date_range'."
    )


class LLMIntentError(RuntimeError):
    """Raised when the LLM could not produce a usable intent."""


def _midnight(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_iso_date(value: str, now: datetime) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo)
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def intent_to_window(intent: _LLMIntent, *, now: datetime) -> tuple[datetime | None, datetime | None, str]:
    """Turn a structured intent into ``(since, until, cleaned_query)``."""
    since: datetime | None = None
    until: datetime | None = None
    kind = intent.time_type
    if kind == "today":
        since = _midnight(now)
    elif kind == "yesterday":
        since = _midnight(now) - timedelta(days=1)
        until = _midnight(now)
    elif kind == "this_week":
        since = _midnight(now) - timedelta(days=now.weekday())
    elif kind == "last_week":
        this_monday = _midnight(now) - timedelta(days=now.weekday())
        since = this_monday - timedelta(days=7)
        until = this_monday
    elif kind == "this_month":
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif kind == "last_month":
        this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        since = (this_month - timedelta(days=1)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        until = this_month
    elif kind == "this_year":
        since = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif kind == "past_days":
        days = intent.past_days or 7
        since = now - timedelta(days=min(max(days, 1), 365))
    elif kind == "date_range":
        if intent.start_date:
            since = _parse_iso_date(intent.start_date, now)
        if intent.end_date:
            until = _parse_iso_date(intent.end_date, now)
            if until is not None:
                until = until + timedelta(days=1)  # inclusive end -> exclusive bound
    cleaned = intent.search_query.strip()
    return since, until, cleaned


class LLMIntentParser:
    """LLM-backed query intent parser (single request, structured output)."""

    name = "llm-intent"
    uses_llm = True

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.max_attempts = max(1, settings.max_retries + 1)
        self._llm: Any | None = None

    # ------------------------------------------------------------------
    def _get_llm(self) -> Any:
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise LLMIntentError("langchain-openai is not installed") from exc
            self._llm = ChatOpenAI(
                model=self.settings.model,
                temperature=self.settings.temperature,
                base_url=self.settings.resolved_base_url(),
                api_key=SecretStr(self.settings.resolved_api_key()),
                timeout=self.settings.timeout_s,
                max_retries=0,
            )
        return self._llm

    def _structured(self) -> Any:
        return self._get_llm().with_structured_output(_LLMIntent, include_raw=True)

    # ------------------------------------------------------------------
    async def parse(
        self, query: str, *, now: datetime | None = None
    ) -> tuple[datetime | None, datetime | None, str]:
        """Parse ``query`` into ``(since, until, cleaned_query)`` via the LLM."""
        now = now or utcnow()
        prompt = INTENT_PROMPT.format(now=now.isoformat(), query=query)
        messages = [("system", INTENT_SYSTEM_PROMPT), ("human", prompt)]
        intent = await self._invoke(messages)
        since, until, cleaned = intent_to_window(intent, now=now)
        if not cleaned:
            cleaned = query.strip()
        return since, until, cleaned

    async def _invoke(self, messages: Any) -> _LLMIntent:
        error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                result = await self._structured().ainvoke(messages)
            except Exception as exc:  # noqa: BLE001 - provider specific
                error = exc
                log.debug("intent structured call failed: %r", exc)
                continue
            return self._extract(result)
        raise LLMIntentError(str(error) if error else "LLM returned no intent result")

    @staticmethod
    def _extract(result: Any) -> _LLMIntent:
        if isinstance(result, _LLMIntent):
            return result
        if isinstance(result, dict):
            return _LLMIntent.model_validate(result)
        # include_raw=True 的返回：{"raw": ..., "parsed": ...}
        raw = result
        parsed = getattr(result, "parsed", None)
        if isinstance(parsed, _LLMIntent):
            return parsed
        if isinstance(parsed, dict):
            return _LLMIntent.model_validate(parsed)
        text = getattr(raw, "content", "") or ""
        if isinstance(text, list):  # some providers return content blocks
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in text
            )
        try:
            return _LLMIntent.model_validate(json.loads(text))
        except (ValueError, TypeError) as exc:
            raise LLMIntentError(f"could not parse intent output: {exc}") from exc


async def parse_query_intent(
    query: str,
    llm_settings: LLMSettings,
    *,
    now: datetime | None = None,
) -> tuple[datetime | None, datetime | None, str]:
    """Best-effort intent parsing: LLM when available, regex otherwise.

    @returns (since, until, cleaned_query)，语义与 :func:`parse_time_window`
             完全一致，可直接替换使用。
    """
    if not llm_settings.configured:
        return parse_time_window(query, now=now)
    try:
        parser = LLMIntentParser(llm_settings)
        return await parser.parse(query, now=now)
    except Exception as exc:  # noqa: BLE001 - degrade to the regex parser
        log.warning("LLM intent parsing failed (%r), falling back to regex", exc)
        return parse_time_window(query, now=now)


__all__ = [
    "LLMIntentError",
    "LLMIntentParser",
    "intent_to_window",
    "parse_query_intent",
]
