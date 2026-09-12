"""Domain models shared by every layer of the agent.

The models double as the *contract* advertised in the A2A agent card: the
``inputSchema`` of each skill maps onto :class:`SkillRequest` and the
``outputSchema`` onto :class:`NewsResult`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .text_utils import normalize_url


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def make_article_id(url: str, title: str = "") -> str:
    key = normalize_url(url) or title
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class SkillMode(str, Enum):
    """The three skills exposed in the agent card."""

    FETCH = "fetch_news"
    SUMMARIZE = "summarize_news"
    TREND = "analyze_trend"

    @classmethod
    def coerce(cls, value: Any) -> "SkillMode":
        if isinstance(value, cls):
            return value
        raw = str(value or "").strip().lower()
        aliases = {
            "fetch": cls.FETCH,
            "fetch_news": cls.FETCH,
            "fetchnews": cls.FETCH,
            "summarize": cls.SUMMARIZE,
            "summarise": cls.SUMMARIZE,
            "summarize_news": cls.SUMMARIZE,
            "summary": cls.SUMMARIZE,
            "analyze": cls.TREND,
            "analyse": cls.TREND,
            "trend": cls.TREND,
            "analyze_trend": cls.TREND,
            "analyze_trends": cls.TREND,
        }
        return aliases.get(raw, cls.SUMMARIZE)


class ErrorCode(str, Enum):
    """Stable machine readable error codes of the degradation contract."""

    INVALID_REQUEST = "invalid_request"
    NO_RESULTS = "no_results"
    SOURCE_UNAVAILABLE = "source_unavailable"
    FETCH_TIMEOUT = "fetch_timeout"
    RATE_LIMITED = "rate_limited"
    PARSE_ERROR = "parse_error"
    LLM_FAILED = "llm_failed"
    LLM_TIMEOUT = "llm_timeout"
    CONTEXT_TOO_LONG = "context_too_long"
    CACHE_ERROR = "cache_error"
    TASK_TIMEOUT = "task_timeout"
    TASK_CANCELED = "task_canceled"
    INTERNAL_ERROR = "internal_error"


class ErrorInfo(BaseModel):
    """A single recoverable/terminal problem reported back to the caller."""

    code: ErrorCode
    message: str
    source: str | None = None
    stage: str | None = None
    retryable: bool = False


class RawArticle(BaseModel):
    """An article exactly as returned by a :class:`NewsSource`."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    url: str
    source: str
    published_at: datetime | None = None
    summary: str | None = None
    content: str | None = None
    language: str | None = None
    author: str | None = None
    image_url: str | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    duplicate_sources: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", mode="before")
    @classmethod
    def _aware(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return ensure_aware(value)
        return value

    @property
    def text(self) -> str:
        parts = [self.title or "", self.summary or "", self.content or ""]
        return "\n".join(part for part in parts if part).strip()

    def excerpt(self, limit: int = 800) -> str:
        for candidate in (self.content, self.summary):
            if candidate:
                return candidate.strip()[:limit]
        return ""


class AnalyzedArticle(BaseModel):
    """Per-article structured extraction (LLM or heuristic)."""

    id: str
    title: str
    url: str
    source: str
    published_at: datetime | None = None
    language: str | None = None
    relevance: float = 0.0
    entities: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    #: -1.0 (very negative) .. 1.0 (very positive)
    sentiment_score: float = 0.0
    stance: str | None = None
    key_points: list[str] = Field(default_factory=list)
    summary: str | None = None
    duplicate_sources: list[str] = Field(default_factory=list)


class TrendInsight(BaseModel):
    """Topic level aggregation used by the ``analyze_trend`` skill."""

    topic: str
    mentions: int
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    average_sentiment: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    representative_urls: list[str] = Field(default_factory=list)


class NewsResult(BaseModel):
    """The final artifact returned to the A2A caller."""

    query: str
    mode: SkillMode = SkillMode.SUMMARIZE
    language: str = "zh"
    generated_at: datetime = Field(default_factory=utcnow)
    duration_ms: int = 0
    #: True when one or more stages degraded (cache hit fallback, heuristic
    #: analysis instead of LLM, partial timeout, ...).
    degraded: bool = False
    counts: dict[str, int] = Field(default_factory=dict)
    summary: str = ""
    articles: list[AnalyzedArticle] = Field(default_factory=list)
    trends: list[TrendInsight] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(
            error.code
            in {
                ErrorCode.INVALID_REQUEST,
                ErrorCode.INTERNAL_ERROR,
                ErrorCode.NO_RESULTS,
            }
            for error in self.errors
        )


class SkillRequest(BaseModel):
    """Validated, normalised invocation of one skill."""

    skill: SkillMode = SkillMode.SUMMARIZE
    query: str
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 15
    language: str = "zh"
    sources: list[str] | None = None
    threshold: float | None = None
    include_analyzed: bool = True
    context_id: str | None = None

    @field_validator("since", "until", mode="before")
    @classmethod
    def _aware(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return ensure_aware(value)
        if isinstance(value, str) and value:
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            return ensure_aware(parsed)
        return value

    @field_validator("query", mode="before")
    @classmethod
    def _clean_query(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    def resolved_since(self, default_days: int = 7) -> datetime:
        return self.since or (utcnow() - timedelta(days=default_days))

    def cache_key(self, default_days: int = 7) -> str:
        parts = [
            self.query.lower(),
            self.language,
            str(self.limit),
            ",".join(sorted(self.sources or [])),
            self.resolved_since(default_days).strftime("%Y-%m-%dT%H"),
            (self.until or utcnow()).strftime("%Y-%m-%dT%H"),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
