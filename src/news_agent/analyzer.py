"""Structured analysis + summarisation (TODO items 9/11/12 and the LLM part).

Three implementations of the same tiny protocol:

``HeuristicAnalyzer``
    Zero-dependency, deterministic, offline.  Always available and used as the
    degradation target when the LLM is not configured or fails.
``LLMAnalyzer``
    Structured extraction through an OpenAI-compatible endpoint (OpenAI, vLLM,
    DeepSeek, ...), batched + concurrency limited, with map-reduce summaries.
``FallbackAnalyzer``
    Wraps the two: prefer the LLM, degrade to heuristics, and report the
    degradation through the :class:`~news_agent.runtime.RunContext`.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, SecretStr
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

from .config import LLMSettings
from .models import AnalyzedArticle, ErrorCode, RawArticle, SkillRequest
from .runtime import RunContext, get_logger
from .text_utils import split_sentences, tokenize, truncate

log = get_logger("analyzer")

# ---------------------------------------------------------------------------
# tiny lexicons for the offline sentiment estimator
# ---------------------------------------------------------------------------
POSITIVE_WORDS = (
    "增长", "突破", "利好", "上涨", "提升", "领先", "成功", "创新", "改善", "回暖",
    "支持", "合作", "签约", "获奖", "看好", "上調", "上调", "扩产", "盈利", "超预期",
    "增长点", "机会", "繁荣", "复苏", "获批", "量产", "record", "growth", "surge",
    "profit", "beat", "success", "breakthrough", "approve", "partnership", "rally",
)
NEGATIVE_WORDS = (
    "下跌", "下滑", "亏损", "风险", "警告", "处罚", "违规", "延期", "裁员", "破产",
    "下滑", "质疑", "争议", "召回", "事故", "收紧", "监管", "压力", "sla", "conflict",
    "fall", "drop", "loss", "risk", "warning", "fine", "lawsuit", "delay", "layoff",
    "bankrupt", "probe", "recall", "accident", "scrutiny", "pressure",
)

_ENTITY_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9&.\-]{2,}(?:\s+[A-Z][A-Za-z0-9&.\-]{2,}){0,3})")
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _lexicon_sentiment(
    text: str,
) -> tuple[Literal["positive", "neutral", "negative"], float]:
    if not text:
        return "neutral", 0.0
    positive = sum(text.count(word) for word in POSITIVE_WORDS)
    negative = sum(text.count(word) for word in NEGATIVE_WORDS)
    total = positive + negative
    if total == 0:
        return "neutral", 0.0
    score = (positive - negative) / total
    if score > 0.2:
        return "positive", round(score, 3)
    if score < -0.2:
        return "negative", round(score, 3)
    return "neutral", round(score, 3)


def extract_entities(text: str, query: str = "", top_n: int = 5) -> list[str]:
    """Cheap entity extraction: latin proper nouns + salient CJK terms."""
    if not text:
        return []
    entities: list[str] = []

    for match in _ENTITY_TOKEN_RE.findall(text):
        cleaned = match.strip()
        if 2 <= len(cleaned) <= 40:
            entities.append(cleaned)

    # CJK: prefer multi-character terms from the title-like beginning
    query_tokens = set(tokenize(query)) if query else set()
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 2:
            continue
        counts[token] = counts.get(token, 0) + 1
    cjk_ranked = sorted(
        counts.items(),
        key=lambda item: (
            0 if item[0] in query_tokens else 1,
            -len(item[0]) * item[1],
        ),
    )
    for token, _count in cjk_ranked:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,6}", token):
            entities.append(token)
        if len(entities) >= top_n * 2:
            break

    unique: list[str] = []
    for entity in entities:
        if entity and entity not in unique:
            unique.append(entity)
    return unique[:top_n]


def _key_points(text: str, top_n: int = 3) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return [truncate(text, 120)] if text else []
    ranked = sorted(
        sentences,
        key=lambda sentence: (sum(1 for _ in tokenize(sentence)), len(sentence)),
        reverse=True,
    )
    points: list[str] = []
    for sentence in ranked[: top_n * 2]:
        cleaned = truncate(sentence, 140)
        if cleaned and cleaned not in points:
            points.append(cleaned)
        if len(points) >= top_n:
            break
    return points


def light_analyze(articles: Iterable[RawArticle]) -> list[AnalyzedArticle]:
    """No-LLM, no-lexicon conversion used by the ``fetch_news`` skill."""
    return [
        AnalyzedArticle(
            id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            published_at=article.published_at,
            language=article.language,
            summary=article.summary or article.excerpt(160) or None,
            duplicate_sources=list(article.duplicate_sources),
        )
        for article in articles
    ]


class Analyzer(Protocol):
    """Structural interface implemented by every analyser.

    ``HeuristicAnalyzer``, ``LLMAnalyzer`` and ``FallbackAnalyzer`` all satisfy
    it, and so does any custom analyser (including test doubles) — the graph only
    depends on this behaviour, not on a concrete class.
    """

    name: str
    uses_llm: bool

    async def analyze(
        self,
        request: SkillRequest,
        articles: Sequence[RawArticle],
        ctx: RunContext | None = None,
    ) -> list[AnalyzedArticle]: ...

    async def summarize(
        self,
        request: SkillRequest,
        articles: Sequence[AnalyzedArticle],
        ctx: RunContext | None = None,
        *,
        previous: str | None = None,
    ) -> str: ...


class HeuristicAnalyzer:
    """Deterministic offline analyser."""

    name = "heuristic"
    uses_llm = False

    async def analyze(
        self,
        request: SkillRequest,
        articles: Sequence[RawArticle],
        ctx: RunContext | None = None,
    ) -> list[AnalyzedArticle]:
        results: list[AnalyzedArticle] = []
        for article in articles:
            text = f"{article.title}。{article.summary or ''}{article.excerpt(600)}"
            sentiment, score = _lexicon_sentiment(text)
            stance = {"positive": "支持/看好", "negative": "质疑/担忧", "neutral": None}[sentiment]
            results.append(
                AnalyzedArticle(
                    id=article.id,
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    published_at=article.published_at,
                    language=article.language,
                    entities=extract_entities(text, request.query),
                    events=_key_points(f"{article.title}。{article.summary or ''}", 2),
                    sentiment=sentiment,  # type: ignore[arg-type]
                    sentiment_score=score,
                    stance=stance,
                    key_points=_key_points(text, 3),
                    summary=truncate(article.summary or article.excerpt(300), 200) or article.title,
                    duplicate_sources=list(article.duplicate_sources),
                )
            )
        return results

    async def summarize(
        self,
        request: SkillRequest,
        articles: Sequence[AnalyzedArticle],
        ctx: RunContext | None = None,
        *,
        previous: str | None = None,
    ) -> str:
        if not articles:
            return ""
        positives = sum(1 for item in articles if item.sentiment == "positive")
        negatives = sum(1 for item in articles if item.sentiment == "negative")
        sources = sorted({item.source for item in articles})
        headline = (
            f"共汇总 {len(articles)} 篇关于「{request.query}」的报道"
            f"（来源：{', '.join(sources[:5])}）。"
            f"其中偏正面 {positives} 篇、偏负面 {negatives} 篇。"
        )
        bullets = []
        for item in articles[:5]:
            bullet = item.summary or item.title
            if bullet:
                bullets.append(f"- {truncate(bullet, 110)}")
        body = "\n".join(bullets)
        summary = f"{headline}\n{body}".strip()
        if previous:
            summary = f"{previous}\n{summary}"
        return summary


# ---------------------------------------------------------------------------
# LLM powered analyser
# ---------------------------------------------------------------------------
class _ItemAnalysis(BaseModel):
    """Structured extraction of a single article."""

    index: int = Field(description="The index of the article as provided in the input.")
    entities: list[str] = Field(
        default_factory=list, description="Named entities: companies, people, products, places."
    )
    events: list[str] = Field(
        default_factory=list, description="Concrete events described by the article."
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        default="neutral", description="Overall sentiment towards the query topic."
    )
    sentiment_score: float = Field(
        default=0.0, description="Sentiment score between -1 (very negative) and 1 (very positive)."
    )
    stance: str | None = Field(
        default=None, description="The article's stance / perspective, if any (short phrase)."
    )
    key_points: list[str] = Field(default_factory=list, description="2-4 key points.")
    summary: str = Field(default="", description="One sentence summary in the requested language.")


class _BatchAnalysis(BaseModel):
    results: list[_ItemAnalysis] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a meticulous news analyst. You extract structured facts from news "
    "articles. Only use information present in the provided text, never invent "
    "facts. Always answer with the requested JSON structure."
)

_MAP_PROMPT = """News articles related to the query "{query}":

{payload}

Return one analysis object per article. Rules:
- echo the exact `index` of the article
- `entities`, `events`, `key_points` must be short strings
- `summary` is ONE sentence in {language_name}
- `sentiment` is the sentiment towards the query "{query}" (not towards the publisher)
- keep the total answer concise
"""

_REDUCE_PROMPT = """You are given partial summaries about the query "{query}".

{payload}

Write a single coherent synthesis in {language_name} of at most 300 words:
1. one sentence on the overall picture,
2. 3-5 bullet points on the most important facts or divergences,
3. one closing sentence on what remains uncertain or contested.
Do not add information that is not in the partial summaries.
"""

_CHUNK_PROMPT = """Summarise the following batch of news items about "{query}" in {language_name}.
Focus on facts, numbers, named entities and any disagreement between sources.
Output 2-4 bullet points, nothing else.

{payload}
"""

_LANGUAGE_NAMES = {"zh": "Simplified Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}


class LLMError(RuntimeError):
    """Raised when the LLM could not produce a usable answer."""


class LLMAnalyzer:
    """Batched, concurrency limited structured extraction via an LLM."""

    name = "llm"
    uses_llm = True

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.max_attempts = max(1, settings.max_retries + 1)
        self._llm: Any | None = None
        self._structured_cache: dict[str, Any] = {}
        self._preferred_method: str | None = None

    # ------------------------------------------------------------------
    def _get_llm(self) -> Any:
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise LLMError("langchain-openai is not installed") from exc
            self._llm = ChatOpenAI(
                model=self.settings.model,
                temperature=self.settings.temperature,
                base_url=self.settings.resolved_base_url(),
                api_key=SecretStr(self.settings.resolved_api_key()),
                timeout=self.settings.timeout_s,
                max_retries=0,
            )
        return self._llm

    def _structured(self, method: str) -> Any:
        if method not in self._structured_cache:
            self._structured_cache[method] = self._get_llm().with_structured_output(
                _BatchAnalysis, method=method, include_raw=True
            )
        return self._structured_cache[method]

    def _methods(self) -> list[str]:
        if self._preferred_method:
            return [self._preferred_method]
        configured = self.settings.structured_method
        order = [configured, "json_schema", "function_calling", "json_mode"]
        seen: list[str] = []
        for method in order:
            if method not in seen:
                seen.append(method)
        return seen

    # ------------------------------------------------------------------
    def _language_name(self, language: str | None) -> str:
        return _LANGUAGE_NAMES.get((language or "zh").lower()[:2], "Chinese")

    @staticmethod
    def _payload(items: Sequence[tuple[int, RawArticle]], max_chars: int) -> str:
        rows = []
        for index, article in items:
            rows.append(
                {
                    "index": index,
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat()
                    if article.published_at
                    else None,
                    "excerpt": article.excerpt(max_chars),
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=1)

    @staticmethod
    def _extract(parsed: Any, raw: Any) -> _BatchAnalysis:
        if isinstance(parsed, _BatchAnalysis):
            return parsed
        if isinstance(parsed, dict) and "results" in parsed:
            return _BatchAnalysis.model_validate(parsed)
        text = getattr(raw, "content", "") or ""
        if isinstance(text, list):  # some providers return content blocks
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in text
            )
        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(1)
        try:
            return _BatchAnalysis.model_validate(json.loads(text))
        except (ValueError, TypeError) as exc:
            raise LLMError(f"could not parse structured output: {exc}") from exc

    async def _call_batch_once(self, messages: Any, ctx: RunContext | None) -> _BatchAnalysis:
        error: Exception | None = None
        for method in self._methods():
            try:
                result = await self._structured(method).ainvoke(messages)
            except Exception as exc:  # noqa: BLE001 - provider specific
                error = exc
                log.debug("structured output via %s failed: %r", method, exc)
                continue
            self._preferred_method = method
            if isinstance(result, dict):
                self._note_usage(ctx, result.get("raw"))
                return self._extract(result.get("parsed"), result.get("raw"))
            return self._extract(result, None)
        raise LLMError(str(error) if error else "LLM returned no result")

    async def _call_batch(
        self,
        request: SkillRequest,
        items: Sequence[tuple[int, RawArticle]],
        ctx: RunContext | None = None,
    ) -> _BatchAnalysis:
        payload = self._payload(items, self.settings.max_excerpt_chars)
        prompt = _MAP_PROMPT.format(
            query=request.query,
            payload=payload,
            language_name=self._language_name(request.language),
        )
        messages = [("system", SYSTEM_PROMPT), ("human", prompt)]
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential_jitter(initial=0.6, max=6.0),
                reraise=True,
            ):
                with attempt:
                    return await self._call_batch_once(messages, ctx)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise provider errors
            raise LLMError(f"LLM call failed after {self.max_attempts} attempts: {exc}") from exc
        raise LLMError("LLM call failed")

    @staticmethod
    def _note_usage(ctx: RunContext | None, raw: Any) -> None:
        if ctx is not None:
            ctx.note_llm_usage(getattr(raw, "usage_metadata", None))

    # ------------------------------------------------------------------
    async def analyze(
        self,
        request: SkillRequest,
        articles: Sequence[RawArticle],
        ctx: RunContext | None = None,
    ) -> list[AnalyzedArticle]:
        if not articles:
            return []
        batch_size = max(1, self.settings.batch_size)
        indexed = list(enumerate(articles))
        batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]

        semaphore = asyncio.Semaphore(max(1, self.settings.concurrency))

        async def _run(batch: list[tuple[int, RawArticle]]) -> dict[int, _ItemAnalysis]:
            async with semaphore:
                parsed = await self._call_batch(request, batch, ctx)
                return {item.index: item for item in parsed.results}

        outcomes = await asyncio.gather(
            *(_run(batch) for batch in batches), return_exceptions=True
        )

        merged: dict[int, _ItemAnalysis] = {}
        failures: list[str] = []
        for batch, outcome in zip(batches, outcomes):
            if isinstance(outcome, BaseException):
                failures.append(f"batch of {len(batch)} articles: {outcome}")
                if ctx is not None:
                    ctx.add_error(
                        ErrorCode.LLM_FAILED,
                        f"LLM batch failed: {outcome}",
                        stage="analyze",
                        retryable=True,
                    )
                continue
            merged.update(outcome)

        heuristic = HeuristicAnalyzer()
        missing = [(index, article) for index, article in indexed if index not in merged]
        heuristic_results: dict[int, AnalyzedArticle] = {}
        if missing:
            if ctx is not None:
                ctx.add_warning(
                    f"{len(missing)} 篇文章未获得 LLM 结构化结果，已使用启发式分析补齐"
                )
            fallback = await heuristic.analyze(
                request, [article for _, article in missing], ctx
            )
            heuristic_results = {
                index: item for (index, _), item in zip(missing, fallback)
            }
        if failures and ctx is not None:
            ctx.count("llm_batch_failures", len(failures))

        results: list[AnalyzedArticle] = []
        for index, article in indexed:
            item = merged.get(index)
            if item is not None:
                results.append(
                    AnalyzedArticle(
                        id=article.id,
                        title=article.title,
                        url=article.url,
                        source=article.source,
                        published_at=article.published_at,
                        language=article.language,
                        entities=item.entities[:8],
                        events=item.events[:6],
                        sentiment=item.sentiment,
                        sentiment_score=round(max(-1.0, min(1.0, item.sentiment_score)), 3),
                        stance=(item.stance or None),
                        key_points=item.key_points[:5],
                        summary=item.summary or article.summary,
                        duplicate_sources=list(article.duplicate_sources),
                    )
                )
            else:
                results.append(heuristic_results[index])
        return results

    # ------------------------------------------------------------------
    async def summarize(
        self,
        request: SkillRequest,
        articles: Sequence[AnalyzedArticle],
        ctx: RunContext | None = None,
        *,
        previous: str | None = None,
    ) -> str:
        if not articles:
            return ""
        chunk_size = max(1, self.settings.summary_chunk_size)
        chunks = [
            list(articles[i : i + chunk_size])
            for i in range(0, len(articles), chunk_size)
        ]

        # map ---------------------------------------------------------
        semaphore = asyncio.Semaphore(max(1, self.settings.concurrency))

        async def _map(chunk: list[AnalyzedArticle]) -> str:
            async with semaphore:
                return await self._summarize_chunk(request, chunk, ctx)

        partials = await asyncio.gather(*(_map(chunk) for chunk in chunks), return_exceptions=True)
        usable: list[str] = []
        for outcome in partials:
            if isinstance(outcome, BaseException):
                if ctx is not None:
                    ctx.add_error(
                        ErrorCode.LLM_FAILED,
                        f"partial summary failed: {outcome}",
                        stage="summarize",
                        retryable=True,
                    )
                continue
            if outcome.strip():
                usable.append(outcome.strip())

        if not usable:
            raise LLMError("all partial summaries failed")

        # reduce ------------------------------------------------------
        body = "\n\n".join(f"### 批次 {index + 1}\n{text}" for index, text in enumerate(usable))
        try:
            summary = await self._reduce(request, body, ctx)
        except Exception as exc:  # noqa: BLE001 - degrade to concatenation
            if ctx is not None:
                ctx.add_warning(f"汇总阶段失败，已退化为直接拼接分批小结（{exc}）")
            summary = "\n\n".join(usable)
        if previous:
            summary = f"{previous}\n\n{summary}"
        return summary.strip()

    async def _summarize_chunk(
        self,
        request: SkillRequest,
        chunk: list[AnalyzedArticle],
        ctx: RunContext | None = None,
    ) -> str:
        payload = "\n".join(
            f"- [{(item.published_at.isoformat() if item.published_at else 'n/a')}] "
            f"{item.title} | {item.source} | {item.summary or ''}"
            for item in chunk
        )
        prompt = _CHUNK_PROMPT.format(
            query=request.query,
            payload=payload,
            language_name=self._language_name(request.language),
        )
        return await self._call_text(prompt, ctx)

    async def _reduce(
        self, request: SkillRequest, payload: str, ctx: RunContext | None = None
    ) -> str:
        prompt = _REDUCE_PROMPT.format(
            query=request.query,
            payload=payload,
            language_name=self._language_name(request.language),
        )
        return await self._call_text(prompt, ctx)

    async def _call_text(self, prompt: str, ctx: RunContext | None = None) -> str:
        async def _once() -> str:
            try:
                message = await self._get_llm().ainvoke(
                    [("system", SYSTEM_PROMPT), ("human", prompt)]
                )
            except Exception as exc:  # noqa: BLE001 - provider specific
                raise LLMError(str(exc)) from exc
            self._note_usage(ctx, message)
            content = getattr(message, "content", "")
            if isinstance(content, list):  # pragma: no cover - multi-part content
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            text = str(content).strip()
            if not text:
                raise LLMError("empty completion")
            return text

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential_jitter(initial=0.6, max=6.0),
                reraise=True,
            ):
                with attempt:
                    return await _once()
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise provider errors
            raise LLMError(f"LLM call failed after {self.max_attempts} attempts: {exc}") from exc
        raise LLMError("LLM call failed")


class FallbackAnalyzer:
    """Prefer the LLM, transparently degrade to heuristics."""

    def __init__(self, primary: Analyzer, fallback: Analyzer) -> None:
        self.primary: Analyzer = primary
        self.fallback: Analyzer = fallback
        self.name = "llm+heuristic"
        self.uses_llm = True
        self.degraded = False

    async def analyze(
        self,
        request: SkillRequest,
        articles: Sequence[RawArticle],
        ctx: RunContext | None = None,
    ) -> list[AnalyzedArticle]:
        try:
            results = await self.primary.analyze(request, articles, ctx)
            if len(results) == len(articles):
                return results
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail
            self.degraded = True
            if ctx is not None:
                ctx.add_error(
                    ErrorCode.LLM_FAILED,
                    f"LLM analysis unavailable, degraded to heuristics: {exc}",
                    stage="analyze",
                    retryable=True,
                )
                ctx.add_warning("LLM 分析不可用，已降级为启发式分析")
            else:  # pragma: no cover - no context available
                log.warning("LLM analysis failed: %r", exc)
        return await self.fallback.analyze(request, articles, ctx)

    async def summarize(
        self,
        request: SkillRequest,
        articles: Sequence[AnalyzedArticle],
        ctx: RunContext | None = None,
        *,
        previous: str | None = None,
    ) -> str:
        try:
            return await self.primary.summarize(request, articles, ctx, previous=previous)
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail
            self.degraded = True
            if ctx is not None:
                ctx.add_error(
                    ErrorCode.LLM_FAILED,
                    f"LLM summarisation unavailable: {exc}",
                    stage="summarize",
                    retryable=True,
                )
                ctx.add_warning("LLM 摘要不可用，已使用启发式摘要")
            else:  # pragma: no cover
                log.warning("LLM summarisation failed: %r", exc)
            return await self.fallback.summarize(request, articles, ctx, previous=previous)


def build_analyzer(settings: Any) -> Analyzer:
    """Return the best analyser available for the current configuration."""
    heuristic = HeuristicAnalyzer()
    if settings.llm.configured:
        return FallbackAnalyzer(LLMAnalyzer(settings.llm), heuristic)
    return heuristic
