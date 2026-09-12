"""LLM based re-ranking of the relevance-filtered candidates.

The lexical filter in :mod:`news_agent.relevance` is cheap and recall oriented;
it still lets near-misses through (a shared generic token, an accidental CJK
bigram) and it cannot judge whether an article *actually* answers the query.

This module adds a single, optional LLM pass over the candidates that survived
the lexical filter: given the query, its expansion keywords and the candidate
headlines, the model returns a relevance score per candidate.  The scores are
blended into the lexical ones so that:

* the *selection* (threshold) still follows the lexical score — the LLM cannot
  silently reduce recall;
* the *ordering* (and therefore which articles fit into the limited analysis
  budget) follows the blended score.

Every failure degrades to the lexical ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field, SecretStr

from .config import LLMSettings
from .models import RawArticle
from .runtime import RunContext, get_logger

log = get_logger("rerank")

RERANK_SYSTEM_PROMPT = (
    "You are a news relevance judge. You score how well each candidate article "
    "matches the user's information need. Judge the topic, not the publisher or "
    "the writing style. Always answer with the requested JSON structure."
)

RERANK_PROMPT = """Query: {query}
{keywords_block}
Candidates:
{payload}

For every candidate return `index` and `relevance` in [0, 1]:
- 1.0 = directly about the query topic, a reader asking this question must see it
- 0.5 = related but only partially on topic, or about a different angle
- 0.1 = mentions a keyword but is not really about the topic
- 0.0 = off topic / spam

Rules:
- echo the exact `index` of each candidate, do not invent indices
- judge against the query topic, not against the language of the candidate
- do not add any text outside the JSON structure
"""

_KEYWORDS_BLOCK = "Related keywords (any language counts as a match): {keywords}\n"


class _ItemScore(BaseModel):
    index: int = Field(description="The index of the candidate as provided.")
    relevance: float = Field(
        default=0.0, description="Relevance of the candidate to the query, in [0, 1]."
    )


class _RerankOutput(BaseModel):
    results: list[_ItemScore] = Field(default_factory=list)


class RerankError(RuntimeError):
    """Raised when the LLM could not produce usable relevance scores."""


class Reranker(Protocol):
    """Structural interface for re-rankers (LLM or any custom implementation)."""

    name: str
    uses_llm: bool

    async def rerank(
        self,
        query: str,
        keywords: Sequence[str],
        candidates: Sequence[RawArticle],
        *,
        ctx: RunContext | None = None,
    ) -> dict[str, float]: ...


class LLMReranker:
    """Scores candidates with one structured-output LLM call."""

    name = "llm-rerank"
    uses_llm = True

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._llm: Any | None = None

    # ------------------------------------------------------------------
    def _get_llm(self) -> Any:
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RerankError("langchain-openai is not installed") from exc
            self._llm = ChatOpenAI(
                model=self.settings.model,
                temperature=0.0,
                base_url=self.settings.resolved_base_url(),
                api_key=SecretStr(self.settings.resolved_api_key()),
                timeout=self.settings.timeout_s,
                max_retries=0,
            )
        return self._llm

    def _structured(self) -> Any:
        return self._get_llm().with_structured_output(_RerankOutput, include_raw=True)

    # ------------------------------------------------------------------
    async def rerank(
        self,
        query: str,
        keywords: Sequence[str],
        candidates: Sequence[RawArticle],
        *,
        ctx: RunContext | None = None,
    ) -> dict[str, float]:
        """Return ``{article_id: relevance}`` for the candidates the model scored."""
        if not candidates:
            return {}
        excerpt_chars = max(40, self.settings.rerank_max_excerpt_chars)
        payload = "\n".join(
            f"[{index}] {article.title} | {article.source} | "
            f"{(article.summary or article.excerpt(excerpt_chars))[:excerpt_chars]}"
            for index, article in enumerate(candidates)
        )
        keywords_block = (
            _KEYWORDS_BLOCK.format(keywords=", ".join(keywords)) if keywords else ""
        )
        prompt = RERANK_PROMPT.format(
            query=query, keywords_block=keywords_block, payload=payload
        )
        try:
            result = await self._structured().ainvoke(
                [("system", RERANK_SYSTEM_PROMPT), ("human", prompt)]
            )
        except Exception as exc:  # noqa: BLE001 - provider specific
            raise RerankError(str(exc)) from exc

        parsed, raw = self._unwrap(result, ctx)
        scores: dict[str, float] = {}
        for item in parsed.results:
            if 0 <= item.index < len(candidates):
                scores[candidates[item.index].id] = max(0.0, min(1.0, item.relevance))
        if not scores:
            raise RerankError("LLM returned no usable relevance scores")
        return scores

    @staticmethod
    def _unwrap(result: Any, ctx: RunContext | None) -> tuple[_RerankOutput, Any]:
        if isinstance(result, _RerankOutput):
            return result, None
        if isinstance(result, dict) and "parsed" in result:
            raw = result.get("raw")
            if ctx is not None:
                ctx.note_llm_usage(getattr(raw, "usage_metadata", None))
            parsed = result.get("parsed")
            if isinstance(parsed, _RerankOutput):
                return parsed, raw
            if isinstance(parsed, dict):
                return _RerankOutput.model_validate(parsed), raw
            raise RerankError("could not parse rerank output")
        if isinstance(result, _RerankOutput):  # pragma: no cover - defensive
            return result, None
        if isinstance(result, dict):
            return _RerankOutput.model_validate(result), None
        raise RerankError("could not parse rerank output")


def build_reranker(settings: Any) -> Reranker | None:
    """Return an LLM re-ranker when configured and enabled, else ``None``."""
    if not getattr(settings, "rerank_enabled", True):
        return None
    llm = getattr(settings, "llm", None)
    if llm is not None and llm.configured:
        return LLMReranker(llm)
    return None


def blend_scores(lexical: float, llm_relevance: float, *, weight: float = 0.6) -> float:
    """Blend the LLM relevance into the lexical score (LLM gets ``weight``)."""
    blended = weight * llm_relevance + (1.0 - weight) * lexical
    return max(0.0, min(1.0, blended))


__all__ = [
    "LLMReranker",
    "RerankError",
    "Reranker",
    "blend_scores",
    "build_reranker",
]
