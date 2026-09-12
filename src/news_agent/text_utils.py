"""Small text helpers used across the agent (tokenisation, HTML stripping, ...).

The tokeniser is CJK aware: Chinese text is split into character bigrams which
is a decent proxy for words without pulling in a segmentation dependency, while
latin text is split on word boundaries and stop-word filtered.
"""

from __future__ import annotations

import html
import json
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LATIN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'’\-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\.])\s*|\n+")
_CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*(.*?)\s*```\s*$", re.DOTALL)
_TEXT_WRAPPER_KEYS = ("synthesis", "summary", "text", "content", "result", "answer", "output")

_TRACKING_PREFIXES = ("utm_", "spm", "from", "ref", "referrer", "cmpid", "ncid", "fbclid", "gclid")

STOPWORDS: set[str] = {
    # english
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "by", "as", "it", "its", "this",
    "that", "these", "those", "from", "after", "before", "over", "under", "vs",
    "how", "why", "what", "when", "who", "will", "has", "have", "had", "not",
    "but", "than", "then", "into", "out", "up", "down", "new", "says", "said",
    # chinese
    "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "与", "及", "或", "为", "对", "被", "把", "从", "等", "中",
    "后", "前", "报道", "消息", "记者", "表示", "认为", "今日", "昨天", "今天",
    "最新", "相关", "全文", "阅读", "来源", "编辑", "责编", "原标题",
}


def strip_html(text: str | None) -> str:
    """Remove markup and unescape entities."""
    if not text:
        return ""
    cleaned = _SCRIPT_RE.sub(" ", text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", str(text)).strip()


def clean_llm_text(text: str | None) -> str:
    """Normalise raw LLM prose: strip code fences and unwrap JSON envelopes.

    线上真实案例：摘要请求返回了

        ```json
        {"synthesis": "近期科技新闻涵盖了……"}
        ```

    直接展示给调用方就成了原始 JSON，因此这里统一解包。
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    fenced = _CODE_FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return raw
        if isinstance(parsed, dict):
            for key in _TEXT_WRAPPER_KEYS:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            parts = [
                value.strip()
                for value in parsed.values()
                if isinstance(value, str) and value.strip()
            ]
            if parts:
                return "\n".join(parts)
        elif isinstance(parsed, list):
            parts = [str(value).strip() for value in parsed if str(value).strip()]
            if parts:
                return "\n".join(parts)
    return raw


def truncate(text: str | None, limit: int, suffix: str = "…") -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def tokenize(text: str | None, *, cjk_bigram: bool = True) -> list[str]:
    """Return a stop-word filtered bag of tokens."""
    text = clean_text(text)
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []

    for word in _LATIN_RE.findall(lowered):
        if len(word) < 2 or word in STOPWORDS:
            continue
        tokens.append(word)

    for run in _CJK_RE.findall(text):
        if run in STOPWORDS:
            continue
        if len(run) == 1:
            if run not in STOPWORDS:
                tokens.append(run)
            continue
        if cjk_bigram:
            for index in range(len(run) - 1):
                gram = run[index : index + 2]
                if gram not in STOPWORDS:
                    tokens.append(gram)
        else:
            tokens.append(run)
            for char in run:
                if char not in STOPWORDS:
                    tokens.append(char)
    return tokens


def token_set(text: str | None) -> set[str]:
    return set(tokenize(text))


def normalize_for_compare(text: str | None) -> str:
    """Lower-cased, punctuation/space-free form used for fuzzy comparison."""
    if not text:
        return ""
    lowered = str(text).lower()
    return re.sub(r"[\s\u3000\W_]+", "", lowered, flags=re.UNICODE)


def split_sentences(text: str | None) -> list[str]:
    if not text:
        return []
    chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(clean_text(text))]
    return [chunk for chunk in chunks if len(chunk) > 4]


def normalize_url(url: str | None) -> str:
    """Canonical URL form used for de-duplication and article ids."""
    if not url:
        return ""
    raw = str(url).strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:  # pragma: no cover - defensive
        return raw.lower()
    if not parsed.scheme and not parsed.netloc:
        return normalize_for_compare(raw)

    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif netloc.endswith(":443"):
        netloc = netloc[:-4]

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            netloc,
            path,
            "",
            urlencode(sorted(query_pairs)),
            "",
        )
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
