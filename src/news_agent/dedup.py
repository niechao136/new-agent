"""Cross-source de-duplication (TODO item 6).

Three complementary strategies are combined, cheapest first:

1. **exact URL** match on the normalised URL (tracking params stripped);
2. **exact normalised title** match;
3. fuzzy match -- 64 bit SimHash Hamming distance on ``title + summary``
   and/or a ``difflib`` title similarity ratio.

Candidate generation is blocked (SimHash bands + shared tokens) so the fuzzy
step stays cheap even with a few hundred articles.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher

from .models import RawArticle
from .text_utils import normalize_for_compare, normalize_url, tokenize

_BANDS = 4  # 4 x 16 bit bands for simhash blocking


def simhash(text: str, bits: int = 64) -> int:
    """Deterministic 64-bit SimHash of a text (no external dependency)."""
    tokens = tokenize(text)
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for index in range(bits):
            vector[index] += 1 if (value >> index) & 1 else -1
    fingerprint = 0
    for index in range(bits):
        if vector[index] > 0:
            fingerprint |= 1 << index
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _bands(fingerprint: int, bits: int = 64) -> list[tuple[int, int]]:
    width = bits // _BANDS
    return [
        (band, (fingerprint >> (band * width)) & ((1 << width) - 1))
        for band in range(_BANDS)
    ]


def title_similarity(left: str, right: str) -> float:
    left_norm = normalize_for_compare(left)
    right_norm = normalize_for_compare(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    shorter, longer = sorted((left_norm, right_norm), key=len)
    if len(shorter) >= 6 and shorter in longer:
        return 0.95
    return SequenceMatcher(None, left_norm, right_norm).ratio()


class Deduplicator:
    """Collapses articles reporting the same story into one representative."""

    def __init__(self, *, title_ratio: float = 0.86, simhash_distance: int = 3) -> None:
        self.title_ratio = title_ratio
        self.simhash_distance = simhash_distance

    # ------------------------------------------------------------------
    def dedupe(self, articles: Iterable[RawArticle]) -> list[RawArticle]:
        kept: list[RawArticle] = []
        fingerprints: list[int] = []
        normalized_titles: list[str] = []
        simhash_index: dict[tuple[int, int], list[int]] = defaultdict(list)
        token_index: dict[str, list[int]] = defaultdict(list)
        url_index: dict[str, int] = {}
        title_index: dict[str, int] = {}

        for article in articles:
            url_key = normalize_url(article.url) or f"id:{article.id}"
            title_key = normalize_for_compare(article.title)
            fingerprint = simhash(f"{article.title} {article.summary or ''}")
            title_tokens = set(tokenize(article.title))

            match: int | None = None

            # 1) cheap exact lookups -------------------------------------
            if url_key and url_key in url_index:
                match = url_index[url_key]
            elif title_key and title_key in title_index:
                match = title_index[title_key]

            # 2) fuzzy lookup -------------------------------------------
            if match is None:
                candidates: set[int] = set()
                for band in _bands(fingerprint):
                    candidates.update(simhash_index.get(band, ()))
                for token in list(title_tokens)[:8]:
                    bucket = token_index.get(token)
                    if bucket and len(bucket) <= 40:
                        candidates.update(bucket)
                for index in sorted(candidates)[:40]:
                    if (
                        hamming_distance(fingerprint, fingerprints[index])
                        <= self.simhash_distance
                    ):
                        match = index
                        break
                    if title_similarity(article.title, kept[index].title) >= self.title_ratio:
                        match = index
                        break

            if match is not None:
                self._merge(kept[match], article)
                continue

            # 3) register as a new representative -----------------------
            index = len(kept)
            kept.append(article)
            fingerprints.append(fingerprint)
            normalized_titles.append(title_key)
            if url_key:
                url_index[url_key] = index
            if title_key:
                title_index[title_key] = index
            for band in _bands(fingerprint):
                simhash_index[band].append(index)
            for token in list(title_tokens)[:10]:
                token_index[token].append(index)

        return kept

    # ------------------------------------------------------------------
    @staticmethod
    def _merge(target: RawArticle, duplicate: RawArticle) -> None:
        """Merge ``duplicate`` into ``target`` keeping target's identity."""
        if duplicate.source and duplicate.source not in target.duplicate_sources:
            if duplicate.source != target.source:
                target.duplicate_sources.append(duplicate.source)
        if duplicate.id and duplicate.id not in target.duplicate_sources and duplicate.id != target.id:
            target.extra.setdefault("duplicate_ids", [])
            if duplicate.id not in target.extra["duplicate_ids"]:
                target.extra["duplicate_ids"].append(duplicate.id)
        # enrich missing fields with whatever the duplicate knows
        if not target.content and duplicate.content:
            target.content = duplicate.content
        if not target.summary and duplicate.summary:
            target.summary = duplicate.summary
        if target.published_at is None and duplicate.published_at is not None:
            target.published_at = duplicate.published_at
        elif (
            target.published_at is not None
            and duplicate.published_at is not None
            and duplicate.published_at < target.published_at
        ):
            # keep the earliest publication time (a strong freshness signal)
            target.published_at = duplicate.published_at
        if not target.image_url and duplicate.image_url:
            target.image_url = duplicate.image_url
        if len(duplicate.title) > len(target.title) and duplicate.title not in target.title:
            target.extra.setdefault("alt_titles", [])
            if duplicate.title not in target.extra["alt_titles"]:
                target.extra["alt_titles"].append(duplicate.title)
