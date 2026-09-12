"""SQLite backed caching + incremental article history (TODO item 8).

Design notes
------------
* ``article_cache``  -- request scoped result of the *fetch* stage, keyed by
  query/time-window/sources.  Avoids hammering upstream feeds for the same
  question within ``cache_ttl_s``.
* ``result_cache``   -- full :class:`NewsResult` payloads (optional, used by the
  executor to answer repeated identical tasks instantly).
* ``article_history``-- every article ever seen.  Storing the first/last seen
  timestamp is the basis for incremental updates ("what is new since the last
  run?").

The module deliberately uses the stdlib ``sqlite3`` driver executed through
``asyncio.to_thread`` -- zero extra dependency and no event-loop blocking.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import NewsResult, RawArticle, utcnow
from .runtime import get_logger

log = get_logger("cache")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS article_cache (
    cache_key  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS result_cache (
    cache_key  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS article_history (
    id            TEXT PRIMARY KEY,
    url           TEXT,
    title         TEXT,
    source        TEXT,
    published_at  TEXT,
    first_seen_at REAL NOT NULL,
    last_seen_at  REAL NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_article_history_published
    ON article_history(published_at);
CREATE INDEX IF NOT EXISTS idx_article_history_source
    ON article_history(source);
"""


class SqliteCache:
    """Small async-friendly wrapper around a single SQLite file."""

    def __init__(self, path: str | Path, *, ttl_s: int = 900, enabled: bool = True) -> None:
        self.path = Path(path)
        self.ttl_s = ttl_s
        self.enabled = enabled
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._ready = False
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.enabled and self._ready

    async def init(self) -> None:
        if not self.enabled or self._ready:
            return
        try:
            await asyncio.to_thread(self._connect)
            self._ready = True
        except Exception as exc:  # pragma: no cover - disk issues
            self.last_error = repr(exc)
            log.warning("cache disabled, initialisation failed: %r", exc)
            self._ready = False

    def _connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(_SCHEMA)
        connection.commit()
        self._connection = connection

    async def aclose(self) -> None:
        if self._connection is not None:
            await asyncio.to_thread(self._connection.close)
            self._connection = None
        self._ready = False

    # ------------------------------------------------------------------
    # low level helpers
    # ------------------------------------------------------------------
    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        connection = self._connection
        if connection is None:
            return
        with self._lock:
            connection.execute(sql, params)
            connection.commit()

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        connection = self._connection
        if connection is None:
            return []
        with self._lock:
            cursor = connection.execute(sql, params)
            return cursor.fetchall()

    async def _safe(self, func, *args):
        """Run a blocking cache operation, swallowing (and logging) failures."""
        if not self.available:
            return None
        try:
            return await asyncio.to_thread(func, *args)
        except Exception as exc:  # pragma: no cover - cache must never break a run
            self.last_error = repr(exc)
            log.warning("cache operation failed: %r", exc)
            return None

    # ------------------------------------------------------------------
    # fetch stage cache
    # ------------------------------------------------------------------
    async def get_articles(self, cache_key: str) -> list[RawArticle] | None:
        rows = await self._safe(self._get_articles_sync, cache_key)
        if not rows:
            return None
        try:
            return [RawArticle.model_validate(item) for item in json.loads(rows[0][0])]
        except Exception:  # pragma: no cover - schema drift
            return None

    def _get_articles_sync(self, cache_key: str) -> list[tuple[str]]:
        cutoff = time.time() - self.ttl_s
        rows = self._query(
            "SELECT payload FROM article_cache WHERE cache_key = ? AND created_at >= ?",
            (cache_key, cutoff),
        )
        if not rows:
            self._execute("DELETE FROM article_cache WHERE cache_key = ?", (cache_key,))
        return rows

    async def set_articles(self, cache_key: str, articles: Iterable[RawArticle]) -> None:
        payload = json.dumps(
            [article.model_dump(mode="json") for article in articles], ensure_ascii=False
        )
        await self._safe(self._set_articles_sync, cache_key, payload)

    def _set_articles_sync(self, cache_key: str, payload: str) -> None:
        self._execute(
            "INSERT OR REPLACE INTO article_cache(cache_key, payload, created_at) "
            "VALUES (?, ?, ?)",
            (cache_key, payload, time.time()),
        )

    # ------------------------------------------------------------------
    # full result cache
    # ------------------------------------------------------------------
    async def get_result(self, cache_key: str) -> NewsResult | None:
        rows = await self._safe(self._get_result_sync, cache_key)
        if not rows:
            return None
        try:
            return NewsResult.model_validate(json.loads(rows[0][0]))
        except Exception:  # pragma: no cover - schema drift
            return None

    def _get_result_sync(self, cache_key: str) -> list[tuple[str]]:
        cutoff = time.time() - self.ttl_s
        return self._query(
            "SELECT payload FROM result_cache WHERE cache_key = ? AND created_at >= ?",
            (cache_key, cutoff),
        )

    async def set_result(self, cache_key: str, result: NewsResult) -> None:
        payload = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        await self._safe(self._set_result_sync, cache_key, payload)

    def _set_result_sync(self, cache_key: str, payload: str) -> None:
        self._execute(
            "INSERT OR REPLACE INTO result_cache(cache_key, payload, created_at) "
            "VALUES (?, ?, ?)",
            (cache_key, payload, time.time()),
        )

    # ------------------------------------------------------------------
    # incremental history
    # ------------------------------------------------------------------
    async def upsert_history(self, articles: Iterable[RawArticle]) -> list[str]:
        """Persist articles and return the ids that were never seen before."""
        payloads = [
            (
                article.id,
                article.url,
                article.title,
                article.source,
                article.published_at.isoformat() if article.published_at else None,
                json.dumps(article.model_dump(mode="json"), ensure_ascii=False),
            )
            for article in articles
        ]
        if not payloads:
            return []
        new_ids = await self._safe(self._upsert_history_sync, payloads)
        return new_ids or []

    def _upsert_history_sync(
        self, payloads: list[tuple[str, str, str, str, str | None, str]]
    ) -> list[str]:
        now = time.time()
        new_ids: list[str] = []
        connection = self._connection
        if connection is None:
            return []
        with self._lock:
            for article_id, url, title, source, published_at, payload in payloads:
                existing = connection.execute(
                    "SELECT 1 FROM article_history WHERE id = ?", (article_id,)
                ).fetchone()
                if existing:
                    connection.execute(
                        "UPDATE article_history SET last_seen_at = ?, payload = ? WHERE id = ?",
                        (now, payload, article_id),
                    )
                else:
                    new_ids.append(article_id)
                    connection.execute(
                        "INSERT INTO article_history"
                        "(id, url, title, source, published_at, first_seen_at, last_seen_at, payload)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (article_id, url, title, source, published_at, now, now, payload),
                    )
            connection.commit()
        return new_ids

    # ------------------------------------------------------------------
    async def stats(self) -> dict[str, int]:
        rows = await self._safe(self._stats_sync)
        if not rows:
            return {"article_cache": 0, "result_cache": 0, "article_history": 0}
        return {
            "article_cache": rows[0][0],
            "result_cache": rows[1][0],
            "article_history": rows[2][0],
        }

    def _stats_sync(self) -> list[tuple[int]]:
        return [
            self._query("SELECT COUNT(*) FROM article_cache")[0],
            self._query("SELECT COUNT(*) FROM result_cache")[0],
            self._query("SELECT COUNT(*) FROM article_history")[0],
        ]

    def purge(self, *, keep_seconds: int | None = None) -> None:  # pragma: no cover
        cutoff = time.time() - (keep_seconds if keep_seconds is not None else self.ttl_s)
        self._execute("DELETE FROM article_cache WHERE created_at < ?", (cutoff,))
        self._execute("DELETE FROM result_cache WHERE created_at < ?", (cutoff,))
        log.info("cache purged at %s", utcnow().isoformat())
