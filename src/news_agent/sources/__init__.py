"""Pluggable news acquisition backends."""

from __future__ import annotations

from .api import GNewsSource, NewsAPISource
from .base import NewsSource
from .http import (
    HttpClient,
    HttpClientProtocol,
    HttpResponse,
    SourceError,
    gather_limited,
)
from .mock import MockSource
from .registry import SOURCE_TYPES, SourceRegistry
from .rss import RSSSource

__all__ = [
    "NewsSource",
    "RSSSource",
    "NewsAPISource",
    "GNewsSource",
    "MockSource",
    "HttpClient",
    "HttpClientProtocol",
    "HttpResponse",
    "SourceError",
    "SourceRegistry",
    "SOURCE_TYPES",
    "gather_limited",
]
