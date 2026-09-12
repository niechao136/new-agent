"""HTTP plumbing shared by the concrete news sources."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from ..models import ErrorCode


@runtime_checkable
class HttpResponse(Protocol):
    """Structural view of an HTTP response, as used by the news sources.

    Both :class:`httpx.Response` and the stubs in the test-suite satisfy it, so
    sources can be exercised without any network access.
    """

    @property
    def content(self) -> bytes: ...

    @property
    def status_code(self) -> int: ...

    def json(self) -> Any: ...


@runtime_checkable
class HttpClientProtocol(Protocol):
    """Structural interface for the HTTP client injected into every source."""

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse: ...


class SourceError(Exception):
    """Normalised source level failure.

    Every source raises this so the calling node can build a machine readable
    :class:`~news_agent.models.ErrorInfo` and decide whether to retry.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class HttpClient:
    """Thin async wrapper adding User-Agent rotation, timeouts and error mapping."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._user_agents = list(settings.user_agents) or ["news-agent/0.1"]
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(settings.fetch_timeout_s),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        self._index = random.randrange(len(self._user_agents))

    def _next_user_agent(self) -> str:
        self._index = (self._index + 1) % len(self._user_agents)
        return self._user_agents[self._index]

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        merged_headers = {"User-Agent": self._next_user_agent(), "Accept": "*/*"}
        if headers:
            merged_headers.update(headers)
        try:
            request_kwargs: dict = {
                "headers": merged_headers,
                "timeout": timeout or self._settings.fetch_timeout_s,
            }
            # 注意：httpx 在传入 params 时会「替换」掉 URL 原有的查询串，
            # 传空 dict 会把内联在 URL 里的 ?q=... 清空，导致目标返回 404 / HTML。
            # 因此仅在 params 非空时才传给 httpx，保留 URL 自带查询。
            if params:
                request_kwargs["params"] = params
            response = await self._client.get(url, **request_kwargs)
        except httpx.TimeoutException as exc:
            raise SourceError(
                ErrorCode.FETCH_TIMEOUT, f"request to {url} timed out", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise SourceError(
                ErrorCode.SOURCE_UNAVAILABLE,
                f"transport error for {url}: {exc}",
                retryable=True,
            ) from exc

        if response.status_code == 429:
            raise SourceError(
                ErrorCode.RATE_LIMITED,
                f"{url} rate limited the request",
                retryable=True,
                status_code=429,
            )
        if response.status_code >= 500:
            raise SourceError(
                ErrorCode.SOURCE_UNAVAILABLE,
                f"{url} returned HTTP {response.status_code}",
                retryable=True,
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise SourceError(
                ErrorCode.SOURCE_UNAVAILABLE,
                f"{url} returned HTTP {response.status_code}",
                retryable=False,
                status_code=response.status_code,
            )
        return response

    async def aclose(self) -> None:
        await self._client.aclose()


async def gather_limited(
    coroutines: list[Any], *, concurrency: int
) -> list[Any]:
    """``asyncio.gather`` honouring a maximum concurrency."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_guarded(coro) for coro in coroutines))
