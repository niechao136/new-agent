"""Inbound authentication middleware for the HTTP / A2A endpoints.

Implements a static API-key scheme (see :class:`~news_agent.config.AuthSettings`)
as a pure ASGI middleware so it also covers the protocol routes that the
``a2a-sdk`` registers dynamically (JSON-RPC ``POST /``, REST ``/message:send``,
SSE streams, …), which are invisible to router-level FastAPI dependencies.

Accepted credentials (first match wins):

* ``Authorization: Bearer <key>``
* ``<api_key_header>: <key>`` (``X-API-Key`` by default)

Always public (no key required):

* ``/healthz`` and ``/readyz`` -- container/orchestrator health probes
* ``/.well-known/agent-card.json`` and ``/.well-known/agent.json`` --
  A2A discovery; gateways must be able to read the card (which itself
  advertises the security requirements) before authenticating.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from ..config import AuthSettings
from ..runtime import get_logger

log = get_logger("a2a.auth")

#: Paths that never require authentication.
PUBLIC_PATHS = frozenset({"/healthz", "/readyz"})
#: Path prefixes that never require authentication (A2A discovery).
PUBLIC_PREFIXES = ("/.well-known/",)

_AUTH_SCHEME = "bearer "

ASGIApp = Callable[..., Awaitable[None]]


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def extract_api_key(headers: Headers, api_key_header: str) -> str | None:
    """Extract the presented API key from request headers, if any."""
    raw = headers.get(api_key_header)
    if raw:
        return raw.strip()
    authorization = headers.get("authorization")
    if authorization and authorization.lower().startswith(_AUTH_SCHEME):
        return authorization[len(_AUTH_SCHEME) :].strip()
    return None


def key_matches(presented: str, valid_keys: list[str]) -> bool:
    """Constant-time check of the presented key against the configured ones."""
    presented_bytes = presented.encode("utf-8")
    for valid in valid_keys:
        if hmac.compare_digest(presented_bytes, valid.encode("utf-8")):
            return True
    return False


def unauthorized_response(message: str) -> JSONResponse:
    return JSONResponse(
        {"detail": message},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


class AuthMiddleware:
    """Pure ASGI middleware enforcing :class:`AuthSettings` on HTTP requests.

    Disabled when ``auth.enabled`` is false or no API keys are configured, so
    existing deployments keep working without changes.
    """

    def __init__(self, app: ASGIApp, auth: AuthSettings) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http" or not self.auth.configured:
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if _is_public(path):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        presented = extract_api_key(headers, self.auth.api_key_header)
        if presented is None:
            log.info("auth: rejected %s %s (missing credentials)", scope.get("method"), path)
            await unauthorized_response(
                "Missing API key: pass 'Authorization: Bearer <key>' "
                f"or '{self.auth.api_key_header}: <key>'."
            )(scope, receive, send)
            return
        if not key_matches(presented, self.auth.api_keys):
            log.info("auth: rejected %s %s (invalid key)", scope.get("method"), path)
            await unauthorized_response("Invalid API key.")(scope, receive, send)
            return

        await self.app(scope, receive, send)


__all__ = [
    "AuthMiddleware",
    "PUBLIC_PATHS",
    "PUBLIC_PREFIXES",
    "extract_api_key",
    "key_matches",
]
