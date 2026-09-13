"""FastAPI application wiring the official ``a2a-sdk`` server to the news agent.

The protocol surface is **entirely** provided by the SDK:

===============================  =========================================
Endpoint                         Provided by
===============================  =========================================
``/.well-known/agent-card.json``  ``sdk.create_agent_card_routes``
``/.well-known/agent.json``       ``sdk.create_agent_card_routes`` (0.3 path)
``POST /``                        ``sdk.create_jsonrpc_routes`` (v1.0 + v0.3)
``/message:send`` …               ``sdk.create_rest_routes``
``/tasks/{id}``, ``/tasks`` …     ``sdk.create_rest_routes``
===============================  =========================================

This module only adds operational endpoints (``/healthz``, ``/metrics``,
``/skills`` and a discovery index) and manages the ``NewsAgent`` lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from a2a.server.events import InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2
from a2a.utils.constants import DEFAULT_RPC_URL
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import Settings, load_settings
from ..graph.agent import NewsAgent
from ..runtime import Metrics, configure_logging, get_logger
from .auth import AuthMiddleware
from .card import (
    AGENT_CARD_LEGACY_PATH,
    AGENT_CARD_WELL_KNOWN_PATH,
    OUTPUT_SCHEMA,
    build_agent_card,
    skill_descriptions,
    skill_ids,
)
from .executor import NewsAgentExecutor

log = get_logger("a2a.server")


def create_app(
    settings: Settings | None = None,
    *,
    agent: NewsAgent | None = None,
    metrics: Metrics | None = None,
    task_store: InMemoryTaskStore | None = None,
    executor: NewsAgentExecutor | None = None,
    enable_v0_3_compat: bool = True,
) -> FastAPI:
    """Build the news agent A2A application.

    Args:
        settings: runtime configuration (loaded from the environment if omitted).
        agent: pre-built :class:`NewsAgent` (tests / embedding).  When omitted the
            agent is created from ``settings`` during the ASGI lifespan.
        metrics: shared metrics registry.
        task_store: override the SDK task store (defaults to in-memory).
        executor: override the SDK agent executor (mainly for tests).
        enable_v0_3_compat: also accept A2A 0.3 method names (``message/send``,
            ``tasks/get``, …) next to the 1.0 names (``SendMessage``, …).
    """
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    metrics = metrics or (agent.metrics if agent is not None else Metrics())
    owns_agent = agent is None

    holder: dict[str, NewsAgent | None] = {"agent": agent}
    init_lock = asyncio.Lock()

    async def ensure_agent() -> NewsAgent:
        """Create the agent on demand (idempotent, works without ASGI lifespan)."""
        if holder["agent"] is None:
            async with init_lock:
                if holder["agent"] is None:
                    holder["agent"] = await NewsAgent.create(settings, metrics=metrics)
                    app.state.agent = holder["agent"]
        assert holder["agent"] is not None
        return holder["agent"]

    card: a2a_pb2.AgentCard = build_agent_card(settings)
    store = task_store or InMemoryTaskStore()
    news_executor = executor or NewsAgentExecutor(ensure_agent, settings, metrics=metrics)

    # The SDK handler owns the state machine, the task store and the event
    # queues; cancellation, resubscription and SSE all come for free.
    request_handler = DefaultRequestHandler(
        agent_executor=news_executor,
        task_store=store,
        agent_card=card,
        queue_manager=InMemoryQueueManager(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await ensure_agent()
        log.info(
            "news-agent ready: %s (sources=%s, skills=%s)",
            settings.resolve_agent_url(),
            holder["agent"].registry.names() if holder["agent"] else [],
            skill_ids(),
        )
        try:
            yield
        finally:
            await news_executor.aclose()
            await request_handler.aclose()
            if owns_agent and holder["agent"] is not None:
                await holder["agent"].aclose()

    app = FastAPI(
        title=settings.agent_name,
        description=settings.agent_description,
        version=settings.agent_version,
        lifespan=lifespan,
    )
    # Inbound authentication (API key). Registered as ASGI middleware so it
    # also covers the SDK-generated protocol routes below. When disabled the
    # middleware passes everything through.
    app.add_middleware(AuthMiddleware, auth=settings.auth)
    app.state.settings = settings
    app.state.agent = agent
    app.state.agent_card = card
    app.state.metrics = metrics
    app.state.task_store = store
    app.state.news_executor = news_executor
    app.state.request_handler = request_handler

    # ------------------------------------------------------------------
    # operational endpoints -- registered *before* the A2A REST routes so the
    # SDK's `/{tenant}` mount cannot shadow them.
    # ------------------------------------------------------------------
    @app.get("/", tags=["meta"])
    async def index() -> dict[str, Any]:
        return {
            "name": settings.agent_name,
            "version": settings.agent_version,
            "protocolVersion": card.supported_interfaces[0].protocol_version,
            "description": settings.agent_description,
            "skills": skill_ids(),
            "agentCard": AGENT_CARD_WELL_KNOWN_PATH,
            "agentCardLegacy": AGENT_CARD_LEGACY_PATH,
            "jsonrpc": DEFAULT_RPC_URL,
            "transports": [
                interface.protocol_binding for interface in card.supported_interfaces
            ],
            "docs": "/docs",
        }

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict[str, Any]:
        """Liveness: never builds the agent, only reports what is known."""
        current: NewsAgent | None = holder["agent"]
        return {
            "status": "ok",
            "agent": settings.agent_name,
            "version": settings.agent_version,
            "agentReady": current is not None,
            "sources": current.registry.names() if current is not None else [],
            "skills": skill_ids(),
            "inFlight": news_executor.in_flight(),
        }

    @app.get("/metrics", tags=["meta"])
    async def read_metrics() -> dict[str, Any]:
        current: NewsAgent | None = holder["agent"]
        payload: dict[str, Any] = {
            "metrics": metrics.snapshot(),
            "tasks": {
                "backend": type(store).__name__,
                "inFlight": news_executor.in_flight(),
            },
        }
        if current is not None and current.cache is not None:
            payload["cache"] = await current.cache.stats()
        return payload

    @app.get("/skills", tags=["meta"])
    async def skills() -> dict[str, Any]:
        return {"skills": skill_descriptions(), "resultSchema": OUTPUT_SCHEMA}

    @app.get("/readyz", tags=["meta"])
    async def readyz() -> JSONResponse:
        """Readiness: builds the agent on demand so the first real task is fast."""
        try:
            await ensure_agent()
        except Exception as exc:  # noqa: BLE001 - report instead of a 500
            log.exception("agent initialisation failed")
            return JSONResponse({"ready": False, "error": str(exc)}, status_code=503)
        return JSONResponse({"ready": True}, status_code=200)

    # ------------------------------------------------------------------
    # A2A protocol surface (SDK provided)
    # ------------------------------------------------------------------
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=[
            *create_agent_card_routes(card, card_url=AGENT_CARD_WELL_KNOWN_PATH),
            # legacy 0.3 discovery path, kept for gateways that still use it
            *create_agent_card_routes(card, card_url=AGENT_CARD_LEGACY_PATH),
        ],
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler,
            rpc_url=DEFAULT_RPC_URL,
            enable_v0_3_compat=enable_v0_3_compat,
        ),
        rest_routes=create_rest_routes(
            request_handler, enable_v0_3_compat=enable_v0_3_compat
        ),
    )
    return app


def app_factory() -> FastAPI:  # pragma: no cover - uvicorn --factory entrypoint
    return create_app()


__all__ = ["create_app", "app_factory"]
