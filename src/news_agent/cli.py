"""Command line interface: ``news-agent serve|card|run|call|skills``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from google.protobuf import json_format

from . import __version__
from .config import Settings
from .models import SkillMode
from .runtime import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-agent",
        description="A2A news agent (LangGraph + FastAPI)",
    )
    parser.add_argument("--version", action="version", version=f"news-agent {__version__}")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the A2A HTTP server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")

    sub.add_parser("card", help="print the A2A agent card as JSON")
    sub.add_parser("skills", help="list the advertised skills")
    sub.add_parser("sources", help="list configured news sources")

    run = sub.add_parser("run", help="run a skill locally (no HTTP server)")
    run.add_argument("query", help="关键词 / 话题")
    run.add_argument(
        "--skill",
        default=None,
        choices=[mode.value for mode in SkillMode] + ["fetch", "summarize", "trend"],
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--language", default=None)
    run.add_argument("--since", default=None, help="ISO 8601，例如 2026-09-01")
    run.add_argument("--sources", default=None, help="逗号分隔的源名称")
    run.add_argument("--threshold", type=float, default=None)
    run.add_argument("--mock", action="store_true", help="使用内置 mock 源（离线）")
    run.add_argument("--no-llm", action="store_true", help="禁用 LLM，走启发式分析")
    run.add_argument("--no-cache", action="store_true", help="禁用 SQLite 缓存")
    run.add_argument("--json", action="store_true", help="输出完整 JSON")

    call = sub.add_parser("call", help="call a running agent over A2A")
    call.add_argument("query")
    call.add_argument("--base-url", default="http://localhost:8080")
    call.add_argument("--skill", default="summarize_news")
    call.add_argument("--limit", type=int, default=None)
    call.add_argument("--language", default=None)
    call.add_argument("--stream", action="store_true", help="使用 SSE 流式接收进度")
    call.add_argument("--json", action="store_true")

    task = sub.add_parser("task", help="inspect a task on a running agent")
    task.add_argument("task_id")
    task.add_argument("--base-url", default="http://localhost:8080")
    task.add_argument("--cancel", action="store_true")

    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    env: dict[str, str] = {}
    if getattr(args, "mock", False):
        env["NEWS_AGENT_USE_MOCK"] = "1"
    if getattr(args, "no_llm", False):
        env["NEWS_AGENT_LLM_ENABLED"] = "0"
    if getattr(args, "no_cache", False):
        env["NEWS_AGENT_CACHE_ENABLED"] = "0"
    if getattr(args, "log_level", None):
        env["NEWS_AGENT_LOG_LEVEL"] = args.log_level
    return Settings.load(env or None)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings_from_args(args)
    configure_logging(settings.log_level)
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"news-agent {__version__} serving on http://{host}:{port}")
    print(f"agent card: http://{host}:{port}/.well-known/agent.json")
    if args.reload:
        uvicorn.run(
            "news_agent.a2a.server:app_factory",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
    else:
        from .a2a.server import create_app

        uvicorn.run(create_app(settings), host=host, port=port)
    return 0


def _cmd_card(args: argparse.Namespace) -> int:
    from .a2a.card import agent_card_dict

    print(json.dumps(agent_card_dict(_settings_from_args(args)), ensure_ascii=False, indent=2))
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    from .a2a.card import skill_descriptions

    for skill in skill_descriptions():
        print(f"- {skill['id']}: {skill['name']}")
        print(f"    {skill['description']}")
        print(f"    examples: {'; '.join(skill['examples'])}")
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    if not settings.sources:
        print("no sources configured")
        return 0
    for source in settings.sources:
        state = "enabled" if source.enabled else "disabled"
        detail = ",".join(source.feeds) if source.feeds else source.base_url or ""
        print(f"- {source.name} [{source.type}] {state} {detail}")
    print(f"llm: {'on' if settings.llm.configured else 'off (heuristic mode)'}")
    return 0


async def _run_local(args: argparse.Namespace) -> int:
    from .graph.agent import NewsAgent
    from .models import SkillRequest

    settings = _settings_from_args(args)
    configure_logging(settings.log_level)
    request = SkillRequest(
        skill=SkillMode.coerce(args.skill or settings.default_mode),
        query=args.query,
        since=args.since,
        limit=args.limit or settings.default_limit,
        language=args.language or settings.default_language,
        sources=[item.strip() for item in args.sources.split(",")] if args.sources else None,
        threshold=args.threshold,
    )
    agent = await NewsAgent.create(settings)
    try:
        result = await agent.run(request)
    finally:
        await agent.aclose()

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0 if result.articles else 1

    print(f"skill      : {result.mode.value}")
    print(f"query      : {result.query}")
    print(f"articles   : {result.counts.get('analyzed', 0)} (fetched {result.counts.get('fetched', 0)}, "
          f"duplicates removed {result.counts.get('duplicates_removed', 0)})")
    print(f"duration   : {result.duration_ms} ms   degraded: {result.degraded}")
    if result.warnings:
        print("warnings   :")
        for warning in result.warnings:
            print(f"  ! {warning}")
    if result.errors:
        print("errors     :")
        for error in result.errors:
            print(f"  x [{error.code.value}] {error.message}")
    if result.summary:
        print("\n--- summary ---")
        print(result.summary)
    if result.trends:
        print("\n--- trends ---")
        for trend in result.trends:
            print(
                f"  · {trend.topic} x{trend.mentions} "
                f"({trend.sentiment}, {trend.average_sentiment:+.2f})"
            )
    if result.articles:
        print("\n--- articles ---")
        for article in result.articles:
            published = article.published_at.date().isoformat() if article.published_at else "n/a"
            print(
                f"  [{article.sentiment:>8}] {published} {article.title} "
                f"({article.source}, r={article.relevance:.2f})"
            )
            print(f"             {article.url}")
    return 0 if result.articles else 1


async def _run_call(args: argparse.Namespace) -> int:
    from .a2a.client import NewsA2AClient, result_from_task, state_name

    settings = _settings_from_args(args)
    async with NewsA2AClient(
        args.base_url, streaming=True, timeout=max(60.0, settings.task_timeout_s)
    ) as client:
        card = await client.connect()
        print(
            f"connected to {card.name} {card.version} "
            f"(A2A {card.supported_interfaces[0].protocol_version}, "
            f"skills: {', '.join(skill.id for skill in card.skills)})"
        )
        kwargs: dict[str, Any] = {
            "skill": args.skill,
            "limit": args.limit,
            "language": args.language,
        }
        task = None
        last_stage = None
        async for update in client.send(args.query, **kwargs):
            if update.kind == "status":
                if args.stream:
                    marker = "·" if update.stage != last_stage else " "
                    print(f"  {marker} {update.message}")
                last_stage = update.stage
            elif update.kind == "task":
                current = update.task
                if current is None:  # pragma: no cover - defensive
                    continue
                if args.stream and update.is_terminal:
                    print(f"  → task {state_name(current.status.state)}")
                task = current

    if task is None:
        print("the agent did not return a task", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(json_format.MessageToDict(task), ensure_ascii=False, indent=2))
        return 0

    payload = result_from_task(task) or {}
    state = state_name(task.status.state)
    print(f"task       : {task.id} -> {state}")
    print(f"counts     : {payload.get('counts')}")
    print(f"degraded   : {payload.get('degraded')}")
    if payload.get("warnings"):
        for warning in payload["warnings"]:
            print(f"  ! {warning}")
    if payload.get("errors"):
        for error in payload["errors"]:
            print(f"  x [{error.get('code')}] {error.get('message')}")
    if payload.get("summary"):
        print("\n--- summary ---")
        print(payload["summary"])
    for trend in payload.get("trends") or []:
        print(
            f"  trend: {trend['topic']} x{trend['mentions']} "
            f"({trend['sentiment']}, {trend['average_sentiment']:+.2f})"
        )
    return 0 if state == "completed" else 1


async def _run_task(args: argparse.Namespace) -> int:
    from .a2a.client import NewsA2AClient

    async with NewsA2AClient(args.base_url, timeout=120) as client:
        if args.cancel:
            task = await client.cancel_task(args.task_id)
        else:
            task = await client.get_task(args.task_id)
    print(
        json.dumps(json_format.MessageToDict(task), ensure_ascii=False, indent=2)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "card":
        return _cmd_card(args)
    if args.command == "skills":
        return _cmd_skills(args)
    if args.command == "sources":
        return _cmd_sources(args)
    if args.command == "run":
        return asyncio.run(_run_local(args))
    if args.command == "call":
        return asyncio.run(_run_call(args))
    if args.command == "task":
        return asyncio.run(_run_task(args))
    parser.error(f"unknown command: {args.command}")  # pragma: no cover - argparse exits


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
