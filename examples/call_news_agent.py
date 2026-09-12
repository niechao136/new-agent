"""End-to-end A2A call example built on the official ``a2a-sdk`` client.

Shows how another agent of the platform (the default agent or a custom one)
talks to the news agent:

1. resolve the agent card (``/.well-known/agent-card.json``)
2. send a ``SendMessage`` request and consume the streamed ``TaskUpdate``s
   (task → working status with ``stage`` metadata → artifact → completed)
3. read the ``news-result`` artifact and print the structured result
4. resubscribe / cancel / list tasks through the same client

Run it against a locally started agent:

    news-agent serve --port 8080                    # terminal 1
    python examples/call_news_agent.py              # terminal 2
    python examples/call_news_agent.py --skill analyze_trend --query "固态电池" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.protobuf import json_format  # noqa: E402

from news_agent.a2a.client import (  # noqa: E402
    NewsA2AClient,
    result_from_task,
    state_name,
)
from news_agent.models import SkillMode  # noqa: E402


def show_card(card: Any) -> None:
    print(f"agent       : {card.name} {card.version}")
    print(f"description : {card.description}")
    for interface in card.supported_interfaces:
        print(f"interface   : {interface.protocol_binding} v{interface.protocol_version} -> {interface.url}")
    print(f"streaming   : {card.capabilities.streaming}")
    for skill in card.skills:
        print(f"  - {skill.id:15s} {skill.name}")


def show_result(task: Any) -> None:
    payload = result_from_task(task) or {}
    counts = payload.get("counts", {})
    print(f"task        : {task.id} -> {state_name(task.status.state)}")
    print(f"mode        : {payload.get('mode')}")
    print(
        "counts      : "
        f"fetched={counts.get('fetched')} deduped={counts.get('after_dedup')} "
        f"selected={counts.get('selected')} analyzed={counts.get('analyzed')}"
    )
    print(f"degraded    : {payload.get('degraded')}")
    for warning in payload.get("warnings", []):
        print(f"  ! {warning}")
    for error in payload.get("errors", []):
        print(f"  x [{error['code']}] {error['message']}")

    if payload.get("summary"):
        print("\n--- summary ---")
        print(payload["summary"])

    for trend in payload.get("trends", []):
        print(
            f"  trend: {trend['topic']} x{trend['mentions']} "
            f"({trend['sentiment']}, {trend['average_sentiment']:+.2f})"
        )

    print("\n--- articles ---")
    for article in payload.get("articles", []):
        published = (article.get("published_at") or "n/a")[:10]
        print(
            f"  [{article.get('sentiment', 'n/a'):>8}] {published} "
            f"{article.get('title')} ({article.get('source')})"
        )
        print(f"             {article.get('url')}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Call the news agent over A2A")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--query", default="人形机器人")
    parser.add_argument(
        "--skill", default="summarize_news", choices=[mode.value for mode in SkillMode]
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--since", default=None, help="ISO 8601, e.g. 2026-09-01")
    parser.add_argument("--quiet", action="store_true", help="do not print progress")
    parser.add_argument("--json", action="store_true", help="dump the raw task JSON")
    parser.add_argument("--list-tasks", action="store_true", help="also list agent tasks")
    args = parser.parse_args()

    async with NewsA2AClient(args.base_url, streaming=True, timeout=300) as client:
        card = await client.connect()
        show_card(card)
        print()

        print(">>> SendMessage (streaming)")
        task = None
        seen_stage: str | None = None
        async for update in client.send(
            args.query,
            skill=args.skill,
            limit=args.limit,
            language=args.language,
            since=args.since,
        ):
            if update.kind == "status" and not args.quiet:
                marker = "·" if update.stage != seen_stage else " "
                print(f"  {marker} {update.message}")
                seen_stage = update.stage
            elif update.kind == "artifact" and not args.quiet:
                print("  · artifact received")
            elif update.kind == "task":
                task = update.task

        if task is None:
            print("the agent did not return a task", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(json_format.MessageToDict(task), ensure_ascii=False, indent=2))
        else:
            print()
            show_result(task)

        # --- other client capabilities come straight from the SDK ----------
        fetched = await client.get_task(task.id)
        assert fetched.id == task.id
        if args.list_tasks:
            listing = await client.list_tasks(page_size=10)
            print(f"\nagent tasks : {len(listing.tasks)}")

        return 0 if state_name(task.status.state) == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
