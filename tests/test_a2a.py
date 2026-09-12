"""A2A layer tests: agent card, task lifecycle, streaming, REST, SDK client.

The protocol surface is provided by ``a2a-sdk``; these tests therefore check our
*wiring* (card contents, skill parsing, executor behaviour, degradation) plus the
end-to-end round trip over the SDK transports (both A2A 1.0 methods and the 0.3
compatibility names).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2
from fastapi.testclient import TestClient

from news_agent.a2a.card import (
    AGENT_CARD_LEGACY_PATH,
    AGENT_CARD_WELL_KNOWN_PATH,
    SKILL_SCHEMA_EXTENSION_URI,
    build_agent_card,
    skill_ids,
)
from news_agent.a2a.client import NewsA2AClient, result_from_task, state_name
from news_agent.a2a.executor import (
    NewsAgentExecutor,
    SkillRequestError,
    parse_skill_request,
)
from news_agent.a2a.server import create_app
from news_agent.config import LLMSettings, Settings, SourceConfig
from news_agent.models import ErrorCode, NewsResult, SkillMode
from news_agent.runtime import RunContext

BASE_URL = "http://test"

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        agent_url=BASE_URL,
        sources=[SourceConfig(name="mock", type="mock")],
        llm=LLMSettings(enabled=False),
        cache_enabled=False,
        cache_path=str(tmp_path / "a2a.sqlite3"),
        log_level="WARNING",
        default_limit=6,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def http(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        yield client


@pytest.fixture
def sync_client(app):
    """Starlette TestClient (runs the ASGI lifespan)."""
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def message_payload(
    query: str = "人形机器人",
    *,
    skill: str = "summarize_news",
    limit: int = 5,
    **extra: Any,
) -> dict[str, Any]:
    data = {"skill": skill, "query": query, "limit": limit, **extra}
    return {
        "messageId": "msg-1",
        "role": "ROLE_USER",
        "parts": [{"data": data}],
    }


def legacy_message_payload(query: str = "人形机器人", **extra: Any) -> dict[str, Any]:
    data = {"skill": "summarize_news", "query": query, "limit": 5, **extra}
    return {
        "messageId": "msg-legacy",
        "role": "user",
        "parts": [{"kind": "data", "data": data}],
    }


async def rpc(
    http: httpx.AsyncClient,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    version: str | None = "1.0",
    request_id: int | str = 1,
) -> dict[str, Any]:
    headers = {"A2A-Version": version} if version else {}
    response = await http.post(
        "/",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def task_of(payload: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPC result may be a Task, a Message or ``{"task": ...}``."""
    result = payload.get("result")
    assert result is not None, payload
    if "task" in result:
        return result["task"]
    return result


# ---------------------------------------------------------------------------
# discovery / card
# ---------------------------------------------------------------------------
async def test_agent_card_on_both_well_known_paths(http):
    card = (await http.get(AGENT_CARD_WELL_KNOWN_PATH)).json()
    assert card["name"] == "news-agent"
    assert [skill["id"] for skill in card["skills"]] == [
        "fetch_news",
        "summarize_news",
        "analyze_trend",
    ]
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert card["supportedInterfaces"][0]["url"] == BASE_URL
    # JSON schemas are published through an extension (A2A 1.0 skills have none)
    extensions = {ext["uri"]: ext for ext in card["capabilities"]["extensions"]}
    params = extensions[SKILL_SCHEMA_EXTENSION_URI]["params"]
    assert params["requestSchema"]["required"] == ["query"]
    assert "articles" in params["resultSchema"]["properties"]

    legacy = await http.get(AGENT_CARD_LEGACY_PATH)
    assert legacy.status_code == 200
    assert legacy.json()["name"] == "news-agent"


def test_build_agent_card_is_a_proto_with_all_skills():
    card = build_agent_card(Settings())
    assert isinstance(card, a2a_pb2.AgentCard)
    assert [skill.id for skill in card.skills] == skill_ids()
    assert card.capabilities.streaming
    assert card.default_output_modes


async def test_meta_endpoints(http):
    index = (await http.get("/")).json()
    assert index["protocolVersion"] == "1.0"
    assert index["skills"] == skill_ids()

    # readiness builds the agent on demand (ASGITransport does not run lifespan)
    assert (await http.get("/readyz")).json() == {"ready": True}

    health = (await http.get("/healthz")).json()
    assert health["status"] == "ok"
    assert health["agentReady"] is True
    assert health["sources"] == ["mock"]

    metrics = (await http.get("/metrics")).json()
    assert "metrics" in metrics
    assert metrics["tasks"]["backend"] == "InMemoryTaskStore"

    skills = (await http.get("/skills")).json()
    assert len(skills["skills"]) == 3
    assert skills["resultSchema"]["title"] == "NewsResult"


def test_lifespan_boots_with_sync_client(sync_client):
    assert sync_client.get("/healthz").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# A2A 1.0 JSON-RPC
# ---------------------------------------------------------------------------
async def test_send_message_returns_completed_task_with_artifact(http):
    payload = await rpc(http, "SendMessage", {"message": message_payload()})
    assert payload.get("error") is None
    task = task_of(payload)

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"], task

    data = task["artifacts"][0]["parts"][0]["data"]
    assert data["query"] == "人形机器人"
    assert data["mode"] == "summarize_news"
    assert data["articles"]
    assert data["counts"]["analyzed"] >= 1
    assert data["counts"]["duplicates_removed"] >= 1
    # the text part mirrors the summary for plain-text callers
    text_part = task["artifacts"][0]["parts"][1]
    assert text_part["text"]


async def test_fetch_news_skill_skips_llm_analysis(http):
    payload = await rpc(
        http, "SendMessage", {"message": message_payload(skill="fetch_news", limit=4)}
    )
    data = task_of(payload)["artifacts"][0]["parts"][0]["data"]
    assert data["mode"] == "fetch_news"
    assert data["summary"] == ""
    assert all(article["sentiment"] == "neutral" for article in data["articles"])
    assert all(not article["entities"] for article in data["articles"])


async def test_analyze_trend_skill_returns_trends(http):
    payload = await rpc(
        http, "SendMessage", {"message": message_payload(skill="analyze_trend", limit=6)}
    )
    data = task_of(payload)["artifacts"][0]["parts"][0]["data"]
    assert data["mode"] == "analyze_trend"
    assert data["trends"]


async def test_text_only_message_and_metadata_query(http):
    # "<skill>: <query>" as plain text
    payload = await rpc(
        http,
        "SendMessage",
        {"message": {"messageId": "m", "role": "ROLE_USER",
                     "parts": [{"text": "fetch_news: 固态电池"}]}},
    )
    data = task_of(payload)["artifacts"][0]["parts"][0]["data"]
    assert data["mode"] == "fetch_news"
    assert data["query"] == "固态电池"

    # query provided through the request metadata
    payload = await rpc(
        http,
        "SendMessage",
        {
            "message": {"messageId": "m2", "role": "ROLE_USER",
                        "parts": [{"text": "no query in text"}]},
            "metadata": {"query": "量子计算", "skill": "fetch_news", "limit": 3},
        },
    )
    data = task_of(payload)["artifacts"][0]["parts"][0]["data"]
    assert data["query"] == "量子计算"


async def test_get_task_list_tasks_and_cancel(http):
    created = task_of(await rpc(http, "SendMessage", {"message": message_payload(limit=3)}))
    task_id = created["id"]

    fetched = await rpc(http, "GetTask", {"id": task_id, "historyLength": 50})
    assert task_of(fetched)["id"] == task_id
    assert task_of(fetched)["status"]["state"] == "TASK_STATE_COMPLETED"

    listing = await rpc(http, "ListTasks", {"pageSize": 10})
    assert any(task["id"] == task_id for task in listing["result"]["tasks"])

    # a finished task is never silently re-opened
    cancelled = await rpc(http, "CancelTask", {"id": task_id})
    if cancelled.get("error") is None:
        assert task_of(cancelled)["status"]["state"] in {
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
        }


async def test_streaming_message_yields_status_and_artifact_events(http):
    body = {
        "jsonrpc": "2.0",
        "id": "stream",
        "method": "SendStreamingMessage",
        "params": {"message": message_payload(limit=4)},
    }
    response = await http.post("/", json=body, headers={"A2A-Version": "1.0"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    raw = response.text
    assert "data:" in raw
    assert '"task"' in raw
    assert "TASK_STATE_WORKING" in raw
    assert "statusUpdate" in raw
    assert "artifactUpdate" in raw
    assert "TASK_STATE_COMPLETED" in raw
    # progress metadata carries the pipeline stage
    assert "fetch" in raw and "summarize" in raw


# ---------------------------------------------------------------------------
# protocol errors / degradation
# ---------------------------------------------------------------------------
async def test_missing_query_rejects_the_task(http):
    payload = await rpc(
        http, "SendMessage", {"message": message_payload(query="")}
    )
    task = task_of(payload)
    assert task["status"]["state"] == "TASK_STATE_REJECTED"
    text = json.dumps(task, ensure_ascii=False)
    assert "query" in text
    assert "analyze_trend" in text  # valid skills are listed back to the caller


async def test_unknown_skill_rejects_the_task(http):
    payload = await rpc(
        http, "SendMessage", {"message": message_payload(skill="make_coffee")}
    )
    assert task_of(payload)["status"]["state"] == "TASK_STATE_REJECTED"


async def test_unknown_method_and_unknown_task(http):
    error = (await rpc(http, "NoSuchMethod", {}))["error"]
    assert error["code"] == -32601  # JSON-RPC "method not found"

    missing = await rpc(http, "GetTask", {"id": "does-not-exist"})
    assert missing["error"] is not None


async def test_version_header_is_required_for_v1_methods(http):
    response = await http.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {"message": message_payload()},
        },
    )
    # no A2A-Version header -> the SDK treats it as v0.3 and rejects v1.0 methods
    payload = response.json()
    assert payload.get("error") is not None or payload.get("result") is None


# ---------------------------------------------------------------------------
# A2A 0.3 compatibility + REST transport
# ---------------------------------------------------------------------------
async def test_v0_3_compat_message_send(http):
    payload = await rpc(
        http, "message/send", {"message": legacy_message_payload()}, version=None
    )
    assert payload.get("error") is None
    task = task_of(payload)
    assert task["status"]["state"] == "completed"
    assert task["artifacts"]

    fetched = await rpc(http, "tasks/get", {"id": task["id"]}, version=None)
    assert task_of(fetched)["id"] == task["id"]


async def test_rest_transport_message_send(http):
    response = await http.post(
        "/message:send",
        json={"message": message_payload(limit=3)},
        headers={"A2A-Version": "1.0"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    task = body.get("task") or body
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"

    got = await http.get(
        f"/tasks/{task['id']}", headers={"A2A-Version": "1.0"}
    )
    assert got.status_code == 200
    assert (got.json().get("id") or got.json().get("task", {}).get("id")) == task["id"]


# ---------------------------------------------------------------------------
# SDK client round trip
# ---------------------------------------------------------------------------
async def test_sdk_client_streams_progress_and_returns_result(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as raw_http:
        client = NewsA2AClient(BASE_URL, httpx_client=raw_http, interface_url=BASE_URL)
        try:
            card = await client.connect()
            assert card.name == "news-agent"

            stages: list[str] = []
            task = None
            async for update in client.send("人形机器人", skill="analyze_trend", limit=5):
                if update.kind == "status" and update.stage:
                    stages.append(update.stage)
                elif update.kind == "task":
                    task = update.task

            assert task is not None
            assert state_name(task.status.state) == "completed"
            assert {"fetch", "filter", "analyze", "summarize"} <= set(stages)

            payload = result_from_task(task)
            assert payload is not None
            assert payload["trends"]

            # the same terminal task is retrievable / listed
            fetched = await client.get_task(task.id)
            assert fetched.id == task.id
            listing = await client.list_tasks()
            assert any(item.id == task.id for item in listing.tasks)
        finally:
            await client.close()


async def test_sdk_client_run_helper(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as raw_http:
        client = NewsA2AClient(BASE_URL, httpx_client=raw_http, interface_url=BASE_URL)
        try:
            task = await client.run("固态电池", skill="fetch_news", limit=3)
        finally:
            await client.close()
    payload = result_from_task(task)
    assert payload is not None
    assert payload["mode"] == "fetch_news"
    assert payload["articles"]


# ---------------------------------------------------------------------------
# executor units
# ---------------------------------------------------------------------------
def test_parse_skill_request_variants(settings):
    request = parse_skill_request(text="fetch_news: 固态电池", settings=settings)
    assert request.skill is SkillMode.FETCH
    assert request.query == "固态电池"
    assert request.limit == settings.default_limit

    request = parse_skill_request(
        data={"skill": "analyze_trend", "query": "AI", "limit": 999, "threshold": 2},
        settings=settings,
    )
    assert request.skill is SkillMode.TREND
    assert request.limit == settings.max_limit  # clamped
    assert request.threshold == 1.0  # clamped

    request = parse_skill_request(
        data={"skill": "fetch_news", "query": "q", "sources": "mock, google-news"},
        settings=settings,
    )
    assert request.sources == ["mock", "google-news"]

    request = parse_skill_request(text="纯文本关键词", settings=settings)
    assert request.skill is SkillMode.SUMMARIZE  # default mode
    assert request.query == "纯文本关键词"

    request = parse_skill_request(
        text='{"skill": "fetch_news", "query": "JSON 形式"}', settings=settings
    )
    assert request.skill is SkillMode.FETCH
    assert request.query == "JSON 形式"

    request = parse_skill_request(metadata={"query": "来自 metadata"}, settings=settings)
    assert request.query == "来自 metadata"


def test_parse_skill_request_errors(settings):
    with pytest.raises(SkillRequestError):
        parse_skill_request(text="   ", settings=settings)
    with pytest.raises(SkillRequestError):
        parse_skill_request(data={"skill": "nope", "query": "x"}, settings=settings)
    with pytest.raises(SkillRequestError):
        parse_skill_request(data={"query": "x", "limit": "many"}, settings=settings)
    with pytest.raises(SkillRequestError):
        parse_skill_request(data={"query": "x", "threshold": "high"}, settings=settings)


async def test_executor_timeout_returns_partial_result(agent, settings):
    executor = NewsAgentExecutor(lambda: _awaitable(agent), settings)

    ctx = RunContext()
    ctx.set_partial(NewsResult(query="人形机器人", summary="部分摘要"))
    request = parse_skill_request(data={"query": "人形机器人"}, settings=settings)

    result = executor._timeout_result(request, ctx, 30.0)  # noqa: SLF001

    assert result.degraded is True
    assert result.summary == "部分摘要"
    assert any(error.code is ErrorCode.TASK_TIMEOUT for error in result.errors)
    assert result.warnings


async def _awaitable(agent):
    return agent


def test_app_is_wired_with_sdk_components(app):
    """The protocol layer must be the SDK's, not a hand-rolled one."""
    assert isinstance(app.state.request_handler, DefaultRequestHandler)
    assert isinstance(app.state.task_store, InMemoryTaskStore)
    assert isinstance(app.state.news_executor, NewsAgentExecutor)
    assert isinstance(app.state.agent_card, a2a_pb2.AgentCard)
