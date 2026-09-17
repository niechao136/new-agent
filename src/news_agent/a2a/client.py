"""Thin convenience layer over the official A2A client (``a2a-sdk``).

It removes the protocol boilerplate so other agents / the CLI can do::

    async with NewsA2AClient("http://localhost:9901") as client:
        async for update in client.send("人形机器人", skill="analyze_trend"):
            print(update.stage, update.message)
        task = update.task          # terminal task with the news-result artifact

Everything on the wire (agent card resolution, JSON-RPC 1.0 / 0.3, SSE
streaming, task polling) is handled by the SDK.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from a2a.client import (
    A2ACardResolver,
    AgentCardResolutionError,
    Client,
    ClientConfig,
    ClientFactory,
)
from a2a.helpers.proto_helpers import (
    get_artifact_text,
    get_message_text,
    new_data_part,
    new_text_part,
)
from a2a.types import a2a_pb2
from a2a.utils.constants import TransportProtocol
from google.protobuf import json_format

from ..models import SkillMode

#: States that end a task.
TERMINAL_STATES = frozenset(
    {
        a2a_pb2.TASK_STATE_COMPLETED,
        a2a_pb2.TASK_STATE_FAILED,
        a2a_pb2.TASK_STATE_CANCELED,
        a2a_pb2.TASK_STATE_REJECTED,
    }
)


def state_name(state: int) -> str:
    """``TASK_STATE_COMPLETED`` -> ``completed``."""
    name = a2a_pb2.TaskState.Name(state)
    return name.removeprefix("TASK_STATE_").lower()


def _origin_url(url: str) -> str:
    """取 URL 的 ``scheme://host[:port]`` 部分；本就不带路径时原样返回（查询串保留）。"""
    parts = urlsplit(url)
    if not parts.path or parts.path == "/":
        return url
    return urlunsplit((parts.scheme, parts.netloc, "", parts.query, ""))


@dataclass
class TaskUpdate:
    """A normalised view over the SDK's ``StreamResponse`` events."""

    kind: str  #: ``task`` | ``status`` | ``artifact`` | ``message``
    state: str | None = None
    stage: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifact: a2a_pb2.Artifact | None = None
    task: a2a_pb2.Task | None = None

    @property
    def is_terminal(self) -> bool:
        return self.task is not None and self.task.status.state in TERMINAL_STATES


class NewsA2AClient:
    """Minimal A2A client for the news agent."""

    def __init__(
        self,
        base_url: str,
        *,
        streaming: bool = True,
        polling: bool = False,
        timeout: float = 300.0,
        httpx_client: httpx.AsyncClient | None = None,
        interface_url: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.streaming = streaming
        self.polling = polling
        self.timeout = timeout
        self._http = httpx_client or httpx.AsyncClient(timeout=timeout)
        self._owns_http = httpx_client is None
        self._interface_url = interface_url
        self._client: Client | None = None
        self.card: a2a_pb2.AgentCard | None = None

    # ------------------------------------------------------------------
    async def __aenter__(self) -> "NewsA2AClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._owns_http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    async def fetch_card(self) -> a2a_pb2.AgentCard:
        """Resolve the agent card (``/.well-known/agent-card.json``).

        带路径的 ``base_url`` 解析失败时回退到 origin 再试一次：调用方常把 RPC
        端点（如 ``http://host:9901/a2a``）当作服务地址填入，此时 well-known
        会被拼到路径下而 404，而卡片实际挂在 origin 下。
        """
        try:
            self.card = await A2ACardResolver(self._http, self.base_url).get_agent_card()
        except (AgentCardResolutionError, httpx.HTTPError):
            origin = _origin_url(self.base_url)
            if origin == self.base_url:
                raise
            self.card = await A2ACardResolver(self._http, origin).get_agent_card()
        return self.card

    async def connect(self) -> a2a_pb2.AgentCard:
        """Resolve the card and build the SDK client (idempotent)."""
        if self._client is not None and self.card is not None:
            return self.card
        card = self.card or await self.fetch_card()
        if self._interface_url:
            for interface in card.supported_interfaces:
                interface.url = self._interface_url
        config = ClientConfig(
            streaming=self.streaming,
            polling=self.polling,
            httpx_client=self._http,
            supported_protocol_bindings=[TransportProtocol.JSONRPC],
        )
        self._client = ClientFactory(config).create(card)
        return card

    @property
    def client(self) -> Client:
        if self._client is None:  # pragma: no cover - usage error
            raise RuntimeError("client is not connected; call connect() first")
        return self._client

    async def _ensure_connected(self) -> None:
        """Transparently connect on first use."""
        if self._client is None:
            await self.connect()

    # ------------------------------------------------------------------
    @staticmethod
    def build_request(
        query: str,
        *,
        skill: str | SkillMode = SkillMode.SUMMARIZE,
        limit: int | None = None,
        language: str | None = None,
        since: str | None = None,
        until: str | None = None,
        sources: list[str] | None = None,
        threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> a2a_pb2.SendMessageRequest:
        """Build the ``SendMessageRequest`` (DataPart + human readable TextPart)."""
        skill_id = skill.value if isinstance(skill, SkillMode) else str(skill)
        payload: dict[str, Any] = {"skill": skill_id, "query": query}
        if limit is not None:
            payload["limit"] = limit
        if language:
            payload["language"] = language
        if since:
            payload["since"] = since
        if until:
            payload["until"] = until
        if sources:
            payload["sources"] = list(sources)
        if threshold is not None:
            payload["threshold"] = threshold

        request = a2a_pb2.SendMessageRequest(
            message=a2a_pb2.Message(
                message_id=uuid.uuid4().hex,
                role=a2a_pb2.ROLE_USER,
                parts=[
                    new_data_part(payload, media_type="application/json"),
                    new_text_part(f"{skill_id}: {query}", media_type="text/plain"),
                ],
            )
        )
        if metadata:
            request.metadata.update(metadata)
        return request

    # ------------------------------------------------------------------
    async def send(self, query: str, **kwargs: Any) -> AsyncIterator[TaskUpdate]:
        """Send a message and yield normalised updates until the task ends.

        The final yielded update has ``kind == "task"`` and carries the terminal
        ``Task`` (including the ``news-result`` artifact).
        """
        await self._ensure_connected()
        request = self.build_request(query, **kwargs)
        async for response in self.client.send_message(request):
            if response.HasField("task"):
                task = response.task
                yield TaskUpdate(kind="task", state=state_name(task.status.state), task=task)
                if task.status.state in TERMINAL_STATES:
                    return
            elif response.HasField("status_update"):
                update = response.status_update
                metadata = (
                    json_format.MessageToDict(update.metadata)
                    if update.HasField("metadata")
                    else {}
                )
                text = ""
                if update.status.HasField("message"):
                    text = get_message_text(update.status.message)
                state = state_name(update.status.state)
                yield TaskUpdate(
                    kind="status",
                    state=state,
                    stage=metadata.get("stage"),
                    message=text,
                    data=metadata.get("data") or {},
                )
                if update.status.state in TERMINAL_STATES:
                    task = await self._safe_get_task(update.task_id)
                    if task is not None:
                        yield TaskUpdate(
                            kind="task", state=state_name(task.status.state), task=task
                        )
                    return
            elif response.HasField("artifact_update"):
                artifact = response.artifact_update.artifact
                yield TaskUpdate(
                    kind="artifact",
                    message=get_artifact_text(artifact),
                    artifact=artifact,
                )
            elif response.HasField("message"):
                yield TaskUpdate(kind="message", message=get_message_text(response.message))

    async def run(self, query: str, **kwargs: Any) -> a2a_pb2.Task:
        """Send a message and return the terminal task."""
        task: a2a_pb2.Task | None = None
        async for update in self.send(query, **kwargs):
            if update.task is not None:
                task = update.task
        if task is None:  # pragma: no cover - protocol violation
            raise RuntimeError("the agent did not return a task")
        return task

    async def get_task(self, task_id: str, *, history_length: int = 20) -> a2a_pb2.Task:
        await self._ensure_connected()
        return await self.client.get_task(
            a2a_pb2.GetTaskRequest(id=task_id, history_length=history_length)
        )

    async def cancel_task(self, task_id: str) -> a2a_pb2.Task:
        await self._ensure_connected()
        return await self.client.cancel_task(a2a_pb2.CancelTaskRequest(id=task_id))

    async def list_tasks(self, *, page_size: int = 20) -> a2a_pb2.ListTasksResponse:
        await self._ensure_connected()
        return await self.client.list_tasks(a2a_pb2.ListTasksRequest(page_size=page_size))

    async def subscribe(self, task_id: str) -> AsyncIterator[TaskUpdate]:
        """Resubscribe to a running task's event stream."""
        await self._ensure_connected()
        async for event in self.client.subscribe(a2a_pb2.SubscribeToTaskRequest(id=task_id)):
            if event.HasField("status_update"):
                update = event.status_update
                metadata = (
                    json_format.MessageToDict(update.metadata)
                    if update.HasField("metadata")
                    else {}
                )
                yield TaskUpdate(
                    kind="status",
                    state=state_name(update.status.state),
                    stage=metadata.get("stage"),
                    message=get_message_text(update.status.message)
                    if update.status.HasField("message")
                    else "",
                )
            elif event.HasField("task"):
                yield TaskUpdate(
                    kind="task",
                    state=state_name(event.task.status.state),
                    task=event.task,
                )

    # ------------------------------------------------------------------
    async def _safe_get_task(self, task_id: str) -> a2a_pb2.Task | None:
        try:
            return await self.get_task(task_id)
        except Exception as exc:  # noqa: BLE001 - the artifacts are optional
            import logging

            logging.getLogger("news_agent.a2a.client").warning(
                "could not fetch final task %s: %r", task_id, exc
            )
            return None


def _restore_integers(value: Any) -> Any:
    """``google.protobuf.Value`` has a single number type (double).

    ``MessageToDict`` therefore turns every count into a float; restore integral
    values so consumers see ``10`` instead of ``10.0``.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _restore_integers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_integers(item) for item in value]
    return value


def result_from_task(task: a2a_pb2.Task) -> dict[str, Any] | None:
    """Extract the ``NewsResult`` dict from a task's artifacts."""
    for artifact in task.artifacts:
        for part in artifact.parts:
            if part.HasField("data"):
                return _restore_integers(json_format.MessageToDict(part.data))
    return None


__all__ = [
    "NewsA2AClient",
    "TaskUpdate",
    "TERMINAL_STATES",
    "result_from_task",
    "state_name",
]
