"""A2A layer built on the official ``a2a-sdk``.

No hand-rolled protocol types: the agent card, the task state machine, the JSON-RPC
/ REST transports and the SSE event stream all come from the SDK.  This package
only contains the news specific glue:

* :mod:`news_agent.a2a.card`     -- builds the SDK ``AgentCard`` (+ JSON schemas)
* :mod:`news_agent.a2a.executor` -- ``AgentExecutor`` bridging LangGraph <-> A2A
* :mod:`news_agent.a2a.server`   -- FastAPI app assembling SDK routes
* :mod:`news_agent.a2a.client`   -- convenience wrapper over the SDK client
"""

from __future__ import annotations

from .card import (
    AGENT_CARD_LEGACY_PATH,
    AGENT_CARD_WELL_KNOWN_PATH,
    OUTPUT_SCHEMA,
    PROGRESS_EXTENSION_URI,
    SKILL_SCHEMA_EXTENSION_URI,
    agent_card_dict,
    build_agent_card,
    skill_descriptions,
    skill_ids,
)
from .client import NewsA2AClient, TaskUpdate, result_from_task, state_name
from .executor import (
    NewsAgentExecutor,
    SkillRequestError,
    TERMINAL_STATES,
    parse_skill_request,
)
from .server import create_app

__all__ = [
    "create_app",
    "build_agent_card",
    "agent_card_dict",
    "skill_descriptions",
    "skill_ids",
    "OUTPUT_SCHEMA",
    "AGENT_CARD_WELL_KNOWN_PATH",
    "AGENT_CARD_LEGACY_PATH",
    "SKILL_SCHEMA_EXTENSION_URI",
    "PROGRESS_EXTENSION_URI",
    "NewsAgentExecutor",
    "parse_skill_request",
    "SkillRequestError",
    "TERMINAL_STATES",
    "NewsA2AClient",
    "TaskUpdate",
    "result_from_task",
    "state_name",
]
