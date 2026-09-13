"""Agent Card construction using the official ``a2a-sdk`` types.

The card is the *contract* advertised to the A2A gateway.  Everything here is
built from :mod:`a2a.types` (protobuf) types -- no hand-rolled protocol models.

A2A 1.0 ``AgentSkill`` has no schema fields, so the JSON schemas of the request
and of the :class:`~news_agent.models.NewsResult` response are published as an
*agent extension* (``AgentExtension.params``) instead.  Clients that do not
understand the extension simply ignore it, while stricter platforms can validate
before sending.
"""

from __future__ import annotations

from typing import Any

from a2a.types import a2a_pb2
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from google.protobuf import json_format, struct_pb2

from ..config import Settings
from ..models import SkillMode

#: Well-known path used by A2A >= 1.0.
AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"
#: Well-known path used by A2A 0.3 gateways (kept for compatibility).
AGENT_CARD_LEGACY_PATH = "/.well-known/agent.json"

#: Extension carrying the concrete JSON schemas of this agent's skills.
SKILL_SCHEMA_EXTENSION_URI = "https://news-agent.dev/a2a/skill-schemas"
#: Extension describing the progress metadata published on status updates.
PROGRESS_EXTENSION_URI = "https://news-agent.dev/a2a/task-progress"

_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "NewsRequest：作为 Message 的 DataPart.data 发送（或使用纯文本 "
        '"<skill>: <query>"），也可放在请求 metadata 中。'
    ),
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 300,
            "description": "关键词 / 领域 / 话题，例如 “人形机器人” 或 “AI regulation”。",
        },
        "skill": {
            "type": "string",
            "enum": [mode.value for mode in SkillMode],
            "default": SkillMode.SUMMARIZE.value,
            "description": "要执行的技能；缺省时取服务端 DEFAULT_MODE。",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string", "maxLength": 64},
            "maxItems": 8,
            "description": (
                "相关性匹配用的扩展关键词（同义词 / 英文译名等，可选）。"
                "缺省时由服务端 LLM 意图解析自动生成。"
            ),
        },
        "since": {
            "type": "string",
            "format": "date-time",
            "description": "只返回该时间之后发布的新闻（ISO 8601，可选）。",
        },
        "until": {
            "type": "string",
            "format": "date-time",
            "description": "只返回该时间之前发布的新闻（ISO 8601，可选）。",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 15,
            "description": "返回的最大文章数量。",
        },
        "language": {
            "type": "string",
            "default": "zh",
            "description": "结果与摘要语言（ISO-639-1，如 zh / en）。",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "限定使用的新闻源名称；缺省表示全部已启用源。",
        },
        "threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "相关度过滤阈值，覆盖服务端默认值。",
        },
    },
    "required": ["query"],
    "additionalProperties": True,
}

_ARTICLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "url": {"type": "string", "format": "uri"},
        "source": {"type": "string"},
        "published_at": {"type": "string", "format": "date-time", "nullable": True},
        "language": {"type": "string", "nullable": True},
        "relevance": {"type": "number", "description": "0-1 相关度"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "sentiment_score": {"type": "number", "description": "-1..1 情感得分"},
        "stance": {"type": "string", "nullable": True},
        "entities": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": {"type": "string"}},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "nullable": True},
        "duplicate_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "title", "url", "source"],
}

_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "enum": [
                "invalid_request",
                "no_results",
                "source_unavailable",
                "fetch_timeout",
                "rate_limited",
                "parse_error",
                "llm_failed",
                "llm_timeout",
                "context_too_long",
                "cache_error",
                "task_timeout",
                "task_canceled",
                "internal_error",
            ],
        },
        "message": {"type": "string"},
        "source": {"type": "string", "nullable": True},
        "stage": {"type": "string", "nullable": True},
        "retryable": {"type": "boolean"},
    },
    "required": ["code", "message"],
}

#: JSON schema of the ``news-result`` artifact data part.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "NewsResult",
    "description": "新闻抓取 + 结构化分析 + 汇总摘要的结果对象",
    "properties": {
        "query": {"type": "string"},
        "mode": {"type": "string", "enum": [mode.value for mode in SkillMode]},
        "language": {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "duration_ms": {"type": "integer"},
        "degraded": {
            "type": "boolean",
            "description": "true 表示部分阶段降级（缓存回退 / 启发式替代 LLM / 超时部分结果）",
        },
        "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        "summary": {"type": "string", "description": "话题级聚合摘要"},
        "articles": {"type": "array", "items": _ARTICLE_SCHEMA},
        "trends": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "mentions": {"type": "integer"},
                    "sentiment": {"type": "string"},
                    "average_sentiment": {"type": "number"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "representative_urls": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "errors": {"type": "array", "items": _ERROR_SCHEMA},
        "timings_ms": {"type": "object", "additionalProperties": {"type": "number"}},
        "metrics": {"type": "object"},
    },
    "required": ["query", "mode", "articles"],
}

_COMMON_TAGS = ["news", "news-agent", "rss", "analysis", "summarization"]

_SKILLS: list[dict[str, Any]] = [
    {
        "id": SkillMode.FETCH.value,
        "name": "Fetch news",
        "description": (
            "按关键词抓取新闻原文元数据（标题、来源、发布时间、摘要、链接），"
            "完成跨源去重与相关性过滤，不做 LLM 分析，响应最快。"
        ),
        "tags": [*_COMMON_TAGS, "fetch", "dedup"],
        "examples": ["fetch_news: 抓取最近 3 天关于“人形机器人”的新闻，最多 20 条"],
    },
    {
        "id": SkillMode.SUMMARIZE.value,
        "name": "Summarize news",
        "description": (
            "抓取 + 去重 + 相关性过滤 + 结构化抽取（实体/事件/情感/立场）+ "
            "map-reduce 话题级摘要。适合“最近发生了什么”这类问题。"
        ),
        "tags": [*_COMMON_TAGS, "summarize", "structured-output"],
        "examples": ["summarize_news: 总结本周关于“固态电池”的新闻并给出正负面判断"],
    },
    {
        "id": SkillMode.TREND.value,
        "name": "Analyze trend",
        "description": (
            "在 summarize_news 的基础上做话题级聚合，输出高频实体/主题、"
            "提及次数、整体情感倾向与代表性链接，用于趋势研判。"
        ),
        "tags": [*_COMMON_TAGS, "trend", "aggregation"],
        "examples": ["analyze_trend: 分析最近一个月“生成式 AI 监管”的舆论趋势"],
    },
]


def skill_ids() -> list[str]:
    """Skill ids advertised by this agent."""
    return [skill["id"] for skill in _SKILLS]


def _struct(payload: dict[str, Any]) -> struct_pb2.Struct:
    return json_format.ParseDict(payload, struct_pb2.Struct())


def security_declaration(settings: Settings) -> tuple[dict[str, a2a_pb2.SecurityScheme], list[a2a_pb2.SecurityRequirement]]:
    """A2A ``securitySchemes`` / ``security`` advertised when auth is enabled.

    Callers may use either the ``Authorization: Bearer`` header or the
    dedicated API-key header (see :class:`~news_agent.config.AuthSettings`).
    """
    if not settings.auth.configured:
        return {}, []
    schemes = {
        "bearer": a2a_pb2.SecurityScheme(
            http_auth_security_scheme=a2a_pb2.HTTPAuthSecurityScheme(
                scheme="bearer",
                description="API key as 'Authorization: Bearer <key>'.",
            )
        ),
        "api_key": a2a_pb2.SecurityScheme(
            api_key_security_scheme=a2a_pb2.APIKeySecurityScheme(
                location="header",
                name=settings.auth.api_key_header,
                description=f"API key as '{settings.auth.api_key_header}: <key>'.",
            )
        ),
    }
    requirements = [
        a2a_pb2.SecurityRequirement(schemes={"bearer": a2a_pb2.StringList()}),
        a2a_pb2.SecurityRequirement(schemes={"api_key": a2a_pb2.StringList()}),
    ]
    return schemes, requirements


def build_agent_card(settings: Settings) -> a2a_pb2.AgentCard:
    """Build the A2A ``AgentCard`` for this news agent."""
    url = settings.resolve_agent_url()
    skill_schema_extension = a2a_pb2.AgentExtension(
        uri=SKILL_SCHEMA_EXTENSION_URI,
        description=(
            "JSON schemas for the DataPart payload (requestSchema) and for the "
            "result artifact (resultSchema) of every skill."
        ),
        required=False,
        params=_struct({"requestSchema": _REQUEST_SCHEMA, "resultSchema": OUTPUT_SCHEMA}),
    )
    progress_extension = a2a_pb2.AgentExtension(
        uri=PROGRESS_EXTENSION_URI,
        description=(
            "Working status updates carry metadata {stage, kind, data} where "
            "stage ∈ fetch|filter|analyze|summarize|format."
        ),
        required=False,
    )
    security_schemes, security_requirements = security_declaration(settings)

    return a2a_pb2.AgentCard(
        name=settings.agent_name,
        description=settings.agent_description,
        version=settings.agent_version,
        documentation_url=f"{url}/docs",
        provider=a2a_pb2.AgentProvider(organization="news-agent", url=url),
        supported_interfaces=[
            a2a_pb2.AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            ),
            a2a_pb2.AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.HTTP_JSON.value,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            ),
        ],
        capabilities=a2a_pb2.AgentCapabilities(
            streaming=True,
            push_notifications=False,
            extensions=[skill_schema_extension, progress_extension],
        ),
        default_input_modes=["application/json", "text/plain"],
        default_output_modes=["application/json", "text/plain"],
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        skills=[
            a2a_pb2.AgentSkill(
                id=skill["id"],
                name=skill["name"],
                description=skill["description"],
                tags=list(skill["tags"]),
                examples=list(skill["examples"]),
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"],
            )
            for skill in _SKILLS
        ],
    )


def agent_card_dict(settings: Settings) -> dict[str, Any]:
    """Card as a JSON-friendly dict (camelCase, proto JSON mapping)."""
    return json_format.MessageToDict(
        build_agent_card(settings), preserving_proto_field_name=False
    )


def skill_descriptions() -> list[dict[str, Any]]:
    """Skill metadata incl. the JSON schemas (used by ``GET /skills`` / CLI)."""
    return [
        {**skill, "inputSchema": _REQUEST_SCHEMA, "outputSchema": OUTPUT_SCHEMA}
        for skill in _SKILLS
    ]


__all__ = [
    "AGENT_CARD_LEGACY_PATH",
    "AGENT_CARD_WELL_KNOWN_PATH",
    "OUTPUT_SCHEMA",
    "PROGRESS_EXTENSION_URI",
    "SKILL_SCHEMA_EXTENSION_URI",
    "agent_card_dict",
    "build_agent_card",
    "security_declaration",
    "skill_descriptions",
    "skill_ids",
]
