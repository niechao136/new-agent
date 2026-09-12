"""LLM 意图解析（intent）测试。"""

from datetime import datetime, timedelta, timezone

import pytest

from news_agent.a2a import aparse_skill_request
from news_agent.config import LLMSettings, Settings
from news_agent.intent import (
    LLMIntentError,
    LLMIntentParser,
    QueryIntent,
    _LLMIntent,
    intent_to_window,
    parse_query_intent,
)

# 2026-09-12 是周六
NOW = datetime(2026, 9, 12, 5, 0, 0, tzinfo=timezone.utc)


class _FakeStructured:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def ainvoke(self, messages):
        return self._behaviour()


class _FakeLLM:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def with_structured_output(self, schema, include_raw=True):
        return _FakeStructured(self._behaviour)


def _parser_with(result=None, error: Exception | None = None) -> LLMIntentParser:
    parser = LLMIntentParser(
        LLMSettings(base_url="http://localhost:8000/v1", api_key="test")
    )

    def _behaviour():
        if error is not None:
            raise error
        return result

    parser._llm = _FakeLLM(_behaviour)  # noqa: SLF001 - 测试桩注入
    return parser


def _settings(llm: LLMSettings | None = None) -> Settings:
    return Settings(llm=llm or LLMSettings(enabled=False))


# ---------------------------------------------------------------------------
# intent_to_window
# ---------------------------------------------------------------------------
def test_today():
    intent = _LLMIntent(search_query="科技新闻", time_type="today")
    since, until, cleaned = intent_to_window(intent, now=NOW)
    assert since == datetime(2026, 9, 12, tzinfo=timezone.utc)
    assert until is None
    assert cleaned == "科技新闻"


def test_past_days():
    intent = _LLMIntent(search_query="AI", time_type="past_days", past_days=3)
    since, _, _ = intent_to_window(intent, now=NOW)
    assert (NOW - since).days == 3


def test_date_range_end_inclusive():
    intent = _LLMIntent(
        search_query="汇率",
        time_type="date_range",
        start_date="2026-09-01",
        end_date="2026-09-10",
    )
    since, until, _ = intent_to_window(intent, now=NOW)
    assert since == datetime(2026, 9, 1, tzinfo=timezone.utc)
    # 包含结束日 → until 为次日零点
    assert until == datetime(2026, 9, 11, tzinfo=timezone.utc)


def test_this_year():
    intent = _LLMIntent(search_query="新能源", time_type="this_year")
    since, _, _ = intent_to_window(intent, now=NOW)
    assert since == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_none_type_no_window():
    intent = _LLMIntent(search_query="机器人", time_type="none")
    since, until, _ = intent_to_window(intent, now=NOW)
    assert since is None
    assert until is None


# ---------------------------------------------------------------------------
# LLMIntentParser
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parser_returns_intent_with_keywords():
    parser = _parser_with(
        _LLMIntent(
            search_query="固态电池",
            keywords=["solid-state battery", "全固态电池", "固态电池"],
            time_type="today",
        )
    )
    intent = await parser.parse("帮我汇总一下今天的固态电池新闻", now=NOW)

    assert isinstance(intent, QueryIntent)
    assert intent.search_query == "固态电池"
    assert intent.since == datetime(2026, 9, 12, tzinfo=timezone.utc)
    # 与 search_query 重复的关键词被剔除
    assert intent.keywords == ["solid-state battery", "全固态电池"]


@pytest.mark.asyncio
async def test_parser_empty_query_falls_back_to_raw():
    parser = _parser_with(_LLMIntent(search_query="  ", time_type="none"))
    intent = await parser.parse("科技新闻", now=NOW)
    assert intent.search_query == "科技新闻"


@pytest.mark.asyncio
async def test_parser_raises_on_llm_error():
    parser = _parser_with(error=RuntimeError("boom"))
    with pytest.raises(LLMIntentError):
        await parser.parse("科技新闻", now=NOW)


# ---------------------------------------------------------------------------
# parse_query_intent（LLM 优先 + 正则回退）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unconfigured_settings_uses_regex():
    intent = await parse_query_intent(
        "帮我汇总一下本周的科技新闻", LLMSettings(enabled=False), now=NOW
    )
    assert intent.since == datetime(2026, 9, 7, tzinfo=timezone.utc)  # 本周一
    assert intent.search_query == "科技新闻"
    assert intent.keywords == []


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_regex():
    llm = LLMSettings(base_url="http://localhost:8000/v1", api_key="test", max_retries=0)
    intent = await parse_query_intent("帮我汇总一下本周的科技新闻", llm, now=NOW)
    assert intent.since == datetime(2026, 9, 7, tzinfo=timezone.utc)
    assert intent.search_query == "科技新闻"


@pytest.mark.asyncio
async def test_llm_success_path(monkeypatch):
    llm = LLMSettings(base_url="http://localhost:8000/v1", api_key="test")

    async def _fake_parse(self, query, *, now=None):
        return QueryIntent(
            search_query="固态电池",
            keywords=["solid-state battery"],
            since=NOW - timedelta(days=2),
        )

    monkeypatch.setattr("news_agent.intent.LLMIntentParser.parse", _fake_parse)
    intent = await parse_query_intent("最近两天的固态电池新闻", llm, now=NOW)
    assert intent.search_query == "固态电池"
    assert intent.keywords == ["solid-state battery"]
    assert (NOW - intent.since).days == 2


# ---------------------------------------------------------------------------
# aparse_skill_request（executor 集成）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aparse_uses_intent_parser(monkeypatch):
    async def _fake_intent(query, llm_settings, *, now=None):
        return QueryIntent(
            search_query="科技新闻",
            keywords=["tech news", "technology"],
            since=datetime(2026, 9, 7, tzinfo=timezone.utc),
        )

    monkeypatch.setattr("news_agent.a2a.executor.parse_query_intent", _fake_intent)
    request = await aparse_skill_request(
        data={"query": "请帮我汇总一下本周的科技新闻"},
        settings=_settings(LLMSettings(base_url="http://x", api_key="k")),
    )
    assert request.query == "科技新闻"
    assert request.since == datetime(2026, 9, 7, tzinfo=timezone.utc)
    assert request.keywords == ["tech news", "technology"]


@pytest.mark.asyncio
async def test_aparse_merges_caller_keywords(monkeypatch):
    async def _fake_intent(query, llm_settings, *, now=None):
        return QueryIntent(search_query=query, keywords=["solid-state battery"])

    monkeypatch.setattr("news_agent.a2a.executor.parse_query_intent", _fake_intent)
    request = await aparse_skill_request(
        data={"query": "固态电池", "keywords": ["固态电池", "全固态电池", "solid-state battery"]},
        settings=_settings(),
    )
    assert request.keywords == ["solid-state battery", "全固态电池"]


@pytest.mark.asyncio
async def test_aparse_skips_intent_when_since_given(monkeypatch):
    async def _fail(*args, **kwargs):
        raise AssertionError("intent parser should not be called")

    monkeypatch.setattr("news_agent.a2a.executor.parse_query_intent", _fail)
    explicit = datetime(2026, 9, 1, tzinfo=timezone.utc)
    request = await aparse_skill_request(
        data={"query": "科技新闻", "since": explicit.isoformat()},
        settings=_settings(),
    )
    assert request.query == "科技新闻"
    assert request.since is not None
