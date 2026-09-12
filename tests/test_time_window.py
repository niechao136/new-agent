"""时间窗口解析与默认窗口测试。"""

from datetime import datetime, timedelta, timezone

from news_agent.models import SkillRequest
from news_agent.time_window import parse_time_window


def _now() -> datetime:
    # 2026-09-12 是周六
    return datetime(2026, 9, 12, 5, 0, 0, tzinfo=timezone.utc)


def test_this_week_parses_to_monday_and_cleans_query():
    since, until, cleaned = parse_time_window("帮我汇总一下本周的科技新闻", now=_now())
    assert since == datetime(2026, 9, 7, 0, 0, 0, tzinfo=timezone.utc)  # 本周一
    assert until is None
    assert cleaned == "科技新闻"


def test_recent_n_days():
    since, _, cleaned = parse_time_window("最近7天的AI新闻", now=_now())
    assert since is not None
    assert (_now() - since).days == 7
    assert "AI" in cleaned


def test_english_past_n_days():
    since, _, _ = parse_time_window("tech news from the past 3 days", now=_now())
    assert since is not None
    assert (_now() - since).days == 3


def test_yesterday_sets_until():
    since, until, _ = parse_time_window("昨天的新闻", now=_now())
    assert since == datetime(2026, 9, 11, 0, 0, 0, tzinfo=timezone.utc)
    assert until == datetime(2026, 9, 12, 0, 0, 0, tzinfo=timezone.utc)


def test_last_week_sets_range():
    since, until, _ = parse_time_window("上周的科技新闻", now=_now())
    assert since == datetime(2026, 8, 31, 0, 0, 0, tzinfo=timezone.utc)
    assert until == datetime(2026, 9, 7, 0, 0, 0, tzinfo=timezone.utc)


def test_no_time_expression_returns_none_window():
    since, until, cleaned = parse_time_window("科技新闻", now=_now())
    assert since is None
    assert until is None
    assert cleaned == "科技新闻"


def test_cleaned_query_never_empties():
    # 查询只有时间词时，清理后不能为空
    _, _, cleaned = parse_time_window("本周", now=_now())
    assert cleaned


def test_with_default_window_fills_since():
    req = SkillRequest(query="科技新闻")
    assert req.since is None
    req.with_default_window(7)
    assert req.since is not None
    age = datetime.now(timezone.utc) - req.since
    assert timedelta(days=7, seconds=-60) <= age <= timedelta(days=7, seconds=60)


def test_with_default_window_respects_explicit_since():
    explicit = datetime(2026, 9, 1, tzinfo=timezone.utc)
    req = SkillRequest(query="科技新闻", since=explicit)
    req.with_default_window(7)
    assert req.since == explicit


def test_with_default_window_zero_disables():
    req = SkillRequest(query="科技新闻")
    req.with_default_window(0)
    assert req.since is None
