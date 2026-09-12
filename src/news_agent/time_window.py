"""自然语言时间窗口解析。

用户输入通常是「帮我汇总一下本周的科技新闻」这样的整句，其中包含时间
表达（本周/昨天/最近7天…）。此前这些词只被当作搜索关键词原样发给新闻源，
既没有时间窗口过滤（旧闻混入），也拖累了搜索相关性。

``parse_time_window`` 从查询中识别时间表达，返回 ``(since, until,
cleaned_query)``；无法识别时返回 ``(None, None, 清理后的原查询)``。
清理逻辑会同时去掉「帮我/汇总一下」等与搜索无关的口语化填充词。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .models import utcnow

# 匹配顺序即优先级：带数字的相对天数必须先于裸「最近/近日」。
_RELATIVE_DAYS_ZH = re.compile(r"(?:最近|近|过去|这)\s*(\d{1,3})\s*[天日]")
_RELATIVE_DAYS_EN = re.compile(r"\b(?:last|past|recent)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)
_TODAY = re.compile(r"今天|今日|\btoday\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"昨天|昨日|\byesterday\b", re.IGNORECASE)
_THIS_WEEK = re.compile(r"本周|这周|这一周|这个星期|本星期|\bthis week\b", re.IGNORECASE)
_LAST_WEEK = re.compile(r"上周|上星期|\blast week\b", re.IGNORECASE)
_THIS_MONTH = re.compile(r"本月|这个月|本月份|\bthis month\b", re.IGNORECASE)
_LAST_MONTH = re.compile(r"上个月|上月|\blast month\b", re.IGNORECASE)
_THIS_YEAR = re.compile(r"今年|\bthis year\b", re.IGNORECASE)
_BARE_RECENT = re.compile(r"近日|最近|近期|\brecently\b|\blately\b", re.IGNORECASE)

# 与搜索无关的口语化填充词（仅在清理后仍保留足够内容时才生效）。
_FILLERS = re.compile(
    r"帮我|请你|麻烦你?|给我|我想|我要|"
    r"汇总|总结|归纳|梳理|整理|概括|搜集|收集|查一下|搜索|检索|"
    r"相关|一下|看看|说说|讲讲"
)
_DANGLING = re.compile(r"^[的了吗呢吧啊，。、！？\s]+|[的了吗呢吧啊，。、！？\s]+$")

_TIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    _RELATIVE_DAYS_EN,
    _TODAY,
    _YESTERDAY,
    _THIS_WEEK,
    _LAST_WEEK,
    _THIS_MONTH,
    _LAST_MONTH,
    _THIS_YEAR,
    _BARE_RECENT,
)


def _midnight(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _clean_query(text: str) -> str:
    cleaned = _FILLERS.sub("", text)
    cleaned = _DANGLING.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # 清理后太短则保留原文，避免把有效关键词全部删光
    return cleaned if len(cleaned) >= 2 else text.strip()


def parse_time_window(
    query: str, *, now: datetime | None = None
) -> tuple[datetime | None, datetime | None, str]:
    """从查询中解析时间窗口。

    @returns (since, until, cleaned_query)；未识别出时间表达时
             since/until 为 None，cleaned_query 仍会做填充词清理。
    """
    now = now or utcnow()
    text = query.strip()
    since: datetime | None = None
    until: datetime | None = None

    if m := _RELATIVE_DAYS_ZH.search(text):
        since = now - timedelta(days=min(int(m.group(1)), 365))
    elif m := _RELATIVE_DAYS_EN.search(text):
        since = now - timedelta(days=min(int(m.group(1)), 365))
    elif _TODAY.search(text):
        since = _midnight(now)
    elif _YESTERDAY.search(text):
        since = _midnight(now) - timedelta(days=1)
        until = _midnight(now)
    elif _THIS_WEEK.search(text):
        since = _midnight(now) - timedelta(days=now.weekday())
    elif _LAST_WEEK.search(text):
        this_monday = _midnight(now) - timedelta(days=now.weekday())
        since = this_monday - timedelta(days=7)
        until = this_monday
    elif _THIS_MONTH.search(text):
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif _LAST_MONTH.search(text):
        this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = this_month - timedelta(days=1)
        since = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        until = this_month
    elif _THIS_YEAR.search(text):
        since = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif _BARE_RECENT.search(text):
        since = now - timedelta(days=7)

    if m := _RELATIVE_DAYS_ZH.search(text):
        cleaned_source = _RELATIVE_DAYS_ZH.sub(" ", text, count=1)
    else:
        cleaned_source = text
    for pattern in _TIME_PATTERNS:
        cleaned_source = pattern.sub(" ", cleaned_source)
    cleaned = _clean_query(cleaned_source)
    # 查询只有时间词时清理结果可能为空，此时回退到原始查询
    return since, until, (cleaned or text.strip() or query.strip())
