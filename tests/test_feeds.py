"""默认 RSS 源配置测试：覆盖国外媒体、多行业源与栏目领域标签。"""

from news_agent.config import TOPIC_RSS_FEEDS, Settings
from news_agent.relevance import TOPIC_QUERY_TERMS

KNOWN_TOPICS = set(TOPIC_QUERY_TERMS)


def _source_names() -> set[str]:
    return {source.name for source in Settings.default_sources()}


def test_international_general_feeds_enabled():
    names = _source_names()
    for name in (
        "bbc-world",
        "aljazeera",
        "npr",
        "dw",
        "france24",
        "cna",
        "japantimes",
    ):
        assert name in names, f"missing international feed: {name}"


def test_industry_feeds_enabled():
    names = _source_names()
    for name in (
        "cnbc-top",
        "marketwatch",
        "wsj-markets",
        "economist-finance",
        "bbc-sport",
        "skysports",
        "variety",
        "deadline",
        "stat-health",
    ):
        assert name in names, f"missing industry feed: {name}"


def test_domestic_tech_feeds_enabled():
    names = _source_names()
    for name in ("sspai", "solidot", "ithome", "ifanr"):
        assert name in names, f"missing domestic tech feed: {name}"


def test_dead_feeds_are_gone():
    """生产验证已失效的源不应再出现（避免每次跑都产生错误）。"""
    names = _source_names()
    for name in ("36kr", "huxiu", "espn-sports", "cnbc-economy", "nikkei-asia"):
        assert name not in names, f"dead feed still configured: {name}"


def test_every_topic_feed_declares_a_known_topic():
    for name, (url, topic) in TOPIC_RSS_FEEDS.items():
        assert url.startswith("http"), f"invalid feed url for {name}: {url}"
        assert topic in KNOWN_TOPICS, f"unknown topic for {name}: {topic}"
        assert "{" not in url, f"topic feed must not use placeholders: {name}"


def test_topic_feeds_expose_topic_to_sources():
    by_name = {source.name: source for source in Settings.default_sources()}
    assert by_name["techcrunch"].topic == "tech"
    assert by_name["wsj-markets"].topic == "business"
    assert by_name["bbc-sport"].topic == "sports"
    # 搜索类源没有固定栏目
    assert by_name["google-news"].topic is None


def test_all_topics_have_at_least_one_feed():
    covered = {topic for _url, topic in TOPIC_RSS_FEEDS.values()}
    for topic in KNOWN_TOPICS:
        assert topic in covered, f"topic without any feed: {topic}"
