"""默认 RSS 源配置测试：覆盖国外媒体与多行业源。"""

from news_agent.config import Settings


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
        "nikkei-asia",
    ):
        assert name in names, f"missing international feed: {name}"


def test_industry_feeds_enabled():
    names = _source_names()
    for name in (
        "cnbc-economy",
        "marketwatch",
        "economist-finance",
        "espn-sports",
        "variety-entertainment",
        "stat-health",
    ):
        assert name in names, f"missing industry feed: {name}"


def test_topic_feeds_are_enabled_sources():
    names = _source_names()
    for name in ("36kr", "techcrunch", "bbc-tech"):
        assert name in names


def test_all_topic_feeds_have_urls():
    from news_agent.config import TOPIC_RSS_FEEDS

    for name, url in TOPIC_RSS_FEEDS.items():
        assert url.startswith("http"), f"invalid feed url for {name}: {url}"
