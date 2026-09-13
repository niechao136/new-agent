"""Runtime configuration.

Everything is driven by environment variables prefixed with ``NEWS_AGENT_`` so
that the same image can be deployed as a container without code changes.  The
settings object is a plain pydantic model which makes it trivial to construct a
customised instance in tests (``Settings(llm=LLMSettings(enabled=False))``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "NEWS_AGENT_"

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "news-agent/0.1 (+https://example.com/news-agent)",
]

# Google / Bing news RSS both accept a free-text query and need no API key,
# which makes them the default "always available" sources.
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
)
BING_NEWS_RSS = "https://www.bing.com/news/search?q={query}&format=RSS&setlang={hl}"

#: 零配置主题 RSS 源（无需 API key）。它们不是关键词搜索，而是固定栏目流：
#: 抓回后由相关性过滤按查询取舍，作为 Google/Bing 搜索之外的补充。
#: 值为 ``(feed_url, topic)``；``topic`` 用于「泛化查询」按栏目路由
#: （见 ``news_agent.relevance``），因此每个源必须归类。
#: 所有 URL 都已用 ``curl`` 在生产环境验证过（返回条目数 > 0）。
TOPIC_RSS_FEEDS: dict[str, tuple[str, str]] = {
    # --- 国内科技 ---
    "sspai": ("https://sspai.com/feed", "tech"),
    "solidot": ("https://www.solidot.org/index.rss", "tech"),
    "ithome": ("https://www.ithome.com/rss/", "tech"),
    "ifanr": ("https://www.ifanr.com/feed", "tech"),
    # --- 国外科技 ---
    "techcrunch": ("https://techcrunch.com/feed/", "tech"),
    "the-verge": ("https://www.theverge.com/rss/index.xml", "tech"),
    "bbc-tech": ("https://feeds.bbci.co.uk/news/technology/rss.xml", "tech"),
    # --- 国际综合 ---
    "bbc-world": ("https://feeds.bbci.co.uk/news/world/rss.xml", "world"),
    "aljazeera": ("https://www.aljazeera.com/xml/rss/all.xml", "world"),
    "npr": ("https://feeds.npr.org/1001/rss.xml", "world"),
    "dw": ("https://rss.dw.com/rdf/rss-en-all", "world"),
    "france24": ("https://www.france24.com/en/rss", "world"),
    "cna": ("https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml", "world"),
    "japantimes": ("https://www.japantimes.co.jp/feed/", "world"),
    # --- 财经/商业 ---
    "cnbc-top": ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "business"),
    "marketwatch": (
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "business",
    ),
    "wsj-markets": ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "business"),
    "economist-finance": (
        "https://www.economist.com/finance-and-economics/rss.xml",
        "business",
    ),
    # --- 体育 ---
    "bbc-sport": ("https://feeds.bbci.co.uk/sport/rss.xml", "sports"),
    "skysports": ("https://www.skysports.com/rss/12040", "sports"),
    # --- 娱乐 ---
    "variety": ("https://variety.com/feed/", "entertainment"),
    "deadline": ("https://deadline.com/feed/", "entertainment"),
    # --- 健康 ---
    "stat-health": ("https://www.statnews.com/feed/", "health"),
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:  # pragma: no cover - defensive
        return default


def _env_list(name: str) -> list[str]:
    raw = _env(name)
    if not raw:
        return []
    if raw.strip().startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return []
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


class SourceConfig(BaseModel):
    """Declaration of a single news source."""

    name: str
    type: Literal["rss", "newsapi", "gnews", "mock"]
    enabled: bool = True
    weight: float = 1.0
    #: RSS feed templates. ``{query}``, ``{language}``, ``{hl}``, ``{gl}``,
    #: ``{ceid}`` placeholders are substituted per request.
    feeds: list[str] = Field(default_factory=list)
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float | None = None
    #: 栏目领域（tech/business/world/sports/entertainment/health）。搜索类源
    #: （Google/Bing/NewsAPI）没有固定领域，保持 None。
    topic: str | None = None

    @field_validator("feeds", mode="before")
    @classmethod
    def _coerce_feeds(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class LLMSettings(BaseModel):
    """LLM configuration (OpenAI compatible endpoint, e.g. vLLM)."""

    enabled: bool = True
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    timeout_s: float = 60.0
    max_retries: int = 2
    #: Max concurrent in-flight LLM requests (protects the inference server).
    concurrency: int = 4
    #: How many articles are packed into one structured-extraction request.
    batch_size: int = 6
    #: Max characters of article body sent to the model.
    max_excerpt_chars: int = 800
    #: Map-reduce summary: articles per map chunk.
    summary_chunk_size: int = 6
    #: LLM re-ranking: how many filtered candidates are sent to the judge model.
    rerank_max_candidates: int = 30
    #: LLM re-ranking: max characters of the article excerpt per candidate.
    rerank_max_excerpt_chars: int = 200
    structured_method: Literal["json_schema", "function_calling", "json_mode"] = (
        "json_schema"
    )

    @property
    def configured(self) -> bool:
        return bool(self.enabled and (self.api_key or self.base_url))

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"

    def resolved_base_url(self) -> str | None:
        return self.base_url or os.environ.get("OPENAI_BASE_URL")


class AuthSettings(BaseModel):
    """Inbound authentication for the HTTP / A2A endpoints.

    Static API-key based scheme: callers must present one of the configured
    keys either as ``Authorization: Bearer <key>`` or in the dedicated header
    (``X-API-Key`` by default).  Health probes (``/healthz``, ``/readyz``) and
    the Agent Card discovery paths (``/.well-known/*``) always stay public so
    orchestrators and A2A gateways can discover the agent.

    Configure via ``NEWS_AGENT_AUTH_ENABLED`` / ``NEWS_AGENT_AUTH_API_KEYS``
    (comma separated) / ``NEWS_AGENT_AUTH_API_KEY_HEADER``.
    """

    enabled: bool = False
    #: Valid API keys. An empty list disables enforcement even when ``enabled``.
    api_keys: list[str] = Field(default_factory=list)
    #: Header accepted as an alternative to ``Authorization: Bearer``.
    api_key_header: str = "X-API-Key"

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.api_keys)


class Settings(BaseModel):
    """Top level settings object."""

    # --- identity ---------------------------------------------------------
    agent_name: str = "news-agent"
    agent_version: str = "0.1.0"
    agent_description: str = (
        "News agent: multi-source news fetching, de-duplication, structured "
        "analysis (entities / events / sentiment / stance) and map-reduce "
        "summarisation, exposed over the A2A protocol."
    )
    agent_url: str = "http://localhost:9901"
    host: str = "0.0.0.0"
    port: int = 9901
    log_level: str = "INFO"

    # --- request defaults -------------------------------------------------
    default_language: str = "zh"
    default_limit: int = 15
    max_limit: int = 50
    default_mode: str = "summarize_news"
    #: 查询未携带时间表达时的默认抓取窗口（天）。0 表示不做时间过滤。
    default_window_days: int = 7

    # --- reliability ------------------------------------------------------
    task_timeout_s: float = 180.0
    fetch_timeout_s: float = 15.0
    source_concurrency: int = 8
    fetch_per_source_multiplier: float = 2.0
    max_articles_for_llm: int = 20

    # --- filtering --------------------------------------------------------
    relevance_threshold: float = 0.2
    dedup_title_ratio: float = 0.86
    dedup_simhash_distance: int = 3
    trends_top_n: int = 8
    #: LLM 精排：对通过阈值过滤的候选做一次相关性重排（需要 LLM 配置）。
    rerank_enabled: bool = True
    #: 来源多样性：限制单一来源占据结果的比例（limit/3，至少 3 篇）。
    source_diversity: bool = True
    #: 垃圾内容（博彩/SEO 站群/促销页）过滤开关与判定阈值。
    spam_filter_enabled: bool = True
    spam_threshold: float = 0.6

    # --- cache ------------------------------------------------------------
    cache_enabled: bool = True
    cache_path: str = ".cache/news_agent.sqlite3"
    cache_ttl_s: int = 900

    # --- misc -------------------------------------------------------------
    user_agents: list[str] = Field(default_factory=lambda: list(DEFAULT_USER_AGENTS))
    llm: LLMSettings = Field(default_factory=LLMSettings)
    sources: list[SourceConfig] = Field(default_factory=list)
    #: inbound authentication of the HTTP / A2A endpoints
    auth: AuthSettings = Field(default_factory=AuthSettings)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def default_sources(cls, *, extra_feeds: list[str] | None = None) -> list[SourceConfig]:
        sources = [
            SourceConfig(
                name="google-news",
                type="rss",
                feeds=[GOOGLE_NEWS_RSS],
                weight=1.0,
            ),
            SourceConfig(name="bing-news", type="rss", feeds=[BING_NEWS_RSS], weight=0.8),
        ]
        if _env_bool("TOPIC_FEEDS", True):
            for topic_name, (topic_feed, domain) in TOPIC_RSS_FEEDS.items():
                sources.append(
                    SourceConfig(
                        name=topic_name,
                        type="rss",
                        feeds=[topic_feed],
                        topic=domain,
                        weight=0.6,
                    )
                )
        if extra_feeds:
            sources.append(
                SourceConfig(name="custom-rss", type="rss", feeds=list(extra_feeds), weight=1.0)
            )
        api_key = _env("NEWSAPI_KEY")
        sources.append(
            SourceConfig(
                name="newsapi",
                type="newsapi",
                enabled=bool(api_key),
                api_key=api_key,
                base_url="https://newsapi.org/v2/everything",
                weight=1.2,
            )
        )
        gnews_key = _env("GNEWS_KEY")
        sources.append(
            SourceConfig(
                name="gnews",
                type="gnews",
                enabled=bool(gnews_key),
                api_key=gnews_key,
                base_url="https://gnews.io/api/v4/search",
                weight=1.2,
            )
        )
        return sources

    @classmethod
    def load(cls, env: dict[str, str] | None = None) -> "Settings":
        """Build settings from environment variables."""
        original = None
        if env is not None:
            original = dict(os.environ)
            os.environ.update({k: str(v) for k, v in env.items()})
        try:
            use_mock = _env_bool("USE_MOCK", False)
            if use_mock:
                sources = [SourceConfig(name="mock", type="mock", weight=1.0)]
            else:
                sources = cls.default_sources(extra_feeds=_env_list("EXTRA_RSS_FEEDS"))

            llm = LLMSettings(
                enabled=_env_bool("LLM_ENABLED", True),
                model=_env("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
                base_url=_env("LLM_BASE_URL"),
                api_key=_env("LLM_API_KEY"),
                temperature=_env_float("LLM_TEMPERATURE", 0.1),
                timeout_s=_env_float("LLM_TIMEOUT_S", 60.0),
                max_retries=_env_int("LLM_MAX_RETRIES", 2),
                concurrency=_env_int("LLM_CONCURRENCY", 4),
                batch_size=_env_int("LLM_BATCH_SIZE", 6),
                max_excerpt_chars=_env_int("LLM_MAX_EXCERPT_CHARS", 800),
                summary_chunk_size=_env_int("LLM_SUMMARY_CHUNK_SIZE", 6),
                rerank_max_candidates=_env_int("LLM_RERANK_MAX_CANDIDATES", 30),
                rerank_max_excerpt_chars=_env_int("LLM_RERANK_MAX_EXCERPT_CHARS", 200),
            )

            auth = AuthSettings(
                enabled=_env_bool("AUTH_ENABLED", False),
                api_keys=_env_list("AUTH_API_KEYS"),
                api_key_header=_env("AUTH_API_KEY_HEADER", "X-API-Key")
                or "X-API-Key",
            )

            return cls(
                agent_name=_env("AGENT_NAME", "news-agent") or "news-agent",
                agent_version=_env("AGENT_VERSION", "0.1.0") or "0.1.0",
                agent_url=_env("AGENT_URL", "http://localhost:9901") or "http://localhost:9901",
                host=_env("HOST", "0.0.0.0") or "0.0.0.0",
                port=_env_int("PORT", 9901),
                log_level=_env("LOG_LEVEL", "INFO") or "INFO",
                default_language=_env("DEFAULT_LANGUAGE", "zh") or "zh",
                default_limit=_env_int("DEFAULT_LIMIT", 15),
                max_limit=_env_int("MAX_LIMIT", 50),
                default_mode=_env("DEFAULT_MODE", "summarize_news") or "summarize_news",
                default_window_days=_env_int("DEFAULT_WINDOW_DAYS", 7),
                task_timeout_s=_env_float("TASK_TIMEOUT_S", 180.0),
                fetch_timeout_s=_env_float("FETCH_TIMEOUT_S", 15.0),
                source_concurrency=_env_int("SOURCE_CONCURRENCY", 8),
                fetch_per_source_multiplier=_env_float("FETCH_PER_SOURCE_MULTIPLIER", 2.0),
                max_articles_for_llm=_env_int("MAX_ARTICLES_FOR_LLM", 20),
                relevance_threshold=_env_float("RELEVANCE_THRESHOLD", 0.2),
                dedup_title_ratio=_env_float("DEDUP_TITLE_RATIO", 0.86),
                dedup_simhash_distance=_env_int("DEDUP_SIMHASH_DISTANCE", 3),
                trends_top_n=_env_int("TRENDS_TOP_N", 8),
                rerank_enabled=_env_bool("RERANK_ENABLED", True),
                source_diversity=_env_bool("SOURCE_DIVERSITY", True),
                spam_filter_enabled=_env_bool("SPAM_FILTER_ENABLED", True),
                spam_threshold=_env_float("SPAM_THRESHOLD", 0.6),
                cache_enabled=_env_bool("CACHE_ENABLED", True),
                cache_path=_env("CACHE_PATH", ".cache/news_agent.sqlite3")
                or ".cache/news_agent.sqlite3",
                cache_ttl_s=_env_int("CACHE_TTL_S", 900),
                llm=llm,
                sources=sources,
                auth=auth,
            )
        finally:
            if original is not None:
                os.environ.clear()
                os.environ.update(original)

    # ------------------------------------------------------------------
    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]

    def cache_file(self) -> Path:
        return Path(self.cache_path).expanduser()

    def resolve_agent_url(self) -> str:
        return self.agent_url.rstrip("/")


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Convenience wrapper around :meth:`Settings.load`."""
    return Settings.load(env)
