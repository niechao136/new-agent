# news-agent

一个通过 **A2A 协议**对外暴露能力的新闻 Agent：多源抓取 → 去重 → 相关性过滤 → 结构化分析
（实体/事件/情感/立场）→ map-reduce 话题级摘要 → 按 A2A 输出 schema 组装结果。

技术栈：**LangGraph**（分析子图）+ **a2a-sdk**（A2A 协议层）+ **FastAPI**（HTTP 承载）+ **Pydantic**（领域模型）。
协议层（Agent Card / 任务状态机 / JSON-RPC 与 REST 传输 / SSE 流式 / 客户端）**全部由官方
[`a2a-sdk`](https://pypi.org/project/a2a-sdk/) 提供**，本项目只实现新闻业务与二者的胶水层。

---

## 1. 特性一览

| 能力 | 说明 |
| --- | --- |
| 三个 A2A skill | `fetch_news`、`summarize_news`、`analyze_trend` |
| 标准协议实现 | Agent Card、任务生命周期、JSON-RPC 1.0 + 0.3 兼容、HTTP+JSON REST、SSE 订阅全部来自 `a2a-sdk` |
| 异步任务模型 | `submitted → working → completed / failed / canceled / rejected`，支持轮询、订阅、取消、列表 |
| 阶段进度推送 | 标准 `TaskStatusUpdateEvent` 携带 `{stage, kind, data}` 元数据（fetch/filter/analyze/summarize/format） |
| 多源抓取 | RSS（Google News / Bing News 搜索 + 23 个免 key 栏目源，覆盖科技/财经/国际/体育/娱乐/健康）、NewsAPI、GNews、内置 mock 源 |
| 统一源抽象 | `NewsSource.fetch(query, since, limit)`，新增源只需实现 `_fetch` |
| 质量过滤 | 剔除博彩/SEO 站群/促销页（品牌词 + 语义词累计 + 域名特征），避免"标题复述查询词"的垃圾页挤占结果 |
| 三重去重 | URL 规范化 → 标题规范化 → SimHash/标题相似度模糊匹配 |
| 相关性过滤 | 可解释的 token 重叠打分 + 连续匹配约束（抑制 CJK 跨界 bigram 误召回）+ 多语言关键词匹配（中英互查）；来源权重排序 + 来源配额去偏 |
| 泛化查询路由 | 「科技新闻」这类无具体对象的查询按来源栏目领域匹配（并给基础分），避免只有中文源能召回、英文源全被丢弃 |
| 可选 LLM 精排 | 对通过阈值的候选做一次结构化相关性重排，分数融合进排序；失败自动退回词面排序 |
| 结构化抽取 | LLM structured output（OpenAI 兼容 / vLLM），分 batch + 并发上限 |
| 长文本处理 | map-reduce 摘要，避免超长 prompt；分批小结失败可退化为拼接 |
| 双缓存 | SQLite 抓取缓存（TTL）+ 结果缓存 + 文章历史（增量识别新文章） |
| 稳定降级 | 源不可用/超时/LLM 失败/任务超时都返回**部分结果 + 结构化错误码**，绝不静默；`degraded` 只在结果真的受影响（无结果、多数源失败、LLM 降级）时置位，少数源失败仅告警 |
| 可观测性 | 每节点耗时、抓取成功率、LLM token 数、`/metrics` 指标快照 |
| 离线可跑 | 不配置任何 key（甚至无网络）也能用 mock 源 + 启发式分析完整跑通 |
| 容器化部署 | 多阶段 `Dockerfile`（依赖锁文件安装、非 root、内置健康检查）+ `docker-compose.yml` |

---

## 2. 目录结构

```
src/news_agent/
  config.py           # 环境变量驱动的配置（Settings / LLMSettings / SourceConfig）
  models.py           # 领域模型：RawArticle / AnalyzedArticle / TrendInsight / NewsResult / SkillRequest
  text_utils.py       # CJK 感知分词、URL 规范化、HTML 清洗
  runtime.py          # Metrics + RunContext（进度事件 / 阶段耗时 / 部分结果）
  dedup.py            # 去重（URL / 标题 / SimHash）
  quality.py          # 内容质量过滤（博彩/SEO 站群/促销页）
  intent.py           # 查询意图解析（主题关键词 + 时间窗口，LLM 优先 / 正则回退）
  relevance.py        # 相关性打分、栏目路由与筛选
  rerank.py           # 可选 LLM 精排（与词面分融合）
  trends.py           # 话题级趋势聚合
  cache.py            # SQLite 缓存 + 增量文章历史
  analyzer.py         # HeuristicAnalyzer / LLMAnalyzer / FallbackAnalyzer
  sources/            # NewsSource 抽象 + RSS / NewsAPI / GNews / Mock + 注册表
  graph/              # LangGraph 子图：state / nodes / builder / agent facade
  a2a/                # ← 基于 a2a-sdk 的协议适配层
    card.py           #   构建 SDK 的 AgentCard（含 JSON Schema 扩展）
    executor.py       #   AgentExecutor：RequestContext ⇄ SkillRequest ⇄ LangGraph
    server.py         #   组装 SDK 路由的 FastAPI app（+ 运维端点）
    client.py         #   SDK 客户端的便捷封装（CLI / examples 使用）
  cli.py              # serve | card | skills | sources | run | call | task | healthcheck
src/main.py           # 兼容入口（python src/main.py ...）
examples/call_news_agent.py   # 端到端 A2A 调用示例（SDK 客户端）
tests/                # 单元/集成测试（全部离线）
Dockerfile            # 多阶段镜像：uv 锁定依赖 → 自包含 venv，非 root 运行
docker-compose.yml    # 一键部署（健康检查、数据卷、环境变量透传）
.dockerignore         # 构建上下文裁剪（排除 .venv/.env/缓存/测试）
```

---

## 3. 快速开始

### 3.1 安装

```bash
pip install -e ".[dev]"        # 或：uv sync --extra dev
```

### 3.2 完全离线试跑（推荐先跑这个）

```bash
# 使用内置 mock 新闻源 + 启发式分析，不访问网络、不调用 LLM
python -m news_agent.cli run "人形机器人" --mock --no-llm --limit 6
python -m news_agent.cli run "人形机器人" --mock --no-llm --skill analyze_trend --json
```

### 3.3 启动 A2A 服务

```bash
python -m news_agent.cli serve --port 9901
# 或
uvicorn news_agent.a2a.server:app_factory --factory --host 0.0.0.0 --port 9901
```

```bash
curl http://localhost:9901/.well-known/agent-card.json     # Agent Card（A2A 1.0 路径）
curl http://localhost:9901/.well-known/agent.json          # Agent Card（0.3 兼容路径）
curl http://localhost:9901/healthz                         # 存活探针
curl http://localhost:9901/readyz                          # 就绪探针（会预热 agent）
curl http://localhost:9901/metrics                         # 指标快照
curl http://localhost:9901/skills                          # skill 清单 + 输出 JSON Schema

# 提交任务（A2A 1.0 JSON-RPC 方法名，需要 A2A-Version 头）
curl -s http://localhost:9901/ \
  -H 'content-type: application/json' -H 'A2A-Version: 1.0' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
  "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [
      {"data": {"skill": "summarize_news", "query": "固态电池", "limit": 10, "language": "zh"}}
  ]}}
}'

# 0.3 兼容写法的同一个任务（无需 A2A-Version 头）
curl -s http://localhost:9901/ -H 'content-type: application/json' -d '{
  "jsonrpc": "2.0", "id": 2, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "m2",
    "parts": [{"kind": "data", "data": {"query": "固态电池", "limit": 10}}]}}
}'

curl -s http://localhost:9901/ -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":3,"method":"GetTask","params":{"id":"<taskId>","historyLength":50}}'

# HTTP+JSON（REST）绑定
curl -s -X POST http://localhost:9901/message:send -H 'A2A-Version: 1.0' \
  -H 'content-type: application/json' \
  -d '{"message":{"messageId":"m3","role":"ROLE_USER","parts":[{"data":{"query":"固态电池"}}]}}'
```

### 3.4 用客户端调用（等价于其他 agent 的调用方式）

```bash
python -m news_agent.cli call "人形机器人" --skill analyze_trend --limit 5 --stream
python -m news_agent.cli task <taskId>              # 查看任务
python -m news_agent.cli task <taskId> --cancel     # 取消任务
python examples/call_news_agent.py --stream --list-tasks
```

也可以直接用 SDK 客户端：

```python
from a2a.client import ClientConfig, ClientFactory           # a2a-sdk
from a2a.helpers.proto_helpers import new_data_part, new_text_part
from a2a.types import a2a_pb2

card = ...  # A2ACardResolver 解析 /.well-known/agent-card.json
client = ClientFactory(ClientConfig(streaming=True)).create(card)
request = a2a_pb2.SendMessageRequest(message=a2a_pb2.Message(
    message_id="m1", role=a2a_pb2.ROLE_USER,
    parts=[new_data_part({"skill": "summarize_news", "query": "人形机器人"}, "application/json")],
))
async for event in client.send_message(request):   # StreamResponse
    ...                                            # task / statusUpdate / artifactUpdate
```

### 3.5 配置真实新闻源与 LLM

```bash
cp .env.example .env
export NEWS_AGENT_LLM_BASE_URL=http://localhost:8000/v1   # vLLM / OpenAI / DeepSeek ...
export NEWS_AGENT_LLM_API_KEY=sk-xxx
export NEWS_AGENT_LLM_MODEL=Qwen2.5-72B-Instruct
export NEWS_AGENT_NEWSAPI_KEY=xxx        # 可选，启用 NewsAPI
export NEWS_AGENT_GNEWS_KEY=xxx          # 可选，启用 GNews
```

> 默认源是 Google News / Bing News 的 RSS 搜索，**无需 API key**。
> 不配置 LLM 时自动走启发式分析，接口与输出 schema 完全一致，只是 `metrics.analyzer` 变为 `heuristic`。

### 3.6 Docker 部署

镜像两阶段构建：第一段用 `uv` 按 `uv.lock` 把依赖装进自包含的 `/opt/venv`，
第二段只拷贝这个 venv（不带编译器、uv、测试代码），以 UID 10001 非 root 运行。

builder 阶段的层顺序是刻意安排的，改代码时不会触发依赖重装：

| 层 | 输入 | 何时失效 |
| --- | --- | --- |
| 依赖层 | `pyproject.toml` + `uv.lock` | 只有增删/升级依赖时（≈25s） |
| 项目层 | `README.md` + `src/` | 改源码或文档时（≈3s） |

`README.md` 必须在项目层：`pyproject.toml` 把它声明为项目 readme，放进依赖层会导致
「改一行文档就重新下载 67 个依赖」。

```bash
# 构建 + 运行
docker build -t news-agent:0.1.0 .
docker run --rm -p 9901:9901 \
  -e NEWS_AGENT_AGENT_URL=http://localhost:9901 \
  -v news-agent-data:/data \
  news-agent:0.1.0

# 或使用 compose（同目录存在 .env 时自动用于 ${VAR:-default} 替换）
cp .env.example .env          # 可选：填 LLM / 新闻源 key
docker compose up --build -d
docker compose logs -f news-agent
docker compose down           # 加 -v 连数据卷一起删
```

容器内的约定：

| 项 | 值 |
| --- | --- |
| 监听地址 | `NEWS_AGENT_HOST=0.0.0.0`、`NEWS_AGENT_PORT=9901`（改这两个变量即可换端口，无需重建镜像） |
| 缓存与历史 | `NEWS_AGENT_CACHE_PATH=/data/news_agent.sqlite3`，用卷 `news-agent-data` 持久化 |
| 运行用户 | `app`（UID/GID 10001），`/data` 归其所有 |
| 健康检查 | `news-agent healthcheck` 探 `/healthz`；设 `NEWS_AGENT_HEALTHCHECK_ARGS=--ready` 可改用 `/readyz`（会预热 agent） |
| 启动命令 | `news-agent serve`（uvicorn，收到 SIGTERM 优雅退出） |

常用操作：

```bash
docker compose exec news-agent news-agent card          # 查看容器内的 Agent Card
docker compose exec news-agent news-agent healthcheck   # 手动探活
curl -s localhost:9901/.well-known/agent-card.json | head -c 200

# 完全离线冒烟（mock 源 + 启发式分析，不访问网络与 LLM）
docker compose run --rm -e NEWS_AGENT_USE_MOCK=1 -e NEWS_AGENT_LLM_ENABLED=0 \
  news-agent run "人形机器人" --limit 5

# 在宿主机上调用容器里的 agent
python -m news_agent.cli call "固态电池" --base-url http://localhost:9901
```

> **`NEWS_AGENT_AGENT_URL` 必须是调用方能访问的地址**：它是 Agent Card 里
> `supportedInterfaces[].url` 的值，标准 A2A 客户端（含 `a2a-sdk`）会按它发起后续调用。
> 容器里不要写 `127.0.0.1`（跨容器的调用方会连到它自己），生产环境应写对外域名。
> 本项目自带的 `news-agent call --base-url ...` 会在本地覆盖卡片地址，方便端口映射/调试场景。

部署注意：

- 镜像按**单副本**设计：任务状态默认在进程内存（SDK `InMemoryTaskStore`）。
  多副本需要换成 SDK 的 `DatabaseTaskStore` 并让所有副本共享同一数据库（见 §10）；
- 多架构构建：`docker buildx build --platform linux/amd64,linux/arm64 -t news-agent:0.1.0 --push .`
  （依赖均有对应 wheel，镜像内不需要编译工具链）；
- 依赖层只在 `pyproject.toml` / `uv.lock` 变化时重建，改业务代码只重建项目层。

---

## 4. 配置项

所有变量以 `NEWS_AGENT_` 为前缀（`.env.example` 有完整清单）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `AGENT_NAME` / `AGENT_VERSION` / `AGENT_URL` | news-agent / 0.1.0 / http://localhost:9901 | Agent Card 身份信息；`AGENT_URL` 同时是 card 的 `supportedInterfaces[].url`，**容器/网关环境必须设为调用方可达的地址** |
| `HOST` / `PORT` | 0.0.0.0 / 9901 | 服务监听地址 |
| `DEFAULT_LANGUAGE` / `DEFAULT_LIMIT` / `MAX_LIMIT` | zh / 15 / 50 | 请求默认值与上限 |
| `DEFAULT_MODE` | summarize_news | 未显式指定 skill 时的默认技能 |
| `TASK_TIMEOUT_S` | 180 | 单任务最大执行时间（超时返回部分结果） |
| `FETCH_TIMEOUT_S` | 15 | 单个源单次请求超时 |
| `SOURCE_CONCURRENCY` | 8 | 并发源数量上限（源较多时决定抓取总耗时） |
| `RELEVANCE_THRESHOLD` | 0.2 | 相关性阈值 |
| `RERANK_ENABLED` | 1 | 是否启用 LLM 精排（需配置 LLM，否则自动跳过） |
| `SOURCE_DIVERSITY` | 1 | 是否限制单一来源占比（上限 `ceil(limit/3)`，至少 3 篇） |
| `SPAM_FILTER_ENABLED` / `SPAM_THRESHOLD` | 1 / 0.6 | 垃圾内容（博彩/SEO 站群/促销页）过滤开关与阈值 |
| `TOPIC_FEEDS` | 1 | 是否启用 23 个免 key 栏目源（关闭后只剩搜索类源） |
| `MAX_ARTICLES_FOR_LLM` | 20 | 送 LLM 的文章上限，超出部分走启发式 |
| `LLM_RERANK_MAX_CANDIDATES` / `LLM_RERANK_MAX_EXCERPT_CHARS` | 30 / 200 | 精排候选数与每篇摘要字符上限 |
| `CACHE_ENABLED` / `CACHE_PATH` / `CACHE_TTL_S` | 1 / .cache/news_agent.sqlite3 / 900 | 抓取缓存 |
| `LLM_ENABLED` / `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | 1 / gpt-4o-mini / - / - | OpenAI 兼容端点（兼容 `OPENAI_API_KEY` / `OPENAI_BASE_URL`） |
| `LLM_BATCH_SIZE` / `LLM_CONCURRENCY` / `LLM_SUMMARY_CHUNK_SIZE` | 6 / 4 / 6 | 批大小、并发上限、map-reduce 分块 |
| `USE_MOCK` | 0 | 强制使用确定性 mock 源（离线） |
| `EXTRA_RSS_FEEDS` | - | 追加自定义 RSS（含 `{query}` 占位符即变为关键词搜索） |

---

## 5. A2A 契约（由 `a2a-sdk` 实现）

### 5.1 Agent Card

`GET /.well-known/agent-card.json`（A2A 1.0）与 `GET /.well-known/agent.json`（0.3 兼容）返回同一张卡片：

```jsonc
{
  "name": "news-agent",
  "description": "...",
  "version": "0.1.0",
  "supportedInterfaces": [
    { "url": "http://localhost:9901", "protocolBinding": "JSONRPC",   "protocolVersion": "1.0" },
    { "url": "http://localhost:9901", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0" }
  ],
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "extensions": [
      {
        "uri": "https://news-agent.dev/a2a/skill-schemas",
        // A2A 1.0 的 AgentSkill 没有 schema 字段，因此把 JSON Schema 挂在扩展上
        "params": { "requestSchema": { ... }, "resultSchema": { ... } }
      },
      { "uri": "https://news-agent.dev/a2a/task-progress", "description": "statusUpdate.metadata = {stage, kind, data}" }
    ]
  },
  "defaultInputModes": ["application/json", "text/plain"],
  "defaultOutputModes": ["application/json", "text/plain"],
  "skills": [
    { "id": "fetch_news",     "name": "Fetch news",     "tags": [...], "examples": [...] },
    { "id": "summarize_news", "name": "Summarize news", "tags": [...], "examples": [...] },
    { "id": "analyze_trend",  "name": "Analyze trend",  "tags": [...], "examples": [...] }
  ]
}
```

请求参数与结果对象的 JSON Schema 也通过 `GET /skills` 直接暴露。

### 5.2 三个 skill

| skill | 做什么 | 是否调用 LLM | 特有输出 |
| --- | --- | --- | --- |
| `fetch_news` | 抓取 + 去重 + 相关性过滤，只返回元数据 | 否（最快） | 无 `summary`，`articles` 仅含 `relevance` + 元数据 |
| `summarize_news` | 全流程 + 结构化抽取 + 聚合摘要 | 是（可降级） | `summary`、`articles[].entities/events/sentiment/stance/key_points` |
| `analyze_trend` | 在 `summarize_news` 基础上做话题聚合 | 是（可降级） | `trends[]`（topic/mentions/sentiment/keywords/代表性链接） |

**输入参数**——三种等价写法（服务端都会解析）：

1. `DataPart`：`{"kind":"data","data":{"query":"...","skill":"...","limit":10,...}}`
2. 纯文本：`"summarize_news: 固态电池"` 或直接 `"固态电池"`
3. 请求 `metadata`：`{"metadata":{"query":"固态电池","skill":"fetch_news"}}`

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `query` | string | 必填 | 关键词 / 领域 / 话题 |
| `skill` | enum | `summarize_news` | `fetch_news` / `summarize_news` / `analyze_trend` |
| `since` / `until` | ISO 8601 | - | 时间窗口 |
| `keywords` | string[] | LLM 解析结果 | 相关性匹配用的扩展关键词（同义词/英文译名），不改变抓取请求 |
| `limit` | int | 15 | 返回文章数上限（服务端 clamp 到 `MAX_LIMIT`） |
| `language` | string | zh | 结果与摘要语言 |
| `sources` | string[] | 全部启用源 | 指定新闻源 |
| `threshold` | float | 服务端配置 | 覆盖相关性阈值 |

**输出**：任务产物 `news-result`，包含两个 part：

* `DataPart`：完整的 `NewsResult`（JSON）
* `TextPart`：摘要纯文本，方便只支持文本的调用方

```jsonc
{
  "query": "固态电池", "mode": "summarize_news", "language": "zh",
  "generated_at": "2026-09-12T01:06:41Z", "duration_ms": 3421, "degraded": false,
  "counts": { "fetched": 42, "duplicates_removed": 9, "after_dedup": 33,
              "selected": 10, "analyzed": 10, "sources_used": 3, "errors": 0, "warnings": 0 },
  "summary": "……话题级聚合摘要……",
  "articles": [
    { "id": "0f4a488a3a728685", "title": "……", "url": "https://…", "source": "……",
      "published_at": "2026-09-11T22:10:00Z", "relevance": 0.92,
      "sentiment": "positive", "sentiment_score": 0.75, "stance": "支持/看好",
      "entities": ["…"], "events": ["…"], "key_points": ["…"], "summary": "……",
      "duplicate_sources": ["…"] }
  ],
  "trends": [ { "topic": "固态电池", "mentions": 7, "sentiment": "positive",
                "average_sentiment": 0.42, "keywords": ["…"], "representative_urls": ["…"] } ],
  "warnings": [], "errors": [],
  "timings_ms": { "fetch": 1180.2, "filter": 12.4, "analyze": 2103.7, "summarize": 96.1, "format": 1.1 },
  "metrics": { "analyzer": "llm+heuristic", "articles_fetched": 42, "llm_prompt_tokens": 8123 }
}
```

> A2A 的 `Part.data` 是 `google.protobuf.Value`，数字统一为 double；
> 用 `news_agent.a2a.client.result_from_task(task)` 读取时会自动把整数值还原成 int。

### 5.3 端点与方法

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/.well-known/agent-card.json` · `/.well-known/agent.json` | Agent Card |
| POST | `/` | JSON-RPC 2.0（见下表方法名） |
| POST | `/message:send` · `/message:stream` | HTTP+JSON 绑定 |
| GET / POST | `/tasks/{id}` · `/tasks/{id}:subscribe` · `/tasks/{id}:cancel` | HTTP+JSON 绑定 |
| GET | `/tasks` | 任务列表 |
| GET | `/healthz` · `/readyz` · `/metrics` · `/skills` · `/docs` | 运维 / 自描述（本项目新增，注册在 SDK 路由之前） |

JSON-RPC 方法名（**1.0 使用 PascalCase，需要 `A2A-Version: 1.0` 请求头**；同时开启 0.3 兼容名）：

| A2A 1.0 | A2A 0.3 兼容 | 说明 |
| --- | --- | --- |
| `SendMessage` | `message/send` | 提交任务（非流式，等待终态后返回 Task） |
| `SendStreamingMessage` | `message/stream` | 提交任务并以 SSE 返回事件流 |
| `GetTask` | `tasks/get` | 查询任务 |
| `ListTasks` | - | 任务列表 |
| `CancelTask` | `tasks/cancel` | 取消任务 |
| `SubscribeToTask` | `tasks/resubscribe` | 重新订阅运行中任务的事件流 |
| `GetExtendedAgentCard` | `agent/getAuthenticatedExtendedCard` | 扩展卡片（未配置时返回协议错误） |

### 5.4 任务状态机与流式事件

```
submitted ──▶ working ──▶ completed
                   │
                   ├──▶ failed        （没有任何可用结果 / 内部错误）
                   ├──▶ canceled      （调用方 CancelTask）
                   ├──▶ rejected      （参数非法，如缺 query / 未知 skill）
                   └──▶ completed(degraded=true)  （超时或降级，仍带部分结果）
```

JSON-RPC 里状态值是 proto 枚举名（`TASK_STATE_COMPLETED` 等）；用 `state_name(state)` 可得到 `completed`。

SSE 流中的事件类型（`StreamResponse` 的 oneof）：

| 事件 | 内容 |
| --- | --- |
| `task` | 任务快照（首个事件；`SendStreamingMessage` 会带上） |
| `statusUpdate` | 状态流转；**working 阶段携带 `metadata = {stage, kind, data}`**，`stage ∈ fetch/filter/analyze/summarize/format`，`message.parts[].text` 是人类可读进度 |
| `artifactUpdate` | `news-result` 产物（data + text 两个 part） |

### 5.5 错误与降级契约

**协议层错误**由 SDK 按 A2A/JSON-RPC 规范返回：

| 场景 | 表现 |
| --- | --- |
| 未知方法 | JSON-RPC error `-32601 MethodNotFound` |
| 参数不合法 | `-32602` / `InvalidParamsError`（`GetTask` 校验等） |
| 任务不存在 | `TaskNotFoundError` → `-32001` |
| 任务不可取消 | `TaskNotCancelableError` → `-32002` |
| 版本不匹配 | `VersionNotSupportedError`（未带 `A2A-Version` 时按 0.3 处理） |
| 请求缺 `query` / skill 未知 | 任务被 **rejected**，`status.message` 里同时给出文本说明与结构化 `{"error":"invalid_request","validSkills":[...]}` |

**业务层错误**放在产物 `NewsResult.errors[]`，使用稳定错误码，调用方据此决策：

| code | 含义 | retryable |
| --- | --- | --- |
| `invalid_request` | 参数非法 | 否 |
| `no_results` | 所有源都没有结果（任务为 `failed`） | 是 |
| `source_unavailable` / `fetch_timeout` / `rate_limited` / `parse_error` | 单个源失败（不影响其他源） | 多数为是 |
| `llm_failed` / `llm_timeout` / `context_too_long` | LLM 阶段失败（已降级为启发式） | 是 |
| `cache_error` | 缓存不可用（已跳过缓存继续执行） | 是 |
| `task_timeout` | 超过 `TASK_TIMEOUT_S`（返回部分结果，`degraded=true`） | 是 |
| `task_canceled` / `internal_error` | 取消 / 内部异常 | - |

降级策略（保证调用方永远拿到结构稳定的响应）：

1. **单源失败** → 记录 `errors[]`，继续用其余源的结果；
2. **抓取结果为空** → `no_results` + 空 `articles`，任务状态 `failed`（不抛异常、不挂起）；
3. **LLM 不可用** → 启发式抽取/摘要兜底，`warnings[]` 说明，`degraded=true`；
4. **LLM 部分 batch 失败** → 失败批次的文章用启发式补齐，其余保持 LLM 结果；
5. **摘要 reduce 失败** → 退化为拼接分批小结；
6. **整体超时** → 返回已完成的阶段结果（`RunContext.partial`），`degraded=true`；
7. **缓存故障** → 静默跳过缓存，不影响主流程。

---

## 6. LangGraph 子图

```
START ─▶ fetch ─▶ filter ─┬─▶ (skill = fetch_news / 无结果) ─────────────▶ format ─▶ END
                          └─▶ analyze ─▶ summarize ─────────────────────▶ format
```

| 节点 | 职责 |
| --- | --- |
| `fetch_node` | 并发调用多个 `NewsSource`（`asyncio.gather` + 并发上限 + 每源超时 + tenacity 重试）；命中缓存则跳过抓取；写入增量历史 |
| `filter_node` | 质量过滤（博彩/SEO 站群/促销页）→ 三重去重 → 相关性打分（连续匹配约束 + 多关键词 + 泛化查询栏目路由）→ 来源权重排序 → 来源配额去偏 → 阈值过滤 → 数量截断（低于阈值时按 best-effort 补齐并告警）→ 可选 LLM 精排；分数写入 state 供下游复用 |
| `analyze_node` | 分批 + 并发受限的 LLM structured output（实体/事件/情感/立场/要点/摘要）；超预算与失败批次用启发式补齐 |
| `summarize_node` | map-reduce：分批小结 → 合并综述（避免一次性塞进 context） |
| `format_node` | 组装 `NewsResult`（含 trends、counts、timings、metrics），写入 `RunContext.partial` |

State（`NewsState`）承载业务数据；**进度、耗时、错误、部分结果**放在 `RunContext`，
即使图被超时中断也能取回已完成阶段的结果。执行器把它桥接到 A2A：

```
RequestContext ──parse──▶ SkillRequest ──▶ LangGraph ──▶ NewsResult
                                                          │
RunContext.emit(...) ──▶ TaskStatusUpdateEvent(working, metadata={stage,...})
                                                          ▼
                                            TaskArtifactUpdateEvent(news-result)
                                                          ▼
                                            TaskStatusUpdateEvent(completed/failed)
```

---

## 7. 可观测性

- 结构化日志：`news_agent.run` 打印每个阶段的开始/结束、每源返回数量、每阶段耗时；
- `RunContext.timings` → `NewsResult.timings_ms`（fetch/filter/analyze/summarize/format）；
- `Metrics` 计数器/直方图：`articles_fetched`、`articles_deduped`、`fetch_cache_hit`、
  `articles_new_since_last_run`、`llm_prompt_tokens`、`llm_completion_tokens`、`stage_ms.*`、
  `a2a_tasks_submitted/completed/failed/rejected/canceled/timeouts`；
- `GET /metrics` 返回 counters + histogram 分位数 + 任务/缓存统计。

---

## 8. 测试与类型检查

```bash
python -m pytest -q          # 68 passed（全部离线、无需网络与 API key）
basedpyright                 # 0 errors, 0 warnings（uv run basedpyright）
```

**类型检查策略**（配置见 `pyproject.toml` 的 `[tool.basedpyright]`）：

* 使用 `standard` 模式 —— 参数/返回值/赋值类型不匹配、Optional 访问、未绑定变量、
  未使用的导入与变量、已弃用 API 等**能抓真实缺陷的规则全部开启，当前为 0 error**；
* 关闭的仅是"类型标注完备性"类规则（`reportExplicitAny`、`reportUnknown*`、
  `reportMissingTypeStubs` 等）：本项目需要与动态类型的三方 API 互操作
  （LangGraph 编译图与 LangChain Runnable、protobuf `Struct` 的任意 JSON 负载、
  部分三方库缺少存根），全量标注只会引入大量无意义的 `Any`。
* 源与 HTTP 客户端、分析器都通过 `Protocol` 描述（`HttpClientProtocol`、
  `Analyzer`），因此测试替身无需 `cast`/`type: ignore` 即可通过类型检查。

覆盖点：URL/SimHash 去重、相关性打分、SQLite 缓存 TTL 与增量、RSS/NewsAPI/GNews 源解析与错误分类、
map-reduce 分块与降级、批处理 + 启发式补齐、启发式分析、图端到端（三个 skill、缓存命中、
空结果降级、排序稳定性）、A2A Agent Card（含双 well-known 路径与 schema 扩展）、A2A 1.0 JSON-RPC、
0.3 兼容方法名、HTTP+JSON REST、SSE 事件流与阶段元数据、参数校验/unknown method/task not found、
SDK 客户端端到端（流式进度 + 产物读取 + 任务列表）、executor 超时部分结果。

---

## 9. 接入多 Agent 平台（TODO items 20–22）

**20. 注册为可发布的自定义 agent**

- 用 `news-agent serve`（或容器化）暴露 A2A 服务，网关侧配置：
  - Agent Card：`<base>/.well-known/agent-card.json`（若网关仍用 0.3，可读 `<base>/.well-known/agent.json`）
  - JSON-RPC：`<base>/`；HTTP+JSON：`<base>/message:send`、`<base>/tasks/{id}`
  - 订阅：`<base>/tasks/{id}:subscribe` 或 JSON-RPC `SubscribeToTask`
- 在 admin center 里登记为自定义 agent，并在统一路由里绑定上述 endpoint；
  平台侧工具只需把 `skill + 参数` 放进 `DataPart` 即可。

**21. 是否需要 A2A 可发现**

本 agent 是标准 A2A 1.0 server（card 由 SDK 生成，`supportedInterfaces` 同时声明
`JSONRPC` 与 `HTTP+JSON`），可以直接接入网关的发现流程。需要对齐的点：

1. 调用 v1.0 方法名时带上 `A2A-Version: 1.0`（缺省按 0.3 处理）；
2. 若网关仍使用 0.3 名称，服务端已开启 `enable_v0_3_compat`，无需额外配置；
3. 任务状态使用 `TASK_STATE_*` 枚举；结果从 `artifacts[].parts[]` 读取（`data` + `text` 双形态）；
4. `contextId` 透传：调用方在 message 上带 `contextId`，服务端会沿用（同一上下文的任务可被 `ListTasks` 按 context 过滤）。

**22. 调用示例**

`examples/call_news_agent.py` 是端到端示例：解析 Card → `SendMessage` 流式消费
（task → working/进度 → artifact → completed）→ 读取 `news-result` → `GetTask` / `ListTasks`。
其他 agent 只要复用 `a2a-sdk` 的 `ClientFactory`（或本项目 `NewsA2AClient` 的写法）即可。

---

## 10. 后续可扩展点

- **任务持久化**：`InMemoryTaskStore` 换成 SDK 自带的 `DatabaseTaskStore`（`a2a-sdk[db]`，SQLAlchemy），
  仅需在 `a2a/server.py` 替换一行；多副本部署时再配合 `DatabasePushNotificationConfigStore`；
- **推送通知**：配置 `PushNotificationSender` 即可支持 `capabilities.pushNotifications`；
- **交互式追问**：`TaskUpdater.requires_input()` + 续跑同一 task（`RequestContext.current_task`）；
- 用 embedding 相似度替换 `relevance.py` 的关键词打分（`rank_articles` 是替换点）；
- 把 `SqliteCache` 换成 Postgres 以支持多实例共享缓存；
- 为 `analyze_node` 增加按域名的抽取模板（财经/科技/政策）；
- 为卡片补 `signatures`（`a2a.utils.signing`）以便网关做来源校验。
