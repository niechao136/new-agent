基于你的技术栈（LangGraph + FastAPI）以及正在做的多Agent平台（A2A 网关架构），这个新闻 agent 应该设计成一个**独立的 LangGraph 子图/agent**，通过 A2A 协议暴露 skill，可被你平台里的其他 agent（包括默认 agent 和自定义 agent）调用。下面是具体的 TODO 步骤。

## 一、需求与接口设计（先定契约，再写代码）

1. 明确 Agent Card：定义这个新闻 agent 对外暴露的 skill（如 `fetch_news`、`summarize_news`、`analyze_trend`），包括每个 skill 的输入参数（关键词/领域/时间范围/语言/数量）和输出 schema（结构化 JSON：标题、来源、发布时间、摘要、情感/立场标签、原文链接）。
2. 确定调用模式：同步返回 vs 流式返回（新闻抓取+分析可能耗时较长，建议走 A2A 的 task 异步模式，支持 `submitted → working → completed` 状态，而不是一次性阻塞响应）。
3. 定义错误与降级契约：抓取失败、源不可用、内容过长超 context 时的返回格式，保证调用方 agent 能稳定处理。

## 二、数据抓取层

4. 选定新闻源：RSS（feedparser）、新闻聚合 API（如 NewsAPI/Bing News/GNews）、或指定站点爬虫，建议优先用结构化 API/RSS，降低反爬维护成本。
5. 实现统一的 `NewsSource` 抽象接口（`fetch(query, since, limit) -> List[RawArticle]`），每个具体源实现该接口，方便后续增减源。
6. 加入去重逻辑（按 URL / 标题相似度 / simhash）避免同一事件多源重复。
7. 加入抓取限流与重试（tenacity）、超时控制、User-Agent 管理。
8. 设计原始数据缓存层（Redis 或本地 SQLite/Postgres），按 query+时间窗口做缓存，避免重复抓取同一批新闻，同时作为增量更新的基础。

## 三、LangGraph 分析流程设计

9. 用 LangGraph 建子图，节点建议拆成：
   - `fetch_node`：并行调用多个 `NewsSource`，聚合原始文章
   - `filter_node`：去重、相关性过滤（可用 embedding 相似度或关键词打分剔除噪声）
   - `analyze_node`：对每篇/每批文章做结构化抽取（LLM structured output：实体、事件、情感、立场），可参考你已有的 [[form-filling-pipeline]] 里 LangChain + vLLM 结构化输出的经验
   - `summarize_node`：对分析结果做聚合摘要（单篇摘要 + 多篇话题级汇总，可用 map-reduce 或 refine 策略处理长文本）
   - `format_node`：按 A2A 输出 schema 组装最终结果
10. 定义 State：包含 query、raw_articles、filtered_articles、analyzed_results、final_summary，以及错误/进度字段。
11. 长文本/多篇文章场景下用 map-reduce 摘要而非一次性塞进 context，避免超长 prompt。
12. 如果新闻量大，考虑对 `analyze_node` 做批处理 + 并发（asyncio.gather），控制并发数防止打爆 LLM 服务。

## 四、A2A 服务封装

13. 用 FastAPI 实现 A2A server 端点（`/.well-known/agent.json` 返回 Agent Card；`/tasks/send` 或对应 A2A 版本的任务接口）。
14. 把 LangGraph 子图包装成 task executor：接收 A2A 请求 → 转换为 LangGraph 输入 state → 执行 → 转换为 A2A 输出消息。
15. 实现任务状态管理（in-memory dict 或 Redis），支持调用方轮询/订阅任务进度，尤其适合新闻抓取这种非实时场景。
16. 如需要流式，支持 SSE 推送阶段性结果（如"已抓取N篇"→"分析完成"→"摘要生成中"）。

## 五、可靠性与可观测性

17. 加日志埋点（每个节点耗时、抓取成功率、LLM 调用 token 数），方便后续排查慢查询。
18. 加超时兜底：整体任务设置最大执行时间，超时返回部分结果而不是无响应。
19. 加简单的单元测试：mock 新闻源响应，测试 filter/summarize 节点逻辑；集成测试跑一次真实小规模抓取。

## 六、接入你的多Agent平台

20. 按照你平台的"自定义 agent 通过 admin center 发布并绑定 A2A"的架构，把新闻 agent 注册为一个可发布的自定义 agent，走统一路由配置。
21. 决定这个 agent 是否需要被 A2A 发现（对应你之前记录的开放问题"custom agents 是否也应该是 A2A 可发现的 server"），如果是，确保 Agent Card 格式与你网关约定一致。
22. 编写调用示例：从默认 agent 或另一个自定义 agent 通过 A2A 工具调用这个新闻 agent，验证端到端链路。

---

## 实现对照表

| # | 完成情况 | 落地位置 |
| --- | --- | --- |
| 1 | ✅ Agent Card 定义 3 个 skill + 输入/输出 JSON Schema | `src/news_agent/a2a/card.py`、`models.py`（`SkillRequest` / `NewsResult`） |
| 2 | ✅ 异步 task 模式 `submitted → working → completed/failed/canceled` | `a2a/executor.py`、`a2a/store.py`、`a2a/models.py` |
| 3 | ✅ 错误与降级契约（稳定错误码 + 永不抛栈） | `models.py::ErrorCode`、README §5.5 |
| 4 | ✅ 新闻源：RSS（Google/Bing，免 key）+ NewsAPI + GNews + mock | `sources/rss.py`、`sources/api.py`、`sources/mock.py` |
| 5 | ✅ `NewsSource.fetch(query, since, limit)` 统一抽象 + 注册表 | `sources/base.py`、`sources/registry.py` |
| 6 | ✅ 去重：URL 规范化 / 标题规范化 / SimHash + 标题相似度 | `dedup.py` |
| 7 | ✅ 限流重试（tenacity）、超时（每源 + 整体）、User-Agent 轮换 | `sources/base.py`、`sources/http.py` |
| 8 | ✅ SQLite 缓存（query+时间窗，TTL）+ 文章历史（增量新文章） | `cache.py` |
| 9 | ✅ LangGraph 子图：fetch / filter / analyze / summarize / format | `graph/nodes.py`、`graph/builder.py` |
| 10 | ✅ State 定义 | `graph/state.py`（业务数据入 State，进度/耗时可观测数据放 `RunContext`） |
| 11 | ✅ map-reduce 摘要（分批小结 → 合并综述，reduce 失败退化为拼接） | `analyzer.py::LLMAnalyzer.summarize` |
| 12 | ✅ 分析批处理 + `asyncio.gather` + 并发信号量（LLM 预算可配） | `analyzer.py::LLMAnalyzer.analyze`、`sources/http.py::gather_limited` |
| 13 | ✅ FastAPI A2A server：Agent Card + 任务接口，**协议层全部由 `a2a-sdk` 提供**（`A2AFastAPI` 路由 / `DefaultRequestHandler` / JSON-RPC 1.0 + 0.3 兼容 / HTTP+JSON） | `a2a/server.py`、`a2a/card.py` |
| 14 | ✅ 子图包装为 SDK 的 `AgentExecutor`（`RequestContext` ⇄ `SkillRequest` ⇄ LangGraph） | `a2a/executor.py`、`graph/agent.py` |
| 15 | ✅ 任务状态管理使用 SDK 的 `InMemoryTaskStore` + `DefaultRequestHandler`，轮询/订阅/取消/列表均为标准能力 | `a2a/server.py`（换 `DatabaseTaskStore` 即可持久化） |
| 16 | ✅ SSE 阶段进度推送：SDK 事件队列 + 标准 `TaskStatusUpdateEvent`（`metadata={stage,kind,data}`） | `a2a/executor.py::_publish_progress` |
| 17 | ✅ 日志埋点：节点耗时、抓取成功率、LLM token 数、`/metrics` | `runtime.py`、`graph/nodes.py` |
| 18 | ✅ 超时兜底：返回已完成阶段的部分结果（`RunContext.partial`） | `a2a/executor.py::_timeout_result` |
| 19 | ✅ 68 个单元/集成测试（mock 源响应、filter/summarize 节点、A2A 协议与 SDK 客户端端到端） | `tests/` |
| 20 | ◑ 平台侧注册/绑定 A2A 需在 admin center 操作（代码侧契约与文档已就绪） | README §9-20 |
| 21 | ✅ Agent Card 由 SDK 生成（`supportedInterfaces` / `capabilities` / `skills`），同时暴露 1.0 与 0.3 well-known 路径，兼容 PascalCase 与 `message/send` 两套方法名 | `a2a/card.py`、README §9-21 |
| 22 | ✅ 端到端调用示例（发现 Card → 提交 → 轮询 → SSE → 读 artifact） | `examples/call_news_agent.py`、`a2a/client.py`、`news-agent call` |

验证方式：

```bash
python -m pytest -q                                  # 64 passed，全程离线
python -m news_agent.cli run "人形机器人" --mock --no-llm --limit 6
python -m news_agent.cli serve --port 8080           # 另开终端跑 examples/call_news_agent.py
```
