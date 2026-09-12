"""news-agent: an A2A-exposed news agent built on LangGraph + FastAPI.

The package is organised in the following layers:

* :mod:`news_agent.sources`    -- pluggable news acquisition backends
* :mod:`news_agent.dedup`      -- cross-source de-duplication
* :mod:`news_agent.relevance`  -- relevance scoring / noise filtering
* :mod:`news_agent.cache`      -- SQLite cache + incremental article history
* :mod:`news_agent.llm`        -- LLM analyser (with heuristic degradation)
* :mod:`news_agent.graph`      -- the LangGraph sub-graph (the actual agent)
* :mod:`news_agent.a2a`        -- A2A server, agent card, task executor
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
