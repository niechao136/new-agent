"""Assembles the LangGraph sub-graph (TODO item 9)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import NewsGraphNodes
from .state import NewsState

NODE_FETCH = "fetch"
NODE_FILTER = "filter"
NODE_ANALYZE = "analyze"
NODE_SUMMARIZE = "summarize"
NODE_FORMAT = "format"


def build_news_graph(nodes: NewsGraphNodes) -> Any:
    """Compile the news sub-graph.

    ``fetch -> filter -> (analyze -> summarize)? -> format``

    ``fetch_news`` short-circuits straight to ``format`` after the filter stage;
    the analysis/summarisation branch is only taken by ``summarize_news`` and
    ``analyze_trend``.
    """
    graph = StateGraph(NewsState)
    graph.add_node(NODE_FETCH, nodes.fetch_node)
    graph.add_node(NODE_FILTER, nodes.filter_node)
    graph.add_node(NODE_ANALYZE, nodes.analyze_node)
    graph.add_node(NODE_SUMMARIZE, nodes.summarize_node)
    graph.add_node(NODE_FORMAT, nodes.format_node)

    graph.add_edge(START, NODE_FETCH)
    graph.add_edge(NODE_FETCH, NODE_FILTER)
    graph.add_conditional_edges(
        NODE_FILTER,
        nodes.route_after_filter,
        {NODE_ANALYZE: NODE_ANALYZE, NODE_FORMAT: NODE_FORMAT},
    )
    graph.add_edge(NODE_ANALYZE, NODE_SUMMARIZE)
    graph.add_edge(NODE_SUMMARIZE, NODE_FORMAT)
    graph.add_edge(NODE_FORMAT, END)

    return graph.compile()
