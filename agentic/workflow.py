from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from agentic.agents import (
    SupportState,
    classify_issue_node,
    customer_context_node,
    kb_retrieval_node,
    resolver_node,
    escalation_decider_node,
    create_escalation_node,
    respond_node,
    route_customer_context,
    route_escalation_creation,
)

import agentic.tools.tools


def orchestrator():
    graph = StateGraph(SupportState)

    graph.add_node("classify_issue", classify_issue_node)
    graph.add_node("retrieve_customer_context", customer_context_node)
    graph.add_node("retrieve_kb", kb_retrieval_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("escalation_decider", escalation_decider_node)
    graph.add_node("create_escalation", create_escalation_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("classify_issue")

    graph.add_conditional_edges(
        "classify_issue",
        route_customer_context,
        {
            "retrieve_customer_context": "retrieve_customer_context",
            "retrieve_kb": "retrieve_kb",
        },
    )

    graph.add_edge("retrieve_customer_context", "retrieve_kb")
    graph.add_edge("retrieve_kb", "resolver")
    graph.add_edge("resolver", "escalation_decider")

    graph.add_conditional_edges(
        "escalation_decider",
        route_escalation_creation,
        {
            "create_escalation": "create_escalation",
            "respond": "respond",
        },
    )

    graph.add_edge("create_escalation", "respond")
    graph.set_finish_point("respond")
    

    return graph.compile(checkpointer=MemorySaver())


app = orchestrator()

  