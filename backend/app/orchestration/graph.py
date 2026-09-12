"""SatQuery AI -- LangGraph StateGraph Assembly & Compilation.

Implements the complete stateful graph flow (sections 2, 26, 29):
1. validate_inputs
2. interpret_query
3. clarification_check (LangGraph interrupt)
4. constrained_router
5. check_cache
6. dispatch_specialist
7. validate_specialist_output
8. run_evidence_engine
9. confidence_check (ACCEPT | WARN | ABSTAIN)
10. compose_response / abstention_response
11. persist_result
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.orchestration.nodes import (
    abstention_response_node,
    check_cache_node,
    clarification_check_node,
    compose_response_node,
    compose_response_with_warning_node,
    confidence_check_node,
    constrained_router_node,
    dispatch_specialist_node,
    interpret_query_node,
    persist_result_node,
    run_evidence_engine_node,
    validate_evidence_contract_node,
    validate_inputs_node,
    validate_specialist_output_node,
)
from app.orchestration.state import SatQueryState


# ─── Conditional Routing Functions ───────────────────────────────────────────

def route_after_validation(state: SatQueryState) -> Literal["interpret_query", "persist_result"]:
    """If input validation failed, bypass downstream pipeline directly to persist_result."""
    if state.get("error") or state.get("status") == "failed":
        return "persist_result"
    return "interpret_query"


def route_after_clarification(state: SatQueryState) -> Literal["constrained_router", "__end__"]:
    """If clarification is required and awaiting input, end the current step (interrupt)."""
    if state.get("needs_clarification"):
        return "__end__"
    return "constrained_router"


def route_after_cache(state: SatQueryState) -> Literal["run_evidence_engine", "dispatch_specialist"]:
    """If a valid cache hit exists, bypass specialist dispatch directly to evidence engine."""
    if state.get("cache_hit"):
        return "run_evidence_engine"
    return "dispatch_specialist"


def route_after_confidence(state: SatQueryState) -> Literal["abstention_response", "compose_response", "compose_response_with_warning"]:
    """Branches according to Section 18: ACCEPT -> compose_response, WARN -> compose_response_with_warning, ABSTAIN -> abstention."""
    decision = state.get("confidence", {}).get("decision", "ACCEPT")
    if decision == "ABSTAIN":
        return "abstention_response"
    elif decision == "WARN":
        return "compose_response_with_warning"
    return "compose_response"


# ─── Graph Builder ───────────────────────────────────────────────────────────

def build_satquery_graph() -> StateGraph:
    """Assembles the SatQuery StateGraph with all nodes, transitions, and conditional branches."""
    workflow = StateGraph(SatQueryState)

    # Register discrete nodes
    workflow.add_node("validate_inputs", validate_inputs_node)
    workflow.add_node("interpret_query", interpret_query_node)
    workflow.add_node("clarification_check", clarification_check_node)
    workflow.add_node("constrained_router", constrained_router_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("dispatch_specialist", dispatch_specialist_node)
    workflow.add_node("validate_specialist_output", validate_specialist_output_node)
    workflow.add_node("run_evidence_engine", run_evidence_engine_node)
    workflow.add_node("validate_evidence_contract", validate_evidence_contract_node)
    workflow.add_node("confidence_check", confidence_check_node)
    workflow.add_node("compose_response", compose_response_node)
    workflow.add_node("compose_response_with_warning", compose_response_with_warning_node)
    workflow.add_node("abstention_response", abstention_response_node)
    workflow.add_node("persist_result", persist_result_node)

    # Define edges and conditional routing
    workflow.add_edge(START, "validate_inputs")

    workflow.add_conditional_edges(
        "validate_inputs",
        route_after_validation,
        {
            "interpret_query": "interpret_query",
            "persist_result": "persist_result",
        },
    )

    workflow.add_edge("interpret_query", "clarification_check")

    workflow.add_conditional_edges(
        "clarification_check",
        route_after_clarification,
        {
            "constrained_router": "constrained_router",
            "__end__": END,
        },
    )

    workflow.add_edge("constrained_router", "check_cache")

    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {
            "run_evidence_engine": "run_evidence_engine",
            "dispatch_specialist": "dispatch_specialist",
        },
    )

    workflow.add_edge("dispatch_specialist", "validate_specialist_output")
    workflow.add_edge("validate_specialist_output", "run_evidence_engine")
    workflow.add_edge("run_evidence_engine", "validate_evidence_contract")
    workflow.add_edge("validate_evidence_contract", "confidence_check")

    workflow.add_conditional_edges(
        "confidence_check",
        route_after_confidence,
        {
            "abstention_response": "abstention_response",
            "compose_response": "compose_response",
            "compose_response_with_warning": "compose_response_with_warning",
        },
    )

    workflow.add_edge("compose_response", "persist_result")
    workflow.add_edge("compose_response_with_warning", "persist_result")
    workflow.add_edge("abstention_response", "persist_result")
    workflow.add_edge("persist_result", END)

    return workflow


# ─── Checkpointer & Runtime Compilation ──────────────────────────────────────

_compiled_graph = None
_checkpointer_instance: Optional[BaseCheckpointSaver] = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Returns thread-level checkpointer (SqliteSaver in local dev, MemorySaver fallback)."""
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    settings = get_settings()
    try:
        db_path = settings.state_dir / "langgraph_checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer_instance = SqliteSaver(conn)
    except Exception:
        # MemorySaver fallback for testing or ephemeral environments
        _checkpointer_instance = MemorySaver()

    return _checkpointer_instance


def get_satquery_pipeline(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Returns a compiled, thread-persistent SatQuery LangGraph pipeline."""
    global _compiled_graph
    if _compiled_graph is not None and checkpointer is None:
        return _compiled_graph

    cp = checkpointer or get_checkpointer()
    graph = build_satquery_graph()
    compiled = graph.compile(checkpointer=cp)
    if checkpointer is None:
        _compiled_graph = compiled
    return compiled
