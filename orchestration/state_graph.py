"""
LangGraph workflow: orchestrator → route by next_action → client_comms | provider_comms | provider_rpa | END.

Run with: graph.invoke(initial_state(loa_id="L001"))
Result is the final state dict (e.g. generated_message, next_action, error).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal

from langgraph.graph import END, START, StateGraph

from agents.client_comms import client_communication_agent
from agents.document_processing import document_processing_agent
from agents.provider_comms import provider_communication_agent
from agents.provider_rpa import provider_rpa_agent
from agents.workflow_orchestrator import workflow_orchestrator_agent
from models.database import DocumentSubmission, LOAWorkflow, session_scope

from orchestration.state import ChaserState
from orchestration.workflow_states import (
    AWAITING_CLIENT_SIGNATURE,
    CLIENT_DOCUMENTS_REJECTED,
    DOCUMENT_AWAITING_VERIFICATION,
    SIGNED_LOA_READY_FOR_PROVIDER,
)


def _orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run workflow orchestrator; populates next_action, client_id, current_state, etc."""
    return workflow_orchestrator_agent(state)


def _route_after_orchestrator(
    state: Dict[str, Any],
) -> Literal["prepare_client", "provider_comms", "provider_rpa", "document_processing", "__end__"]:
    """Route to the next node based on next_action from the orchestrator."""
    next_action = (state.get("next_action") or "").strip()
    if next_action in ("client_communication", "client_notification"):
        return "prepare_client"
    if next_action == "provider_submission":
        return "provider_rpa"
    if next_action in (
        "provider_follow_up",
        "provider_urgent_follow_up",
        "provider_clarification",
    ):
        return "provider_comms"
    if next_action == "document_verification":
        return "document_processing"
    return "__end__"


def _document_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run document verification for pending_document_id; state must have document_id."""
    state = {**state, "run_ocr": True}
    return document_processing_agent(state)


def _post_document_verification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    After document_processing: clear pending_document_id; if validation passed
    set Client Signed and signature_verified; if failed set state to Client Documents Rejected
    and route to prepare_client for document_request.
    """
    loa_id = state.get("loa_id")
    validation_passed = state.get("validation_passed", False)
    quality_issues = state.get("quality_issues") or ""
    document_type = ""

    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if loa:
            loa.pending_document_id = None
            loa.updated_at = datetime.utcnow()
            if validation_passed:
                loa.signature_verified = True
                loa.current_state = SIGNED_LOA_READY_FOR_PROVIDER
                loa.days_in_current_state = 0
            else:
                loa.current_state = CLIENT_DOCUMENTS_REJECTED
                loa.days_in_current_state = 0
        if not validation_passed and state.get("document_id"):
            doc = db.query(DocumentSubmission).filter_by(document_id=state["document_id"]).first()
            if doc and doc.document_type:
                document_type = doc.document_type

    if validation_passed:
        return {**state, "next_action": "complete", "reasoning": "LOA state set to Signed LOA - Ready for Provider; verification complete."}
    # Failed: add document_type to context so client message can reference the rejected document
    missing_label = f"{document_type} (signed LOA or supporting documents)" if document_type else "signed LOA or supporting documents"
    return {
        **state,
        "current_state": CLIENT_DOCUMENTS_REJECTED,
        "next_action": "client_communication",
        "communication_type": "document_request",
        "reasoning": "LOA state set to Client Documents Rejected; routing to client comms (validation failed).",
        "context": {
            "missing_documents": [missing_label],
            "quality_issues": quality_issues,
            "message": "Documents did not pass verification. Please resubmit.",
            "document_type": document_type,
        },
    }


def _prepare_client_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Set communication_type from next_action/current_state then run client_communication_agent."""
    next_action = state.get("next_action", "")
    current_state = state.get("current_state", "")
    if next_action == "client_notification":
        state = {**state, "communication_type": "status_update"}
    elif next_action == "client_communication":
        if state.get("communication_type") == "document_request":
            pass  # already set by post_document_verification (Client Documents Rejected)
        elif current_state == AWAITING_CLIENT_SIGNATURE:
            state = {**state, "communication_type": "loa_signature_request"}
        else:
            state = {**state, "communication_type": "general"}
    elif state.get("communication_type") != "document_request":
        state = {**state, "communication_type": "general"}
    return client_communication_agent(state)


def _provider_comms_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Draft provider-facing message (follow-up, urgent, or clarification)."""
    return provider_communication_agent(state)


def _provider_rpa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Submit LOA to provider (sets rpa_action=submit_loa)."""
    state = {**state, "rpa_action": "submit_loa"}
    return provider_rpa_agent(state)


def _route_after_post_doc(
    state: Dict[str, Any],
) -> Literal["prepare_client", "__end__"]:
    """After document verification: if validation failed, go to prepare_client for document_request; else END."""
    if state.get("validation_passed"):
        return "__end__"
    return "prepare_client"


def _build_graph() -> StateGraph:
    """Build and return the StateGraph (not compiled)."""
    builder = StateGraph(ChaserState)

    builder.add_node("orchestrator", _orchestrator_node)
    builder.add_node("prepare_client", _prepare_client_node)
    builder.add_node("provider_comms", _provider_comms_node)
    builder.add_node("provider_rpa", _provider_rpa_node)
    builder.add_node("document_processing", _document_processing_node)
    builder.add_node("post_document_verification", _post_document_verification_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "prepare_client": "prepare_client",
            "provider_comms": "provider_comms",
            "provider_rpa": "provider_rpa",
            "document_processing": "document_processing",
            "__end__": END,
        },
    )
    builder.add_edge("document_processing", "post_document_verification")
    builder.add_conditional_edges(
        "post_document_verification",
        _route_after_post_doc,
        {"prepare_client": "prepare_client", "__end__": END},
    )
    builder.add_edge("prepare_client", END)
    builder.add_edge("provider_comms", END)
    builder.add_edge("provider_rpa", END)

    return builder


def get_chaser_graph():
    """
    Return the compiled LangGraph for the chaser workflow.

    Usage:
        from orchestration import get_chaser_graph, initial_state
        graph = get_chaser_graph()
        result = graph.invoke(initial_state(loa_id="L001"))
    """
    return _build_graph().compile()
