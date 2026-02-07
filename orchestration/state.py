"""
Shared state schema for the Agentic Chaser workflow graph.

All agents receive and return a Dict[str, Any] compatible with ChaserState.
TypedDict is for static typing and documentation; runtime state remains a plain dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict


class ChaserState(TypedDict, total=False):
    """State dict passed through the graph and into every agent. All keys optional."""

    # Orchestrator / routing
    loa_id: str
    client_id: str
    next_action: str
    current_state: str
    priority_score: float
    sla_days_remaining: Optional[int]
    needs_advisor_intervention: bool
    provider: str
    reference_number: Optional[str]
    client_name: str
    error: str

    # Client communication
    communication_type: str
    context: Dict[str, Any]
    generated_message: str
    communication_channel: str
    message_sent: bool

    # Provider RPA
    rpa_action: str
    rpa_success: bool
    rpa_message: str
    provider_status_text: Optional[str]

    # Document processing
    document_id: str
    run_ocr: bool
    ocr_text: Optional[str]
    ocr_confidence_score: Optional[float]
    validation_passed: bool
    manual_review_required: bool
    quality_issues: str

    # Sentiment / response parser / predictive (optional)
    message_text: str
    message_id: str
    persist: bool
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    raw_message: str
    parsed_intent: str
    key_facts: list
    action_items: list
    contains_question: bool
    completion_signals: list
    parsed_summary: str
    delay_risk: str
    delay_risk_reason: str
    recommended_action: str
    predicted_completion_days: Optional[int]
    insight_summary: str
    horizon_days: int


def initial_state(loa_id: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Build initial state for graph invocation.

    Args:
        loa_id: LOA identifier (required for orchestrator).
        **kwargs: Any additional state keys (e.g. context, client_id).

    Returns:
        Dict suitable for graph.invoke(initial_state(loa_id="L001")).
    """
    state: Dict[str, Any] = {"loa_id": loa_id}
    state.update(kwargs)
    return state
