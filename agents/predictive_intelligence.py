"""
Agent 8: Predictive Intelligence Agent

Uses workflow data and optional ML/LLM to produce predictions and
recommendations: delay risk, likely completion, next best action,
and short explanations for the advisor or dashboard.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.sentiment_for_priority import (
    apply_sentiment_to_priority,
    get_client_sentiment_for_priority,
    get_provider_sentiment_for_priority,
)
from models.database import LOAWorkflow, get_session
from models.ml_models import calculate_priority_score
from orchestration.workflow_states import (
    AWAITING_CLIENT_SIGNATURE,
    CASE_COMPLETE,
    CLIENT_DOCUMENTS_REJECTED,
    DOCUMENT_AWAITING_VERIFICATION,
    PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT,
    PROVIDER_RESPONSE_INCOMPLETE,
    SIGNED_LOA_READY_FOR_PROVIDER,
    SUBMITTED_TO_PROVIDER,
    WITH_PROVIDER_PROCESSING,
)
from utils.llm_helpers import chat_completion


def predictive_intelligence_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce predictions and recommendations for an LOA or client.

    Expected state keys:
        - loa_id: str – LOA to analyse (optional if client_id + provider given)
        - client_id: str – optional, for client-level view
        - horizon_days: int – days ahead to consider (default 14)

    Updates state with:
        - delay_risk: str – "low" | "medium" | "high"
        - delay_risk_reason: str
        - recommended_action: str
        - predicted_completion_days: int | None
        - insight_summary: str – short narrative for advisor
    """
    loa_id = state.get("loa_id")
    client_id = state.get("client_id")
    horizon_days = int(state.get("horizon_days", 14))

    if not loa_id and not client_id:
        state["error"] = "Provide loa_id or client_id to predictive_intelligence_agent"
        state.setdefault("delay_risk", "unknown")
        state.setdefault("delay_risk_reason", "")
        state.setdefault("recommended_action", "")
        state.setdefault("predicted_completion_days", None)
        state.setdefault("insight_summary", "")
        return state

    with get_session() as db:
        if loa_id:
            loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
            loas = [loa] if loa else []
        else:
            loas = db.query(LOAWorkflow).filter_by(client_id=client_id).all()

        if not loas:
            state["error"] = f"No LOA found for loa_id={loa_id!r} or client_id={client_id!r}"
            state["delay_risk"] = "unknown"
            state["delay_risk_reason"] = ""
            state["recommended_action"] = ""
            state["predicted_completion_days"] = None
            state["insight_summary"] = ""
            return state

        # Use first LOA for priority and state (or worst case if multiple)
        loa = max(loas, key=lambda x: (x.priority_score or 0, -(x.sla_days_remaining or 999)))
        client_age = loa.client.age if loa.client else 50
        base_priority = calculate_priority_score(
            days_in_state=loa.days_in_current_state,
            sla_overdue=max(0, (loa.sla_days or 0) - (loa.sla_days_remaining or 0)),
            client_age_55_plus=(client_age >= 55),
            doc_quality_score=loa.document_quality_score or 75.0,
        )
        client_label, client_score = get_client_sentiment_for_priority(loa.client_id)
        provider_label, provider_score = get_provider_sentiment_for_priority(loa.loa_id)
        priority = apply_sentiment_to_priority(
            base_priority, client_label, client_score, provider_label, provider_score
        )

        delay_risk, delay_reason = _assess_delay_risk(loa, priority)
        recommended = _recommend_action(loa, priority, delay_risk)
        pred_days = _estimate_completion_days(loa, delay_risk)
        # Copy attributes needed after session closes (avoid detached instance access)
        provider = loa.provider
        current_state = loa.current_state
        days_in_state = loa.days_in_current_state
        sla_remaining = loa.sla_days_remaining
        client_sentiment_label = client_label
        client_sentiment_score = client_score

    state["delay_risk"] = delay_risk
    state["delay_risk_reason"] = delay_reason
    state["recommended_action"] = recommended
    state["predicted_completion_days"] = pred_days
    state["priority_score"] = priority

    # Optional LLM insight (graceful if Ollama unavailable)
    try:
        state["insight_summary"] = _generate_insight(
            provider=provider,
            current_state=current_state,
            days_in_state=days_in_state,
            sla_remaining=sla_remaining,
            delay_risk=delay_risk,
            recommended_action=recommended,
            client_sentiment_label=client_sentiment_label,
            client_sentiment_score=client_sentiment_score,
        )
    except Exception:
        state["insight_summary"] = f"{delay_risk} delay risk. {recommended}"

    return state


def _assess_delay_risk(loa: LOAWorkflow, priority: float) -> tuple[str, str]:
    if loa.sla_days_remaining is not None and loa.sla_days_remaining < 0:
        return "high", "SLA already overdue."
    if priority >= 7.0:
        return "high", "Priority score indicates urgent intervention needed."
    if loa.sla_days_remaining is not None and loa.sla_days_remaining < 3:
        return "high", "SLA due within 3 days."
    if loa.days_in_current_state > 15 or (loa.sla_days_remaining is not None and loa.sla_days_remaining < 7):
        return "medium", "Extended time in current state or SLA within a week."
    return "low", "On track; no immediate delay indicators."


def _recommend_action(loa: LOAWorkflow, priority: float, delay_risk: str) -> str:
    if delay_risk == "high":
        if loa.current_state in (SUBMITTED_TO_PROVIDER, WITH_PROVIDER_PROCESSING):
            return "Urgent provider follow-up or escalate to advisor."
        if loa.current_state == AWAITING_CLIENT_SIGNATURE:
            return "Chase client for LOA signature."
        if loa.current_state == CLIENT_DOCUMENTS_REJECTED:
            return "Chase client to resubmit documents."
        return "Escalate to advisor for immediate action."
    if delay_risk == "medium":
        if loa.current_state in (SUBMITTED_TO_PROVIDER, WITH_PROVIDER_PROCESSING):
            return "Send polite provider follow-up."
        if loa.current_state == AWAITING_CLIENT_SIGNATURE:
            return "Send LOA reminder to client."
        if loa.current_state == CLIENT_DOCUMENTS_REJECTED:
            return "Send document resubmission request to client."
        return "Monitor and follow up if no change in 3 days."
    if loa.current_state == SIGNED_LOA_READY_FOR_PROVIDER:
        return "Submit LOA to provider."
    if loa.current_state == PROVIDER_RESPONSE_INCOMPLETE:
        return "Respond to provider with requested clarification."
    if loa.current_state == DOCUMENT_AWAITING_VERIFICATION:
        return "Document verification will run on next chaser cycle."
    return "Continue monitoring."


def _estimate_completion_days(loa: LOAWorkflow, delay_risk: str) -> int | None:
    if loa.current_state == CASE_COMPLETE:
        return 0
    remaining = loa.sla_days_remaining
    if remaining is not None:
        return max(0, remaining)
    # Rough default by state
    defaults = {
        AWAITING_CLIENT_SIGNATURE: 14,
        DOCUMENT_AWAITING_VERIFICATION: 2,
        CLIENT_DOCUMENTS_REJECTED: 10,
        SIGNED_LOA_READY_FOR_PROVIDER: 12,
        SUBMITTED_TO_PROVIDER: 10,
        WITH_PROVIDER_PROCESSING: 7,
        PROVIDER_RESPONSE_INCOMPLETE: 14,
        PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT: 2,
    }
    base = defaults.get(loa.current_state, 10)
    if delay_risk == "high":
        return base + 7
    if delay_risk == "medium":
        return base + 3
    return base


def _generate_insight(
    provider: str,
    current_state: str,
    days_in_state: int,
    sla_remaining: int | None,
    delay_risk: str,
    recommended_action: str,
    client_sentiment_label: str | None = None,
    client_sentiment_score: float | None = None,
) -> str:
    sentiment_note = ""
    if client_sentiment_label and client_sentiment_label != "Neutral":
        sentiment_note = f" Client sentiment: {client_sentiment_label}."
    prompt = f"""You are a UK financial advice operations analyst. In one short sentence (max 25 words), summarise the situation for an advisor. Be factual and actionable.

Provider: {provider}. State: {current_state}. Days in state: {days_in_state}. SLA days remaining: {sla_remaining}. Delay risk: {delay_risk}. Recommended: {recommended_action}.{sentiment_note}"""
    return chat_completion(
        "Output only the single-sentence summary, no preamble.",
        prompt,
        temperature=0.2,
    ).strip()
