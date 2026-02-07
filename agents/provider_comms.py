"""
Agent 2: Provider Communication Agent

Handles professional communication with pension providers.
- LOA submission cover letters
- Follow-up chasers (when provider is processing)
- Urgent follow-up when SLA is approaching
- Clarification responses when provider needs more info
"""

from __future__ import annotations

from typing import Any, Dict

from models.database import LOAWorkflow, get_session
from utils.llm_helpers import chat_completion


def provider_communication_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate or draft provider-facing communication.

    Args:
        state: Current workflow state containing:
            - loa_id: LOA identifier
            - next_action: provider_submission | provider_follow_up |
                           provider_urgent_follow_up | provider_clarification
            - context: Optional extra context (reference_number, etc.)

    Returns:
        Updated state with generated message and metadata
    """
    loa_id = state.get("loa_id")
    next_action = state.get("next_action", "provider_follow_up")
    context = state.get("context", {})

    if not loa_id:
        state["error"] = "No LOA ID provided"
        return state

    with get_session() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()

        if not loa:
            state["error"] = f"LOA {loa_id} not found"
            return state

        provider = loa.provider
        client_name = loa.client.name if loa.client else "Unknown"
        reference_number = loa.reference_number or context.get("reference_number", "")
        days_in_state = loa.days_in_current_state
        sla_days_remaining = loa.sla_days_remaining
        case_type = loa.case_type or "Pension Consolidation"

        provider_context = {
            "provider": provider,
            "client_name": client_name,
            "reference_number": reference_number,
            "days_in_state": days_in_state,
            "sla_days_remaining": sla_days_remaining,
            "case_type": case_type,
            "current_state": loa.current_state,
        }

    if next_action == "provider_submission":
        message = generate_submission_cover(provider_context, context)
        state["communication_type"] = "submission"
    elif next_action == "provider_urgent_follow_up":
        message = generate_urgent_follow_up(provider_context, context)
        state["communication_type"] = "urgent_follow_up"
    elif next_action == "provider_follow_up":
        message = generate_follow_up(provider_context, context)
        state["communication_type"] = "follow_up"
    elif next_action == "provider_clarification":
        message = generate_clarification_response(provider_context, context)
        state["communication_type"] = "clarification"
    else:
        message = generate_follow_up(provider_context, context)
        state["communication_type"] = "follow_up"

    state["generated_message"] = message
    state["provider"] = provider
    state["message_sent"] = False  # Typically sent via portal or email by advisor/RPA
    state["reasoning"] = f"Drafted provider message ({len(message)} chars)."
    return state


def generate_submission_cover(
    provider_context: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """Generate cover letter/email when submitting LOA to provider."""

    system_prompt = f"""You are a professional UK financial advice firm operations assistant.
Draft a concise, formal cover letter/email to accompany a Letter of Authority (LOA) submission.

Provider: {provider_context['provider']}
Client name: {provider_context['client_name']}
Case type: {provider_context['case_type']}

Guidelines:
- Be professional and formal (B2B)
- State that the signed LOA is attached and request pension information/transfer value
- Include client name and any reference we have (if provided)
- Request confirmation of receipt and expected timescale if known
- Use UK English spelling
- Keep to one short paragraph plus sign-off
"""
    ref_line = f" Our reference: {provider_context['reference_number']}." if provider_context.get("reference_number") else ""
    user_prompt = f"""Draft a cover email to {provider_context['provider']} submitting the signed LOA for {provider_context['client_name']} ({provider_context['case_type']}).{ref_line}
Request pension information / transfer value and ask for confirmation of receipt."""

    return chat_completion(system_prompt, user_prompt, temperature=0.3)


def generate_follow_up(
    provider_context: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """Generate polite follow-up when provider has had the LOA for a while."""

    days = provider_context.get("days_in_state", 0)
    ref = provider_context.get("reference_number", "")

    system_prompt = f"""You are a professional UK financial advice firm operations assistant.
Draft a polite follow-up email to a pension provider chasing progress on an LOA.

Provider: {provider_context['provider']}
Client: {provider_context['client_name']}
Days since submission / in current state: {days}
Reference (if any): {ref or 'Not yet assigned'}

Guidelines:
- Be professional and courteous (B2B)
- Reference the client and LOA/request
- Ask for a status update and any expected date for response
- Do not sound accusatory; assume normal processing delays
- Use UK English spelling
- Keep it brief
"""
    user_prompt = f"""Draft a polite follow-up to {provider_context['provider']} chasing the LOA for {provider_context['client_name']} (with us for {days} days). Ask for a status update."""

    return chat_completion(system_prompt, user_prompt, temperature=0.3)


def generate_urgent_follow_up(
    provider_context: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """Generate urgent follow-up when SLA is approaching or overdue."""

    days_remaining = provider_context.get("sla_days_remaining")
    days_in_state = provider_context.get("days_in_state", 0)
    ref = provider_context.get("reference_number", "")

    system_prompt = f"""You are a professional UK financial advice firm operations assistant.
Draft an urgent but still professional follow-up to a pension provider. Our internal SLA is at risk or overdue.

Provider: {provider_context['provider']}
Client: {provider_context['client_name']}
Days in current state: {days_in_state}
SLA days remaining: {days_remaining}
Reference (if any): {ref or 'Not yet assigned'}

Guidelines:
- Be firm and clear about urgency without being rude
- State that we need the information to meet our client commitment
- Request an immediate status update and a committed date if possible
- Use UK English spelling
- Keep it short and actionable
"""
    user_prompt = f"""Draft an urgent follow-up to {provider_context['provider']} for the LOA regarding {provider_context['client_name']}. SLA remaining: {days_remaining} days. Request immediate status and a committed response date."""

    return chat_completion(system_prompt, user_prompt, temperature=0.3)


def generate_clarification_response(
    provider_context: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """Generate response when provider has requested clarification or more information."""

    provider_question = context.get("provider_question", "additional information or clarification")
    ref = provider_context.get("reference_number", "")

    system_prompt = f"""You are a professional UK financial advice firm operations assistant.
Draft a response to a pension provider who has requested clarification or more information regarding an LOA.

Provider: {provider_context['provider']}
Client: {provider_context['client_name']}
Reference (if any): {ref or 'Not yet assigned'}

Provider has asked for: {provider_question}

Guidelines:
- Be professional and helpful (B2B)
- Acknowledge their request
- Either provide the information if we have it, or state that we will obtain it from the client and respond by [date]
- Use UK English spelling
- Keep it clear and concise
"""
    user_prompt = f"""Draft a response to {provider_context['provider']} regarding their request for: {provider_question}. Client: {provider_context['client_name']}. Be helpful and set clear next steps."""

    return chat_completion(system_prompt, user_prompt, temperature=0.3)
