"""
Agent 1: Client Communication Agent

Handles personalized, empathetic communication with clients.
- Document requests
- LOA signature reminders
- Post-advice item follow-ups
- Status updates
"""

from __future__ import annotations

from typing import Any, Dict

from models.database import ClientProfile, LOAWorkflow, get_session
from utils.llm_helpers import chat_completion


def client_communication_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate personalized client communication.
    
    Args:
        state: Current workflow state containing:
            - client_id: Client identifier
            - communication_type: Type of communication needed
            - context: Additional context (LOA state, missing docs, etc.)
    
    Returns:
        Updated state with generated message
    """
    client_id = state.get("client_id")
    comm_type = state.get("communication_type", "general")
    context = state.get("context", {})
    
    # Get client info from database
    with get_session() as db:
        client = db.query(ClientProfile).filter_by(client_id=client_id).first()
        
        if not client:
            state["error"] = f"Client {client_id} not found"
            return state
        
        # Build context for LLM
        client_context = {
            "name": client.name,
            "age": client.age,
            "communication_preference": client.communication_preference,
            "responsiveness": client.document_responsiveness,
        }
    
    # Generate message based on type
    if comm_type == "loa_signature_request":
        message = generate_loa_signature_request(client_context, context)
    
    elif comm_type == "document_request":
        message = generate_document_request(client_context, context)

    elif comm_type == "fact_find_document_request":
        message = generate_fact_find_document_request(client_context, context)

    elif comm_type == "post_advice_reminder":
        message = generate_post_advice_reminder(client_context, context)
    
    elif comm_type == "status_update":
        message = generate_status_update(client_context, context)
    
    else:
        message = generate_general_message(client_context, context)
    
    # Store message in state
    state["generated_message"] = message
    state["communication_channel"] = client_context["communication_preference"]
    state["message_sent"] = True
    state["reasoning"] = f"Generated {comm_type} message ({len(message)} chars)."
    return state


def generate_loa_signature_request(
    client: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Generate personalized LOA signature request."""
    
    provider = context.get("provider", "your pension provider")
    days_waiting = context.get("days_waiting", 0)
    
    system_prompt = f"""You are a helpful UK financial advisor assistant.
Generate a friendly, professional email requesting LOA signature.

Client details:
- Name: {client['name']}
- Age: {client['age']}
- Communication style preference: {client['communication_preference']}
- Document responsiveness: {client['responsiveness']}

Context:
- Provider: {provider}
- Days since LOA sent: {days_waiting}

Guidelines:
- Be warm and empathetic
- Explain WHY the LOA is needed (to get pension information)
- Keep it brief and clear
- Include a clear call-to-action
- Use UK English spelling
"""
    
    user_prompt = f"""Write a personalized email to {client['name']} asking them to sign and return the Letter of Authority for {provider}.
    
Make it conversational and explain that this allows us to request their pension details on their behalf."""
    
    message = chat_completion(system_prompt, user_prompt, temperature=0.4)
    
    return message


def generate_document_request(
    client: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Generate personalized document request. When quality_issues or message (rejection reason) is present, explain why we are reaching out again."""
    missing_docs = context.get("missing_documents", [])
    docs_str = ", ".join(missing_docs)
    quality_issues = context.get("quality_issues") or context.get("rejection_reason") or ""
    reason_message = (context.get("message") or "").strip()
    is_resubmit = bool(quality_issues or reason_message)

    resubmit_instruction = ""
    if is_resubmit:
        resubmit_instruction = f"""
The client previously submitted document(s) that did not meet our requirements. You must explain why we are reaching out again.
- Quality/verification issues: {quality_issues or "Not specified"}
- Internal note: {reason_message or "Please resubmit with a clearer or valid document."}
Ask the client to resubmit and briefly explain what was wrong (e.g. image unclear, expired document, wrong type) in a helpful, non-technical way.
"""

    system_prompt = f"""You are a helpful UK financial advisor assistant.
Generate a friendly email requesting missing documents.

Client details:
- Name: {client['name']}
- Responsiveness: {client['responsiveness']}

Missing documents: {docs_str}
{resubmit_instruction}

Guidelines:
- Be patient and helpful
- Explain WHY each document is needed
- Provide examples (e.g., "recent utility bill like gas or electric")
- Keep tone encouraging, not demanding
- Use UK English spelling
"""

    user_prompt = f"""Write a personalized email to {client['name']} requesting these documents: {docs_str}."""
    if is_resubmit:
        user_prompt += f" Explain that the previous submission did not pass our verification (reason: {quality_issues or reason_message}) and ask them to resubmit with a clearer or valid document."
    else:
        user_prompt += " Explain what each is and why we need it. Make it sound helpful, not bureaucratic."

    message = chat_completion(system_prompt, user_prompt, temperature=0.4)
    return message


def generate_fact_find_document_request(
    client: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Generate personalized fact-find document request for the advice process."""
    missing_docs = context.get("missing_documents", [])
    docs_str = ", ".join(missing_docs)

    system_prompt = f"""You are a helpful UK financial advisor assistant.
Generate a friendly email requesting fact-find documents needed for the advice process.

Client details:
- Name: {client['name']}
- Responsiveness: {client['responsiveness']}

Missing fact-find documents: {docs_str}

These are standard documents we need to complete our fact-find (e.g. proof of identity for AML, proof of address, pension statements for transfer analysis, payslips/P60s for contribution and tax planning). Explain briefly why each is needed and give practical examples where helpful (e.g. "recent utility bill or bank statement dated within 3 months").

Guidelines:
- Be warm and professional
- Explain that these documents are part of our standard fact-find to give them the best advice
- Keep tone encouraging, not bureaucratic
- Use UK English spelling
"""

    user_prompt = f"""Write a personalized email to {client['name']} requesting these fact-find documents: {docs_str}.

Explain what each is and why we need it for the advice process. Make it helpful and clear."""

    message = chat_completion(system_prompt, user_prompt, temperature=0.4)
    return message


def generate_post_advice_reminder(
    client: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Generate post-advice item reminder."""
    
    item_type = context.get("item_type", "form")
    days_outstanding = context.get("days_outstanding", 0)
    deadline_days = context.get("deadline_days", 14)
    
    system_prompt = f"""You are a helpful UK financial advisor assistant.
Generate a friendly reminder about an outstanding post-advice item.

Client details:
- Name: {client['name']}

Item details:
- Type: {item_type}
- Days outstanding: {days_outstanding}
- Days until deadline: {deadline_days}

Guidelines:
- Be friendly but create appropriate urgency if deadline approaching
- Explain WHY this item matters (e.g., risk questionnaire determines investment strategy)
- Offer help if they're stuck
- Keep it short and actionable
"""
    
    user_prompt = f"""Write a reminder to {client['name']} about their outstanding {item_type}.
    
Days outstanding: {days_outstanding}
Days until deadline: {deadline_days}

Make it helpful and explain why completing this is important."""
    
    message = chat_completion(system_prompt, user_prompt, temperature=0.4)
    
    return message


def generate_status_update(
    client: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Generate proactive status update."""
    
    update_type = context.get("update_type", "progress")
    provider = context.get("provider", "provider")
    
    system_prompt = f"""You are a helpful UK financial advisor assistant.
Generate a proactive status update for the client.

Client details:
- Name: {client['name']}

Update context:
- Type: {update_type}
- Provider: {provider}

Guidelines:
- Be proactive and transparent
- Set realistic expectations
- Reassure without overpromising
- Keep it brief
"""
    
    user_prompt = f"""Write a brief status update email to {client['name']} about their case with {provider}.

Context: {context.get('message', 'General progress update')}

Keep it positive and informative."""
    
    message = chat_completion(system_prompt, user_prompt, temperature=0.4)
    
    return message


def generate_general_message(
    client: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Generate general client communication."""
    
    message_context = context.get("message", "")
    
    system_prompt = f"""You are a helpful UK financial advisor assistant.
Generate a personalized message for the client.

Client details:
- Name: {client['name']}

Guidelines:
- Be warm and professional
- Use UK English spelling
- Keep it conversational
"""
    
    user_prompt = f"""Write a message to {client['name']}.

Context: {message_context}"""
    
    message = chat_completion(system_prompt, user_prompt, temperature=0.4)
    
    return message