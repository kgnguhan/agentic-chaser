"""
Agent 6: Workflow Orchestrator & Priority Manager

The CORE BRAIN of the system.
- Tracks all LOA/document/task states
- Calculates priority scores
- Decides what needs action TODAY
- Routes to appropriate agents
- Identifies stuck vs slow cases
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from collections import defaultdict

from agents.fact_find_chasing import get_fact_find_status, get_missing_fact_find_documents
from agents.sentiment_for_priority import (
    apply_sentiment_to_priority,
    get_client_sentiment_for_priority,
    get_provider_sentiment_for_priority,
)
from models.database import (
    ClientProfile,
    DocumentSubmission,
    LOAWorkflow,
    PostAdviceItem,
    get_session,
    session_scope,
)
from models.ml_models import calculate_priority_score
from orchestration.workflow_states import (
    AWAITING_CLIENT_SIGNATURE,
    CASE_COMPLETE,
    CLIENT_CHASE_STATES,
    CLIENT_DOCUMENTS_REJECTED,
    DOCUMENT_AWAITING_VERIFICATION,
    LINK_DOCUMENT_ALLOWED_STATES,
    MARK_PROVIDER_INFO_RECEIVED_ALLOWED,
    PROVIDER_CHASE_STATES,
    PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT,
    PROVIDER_RESPONSE_INCOMPLETE,
    SIGNED_LOA_READY_FOR_PROVIDER,
    SUBMITTED_TO_PROVIDER,
    WITH_PROVIDER_PROCESSING,
)


def workflow_orchestrator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrator agent - the central decision maker.
    
    Responsibilities:
    - Calculate priority scores for all active LOAs
    - Determine next action based on state and priority
    - Flag items needing advisor intervention
    - Update workflow states
    
    Args:
        state: Current workflow state from LangGraph
    
    Returns:
        Updated state with routing decisions
    """
    loa_id = state.get("loa_id")
    
    if not loa_id:
        state["error"] = "No LOA ID provided"
        state["next_action"] = "error"
        return state
    
    # Get LOA from database
    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        
        if not loa:
            state["error"] = f"LOA {loa_id} not found"
            state["next_action"] = "error"
            return state
        
        # Get client age for priority calculation
        client_age = loa.client.age if loa.client else 50

        # Base priority then sentiment-adjusted
        base_priority = calculate_priority_score(
            days_in_state=loa.days_in_current_state,
            sla_overdue=loa.sla_days - loa.sla_days_remaining if loa.sla_days_remaining else 0,
            client_age_55_plus=(client_age >= 55),
            doc_quality_score=loa.document_quality_score,
        )
        client_label, client_score = get_client_sentiment_for_priority(loa.client_id)
        provider_label, provider_score = get_provider_sentiment_for_priority(loa_id)
        priority = apply_sentiment_to_priority(
            base_priority, client_label, client_score, provider_label, provider_score
        )
        # Update priority in database
        loa.priority_score = priority
        
        # Determine if advisor intervention needed
        if priority > 7.0 or (loa.sla_days_remaining and loa.sla_days_remaining < 0):
            loa.needs_advisor_intervention = True
            loa.sla_overdue = True
        
        # Update state based on priority and current workflow state
        state["priority_score"] = priority
        state["current_state"] = loa.current_state
        state["sla_days_remaining"] = loa.sla_days_remaining
        state["needs_advisor_intervention"] = loa.needs_advisor_intervention
        state["provider"] = loa.provider
        state["reference_number"] = loa.reference_number
        state["client_id"] = loa.client_id
        state["client_name"] = loa.client.name if loa.client else "Unknown"

        # Copy fields needed for routing (avoid passing ORM instance out of session)
        loa_data = {
            "current_state": loa.current_state,
            "days_in_current_state": loa.days_in_current_state,
            "sla_days_remaining": loa.sla_days_remaining,
            "pending_document_id": getattr(loa, "pending_document_id", None),
        }
        next_action = decide_next_action(loa_data, state)
        state["next_action"] = next_action
        if next_action == "document_verification" and loa_data.get("pending_document_id"):
            state["document_id"] = loa_data["pending_document_id"]

        db.commit()

    priority = state.get("priority_score")
    current_state = state.get("current_state", "")
    next_action = state.get("next_action", "")
    state["reasoning"] = f"Priority {priority}; state {current_state} → next_action {next_action}."
    return state


def decide_next_action(loa_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """
    Core routing logic - decides what happens next.
    Uses a plain dict (no ORM) to avoid detached instance access.

    Priority order:
    1. Advisor escalation (if critical)
    2. Provider follow-up (if stuck with provider)
    3. Client communication (if waiting on client)
    4. Document processing (if docs need validation)
    5. Monitor (if progressing normally)

    Args:
        loa_data: Dict with current_state, days_in_current_state, sla_days_remaining
        state: Current state dict

    Returns:
        Next action string for routing
    """
    priority = state["priority_score"]
    current_state = loa_data.get("current_state", "")
    days_in_state = loa_data.get("days_in_current_state", 0)
    sla_remaining = loa_data.get("sla_days_remaining")

    # Critical cases - immediate advisor intervention
    if priority > 7.0:
        return "escalate_to_advisor"

    # State-based routing
    if current_state == AWAITING_CLIENT_SIGNATURE:
        return "client_communication"  # chase for LOA signature
    if current_state == DOCUMENT_AWAITING_VERIFICATION:
        if loa_data.get("pending_document_id"):
            return "document_verification"
        return "monitor"
    if current_state == CLIENT_DOCUMENTS_REJECTED:
        return "client_communication"  # document_request to resubmit
    if current_state == SIGNED_LOA_READY_FOR_PROVIDER:
        return "provider_submission"
    if current_state == SUBMITTED_TO_PROVIDER:
        if days_in_state > 5:
            return "provider_follow_up"
        return "monitor"
    if current_state == WITH_PROVIDER_PROCESSING:
        if sla_remaining is not None and sla_remaining < 3:
            return "provider_urgent_follow_up"
        if days_in_state > 15:
            return "provider_follow_up"
        return "monitor"
    if current_state == PROVIDER_RESPONSE_INCOMPLETE:
        return "provider_clarification"
    if current_state == PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT:
        return "client_notification"
    if current_state == CASE_COMPLETE:
        return "complete"
    return "monitor"


def _chase_type_for_state(current_state: str) -> str:
    """Return 'client' or 'provider' for dashboard segregation."""
    if current_state in CLIENT_CHASE_STATES:
        return "client"
    if current_state in PROVIDER_CHASE_STATES:
        return "provider"
    return "other"


def get_priority_queue(
    limit: Optional[int] = 10,
    chase_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get top priority LOAs that need action today.

    Used by dashboard to show "Today's Actions" and Client vs Provider chasing views.

    Args:
        limit: Maximum number of items to return; None for no limit (all active LOAs).
        chase_type: If "client", only LOAs needing client action. If "provider", only LOAs needing provider action.
                    None for all active LOAs.

    Returns:
        List of LOA dicts with priority info and chase_type.
    """
    with session_scope() as db:
        query = (
            db.query(LOAWorkflow)
            .filter(LOAWorkflow.current_state != CASE_COMPLETE)
            .order_by(LOAWorkflow.priority_score.desc())
        )
        if chase_type == "client":
            query = query.filter(LOAWorkflow.current_state.in_(CLIENT_CHASE_STATES))
        elif chase_type == "provider":
            query = query.filter(LOAWorkflow.current_state.in_(PROVIDER_CHASE_STATES))
        if limit is not None:
            query = query.limit(limit)
        loas = query.all()

        results = []
        for loa in loas:
            ct = _chase_type_for_state(loa.current_state)
            results.append({
                "loa_id": loa.loa_id,
                "client_name": loa.client.name if loa.client else "Unknown",
                "provider": loa.provider,
                "current_state": loa.current_state,
                "priority_score": round(loa.priority_score, 2),
                "days_in_state": loa.days_in_current_state,
                "sla_days_remaining": loa.sla_days_remaining,
                "needs_intervention": loa.needs_advisor_intervention,
                "chase_type": ct,
            })
        return results


def update_workflow_state(loa_id: str, new_state: str) -> bool:
    """
    Update LOA workflow state.
    
    Args:
        loa_id: LOA identifier
        new_state: New state to transition to
    
    Returns:
        True if successful, False otherwise
    """
    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        
        if not loa:
            return False
        
        # Update state
        old_state = loa.current_state
        loa.current_state = new_state
        
        # Reset days in state counter
        if old_state != new_state:
            loa.days_in_current_state = 0
        
        # Update timestamp
        loa.updated_at = datetime.utcnow()
        
        db.commit()
        
        return True


def tick_loa_time() -> int:
    """
    Advance time for all active LOAs: increment days_in_current_state by 1,
    decrement sla_days_remaining by 1 (allow negative for overdue).
    Call at the start of each chaser run.
    Returns count of LOAs updated.
    """
    with session_scope() as db:
        loas = db.query(LOAWorkflow).filter(LOAWorkflow.current_state != CASE_COMPLETE).all()
        for loa in loas:
            loa.days_in_current_state = (loa.days_in_current_state or 0) + 1
            if loa.sla_days_remaining is not None:
                loa.sla_days_remaining = loa.sla_days_remaining - 1
            loa.updated_at = datetime.utcnow()
        db.commit()
        return len(loas)


def persist_escalation(loa_id: str) -> bool:
    """
    Set needs_advisor_intervention and escalated_at for an LOA (used by autonomic chaser).
    Returns True if updated, False if LOA not found.
    """
    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if not loa:
            return False
        loa.needs_advisor_intervention = True
        loa.escalated_at = datetime.utcnow()
        loa.updated_at = datetime.utcnow()
        db.commit()
        return True


def mark_provider_info_received(loa_id: str) -> bool:
    """
    Transition LOA to Provider Info Received – Notify Client when provider has supplied the necessary information.
    Valid only when current_state is one of the provider-side states.
    Returns True if updated, False if invalid state or LOA not found.
    """
    allowed = MARK_PROVIDER_INFO_RECEIVED_ALLOWED
    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if not loa or loa.current_state not in allowed:
            return False
        loa.current_state = PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT
        loa.days_in_current_state = 0
        loa.updated_at = datetime.utcnow()
        db.commit()
        return True


def link_document_to_loa(document_id: str, loa_id: str) -> bool:
    """
    Link a document to an LOA for verification (e.g. signed LOA submitted by client).
    Document must belong to the same client as the LOA.
    LOA must be in Awaiting Client Signature or Client Documents Rejected (then we move to Document Awaiting Verification).
    Sets DocumentSubmission.loa_id, LOAWorkflow.pending_document_id, and state to Document Awaiting Verification.
    Returns True if updated, False on validation failure.
    """
    with session_scope() as db:
        doc = db.query(DocumentSubmission).filter_by(document_id=document_id).first()
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if not doc or not loa:
            return False
        if doc.client_id != loa.client_id:
            return False
        if loa.current_state not in LINK_DOCUMENT_ALLOWED_STATES:
            return False
        doc.loa_id = loa_id
        loa.pending_document_id = document_id
        loa.current_state = DOCUMENT_AWAITING_VERIFICATION
        loa.days_in_current_state = 0
        loa.updated_at = datetime.utcnow()
        db.commit()
        return True


def get_loa_detail(loa_id: str) -> Dict[str, Any] | None:
    """
    Load one LOA with client and priority breakdown for the detail panel.
    Returns a single dict (no ORM); None if LOA not found.
    """
    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if not loa:
            return None
        client = loa.client
        client_age = client.age if client else None
        client_age_55_plus = client_age is not None and client_age >= 55
        sla_days_val = loa.sla_days or 15
        sla_remaining = loa.sla_days_remaining
        sla_overdue_days = (
            max(0, sla_days_val - (sla_remaining or 0))
            if sla_remaining is not None
            else 0
        )
        priority = calculate_priority_score(
            days_in_state=loa.days_in_current_state,
            sla_overdue=sla_overdue_days,
            client_age_55_plus=client_age_55_plus,
            doc_quality_score=loa.document_quality_score or 75.0,
        )
        out: Dict[str, Any] = {
            "loa_id": loa.loa_id,
            "client_id": loa.client_id,
            "provider": loa.provider,
            "case_type": loa.case_type or "Pension Consolidation",
            "current_state": loa.current_state,
            "priority_score": round(priority, 2),
            "days_in_current_state": loa.days_in_current_state,
            "sla_days": sla_days_val,
            "sla_days_remaining": loa.sla_days_remaining,
            "document_quality_score": loa.document_quality_score,
            "signature_verified": loa.signature_verified,
            "reference_number": loa.reference_number,
            "needs_advisor_intervention": loa.needs_advisor_intervention,
            "sla_overdue": loa.sla_overdue,
            "pending_document_id": getattr(loa, "pending_document_id", None),
            "created_at": loa.created_at,
            "updated_at": loa.updated_at,
            "priority_breakdown": {
                "days_in_state": loa.days_in_current_state,
                "sla_overdue": sla_overdue_days,
                "client_age_55_plus": client_age_55_plus,
                "doc_quality_score": loa.document_quality_score or 75.0,
            },
        }
        if client:
            out["client"] = {
                "client_id": client.client_id,
                "name": client.name,
                "age": client.age,
                "employment_type": client.employment_type,
                "annual_income": client.annual_income,
                "existing_pensions_count": client.existing_pensions_count,
                "risk_profile": client.risk_profile,
                "communication_preference": client.communication_preference,
                "document_responsiveness": client.document_responsiveness,
            }
        else:
            out["client"] = {"client_id": loa.client_id, "name": "Unknown"}
        return out


POST_ADVICE_COMPLETED_STATE = "Completed"


def get_post_advice_chase_queue(limit: int | None = 50) -> List[Dict[str, Any]]:
    """
    Return post-advice items that need chasing (not Completed).
    Order by days_until_deadline asc (nulls last), then days_outstanding desc.
    """
    with session_scope() as db:
        query = (
            db.query(PostAdviceItem)
            .filter(PostAdviceItem.current_state != POST_ADVICE_COMPLETED_STATE)
            .order_by(
                PostAdviceItem.days_until_deadline.asc().nullslast(),
                PostAdviceItem.days_outstanding.desc(),
            )
        )
        if limit is not None:
            query = query.limit(limit)
        items = query.all()
        result = []
        for item in items:
            client_name = item.client.name if item.client else "Unknown"
            result.append({
                "item_id": item.item_id,
                "client_id": item.client_id,
                "client_name": client_name,
                "item_type": item.item_type,
                "current_state": item.current_state or "Pending",
                "days_outstanding": item.days_outstanding or 0,
                "days_until_deadline": item.days_until_deadline,
            })
    return result


def get_client_list() -> List[Dict[str, Any]]:
    """
    List all clients with pending LOA counts and stage breakdown, plus pending document count.
    For dashboard client table.
    """
    with session_scope() as db:
        clients = db.query(ClientProfile).order_by(ClientProfile.name).all()
        result = []
        for c in clients:
            loas = [x for x in c.loa_workflows if x.current_state != CASE_COMPLETE]
            by_state: Dict[str, int] = defaultdict(int)
            for loa in loas:
                by_state[loa.current_state or "Unknown"] += 1
            docs = list(c.documents) if c.documents else []
            pending_docs = sum(1 for d in docs if d.validation_passed is False or d.manual_review_required is True)
            result.append({
                "client_id": c.client_id,
                "name": c.name,
                "pending_loas": len(loas),
                "stages": dict(by_state),
                "stages_summary": ", ".join(f"{s}: {n}" for s, n in sorted(by_state.items())),
                "pending_documents": pending_docs,
                "total_documents": len(docs),
            })
        return result


def get_client_detail(client_id: str) -> Dict[str, Any] | None:
    """Full client profile plus list of their LOAs and document submissions for detail panel."""
    with session_scope() as db:
        client = db.query(ClientProfile).filter_by(client_id=client_id).first()
        if not client:
            return None
        loa_list = []
        for loa in client.loa_workflows:
            loa_list.append({
                "loa_id": loa.loa_id,
                "provider": loa.provider,
                "current_state": loa.current_state,
                "priority_score": round(loa.priority_score, 2),
                "days_in_current_state": loa.days_in_current_state,
                "sla_days_remaining": loa.sla_days_remaining,
                "needs_advisor_intervention": loa.needs_advisor_intervention,
            })
        doc_list = []
        for d in client.documents:
            doc_list.append({
                "document_id": d.document_id,
                "document_type": d.document_type,
                "document_subtype": d.document_subtype,
                "validation_passed": d.validation_passed,
                "manual_review_required": d.manual_review_required,
                "submitted_at": d.submitted_at,
                "processed_at": d.processed_at,
            })
        post_advice_list = []
        for item in client.post_advice_items or []:
            post_advice_list.append({
                "item_id": item.item_id,
                "item_type": item.item_type,
                "current_state": item.current_state or "Pending",
                "days_outstanding": item.days_outstanding or 0,
                "days_until_deadline": item.days_until_deadline,
            })
        fact_find_status = get_fact_find_status(client.client_id)
        return {
            "client_id": client.client_id,
            "name": client.name,
            "age": client.age,
            "employment_type": client.employment_type,
            "annual_income": client.annual_income,
            "existing_pensions_count": client.existing_pensions_count,
            "risk_profile": client.risk_profile,
            "communication_preference": client.communication_preference,
            "document_responsiveness": client.document_responsiveness,
            "loas": loa_list,
            "documents": doc_list,
            "post_advice_items": post_advice_list,
            "fact_find_status": fact_find_status,
            "missing_fact_find_documents": fact_find_status["missing_documents"],
            "pending_loas": sum(1 for x in loa_list if x["current_state"] != CASE_COMPLETE),
            "pending_documents": sum(1 for x in doc_list if x["validation_passed"] is False or x["manual_review_required"] is True),
        }


def get_provider_list() -> List[Dict[str, Any]]:
    """
    List distinct providers with pending LOA counts and stage breakdown.
    For dashboard provider table.
    """
    with session_scope() as db:
        rows = db.query(LOAWorkflow.provider).distinct().all()
        providers = [r[0] for r in rows if r[0]]
        result = []
        for prov in sorted(providers):
            loas = db.query(LOAWorkflow).filter(
                LOAWorkflow.provider == prov,
                LOAWorkflow.current_state != CASE_COMPLETE,
            ).all()
            by_state: Dict[str, int] = defaultdict(int)
            for loa in loas:
                by_state[loa.current_state or "Unknown"] += 1
            result.append({
                "provider": prov,
                "pending_loas": len(loas),
                "stages": dict(by_state),
                "stages_summary": ", ".join(f"{s}: {n}" for s, n in sorted(by_state.items())),
            })
        return result


def get_provider_detail(provider: str) -> Dict[str, Any] | None:
    """Provider name plus list of LOAs for that provider for detail panel."""
    with session_scope() as db:
        loas = db.query(LOAWorkflow).filter(
            LOAWorkflow.provider == provider,
            LOAWorkflow.current_state != CASE_COMPLETE,
        ).order_by(LOAWorkflow.priority_score.desc()).all()
        loa_list = []
        for loa in loas:
            client_name = loa.client.name if loa.client else "Unknown"
            loa_list.append({
                "loa_id": loa.loa_id,
                "client_id": loa.client_id,
                "client_name": client_name,
                "current_state": loa.current_state,
                "priority_score": round(loa.priority_score, 2),
                "days_in_current_state": loa.days_in_current_state,
                "sla_days_remaining": loa.sla_days_remaining,
                "needs_advisor_intervention": loa.needs_advisor_intervention,
            })
        return {
            "provider": provider,
            "pending_loas": len(loa_list),
            "loas": loa_list,
        }