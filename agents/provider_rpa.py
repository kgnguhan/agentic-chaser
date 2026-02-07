"""
Agent 7: Provider Portal RPA Agent

Stub for provider portal automation: submit LOA, check status, download
documents. In production this would drive a browser (e.g. Playwright/Selenium)
or provider APIs. This implementation returns structured outcomes so the
orchestrator can proceed; actual automation can be wired in later.
"""

from __future__ import annotations

from typing import Any, Dict

from models.database import LOAWorkflow, session_scope


def provider_rpa_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a provider-portal action for an LOA (stub implementation).

    Expected state keys:
        - loa_id: str
        - rpa_action: str – "submit_loa" | "check_status" | "download_documents" | "none"

    Updates state with:
        - rpa_success: bool
        - rpa_message: str – human-readable result or error
        - reference_number: str | None – provider reference when available
        - provider_status_text: str | None – status from portal when action is check_status
    """
    loa_id = state.get("loa_id")
    action = (state.get("rpa_action") or "none").strip().lower()

    if not loa_id:
        state["rpa_success"] = False
        state["rpa_message"] = "No loa_id provided to provider_rpa_agent"
        state["reference_number"] = None
        state["provider_status_text"] = None
        return state

    if action not in ("submit_loa", "check_status", "download_documents", "none"):
        state["rpa_success"] = False
        state["rpa_message"] = f"Unknown rpa_action: {state.get('rpa_action')}"
        state["reference_number"] = None
        state["provider_status_text"] = None
        return state

    if action == "none":
        state["rpa_success"] = True
        state["rpa_message"] = "No RPA action requested."
        state["reference_number"] = state.get("reference_number")
        state["provider_status_text"] = None
        return state

    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if not loa:
            state["rpa_success"] = False
            state["rpa_message"] = f"LOA {loa_id} not found"
            state["reference_number"] = None
            state["provider_status_text"] = None
            return state
        # Copy attributes needed after session closes (avoid detached instance access)
        provider = loa.provider
        client_name = loa.client.name if loa.client else "Unknown"
        reference_number = loa.reference_number
        current_state = loa.current_state

    # Stub: simulate success and optional reference/status
    # In production: call Playwright/Selenium or provider API here
    if action == "submit_loa":
        state["rpa_success"] = True
        state["rpa_message"] = f"Submission to {provider} simulated for {client_name}. Wire real RPA/API for production."
        state["reference_number"] = reference_number or None
        state["provider_status_text"] = None
        _maybe_update_state_after_submit(loa_id)

    elif action == "check_status":
        state["rpa_success"] = True
        state["rpa_message"] = f"Status check for {provider} simulated. Wire real RPA/API for production."
        state["reference_number"] = reference_number
        state["provider_status_text"] = f"Simulated: {current_state}"

    elif action == "download_documents":
        state["rpa_success"] = True
        state["rpa_message"] = f"Document download from {provider} simulated. Wire real RPA/API for production."
        state["reference_number"] = reference_number
        state["provider_status_text"] = None

    state["reasoning"] = f"RPA: rpa_success={state.get('rpa_success')}; { (state.get('rpa_message') or '')[:80]}."
    return state


def _maybe_update_state_after_submit(loa_id: str) -> None:
    """Optionally move LOA to Submitted to Provider when RPA reports successful submit."""
    from orchestration.workflow_states import SIGNED_LOA_READY_FOR_PROVIDER, SUBMITTED_TO_PROVIDER
    with session_scope() as db:
        loa = db.query(LOAWorkflow).filter_by(loa_id=loa_id).first()
        if loa and loa.current_state == SIGNED_LOA_READY_FOR_PROVIDER:
            loa.current_state = SUBMITTED_TO_PROVIDER
            loa.days_in_current_state = 0
