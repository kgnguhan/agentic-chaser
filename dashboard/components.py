"""
Dashboard UI components and workflow runner for Agentic Chaser.

Uses Streamlit for rendering. run_workflow and get_predictive_insight are pure
functions; render_* functions use st.* and must be called in a Streamlit context.
"""

from __future__ import annotations

from typing import Any, Generator

import streamlit as st

from agents.predictive_intelligence import predictive_intelligence_agent
from agents.client_comms import client_communication_agent
from agents.document_processing import document_processing_agent
from agents.document_upload import (
    client_has_accepted_document,
    create_document_submission_from_upload,
    UPLOAD_DOCUMENT_TYPES,
)
from agents.fact_find_chasing import (
    document_type_to_category_label,
    get_fact_find_chase_queue,
)
from agents.workflow_orchestrator import (
    get_client_detail,
    get_client_list,
    get_loa_detail,
    get_post_advice_chase_queue,
    get_priority_queue,
    get_provider_detail,
    get_provider_list,
    link_document_to_loa,
    mark_provider_info_received,
)
from orchestration import get_chaser_graph, initial_state
from orchestration.workflow_states import (
    CASE_COMPLETE,
    CLIENT_DOCUMENTS_REJECTED,
    LINK_DOCUMENT_ALLOWED_STATES,
    MARK_PROVIDER_INFO_RECEIVED_ALLOWED,
)

# Priority queue table: internal keys and display headers (order preserved)
PRIORITY_QUEUE_COLS = [
    "loa_id",
    "client_name",
    "provider",
    "current_state",
    "priority_score",
    "days_in_state",
    "sla_days_remaining",
    "needs_intervention",
]
COLUMN_LABELS: dict[str, str] = {
    "loa_id": "LOA ID",
    "client_name": "Client",
    "provider": "Provider",
    "current_state": "Workflow state",
    "priority_score": "Priority (0–10)",
    "days_in_state": "Days in state",
    "sla_days_remaining": "SLA days left",
    "needs_intervention": "Needs intervention",
}

# Per-column description (shown as tooltip on column header hover)
COLUMN_HELP: dict[str, str] = {
    "loa_id": "Letter of Authority identifier.",
    "client_name": "Client name.",
    "provider": "Pension provider.",
    "current_state": "Workflow stage (e.g. Awaiting Client Signature, Document Awaiting Verification, Client Documents Rejected, Signed LOA - Ready for Provider, Submitted to Provider, With Provider - Processing, Provider Response Incomplete, Provider Info Received - Notify Client, Case Complete).",
    "priority_score": "Urgency score 0–10 from the ML model (days in state, SLA overdue, client age 55+, document quality). **Legend:** 0–2 Low, 2–5 Medium, 5–7 High, 7–10 Critical (advisor intervention recommended).",
    "days_in_state": "Days spent in the current workflow state.",
    "sla_days_remaining": "Days until the internal SLA target for this state. Stored per LOA; **negative** = already overdue.",
    "needs_intervention": "Flag set when the case needs advisor attention (e.g. priority > 7 or SLA overdue).",
}

# Display names for graph nodes (for pipeline and steps list)
NODE_DISPLAY_NAMES: dict[str, str] = {
    "orchestrator": "Orchestrator",
    "prepare_client": "Client comms",
    "provider_comms": "Provider comms",
    "provider_rpa": "Provider RPA",
    "document_processing": "Document processing",
    "post_document_verification": "Post document verification",
}


def format_step_summary(node_name: str, update: dict[str, Any]) -> str:
    """Return a short human-readable summary of what this step produced (node-specific)."""
    if update.get("error"):
        return f"Error: {update['error'][:100]}"
    node = (node_name or "").strip().lower()
    # Orchestrator: next_action, priority_score, client
    if "orchestrator" in node:
        parts: list[str] = []
        if "next_action" in update:
            parts.append(f"Set next_action to **{update['next_action']}**")
        if "priority_score" in update:
            parts.append(f"priority_score {update['priority_score']}")
        if "client_name" in update:
            parts.append(f"client: {update['client_name']}")
        if "current_state" in update:
            parts.append(f"workflow state: {update['current_state']}")
        return "; ".join(parts) if parts else "Orchestrator ran."
    # Client comms (prepare_client + client_communication_agent)
    if "client" in node and "comms" in node:
        msg = update.get("generated_message") or update.get("message_text")
        comm_type = update.get("communication_type", "")
        if msg:
            return f"Generated client message ({len(str(msg))} chars)" + (
                f"; type: {comm_type}" if comm_type else ""
            )
        return f"communication_type: {comm_type}" if comm_type else "Client communication prepared."
    # Provider comms
    if "provider" in node and "comms" in node:
        msg = update.get("generated_message")
        if msg:
            return f"Drafted provider message ({len(str(msg))} chars)."
        return "Provider communication drafted."
    # Provider RPA
    if "rpa" in node or "provider rpa" in node:
        success = update.get("rpa_success")
        rpa_msg = (update.get("rpa_message") or "")[:80]
        if success is not None:
            return f"rpa_success: {success}; {rpa_msg}" if rpa_msg else f"rpa_success: {success}."
        return rpa_msg or "Provider RPA step completed."
    # Document processing
    if "document processing" in node or "document_processing" in node:
        v = update.get("validation_passed")
        if v is not None:
            return f"validation_passed: {v}; quality_issues: {update.get('quality_issues', '') or '—'}"
        return "Document verification ran."
    # Post document verification
    if "post document" in node or "post_document" in node:
        return "Updated LOA state; route to client if failed."
    # Fallback: key fields
    parts = []
    if "next_action" in update:
        parts.append(f"next_action: {update['next_action']}")
    if "priority_score" in update:
        parts.append(f"priority_score: {update['priority_score']}")
    msg = update.get("generated_message") or update.get("message_text")
    if msg:
        parts.append(f"message ({len(str(msg))} chars)")
    if "rpa_success" in update:
        parts.append(f"rpa_success: {update['rpa_success']}")
    if not parts:
        keys = [k for k in list(update.keys())[:5] if not k.startswith("_")]
        parts.append(", ".join(keys) if keys else "—")
    return "; ".join(parts)


def run_workflow_with_steps(loa_id: str) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """
    Stream the chaser workflow and yield (node_display_name, update) for each step.
    After the stream ends, the caller can use the accumulated state (merged from all updates)
    as the final result. Yields ("__final__", final_state) as the last item so the UI can
    call render_workflow_result(final_state). On exception, yields ("__error__", {"error": "..."}).
    """
    last_yielded: str | None = None
    try:
        graph = get_chaser_graph()
        state = initial_state(loa_id=loa_id)
        accumulated: dict[str, Any] = dict(state)
        stream = graph.stream(accumulated, stream_mode="updates")
        for chunk in stream:
            if not isinstance(chunk, dict):
                continue
            for raw_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                display_name = NODE_DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " ").title())
                accumulated.update(update)
                last_yielded = raw_name
                yield (display_name, update)
        yield ("__final__", accumulated)
    except Exception as e:
        # #region agent log
        import os
        import traceback
        try:
            log_path = r"c:\agentic-chaser\.cursor\debug.log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            tb = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    '{"location":"components.run_workflow_with_steps","message":"exception","data":{"error":'
                    + repr(str(e)[:200])
                    + ',"last_yielded":'
                    + repr(last_yielded)
                    + ',"traceback":'
                    + repr(tb[:2000])
                    + '},"hypothesisId":"H1"}\n'
                )
        except Exception:
            pass
        # #endregion
        yield ("__error__", {"error": str(e)})


def run_workflow(loa_id: str) -> dict[str, Any]:
    """
    Run the chaser workflow graph for the given LOA.

    Returns the final state dict (next_action, generated_message, error, etc.).
    On exception, returns a dict with an "error" key.
    """
    try:
        graph = get_chaser_graph()
        state = initial_state(loa_id=loa_id)
        result = graph.invoke(state)
        return dict(result) if hasattr(result, "keys") else result
    except Exception as e:
        return {"error": str(e)}


def get_predictive_insight(loa_id: str) -> dict[str, Any]:
    """Return predictive intelligence state for the LOA (insight_summary, delay_risk, recommended_action)."""
    try:
        return predictive_intelligence_agent({"loa_id": loa_id})
    except Exception as e:
        return {"error": str(e)}


def render_dashboard_kpis_and_charts(items: list[dict[str, Any]]) -> None:
    """Render KPI metrics row. Uses full list for aggregates."""
    if not items:
        st.caption("No active cases — KPIs will appear when you have LOAs not yet Case Complete.")
        return
    total = len(items)
    high_priority = sum(1 for x in items if (x.get("priority_score") or 0) >= 7)
    sla_at_risk = sum(
        1 for x in items
        if x.get("sla_days_remaining") is not None and x.get("sla_days_remaining") <= 2
    )
    needs_intervention = sum(1 for x in items if x.get("needs_intervention"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total active cases", total)
    with col2:
        st.metric("High priority (score ≥ 7)", high_priority)
    with col3:
        st.metric("SLA at risk (≤ 2 days left)", sla_at_risk)
    with col4:
        st.metric("Needs intervention", needs_intervention)


def build_priority_queue_table(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build (rows, column_config) for the priority queue table so the app can render
    st.dataframe(..., selection_mode="single-row", on_select="rerun") and read selection.
    Returns ([], {}) when items is empty.
    """
    if not items:
        return [], {}
    rows = [{COLUMN_LABELS[k]: x.get(k) for k in PRIORITY_QUEUE_COLS if k in x} for x in items]
    column_config = {
        COLUMN_LABELS[k]: st.column_config.Column(COLUMN_LABELS[k], help=COLUMN_HELP[k])
        for k in PRIORITY_QUEUE_COLS
    }
    return rows, column_config


def render_priority_queue(items: list[dict[str, Any]]) -> None:
    """Render the priority queue table with descriptive headers. Handles empty list with a message."""
    if not items:
        st.info("No LOAs needing action. Run **python main.py seed** to load test data.")
        return
    rows, column_config = build_priority_queue_table(items)
    st.dataframe(
        rows,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )


def render_loa_detail_panel(loa_id: str, show_link_document: bool = False) -> None:
    """Show task description, column details with explanations, client details, and detailed status."""
    detail = get_loa_detail(loa_id)
    if detail is None:
        st.warning("LOA not found.")
        return
    client = detail.get("client") or {}
    client_name = client.get("name", "Unknown")

    st.markdown(
        f"**{detail.get('loa_id', '')}** — {client_name} — {detail.get('provider', '')} "
        f"({detail.get('case_type', '')})"
    )
    st.caption("Task summary")

    with st.expander("Column details", expanded=True):
        pb = detail.get("priority_breakdown") or {}
        sla_days = detail.get("sla_days") or 15
        sla_remaining = detail.get("sla_days_remaining")
        for k in PRIORITY_QUEUE_COLS:
            label = COLUMN_LABELS[k]
            if k == "client_name":
                val = client_name
            elif k == "loa_id":
                val = detail.get("loa_id", "")
            elif k == "provider":
                val = detail.get("provider", "")
            elif k == "current_state":
                val = detail.get("current_state", "")
            elif k == "priority_score":
                val = detail.get("priority_score", "")
                st.markdown(
                    f"**{label}**: {val} — Calculated from: days in state = {pb.get('days_in_state', 0)}, "
                    f"SLA overdue = {pb.get('sla_overdue', 0)}, client 55+ = {pb.get('client_age_55_plus', False)}, "
                    f"document quality = {pb.get('doc_quality_score', 75)}. "
                    "Legend: 0–2 Low, 2–5 Medium, 5–7 High, 7–10 Critical (advisor intervention recommended)."
                )
                continue
            elif k == "days_in_state":
                val = detail.get("days_in_current_state", "")
            elif k == "sla_days_remaining":
                val = sla_remaining if sla_remaining is not None else "—"
                st.markdown(
                    f"**{label}**: {val} — Days until internal SLA target for this state (total SLA = {sla_days} days). "
                    "Negative = already overdue."
                )
                continue
            elif k == "needs_intervention":
                val = detail.get("needs_advisor_intervention", False)
            else:
                val = detail.get(k, "")
            st.markdown(f"**{label}**: {val}")

    with st.expander("Client details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            for key in ("name", "age", "employment_type", "annual_income", "existing_pensions_count"):
                v = client.get(key)
                st.text(f"{key.replace('_', ' ').title()}: {v if v is not None else '—'}")
        with c2:
            for key in ("risk_profile", "communication_preference", "document_responsiveness"):
                v = client.get(key)
                st.text(f"{key.replace('_', ' ').title()}: {v if v is not None else '—'}")

    with st.expander("Detailed status", expanded=False):
        state = detail.get("current_state", "")
        days = detail.get("days_in_current_state", 0)
        remaining = detail.get("sla_days_remaining")
        remaining_str = str(remaining) if remaining is not None else "—"
        st.markdown(
            f"**Workflow state:** {state} · **Days in state:** {days} · "
            f"**SLA days left:** {remaining_str} (target {sla_days} days)"
        )
        st.markdown(
            f"**Needs advisor intervention:** {detail.get('needs_advisor_intervention', False)} · "
            f"**SLA overdue:** {detail.get('sla_overdue', False)} · "
            f"**Reference:** {detail.get('reference_number') or '—'}"
        )
        created = detail.get("created_at")
        updated = detail.get("updated_at")
        if created or updated:
            st.caption(f"Created: {created} · Updated: {updated}")
        if state in MARK_PROVIDER_INFO_RECEIVED_ALLOWED:
            if st.button("Mark provider info received", key=f"mark_info_{loa_id}"):
                if mark_provider_info_received(loa_id):
                    st.success("State set to Provider Info Received - Notify Client.")
                    st.rerun()
                else:
                    st.error("Could not update (invalid state or LOA not found).")
        if show_link_document and state in LINK_DOCUMENT_ALLOWED_STATES:
            client_id = detail.get("client_id")
            if client_id:
                st.markdown("**Upload document**")
                loa_upload_types = [t for t in UPLOAD_DOCUMENT_TYPES if t in ("Passport", "Driving Licence", "Utility Bill", "Bank Statement", "Council Tax", "Pension Statement", "Investment Statement", "Payslip", "P60")]
                upload_doc_type = st.selectbox("Document type", loa_upload_types, key=f"loa_upload_type_{loa_id}")
                upload_file = st.file_uploader("File (PDF or image)", type=["pdf", "png", "jpg", "jpeg"], key=f"loa_upload_file_{loa_id}")
                if st.button("Upload and link to LOA", key=f"loa_upload_btn_{loa_id}") and upload_file:
                    try:
                        info = create_document_submission_from_upload(
                            client_id=client_id,
                            document_type=upload_doc_type,
                            file_bytes=upload_file.read(),
                            filename=upload_file.name,
                            loa_id=loa_id,
                        )
                        if link_document_to_loa(info["document_id"], loa_id):
                            st.success("Document uploaded and linked. Run workflow to verify.")
                            st.rerun()
                        else:
                            st.error("Document saved but could not link to LOA (invalid state).")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

                st.markdown("**Link existing document to LOA**")
                client_data = get_client_detail(client_id)
                doc_list = (client_data or {}).get("documents") or []
                if doc_list:
                    doc_options = [f"{d.get('document_id', '')} — {d.get('document_type', '')}" for d in doc_list]
                    doc_ids = [d.get("document_id") for d in doc_list]
                    doc_choice = st.selectbox("Document", range(len(doc_options)), format_func=lambda i: doc_options[i], key=f"link_doc_{loa_id}")
                    if st.button("Link document to LOA", key=f"link_btn_{loa_id}"):
                        if doc_choice is not None and doc_choice < len(doc_ids):
                            doc_id = doc_ids[doc_choice]
                            if link_document_to_loa(doc_id, loa_id):
                                st.success("Document linked. State set to Document Awaiting Verification; chaser will verify on next run.")
                                st.rerun()
                            else:
                                st.error("Could not link (document or LOA invalid, or LOA not in Awaiting Client Signature / Client Documents Rejected).")
                else:
                    st.caption("No existing documents for this client.")
            else:
                st.caption("Client not found.")


# Client table: columns and help for selection + detail
CLIENT_TABLE_COLS = ["client_id", "name", "pending_loas", "stages_summary", "pending_documents"]
CLIENT_TABLE_LABELS = {
    "client_id": "Client ID",
    "name": "Name",
    "pending_loas": "Pending LOAs",
    "stages_summary": "Stages (count by state)",
    "pending_documents": "Pending documents",
}
CLIENT_TABLE_HELP = {
    "client_id": "Unique client identifier.",
    "name": "Client name.",
    "pending_loas": "Number of LOAs not yet Case Complete.",
    "stages_summary": "Breakdown of pending LOAs by workflow state (e.g. Prepared: 2, Provider Submitted: 1).",
    "pending_documents": "Document submissions pending validation or manual review.",
}


def build_client_table_data(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build (rows, column_config) for the client table with row selection."""
    if not items:
        return [], {}
    rows = [{CLIENT_TABLE_LABELS[k]: x.get(k) for k in CLIENT_TABLE_COLS if k in x} for x in items]
    column_config = {
        CLIENT_TABLE_LABELS[k]: st.column_config.Column(CLIENT_TABLE_LABELS[k], help=CLIENT_TABLE_HELP[k])
        for k in CLIENT_TABLE_COLS
    }
    return rows, column_config


def render_client_detail_panel(client_id: str) -> None:
    """Show full client profile, their LOAs by stage, and document submissions."""
    detail = get_client_detail(client_id)
    if detail is None:
        st.warning("Client not found.")
        return
    st.markdown(f"**{detail.get('name', '')}** ({detail.get('client_id', '')})")
    st.caption("Client summary")

    with st.expander("Profile", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            for key in ("client_id", "name", "age", "employment_type", "annual_income", "existing_pensions_count"):
                v = detail.get(key)
                st.text(f"{key.replace('_', ' ').title()}: {v if v is not None else '—'}")
        with c2:
            for key in ("risk_profile", "communication_preference", "document_responsiveness"):
                v = detail.get(key)
                st.text(f"{key.replace('_', ' ').title()}: {v if v is not None else '—'}")

    with st.expander("Pending LOAs by stage", expanded=True):
        st.caption(f"Total pending: {detail.get('pending_loas', 0)}")
        loas = detail.get("loas") or []
        if not loas:
            st.caption("No LOAs.")
        else:
            for loa in loas:
                if loa.get("current_state") == CASE_COMPLETE:
                    continue
                st.markdown(
                    f"**{loa.get('loa_id', '')}** — {loa.get('provider', '')} · "
                    f"State: {loa.get('current_state', '')} · Priority: {loa.get('priority_score', '')} · "
                    f"Days in state: {loa.get('days_in_current_state', 0)} · "
                    f"SLA left: {loa.get('sla_days_remaining') if loa.get('sla_days_remaining') is not None else '—'}"
                )

    with st.expander("Document submissions", expanded=False):
        st.caption(f"Pending (validation/review): {detail.get('pending_documents', 0)} of {len(detail.get('documents') or [])}")
        docs = detail.get("documents") or []
        if not docs:
            st.caption("No document submissions.")
        else:
            for d in docs:
                status = "Pending" if (d.get("validation_passed") is False or d.get("manual_review_required") is True) else "OK"
                st.markdown(
                    f"**{d.get('document_id', '')}** — {d.get('document_type', '')} · "
                    f"Validation: {d.get('validation_passed')} · Manual review: {d.get('manual_review_required')} · {status}"
                )

    fact_find_status = detail.get("fact_find_status") or {}
    received_count = fact_find_status.get("received_count", 0)
    required_count = fact_find_status.get("required_count", 0)
    missing_ff = fact_find_status.get("missing_documents") or detail.get("missing_fact_find_documents") or []
    with st.expander("Fact-find documents (partial submission status)", expanded=bool(missing_ff) or required_count > 0):
        st.caption(f"**{received_count} of {required_count}** fact-find documents received. Chasing only for missing categories.")
        if not missing_ff:
            st.caption("All required fact-find document categories are satisfied.")
        else:
            st.caption("Chase client for these fact-find documents only:")
            for doc in missing_ff:
                st.markdown(f"- {doc}")

    post_advice = detail.get("post_advice_items") or []
    with st.expander("Post-advice items", expanded=bool(post_advice)):
        if not post_advice:
            st.caption("No post-advice items.")
        else:
            for item in post_advice:
                st.markdown(
                    f"**{item.get('item_id', '')}** — {item.get('item_type', '')} · "
                    f"State: {item.get('current_state', '')} · "
                    f"Days outstanding: {item.get('days_outstanding', 0)} · "
                    f"Days until deadline: {item.get('days_until_deadline') if item.get('days_until_deadline') is not None else '—'}"
                )


# Provider table: columns and help
PROVIDER_TABLE_COLS = ["provider", "pending_loas", "stages_summary"]
PROVIDER_TABLE_LABELS = {
    "provider": "Provider",
    "pending_loas": "Pending LOAs",
    "stages_summary": "Stages (count by state)",
}
PROVIDER_TABLE_HELP = {
    "provider": "Pension provider name.",
    "pending_loas": "Number of LOAs with this provider not yet Case Complete.",
    "stages_summary": "Breakdown of pending LOAs by workflow state.",
}


def build_provider_table_data(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build (rows, column_config) for the provider table with row selection."""
    if not items:
        return [], {}
    rows = [{PROVIDER_TABLE_LABELS[k]: x.get(k) for k in PROVIDER_TABLE_COLS if k in x} for x in items]
    column_config = {
        PROVIDER_TABLE_LABELS[k]: st.column_config.Column(PROVIDER_TABLE_LABELS[k], help=PROVIDER_TABLE_HELP[k])
        for k in PROVIDER_TABLE_COLS
    }
    return rows, column_config


def render_provider_detail_panel(provider: str) -> None:
    """Show provider name and list of pending LOAs with client and stage."""
    detail = get_provider_detail(provider)
    if detail is None:
        st.warning("Provider not found.")
        return
    st.markdown(f"**{detail.get('provider', '')}** — {detail.get('pending_loas', 0)} pending LOAs")
    st.caption("Provider summary")

    with st.expander("Pending LOAs", expanded=True):
        loas = detail.get("loas") or []
        if not loas:
            st.caption("No pending LOAs.")
        else:
            for loa in loas:
                st.markdown(
                    f"**{loa.get('loa_id', '')}** — {loa.get('client_name', '')} · "
                    f"State: {loa.get('current_state', '')} · Priority: {loa.get('priority_score', '')} · "
                    f"Days in state: {loa.get('days_in_current_state', 0)} · "
                    f"SLA left: {loa.get('sla_days_remaining') if loa.get('sla_days_remaining') is not None else '—'}"
                )


FACT_FIND_QUEUE_COLS = ["client_id", "client_name", "received_required", "missing_documents"]
FACT_FIND_QUEUE_LABELS = {
    "client_id": "Client ID",
    "client_name": "Name",
    "received_required": "Received (of required)",
    "missing_documents": "Missing documents",
}


def build_fact_find_queue_table(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build (rows, column_config) for the fact-find chase queue. Shows X of Y received; missing_documents as comma-separated."""
    if not items:
        return [], {}
    rows = []
    for x in items:
        rc = x.get("received_count")
        rq = x.get("required_count")
        status = f"{rc} / {rq}" if rc is not None and rq is not None else "—"
        row = {
            FACT_FIND_QUEUE_LABELS["client_id"]: x.get("client_id"),
            FACT_FIND_QUEUE_LABELS["client_name"]: x.get("client_name"),
            FACT_FIND_QUEUE_LABELS["received_required"]: status,
            FACT_FIND_QUEUE_LABELS["missing_documents"]: ", ".join(x.get("missing_documents") or []),
        }
        rows.append(row)
    column_config = {
        FACT_FIND_QUEUE_LABELS[k]: st.column_config.Column(FACT_FIND_QUEUE_LABELS[k])
        for k in FACT_FIND_QUEUE_COLS
    }
    return rows, column_config


POST_ADVICE_QUEUE_COLS = ["item_id", "client_id", "client_name", "item_type", "current_state", "days_outstanding", "days_until_deadline"]
POST_ADVICE_QUEUE_LABELS = {
    "item_id": "Item ID",
    "client_id": "Client ID",
    "client_name": "Client",
    "item_type": "Type",
    "current_state": "State",
    "days_outstanding": "Days outstanding",
    "days_until_deadline": "Days to deadline",
}


def build_post_advice_queue_table(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build (rows, column_config) for the post-advice chase queue."""
    if not items:
        return [], {}
    rows = [{POST_ADVICE_QUEUE_LABELS[k]: x.get(k) for k in POST_ADVICE_QUEUE_COLS if k in x} for x in items]
    column_config = {
        POST_ADVICE_QUEUE_LABELS[k]: st.column_config.Column(POST_ADVICE_QUEUE_LABELS[k])
        for k in POST_ADVICE_QUEUE_COLS
    }
    return rows, column_config


def run_fact_find_upload_and_validate(
    client_id: str, document_type: str, file_bytes: bytes, filename: str
) -> Generator[tuple[str, dict[str, Any] | None], None, None]:
    """
    Yield progress messages and final state for fact-find upload + OCR validation.
    Yields: (message, None) for steps; (message, state) at end with validation_passed, quality_issues.
    """
    if client_has_accepted_document(client_id, document_type):
        yield (
            f"{document_type.strip() or 'Document'} is already submitted for this client. You can generate a chase message below.",
            {"validation_passed": False, "quality_issues": "already_submitted"},
        )
        return
    yield ("Saving document…", None)
    try:
        info = create_document_submission_from_upload(
            client_id=client_id,
            document_type=document_type,
            file_bytes=file_bytes,
            filename=filename,
            loa_id=None,
        )
    except Exception as e:
        yield (f"Error saving document: {e}", {"error": str(e)})
        return
    yield ("Document saved.", None)
    yield ("Running OCR and validation…", None)
    state = document_processing_agent(
        {"document_id": info["document_id"], "run_ocr": True}
    )
    if state.get("error"):
        yield (f"Error: {state['error']}", state)
        return
    if state.get("validation_passed"):
        category = document_type_to_category_label(document_type)
        yield (f"Validation passed. Document received for: {category}. Fact-find status updated.", state)
    else:
        issues = state.get("quality_issues") or "Quality check failed"
        if "ocr_unavailable" in issues:
            yield (
                "Validation skipped: OCR engine not available. Install pytesseract and the Tesseract engine to enable document validation (see https://github.com/UB-Mannheim/tesseract/wiki). You can generate a chase message below.",
                state,
            )
        elif "ocr_runtime_error" in issues:
            yield (
                "Validation skipped: OCR failed. Check that the file is an image or PDF and that Tesseract is installed. You can generate a chase message below.",
                state,
            )
        else:
            yield (f"Validation failed: {issues}. You can generate a chase message below.", state)


def run_fact_find_chase(client_id: str, client_name: str, missing_documents: list[str]) -> dict[str, Any]:
    """Run client_communication_agent for fact-find document request; return state with generated_message."""
    state = {
        "client_id": client_id,
        "communication_type": "fact_find_document_request",
        "context": {"missing_documents": missing_documents},
    }
    client_communication_agent(state)
    return state


def run_post_advice_chase(
    client_id: str, item_type: str, days_outstanding: int, days_until_deadline: int | None
) -> dict[str, Any]:
    """Run client_communication_agent for post-advice reminder; return state with generated_message."""
    state = {
        "client_id": client_id,
        "communication_type": "post_advice_reminder",
        "context": {
            "item_type": item_type,
            "days_outstanding": days_outstanding,
            "deadline_days": days_until_deadline if days_until_deadline is not None else 14,
        },
    }
    client_communication_agent(state)
    return state


def render_workflow_result(state: dict[str, Any]) -> None:
    """Display the outcome of a workflow run (next_action, generated_message, error, rpa_message)."""
    if state.get("error"):
        st.error(state["error"])
        return
    next_action = state.get("next_action") or "—"
    st.info(f"**Next action:** {next_action}")
    if state.get("generated_message"):
        with st.expander("Generated message", expanded=True):
            st.text(state["generated_message"])
    if "rpa_success" in state:
        if state.get("rpa_success"):
            st.success(state.get("rpa_message", "RPA completed."))
        else:
            st.warning(state.get("rpa_message", "RPA did not complete."))
