"""
Agentic Chaser – Streamlit dashboard.

LOA and document chaser for advisors: view priority queue, run workflow for an LOA,
see generated messages and next actions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when run via streamlit run dashboard/streamlit_app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from agents.document_upload import UPLOAD_DOCUMENT_TYPES
from agents.workflow_orchestrator import (
    get_client_list,
    get_loa_detail,
    get_post_advice_chase_queue,
    get_priority_queue,
    get_provider_list,
    link_document_to_loa,
    mark_provider_info_received,
)
from orchestration.workflow_states import MARK_PROVIDER_INFO_RECEIVED_ALLOWED
from dashboard.components import (
    build_client_table_data,
    build_fact_find_queue_table,
    build_post_advice_queue_table,
    build_priority_queue_table,
    build_provider_table_data,
    format_step_summary,
    get_fact_find_chase_queue,
    get_predictive_insight,
    render_dashboard_kpis_and_charts,
    render_client_detail_panel,
    render_loa_detail_panel,
    render_priority_queue,
    render_provider_detail_panel,
    render_workflow_result,
    run_fact_find_chase,
    run_fact_find_upload_and_validate,
    run_post_advice_chase,
    run_workflow_with_steps,
)

st.set_page_config(page_title="Agentic Chaser", layout="wide", initial_sidebar_state="expanded")
st.title("Agentic Chaser")
st.caption("LOA and document chaser for advisors. Use the tabs below to switch views.")

# Sidebar: load test data (for empty DB, e.g. on first deploy)
with st.sidebar:
    st.subheader("Data")
    if st.button("Load test data", help="Load sample LOAs and clients from data/test (replaces existing data)"):
        try:
            from scripts.load_test_data import load_test_data
            load_test_data()
            st.success("Test data loaded. Refreshing…")
            st.rerun()
        except FileNotFoundError as e:
            st.error(f"Test data not found: {e}")
        except Exception as e:
            st.error(f"Failed to load test data: {e}")

# Ensure we can load priority queue (DB + models); fetch all active for KPIs/charts
try:
    items = get_priority_queue(limit=None)
except FileNotFoundError:
    st.error("ML models not found. Run: **python main.py train**")
    st.stop()
except Exception:
    st.error("Database not initialized or connection failed. Run: **python main.py init-db** then load test data from the sidebar.")
    st.stop()

table_items = items[:20] if items else []
if "selected_loa_id" not in st.session_state:
    st.session_state.selected_loa_id = None
if "selected_client_id" not in st.session_state:
    st.session_state.selected_client_id = None
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = None

tab_actions, tab_clients, tab_providers = st.tabs([
    "Today's Actions",
    "By client",
    "By provider",
])

# ----- Tab: Today's Actions -----
with tab_actions:
    if st.session_state.get("mark_info_message"):
        st.success(st.session_state.mark_info_message)
        del st.session_state.mark_info_message
    if st.session_state.get("link_doc_message"):
        st.success(st.session_state.link_doc_message)
        del st.session_state.link_doc_message
    render_dashboard_kpis_and_charts(items)
    st.markdown("---")
    tab_client_chase, tab_provider_chase, tab_fact_find, tab_post_advice = st.tabs([
        "Client chasing",
        "Provider chasing",
        "Fact-find chasing",
        "Post-advice chasing",
    ])

    def _run_workflow_section(table_items: list, suffix: str) -> None:
        if not table_items:
            st.info("No LOAs in this view. Click **Load test data** in the sidebar to load sample data, or select the other tab.")
            return
        wf_col1, wf_col2, wf_col3 = st.columns([2, 1, 1])
        with wf_col1:
            options = [f"{x['loa_id']} — {x.get('client_name', '')} ({x.get('provider', '')})" for x in table_items]
            loa_ids = [x["loa_id"] for x in table_items]
            choice = st.selectbox("Select LOA", range(len(options)), format_func=lambda i: options[i], key=f"loa_select_{suffix}")
            wf_selected_loa_id = loa_ids[choice] if choice is not None else loa_ids[0]
        with wf_col2:
            run_clicked = st.button("Run workflow", key=f"run_wf_{suffix}", use_container_width=True)
        with wf_col3:
            selected_item = table_items[choice] if choice is not None else table_items[0]
            can_mark_info = selected_item.get("current_state") in MARK_PROVIDER_INFO_RECEIVED_ALLOWED
            if can_mark_info:
                mark_clicked = st.button("Mark provider info received", key=f"mark_info_{suffix}", use_container_width=True)
            else:
                mark_clicked = False
        if mark_clicked and can_mark_info:
            if mark_provider_info_received(wf_selected_loa_id):
                st.session_state.mark_info_message = "State set to Provider Info Received - Notify Client."
            else:
                st.error("Could not update (invalid state or LOA not found).")
            st.rerun()
        with st.expander("Predictive insight", expanded=False):
            try:
                insight = get_predictive_insight(wf_selected_loa_id)
                if insight.get("error"):
                    st.warning(insight["error"])
                else:
                    st.write("**Summary:**", insight.get("insight_summary", "—"))
                    st.write("**Delay risk:**", insight.get("delay_risk", "—"))
                    st.write("**Recommended action:**", insight.get("recommended_action", "—"))
            except Exception:
                st.warning("Insight unavailable.")
        if run_clicked:
            st.markdown("---")
            pipeline_ph = st.empty()
            steps_ph = st.empty()
            result_ph = st.empty()
            steps_list = []
            completed_nodes = []
            current_node = None
            final_state = None
            agent_label = "—"

            def _render_pipeline(done: list, running: str | None, agent: str) -> None:
                stages = [
                    ("Orchestrator", "Orchestrator" in done, running == "Orchestrator"),
                    ("Route", len(done) >= 1, False),
                    (agent or "Agent", agent and agent in done, running == agent if agent else False),
                    ("End", len(done) >= 2, False),
                ]
                cols = st.columns(4)
                for i, (label, is_done, is_running) in enumerate(stages):
                    with cols[i]:
                        if is_running:
                            st.markdown(f"**{label}**")
                            st.caption("Running…")
                        elif is_done:
                            st.markdown(f"**{label}**")
                            st.caption("Done")
                        else:
                            st.markdown(label)
                            st.caption("Pending")

            for display_name, update in run_workflow_with_steps(wf_selected_loa_id):
                if display_name == "__error__":
                    result_ph.error(update.get("error", "Unknown error"))
                    break
                if display_name == "__final__":
                    final_state = update
                    completed_nodes.append(current_node or "")
                    with pipeline_ph.container():
                        st.markdown("**Pipeline**")
                        _render_pipeline(completed_nodes, None, agent_label if agent_label != "—" else None)
                    break
                if current_node:
                    completed_nodes.append(current_node)
                current_node = display_name
                if display_name in ("Client comms", "Provider comms", "Provider RPA", "Document processing", "Post document verification"):
                    agent_label = display_name
                steps_list.append((display_name, update))
                with pipeline_ph.container():
                    st.markdown("**Pipeline**")
                    _render_pipeline(completed_nodes, current_node, agent_label if agent_label != "—" else None)
                with steps_ph.container():
                    st.markdown("**Workflow steps** (agent reasoning in real time)")
                    for idx, (name, upd) in enumerate(steps_list, 1):
                        reasoning = upd.get("reasoning") or format_step_summary(name, upd)
                        st.markdown(f"{idx}. **{name}** — {reasoning}")
                        with st.expander("Details", expanded=False):
                            for k in ("next_action", "current_state", "priority_score", "validation_passed", "quality_issues", "generated_message", "error", "rpa_success", "rpa_message"):
                                if k in upd and upd[k] is not None:
                                    v = upd[k]
                                    if k == "generated_message" and isinstance(v, str) and len(v) > 200:
                                        st.text_area(k, v, height=120, disabled=True)
                                    else:
                                        st.caption(f"**{k}**: {v}")
            if final_state:
                result_ph.markdown("**Result**")
                render_workflow_result(final_state)

    with tab_client_chase:
        client_items = get_priority_queue(limit=None, chase_type="client")
        client_table_items = client_items[:20]
        st.markdown("**Run workflow** (client-side: Awaiting Client Signature, Document Awaiting Verification, Client Documents Rejected, Provider Info Received - Notify Client)")
        _run_workflow_section(client_table_items, "client")
        st.markdown("---")
        st.markdown("**Priority queue** — select a row for details")
        if not client_table_items:
            render_priority_queue([])
        else:
            rows, column_config = build_priority_queue_table(client_table_items)
            event = st.dataframe(rows, column_config=column_config, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun", height="content", key="df_client")
            sel = getattr(event, "selection", None)
            if sel and getattr(sel, "rows", None) and 0 <= sel.rows[0] < len(client_table_items):
                st.session_state.selected_loa_id = client_table_items[sel.rows[0]]["loa_id"]
        st.markdown("---")
        if st.session_state.selected_loa_id:
            st.markdown("**Task details**")
            render_loa_detail_panel(st.session_state.selected_loa_id, show_link_document=True)
        else:
            st.info("Select a row in the table above to see task details.")

    with tab_provider_chase:
        provider_items = get_priority_queue(limit=None, chase_type="provider")
        provider_table_items = provider_items[:20]
        st.markdown("**Run workflow** (provider-side: Submitted to Provider, With Provider - Processing, Provider Response Incomplete)")
        _run_workflow_section(provider_table_items, "provider")
        st.markdown("---")
        st.markdown("**Priority queue** — select a row for details")
        if not provider_table_items:
            render_priority_queue([])
        else:
            rows, column_config = build_priority_queue_table(provider_table_items)
            event = st.dataframe(rows, column_config=column_config, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun", height="content", key="df_provider")
            sel = getattr(event, "selection", None)
            if sel and getattr(sel, "rows", None) and 0 <= sel.rows[0] < len(provider_table_items):
                st.session_state.selected_loa_id = provider_table_items[sel.rows[0]]["loa_id"]
        st.markdown("---")
        if st.session_state.selected_loa_id:
            st.markdown("**Upload document** (e.g. provider response)")
            prov_loa_id = st.session_state.selected_loa_id
            prov_loa_detail = get_loa_detail(prov_loa_id) if prov_loa_id else None
            if prov_loa_detail and prov_loa_detail.get("client_id"):
                prov_upload_file = st.file_uploader("File (PDF or image)", type=["pdf", "png", "jpg", "jpeg"], key="provider_upload_file")
                if st.button("Upload and attach to case", key="provider_upload_btn") and prov_upload_file:
                    try:
                        from agents.document_upload import create_document_submission_from_upload
                        create_document_submission_from_upload(
                            client_id=prov_loa_detail["client_id"],
                            document_type="Provider response",
                            file_bytes=prov_upload_file.read(),
                            filename=prov_upload_file.name,
                            loa_id=prov_loa_id,
                        )
                        st.success("Document uploaded and attached to case.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
            st.markdown("**Task details**")
            render_loa_detail_panel(st.session_state.selected_loa_id, show_link_document=False)
        else:
            st.info("Select a row in the table above to see task details.")

    with tab_fact_find:
        fact_find_items = get_fact_find_chase_queue(limit=50)
        st.markdown("**Fact-find chasing** — clients missing proof of identity, proof of address, pension statements, etc. Status shows received (of required); we chase only for missing documents.")

        st.markdown("**Upload fact-find document** — specify client and document type; file will be validated with OCR.")
        client_list = get_client_list()
        ff_upload_clients = [(c["client_id"], c["name"]) for c in client_list] if client_list else []
        ff_doc_types = [t for t in UPLOAD_DOCUMENT_TYPES if t not in ("Provider response", "Application Form", "Risk Questionnaire", "AML Verification", "Authority to Proceed", "Annual Review")]
        if ff_upload_clients:
            ff_client_options = [f"{cid} — {name}" for cid, name in ff_upload_clients]
            ff_upload_client_ix = st.selectbox("Client", range(len(ff_client_options)), format_func=lambda i: ff_client_options[i], key="ff_upload_client")
            ff_upload_doc_type = st.selectbox("Document type", ff_doc_types, key="ff_upload_doctype")
            ff_file = st.file_uploader("File (PDF or image)", type=["pdf", "png", "jpg", "jpeg"], key="ff_upload_file")
            if st.button("Upload and validate", key="ff_upload_btn") and ff_file and ff_upload_client_ix is not None:
                client_id = ff_upload_clients[ff_upload_client_ix][0]
                file_bytes = ff_file.read()
                live_log = st.container()
                last_state = None
                with live_log:
                    st.caption("Live log")
                    for msg, state in run_fact_find_upload_and_validate(client_id, ff_upload_doc_type, file_bytes, ff_file.name):
                        st.markdown(f"- {msg}")
                        if state:
                            last_state = state
                if last_state and not last_state.get("validation_passed") and not last_state.get("error"):
                    st.session_state["ff_upload_failed_context"] = {
                        "client_id": client_id,
                        "document_type": ff_upload_doc_type,
                        "quality_issues": last_state.get("quality_issues", ""),
                    }
            if st.session_state.get("ff_upload_failed_context"):
                ctx = st.session_state["ff_upload_failed_context"]
                if st.button("Generate chase message (for last failed upload)", key="ff_chase_after_upload"):
                    from agents.fact_find_chasing import document_type_to_category_label
                    from agents.client_comms import client_communication_agent
                    category = document_type_to_category_label(ctx["document_type"])
                    chase_state = client_communication_agent({
                        "client_id": ctx["client_id"],
                        "communication_type": "document_request",
                        "context": {
                            "missing_documents": [category],
                            "quality_issues": ctx.get("quality_issues", ""),
                            "message": "Document did not pass verification. Please resubmit.",
                        },
                    })
                    if chase_state.get("generated_message"):
                        st.success("Chase message generated.")
                        with st.expander("Generated message", expanded=True):
                            st.text(chase_state["generated_message"])
                    del st.session_state["ff_upload_failed_context"]

        if not fact_find_items:
            st.info("No clients with missing fact-find documents. All required categories are satisfied for active clients.")
        else:
            st.markdown("---")
            ff_rows, ff_config = build_fact_find_queue_table(fact_find_items)
            st.dataframe(ff_rows, column_config=ff_config, use_container_width=True, hide_index=True, height="content")
            st.markdown("**Run chase** — select a client and generate a fact-find document request message.")
            ff_options = [f"{x['client_id']} — {x.get('client_name', '')}" for x in fact_find_items]
            ff_choice = st.selectbox("Client", range(len(ff_options)), format_func=lambda i: ff_options[i], key="ff_select")
            if st.button("Run chase", key="run_ff_chase"):
                entry = fact_find_items[ff_choice] if ff_choice is not None else fact_find_items[0]
                with st.spinner("Generating message…"):
                    state = run_fact_find_chase(
                        entry["client_id"],
                        entry.get("client_name", ""),
                        entry.get("missing_documents", []),
                    )
                if state.get("generated_message"):
                    st.success("Message generated.")
                    with st.expander("Generated message", expanded=True):
                        st.text(state["generated_message"])
                elif state.get("error"):
                    st.error(state["error"])

    with tab_post_advice:
        post_advice_items = get_post_advice_chase_queue(limit=50)
        st.markdown("**Post-advice chasing** — signed application forms, risk questionnaires, AML verification, authority to proceed, annual review responses.")
        if not post_advice_items:
            st.info("No post-advice items to chase. All items are completed.")
        else:
            st.markdown("**Upload document** (e.g. signed form for an item)")
            pa_upload_options = [f"{x['item_id']} — {x.get('client_name', '')} — {x.get('item_type', '')}" for x in post_advice_items]
            pa_upload_ix = st.selectbox("Post-advice item", range(len(pa_upload_options)), format_func=lambda i: pa_upload_options[i], key="pa_upload_select")
            pa_upload_file = st.file_uploader("File (PDF or image)", type=["pdf", "png", "jpg", "jpeg"], key="pa_upload_file")
            if st.button("Upload document", key="pa_upload_btn") and pa_upload_file and pa_upload_ix is not None:
                entry = post_advice_items[pa_upload_ix]
                try:
                    from agents.document_upload import create_document_submission_from_upload
                    create_document_submission_from_upload(
                        client_id=entry["client_id"],
                        document_type=entry.get("item_type", "Application Form"),
                        file_bytes=pa_upload_file.read(),
                        filename=pa_upload_file.name,
                        loa_id=None,
                    )
                    st.success("Document uploaded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
            st.markdown("---")
            pa_rows, pa_config = build_post_advice_queue_table(post_advice_items)
            st.dataframe(pa_rows, column_config=pa_config, use_container_width=True, hide_index=True, height="content")
            st.markdown("**Run chase** — select an item and generate a post-advice reminder message.")
            pa_options = [f"{x['item_id']} — {x.get('client_name', '')} — {x.get('item_type', '')}" for x in post_advice_items]
            pa_choice = st.selectbox("Post-advice item", range(len(pa_options)), format_func=lambda i: pa_options[i], key="pa_select")
            if st.button("Run chase", key="run_pa_chase"):
                entry = post_advice_items[pa_choice] if pa_choice is not None else post_advice_items[0]
                with st.spinner("Generating message…"):
                    state = run_post_advice_chase(
                        entry["client_id"],
                        entry.get("item_type", "form"),
                        entry.get("days_outstanding", 0),
                        entry.get("days_until_deadline"),
                    )
                if state.get("generated_message"):
                    st.success("Message generated.")
                    with st.expander("Generated message", expanded=True):
                        st.text(state["generated_message"])
                elif state.get("error"):
                    st.error(state["error"])

# ----- Tab: By client -----
with tab_clients:
    client_list = get_client_list()
    st.markdown("**Clients** — select a row for details")
    if not client_list:
        st.info("No clients. Click **Load test data** in the sidebar to load sample data.")
    else:
        c_rows, c_config = build_client_table_data(client_list)
        c_event = st.dataframe(
            c_rows,
            column_config=c_config,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            height="content",
        )
        c_sel = getattr(c_event, "selection", None)
        if c_sel and getattr(c_sel, "rows", None):
            idx = c_sel.rows[0]
            if 0 <= idx < len(client_list):
                st.session_state.selected_client_id = client_list[idx]["client_id"]
    st.markdown("---")
    if st.session_state.selected_client_id:
        st.markdown("**Client details**")
        render_client_detail_panel(st.session_state.selected_client_id)
    else:
        st.info("Select a row in the table above to see client details.")

# ----- Tab: By provider -----
with tab_providers:
    provider_list = get_provider_list()
    st.markdown("**Providers** — select a row for details")
    if not provider_list:
        st.info("No providers with pending LOAs.")
    else:
        p_rows, p_config = build_provider_table_data(provider_list)
        p_event = st.dataframe(
            p_rows,
            column_config=p_config,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            height="content",
        )
        p_sel = getattr(p_event, "selection", None)
        if p_sel and getattr(p_sel, "rows", None):
            idx = p_sel.rows[0]
            if 0 <= idx < len(provider_list):
                st.session_state.selected_provider = provider_list[idx]["provider"]
    st.markdown("---")
    if st.session_state.selected_provider:
        st.markdown("**Provider details**")
        render_provider_detail_panel(st.session_state.selected_provider)
    else:
        st.info("Select a row in the table above to see provider details.")

