"""
Load test data from data/test CSV files into the database.

Reads: client_profiles.csv, loa_workflows.csv, document_submissions.csv,
communication_logs.csv, post_advice_items.csv.
Idempotent: clears existing rows in dependency order, then bulk inserts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config.config import settings
from models.database import (
    ClientProfile,
    CommunicationLog,
    DocumentSubmission,
    LOAWorkflow,
    PostAdviceItem,
    session_scope,
)
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


def _bool(s: str) -> bool:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return False
    return str(s).strip().lower() in ("true", "1", "yes")


def _float(s) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _int(s) -> int | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _str(s) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    return t if t else None


def _dt(s) -> datetime | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return None


def _normalize_loa_state(s: str | None) -> str:
    """Map legacy or short state names to current descriptive workflow states."""
    if not s:
        return AWAITING_CLIENT_SIGNATURE
    v = s.strip()
    legacy_to_new = {
        "Prepared": AWAITING_CLIENT_SIGNATURE,
        "Awaiting Document Verification": DOCUMENT_AWAITING_VERIFICATION,
        "Client Signed": SIGNED_LOA_READY_FOR_PROVIDER,
        "Provider Submitted": SUBMITTED_TO_PROVIDER,
        "Provider Processing": WITH_PROVIDER_PROCESSING,
        "Incomplete Info": PROVIDER_RESPONSE_INCOMPLETE,
        "Provider Info Incomplete": PROVIDER_RESPONSE_INCOMPLETE,
        "Info Received": PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT,
        "Complete": CASE_COMPLETE,
    }
    return legacy_to_new.get(v, v)


def load_test_data() -> None:
    """Load all CSVs from data/test into the database. Idempotent."""
    test_dir: Path = settings.paths.test_data_dir
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")

    with session_scope() as db:
        # Clear in reverse dependency order (session_scope commits on exit)
        db.query(CommunicationLog).delete()
        db.query(PostAdviceItem).delete()
        db.query(DocumentSubmission).delete()
        db.query(LOAWorkflow).delete()
        db.query(ClientProfile).delete()

    # Load and insert in dependency order
    # 1. Client profiles
    df_clients = pd.read_csv(test_dir / "client_profiles.csv")
    with session_scope() as db:
        for _, row in df_clients.iterrows():
            db.add(ClientProfile(
                client_id=str(row["client_id"]),
                name=str(row["name"]),
                age=_int(row.get("age")),
                employment_type=_str(row.get("employment_type")),
                annual_income=_float(row.get("annual_income")),
                existing_pensions_count=_int(row.get("existing_pensions_count")) or 0,
                risk_profile=_str(row.get("risk_profile")),
                communication_preference=_str(row.get("communication_preference")) or "Email",
                document_responsiveness=_str(row.get("document_responsiveness")),
            ))

    # 2. LOA workflows
    df_loas = pd.read_csv(test_dir / "loa_workflows.csv")
    with session_scope() as db:
        for _, row in df_loas.iterrows():
            db.add(LOAWorkflow(
                loa_id=str(row["loa_id"]),
                client_id=str(row["client_id"]),
                provider=str(row["provider"]),
                case_type=_str(row.get("case_type")) or "Pension Consolidation",
                current_state=_normalize_loa_state(_str(row.get("current_state")) or AWAITING_CLIENT_SIGNATURE),
                priority_score=_float(row.get("priority_score")) or 0.0,
                days_in_current_state=_int(row.get("days_in_current_state")) or 0,
                sla_days=_int(row.get("sla_days")) or 15,
                sla_days_remaining=_int(row.get("sla_days_remaining")),
                document_quality_score=_float(row.get("document_quality_score")) or 75.0,
                signature_verified=_bool(row.get("signature_verified")),
                reference_number=_str(row.get("reference_number")),
                needs_advisor_intervention=_bool(row.get("needs_advisor_intervention")),
                sla_overdue=_bool(row.get("sla_overdue")),
            ))

    # 3. Document submissions
    doc_path = test_dir / "document_submissions.csv"
    if doc_path.exists():
        df_docs = pd.read_csv(doc_path)
        with session_scope() as db:
            for _, row in df_docs.iterrows():
                db.add(DocumentSubmission(
                    document_id=str(row["document_id"]),
                    client_id=str(row["client_id"]),
                    document_type=str(row["document_type"]),
                    document_subtype=_str(row.get("document_subtype")),
                    ocr_confidence_score=_float(row.get("ocr_confidence_score")),
                    quality_issues=_str(row.get("quality_issues")),
                    validation_passed=_bool(row.get("validation_passed")),
                    manual_review_required=_bool(row.get("manual_review_required")),
                    file_path=_str(row.get("file_path")),
                ))

    # 4. Communication logs
    comm_path = test_dir / "communication_logs.csv"
    if comm_path.exists():
        df_comm = pd.read_csv(comm_path)
        with session_scope() as db:
            for _, row in df_comm.iterrows():
                db.add(CommunicationLog(
                    message_id=str(row["message_id"]),
                    client_id=str(row["client_id"]),
                    direction=str(row["direction"]),
                    channel=str(row["channel"]),
                    message_text=_str(row.get("message_text")),
                    sentiment_label=_str(row.get("sentiment_label")),
                    sentiment_score=_float(row.get("sentiment_score")),
                    message_length_words=_int(row.get("message_length_words")) or 0,
                    contains_question=_bool(row.get("contains_question")),
                    response_time_hours=_float(row.get("response_time_hours")),
                ))

    # 5. Post advice items
    pa_path = test_dir / "post_advice_items.csv"
    if pa_path.exists():
        df_pa = pd.read_csv(pa_path)
        with session_scope() as db:
            for _, row in df_pa.iterrows():
                db.add(PostAdviceItem(
                    item_id=str(row["item_id"]),
                    client_id=str(row["client_id"]),
                    item_type=str(row["item_type"]),
                    current_state=_str(row.get("current_state")) or "Pending",
                    days_outstanding=_int(row.get("days_outstanding")) or 0,
                    days_until_deadline=_int(row.get("days_until_deadline")),
                    completion_percentage=_float(row.get("completion_percentage")) or 0.0,
                    sent_via=_str(row.get("sent_via")),
                    opened=_bool(row.get("opened")),
                    last_interaction_date=_dt(row.get("last_interaction_date")),
                    rejection_reason=_str(row.get("rejection_reason")),
                    resubmission_count=_int(row.get("resubmission_count")) or 0,
                ))


if __name__ == "__main__":
    load_test_data()
    print("Test data loaded from data/test.")
