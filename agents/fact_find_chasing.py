"""
Fact-find document chasing: required categories and queue of clients missing documents.

Used by the chaser runner and dashboard to chase clients for proof of identity,
proof of address, pension statements, investment valuations, etc.
"""

from __future__ import annotations

from typing import List

from models.database import ClientProfile, DocumentSubmission, LOAWorkflow, session_scope
from orchestration.workflow_states import CASE_COMPLETE


# Required fact-find categories (ordered for display and chasing)
FACT_FIND_CATEGORIES = [
    "Proof of identity (passport, driving licence)",
    "Proof of address (utility bill, bank statement)",
    "Existing pension statements",
    "Investment valuations",
    "Protection policy documents",
    "Payslips for pension contribution calculations",
    "P60s for tax planning",
]

# Map DocumentSubmission.document_type (case-insensitive) to category index in FACT_FIND_CATEGORIES.
# One document type can satisfy one category only.
DOCUMENT_TYPE_TO_CATEGORY: dict[str, int] = {
    "passport": 0,
    "driving licence": 0,
    "driving license": 0,
    "utility bill": 1,
    "council tax": 1,
    "bank statement": 1,
    "pension statement": 2,
    "investment statement": 3,
    "protection": 4,
    "policy": 4,
    "payslip": 5,
    "p60": 6,
}


def _doc_type_to_category_index(document_type: str) -> int | None:
    """Return category index for this document_type, or None if not a fact-find type."""
    if not document_type:
        return None
    key = (document_type or "").strip().lower()
    return DOCUMENT_TYPE_TO_CATEGORY.get(key)


def document_type_to_category_label(document_type: str) -> str:
    """Return human-readable fact-find category name for a document_type, or the document_type itself if not mapped."""
    idx = _doc_type_to_category_index(document_type)
    if idx is not None:
        return FACT_FIND_CATEGORIES[idx]
    return (document_type or "document").strip()


REQUIRED_FACT_FIND_COUNT = len(FACT_FIND_CATEGORIES)


def get_fact_find_status(client_id: str) -> dict:
    """
    Return received/required counts and lists for partial submission status.
    Keys: received_count, required_count, missing_documents, received_categories.
    """
    received_indices: set[int] = set()
    with session_scope() as db:
        docs = (
            db.query(DocumentSubmission.document_type)
            .filter(
                DocumentSubmission.client_id == client_id,
                DocumentSubmission.validation_passed == True,
            )
            .all()
        )
        for (doc_type,) in docs:
            idx = _doc_type_to_category_index(doc_type or "")
            if idx is not None:
                received_indices.add(idx)
    required_count = REQUIRED_FACT_FIND_COUNT
    received_count = len(received_indices)
    missing = [
        FACT_FIND_CATEGORIES[i]
        for i in range(required_count)
        if i not in received_indices
    ]
    received_categories = [
        FACT_FIND_CATEGORIES[i]
        for i in range(required_count)
        if i in received_indices
    ]
    return {
        "received_count": received_count,
        "required_count": required_count,
        "missing_documents": missing,
        "received_categories": received_categories,
    }


def get_missing_fact_find_documents(client_id: str) -> List[str]:
    """
    Return list of required fact-find category names that the client has not yet
    supplied (no DocumentSubmission with validation_passed=True mapping to that category).
    """
    return get_fact_find_status(client_id)["missing_documents"]


def get_fact_find_chase_queue(
    limit: int | None = 50,
    clients_with_active_loa_only: bool = True,
) -> List[dict]:
    """
    Return list of clients who are missing at least one fact-find document.
    Each entry: client_id, client_name, missing_documents (list of str).
    Optionally restrict to clients that have at least one active (non-Complete) LOA.
    """
    with session_scope() as db:
        if clients_with_active_loa_only:
            # Clients with at least one active LOA
            active_client_ids = {
                row[0]
                for row in db.query(LOAWorkflow.client_id)
                .filter(LOAWorkflow.current_state != CASE_COMPLETE)
                .distinct()
                .all()
            }
            clients = (
                db.query(ClientProfile)
                .filter(ClientProfile.client_id.in_(active_client_ids))
                .order_by(ClientProfile.name)
                .all()
            )
        else:
            clients = db.query(ClientProfile).order_by(ClientProfile.name).all()

        queue: List[dict] = []
        for c in clients:
            status = get_fact_find_status(c.client_id)
            missing = status["missing_documents"]
            if missing:
                queue.append({
                    "client_id": c.client_id,
                    "client_name": c.name or "Unknown",
                    "missing_documents": missing,
                    "received_count": status["received_count"],
                    "required_count": status["required_count"],
                })
        # Order by number of missing docs (desc), then name
        queue.sort(key=lambda x: (-len(x["missing_documents"]), x["client_name"]))
        if limit is not None:
            queue = queue[:limit]
    return queue


def get_fact_find_documents_awaiting_verification(limit: int | None = 50) -> List[dict]:
    """
    Return fact-find documents (loa_id null) that have not yet been processed (processed_at null).
    Each entry: document_id, client_id, document_type, client_name (optional).
    """
    with session_scope() as db:
        query = (
            db.query(DocumentSubmission)
            .filter(
                DocumentSubmission.loa_id.is_(None),
                DocumentSubmission.processed_at.is_(None),
            )
        )
        if limit is not None:
            query = query.limit(limit)
        docs = query.all()
        result = []
        for doc in docs:
            client_name = doc.client.name if doc.client else "Unknown"
            result.append({
                "document_id": doc.document_id,
                "client_id": doc.client_id,
                "document_type": doc.document_type or "",
                "client_name": client_name,
            })
    return result
