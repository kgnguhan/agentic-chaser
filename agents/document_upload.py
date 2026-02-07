"""
Create DocumentSubmission from advisor-uploaded files.

Saves files to data/uploads and creates a DB row for validation and linking.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from models.database import DocumentSubmission, session_scope

# Directory for uploaded files (relative to project root)
UPLOAD_DIR_NAME = "data/uploads"
# Directory for accepted (validation_passed) documents: data/documents/{client_id}/
ACCEPTED_DOCS_DIR_NAME = "data/documents"

# Document types for dropdowns (fact-find + LOA + provider/post-advice)
UPLOAD_DOCUMENT_TYPES = [
    "Passport",
    "Driving Licence",
    "Utility Bill",
    "Bank Statement",
    "Council Tax",
    "Pension Statement",
    "Investment Statement",
    "Payslip",
    "P60",
    "Provider response",
    "Application Form",
    "Risk Questionnaire",
    "AML Verification",
    "Authority to Proceed",
    "Annual Review",
]


def _project_root(root: Path | None = None) -> Path:
    if root is None:
        return Path(__file__).resolve().parent.parent
    return root


def _upload_dir(root: Path | None = None) -> Path:
    """Return Path to upload directory; create if needed."""
    d = _project_root(root) / UPLOAD_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_dir_name(name: str) -> str:
    """Safe for use as a single path segment (no slashes, no dots)."""
    if not name:
        return "unknown"
    s = re.sub(r"[^\w\-]", "_", name.strip())
    return s[:64] or "unknown"


def copy_to_accepted_storage(
    client_id: str,
    document_type: str,
    document_id: str,
    current_file_path: str,
    project_root: Path | None = None,
) -> str | None:
    """
    Copy an accepted document to client-wise storage: data/documents/{client_id}/{DocumentType}_{date}_{suffix}.ext.
    Returns the new absolute path, or None on failure. Does not delete the original file.
    """
    if not current_file_path or not client_id or not document_type:
        return None
    src = Path(current_file_path).resolve()
    if not src.exists():
        return None
    root = _project_root(project_root)
    base = root / ACCEPTED_DOCS_DIR_NAME
    client_dir = base / _safe_dir_name(client_id)
    client_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix or ""
    doc_type_safe = _safe_dir_name(document_type)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    suffix = (document_id or "")[-8:] if document_id else "upload"
    suffix = re.sub(r"[^\w\-]", "_", suffix)
    name = f"{doc_type_safe}_{date_str}_{suffix}{ext}"
    dest = client_dir / name
    try:
        dest.write_bytes(src.read_bytes())
        return str(dest)
    except OSError:
        return None


def _sanitize_filename(name: str) -> str:
    """Keep only safe characters for a filename."""
    if not name:
        return "upload"
    name = re.sub(r"[^\w\-\.]", "_", name)
    return name[:80] or "upload"


def client_has_accepted_document(client_id: str, document_type: str) -> bool:
    """
    Return True if this client already has an accepted (validation_passed) submission
    for the given document type (e.g. Passport). Used to block duplicate acceptance.
    """
    doc_type = (document_type or "").strip()
    if not client_id or not doc_type:
        return False
    with session_scope() as db:
        exists = (
            db.query(DocumentSubmission)
            .filter(
                DocumentSubmission.client_id == client_id,
                DocumentSubmission.document_type == doc_type,
                DocumentSubmission.validation_passed.is_(True),
            )
            .first()
            is not None
        )
    return exists


def create_document_submission_from_upload(
    client_id: str,
    document_type: str,
    file_bytes: bytes,
    filename: str,
    loa_id: str | None = None,
    project_root: Path | None = None,
) -> dict:
    """
    Save uploaded file to data/uploads, create DocumentSubmission row, return info for UI.

    Returns:
        dict with document_id, file_path, client_id, document_type.
    """
    upload_dir = _upload_dir(project_root)
    doc_id = f"DOC_upload_{uuid4().hex[:12]}"
    safe_name = _sanitize_filename(filename)
    ext = Path(filename).suffix or ""
    if ext and ext not in safe_name:
        safe_name = safe_name + ext
    file_path = upload_dir / f"{doc_id}_{safe_name}"
    file_path.write_bytes(file_bytes)
    # Store path as string (relative to project root or absolute)
    path_str = str(file_path)

    with session_scope() as db:
        doc = DocumentSubmission(
            document_id=doc_id,
            client_id=client_id,
            loa_id=loa_id,
            document_type=(document_type or "Document").strip(),
            file_path=path_str,
            validation_passed=False,
            processed_at=None,
        )
        db.add(doc)
        db.commit()

    return {
        "document_id": doc_id,
        "file_path": path_str,
        "client_id": client_id,
        "document_type": document_type.strip(),
    }
