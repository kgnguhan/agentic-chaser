"""
Agent 3: Intelligent Document Processing Agent

Responsibilities:
- Read document metadata from DB
- (Optionally) run OCR on local files (requires pytesseract + Tesseract engine if run_ocr=True)
- Classify basic quality (good / needs review)
- Decide whether validation passes or manual review is required

Works with DocumentSubmission (file_path, ocr_confidence_score, quality_issues).
OCR is optional: if pytesseract or Tesseract engine is not available, run_ocr is a no-op.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from agents.document_upload import copy_to_accepted_storage
from models.database import DocumentSubmission, session_scope

# ---------- Constants ----------

OCR_CONFIDENCE_FAIL = 60.0   # Below this: validation fails + manual review
OCR_CONFIDENCE_REVIEW = 75.0 # Below this: manual review recommended
OCR_CONFIDENCE_ID_STRICT = 80.0  # ID documents require this minimum

ID_DOCUMENT_TYPES = frozenset({"passport", "driving licence", "driving license"})

# Cached availability of Tesseract OCR (True / False)
_ocr_available: bool | None = None
# Set by run_ocr_on_document on exception so agent can surface ocr_runtime_error
_last_ocr_error: str | None = None


def _get_ocr() -> bool:
    """Return True if pytesseract and Tesseract engine are available."""
    global _ocr_available
    if _ocr_available is None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _ocr_available = True
        except Exception:
            _ocr_available = False
    return _ocr_available is True


def document_processing_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document for a given document_id in state.

    Expected state keys:
        - document_id: ID of DocumentSubmission row
        - run_ocr: bool (optional, default False) – if True, runs OCR on file_path when available

    Updates state with:
        - ocr_text
        - ocr_confidence_score
        - validation_passed
        - manual_review_required
        - quality_issues
    """
    document_id = state.get("document_id")
    run_ocr = bool(state.get("run_ocr", False))

    if not document_id:
        state["error"] = "document_id not provided to document_processing_agent"
        return state

    with session_scope() as db:
        doc: DocumentSubmission | None = (
            db.query(DocumentSubmission).filter_by(document_id=document_id).first()
        )

        if doc is None:
            state["error"] = f"Document {document_id} not found"
            return state

        ocr_text, ocr_conf = None, None
        if run_ocr and doc.file_path:
            ocr_text, ocr_conf = run_ocr_on_document(doc.file_path)
            if ocr_conf is not None:
                doc.ocr_confidence_score = float(ocr_conf)
        # When we tried OCR but got no score, distinguish "engine missing" vs "runtime error" vs "not run"
        global _last_ocr_error
        has_engine = _get_ocr()
        ocr_engine_unavailable = bool(
            run_ocr and doc.file_path and doc.ocr_confidence_score is None and not has_engine
        )
        ocr_runtime_error = bool(
            run_ocr and doc.file_path and doc.ocr_confidence_score is None and has_engine and _last_ocr_error is not None
        )
        if ocr_runtime_error:
            _last_ocr_error = None  # consume so next run is clean

        validation_passed, manual_review_required, issues = evaluate_document_quality(
            document_type=doc.document_type or "",
            ocr_confidence=doc.ocr_confidence_score,
            existing_issues=doc.quality_issues or "",
            ocr_engine_unavailable=ocr_engine_unavailable,
            ocr_runtime_error=ocr_runtime_error,
        )

        doc.validation_passed = validation_passed
        doc.manual_review_required = manual_review_required
        doc.quality_issues = issues or ""
        doc.processed_at = datetime.utcnow()

        # Store accepted documents in client-wise folder with clear filenames
        if validation_passed and doc.file_path:
            new_path = copy_to_accepted_storage(
                client_id=doc.client_id,
                document_type=doc.document_type or "",
                document_id=doc.document_id,
                current_file_path=doc.file_path,
            )
            if new_path:
                doc.file_path = new_path

        # Propagate to state
        state["ocr_text"] = ocr_text
        state["ocr_confidence_score"] = doc.ocr_confidence_score
        state["validation_passed"] = validation_passed
        state["manual_review_required"] = manual_review_required
        state["quality_issues"] = doc.quality_issues

    score = state.get("ocr_confidence_score")
    v = state.get("validation_passed")
    issues = state.get("quality_issues") or "—"
    state["reasoning"] = f"OCR confidence {score}; validation_passed {v}; issues: {issues}."
    return state


def _load_image_for_ocr(path: Path):
    """Load a PIL Image from path (image file or first page of PDF). Returns None if unsupported or failed."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(path, first_page=1, last_page=1, dpi=200)
            return pages[0] if pages else None
        except ImportError:
            return None
        except Exception:
            return None
    try:
        from PIL import Image
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def run_ocr_on_document(file_path: str) -> Tuple[str | None, float | None]:
    """
    Run OCR on a document image or PDF using Tesseract (pytesseract).

    Args:
        file_path: Path to image or PDF on disk.

    Returns:
        (full_text, avg_confidence) or (None, None) if OCR is unavailable, file missing, or fails.
    """
    global _last_ocr_error
    path = Path(file_path).resolve()
    if not path.exists():
        return None, None

    if not _get_ocr():
        return None, None

    image = _load_image_for_ocr(path)
    if image is None:
        _last_ocr_error = "unsupported_file"
        return None, None

    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(image, output_type=Output.DICT, lang="eng")
    except Exception as e:
        _last_ocr_error = "ocr_error"
        return None, None

    texts: list[str] = []
    scores: list[float] = []
    n = len(data.get("text", []))
    for i in range(n):
        word = (data.get("text") or [])[i]
        conf = (data.get("conf") or [0])[i]
        if word and word.strip():
            texts.append(str(word).strip())
        try:
            c = float(conf)
            if c > 0:
                scores.append(c)
        except (TypeError, ValueError):
            pass

    if not texts and not scores:
        return None, None

    full_text = " ".join(texts) if texts else ""
    avg_conf = float(sum(scores) / len(scores)) if scores else None
    return (full_text if full_text else None), avg_conf


def evaluate_document_quality(
    document_type: str,
    ocr_confidence: float | None,
    existing_issues: str,
    ocr_engine_unavailable: bool = False,
    ocr_runtime_error: bool = False,
) -> Tuple[bool, bool, str]:
    """
    Heuristic document quality and validation.

    - OCR confidence < OCR_CONFIDENCE_FAIL → validation fail + manual review
    - OCR confidence < OCR_CONFIDENCE_REVIEW → manual review
    - ID documents (passport, driving licence) use a stricter threshold (OCR_CONFIDENCE_ID_STRICT)
    - Existing quality_issues are preserved and new flags appended.
    - When ocr_confidence is None: ocr_engine_unavailable → "ocr_unavailable", ocr_runtime_error → "ocr_runtime_error", else "no_ocr_run".

    Returns:
        (validation_passed, manual_review_required, combined_issues)
    """
    issues: list[str] = []
    if existing_issues:
        issues.extend([i.strip() for i in existing_issues.split(",") if i.strip()])

    if ocr_confidence is None:
        if issues:
            return False, True, ",".join(issues)
        if ocr_engine_unavailable:
            no_ocr_issue = "ocr_unavailable"
        elif ocr_runtime_error:
            no_ocr_issue = "ocr_runtime_error"
        else:
            no_ocr_issue = "no_ocr_run"
        return False, True, ",".join(issues + [no_ocr_issue])

    manual_review_required = False
    validation_passed = True

    if ocr_confidence < OCR_CONFIDENCE_FAIL:
        validation_passed = False
        manual_review_required = True
        issues.append("very_low_ocr_confidence")
    elif ocr_confidence < OCR_CONFIDENCE_REVIEW:
        manual_review_required = True
        issues.append("low_ocr_confidence")

    # Case-insensitive match for ID documents
    doc_type_normalized = (document_type or "").strip().lower()
    if doc_type_normalized in ID_DOCUMENT_TYPES and ocr_confidence < OCR_CONFIDENCE_ID_STRICT:
        manual_review_required = True
        issues.append("id_doc_low_confidence")

    combined_issues = ",".join(sorted(set(issues))) if issues else ""
    return validation_passed, manual_review_required, combined_issues
