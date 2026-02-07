"""
Database models and session management for Agentic Chaser.

Defines 5 core tables:
- ClientProfile
- LOAWorkflow
- DocumentSubmission
- PostAdviceItem
- CommunicationLog
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from config.config import settings

# ---------- Engine & Base ----------

engine = create_engine(
    settings.db.url,
    echo=settings.app.debug,
    pool_pre_ping=True,
)

Base = declarative_base()


# ========== ORM MODELS ==========


class ClientProfile(Base):
    """Client profile with demographics and preferences."""

    __tablename__ = "client_profiles"

    client_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    employment_type = Column(String, nullable=True)
    annual_income = Column(Float, nullable=True)
    existing_pensions_count = Column(Integer, default=0)
    risk_profile = Column(String, nullable=True)
    communication_preference = Column(String, default="Email")
    document_responsiveness = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    loa_workflows = relationship("LOAWorkflow", back_populates="client")
    documents = relationship("DocumentSubmission", back_populates="client")
    communications = relationship("CommunicationLog", back_populates="client")
    post_advice_items = relationship("PostAdviceItem", back_populates="client")


class LOAWorkflow(Base):
    """LOA (Letter of Authority) workflow tracking."""

    __tablename__ = "loa_workflows"

    loa_id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("client_profiles.client_id"), nullable=False)
    provider = Column(String, nullable=False)
    case_type = Column(String, default="Pension Consolidation")

    # Workflow state (see orchestration.workflow_states for canonical names)
    current_state = Column(String, default="Awaiting Client Signature")
    # Client-side: Awaiting Client Signature, Document Awaiting Verification, Client Documents Rejected
    # Then: Signed LOA - Ready for Provider
    # Provider-side: Submitted to Provider, With Provider - Processing, Provider Response Incomplete
    # Hand-back: Provider Info Received - Notify Client. Terminal: Case Complete

    # Priority & SLA tracking
    priority_score = Column(Float, default=0.0)
    days_in_current_state = Column(Integer, default=0)
    sla_days = Column(Integer, default=15)
    sla_days_remaining = Column(Integer, nullable=True)

    # Document tracking
    document_quality_score = Column(Float, default=75.0)
    signature_verified = Column(Boolean, default=False)
    reference_number = Column(String, nullable=True)

    # Flags
    needs_advisor_intervention = Column(Boolean, default=False)
    sla_overdue = Column(Boolean, default=False)
    escalated_at = Column(DateTime, nullable=True)
    pending_document_id = Column(String, nullable=True)  # document_id awaiting verification (no FK to avoid circular dependency)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    client = relationship("ClientProfile", back_populates="loa_workflows")


class DocumentSubmission(Base):
    """Document submission tracking with OCR/validation results."""

    __tablename__ = "document_submissions"

    document_id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("client_profiles.client_id"), nullable=False)
    loa_id = Column(String, ForeignKey("loa_workflows.loa_id"), nullable=True)

    document_type = Column(String, nullable=False)
    document_subtype = Column(String, nullable=True)

    # OCR & validation
    ocr_confidence_score = Column(Float, nullable=True)
    quality_issues = Column(Text, nullable=True)
    validation_passed = Column(Boolean, default=False)
    manual_review_required = Column(Boolean, default=False)

    file_path = Column(String, nullable=True)

    submitted_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    client = relationship("ClientProfile", back_populates="documents")


class PostAdviceItem(Base):
    """Post-advice task/item tracking (forms, questionnaires, AML, etc.)."""

    __tablename__ = "post_advice_items"

    item_id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("client_profiles.client_id"), nullable=False)

    item_type = Column(String, nullable=False)
    # Types: Application Form, Risk Questionnaire, AML Verification, Annual Review

    current_state = Column(String, default="Pending")
    # States: Pending, Sent, Opened, Partially Completed, Completed, Rejected, Resubmitted

    days_outstanding = Column(Integer, default=0)
    days_until_deadline = Column(Integer, nullable=True)
    completion_percentage = Column(Float, default=0.0)

    sent_via = Column(String, nullable=True)
    opened = Column(Boolean, default=False)
    last_interaction_date = Column(DateTime, nullable=True)

    rejection_reason = Column(Text, nullable=True)
    resubmission_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    client = relationship("ClientProfile", back_populates="post_advice_items")


class CommunicationLog(Base):
    """Communication log with sentiment tracking."""

    __tablename__ = "communication_logs"

    message_id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("client_profiles.client_id"), nullable=False)

    direction = Column(String, nullable=False)  # "Client to Advisor" / "Advisor to Client"
    channel = Column(String, nullable=False)  # Email, SMS, WhatsApp, Phone

    message_text = Column(Text, nullable=True)
    sentiment_label = Column(String, nullable=True)  # Positive, Neutral, Frustrated, Confused
    sentiment_score = Column(Float, nullable=True)

    message_length_words = Column(Integer, default=0)
    contains_question = Column(Boolean, default=False)
    response_time_hours = Column(Float, nullable=True)

    sent_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    client = relationship("ClientProfile", back_populates="communications")


# ---------- Session Factory ----------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---------- Session Helpers ----------


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as db:
            client = ClientProfile(client_id="C001", name="Alice")
            db.add(client)
            # Commit happens automatically
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _migrate_escalated_at() -> None:
    """Add escalated_at to loa_workflows if missing (for existing DBs)."""
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(
                text(
                    "ALTER TABLE loa_workflows ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMP"
                )
            )
        elif engine.dialect.name == "sqlite":
            try:
                conn.execute(
                    text(
                        "ALTER TABLE loa_workflows ADD COLUMN escalated_at DATETIME"
                    )
                )
            except Exception:
                pass
        conn.commit()


def _migrate_document_loa_link() -> None:
    """Add loa_id to document_submissions and pending_document_id to loa_workflows (for existing DBs)."""
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(
                text(
                    "ALTER TABLE document_submissions ADD COLUMN IF NOT EXISTS loa_id VARCHAR REFERENCES loa_workflows(loa_id)"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE loa_workflows ADD COLUMN IF NOT EXISTS pending_document_id VARCHAR"
                )
            )
        elif engine.dialect.name == "sqlite":
            try:
                conn.execute(
                    text(
                        "ALTER TABLE document_submissions ADD COLUMN loa_id VARCHAR REFERENCES loa_workflows(loa_id)"
                    )
                )
            except Exception:
                pass
            try:
                conn.execute(
                    text(
                        "ALTER TABLE loa_workflows ADD COLUMN pending_document_id VARCHAR"
                    )
                )
            except Exception:
                pass
        conn.commit()


def _migrate_workflow_states_to_descriptive() -> None:
    """Rename all workflow states to descriptive names (single source of truth in orchestration.workflow_states)."""
    # Order matters: map legacy/short names to new descriptive names (no new name appears as left-hand side)
    updates = [
        ("Incomplete Info", "Provider Response Incomplete"),
        ("Prepared", "Awaiting Client Signature"),
        ("Awaiting Document Verification", "Document Awaiting Verification"),
        ("Client Signed", "Signed LOA - Ready for Provider"),
        ("Provider Submitted", "Submitted to Provider"),
        ("Provider Processing", "With Provider - Processing"),
        ("Provider Info Incomplete", "Provider Response Incomplete"),
        ("Info Received", "Provider Info Received - Notify Client"),
        ("Complete", "Case Complete"),
    ]
    with engine.connect() as conn:
        for old_val, new_val in updates:
            conn.execute(
                text(
                    "UPDATE loa_workflows SET current_state = :new WHERE current_state = :old"
                ),
                {"new": new_val, "old": old_val},
            )
        conn.commit()


def init_db() -> None:
    """Create all database tables and run any pending migrations."""
    Base.metadata.create_all(bind=engine)
    _migrate_escalated_at()
    _migrate_document_loa_link()
    _migrate_workflow_states_to_descriptive()