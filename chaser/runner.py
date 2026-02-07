"""
Autonomic chaser runner: tick time and run one cycle over active LOAs,
then fact-find chase and post-advice chase.
"""

from __future__ import annotations

import logging
from typing import List

from agents.client_comms import client_communication_agent
from agents.document_processing import document_processing_agent
from agents.fact_find_chasing import (
    document_type_to_category_label,
    get_fact_find_chase_queue,
    get_fact_find_documents_awaiting_verification,
)
from agents.workflow_orchestrator import (
    get_post_advice_chase_queue,
    persist_escalation,
    tick_loa_time,
)
from models.database import LOAWorkflow, session_scope
from orchestration import get_chaser_graph, initial_state
from orchestration.workflow_states import CASE_COMPLETE

logger = logging.getLogger(__name__)

PROVIDER_ACTIONS = (
    "provider_follow_up",
    "provider_urgent_follow_up",
    "provider_clarification",
)
CLIENT_ACTIONS = ("client_communication", "client_notification")

FACT_FIND_QUEUE_LIMIT = 50
POST_ADVICE_QUEUE_LIMIT = 50
FACT_FIND_VERIFICATION_LIMIT = 50


def _get_active_loa_ids(skip_escalated: bool = True) -> List[str]:
    """Return loa_ids for LOAs that are not Complete and optionally not escalated."""
    with session_scope() as db:
        q = db.query(LOAWorkflow.loa_id).filter(LOAWorkflow.current_state != CASE_COMPLETE)
        if skip_escalated:
            q = q.filter(LOAWorkflow.escalated_at.is_(None))
        return [row[0] for row in q.all()]


def run_chaser_cycle(skip_escalated: bool = True) -> None:
    """
    Run one chaser cycle: (1) tick time for all active LOAs,
    (2) for each active (and non-escalated) LOA invoke the graph,
    (3) if result is escalate_to_advisor, persist escalation and skip provider chase;
    (4) if result is a provider action, the graph already ran provider_comms — just log.
    """
    updated = tick_loa_time()
    logger.info("tick_loa_time updated %s LOAs", updated)

    loa_ids = _get_active_loa_ids(skip_escalated=skip_escalated)
    if loa_ids:
        graph = get_chaser_graph()
        for loa_id in loa_ids:
            try:
                result = graph.invoke(initial_state(loa_id=loa_id))
                next_action = (result.get("next_action") or "").strip()

                if next_action == "escalate_to_advisor":
                    if persist_escalation(loa_id):
                        logger.info("Escalated LOA %s to advisor", loa_id)
                    continue

                if next_action in PROVIDER_ACTIONS:
                    logger.info(
                        "Chaser ran provider action %s for LOA %s",
                        next_action,
                        loa_id,
                    )
                if next_action in CLIENT_ACTIONS:
                    logger.info(
                        "Chaser ran client action %s for LOA %s",
                        next_action,
                        loa_id,
                    )
            except Exception as e:
                logger.exception("Chaser cycle failed for LOA %s: %s", loa_id, e)
    else:
        logger.info("No active LOAs to chase")

    # Fact-find phase: chase clients missing fact-find documents
    fact_find_queue = get_fact_find_chase_queue(limit=FACT_FIND_QUEUE_LIMIT)
    for entry in fact_find_queue:
        try:
            state = {
                "client_id": entry["client_id"],
                "communication_type": "fact_find_document_request",
                "context": {"missing_documents": entry["missing_documents"]},
            }
            client_communication_agent(state)
            logger.info(
                "Chaser sent fact-find document request to client %s (%s)",
                entry["client_id"],
                entry["client_name"],
            )
        except Exception as e:
            logger.exception(
                "Fact-find chase failed for client %s: %s",
                entry.get("client_id"),
                e,
            )

    # Post-advice phase: chase outstanding post-advice items
    post_advice_queue = get_post_advice_chase_queue(limit=POST_ADVICE_QUEUE_LIMIT)
    for entry in post_advice_queue:
        try:
            state = {
                "client_id": entry["client_id"],
                "communication_type": "post_advice_reminder",
                "context": {
                    "item_type": entry["item_type"],
                    "days_outstanding": entry["days_outstanding"],
                    "deadline_days": entry.get("days_until_deadline") or 14,
                },
            }
            client_communication_agent(state)
            logger.info(
                "Chaser sent post-advice reminder for item %s (client %s)",
                entry["item_id"],
                entry["client_id"],
            )
        except Exception as e:
            logger.exception(
                "Post-advice chase failed for item %s: %s",
                entry.get("item_id"),
                e,
            )

    # Fact-find document verification phase: run OCR/validation on submitted fact-find docs; on failure chase with reason
    awaiting_verification = get_fact_find_documents_awaiting_verification(limit=FACT_FIND_VERIFICATION_LIMIT)
    for entry in awaiting_verification:
        try:
            state = {
                "document_id": entry["document_id"],
                "client_id": entry["client_id"],
                "run_ocr": True,
            }
            state = document_processing_agent(state)
            if state.get("error"):
                logger.warning("Fact-find doc verification failed for %s: %s", entry["document_id"], state["error"])
                continue
            if not state.get("validation_passed", False):
                category_label = document_type_to_category_label(entry["document_type"])
                client_communication_agent({
                    "client_id": entry["client_id"],
                    "communication_type": "document_request",
                    "context": {
                        "missing_documents": [category_label],
                        "quality_issues": state.get("quality_issues") or "",
                        "message": "Document did not pass verification. Please resubmit.",
                    },
                })
                logger.info(
                    "Chaser sent document resubmit request for %s (client %s), reason: %s",
                    entry["document_id"],
                    entry["client_id"],
                    state.get("quality_issues", ""),
                )
        except Exception as e:
            logger.exception(
                "Fact-find document verification failed for %s: %s",
                entry.get("document_id"),
                e,
            )
