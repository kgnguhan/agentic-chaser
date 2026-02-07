"""
Helpers to feed client and provider sentiment into priority score.

Used by predictive_intelligence_agent and workflow_orchestrator so that
priority (and thus delay risk, routing, queue order) reflects sentiment.
"""

from __future__ import annotations

from typing import Tuple

from models.database import CommunicationLog, session_scope

# Labels used by sentiment model (same as sentiment_analysis_agent)
SENTIMENT_LABELS = frozenset({"Positive", "Neutral", "Frustrated", "Confused"})

# Delta to add to base priority (0-10) per label. Negative = lower priority.
LABEL_DELTA = {
    "Frustrated": 0.8,
    "Confused": 0.4,
    "Neutral": 0.0,
    "Positive": -0.2,
}
DEFAULT_DELTA = 0.0

# How many recent client messages to consider
CLIENT_SENTIMENT_LOOKBACK = 5


def get_client_sentiment_for_priority(client_id: str) -> Tuple[str | None, float | None]:
    """
    Get a single representative client sentiment for priority adjustment.

    Queries CommunicationLog for client_id and direction "Client to Advisor",
    latest first (limit N). For rows missing sentiment, runs predict_sentiment
    on message_text and persists to DB. Returns the latest (label, score).
    """
    try:
        from models.ml_models import predict_sentiment
    except Exception:
        return (None, None)

    with session_scope() as db:
        logs = (
            db.query(CommunicationLog)
            .filter(
                CommunicationLog.client_id == client_id,
                CommunicationLog.direction == "Client to Advisor",
            )
            .order_by(CommunicationLog.sent_at.desc())
            .limit(CLIENT_SENTIMENT_LOOKBACK)
            .all()
        )
        if not logs:
            return (None, None)

        for log in logs:
            if log.sentiment_label is not None and log.sentiment_score is not None:
                continue
            if not (log.message_text or "").strip():
                continue
            try:
                label, score = predict_sentiment(log.message_text.strip())
                log.sentiment_label = label
                log.sentiment_score = float(score)
            except (FileNotFoundError, Exception):
                pass
        db.commit()

        # Return latest that has sentiment
        for log in logs:
            if log.sentiment_label is not None and log.sentiment_score is not None:
                return (log.sentiment_label, log.sentiment_score)
    return (None, None)


def get_provider_sentiment_for_priority(loa_id: str) -> Tuple[str | None, float | None]:
    """
    Get provider sentiment for priority adjustment.

    No provider communication storage yet; returns neutral (no adjustment).
    Later: read from provider response log or LOAWorkflow field if added.
    """
    return (None, None)


def sentiment_priority_delta(
    client_label: str | None,
    client_score: float | None,
    provider_label: str | None,
    provider_score: float | None,
) -> float:
    """
    Map sentiment labels to a priority delta (positive = boost priority).

    Frustrated +0.8, Confused +0.4, Neutral 0, Positive -0.2.
    Optional: scale by confidence (score). Currently uses fixed deltas.
    """
    delta = 0.0
    if client_label and client_label in LABEL_DELTA:
        delta += LABEL_DELTA[client_label]
    if provider_label and provider_label in LABEL_DELTA:
        delta += LABEL_DELTA[provider_label]
    return delta


def apply_sentiment_to_priority(
    base_priority: float,
    client_label: str | None,
    client_score: float | None,
    provider_label: str | None,
    provider_score: float | None,
) -> float:
    """
    Apply sentiment delta to base priority and clamp to 0-10.
    """
    delta = sentiment_priority_delta(
        client_label, client_score, provider_label, provider_score
    )
    return float(max(0.0, min(10.0, base_priority + delta)))
