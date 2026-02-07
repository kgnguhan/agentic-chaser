"""
Agent 4: Sentiment Analysis Agent

Analyses client/advisor message sentiment and optionally stores results
in CommunicationLog (sentiment_label, sentiment_score).
Uses the trained ML model from models.ml_models (predict_sentiment).
"""

from __future__ import annotations

from typing import Any, Dict

from models.database import CommunicationLog, get_session
from models.ml_models import predict_sentiment


def sentiment_analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run sentiment analysis on a message and optionally persist to DB.

    Expected state keys:
        - message_text: str – text to analyse (optional if message_id given)
        - message_id: str – CommunicationLog message_id to load and analyse (optional)
        - client_id: str – required if message_id is used (for loading)
        - persist: bool – if True, update CommunicationLog with sentiment (default True when message_id given)

    Updates state with:
        - sentiment_label: Positive | Neutral | Frustrated | Confused
        - sentiment_score: float (confidence 0–1)
    """
    message_text = state.get("message_text")
    message_id = state.get("message_id")
    client_id = state.get("client_id")
    persist = state.get("persist", bool(message_id))

    if message_id and not message_text:
        with get_session() as db:
            log = (
                db.query(CommunicationLog)
                .filter_by(message_id=message_id, client_id=client_id or "")
                .first()
            )
            if log and log.message_text:
                message_text = log.message_text
            else:
                state["error"] = f"Message {message_id} not found or has no text"
                return state

    if not (message_text or "").strip():
        state["error"] = "No message_text provided and no message_id loaded"
        return state

    try:
        label, score = predict_sentiment(message_text.strip())
    except FileNotFoundError:
        state["error"] = "Sentiment model not found. Run train_sentiment_model() first."
        state["sentiment_label"] = None
        state["sentiment_score"] = None
        return state

    state["sentiment_label"] = label
    state["sentiment_score"] = float(score)

    if persist and message_id and client_id:
        with get_session() as db:
            log = (
                db.query(CommunicationLog)
                .filter_by(message_id=message_id, client_id=client_id)
                .first()
            )
            if log:
                log.sentiment_label = label
                log.sentiment_score = float(score)
                if log.message_text:
                    log.message_length_words = len(log.message_text.split())
                db.commit()

    return state
