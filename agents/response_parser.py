"""
Agent 5: Response Parser Agent

Parses client or provider free-text responses (emails, messages) into
structured fields: intent, key facts, action items, questions, and
whether they indicate completion (e.g. signed LOA, documents sent).
Uses LLM for robust parsing when available.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from utils.llm_helpers import chat_completion


def response_parser_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a raw message into structured fields for workflow use.

    Expected state keys:
        - raw_message: str – email or message text to parse
        - context: str or dict (optional) – e.g. "LOA follow-up" or {"type": "client_response"}

    Updates state with:
        - parsed_intent: str – e.g. "signed_and_returning", "question", "delay", "complaint"
        - key_facts: list[str]
        - action_items: list[str]
        - contains_question: bool
        - completion_signals: list[str] – e.g. "signed", "attached", "sent"
        - parsed_summary: str – short human-readable summary
    """
    raw_message = (state.get("raw_message") or "").strip()
    context = state.get("context", "")

    if not raw_message:
        state["error"] = "No raw_message provided to response_parser_agent"
        state.setdefault("parsed_intent", "unknown")
        state.setdefault("key_facts", [])
        state.setdefault("action_items", [])
        state.setdefault("contains_question", False)
        state.setdefault("completion_signals", [])
        state.setdefault("parsed_summary", "")
        return state

    if isinstance(context, dict):
        context_str = json.dumps(context)
    else:
        context_str = str(context)

    system_prompt = """You are a UK financial advice operations assistant. Parse the following message from a client or provider into structured data.

Respond with a single JSON object only, no other text. Use exactly these keys:
- "intent": one of: signed_and_returning, documents_sent, question, delay_explanation, complaint, confirmation, other
- "key_facts": array of short fact strings (e.g. "Will send by Friday", "Passport attached")
- "action_items": array of actions we or they need to take
- "contains_question": true if they asked a question
- "completion_signals": array of strings like "signed", "attached", "sent" if they indicate something is done
- "summary": one short sentence summarising the message

Use UK English. Be concise."""

    user_prompt = f"""Parse this message. Context: {context_str or 'general'}.\n\nMessage:\n{raw_message}"""

    try:
        response = chat_completion(system_prompt, user_prompt, temperature=0.2)
    except Exception as e:
        state["error"] = str(e)
        state["parsed_intent"] = "unknown"
        state["key_facts"] = []
        state["action_items"] = []
        state["contains_question"] = bool(re.search(r"\?", raw_message))
        state["completion_signals"] = _fallback_completion_signals(raw_message)
        state["parsed_summary"] = ""
        return state

    parsed = _extract_json(response)
    if not parsed:
        state["parsed_intent"] = "other"
        state["key_facts"] = []
        state["action_items"] = []
        state["contains_question"] = "?" in raw_message
        state["completion_signals"] = _fallback_completion_signals(raw_message)
        state["parsed_summary"] = raw_message[:200] + ("..." if len(raw_message) > 200 else "")
        return state

    state["parsed_intent"] = parsed.get("intent", "other")
    state["key_facts"] = _ensure_list(parsed.get("key_facts"))
    state["action_items"] = _ensure_list(parsed.get("action_items"))
    state["contains_question"] = bool(parsed.get("contains_question", False))
    state["completion_signals"] = _ensure_list(parsed.get("completion_signals"))
    state["parsed_summary"] = (parsed.get("summary") or "").strip() or "No summary."
    return state


def _ensure_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if val is None:
        return []
    return [str(val)]


def _extract_json(text: str) -> Dict[str, Any] | None:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _fallback_completion_signals(text: str) -> List[str]:
    """Simple keyword-based fallback when LLM is unavailable."""
    text_lower = text.lower()
    signals = []
    if "signed" in text_lower or "signature" in text_lower:
        signals.append("signed")
    if "attach" in text_lower or "enclosed" in text_lower or "attached" in text_lower:
        signals.append("attached")
    if "sent" in text_lower or "posted" in text_lower or "uploaded" in text_lower:
        signals.append("sent")
    return signals
