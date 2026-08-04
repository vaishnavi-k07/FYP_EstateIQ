"""Vapi webhook receiver — turns a finished call into structured lead features.

Vapi POSTs several event types during a call; only ``end-of-call-report``
carries the final transcript, so everything else is acknowledged and ignored.
On that event the transcript is pulled out, run through the Stage E hybrid
extractor (``nlp.predict``), and returned alongside the call metadata.

Downstream stages attach here as they are built: sentiment/intent (Phase 2)
and lead scoring + SHAP (Phase 4) extend the dict ``process_transcript``
returns, and persistence (Phase 5) writes that dict to the ``leads`` table.
Nothing is stubbed in for them — the seam is ``process_transcript``.

Always returns HTTP 200. Vapi retries non-2xx responses, and a replayed call
would duplicate the lead; failures are logged and reported in the body instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from nlp.predict import get_hybrid_extractor

logger = logging.getLogger(__name__)

webhook_bp = Blueprint("webhook", __name__)

END_OF_CALL = "end-of-call-report"


def extract_transcript(message: Dict[str, Any]) -> Optional[str]:
    """Pulls the transcript out of a Vapi message.

    Vapi has moved the transcript between payload shapes across versions, so
    each known location is tried before falling back to reconstructing it from
    the turn-by-turn message list.
    """
    artifact = message.get("artifact") or {}

    for candidate in (artifact.get("transcript"), message.get("transcript")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    turns: List[str] = []
    for turn in artifact.get("messages") or message.get("messages") or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").lower()
        content = turn.get("message") or turn.get("content") or ""
        if role in ("system", "tool") or not str(content).strip():
            continue
        speaker = "Agent" if role in ("assistant", "bot", "ai") else "Customer"
        turns.append(f"{speaker}: {str(content).strip()}")

    return "\n".join(turns) if turns else None


def call_metadata(message: Dict[str, Any]) -> Dict[str, Any]:
    """Call identifiers and timings worth keeping with the lead record."""
    call = message.get("call") or {}
    customer = message.get("customer") or call.get("customer") or {}
    return {
        "call_id": call.get("id") or message.get("callId"),
        "phone_number": customer.get("number"),
        "started_at": message.get("startedAt") or call.get("startedAt"),
        "ended_at": message.get("endedAt") or call.get("endedAt"),
        "ended_reason": message.get("endedReason"),
        "duration_seconds": message.get("durationSeconds")
        or message.get("durationSeconds".lower()),
    }


def process_transcript(transcript: str) -> Dict[str, Any]:
    """Runs the transcript through the extraction pipeline.

    This is the single place downstream AI stages get added, so the webhook
    itself never has to change again:
        features  <- nlp.predict            (done)
        sentiment/intent                    (Phase 2)
        score/category/explanation          (Phase 4)
    """
    extractor = get_hybrid_extractor()
    return {"features": extractor.extract(transcript)}


@webhook_bp.route("/vapi/webhook", methods=["POST"])
def vapi_webhook():
    payload = request.get_json(silent=True)
    if not payload:
        logger.warning("Webhook called with no JSON body.")
        return jsonify({"status": "ignored", "reason": "no JSON body"}), 200

    message = payload.get("message") or {}
    event_type = message.get("type")

    if event_type != END_OF_CALL:
        logger.info("Ignoring Vapi event: %s", event_type)
        return jsonify({"status": "ignored", "event": event_type}), 200

    metadata = call_metadata(message)
    transcript = extract_transcript(message)
    if not transcript:
        logger.warning("End-of-call report for %s carried no transcript.",
                       metadata.get("call_id"))
        return jsonify({
            "status": "ignored",
            "reason": "no transcript",
            "call": metadata,
        }), 200

    logger.info("Processing call %s (%d chars of transcript)",
                metadata.get("call_id"), len(transcript))
    try:
        result = process_transcript(transcript)
    except Exception:  # noqa: BLE001 - a bad call must not break the endpoint
        logger.exception("Extraction failed for call %s", metadata.get("call_id"))
        # Still 200: Vapi retries non-2xx, and a replay would duplicate the lead.
        return jsonify({
            "status": "error",
            "reason": "extraction failed",
            "call": metadata,
        }), 200

    logger.info("Extracted for call %s: %s", metadata.get("call_id"), result["features"])
    return jsonify({
        "status": "processed",
        "call": metadata,
        "transcript": transcript,
        **result,
    }), 200
