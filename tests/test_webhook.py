"""Vapi webhook — payload parsing and endpoint behaviour.

Uses a fake extractor throughout, so no neural model is loaded.
"""

from __future__ import annotations

import pytest

from conftest import end_of_call_payload
from routes.webhook import call_metadata, extract_transcript


# --------------------------------------------------------------------------- #
# Transcript extraction
# --------------------------------------------------------------------------- #


def test_transcript_from_artifact():
    message = {"artifact": {"transcript": "Customer: hello"}}
    assert extract_transcript(message) == "Customer: hello"


def test_transcript_from_top_level():
    """Older payload shape puts the transcript at the top level."""
    assert extract_transcript({"transcript": "Customer: hello"}) == "Customer: hello"


def test_artifact_transcript_wins_over_top_level():
    message = {"transcript": "old", "artifact": {"transcript": "new"}}
    assert extract_transcript(message) == "new"


def test_transcript_rebuilt_from_messages():
    message = {
        "artifact": {
            "messages": [
                {"role": "system", "message": "You are a sales agent."},
                {"role": "assistant", "message": "What are you looking for?"},
                {"role": "user", "message": "A 2 BHK in Pune."},
            ]
        }
    }
    assert extract_transcript(message) == (
        "Agent: What are you looking for?\nCustomer: A 2 BHK in Pune."
    )


def test_system_and_tool_turns_are_dropped():
    message = {"artifact": {"messages": [
        {"role": "system", "message": "prompt"},
        {"role": "tool", "message": "{}"},
    ]}}
    assert extract_transcript(message) is None


def test_blank_transcript_is_none():
    assert extract_transcript({"artifact": {"transcript": "   "}}) is None
    assert extract_transcript({}) is None


def test_malformed_message_entries_are_skipped():
    message = {"artifact": {"messages": ["not a dict", {"role": "user", "message": "hi"}]}}
    assert extract_transcript(message) == "Customer: hi"


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def test_call_metadata():
    message = end_of_call_payload()["message"]
    meta = call_metadata(message)
    assert meta["call_id"] == "call_123"
    assert meta["phone_number"] == "+919876543210"
    assert meta["ended_reason"] == "customer-ended-call"


def test_call_metadata_tolerates_empty_message():
    meta = call_metadata({})
    assert meta["call_id"] is None
    assert meta["phone_number"] is None


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #


def test_end_of_call_is_processed(client, fake_extractor):
    response = client.post("/vapi/webhook", json=end_of_call_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "processed"
    assert body["features"]["city"] == "Pune"
    assert body["call"]["call_id"] == "call_123"
    assert fake_extractor.calls == ["Customer: I want a 2 BHK in Pune."]


@pytest.mark.parametrize(
    "event", ["status-update", "transcript", "speech-update", "hang", None]
)
def test_non_final_events_are_ignored(client, fake_extractor, event):
    response = client.post("/vapi/webhook", json={"message": {"type": event}})
    assert response.status_code == 200
    assert response.get_json()["status"] == "ignored"
    assert fake_extractor.calls == []


def test_missing_transcript_is_ignored(client, fake_extractor):
    payload = end_of_call_payload()
    payload["message"]["artifact"] = {}
    response = client.post("/vapi/webhook", json=payload)
    assert response.status_code == 200
    assert response.get_json()["reason"] == "no transcript"
    assert fake_extractor.calls == []


def test_empty_body_is_handled(client):
    response = client.post("/vapi/webhook", json=None)
    assert response.status_code == 200
    assert response.get_json()["status"] == "ignored"


def test_malformed_json_does_not_500(client):
    response = client.post(
        "/vapi/webhook", data="not json", content_type="application/json"
    )
    assert response.status_code == 200


def test_extraction_failure_still_returns_200(client, monkeypatch):
    """Vapi retries non-2xx, and a replay would duplicate the lead."""
    from routes import webhook

    class Exploding:
        def extract(self, text):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(webhook, "get_hybrid_extractor", lambda: Exploding())
    response = client.post("/vapi/webhook", json=end_of_call_payload())
    assert response.status_code == 200
    assert response.get_json()["status"] == "error"


def test_health_and_home(client):
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"
