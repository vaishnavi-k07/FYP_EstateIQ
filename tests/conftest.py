"""Shared fixtures.

Loading the real MuRIL model takes seconds and hundreds of MB, so tests that
only exercise rules, parsing or HTTP plumbing must not pay for it. Anything
needing the real model uses the session-scoped ``hybrid`` fixture and is marked
``slow``; everything else uses fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: loads the real NER model (seconds, hundreds of MB)"
    )


@pytest.fixture(scope="session")
def rule_extractor():
    """The rule extractor — CSV catalogs only, no neural model."""
    from nlp.extractor import NLPExtractor

    return NLPExtractor()


@pytest.fixture(scope="session")
def hybrid():
    """The real hybrid extractor. Session-scoped: load the model at most once."""
    pytest.importorskip("torch")
    from nlp.predict import HybridExtractor

    models_store = BACKEND_DIR / "models_store" / "ner"
    if not any(models_store.glob("muril_ner_*")):
        pytest.skip("No trained NER model in models_store/ner — run nlp.ner.train.")
    return HybridExtractor()


class FakeExtractor:
    """Stands in for HybridExtractor so webhook tests skip the model load."""

    def __init__(self, result: Dict[str, Any] | None = None) -> None:
        self.result = result or {"city": "Pune", "bhk": 2, "budget_value": 6000000.0}
        self.calls: List[str] = []

    def extract(self, text: str) -> Dict[str, Any]:
        self.calls.append(text)
        return dict(self.result)


@pytest.fixture
def fake_extractor(monkeypatch):
    """Patches the webhook's extractor lookup with a fake."""
    from routes import webhook

    fake = FakeExtractor()
    monkeypatch.setattr(webhook, "get_hybrid_extractor", lambda: fake)
    return fake


@pytest.fixture
def client(fake_extractor):
    """Flask test client with model preloading disabled."""
    from app import create_app

    app = create_app(preload_models=False)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def end_of_call_payload(transcript: str = "Customer: I want a 2 BHK in Pune.",
                        **overrides: Any) -> Dict[str, Any]:
    """A minimal Vapi end-of-call-report payload."""
    message: Dict[str, Any] = {
        "type": "end-of-call-report",
        "call": {"id": "call_123"},
        "customer": {"number": "+919876543210"},
        "startedAt": "2026-08-04T10:00:00Z",
        "endedAt": "2026-08-04T10:03:00Z",
        "endedReason": "customer-ended-call",
        "artifact": {"transcript": transcript},
    }
    message.update(overrides)
    return {"message": message}
