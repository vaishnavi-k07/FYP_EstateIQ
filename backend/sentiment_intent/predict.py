"""Phase 2 inference — sentiment and intent for one transcript.

The contract the rest of the pipeline depends on (CLAUDE.md section 8, Phase 2)::

    {
        "sentiment": "Hesitant",
        "sentiment_confidence": 0.87,
        "intent": "Schedule Visit",
        "intent_confidence": 0.91,
    }

Both heads share the same input treatment as training: only the customer's
turns are scored, agent lines dropped. Feeding the raw transcript in and
letting this module do the stripping keeps the caller from having to know that.

Contains no feature-extraction logic — city/budget/BHK belong to ``nlp``
(CLAUDE.md section 7).

Run:
    python -m sentiment_intent.predict --text "Customer: um, we're still looking around"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    MODELS_STORE,
    customer_turns,
    resolve_latest,
)

logger = logging.getLogger(__name__)


class _Head:
    """One fine-tuned DistilBERT classifier loaded for inference."""

    def __init__(self, model_dir: Path, device: torch.device, max_length: int) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(device)
        self.model.eval()
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        probabilities = torch.softmax(self.model(**encoding).logits[0], dim=-1)
        index = int(probabilities.argmax())
        return {
            "label": self.id2label[index],
            "confidence": round(float(probabilities[index]), 4),
            "probabilities": {
                self.id2label[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
        }


class SentimentIntentPredictor:
    """Loads both heads once and scores transcripts."""

    def __init__(
        self,
        sentiment_dir: Optional[Path] = None,
        intent_dir: Optional[Path] = None,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        resolved = {
            "sentiment": Path(sentiment_dir) if sentiment_dir else resolve_latest("sentiment"),
            "intent": Path(intent_dir) if intent_dir else resolve_latest("intent"),
        }
        missing = [task for task, path in resolved.items() if path is None]
        if missing:
            raise SystemExit(
                f"No trained {'/'.join(missing)} model in {MODELS_STORE}. "
                "Run `python -m sentiment_intent.train_sentiment` and "
                "`python -m sentiment_intent.train_intent` first."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sentiment = _Head(resolved["sentiment"], self.device, max_length)
        self.intent = _Head(resolved["intent"], self.device, max_length)
        logger.info(
            "Loaded sentiment=%s intent=%s on %s",
            self.sentiment.model_dir.name, self.intent.model_dir.name, self.device,
        )

    def predict(self, transcript: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """Scores a raw transcript. Agent turns are stripped automatically."""
        text = customer_turns(transcript or "")
        if not text.strip():
            return {
                "sentiment": None,
                "sentiment_confidence": None,
                "intent": None,
                "intent_confidence": None,
            }

        sentiment = self.sentiment.predict(text)
        intent = self.intent.predict(text)
        result: Dict[str, Any] = {
            "sentiment": sentiment["label"],
            "sentiment_confidence": sentiment["confidence"],
            "intent": intent["label"],
            "intent_confidence": intent["confidence"],
        }
        if include_probabilities:
            result["sentiment_probabilities"] = sentiment["probabilities"]
            result["intent_probabilities"] = intent["probabilities"]
        return result


_INSTANCE: Optional[SentimentIntentPredictor] = None


def get_predictor() -> SentimentIntentPredictor:
    """Process-wide singleton.

    Two DistilBERT checkpoints are ~500 MB and take seconds to load, so a web
    worker loads them once at startup rather than per request — the same
    pattern ``nlp.predict.get_hybrid_extractor`` uses.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SentimentIntentPredictor()
    return _INSTANCE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Transcript text to score.")
    parser.add_argument("--file", type=Path, help="Read the transcript from a file.")
    parser.add_argument("--probabilities", action="store_true",
                        help="Include the full probability distribution per head.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.file:
        transcript = args.file.read_text(encoding="utf-8")
    elif args.text:
        transcript = args.text
    else:
        transcript = sys.stdin.read()

    result = get_predictor().predict(transcript, include_probabilities=args.probabilities)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
