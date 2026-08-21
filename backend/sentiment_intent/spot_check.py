"""Eyeball a sentiment head on short, hand-written utterances.

The phrase-holdout diagnostic answers "does it generalise to emotional ideas it
never saw", but every one of those transcripts still came out of our generator:
same call structure, same requirement sentences, same length. These cases are
none of those things — they are bare one-liners of the kind a real caller
opens with, several of them code-mixed, and none of them built from our pools.

That makes this a weaker measurement than the diagnostic (ten items, no
statistics) and a stronger sanity check (genuinely out-of-distribution input).
Read it for whether the model is reading emotion at all, not for a score.

Run:
    python -m sentiment_intent.spot_check
    python -m sentiment_intent.spot_check --model-dir <path> --compare <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import (  # noqa: E402
    MODELS_STORE,
    SENTIMENT_CLASSES,
    customer_turns,
)

logger = logging.getLogger(__name__)

HYBRID_HOLDOUT_DIR = MODELS_STORE / "holdout" / "sentiment_v4_hybrid-emodistilroberta"

# (expected tone, register, transcript). Expectations are the caller's reading
# of each line, not ground truth from a generator — that is the point.
CASES: Tuple[Tuple[str, str, str], ...] = (
    ("Enthusiastic", "english",
     "Oh wow, this is exactly what we've been looking for! When can we see it?"),
    ("Enthusiastic", "english",
     "yes finally! we're so excited, let's move fast on this one"),
    ("Enthusiastic", "code-mixed",
     "haan bilkul, ye toh perfect hai, hume bahut pasand aaya"),
    ("Frustrated", "english",
     "I've called three times already and no one got back to me."),
    ("Frustrated", "english",
     "honestly this is taking way too long, I'm losing patience"),
    ("Frustrated", "code-mixed",
     "yaar kitni baar bataya hai, phir se same baat"),
    ("Hesitant", "english",
     "hmm, I'm not sure yet, we're still thinking about it"),
    ("Hesitant", "english",
     "maybe, but I'd want to see a few more options first"),
    ("Neutral", "english",
     "we're looking for a 3 BHK in Baner, budget around 90 lakhs"),
    ("Neutral", "english",
     "can you send me the details over email"),
)


@torch.no_grad()
def predict(
    model_dir: Path, texts: Sequence[str], max_length: int = 256
) -> List[Dict[str, Any]]:
    """Returns label, confidence and the full distribution for each text."""
    from transformers import AutoTokenizer

    from sentiment_intent.domain_adapt import load_classifier  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_classifier(str(model_dir)).to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    out: List[Dict[str, Any]] = []
    for text in texts:
        # Same preprocessing as training: customer turns only. A bare utterance
        # has no "Customer:" prefix, so this falls through to the raw string.
        encoded = tokenizer(
            customer_turns(text), truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        probabilities = torch.softmax(model(**encoded).logits.float()[0], dim=-1).cpu()
        best = int(probabilities.argmax())
        out.append({
            "label": id2label[best],
            "confidence": float(probabilities[best]),
            "distribution": {
                id2label[i]: float(probabilities[i]) for i in range(len(probabilities))
            },
        })
    return out


def run(model_dir: Path, cases: Sequence[Tuple[str, str, str]]) -> Dict[str, Any]:
    predictions = predict(model_dir, [text for _, _, text in cases])

    rows: List[Dict[str, Any]] = []
    for (expected, register, text), prediction in zip(cases, predictions):
        rows.append({
            "expected": expected,
            "register": register,
            "transcript": text,
            "predicted": prediction["label"],
            "confidence": prediction["confidence"],
            "correct": prediction["label"] == expected,
            "distribution": prediction["distribution"],
        })

    correct = sum(1 for r in rows if r["correct"])
    polarity_errors = [
        r for r in rows
        if not r["correct"]
        and {r["expected"], r["predicted"]} == {"Enthusiastic", "Frustrated"}
    ]
    code_mixed = [r for r in rows if r["register"] == "code-mixed"]
    return {
        "model_dir": str(model_dir),
        "n": len(rows),
        "correct": correct,
        "polarity_errors": len(polarity_errors),
        "code_mixed_correct": sum(1 for r in code_mixed if r["correct"]),
        "code_mixed_n": len(code_mixed),
        "rows": rows,
    }


def log_report(title: str, result: Dict[str, Any]) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("%s", title)
    logger.info("%s", result["model_dir"])
    logger.info("=" * 78)
    for index, row in enumerate(result["rows"], start=1):
        mark = "OK  " if row["correct"] else "WRONG"
        tag = " [code-mixed]" if row["register"] == "code-mixed" else ""
        logger.info("")
        logger.info("%2d. %s%s", index, row["transcript"], tag)
        logger.info("    expected %-12s  predicted %-12s  confidence %.3f   %s",
                    row["expected"], row["predicted"], row["confidence"], mark)
        spread = "  ".join(
            f"{name[:4]} {value:.2f}" for name, value in row["distribution"].items()
        )
        logger.info("    %s", spread)
    logger.info("")
    logger.info("%d/%d correct | %d Enthusiastic<->Frustrated errors | code-mixed %d/%d",
                result["correct"], result["n"], result["polarity_errors"],
                result["code_mixed_correct"], result["code_mixed_n"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=HYBRID_HOLDOUT_DIR)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="*",
        default=[],
        help="Additional model directories to run the same cases through.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    results = {str(args.model_dir): run(args.model_dir, CASES)}
    log_report("SPOT CHECK", results[str(args.model_dir)])
    for other in args.compare:
        results[str(other)] = run(other, CASES)
        log_report(f"SPOT CHECK — {other.name}", results[str(other)])

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        logger.info("")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
