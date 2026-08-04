"""Fetches and profiles the two corpora Phase 2 trains on. Trains nothing.

Stage 1 (encoder warm-start) uses SST-2, pulled from the HuggingFace Hub and
cached locally. Stage 2 (domain fine-tune) uses the labeled synthetic
transcripts already produced by ``data_generation``.

Run:
    python -m sentiment_intent.prepare_data
    python sentiment_intent/prepare_data.py --no-download   # profile local only

Writes ``data_report.json`` beside this file: split sizes, per-class counts,
majority-class baselines, and transcript-length percentiles. The baselines are
the numbers Phase 2's exit criteria are measured against, so they live in a
committed artifact rather than in someone's terminal scrollback.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
SYNTHETIC_PATH = REPO_ROOT / "data_generation" / "output" / "synthetic_transcripts.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "data_report.json"

# SST-2 over IMDB: single sentences averaging ~10 words are much closer to a
# call-transcript utterance than IMDB's multi-paragraph reviews.
GENERAL_CORPUS = "stanfordnlp/sst2"

# The stage-2 label spaces. Declared here so a corpus that drifts from the
# agreed schemes fails loudly instead of silently training a mismatched head.
SENTIMENT_CLASSES = ("Enthusiastic", "Neutral", "Hesitant", "Frustrated")
INTENT_CLASSES = (
    "Buy",
    "Rent",
    "Inquiry",
    "Schedule Visit",
    "Investment",
    "Request Callback",
)

CUSTOMER_TURN = re.compile(r"^Customer:\s*(.*)$", re.MULTILINE)


def _distribution(values: Sequence[str], expected: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per-class counts plus the majority-class baseline any model must beat."""
    counts = collections.Counter(values)
    total = sum(counts.values())
    summary: Dict[str, Any] = {
        "total": total,
        "n_classes": len(counts),
        "counts": dict(counts.most_common()),
        "proportions": {k: round(v / total, 4) for k, v in counts.most_common()},
        # A classifier that always guesses the biggest class scores this.
        "majority_baseline": round(max(counts.values()) / total, 4) if total else 0.0,
        # >2.0 is where class weighting starts to be worth considering.
        "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 2) if counts else 0.0,
    }
    if expected is not None:
        summary["expected_classes"] = list(expected)
        summary["missing_classes"] = [c for c in expected if c not in counts]
        summary["unexpected_classes"] = [c for c in counts if c not in expected]
    return summary


def profile_general_corpus(
    download: bool = True, previous: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads SST-2 and reports split sizes and class balance.

    With ``download=False`` the previous report's SST-2 section is carried
    forward. Skipping the fetch is a convenience, so it must not silently
    delete numbers an earlier run already collected.
    """
    if not download:
        if previous:
            logger.info("Download skipped; reusing the SST-2 profile from %s", REPORT_PATH.name)
            return previous
        return {"skipped": "download disabled; no previous profile to reuse"}

    from datasets import load_dataset

    logger.info("Loading %s (cached after first run)...", GENERAL_CORPUS)
    dataset = load_dataset(GENERAL_CORPUS)

    names: List[str] = dataset["train"].features["label"].names
    splits: Dict[str, Any] = {}
    for split_name, split in dataset.items():
        counts = collections.Counter(split["label"])
        # GLUE withholds SST-2 test labels: every row is -1. Recording that here
        # stops Phase 2 from "evaluating" on 1821 unlabeled rows.
        labeled = {k: v for k, v in counts.items() if k >= 0}
        splits[split_name] = {
            "rows": len(split),
            "labeled": bool(labeled),
            "counts": {names[k]: v for k, v in sorted(labeled.items())},
            "proportions": {
                names[k]: round(v / sum(labeled.values()), 4)
                for k, v in sorted(labeled.items())
            }
            if labeled
            else {},
        }

    lengths = [len(s.split()) for s in dataset["train"]["sentence"]]
    return {
        "name": GENERAL_CORPUS,
        "role": "stage-1 encoder warm-start only; head is discarded",
        "class_names": names,
        "splits": splits,
        "train_sentence_words": _length_stats(lengths),
        "usable_eval_split": "validation",
    }


def _length_stats(lengths: Sequence[int]) -> Dict[str, float]:
    ordered = sorted(lengths)
    return {
        "mean": round(statistics.mean(ordered), 1),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[int(0.95 * len(ordered))]),
        "max": float(ordered[-1]),
    }


def profile_synthetic_corpus(path: Path = SYNTHETIC_PATH) -> Dict[str, Any]:
    """Reports the stage-2 label distributions and transcript lengths."""
    if not path.exists():
        raise SystemExit(f"Synthetic transcripts not found at {path}. Run data_generation first.")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Loaded %d synthetic transcripts from %s", len(records), path)

    meta = [r["metadata"] for r in records]
    full_words = [len(r["text"].split()) for r in records]
    customer_words = [
        len(" ".join(CUSTOMER_TURN.findall(r["text"])).split()) for r in records
    ]

    pairs = collections.Counter((m["sentiment"], m["intent"]) for m in meta)
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "total": len(records),
        "sentiment": _distribution([m["sentiment"] for m in meta], SENTIMENT_CLASSES),
        "intent": _distribution([m["intent"] for m in meta], INTENT_CLASSES),
        "completeness": _distribution([m["completeness"] for m in meta]),
        "sentiment_x_intent": {
            "cells_populated": len(pairs),
            "cells_possible": len(SENTIMENT_CLASSES) * len(INTENT_CLASSES),
            "min_cell": min(pairs.values()),
            "max_cell": max(pairs.values()),
        },
        # DistilBERT truncates at 512 tokens. Both figures are far below that,
        # so neither framing loses content to truncation.
        "length_words": {
            "full_transcript": _length_stats(full_words),
            "customer_turns_only": _length_stats(customer_words),
        },
    }


def build_report(download: bool = True, report_path: Path = REPORT_PATH) -> Dict[str, Any]:
    previous: Optional[Dict[str, Any]] = None
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8")).get("general_corpus")
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read the previous report at %s; ignoring it.", report_path)

    return {
        "general_corpus": profile_general_corpus(download=download, previous=previous),
        "synthetic_corpus": profile_synthetic_corpus(),
        "label_schemes": {
            "sentiment": list(SENTIMENT_CLASSES),
            "intent": list(INTENT_CLASSES),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip the SST-2 fetch and profile only the local synthetic corpus.",
    )
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    report = build_report(download=not args.no_download, report_path=args.out)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out)

    synth = report["synthetic_corpus"]
    for field in ("sentiment", "intent"):
        dist = synth[field]
        logger.info(
            "%s: %d classes, imbalance %.2fx, majority baseline %.1f%%",
            field,
            dist["n_classes"],
            dist["imbalance_ratio"],
            dist["majority_baseline"] * 100,
        )
        for problem in ("missing_classes", "unexpected_classes"):
            if dist.get(problem):
                logger.warning("%s has %s: %s", field, problem, dist[problem])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
