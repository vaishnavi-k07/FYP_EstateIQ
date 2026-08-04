"""Stage C — surface the un-reviewed tasks inside the Label Studio UI.

Every task in the project already carries a submitted annotation: the 226 that
were never opened still hold their Stage B rule pre-labels. Label Studio's
built-in "annotated / not annotated" filter therefore marks all 400 as done and
gives the annotator no way to find the remaining work.

This writes two fields into each task's ``data`` so the Data Manager can filter
and sort on them:

``needs_review``    "YES — not yet reviewed" / "no — done"
``review_priority`` "1-test" / "2-validation" / "3-train" / "4-done"

Sorting by ``review_priority`` puts the benchmark-critical test and validation
transcripts first. Re-run after annotating to refresh the flags.

The Label Studio server must be stopped. Fields are additive — ``text`` and
``transcript_id`` are untouched, so ``export_dataset.py`` is unaffected.

Run:
    python -m nlp.ner.mark_review_queue --backup ../review_queue_backup.sqlite3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Set

logger = logging.getLogger(__name__)

DEFAULT_DB = (
    Path.home() / "AppData" / "Local" / "label-studio" / "label-studio"
    / "label_studio.sqlite3"
)
NER_DIR = Path(__file__).resolve().parent
DATASET_PATH = NER_DIR / "dataset" / "ner_dataset.jsonl"

NEEDS_REVIEW = "YES - not yet reviewed"
DONE = "no - done"
PRIORITY_BY_SPLIT: Dict[str, str] = {
    "test": "1-test",
    "validation": "2-validation",
    "train": "3-train",
}


def reviewed_ids(dataset_path: Path) -> Set[str]:
    """Transcript ids whose annotation was actually submitted by a human."""
    with dataset_path.open("r", encoding="utf-8") as handle:
        return {
            row["transcript_id"]
            for row in (json.loads(line) for line in handle if line.strip())
            if row.get("reviewed")
        }


def split_by_id(dataset_path: Path) -> Dict[str, str]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return {
            row["transcript_id"]: row["split"]
            for row in (json.loads(line) for line in handle if line.strip())
        }


def apply_flags(
    connection: sqlite3.Connection,
    reviewed: Set[str],
    splits: Dict[str, str],
    dry_run: bool,
) -> Counter:
    tally: Counter = Counter()
    for task_id, raw in connection.execute("SELECT id, data FROM task").fetchall():
        data = json.loads(raw)
        transcript_id = data.get("transcript_id")
        is_reviewed = transcript_id in reviewed

        data["needs_review"] = DONE if is_reviewed else NEEDS_REVIEW
        data["review_priority"] = (
            "4-done" if is_reviewed
            else PRIORITY_BY_SPLIT.get(splits.get(transcript_id, ""), "3-train")
        )
        tally[data["review_priority"]] += 1

        if not dry_run:
            connection.execute(
                "UPDATE task SET data = ? WHERE id = ?",
                (json.dumps(data, ensure_ascii=False), task_id),
            )
    if not dry_run:
        connection.commit()
    return tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--backup", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.db.exists():
        raise SystemExit(f"Label Studio DB not found: {args.db}")
    if not args.dataset.exists():
        raise SystemExit(
            f"{args.dataset} not found — run nlp.ner.export_dataset first."
        )

    if args.backup and not args.dry_run:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.db, args.backup)
        logger.info("Backed up DB to %s", args.backup)

    reviewed = reviewed_ids(args.dataset)
    splits = split_by_id(args.dataset)
    connection = sqlite3.connect(str(args.db))
    tally = apply_flags(connection, reviewed, splits, args.dry_run)

    total = sum(tally.values())
    remaining = total - tally["4-done"]
    logger.info("")
    logger.info("=" * 62)
    logger.info("REVIEW QUEUE FLAGS%s", "  (dry run)" if args.dry_run else "")
    logger.info("=" * 62)
    for key in ("1-test", "2-validation", "3-train", "4-done"):
        label = "already reviewed" if key == "4-done" else "NEEDS REVIEW"
        logger.info("  %-14s %4d   %s", key, tally[key], label)
    logger.info("  %s", "-" * 46)
    logger.info("  %-14s %4d   of %d tasks", "REMAINING", remaining, total)
    logger.info("")
    logger.info("In the Data Manager: filter needs_review = %r, sort by "
                "review_priority ascending.", NEEDS_REVIEW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
