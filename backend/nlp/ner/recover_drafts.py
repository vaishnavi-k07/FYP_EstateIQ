"""Stage C repair — promote unsaved Label Studio drafts to submitted annotations.

Label Studio autosaves in-progress work to ``tasks_annotationdraft``. If the
annotator navigates away without clicking Submit, that work never reaches
``task_completion`` and is therefore invisible to ``export_dataset.py`` — the
export silently keeps the older (often rule-generated) annotation instead.

This finds drafts that are newer than their task's submitted annotation and
copies the draft result over it, marking the annotation as human-authored so
the Stage C completeness audit counts it correctly.

Defaults to a dry run. Back up the SQLite file before running with --apply.

Run:
    python -m nlp.ner.recover_drafts
    python -m nlp.ner.recover_drafts --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB = (
    Path.home()
    / "AppData"
    / "Local"
    / "label-studio"
    / "label-studio"
    / "label_studio.sqlite3"
)

Span = Tuple[int, int, str]


@dataclass
class DraftRecovery:
    """A draft whose content differs from the annotation it would replace."""

    task_id: int
    annotation_id: Optional[int]
    transcript_id: str
    draft_result: List[Dict[str, Any]]
    draft_updated: str
    annotation_updated: str
    annotation_is_human: bool
    added: List[Span]
    removed: List[Span]
    stale_spans: List[Dict[str, Any]]

    @property
    def draft_is_newer(self) -> bool:
        return self.draft_updated > self.annotation_updated

    @property
    def is_recoverable(self) -> bool:
        """Newer *and* still anchored to the task's current text."""
        return self.draft_is_newer and not self.stale_spans


def stale_offsets(
    result: Sequence[Dict[str, Any]], text: str
) -> List[Dict[str, Any]]:
    """Draft spans whose offsets no longer match the task text.

    Task content can be re-imported (e.g. the synthetic pool was regenerated)
    while old drafts stay attached to the reused task id. Such drafts point at
    superseded text, so promoting them would inject spans that do not exist —
    the same offset gate ``export_dataset.py`` enforces, applied earlier.
    """
    stale: List[Dict[str, Any]] = []
    for region in result or []:
        value = region.get("value") or {}
        labels = value.get("labels") or []
        start, end, stored = value.get("start"), value.get("end"), value.get("text", "")
        if not labels or start is None:
            continue
        if not (0 <= start < end <= len(text)) or text[start:end] != stored:
            stale.append(
                {
                    "label": labels[0],
                    "start": start,
                    "end": end,
                    "stored_text": stored,
                    "actual_text": text[start:end] if 0 <= start < end <= len(text) else None,
                }
            )
    return stale


def span_set(result: Sequence[Dict[str, Any]]) -> Set[Span]:
    """Normalizes a Label Studio result list to a comparable set of spans."""
    spans: Set[Span] = set()
    for region in result or []:
        value = region.get("value") or {}
        labels = value.get("labels") or []
        if labels and value.get("start") is not None:
            spans.add((value["start"], value["end"], labels[0]))
    return spans


def find_recoverable(connection: sqlite3.Connection) -> List[DraftRecovery]:
    """Drafts that differ from — and postdate — their submitted annotation."""
    rows = connection.execute(
        """
        SELECT d.task_id, d.annotation_id, d.result, d.updated_at,
               t.data, c.id, c.result, c.updated_at, c.lead_time
        FROM tasks_annotationdraft d
        JOIN task t ON t.id = d.task_id
        LEFT JOIN task_completion c ON c.task_id = d.task_id
        """
    ).fetchall()

    recoveries: List[DraftRecovery] = []
    for (
        task_id,
        _draft_annotation_id,
        draft_result,
        draft_updated,
        task_data,
        annotation_id,
        annotation_result,
        annotation_updated,
        lead_time,
    ) in rows:
        draft = json.loads(draft_result) if draft_result else []
        current = json.loads(annotation_result) if annotation_result else []
        draft_spans, current_spans = span_set(draft), span_set(current)
        if draft_spans == current_spans:
            continue

        task_payload = json.loads(task_data)
        recoveries.append(
            DraftRecovery(
                task_id=task_id,
                annotation_id=annotation_id,
                transcript_id=task_payload.get("transcript_id", f"task_{task_id}"),
                draft_result=draft,
                draft_updated=str(draft_updated),
                annotation_updated=str(annotation_updated),
                annotation_is_human=lead_time is not None,
                added=sorted(draft_spans - current_spans),
                removed=sorted(current_spans - draft_spans),
                stale_spans=stale_offsets(draft, task_payload.get("text", "")),
            )
        )
    return sorted(recoveries, key=lambda r: r.transcript_id)


def log_recoveries(recoveries: Sequence[DraftRecovery]) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("RECOVERABLE DRAFTS (draft content differs from submitted annotation)")
    logger.info("=" * 78)
    if not recoveries:
        logger.info("None — every draft matches its submitted annotation.")
        return

    for recovery in recoveries:
        origin = "human-reviewed" if recovery.annotation_is_human else "RULE PRELABEL"
        logger.info("")
        logger.info(
            "  %s (task %d) — replaces a %s annotation",
            recovery.transcript_id,
            recovery.task_id,
            origin,
        )
        logger.info(
            "      draft %s  vs  annotation %s  -> draft %s",
            recovery.draft_updated,
            recovery.annotation_updated,
            "NEWER" if recovery.draft_is_newer else "older (SKIPPED)",
        )
        if recovery.stale_spans:
            logger.warning(
                "      STALE — %d span(s) do not match the task's current text; "
                "this draft was written against superseded content and is NOT "
                "recoverable.", len(recovery.stale_spans),
            )
            for span in recovery.stale_spans[:5]:
                logger.warning(
                    "        %-14s [%4s:%4s] wanted %r but text holds %r",
                    span["label"], span["start"], span["end"],
                    span["stored_text"], span["actual_text"],
                )
            continue
        for start, end, label in recovery.added:
            logger.info("      + %-14s [%4d:%4d]", label, start, end)
        for start, end, label in recovery.removed:
            logger.info("      - %-14s [%4d:%4d]", label, start, end)


def apply_recoveries(
    connection: sqlite3.Connection, recoveries: Sequence[DraftRecovery]
) -> int:
    """Overwrites each annotation with its newer draft. Returns rows updated."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
    updated = 0
    for recovery in recoveries:
        if not recovery.is_recoverable or recovery.annotation_id is None:
            continue
        connection.execute(
            """
            UPDATE task_completion
               SET result = ?, updated_at = ?, lead_time = COALESCE(lead_time, 0),
                   updated_by_id = COALESCE(updated_by_id, 1)
             WHERE id = ?
            """,
            (json.dumps(recovery.draft_result, ensure_ascii=False), now, recovery.annotation_id),
        )
        updated += 1
    connection.commit()
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    parser.add_argument("--backup", type=Path, default=None, help="Copy the DB here first.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.db.exists():
        raise SystemExit(f"Label Studio DB not found: {args.db}")

    connection = sqlite3.connect(str(args.db))
    recoveries = find_recoverable(connection)
    log_recoveries(recoveries)

    applicable = [r for r in recoveries if r.is_recoverable and r.annotation_id is not None]
    stale = [r for r in recoveries if r.stale_spans]
    logger.info("")
    logger.info("%d draft(s) differ; %d stale (superseded text, skipped); "
                "%d will be promoted.", len(recoveries), len(stale), len(applicable))

    if not args.apply:
        logger.info("")
        logger.info("Dry run — nothing written. Re-run with --apply to promote them.")
        return 0

    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.db, args.backup)
        logger.info("Backed up DB to %s", args.backup)

    updated = apply_recoveries(connection, applicable)
    logger.info("Promoted %d draft(s) into task_completion.", updated)
    logger.info("Re-run nlp.ner.export_dataset to pick them up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
