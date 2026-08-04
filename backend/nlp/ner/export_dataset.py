"""Stage D (part 1) — export Stage C annotations and build the BIO dataset.

Pulls the hand-corrected annotations out of Label Studio, gates them on the
same offset-integrity check Stage B had to pass, converts character spans to
word-level BIO tags, and writes a reproducible stratified train/val/test split.

Training never sees data that failed the gate: if the integrity check finds a
problem, nothing is written and the process exits non-zero.

Run:
    python -m nlp.ner.export_dataset --token <api-token>
    python -m nlp.ner.export_dataset --from-file export.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logger = logging.getLogger(__name__)

NER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
POOL_PATH = PROJECT_ROOT / "data_generation" / "output" / "synthetic_transcripts.jsonl"
DATASET_DIR = NER_DIR / "dataset"

DEFAULT_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
DEFAULT_PROJECT_ID = int(os.environ.get("LABEL_STUDIO_PROJECT_ID", "1"))
DEFAULT_SQLITE = (
    Path.home() / "AppData" / "Local" / "label-studio" / "label-studio"
    / "label_studio.sqlite3"
)

ENTITIES: Tuple[str, ...] = (
    "CITY",
    "AREA",
    "PROPERTY_TYPE",
    "BHK",
    "BUDGET",
    "AMENITY",
    "FURNISHING",
)
LABEL_LIST: List[str] = ["O"] + [f"{prefix}-{e}" for e in ENTITIES for prefix in ("B", "I")]

SEED = 42
SPLIT_RATIOS: Dict[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}

# Words and standalone punctuation, each with character offsets. Keeping
# punctuation as its own token matches the CoNLL convention in schema.md.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def fetch_export(url: str, token: str, project_id: int) -> List[Dict[str, Any]]:
    endpoint = f"{url.rstrip('/')}/api/projects/{project_id}/export?exportType=JSON"
    request = urllib.request.Request(endpoint)
    request.add_header("Authorization", f"Token {token}")
    logger.info("Fetching annotations from %s", endpoint)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def read_sqlite_export(db_path: Path) -> List[Dict[str, Any]]:
    """Builds the same task shape as the REST export, straight from the LS DB.

    Lets the export run with the Label Studio server stopped, which keeps the
    pipeline reproducible from the annotation store alone.
    """
    import sqlite3

    connection = sqlite3.connect(str(db_path))
    rows = connection.execute(
        """
        SELECT t.id, t.data, c.result, c.was_cancelled, c.lead_time, c.updated_at
        FROM task t
        LEFT JOIN task_completion c ON c.task_id = t.id
        ORDER BY t.id
        """
    ).fetchall()

    tasks: Dict[int, Dict[str, Any]] = {}
    for task_id, data, result, was_cancelled, lead_time, updated_at in rows:
        task = tasks.setdefault(
            task_id, {"id": task_id, "data": json.loads(data), "annotations": []}
        )
        if result is None:
            continue
        task["annotations"].append(
            {
                "result": json.loads(result),
                "was_cancelled": bool(was_cancelled),
                "lead_time": lead_time,
                "updated_at": updated_at,
            }
        )
    return list(tasks.values())


REVIEW_CHOICE = "Reviewed"


def has_review_choice(annotation: Dict[str, Any]) -> bool:
    """True when the annotator ticked the 'Reviewed' box from the config."""
    for region in annotation.get("result") or []:
        if region.get("from_name") == "review_status":
            if REVIEW_CHOICE in ((region.get("value") or {}).get("choices") or []):
                return True
    return False


def is_reviewed(task: Dict[str, Any]) -> bool:
    """True when a human actually confirmed this task's annotation.

    Two independent signals, either of which is sufficient:

    * the explicit 'Reviewed' checkbox (see label_studio_config.xml) — the
      reliable one, because it also re-enables Update on an unchanged
      pre-label that is already correct;
    * a non-null ``lead_time``, which Label Studio sets on any human submit.
      This keeps the 208 annotations reviewed before the checkbox existed.

    Annotations bulk-imported as Stage B pre-labels have neither: they carry
    ``lead_time = None`` and were never opened. Treating those as gold would
    make the rule-vs-model benchmark circular, so they are tracked separately.
    """
    for annotation in task.get("annotations") or []:
        if annotation.get("was_cancelled"):
            continue
        if has_review_choice(annotation) or annotation.get("lead_time") is not None:
            return True
    return False


def load_pool(path: Path = POOL_PATH) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return {
            record["transcript_id"]: record
            for record in (json.loads(line) for line in handle if line.strip())
        }


def accepted_spans(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Spans from non-cancelled annotations, de-duplicated and ordered."""
    seen: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for annotation in task.get("annotations") or []:
        if annotation.get("was_cancelled"):
            continue
        for region in annotation.get("result") or []:
            value = region.get("value") or {}
            labels = value.get("labels") or []
            if not labels:
                continue
            key = (value.get("start"), value.get("end"), labels[0])
            seen.setdefault(
                key,
                {
                    "start": value["start"],
                    "end": value["end"],
                    "label": labels[0],
                    "text": value.get("text", ""),
                },
            )
    return sorted(seen.values(), key=lambda s: (s["start"], s["end"]))


# --------------------------------------------------------------------------- #
# Boundary normalization
# --------------------------------------------------------------------------- #

# schema.md section 5: punctuation and connectors are always O, never part of
# an entity span. Browser drag-selection routinely picks up a leading space or
# a trailing sentence period; trimming those is an artifact fix, not a
# relabeling — the entity text itself is untouched.
_TRIM_CHARS = " \t\n\r"
_TRIM_PUNCT = ".,;:!?"


def normalize_span(
    span: Dict[str, Any], raw_text: str
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Trims stray whitespace/punctuation off a span. Returns (span, changed)."""
    start, end = span["start"], span["end"]
    trimmable = _TRIM_CHARS + _TRIM_PUNCT
    while start < end and raw_text[start] in trimmable:
        start += 1
    while end > start and raw_text[end - 1] in trimmable:
        end -= 1

    if end <= start:
        return None, True
    if (start, end) == (span["start"], span["end"]):
        return span, False
    return {**span, "start": start, "end": end, "text": raw_text[start:end]}, True


def normalize_task_spans(
    spans: Sequence[Dict[str, Any]], raw_text: str, transcript_id: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    for span in spans:
        normalized, changed = normalize_span(span, raw_text)
        if changed:
            changes.append(
                {
                    "transcript_id": transcript_id,
                    "label": span["label"],
                    "before": span["text"],
                    "after": normalized["text"] if normalized else None,
                }
            )
        if normalized:
            kept.append(normalized)
    return kept, changes


# --------------------------------------------------------------------------- #
# Integrity gate
# --------------------------------------------------------------------------- #


@dataclass
class IntegrityReport:
    tasks: int = 0
    spans: int = 0
    label_counts: Counter = field(default_factory=Counter)
    offset_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    unknown_labels: List[Dict[str, Any]] = field(default_factory=list)
    out_of_bounds: List[Dict[str, Any]] = field(default_factory=list)
    whitespace_spans: List[Dict[str, Any]] = field(default_factory=list)
    overlaps: List[Dict[str, Any]] = field(default_factory=list)
    text_differs_from_pool: List[str] = field(default_factory=list)
    unannotated: List[str] = field(default_factory=list)
    not_in_pool: List[str] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return (
            len(self.offset_mismatches)
            + len(self.unknown_labels)
            + len(self.out_of_bounds)
            + len(self.whitespace_spans)
            + len(self.text_differs_from_pool)
            + len(self.not_in_pool)
        )

    @property
    def passed(self) -> bool:
        return self.failures == 0


def check_integrity(
    tasks: Sequence[Dict[str, Any]], pool: Dict[str, Dict[str, Any]]
) -> IntegrityReport:
    """Same gate Stage B had to clear, re-run on the hand-corrected export.

    Overlapping spans are recorded but not treated as fatal: BIO tagging can
    only assign one label per token, so overlaps are resolved (longest wins)
    rather than rejected.
    """
    report = IntegrityReport(tasks=len(tasks))

    for task in tasks:
        transcript_id = (task.get("data") or {}).get("transcript_id", f"task_{task.get('id')}")
        raw_text = (task.get("data") or {}).get("text", "")

        source = pool.get(transcript_id)
        if source is None:
            report.not_in_pool.append(transcript_id)
        elif source["text"] != raw_text:
            report.text_differs_from_pool.append(transcript_id)

        spans = accepted_spans(task)
        if not spans:
            report.unannotated.append(transcript_id)

        placed: List[Tuple[int, int]] = []
        for span in spans:
            report.spans += 1
            start, end, label = span["start"], span["end"], span["label"]
            detail = {"transcript_id": transcript_id, "start": start, "end": end, "label": label}

            if label not in ENTITIES:
                report.unknown_labels.append(detail)
            else:
                report.label_counts[label] += 1

            if not isinstance(start, int) or not isinstance(end, int) or not (
                0 <= start < end <= len(raw_text)
            ):
                report.out_of_bounds.append({**detail, "raw_len": len(raw_text)})
                continue

            actual = raw_text[start:end]
            if actual != span["text"]:
                report.offset_mismatches.append(
                    {**detail, "stored_text": span["text"], "raw_substring": actual}
                )
            if not actual.strip():
                report.whitespace_spans.append(detail)
            if any(start < p_end and end > p_start for p_start, p_end in placed):
                report.overlaps.append(detail)
            placed.append((start, end))

    return report


def log_integrity(report: IntegrityReport) -> None:
    logger.info("")
    logger.info("=" * 74)
    logger.info("DATA INTEGRITY CHECK — Stage C export")
    logger.info("=" * 74)
    logger.info("Tasks exported            : %d", report.tasks)
    logger.info("Spans exported            : %d", report.spans)
    logger.info("")
    logger.info("Offset mismatches         : %d   (required: 0)", len(report.offset_mismatches))
    logger.info("Out-of-bounds spans       : %d   (required: 0)", len(report.out_of_bounds))
    logger.info("Whitespace-only spans     : %d   (required: 0)", len(report.whitespace_spans))
    logger.info("Labels outside schema.md  : %d   (required: 0)", len(report.unknown_labels))
    logger.info("Task text != source pool  : %d   (required: 0)", len(report.text_differs_from_pool))
    logger.info("Transcripts not in pool   : %d   (required: 0)", len(report.not_in_pool))
    logger.info("Overlapping spans         : %d   (tolerated, longest wins)", len(report.overlaps))
    logger.info("Tasks with zero spans     : %d   (tolerated, vague transcripts)",
                len(report.unannotated))

    for name, problems in (
        ("OFFSET MISMATCH", report.offset_mismatches),
        ("UNKNOWN LABEL", report.unknown_labels),
        ("OUT OF BOUNDS", report.out_of_bounds),
        ("WHITESPACE SPAN", report.whitespace_spans),
    ):
        for problem in problems[:20]:
            logger.error("  %s: %s", name, problem)

    logger.info("")
    logger.info("Label distribution (%d spans):", report.spans)
    total = sum(report.label_counts.values()) or 1
    for entity in ENTITIES:
        count = report.label_counts.get(entity, 0)
        logger.info("    %-16s %5d   (%5.1f%%)", entity, count, 100.0 * count / total)
    unknown = set(report.label_counts) - set(ENTITIES)
    for entity in sorted(unknown):
        logger.error("    %-16s %5d   <-- NOT IN SCHEMA", entity, report.label_counts[entity])


# --------------------------------------------------------------------------- #
# BIO conversion
# --------------------------------------------------------------------------- #


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]


def resolve_overlaps(spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One label per token, so nested/overlapping spans collapse to the longest."""
    ordered = sorted(spans, key=lambda s: (-(s["end"] - s["start"]), s["start"]))
    kept: List[Dict[str, Any]] = []
    for span in ordered:
        if any(span["start"] < k["end"] and span["end"] > k["start"] for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s["start"])


def to_bio(
    text: str, spans: Sequence[Dict[str, Any]]
) -> Tuple[List[str], List[str], int]:
    """Assigns word-level BIO tags. Returns tokens, tags, and partial-token hits."""
    tokens = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)
    partial = 0

    for span in resolve_overlaps(spans):
        member_indices = [
            index
            for index, (_, start, end) in enumerate(tokens)
            if start < span["end"] and end > span["start"]
        ]
        if not member_indices:
            continue
        # A span that cuts a token in half cannot be represented exactly at
        # word level; the token is absorbed whole and the case is counted.
        first, last = tokens[member_indices[0]], tokens[member_indices[-1]]
        if first[1] < span["start"] or last[2] > span["end"]:
            partial += 1
        for position, index in enumerate(member_indices):
            tags[index] = f"{'B' if position == 0 else 'I'}-{span['label']}"

    return [t[0] for t in tokens], tags, partial


# --------------------------------------------------------------------------- #
# Stratified split
# --------------------------------------------------------------------------- #


def stratum_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Groups transcripts so every hard-case type lands in all three splits."""
    flags = record["metadata"]["flags"]
    return (
        record["ground_truth"]["city"],
        bool(flags["has_negation"]),
        bool(flags["is_code_mixed"]),
        bool(flags["is_multi_city"]),
        bool(flags["is_telegraphic"]),
    )


def stratified_split(
    transcript_ids: Sequence[str], pool: Dict[str, Dict[str, Any]], seed: int = SEED
) -> Dict[str, str]:
    """Deterministic per-stratum round-robin assignment into train/val/test.

    Strata are small (some hold a single transcript), so proportional slicing
    would starve val/test. Dealing each stratum out in a rotating order, with
    the rotation offset seeded per stratum, keeps the global ratios close while
    guaranteeing small strata are spread rather than dumped into train.
    """
    rng = random.Random(seed)
    groups: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    for transcript_id in transcript_ids:
        groups[stratum_key(pool[transcript_id])].append(transcript_id)

    # Deal proportionally: build a repeating pattern matching the target ratios.
    pattern: List[str] = []
    for name, ratio in SPLIT_RATIOS.items():
        pattern.extend([name] * int(round(ratio * 100)))
    rng.shuffle(pattern)

    assignment: Dict[str, str] = {}
    cursor = 0
    for key in sorted(groups, key=lambda k: (str(k))):
        members = sorted(groups[key])
        rng.shuffle(members)
        for transcript_id in members:
            assignment[transcript_id] = pattern[cursor % len(pattern)]
            cursor += 1
    return assignment


def log_review_coverage(rows: Sequence[Dict[str, Any]]) -> None:
    """Stage C completeness — how much of each split is real human annotation.

    Unreviewed tasks still hold the Stage B rule pre-labels verbatim. Any test
    transcript in that state scores the rule extractor against its own output,
    so the NER-vs-rules benchmark is only meaningful over reviewed rows.
    """
    logger.info("")
    logger.info("=" * 74)
    logger.info("STAGE C REVIEW COVERAGE")
    logger.info("=" * 74)
    reviewed_total = sum(1 for row in rows if row["reviewed"])
    logger.info("Human-reviewed        : %d / %d  (%.1f%%)", reviewed_total, len(rows),
                100.0 * reviewed_total / max(len(rows), 1))
    logger.info("Unreviewed pre-labels : %d   <-- still raw rule-extractor output",
                len(rows) - reviewed_total)
    logger.info("")
    logger.info("  %-12s %8s %12s %12s", "split", "n", "reviewed", "unreviewed")
    logger.info("  %s", "-" * 48)
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        reviewed = sum(1 for row in subset if row["reviewed"])
        logger.info("  %-12s %8d %12d %12d", split, len(subset), reviewed,
                    len(subset) - reviewed)

    unreviewed_test = sum(
        1 for row in rows if row["split"] == "test" and not row["reviewed"]
    )
    if unreviewed_test:
        logger.warning("")
        logger.warning(
            "%d test transcript(s) are unreviewed rule pre-labels. The rule "
            "extractor scores ~1.0 F1 on those by construction — run the "
            "benchmark with --reviewed-only for an honest comparison.",
            unreviewed_test,
        )


def log_split(assignment: Dict[str, str], pool: Dict[str, Dict[str, Any]]) -> None:
    logger.info("")
    logger.info("=" * 74)
    logger.info("STRATIFIED SPLIT")
    logger.info("=" * 74)
    splits = ("train", "validation", "test")
    counts = Counter(assignment.values())
    for split in splits:
        logger.info("  %-12s %4d  (%.1f%%)", split, counts[split],
                    100.0 * counts[split] / max(len(assignment), 1))

    logger.info("")
    logger.info("  Hard cases per split (must appear in all three):")
    header = f"    {'flag':<26}" + "".join(f"{s:>13}" for s in splits)
    logger.info(header)
    checks = {
        "has_negation": lambda f: f["has_negation"],
        "is_code_mixed": lambda f: f["is_code_mixed"],
        "is_multi_city": lambda f: f["is_multi_city"],
        "is_telegraphic": lambda f: f["is_telegraphic"],
        "is_rare_property_type": lambda f: f["is_rare_property_type"],
    }
    for name, predicate in checks.items():
        row = {
            split: sum(
                1
                for tid, s in assignment.items()
                if s == split and predicate(pool[tid]["metadata"]["flags"])
            )
            for split in splits
        }
        flag = "" if all(row[s] > 0 for s in splits) else "   <-- MISSING FROM A SPLIT"
        logger.info("    %-26s" + "".join(f"{row[s]:>13}" for s in splits) + "%s", name, flag)

    logger.info("")
    logger.info("    %-26s" + "".join(f"{s:>13}" for s in splits), "city")
    for city in ("Nashik", "Pune"):
        row = {
            split: sum(
                1
                for tid, s in assignment.items()
                if s == split and pool[tid]["ground_truth"]["city"] == city
            )
            for split in splits
        }
        logger.info("    %-26s" + "".join(f"{row[s]:>13}" for s in splits), city)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_dataset(
    rows: Sequence[Dict[str, Any]], assignment: Dict[str, str], report: IntegrityReport
) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = DATASET_DIR / "ner_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d rows)", jsonl_path, len(rows))

    # CoNLL-2003 style, one token per line, blank line between transcripts.
    for split in ("train", "validation", "test"):
        conll_path = DATASET_DIR / f"{split}.conll"
        with conll_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row["split"] != split:
                    continue
                handle.write(f"# transcript_id = {row['transcript_id']}\n")
                for token, tag in zip(row["tokens"], row["ner_tags"]):
                    handle.write(f"{token}\t{tag}\n")
                handle.write("\n")
        logger.info("Wrote %s", conll_path)

    (DATASET_DIR / "label_list.json").write_text(
        json.dumps(LABEL_LIST, indent=2), encoding="utf-8"
    )
    (DATASET_DIR / "split.json").write_text(
        json.dumps({"seed": SEED, "ratios": SPLIT_RATIOS, "assignment": assignment}, indent=2),
        encoding="utf-8",
    )
    (DATASET_DIR / "integrity_report.json").write_text(
        json.dumps(
            {
                "tasks": report.tasks,
                "spans": report.spans,
                "label_counts": dict(report.label_counts),
                "offset_mismatches": report.offset_mismatches,
                "unknown_labels": report.unknown_labels,
                "out_of_bounds": report.out_of_bounds,
                "whitespace_spans": report.whitespace_spans,
                "text_differs_from_pool": report.text_differs_from_pool,
                "overlaps": len(report.overlaps),
                "tasks_without_spans": report.unannotated,
                "passed": report.passed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", DATASET_DIR / "split.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", default=os.environ.get("LABEL_STUDIO_TOKEN"))
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--from-file", type=Path, help="Use a saved export instead of the API.")
    parser.add_argument(
        "--from-sqlite",
        type=Path,
        nargs="?",
        const=DEFAULT_SQLITE,
        help="Read tasks straight from the Label Studio DB (no running server).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.from_sqlite:
        tasks = read_sqlite_export(args.from_sqlite)
        logger.info("Read %d tasks from %s", len(tasks), args.from_sqlite)
    elif args.from_file:
        tasks = json.loads(args.from_file.read_text(encoding="utf-8"))
        logger.info("Loaded %d tasks from %s", len(tasks), args.from_file)
    else:
        if not args.token:
            raise SystemExit("Pass --token or set LABEL_STUDIO_TOKEN.")
        tasks = fetch_export(args.url, args.token, args.project_id)
        logger.info("Exported %d tasks", len(tasks))

    pool = load_pool()
    report = check_integrity(tasks, pool)
    log_integrity(report)

    if not report.passed:
        logger.error("")
        logger.error("INTEGRITY CHECK FAILED — %d problem(s). Nothing written, "
                     "training not started.", report.failures)
        return 1
    logger.info("")
    logger.info("INTEGRITY CHECK PASSED.")

    rows: List[Dict[str, Any]] = []
    partial_total = 0
    all_changes: List[Dict[str, Any]] = []
    transcript_ids = [t["data"]["transcript_id"] for t in tasks]
    assignment = stratified_split(transcript_ids, pool, seed=args.seed)

    for task in tasks:
        transcript_id = task["data"]["transcript_id"]
        text = task["data"]["text"]
        spans, changes = normalize_task_spans(accepted_spans(task), text, transcript_id)
        all_changes.extend(changes)
        tokens, tags, partial = to_bio(text, spans)
        partial_total += partial
        flags = pool[transcript_id]["metadata"]["flags"]
        rows.append(
            {
                "transcript_id": transcript_id,
                "split": assignment[transcript_id],
                "tokens": tokens,
                "ner_tags": tags,
                "text": text,
                "spans": spans,
                "flags": {
                    key: flags[key]
                    for key in (
                        "has_negation",
                        "is_code_mixed",
                        "is_multi_city",
                        "is_telegraphic",
                        "is_rare_property_type",
                    )
                },
                "city": pool[transcript_id]["ground_truth"]["city"],
                "reviewed": is_reviewed(task),
            }
        )

    if all_changes:
        logger.info("")
        logger.info("=" * 74)
        logger.info("SPAN BOUNDARY NORMALIZATION (schema.md section 5: punctuation is O)")
        logger.info("=" * 74)
        logger.info("Trimmed stray whitespace/punctuation from %d hand-drawn span(s):",
                    len(all_changes))
        by_label = Counter(change["label"] for change in all_changes)
        for label, count in by_label.most_common():
            logger.info("    %-16s %3d", label, count)
        logger.info("")
        for change in all_changes:
            logger.info("    %-10s %-14s %r -> %r", change["transcript_id"], change["label"],
                        change["before"], change["after"])

    tag_counts = Counter(tag for row in rows for tag in row["ner_tags"])
    logger.info("")
    logger.info("BIO conversion: %d transcripts, %d tokens, %d entity spans",
                len(rows), sum(len(r["tokens"]) for r in rows),
                sum(1 for r in rows for t in r["ner_tags"] if t.startswith("B-")))
    logger.info("Tokens tagged O: %d (%.1f%%)", tag_counts["O"],
                100.0 * tag_counts["O"] / max(sum(tag_counts.values()), 1))
    if partial_total:
        logger.warning("Spans not aligned to token boundaries (token absorbed whole): %d",
                       partial_total)

    log_review_coverage(rows)
    log_split(assignment, pool)
    write_dataset(rows, assignment, report)
    (DATASET_DIR / "boundary_normalizations.json").write_text(
        json.dumps(all_changes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %s (%d change(s))",
                DATASET_DIR / "boundary_normalizations.json", len(all_changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
