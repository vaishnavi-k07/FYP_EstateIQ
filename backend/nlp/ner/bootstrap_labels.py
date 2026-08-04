"""Stage B — bootstrap NER pre-labels from the rule-based extractor.

Selects a hard-case-stratified subset of the synthetic transcript pool, runs the
existing :class:`~nlp.extractor.NLPExtractor` matching machinery over each one,
and emits Label Studio import JSON whose character offsets index the **raw**
transcript text.

The pre-labels are a starting point for hand-correction in Stage C, not gold.
The rule extractor is known to miss things the NER model should eventually
catch (spelled-out budgets, code-mixed spans, distractor-city mentions); those
gaps are left unlabeled rather than guessed at.

Run:
    python -m nlp.ner.bootstrap_labels              # select, pre-label, verify
    python -m nlp.ner.bootstrap_labels --verify-only  # re-verify saved output
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Private regexes are imported deliberately: the pre-labels must agree with the
# extractor's own matching rules, so they share one definition rather than two.
from nlp.extractor import _BHK_PATTERN, _BUDGET_PATTERN, NLPExtractor  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = BACKEND_DIR.parent
POOL_PATH = PROJECT_ROOT / "data_generation" / "output" / "synthetic_transcripts.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "prelabels"
PRELABELS_PATH = OUTPUT_DIR / "prelabels.json"
MANIFEST_PATH = OUTPUT_DIR / "selection_manifest.json"

SEED = 42
TARGET_SIZE = 400
CITY_SHARE: Dict[str, float] = {"Nashik": 0.60, "Pune": 0.40}

# Entity label names — must match schema.md section 1 exactly.
LABEL_CITY = "CITY"
LABEL_AREA = "AREA"
LABEL_PROPERTY_TYPE = "PROPERTY_TYPE"
LABEL_BHK = "BHK"
LABEL_BUDGET = "BUDGET"
LABEL_AMENITY = "AMENITY"
LABEL_FURNISHING = "FURNISHING"

# Mirrors NLPExtractor.extract_furnishing, most specific first.
_FURNISHING_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsemi[\s-]?furnished\b", re.IGNORECASE),
    re.compile(r"\bfully[\s-]?furnished\b", re.IGNORECASE),
    re.compile(r"\bunfurnished\b", re.IGNORECASE),
    re.compile(r"\bfurnished\b", re.IGNORECASE),
)

RARE_TYPE_QUOTAS: Dict[str, int] = {
    "Warehouse": 25,
    "Farm House": 25,
    "Office Space": 25,
    "Commercial Shop": 25,
}
BUDGET_FORM_QUOTAS: Dict[str, int] = {
    "lakh": 55,
    "cr": 35,
    "range": 33,
    "exact": 32,
    "spelled_out": 30,
    "crore": 25,
}


# --------------------------------------------------------------------------- #
# Raw <-> cleaned offset mapping
# --------------------------------------------------------------------------- #


def build_offset_map(raw_text: str) -> Tuple[str, List[int]]:
    """Reproduces ``TextPreprocessor.clean`` while tracking raw indices.

    ``clean`` is ``" ".join(text.split())`` — whitespace runs collapse to a
    single space. Returns the cleaned text plus, for every cleaned character,
    the index of the raw character it came from. Inserted join-spaces map to the
    start of the whitespace run they replaced.
    """
    cleaned: List[str] = []
    index_map: List[int] = []
    position = 0
    length = len(raw_text)
    is_first_token = True

    while position < length:
        if raw_text[position].isspace():
            position += 1
            continue
        if not is_first_token:
            cleaned.append(" ")
            index_map.append(position)
        is_first_token = False
        while position < length and not raw_text[position].isspace():
            cleaned.append(raw_text[position])
            index_map.append(position)
            position += 1

    return "".join(cleaned), index_map


def map_span_to_raw(
    clean_start: int, clean_end: int, index_map: Sequence[int]
) -> Tuple[int, int]:
    """Translates a half-open cleaned-text span into raw-text offsets."""
    return index_map[clean_start], index_map[clean_end - 1] + 1


# --------------------------------------------------------------------------- #
# Span candidates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpanCandidate:
    """A located entity mention, in cleaned-text offsets."""

    start: int
    end: int
    label: str
    is_synonym: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start


class SpanLocator:
    """Finds every entity mention the rule extractor's catalogs can match.

    Unlike ``NLPExtractor.extract``, which collapses each field to a single
    value, this keeps every mention — including negated ones, which schema.md
    section 3 requires to be tagged like any other span.
    """

    def __init__(self, extractor: NLPExtractor) -> None:
        self.extractor = extractor

    def locate(self, cleaned_text: str) -> List[SpanCandidate]:
        candidates: List[SpanCandidate] = []
        candidates.extend(self._catalog_spans(cleaned_text, self.extractor.cities, LABEL_CITY))
        candidates.extend(self._catalog_spans(cleaned_text, self.extractor.areas, LABEL_AREA))
        candidates.extend(
            self._catalog_spans(cleaned_text, self.extractor.property_types, LABEL_PROPERTY_TYPE)
        )
        candidates.extend(
            self._catalog_spans(
                cleaned_text, self.extractor.property_synonyms, LABEL_PROPERTY_TYPE, is_synonym=True
            )
        )
        candidates.extend(self._catalog_spans(cleaned_text, self.extractor.amenities, LABEL_AMENITY))
        candidates.extend(
            self._catalog_spans(
                cleaned_text, self.extractor.amenity_synonyms, LABEL_AMENITY, is_synonym=True
            )
        )
        candidates.extend(self._regex_spans(cleaned_text, [_BHK_PATTERN], LABEL_BHK))
        candidates.extend(self._regex_spans(cleaned_text, [_BUDGET_PATTERN], LABEL_BUDGET))
        candidates.extend(
            self._regex_spans(cleaned_text, list(_FURNISHING_PATTERNS), LABEL_FURNISHING)
        )
        return self._resolve_overlaps(candidates)

    @staticmethod
    def _catalog_spans(
        text: str, terms: Any, label: str, is_synonym: bool = False
    ) -> List[SpanCandidate]:
        spans: List[SpanCandidate] = []
        for term in terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                spans.append(SpanCandidate(match.start(), match.end(), label, is_synonym))
        return spans

    @staticmethod
    def _regex_spans(
        text: str, patterns: Sequence[re.Pattern[str]], label: str
    ) -> List[SpanCandidate]:
        spans: List[SpanCandidate] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                start, end = SpanLocator._trim(text, match.start(), match.end())
                if end > start:
                    spans.append(SpanCandidate(start, end, label))
        return spans

    @staticmethod
    def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
        """Drops leading/trailing whitespace so spans never straddle a token gap."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _resolve_overlaps(candidates: Sequence[SpanCandidate]) -> List[SpanCandidate]:
        """Longest match wins, synonym breaks an exact-span tie.

        Same rule as ``NLPExtractor._resolve_overlaps``, but applied across all
        entity types at once so a multi-word AREA ("Dwarka Nashik") suppresses a
        CITY match nested inside it.
        """
        ordered = sorted(
            candidates, key=lambda c: (-c.length, 0 if c.is_synonym else 1, c.start, c.label)
        )
        kept: List[SpanCandidate] = []
        for candidate in ordered:
            if any(candidate.start < k.end and candidate.end > k.start for k in kept):
                continue
            kept.append(candidate)
        return sorted(kept, key=lambda c: c.start)


# --------------------------------------------------------------------------- #
# Stratified selection
# --------------------------------------------------------------------------- #


@dataclass
class Quota:
    """A stratification target. ``maximum`` is a soft cap used for ranking."""

    name: str
    target: int
    minimum: int
    maximum: int
    predicate: Callable[[Dict[str, Any]], bool]


@dataclass
class SelectionResult:
    transcripts: List[Dict[str, Any]]
    quota_achieved: Dict[str, int]
    shortfalls: Dict[str, Dict[str, int]] = field(default_factory=dict)
    overshoots: Dict[str, Dict[str, int]] = field(default_factory=dict)
    excluded_no_city: int = 0


def _flags(record: Dict[str, Any]) -> Dict[str, Any]:
    return record["metadata"]["flags"]


def _build_quotas() -> List[Quota]:
    quotas: List[Quota] = [
        Quota("has_negation", 90, 80, 100, lambda r: bool(_flags(r)["has_negation"])),
        Quota("is_code_mixed", 90, 80, 100, lambda r: bool(_flags(r)["is_code_mixed"])),
        Quota("is_multi_city", 45, 40, 50, lambda r: bool(_flags(r)["is_multi_city"])),
        Quota("is_telegraphic", 45, 40, 55, lambda r: bool(_flags(r)["is_telegraphic"])),
    ]
    for type_name, target in RARE_TYPE_QUOTAS.items():
        quotas.append(
            Quota(
                f"rare_type:{type_name}",
                target,
                20,
                30,
                lambda r, n=type_name: _flags(r)["rare_property_type_name"] == n,
            )
        )
    for form, target in BUDGET_FORM_QUOTAS.items():
        quotas.append(
            Quota(
                f"budget_form:{form}",
                target,
                20,
                TARGET_SIZE,
                lambda r, f=form: _flags(r)["budget_form"] == f,
            )
        )
    return quotas


class StratifiedSelector:
    """Greedy quota filler that keeps the 60/40 Nashik/Pune split intact.

    Quotas are filled scarcest-first (smallest pool-to-target ratio). Each pick
    prefers the candidate that also advances the most other unmet quotas while
    pushing the fewest already-satisfied ones past their soft cap. The remainder
    is filled with straightforward (no hard-flag) transcripts.
    """

    def __init__(self, pool: Sequence[Dict[str, Any]], seed: int = SEED) -> None:
        self.quotas = _build_quotas()
        self.rng = random.Random(seed)

        self.excluded_no_city = sum(
            1 for r in pool if r["ground_truth"]["city"] not in CITY_SHARE
        )
        # A transcript with no ground-truth city cannot be assigned to either
        # side of the 60/40 split, so it is not eligible for selection.
        self.pool = [r for r in pool if r["ground_truth"]["city"] in CITY_SHARE]

        self.city_targets = {
            city: round(TARGET_SIZE * share) for city, share in CITY_SHARE.items()
        }
        self.selected: List[Dict[str, Any]] = []
        self.selected_ids: set[str] = set()
        self.city_counts: Dict[str, int] = {city: 0 for city in CITY_SHARE}
        self.counts: Dict[str, int] = {q.name: 0 for q in self.quotas}
        # Stable tie-break that does not depend on pool ordering.
        self._jitter = {r["transcript_id"]: self.rng.random() for r in self.pool}

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _is_hard_case(record: Dict[str, Any]) -> bool:
        flags = _flags(record)
        return any(
            flags[key]
            for key in (
                "has_negation",
                "is_code_mixed",
                "is_multi_city",
                "is_telegraphic",
                "is_rare_property_type",
            )
        )

    def _city_of(self, record: Dict[str, Any]) -> str:
        return record["ground_truth"]["city"]

    def _city_has_room(self, city: str) -> bool:
        return self.city_counts[city] < self.city_targets[city]

    def _neediest_cities(self) -> List[str]:
        """Cities ordered by how far behind their share they are."""
        return sorted(
            (c for c in CITY_SHARE if self._city_has_room(c)),
            key=lambda c: self.city_counts[c] / self.city_targets[c],
        )

    def _matching_quotas(self, record: Dict[str, Any]) -> List[Quota]:
        return [q for q in self.quotas if q.predicate(record)]

    def _rank_key(self, record: Dict[str, Any]) -> Tuple[int, int, float]:
        matched = self._matching_quotas(record)
        helps = sum(1 for q in matched if self.counts[q.name] < q.target)
        overshoots = sum(1 for q in matched if self.counts[q.name] >= q.maximum)
        return (-helps, overshoots, self._jitter[record["transcript_id"]])

    def _select(self, record: Dict[str, Any]) -> None:
        self.selected.append(record)
        self.selected_ids.add(record["transcript_id"])
        self.city_counts[self._city_of(record)] += 1
        for quota in self._matching_quotas(record):
            self.counts[quota.name] += 1

    def _candidates(
        self, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        return [
            r
            for r in self.pool
            if r["transcript_id"] not in self.selected_ids
            and self._city_has_room(self._city_of(r))
            and (predicate is None or predicate(r))
        ]

    # -- phases ----------------------------------------------------------- #

    def _fill_quota(self, quota: Quota) -> None:
        while self.counts[quota.name] < quota.target and len(self.selected) < TARGET_SIZE:
            candidates = self._candidates(quota.predicate)
            if not candidates:
                break
            for city in self._neediest_cities():
                in_city = [r for r in candidates if self._city_of(r) == city]
                if in_city:
                    candidates = in_city
                    break
            self._select(min(candidates, key=self._rank_key))

    def _fill_remainder(self) -> None:
        while len(self.selected) < TARGET_SIZE:
            candidates = self._candidates()
            if not candidates:
                break
            for city in self._neediest_cities():
                in_city = [r for r in candidates if self._city_of(r) == city]
                if in_city:
                    candidates = in_city
                    break
            # Straightforward cases first, but still prefer one that closes an
            # outstanding quota (a budget form, typically) if any remain.
            self._select(
                min(
                    candidates,
                    key=lambda r: (
                        self._rank_key(r)[0],
                        1 if self._is_hard_case(r) else 0,
                        self._rank_key(r)[1],
                        self._jitter[r["transcript_id"]],
                    ),
                )
            )

    def run(self) -> SelectionResult:
        ordered = sorted(
            self.quotas,
            key=lambda q: len([r for r in self.pool if q.predicate(r)]) / max(q.target, 1),
        )
        for quota in ordered:
            self._fill_quota(quota)
        self._fill_remainder()

        shortfalls = {
            q.name: {"achieved": self.counts[q.name], "target": q.target, "minimum": q.minimum}
            for q in self.quotas
            if self.counts[q.name] < q.minimum
        }
        overshoots = {
            q.name: {"achieved": self.counts[q.name], "maximum": q.maximum}
            for q in self.quotas
            if self.counts[q.name] > q.maximum
        }
        return SelectionResult(
            transcripts=sorted(self.selected, key=lambda r: r["transcript_id"]),
            quota_achieved=dict(self.counts),
            shortfalls=shortfalls,
            overshoots=overshoots,
            excluded_no_city=self.excluded_no_city,
        )


# --------------------------------------------------------------------------- #
# Label Studio emission
# --------------------------------------------------------------------------- #


@dataclass
class PreLabelStats:
    tasks: int = 0
    spans: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    negated_spans: int = 0
    unlocatable: List[Dict[str, Any]] = field(default_factory=list)
    tasks_without_spans: List[str] = field(default_factory=list)


class PreLabelBuilder:
    """Turns selected transcripts into gold_reference.json-shaped LS tasks."""

    def __init__(self, extractor: NLPExtractor) -> None:
        self.extractor = extractor
        self.locator = SpanLocator(extractor)

    def build(self, records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], PreLabelStats]:
        tasks: List[Dict[str, Any]] = []
        stats = PreLabelStats()

        for task_id, record in enumerate(records, start=1):
            raw_text = record["text"]
            transcript_id = record["transcript_id"]
            cleaned_text, index_map = build_offset_map(raw_text)

            expected = self.extractor.preprocessor.clean(raw_text)
            if cleaned_text != expected:
                raise AssertionError(
                    f"{transcript_id}: offset map diverged from TextPreprocessor.clean"
                )

            results: List[Dict[str, Any]] = []
            for span_no, candidate in enumerate(self.locator.locate(cleaned_text), start=1):
                raw_start, raw_end = map_span_to_raw(candidate.start, candidate.end, index_map)
                span_text = raw_text[raw_start:raw_end]
                # Guard: never emit a span we cannot reproduce from raw text.
                if not span_text.strip():
                    stats.unlocatable.append(
                        {
                            "transcript_id": transcript_id,
                            "label": candidate.label,
                            "cleaned_text": cleaned_text[candidate.start : candidate.end],
                        }
                    )
                    continue
                results.append(
                    self._result(f"{transcript_id}_{span_no}", raw_start, raw_end, span_text,
                                 candidate.label)
                )
                stats.spans += 1
                stats.label_counts[candidate.label] = (
                    stats.label_counts.get(candidate.label, 0) + 1
                )
                if self.extractor._is_negated(cleaned_text, candidate.start):
                    stats.negated_spans += 1

            if not results:
                stats.tasks_without_spans.append(transcript_id)

            tasks.append(
                {
                    "id": task_id,
                    "data": {"text": raw_text, "transcript_id": transcript_id},
                    "annotations": [
                        {
                            "id": task_id,
                            "completed_by": 1,
                            "result": results,
                            "was_cancelled": False,
                            "ground_truth": False,
                            "lead_time": None,
                        }
                    ],
                }
            )
            stats.tasks += 1

        return tasks, stats

    @staticmethod
    def _result(
        span_id: str, start: int, end: int, text: str, label: str
    ) -> Dict[str, Any]:
        return {
            "value": {"start": start, "end": end, "text": text, "labels": [label]},
            "id": span_id,
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "prediction",
        }


# --------------------------------------------------------------------------- #
# Verification gate
# --------------------------------------------------------------------------- #


def verify_offsets(
    tasks: Sequence[Dict[str, Any]], pool: Optional[Sequence[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Asserts ``raw_text[start:end] == span text`` for every span.

    Also catches out-of-bounds, inverted, empty, whitespace-padded and
    overlapping spans, and labels outside the schema. When ``pool`` is given,
    each task's stored text is first proven byte-identical to the original
    transcript, so offsets are verified against the true raw text rather than a
    possibly-altered copy of it.
    """
    valid_labels = {
        LABEL_CITY, LABEL_AREA, LABEL_PROPERTY_TYPE, LABEL_BHK,
        LABEL_BUDGET, LABEL_AMENITY, LABEL_FURNISHING,
    }
    problems: List[Dict[str, Any]] = []
    pool_text = {r["transcript_id"]: r["text"] for r in pool} if pool else {}
    seen_ids: set[str] = set()

    for task in tasks:
        transcript_id = task["data"].get("transcript_id", f"task_{task['id']}")
        raw_text = task["data"]["text"]
        spans: List[Tuple[int, int]] = []

        if transcript_id in seen_ids:
            problems.append({"transcript_id": transcript_id, "issue": "duplicate_task"})
        seen_ids.add(transcript_id)

        if pool_text:
            original = pool_text.get(transcript_id)
            if original is None:
                problems.append({"transcript_id": transcript_id, "issue": "not_in_source_pool"})
            elif original != raw_text:
                problems.append(
                    {"transcript_id": transcript_id, "issue": "task_text_differs_from_pool"}
                )

        for result in task["annotations"][0]["result"]:
            value = result["value"]
            start, end, text = value["start"], value["end"], value["text"]
            label = value["labels"][0]

            def flag(issue: str, **extra: Any) -> None:
                problems.append(
                    {
                        "transcript_id": transcript_id,
                        "span_id": result["id"],
                        "issue": issue,
                        "start": start,
                        "end": end,
                        "label": label,
                        "stored_text": text,
                        **extra,
                    }
                )

            if not 0 <= start < end <= len(raw_text):
                flag("out_of_bounds", raw_len=len(raw_text))
                continue
            actual = raw_text[start:end]
            if actual != text:
                flag("offset_mismatch", raw_substring=actual)
            if text != text.strip() or not text.strip():
                flag("whitespace_padded_or_empty")
            if label not in valid_labels:
                flag("unknown_label")
            if any(start < e and end > s for s, e in spans):
                flag("overlapping_span")
            spans.append((start, end))

    return problems


def log_sample(tasks: Sequence[Dict[str, Any]], count: int = 5, seed: int = SEED) -> None:
    """Prints N random pre-labeled transcripts with each span's raw substring."""
    sample = random.Random(seed).sample(list(tasks), min(count, len(tasks)))
    logger.info("")
    logger.info("=" * 78)
    logger.info("SAMPLE — %d random pre-labeled transcripts", len(sample))
    logger.info("=" * 78)
    for task in sample:
        raw_text = task["data"]["text"]
        results = task["annotations"][0]["result"]
        logger.info("")
        logger.info("--- %s (task id %s) — %d spans ---",
                    task["data"]["transcript_id"], task["id"], len(results))
        logger.info("TEXT:")
        for line in raw_text.splitlines():
            logger.info("    %s", line)
        logger.info("SPANS:")
        if not results:
            logger.info("    (none)")
        for result in results:
            value = result["value"]
            logger.info(
                "    %-14s [%4d:%4d]  %-28s | raw[start:end] = %s",
                value["labels"][0],
                value["start"],
                value["end"],
                repr(value["text"]),
                repr(raw_text[value["start"] : value["end"]]),
            )


def composition(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts the selection along every stratification axis."""

    def count(predicate: Callable[[Dict[str, Any]], bool]) -> int:
        return sum(1 for r in records if predicate(r))

    def tally(key: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in records:
            value = _flags(record)[key]
            if value is not None:
                result[str(value)] = result.get(str(value), 0) + 1
        return dict(sorted(result.items(), key=lambda kv: -kv[1]))

    cities: Dict[str, int] = {}
    for record in records:
        city = record["ground_truth"]["city"]
        cities[str(city)] = cities.get(str(city), 0) + 1

    return {
        "total": len(records),
        "city_counts": cities,
        "city_pct": {
            city: round(100.0 * n / max(len(records), 1), 1) for city, n in cities.items()
        },
        "has_negation": count(lambda r: bool(_flags(r)["has_negation"])),
        "is_code_mixed": count(lambda r: bool(_flags(r)["is_code_mixed"])),
        "is_multi_city": count(lambda r: bool(_flags(r)["is_multi_city"])),
        "is_telegraphic": count(lambda r: bool(_flags(r)["is_telegraphic"])),
        "is_rare_property_type": count(lambda r: bool(_flags(r)["is_rare_property_type"])),
        "area_contains_city_substring": count(
            lambda r: bool(_flags(r)["area_contains_city_substring"])
        ),
        "no_hard_flags": count(lambda r: not StratifiedSelector._is_hard_case(r)),
        "rare_property_type": tally("rare_property_type_name"),
        "budget_form": tally("budget_form"),
        "bhk_form": tally("bhk_form"),
        "negation_target": tally("negation_target"),
        "distractor_city": tally("distractor_city"),
        "intent": {
            intent: count(lambda r, i=intent: r["metadata"]["intent"] == i)
            for intent in sorted({r["metadata"]["intent"] for r in records})
        },
        "completeness": {
            level: count(lambda r, lv=level: r["metadata"]["completeness"] == lv)
            for level in ("full", "partial", "vague")
        },
    }


def coverage_gaps(
    records: Sequence[Dict[str, Any]], tasks: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Where the rule extractor pre-labels nothing but ground truth says it should.

    These are not defects — they are the cases the NER model is meant to learn
    and that Stage C hand-correction has to supply. Broken out by the surface
    form that caused the miss so annotation effort can be aimed at the worst
    offenders.
    """
    by_id = {r["transcript_id"]: r for r in records}
    field_to_label = {
        "city": LABEL_CITY,
        "area": LABEL_AREA,
        "property_type": LABEL_PROPERTY_TYPE,
        "bhk": LABEL_BHK,
        "budget_text": LABEL_BUDGET,
        "furnishing": LABEL_FURNISHING,
    }
    missing: Dict[str, List[str]] = {name: [] for name in field_to_label}
    missing["amenities"] = []
    budget_miss_by_form: Dict[str, int] = {}
    bhk_miss_by_form: Dict[str, int] = {}
    partial_budget_range = 0
    unlabeled_distractor_cities = 0

    for task in tasks:
        transcript_id = task["data"]["transcript_id"]
        record = by_id[transcript_id]
        truth = record["ground_truth"]
        flags = _flags(record)
        labels = [r["value"]["labels"][0] for r in task["annotations"][0]["result"]]
        span_texts = [r["value"]["text"].lower() for r in task["annotations"][0]["result"]]

        for field_name, label in field_to_label.items():
            if truth.get(field_name) is not None and label not in labels:
                missing[field_name].append(transcript_id)
                if field_name == "budget_text" and flags["budget_form"]:
                    budget_miss_by_form[flags["budget_form"]] = (
                        budget_miss_by_form.get(flags["budget_form"], 0) + 1
                    )
                if field_name == "bhk" and flags["bhk_form"]:
                    bhk_miss_by_form[flags["bhk_form"]] = (
                        bhk_miss_by_form.get(flags["bhk_form"], 0) + 1
                    )

        if len(truth.get("amenities") or []) > labels.count(LABEL_AMENITY):
            missing["amenities"].append(transcript_id)

        # A "265 to 280 lakhs" range pre-labels only its upper bound.
        if flags["budget_form"] == "range" and LABEL_BUDGET in labels:
            partial_budget_range += 1

        distractor = flags.get("distractor_city")
        if distractor and distractor.lower() not in span_texts:
            unlabeled_distractor_cities += 1

    return {
        "note": (
            "Expected rule-extractor blind spots. Stage C hand-correction supplies these."
        ),
        "missing_field_spans": {k: len(v) for k, v in missing.items()},
        "budget_misses_by_form": budget_miss_by_form,
        "bhk_misses_by_form": bhk_miss_by_form,
        "range_budgets_labeled_upper_bound_only": partial_budget_range,
        "unlabeled_distractor_city_mentions": unlabeled_distractor_cities,
        "transcript_ids": {k: v for k, v in missing.items() if v},
    }


def log_coverage_gaps(gaps: Dict[str, Any], total: int) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("KNOWN PRE-LABEL GAPS (to fix by hand in Stage C)")
    logger.info("=" * 78)
    logger.info("Ground-truth field present but no span pre-labeled, of %d transcripts:", total)
    for field_name, count in sorted(
        gaps["missing_field_spans"].items(), key=lambda kv: -kv[1]
    ):
        if count:
            logger.info("    %-16s %4d", field_name, count)
    logger.info("")
    logger.info("BUDGET misses by surface form: %s", gaps["budget_misses_by_form"])
    logger.info("BHK misses by surface form:    %s", gaps["bhk_misses_by_form"])
    logger.info(
        "Range budgets pre-labeled with upper bound only (e.g. '265 to 280 lakhs' "
        "-> '280 lakhs'): %d", gaps["range_budgets_labeled_upper_bound_only"]
    )
    logger.info(
        "Distractor-city mentions left unlabeled (schema.md defines no convention "
        "for these): %d", gaps["unlabeled_distractor_city_mentions"]
    )


def log_composition(comp: Dict[str, Any], result: SelectionResult) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("SELECTION COMPOSITION (%d transcripts)", comp["total"])
    logger.info("=" * 78)
    logger.info("")
    logger.info("City split (target %.0f%% Nashik / %.0f%% Pune):",
                CITY_SHARE["Nashik"] * 100, CITY_SHARE["Pune"] * 100)
    for city, n in comp["city_counts"].items():
        logger.info("    %-10s %4d   (%.1f%%)", city, n, comp["city_pct"][city])
    if result.excluded_no_city:
        logger.info(
            "    (%d pool transcripts with no ground-truth city were excluded "
            "so the split is exact)", result.excluded_no_city
        )

    logger.info("")
    logger.info("Hard-case flags:            selected   requested range")
    targets = {q.name: q for q in _build_quotas()}
    for key in ("has_negation", "is_code_mixed", "is_multi_city", "is_telegraphic"):
        quota = targets[key]
        logger.info("    %-24s %5d   %d-%d", key, comp[key], quota.minimum, quota.maximum)
    logger.info("    %-24s %5d   (no explicit target)", "is_rare_property_type",
                comp["is_rare_property_type"])

    logger.info("")
    logger.info("Rare property types (target 20-30 each for the 4 requested):")
    for name, n in comp["rare_property_type"].items():
        requested = " <- requested" if name in RARE_TYPE_QUOTAS else ""
        logger.info("    %-22s %4d%s", name, n, requested)

    logger.info("")
    logger.info("Budget forms (spread required):")
    for form, n in comp["budget_form"].items():
        logger.info("    %-22s %4d", form, n)
    logger.info("    %-22s %4d", "(no budget stated)",
                comp["total"] - sum(comp["budget_form"].values()))

    logger.info("")
    logger.info("BHK forms:")
    for form, n in comp["bhk_form"].items():
        logger.info("    %-22s %4d", form, n)

    logger.info("")
    logger.info("Negation targets: %s", comp["negation_target"])
    logger.info("Distractor cities present: %d transcripts %s",
                sum(comp["distractor_city"].values()), comp["distractor_city"])
    logger.info("Completeness: %s", comp["completeness"])
    logger.info("Intent: %s", comp["intent"])
    logger.info("Straightforward (no hard flags): %d", comp["no_hard_flags"])

    if result.shortfalls:
        logger.warning("")
        logger.warning("QUOTA SHORTFALLS — these could not be filled from the pool:")
        for name, detail in result.shortfalls.items():
            logger.warning(
                "    %-26s achieved %d, minimum %d, target %d",
                name, detail["achieved"], detail["minimum"], detail["target"],
            )
    else:
        logger.info("")
        logger.info("All stratification quotas met their minimum.")

    if result.overshoots:
        logger.warning("Quotas above their soft cap (flag co-occurrence): %s", result.overshoots)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def load_pool(path: Path = POOL_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], PreLabelStats, SelectionResult, Dict[str, Any]]:
    pool = load_pool()
    logger.info("Loaded %d transcripts from %s", len(pool), POOL_PATH)

    selection = StratifiedSelector(pool, seed=seed).run()
    logger.info("Selected %d transcripts for annotation", len(selection.transcripts))

    extractor = NLPExtractor()
    tasks, stats = PreLabelBuilder(extractor).build(selection.transcripts)
    logger.info("Emitted %d pre-label spans across %d tasks", stats.spans, stats.tasks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PRELABELS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(tasks, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", PRELABELS_PATH)

    comp = composition(selection.transcripts)
    gaps = coverage_gaps(selection.transcripts, tasks)
    manifest = {
        "coverage_gaps": gaps,
        "seed": seed,
        "source_pool": str(POOL_PATH),
        "pool_size": len(pool),
        "selected": len(selection.transcripts),
        "transcript_ids": [r["transcript_id"] for r in selection.transcripts],
        "composition": comp,
        "quota_achieved": selection.quota_achieved,
        "quota_shortfalls": selection.shortfalls,
        "quota_overshoots": selection.overshoots,
        "excluded_no_city": selection.excluded_no_city,
        "prelabel_stats": {
            "tasks": stats.tasks,
            "spans": stats.spans,
            "label_counts": stats.label_counts,
            "spans_in_negation_context": stats.negated_spans,
            "unlocatable_spans": stats.unlocatable,
            "tasks_without_spans": stats.tasks_without_spans,
        },
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", MANIFEST_PATH)

    return tasks, stats, selection, gaps


def verify(
    tasks: Sequence[Dict[str, Any]], pool: Optional[Sequence[Dict[str, Any]]] = None
) -> int:
    logger.info("")
    logger.info("=" * 78)
    logger.info("VERIFICATION GATE — offset check over %d tasks", len(tasks))
    logger.info("=" * 78)
    problems = verify_offsets(tasks, pool)
    total_spans = sum(len(t["annotations"][0]["result"]) for t in tasks)
    if problems:
        logger.error("FAIL — %d problem(s) across %d spans:", len(problems), total_spans)
        for problem in problems:
            logger.error("    %s", problem)
    else:
        logger.info(
            "PASS — %d/%d spans satisfy raw_text[start:end] == span text.", total_spans, total_spans
        )
        logger.info("       No overlaps, no out-of-bounds, no unknown labels, no duplicate tasks.")
        if pool:
            logger.info(
                "       All %d task texts are byte-identical to the source pool transcript.",
                len(tasks),
            )
    return len(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip generation; verify the pre-labels already on disk.",
    )
    args = parser.parse_args()

    # Transcripts contain em-dashes and romanized code-mixed text; keep the
    # console readable on a non-UTF8 Windows codepage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.verify_only:
        with PRELABELS_PATH.open("r", encoding="utf-8") as handle:
            tasks = json.load(handle)
        failures = verify(tasks, load_pool())
        log_sample(tasks, seed=args.seed)
        return 1 if failures else 0

    _, stats, selection, gaps = run(seed=args.seed)

    # Verify what was actually written to disk, not the in-memory objects.
    with PRELABELS_PATH.open("r", encoding="utf-8") as handle:
        saved_tasks = json.load(handle)

    failures = verify(saved_tasks, load_pool())
    log_sample(saved_tasks, seed=args.seed)
    log_composition(composition(selection.transcripts), selection)
    log_coverage_gaps(gaps, len(selection.transcripts))

    logger.info("")
    logger.info("Span label counts: %s", stats.label_counts)
    logger.info("Spans sitting in a negation context (tagged anyway, per schema.md "
                "section 3): %d", stats.negated_spans)
    if stats.tasks_without_spans:
        logger.info("Tasks with zero pre-labels (vague transcripts, expected): %d — %s",
                    len(stats.tasks_without_spans),
                    ", ".join(stats.tasks_without_spans[:10]))
    if stats.unlocatable:
        logger.warning("Unlocatable spans left unlabeled: %d", len(stats.unlocatable))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
