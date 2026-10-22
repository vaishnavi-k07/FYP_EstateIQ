"""Runs the promoted NLP, sentiment and intent models over the 50 unseen cases.

Nothing is trained here. Both modules are loaded exactly the way the webhook
loads them, and which artifact answered is printed before any metric, so a
number can always be traced to a checkpoint.

Scoring
-------
*Extraction* is scored entity-level, the seqeval convention, applied to the
module's actual output contract. ``nlp.predict`` returns normalized field
values, not raw character spans, so a "gold entity" here is a normalized value
per field rather than a span offset — the same TP/FP/FN bookkeeping seqeval
does over spans, one level further down the pipeline. Per field:

    TP  predicted value equals the gold value
    FP  a value was predicted where gold is null, or predicted the wrong value
    FN  gold has a value and prediction is null, or predicted the wrong value

A wrong value therefore costs one FP *and* one FN, which is what seqeval does
with a mislabelled span, and is why precision and recall can both fall on a
single bad guess. ``amenities`` is a set, so its TP/FP/FN come from set
difference. Scoring raw spans instead would need 50 transcripts hand-annotated
at character level and would measure a layer nothing downstream consumes; the
normalization and negation post-processing in Stage E sit *between* the spans
and the contract, and it is the contract that feeds lead scoring.

*Sentiment* is scored the ordinary way: accuracy, macro-F1, per-class
precision/recall/F1, and a 4x4 confusion matrix. Two error families are pulled
out by name because they are not equally costly to a sales team — an
Enthusiastic caller read as Frustrated (or the reverse) inverts the callback
priority, while Hesitant read as Frustrated keeps the lead in roughly the right
queue.

*Intent* is scored the same way as sentiment, with one difference: macro
averages run over the classes that actually occur in the gold labels or the
predictions. The set has no ``Schedule Visit`` case, and averaging a class with
no support in as F1 = 0 would measure the test set, not the model. Every intent
label also carries how it was reached (stated / inferred / ambiguous), and the
metrics are split on that, because most of these callers never state an intent.

Run:
    python -m eval.run_unseen50
    python -m eval.run_unseen50 --out ../docs/unseen50_report.md
    python -m eval.run_unseen50 --skip-sentiment
    python -m eval.run_unseen50 --skip-intent
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore")

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
for path in (str(BACKEND_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval.unseen50_cases import (  # noqa: E402
    CASES,
    ENTITY_FIELDS,
    INTENT_CLASSES,
    SENTIMENT_CLASSES,
    by_intent,
    by_sentiment,
)

logger = logging.getLogger(__name__)

NER_CORPUS = REPO_ROOT / "data_generation" / "output" / "synthetic_transcripts.jsonl"

# Reported as its own entity, but sourced from budget_value: the contract's
# `budget` is the raw surface string and `budget_value` the rupee figure lead
# scoring consumes, so the figure is what gets graded.
# Root cause per observed miss, read off the span trace rather than guessed.
# Keyed (case id, field). Stale entries are harmless — they only render when the
# matching miss actually occurs, so a fix silently removes the note with it.
DIAGNOSIS: Dict[Tuple[str, str], str] = {
    ("E02", "budget_value"): (
        "MuRIL *did* find the span `two and a half crore`. `BudgetParser` cannot "
        "read it: its number-word run breaks at 'half', which is in neither "
        "`_NUMBER_WORDS` nor `_NUMBER_SCALES`, so the run never reaches the unit. "
        "`_resolve_budget` then rejects the unparseable span and the "
        "`_BUDGET_SCAN` fallback fails for the same reason. Fractional spelled "
        "amounts are a gap in the parser, not in the NER model."
    ),
    ("E04", "furnishing"): (
        "MuRIL tagged the adjective `liveable` as FURNISHING. There is no "
        "furnishing catalog to validate against, so `_canonical` title-cases "
        "whatever span it is handed (nlp/predict.py:540) and any model false "
        "positive becomes a furnishing value verbatim."
    ),
    ("E09", "property_type"): (
        "`_NEGATION_CUES` in nlp/predict.py is English-only, so 'Villa nahi "
        "chahiye' is not recognised as negation. The villa span stays active and "
        "wins as the last PROPERTY_TYPE mention, overriding the flat the caller "
        "actually asked for. English negation works — see H11 and F04."
    ),
    ("N10", "budget_value"): (
        "`_BUDGET_SCAN` is anchored on a currency unit (crore/lakh/rupees/rs) so "
        "that bare numbers cannot be mistaken for money. '22000 per month' has no "
        "unit token, and no model BUDGET span survived, so nothing is reported."
    ),
    ("N11", "budget_value"): (
        "Same cause as N10: '12000 a month' carries no currency unit for "
        "`_BUDGET_SCAN` to anchor on. Monthly rent stated as a bare figure is "
        "currently unreachable."
    ),
    ("H02", "property_type"): (
        "No PROPERTY_TYPE span survives — correctly, the caller never names one. "
        "`extract()` then falls back to the rule extractor over the whole "
        "transcript (nlp/predict.py:632), which regex-matches 'house' inside the "
        "amenity 'club house' and maps it through property_type_synonyms.csv. "
        "The fallback exists to recover catalog-collision terms; here it invents "
        "a property type out of an amenity."
    ),
    ("F05", "city"): (
        "MuRIL tagged `January` as CITY. It is the only CITY candidate, so "
        "`_choose_city` returns it through the single-candidate branch "
        "(nlp/predict.py:492) without ever applying the known-city or distractor "
        "filter those lines exist for. Because city then reads non-null, the "
        "area-to-city lookup that would have given Yerawada → Pune never runs."
    ),
    ("F08", "furnishing"): (
        "Same passthrough as E04: MuRIL tagged the literal word `Furnishing` and "
        "`_canonical` title-cased it into a value. The caller said furnishing "
        "does not apply."
    ),
}

_FIELD_LABELS: Dict[str, str] = {
    "city": "CITY",
    "area": "AREA",
    "property_type": "PROPERTY_TYPE",
    "bhk": "BHK",
    "budget_value": "BUDGET",
    "amenities": "AMENITY",
    "furnishing": "FURNISHING",
}


# --------------------------------------------------------------------------- #
# Value comparison
# --------------------------------------------------------------------------- #

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())


def _key(field: str, value: Any) -> Optional[Any]:
    """A comparable form of one field value, or None for 'nothing predicted'."""
    if value is None:
        return None
    if field == "bhk":
        return int(value)
    if field == "budget_value":
        return round(float(value), 2)
    text = " ".join(str(value).split()).casefold()
    return text or None


def _value_set(field: str, value: Any) -> set:
    if field == "amenities":
        return {k for k in (_key(field, v) for v in (value or [])) if k is not None}
    key = _key(field, value)
    return {key} if key is not None else set()


def score_case(expected: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, Any]:
    """TP/FP/FN per entity for one case, plus whether each field matched exactly."""
    per_field: Dict[str, Dict[str, Any]] = {}
    for field in ENTITY_FIELDS:
        gold = _value_set(field, expected.get(field))
        pred = _value_set(field, predicted.get(field))
        per_field[field] = {
            "tp": len(gold & pred),
            "fp": len(pred - gold),
            "fn": len(gold - pred),
            "exact": gold == pred,
            "expected": expected.get(field),
            "predicted": predicted.get(field),
        }
    return per_field


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #

def _ngrams(texts: Sequence[str], max_n: int) -> Dict[int, set]:
    grams: Dict[int, set] = collections.defaultdict(set)
    for text in texts:
        tokens = _normalize(text).split()
        for n in range(3, min(max_n, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                grams[n].add(tuple(tokens[i:i + n]))
    return grams


def _longest_shared_run(text: str, grams: Dict[int, set], max_n: int) -> int:
    words = _normalize(text).split()
    for n in range(min(max_n, len(words)), 2, -1):
        pool = grams.get(n)
        if pool and any(tuple(words[i:i + n]) in pool for i in range(len(words) - n + 1)):
            return n
    return 0


def _training_texts() -> Tuple[List[str], Dict[str, int]]:
    """Every string the two training corpora could have shown a model."""
    texts: List[str] = []
    counts: Dict[str, int] = {}

    if NER_CORPUS.exists():
        rows = [
            json.loads(line)
            for line in NER_CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ner_texts = [r.get("transcript", "") for r in rows if r.get("transcript")]
        texts.extend(ner_texts)
        counts["ner_transcripts"] = len(ner_texts)
    else:
        logger.warning("NER corpus missing at %s — leakage check is partial.", NER_CORPUS)
        counts["ner_transcripts"] = 0

    try:
        from sentiment_intent.demo_set import training_phrases  # noqa: PLC0415

        phrases = training_phrases()
        texts.extend(phrases)
        counts["sentiment_phrases"] = len(phrases)
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.warning("Sentiment phrase pools unavailable (%s).", exc)
        counts["sentiment_phrases"] = 0

    return texts, counts


def leakage_report(max_n: int = 12) -> Dict[str, Any]:
    texts, counts = _training_texts()
    grams = _ngrams(texts, max_n)
    normalized = [(t, _normalize(t).strip()) for t in texts]
    # Only phrase-length strings are checked as substrings; a whole training
    # transcript appearing verbatim would be caught by the shared-run measure
    # anyway, and testing 400 long strings as substrings finds nothing.
    short = [(orig, norm) for orig, norm in normalized if 0 < len(norm.split()) <= 25]

    hits: List[Dict[str, str]] = []
    runs: Dict[str, int] = {}
    for case in CASES:
        spoken = customer_only(case["transcript"])
        haystack = _normalize(spoken)
        for original, norm in short:
            if norm and norm in haystack:
                hits.append({"case": case["id"], "phrase": original})
        runs[case["id"]] = _longest_shared_run(spoken, grams, max_n)

    return {
        "sources": counts,
        "exact_phrase_reuse": hits,
        "longest_shared_word_run": runs,
        "max_shared_run": max(runs.values()) if runs else 0,
    }


def customer_only(transcript: str) -> str:
    """The caller's turns, agent prompts dropped."""
    lines = [
        line.split(":", 1)[1].strip()
        for line in transcript.splitlines()
        if line.lower().startswith("customer:")
    ]
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_extractor() -> Tuple[Any, Dict[str, Any]]:
    """The promoted hybrid, or the rule extractor if MuRIL cannot be loaded."""
    try:
        from nlp.predict import HybridExtractor  # noqa: PLC0415

        hybrid = HybridExtractor()
        return hybrid, {
            "which": "MuRIL hybrid (nlp.predict.HybridExtractor)",
            "artifact": str(hybrid.predictor.model_dir)
            if hasattr(hybrid.predictor, "model_dir")
            else str(hybrid._resolve_model_dir(None)),
            "degraded": False,
        }
    except Exception as exc:
        logger.error("MuRIL hybrid unavailable (%s) — falling back to rules.", exc)
        from nlp.extractor import NLPExtractor  # noqa: PLC0415

        return NLPExtractor(), {
            "which": "RULE-BASED ONLY (nlp.extractor.NLPExtractor)",
            "artifact": "none — no model weights involved",
            "degraded": True,
            "reason": str(exc),
        }


def load_sentiment() -> Tuple[Optional[Any], Dict[str, Any]]:
    """The promoted 4-class sentiment head, if its artifact is on disk."""
    from sentiment_intent.data import (  # noqa: PLC0415
        DEFAULT_MAX_LENGTH,
        MODELS_STORE,
        resolve_latest,
    )

    model_dir = resolve_latest("sentiment")
    if model_dir is None:
        return None, {
            "which": "NOT LOADED",
            "artifact": None,
            "degraded": True,
            "reason": (
                f"No promoted sentiment artifact under {MODELS_STORE} "
                "(the directory does not exist). models_store/ is gitignored, "
                "so the checkpoint is not in this clone."
            ),
        }

    import torch  # noqa: PLC0415

    from sentiment_intent.predict import _Head  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _Head(model_dir, device, DEFAULT_MAX_LENGTH)
    return head, {
        "which": "DistilBERT 4-class sentiment head (sentiment_intent.predict)",
        "artifact": str(model_dir),
        "degraded": False,
        "device": str(device),
    }


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def load_intent() -> Tuple[Optional[Any], Dict[str, Any]]:
    """The promoted 6-class intent head, if its artifact is on disk."""
    from sentiment_intent.data import (  # noqa: PLC0415
        DEFAULT_MAX_LENGTH,
        MODELS_STORE,
        resolve_latest,
    )

    model_dir = resolve_latest("intent")
    if model_dir is None:
        return None, {
            "which": "NOT LOADED",
            "artifact": None,
            "degraded": True,
            "reason": f"No promoted intent artifact under {MODELS_STORE}.",
        }

    import torch  # noqa: PLC0415

    from sentiment_intent.predict import _Head  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _Head(model_dir, device, DEFAULT_MAX_LENGTH)
    return head, {
        "which": "DistilBERT 6-class intent head (sentiment_intent.predict)",
        "artifact": str(model_dir),
        "degraded": False,
        "device": str(device),
    }


def run_extraction(extractor: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in CASES:
        predicted = extractor.extract(case["transcript"])
        scored = score_case(case["expected"], predicted)
        misses = [
            {
                "field": field,
                "expected": scored[field]["expected"],
                "predicted": scored[field]["predicted"],
            }
            for field in ENTITY_FIELDS
            if not scored[field]["exact"]
        ]
        rows.append(
            {
                "id": case["id"],
                "register": case["register"],
                "style": case["style"],
                "tags": list(case["tags"]),
                "note": case["note"],
                "transcript": case["transcript"],
                "expected": case["expected"],
                "predicted": {k: predicted.get(k) for k in ENTITY_FIELDS},
                "raw_predicted": predicted,
                "per_field": scored,
                "misses": misses,
                "all_correct": not misses,
            }
        )
    return rows


def extraction_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_entity: Dict[str, Dict[str, Any]] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    slots_correct = 0
    slots_total = 0

    for field in ENTITY_FIELDS:
        tp = sum(r["per_field"][field]["tp"] for r in rows)
        fp = sum(r["per_field"][field]["fp"] for r in rows)
        fn = sum(r["per_field"][field]["fn"] for r in rows)
        exact = sum(1 for r in rows if r["per_field"][field]["exact"])
        precision, recall, f1 = _prf(tp, fp, fn)
        per_entity[_FIELD_LABELS[field]] = {
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "field_accuracy": exact / len(rows) if rows else 0.0,
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        slots_correct += exact
        slots_total += len(rows)

    micro = _prf(totals["tp"], totals["fp"], totals["fn"])
    macro_f1 = (
        sum(v["f1"] for v in per_entity.values()) / len(per_entity) if per_entity else 0.0
    )

    def subset(predicate) -> Dict[str, Any]:
        chosen = [r for r in rows if predicate(r)]
        if not chosen:
            return {"cases": 0}
        tp = sum(r["per_field"][f]["tp"] for r in chosen for f in ENTITY_FIELDS)
        fp = sum(r["per_field"][f]["fp"] for r in chosen for f in ENTITY_FIELDS)
        fn = sum(r["per_field"][f]["fn"] for r in chosen for f in ENTITY_FIELDS)
        p, r_, f = _prf(tp, fp, fn)
        correct = sum(
            1 for r in chosen for f in ENTITY_FIELDS if r["per_field"][f]["exact"]
        )
        return {
            "cases": len(chosen),
            "micro_precision": p, "micro_recall": r_, "micro_f1": f,
            "field_accuracy": correct / (len(chosen) * len(ENTITY_FIELDS)),
            "fully_correct_cases": sum(1 for r in chosen if r["all_correct"]),
        }

    tags = sorted({t for r in rows for t in r["tags"]})
    return {
        "per_entity": per_entity,
        "micro_precision": micro[0],
        "micro_recall": micro[1],
        "micro_f1": micro[2],
        "macro_f1": macro_f1,
        "field_accuracy": slots_correct / slots_total if slots_total else 0.0,
        "field_slots": slots_total,
        "fully_correct_cases": sum(1 for r in rows if r["all_correct"]),
        "total_cases": len(rows),
        "by_register": {
            reg: subset(lambda r, reg=reg: r["register"] == reg)
            for reg in ("english", "code-mixed")
        },
        "by_style": {
            st: subset(lambda r, st=st: r["style"] == st) for st in ("clean", "messy")
        },
        "by_tag": {tag: subset(lambda r, tag=tag: tag in r["tags"]) for tag in tags},
    }


def run_classifier(head: Any, gold_key: str) -> List[Dict[str, Any]]:
    """Scores every case with one classification head against ``case[gold_key]``."""
    from sentiment_intent.data import customer_turns  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for case in CASES:
        text = customer_turns(case["transcript"])
        result = head.predict(text)
        rows.append(
            {
                "id": case["id"],
                "register": case["register"],
                "style": case["style"],
                "tags": list(case["tags"]),
                "intent_basis": case["intent_basis"],
                "transcript": case["transcript"],
                "scored_text": text,
                "expected": case[gold_key],
                "predicted": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "correct": result["label"] == case[gold_key],
            }
        )
    return rows


def classification_metrics(
    rows: Sequence[Dict[str, Any]], labels: Sequence[str]
) -> Dict[str, Any]:
    """Accuracy, per-class P/R/F1, macro averages and the confusion matrix.

    Macro averages cover the classes present in gold or predictions — for the
    balanced sentiment set that is all four, for intent it drops the class the
    test set never contains.
    """
    labels = list(labels)
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in labels if p != label)
        fp = sum(matrix[g][label] for g in labels if g != label)
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class[label] = {
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    averaged = [
        label for label in labels
        if per_class[label]["support"] or per_class[label]["fp"]
    ]
    correct = sum(1 for r in rows if r["correct"])

    def macro(key: str) -> float:
        return sum(per_class[c][key] for c in averaged) / len(averaged) if averaged else 0.0

    def subset(predicate) -> Dict[str, Any]:
        chosen = [r for r in rows if predicate(r)]
        if not chosen:
            return {"cases": 0}
        present = sorted({r["expected"] for r in chosen})
        f1s = []
        for label in present:
            tp = sum(1 for r in chosen if r["expected"] == label and r["correct"])
            fn = sum(1 for r in chosen if r["expected"] == label and not r["correct"])
            fp = sum(
                1 for r in chosen
                if r["predicted"] == label and r["expected"] != label
            )
            f1s.append(_prf(tp, fp, fn)[2])
        return {
            "cases": len(chosen),
            "accuracy": sum(1 for r in chosen if r["correct"]) / len(chosen),
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }

    return {
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "averaged_over": averaged,
        "per_class": per_class,
        "confusion": matrix,
        "by_register": {
            reg: subset(lambda r, reg=reg: r["register"] == reg)
            for reg in ("english", "code-mixed")
        },
        "by_style": {
            st: subset(lambda r, st=st: r["style"] == st) for st in ("clean", "messy")
        },
        "_subset": subset,
    }


def sentiment_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = classification_metrics(rows, SENTIMENT_CLASSES)
    metrics.pop("_subset")
    dangerous = [
        r for r in rows
        if {r["expected"], r["predicted"]} == {"Enthusiastic", "Frustrated"}
    ]
    safe = [
        r for r in rows
        if {r["expected"], r["predicted"]} == {"Hesitant", "Frustrated"}
    ]
    metrics.update({
        "dangerous_errors": [
            {"id": r["id"], "expected": r["expected"], "predicted": r["predicted"],
             "confidence": r["confidence"]}
            for r in dangerous
        ],
        "safe_errors": [
            {"id": r["id"], "expected": r["expected"], "predicted": r["predicted"],
             "confidence": r["confidence"]}
            for r in safe
        ],
    })
    return metrics


def intent_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = classification_metrics(rows, INTENT_CLASSES)
    subset = metrics.pop("_subset")
    counts = by_intent()
    majority = max(counts, key=counts.get)
    metrics.update({
        "by_basis": {
            basis: subset(lambda r, basis=basis: r["intent_basis"] == basis)
            for basis in ("stated", "inferred", "ambiguous")
        },
        "majority_class": majority,
        "majority_baseline_accuracy": counts[majority] / len(CASES),
    })
    return metrics


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "(none)"
    if isinstance(value, float):
        return f"{value:,.0f}" if value >= 1000 else f"{value:g}"
    return str(value)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _one_line(text: str, width: int = 150) -> str:
    spoken = " ".join(customer_only(text).split())
    return spoken if len(spoken) <= width else spoken[: width - 1] + "…"


def render_report(result: Dict[str, Any]) -> str:
    ex_rows: List[Dict[str, Any]] = result["extraction"]["rows"]
    ex: Dict[str, Any] = result["extraction"]["metrics"]
    leak = result["leakage"]
    out: List[str] = []
    w = out.append

    w("# EstateIQ — held-out evaluation on 50 unseen transcripts\n")
    w("## 0. What actually ran\n")
    w("| Module | Loaded | Artifact |")
    w("|---|---|---|")
    w(f"| Extraction | {result['extractor_info']['which']} | "
      f"`{result['extractor_info']['artifact']}` |")
    w(f"| Sentiment | {result['sentiment_info']['which']} | "
      f"`{result['sentiment_info']['artifact']}` |")
    w(f"| Intent | {result['intent_info']['which']} | "
      f"`{result['intent_info']['artifact']}` |")
    w("")
    for info, name in (
        (result["extractor_info"], "Extraction"),
        (result["sentiment_info"], "Sentiment"),
        (result["intent_info"], "Intent"),
    ):
        if info.get("degraded"):
            w(f"> **{name} ran degraded or not at all.** {info.get('reason', '')}\n")

    counts = by_sentiment()
    w("## 1. The test set\n")
    w(f"50 transcripts, written for this evaluation. Sentiment balance: "
      + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".")
    w(f"Register: {sum(1 for c in CASES if c['register'] == 'english')} English, "
      f"{sum(1 for c in CASES if c['register'] == 'code-mixed')} code-mixed (Hinglish). "
      f"Style: {sum(1 for c in CASES if c['style'] == 'clean')} clean, "
      f"{sum(1 for c in CASES if c['style'] == 'messy')} disfluent.\n")
    w("**Unseen check** — every case measured against the training corpora:\n")
    w(f"- Sources checked: {leak['sources'].get('ner_transcripts', 0)} NER training "
      f"transcripts, {leak['sources'].get('sentiment_phrases', 0)} sentiment "
      f"generator phrases.")
    w(f"- Verbatim training phrases reused: **{len(leak['exact_phrase_reuse'])}**.")
    w(f"- Longest contiguous word run any case shares with training text: "
      f"**{leak['max_shared_run']} words**.\n")
    if leak["exact_phrase_reuse"]:
        w("Reuse found (must be zero):\n")
        for hit in leak["exact_phrase_reuse"]:
            w(f"- `{hit['case']}` — {hit['phrase']!r}")
        w("")

    # ---------------- Extraction ---------------- #
    w("## 2. Extraction — per-case results\n")
    w("`✅` = all seven fields exact. Misses list `field: expected → predicted`.\n")
    w("| # | Reg | Tags | Result | Misses |")
    w("|---|---|---|---|---|")
    for row in ex_rows:
        misses = (
            "; ".join(
                f"**{_FIELD_LABELS[m['field']]}**: {_fmt(m['expected'])} → "
                f"{_fmt(m['predicted'])}"
                for m in row["misses"]
            )
            or "—"
        )
        tags = ", ".join(row["tags"]) or "—"
        w(f"| {row['id']} | {row['register'][:4]} | {tags} | "
          f"{'✅' if row['all_correct'] else '❌'} | {misses} |")
    w("")

    w("### 2.1 Per-entity precision / recall / F1\n")
    w("| Entity | Support | TP | FP | FN | Precision | Recall | F1 | Field acc. |")
    w("|---|---|---|---|---|---|---|---|---|")
    for label, stats in ex["per_entity"].items():
        w(f"| {label} | {stats['support']} | {stats['tp']} | {stats['fp']} | "
          f"{stats['fn']} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
          f"{stats['f1']:.3f} | {_pct(stats['field_accuracy'])} |")
    w(f"| **micro (all)** | {sum(s['support'] for s in ex['per_entity'].values())} | "
      f"{sum(s['tp'] for s in ex['per_entity'].values())} | "
      f"{sum(s['fp'] for s in ex['per_entity'].values())} | "
      f"{sum(s['fn'] for s in ex['per_entity'].values())} | "
      f"{ex['micro_precision']:.3f} | {ex['micro_recall']:.3f} | "
      f"**{ex['micro_f1']:.3f}** | {_pct(ex['field_accuracy'])} |")
    w("")
    w(f"- Overall extraction accuracy (exact field matches): "
      f"**{_pct(ex['field_accuracy'])}** "
      f"({int(round(ex['field_accuracy'] * ex['field_slots']))}/{ex['field_slots']} "
      f"field slots).")
    w(f"- Cases with all seven fields correct: "
      f"**{ex['fully_correct_cases']}/{ex['total_cases']}**.")
    w(f"- Macro-F1 across the seven entities: **{ex['macro_f1']:.3f}**.\n")

    w("### 2.2 Extraction by subset\n")
    w("| Subset | Cases | Micro-F1 | Field acc. | Fully correct |")
    w("|---|---|---|---|---|")
    for group in ("by_register", "by_style", "by_tag"):
        for name, stats in ex[group].items():
            if not stats.get("cases"):
                continue
            w(f"| {name} | {stats['cases']} | {stats['micro_f1']:.3f} | "
              f"{_pct(stats['field_accuracy'])} | "
              f"{stats['fully_correct_cases']}/{stats['cases']} |")
    w("")

    w("### 2.3 Every extraction miss, in full\n")
    misses = [(r, m) for r in ex_rows for m in r["misses"]]
    if not misses:
        w("None.\n")
    else:
        for row, miss in misses:
            w(f"**{row['id']}** ({row['register']}, {row['style']}"
              + (f", {', '.join(row['tags'])}" if row["tags"] else "")
              + f") — `{_FIELD_LABELS[miss['field']]}`")
            w(f"> {_one_line(row['transcript'])}")
            w(f"- expected: `{_fmt(miss['expected'])}`")
            w(f"- predicted: `{_fmt(miss['predicted'])}`")
            if row["note"]:
                w(f"- why the case is hard: {row['note']}")
            cause = DIAGNOSIS.get((row["id"], miss["field"]))
            if cause:
                w(f"- root cause: {cause}")
            w("")

    # ---------------- Sentiment ---------------- #
    w("## 3. Sentiment\n")
    if result["sentiment"] is None:
        w("**Not run.** " + result["sentiment_info"].get("reason", "") + "\n")
        w("Ground-truth sentiment labels for all 50 cases are in "
          "`backend/eval/unseen50_cases.py`; re-running this script once the "
          "artifact is restored fills this section in with no other change.\n")
        w(render_intent(result))
        w(render_summary(result))
        return "\n".join(out)

    sent_rows: List[Dict[str, Any]] = result["sentiment"]["rows"]
    sm: Dict[str, Any] = result["sentiment"]["metrics"]

    w("### 3.1 Per-case results\n")
    w("| # | Reg | Transcript (customer turns) | Expected | Predicted | Conf. | |")
    w("|---|---|---|---|---|---|---|")
    for row in sent_rows:
        w(f"| {row['id']} | {row['register'][:4]} | {_one_line(row['transcript'], 110)} | "
          f"{row['expected']} | {row['predicted']} | {row['confidence']:.2f} | "
          f"{'✅' if row['correct'] else '❌'} |")
    w("")

    w("### 3.2 Headline\n")
    w(f"- Accuracy: **{_pct(sm['accuracy'])}** ({sm['correct']}/{sm['total']})")
    w(f"- Macro-F1: **{sm['macro_f1']:.3f}**")
    w("- Majority-class baseline on this set: "
      f"{_pct(max(by_sentiment().values()) / len(CASES))}\n")

    w("### 3.3 Per-class precision / recall / F1\n")
    w("| Class | Support | TP | FP | FN | Precision | Recall | F1 |")
    w("|---|---|---|---|---|---|---|---|")
    for label, stats in sm["per_class"].items():
        w(f"| {label} | {stats['support']} | {stats['tp']} | {stats['fp']} | "
          f"{stats['fn']} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
          f"{stats['f1']:.3f} |")
    w("")

    w("### 3.4 Confusion matrix (rows = truth, columns = predicted)\n")
    w("| truth \\ pred | " + " | ".join(SENTIMENT_CLASSES) + " |")
    w("|---" * (len(SENTIMENT_CLASSES) + 1) + "|")
    for gold in SENTIMENT_CLASSES:
        cells = []
        for pred in SENTIMENT_CLASSES:
            count = sm["confusion"][gold][pred]
            cells.append(f"**{count}**" if gold == pred and count else str(count))
        w(f"| **{gold}** | " + " | ".join(cells) + " |")
    w("")

    w("### 3.5 Dangerous vs safe confusions\n")
    w(f"- **Enthusiastic ↔ Frustrated (dangerous — inverts callback priority): "
      f"{len(sm['dangerous_errors'])}**")
    for err in sm["dangerous_errors"]:
        w(f"  - `{err['id']}` {err['expected']} → {err['predicted']} "
          f"(conf {err['confidence']:.2f})")
    w(f"- **Hesitant ↔ Frustrated (safe — lead stays in roughly the right queue): "
      f"{len(sm['safe_errors'])}**")
    for err in sm["safe_errors"]:
        w(f"  - `{err['id']}` {err['expected']} → {err['predicted']} "
          f"(conf {err['confidence']:.2f})")
    w("")

    w("### 3.6 English vs code-mixed\n")
    w("| Subset | Cases | Accuracy | Macro-F1 |")
    w("|---|---|---|---|")
    for group in ("by_register", "by_style"):
        for name, stats in sm[group].items():
            if not stats.get("cases"):
                continue
            w(f"| {name} | {stats['cases']} | {_pct(stats['accuracy'])} | "
              f"{stats['macro_f1']:.3f} |")
    w("")

    w("### 3.7 Every sentiment miss, in full\n")
    wrong = [r for r in sent_rows if not r["correct"]]
    if not wrong:
        w("None.\n")
    for row in wrong:
        w(f"**{row['id']}** ({row['register']}, {row['style']}) — expected "
          f"**{row['expected']}**, predicted **{row['predicted']}** at "
          f"{row['confidence']:.2f}")
        w(f"> {_one_line(row['transcript'])}")
        dist = ", ".join(f"{k} {v:.2f}" for k, v in row["probabilities"].items())
        w(f"- full distribution: {dist}")
        w("")

    w(render_intent(result))
    w(render_summary(result))
    return "\n".join(out)


def render_intent(result: Dict[str, Any]) -> str:
    out: List[str] = []
    w = out.append
    w("## 4. Intent\n")
    if result.get("intent") is None:
        w("**Not run.** " + result["intent_info"].get("reason", "") + "\n")
        return "\n".join(out)

    rows: List[Dict[str, Any]] = result["intent"]["rows"]
    im: Dict[str, Any] = result["intent"]["metrics"]
    counts = by_intent()
    basis = collections.Counter(c["intent_basis"] for c in CASES)

    w("> **Read this first.** These 50 calls were written for extraction and "
      "sentiment, not intent, so most callers never say what they want to do: "
      f"{basis['stated']} labels are *stated*, {basis['inferred']} *inferred* and "
      f"{basis['ambiguous']} *ambiguous* (split in §4.5). The set is "
      + ", ".join(f"{k} {v}" for k, v in counts.items())
      + f" — a model that always answered **{im['majority_class']}** would score "
      f"{_pct(im['majority_baseline_accuracy'])}. A class with 1–2 cases moves "
      "its per-class numbers by 50 points or more on a single call.\n")

    w("### 4.1 Per-case results\n")
    w("| # | Basis | Transcript (customer turns) | Expected | Predicted | Conf. | |")
    w("|---|---|---|---|---|---|---|")
    for row in rows:
        w(f"| {row['id']} | {row['intent_basis']} | "
          f"{_one_line(row['transcript'], 100)} | {row['expected']} | "
          f"{row['predicted']} | {row['confidence']:.2f} | "
          f"{'✅' if row['correct'] else '❌'} |")
    w("")

    w("### 4.2 Headline\n")
    w(f"- Accuracy: **{_pct(im['accuracy'])}** ({im['correct']}/{im['total']})")
    w(f"- Majority-class baseline (always {im['majority_class']}): "
      f"{_pct(im['majority_baseline_accuracy'])}")
    w(f"- Macro precision / recall / F1: {im['macro_precision']:.3f} / "
      f"{im['macro_recall']:.3f} / **{im['macro_f1']:.3f}** "
      f"(averaged over: {', '.join(im['averaged_over'])})\n")

    w("### 4.3 Per-class precision / recall / F1\n")
    w("| Class | Support | TP | FP | FN | Precision | Recall | F1 |")
    w("|---|---|---|---|---|---|---|---|")
    for label, stats in im["per_class"].items():
        w(f"| {label} | {stats['support']} | {stats['tp']} | {stats['fp']} | "
          f"{stats['fn']} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
          f"{stats['f1']:.3f} |")
    w("")

    w("### 4.4 Confusion matrix (rows = truth, columns = predicted)\n")
    w("| truth \\ pred | " + " | ".join(INTENT_CLASSES) + " |")
    w("|---" * (len(INTENT_CLASSES) + 1) + "|")
    for gold in INTENT_CLASSES:
        cells = []
        for pred in INTENT_CLASSES:
            count = im["confusion"][gold][pred]
            cells.append(f"**{count}**" if gold == pred and count else str(count))
        w(f"| **{gold}** | " + " | ".join(cells) + " |")
    w("")

    w("### 4.5 By how the label was reached, and by subset\n")
    w("| Subset | Cases | Accuracy | Macro-F1 |")
    w("|---|---|---|---|")
    for group in ("by_basis", "by_register", "by_style"):
        for name, stats in im[group].items():
            if not stats.get("cases"):
                continue
            w(f"| {name} | {stats['cases']} | {_pct(stats['accuracy'])} | "
              f"{stats['macro_f1']:.3f} |")
    w("")

    w("### 4.6 Every intent miss, in full\n")
    wrong = [r for r in rows if not r["correct"]]
    if not wrong:
        w("None.\n")
    for row in wrong:
        w(f"**{row['id']}** ({row['intent_basis']} label, {row['register']}) — "
          f"expected **{row['expected']}**, predicted **{row['predicted']}** at "
          f"{row['confidence']:.2f}")
        w(f"> {_one_line(row['transcript'])}")
        dist = ", ".join(f"{k} {v:.2f}" for k, v in row["probabilities"].items())
        w(f"- full distribution: {dist}")
        w("")
    return "\n".join(out)


# Failure families, so eight scattered misses read as a short list of causes
# rather than eight unrelated incidents. Cases are attributed by (id, field), so
# a family that stops firing disappears from the report on its own.
_FAILURE_FAMILIES: Tuple[Tuple[str, Tuple[Tuple[str, str], ...], str], ...] = (
    (
        "Model spans become furnishing values unchecked",
        (("E04", "furnishing"), ("F08", "furnishing")),
        "FURNISHING is the one model-owned field with no catalog behind it, so "
        "`_canonical` title-cases any span MuRIL emits. Constraining it to the "
        "four canonical values would turn both of these into correct nulls.",
    ),
    (
        "Budget forms the parser cannot reach",
        (("E02", "budget_value"), ("N10", "budget_value"), ("N11", "budget_value")),
        "Fractional spelled amounts ('two and a half crore') and bare monthly "
        "rent with no unit word ('22000 per month'). Both are parser gaps, not "
        "NER gaps — in E02 the model found the right span and the parser "
        "dropped it. Monthly rent matters for the Rent intent specifically.",
    ),
    (
        "Negation is English-only",
        (("E09", "property_type"),),
        "English negation is handled correctly (H11, F04, N09, N12 all pass). "
        "Hindi/Marathi cues — nahi, nako, mat — are absent from "
        "`_NEGATION_CUES`, so a code-mixed caller's ruled-out requirement is "
        "read as a wanted one.",
    ),
    (
        "Unvalidated single-candidate city",
        (("F05", "city"),),
        "`_choose_city` applies its known-city and distractor filters only when "
        "there are two or more candidates, so a lone model false positive is "
        "accepted and suppresses the area-to-city lookup that would have "
        "corrected it.",
    ),
    (
        "Rule fallback invents a property type",
        (("H02", "property_type"),),
        "The whole-transcript rule fallback matches substrings, so 'house' "
        "inside the amenity 'club house' produces a property type the caller "
        "never asked for.",
    ),
)


def render_summary(result: Dict[str, Any]) -> str:
    ex = result["extraction"]["metrics"]
    ex_rows = result["extraction"]["rows"]
    sentiment = result["sentiment"]
    out: List[str] = []
    w = out.append

    w("\n## 5. Final performance matrix — all modules\n")
    w("One run, same 50 transcripts. Extraction is scored per field slot "
      "(7 fields × 50 cases); sentiment and intent per transcript. Precision, "
      "recall and F1 are macro averages unless marked micro.\n")
    w("| Module | Unit | Accuracy | Precision | Recall | F1 | Baseline |")
    w("|---|---|---|---|---|---|---|")
    ents = list(ex["per_entity"].values())
    ex_macro_p = sum(e["precision"] for e in ents) / len(ents)
    ex_macro_r = sum(e["recall"] for e in ents) / len(ents)
    w(f"| NLP extraction (micro) | {ex['field_slots']} field slots | "
      f"{_pct(ex['field_accuracy'])} | {ex['micro_precision']:.3f} | "
      f"{ex['micro_recall']:.3f} | **{ex['micro_f1']:.3f}** | — |")
    w(f"| NLP extraction (macro, 7 fields) | {ex['field_slots']} field slots | "
      f"{_pct(ex['field_accuracy'])} | {ex_macro_p:.3f} | {ex_macro_r:.3f} | "
      f"{ex['macro_f1']:.3f} | — |")
    for key, name, counts in (
        ("sentiment", "Sentiment (4-class)", by_sentiment()),
        ("intent", "Intent (6-class)", by_intent()),
    ):
        block = result.get(key)
        if block is None:
            w(f"| {name} | — | not run | — | — | — | — |")
            continue
        m = block["metrics"]
        w(f"| {name} | {m['total']} transcripts | **{_pct(m['accuracy'])}** | "
          f"{m['macro_precision']:.3f} | {m['macro_recall']:.3f} | "
          f"**{m['macro_f1']:.3f}** | "
          f"majority {_pct(max(counts.values()) / len(CASES))} |")
    w("")
    w("Per-field extraction scores are in §2.1, per-class sentiment in §3.3, "
      "per-class intent in §4.3.\n")

    w("\n## 6. Metrics summary\n")
    w("| | Extraction | Sentiment |")
    w("|---|---|---|")
    w(f"| Ran | {result['extractor_info']['which']} | "
      f"{result['sentiment_info']['which']} |")
    if sentiment:
        sm = sentiment["metrics"]
        w(f"| Headline | micro-F1 **{ex['micro_f1']:.3f}** | accuracy "
          f"**{_pct(sm['accuracy'])}** |")
        w(f"| Secondary | macro-F1 {ex['macro_f1']:.3f} | macro-F1 "
          f"{sm['macro_f1']:.3f} |")
        w(f"| Accuracy | {_pct(ex['field_accuracy'])} of "
          f"{ex['field_slots']} field slots | {sm['correct']}/{sm['total']} "
          f"transcripts |")
        w(f"| Clean sweeps | {ex['fully_correct_cases']}/{ex['total_cases']} "
          f"cases, all 7 fields | — |")
        w(f"| English | micro-F1 {ex['by_register']['english']['micro_f1']:.3f} | "
          f"accuracy {_pct(sm['by_register']['english']['accuracy'])} |")
        w(f"| Code-mixed | "
          f"micro-F1 {ex['by_register']['code-mixed']['micro_f1']:.3f} | "
          f"accuracy {_pct(sm['by_register']['code-mixed']['accuracy'])} |")
        w(f"| Dangerous confusions | — | "
          f"{len(sm['dangerous_errors'])} Enthusiastic↔Frustrated |")
        w(f"| Safe confusions | — | "
          f"{len(sm['safe_errors'])} Hesitant↔Frustrated |")
    else:
        w(f"| Headline | micro-F1 **{ex['micro_f1']:.3f}** | not run |")
        w(f"| Secondary | macro-F1 {ex['macro_f1']:.3f} | not run |")
        w(f"| Accuracy | {_pct(ex['field_accuracy'])} of "
          f"{ex['field_slots']} field slots | not run |")
        w(f"| Clean sweeps | {ex['fully_correct_cases']}/{ex['total_cases']} "
          f"cases, all 7 fields | not run |")
        w(f"| English | micro-F1 {ex['by_register']['english']['micro_f1']:.3f} | "
          f"not run |")
        w(f"| Code-mixed | "
          f"micro-F1 {ex['by_register']['code-mixed']['micro_f1']:.3f} | not run |")
    w("")

    observed = {(r["id"], m["field"]) for r in ex_rows for m in r["misses"]}
    families = [
        (name, [c for c in cases if c in observed], why)
        for name, cases, why in _FAILURE_FAMILIES
    ]
    families = [f for f in families if f[1]]

    w("### 6.1 What is wrong with extraction, grouped\n")
    if not families:
        w("No misses to group.\n")
    for index, (name, cases, why) in enumerate(families, start=1):
        ids = ", ".join(f"`{cid}`" for cid, _ in cases)
        w(f"{index}. **{name}** — {len(cases)} miss"
          f"{'es' if len(cases) != 1 else ''} ({ids}). {why}")
    unexplained = observed - {
        c for _, cases, _ in _FAILURE_FAMILIES for c in cases
    }
    if unexplained:
        w("")
        w("Not yet attributed to a family: "
          + ", ".join(f"`{cid}`/{_FIELD_LABELS[f]}" for cid, f in sorted(unexplained))
          + ". Diagnose before presenting.")
    w("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Write the markdown report here.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write raw results here.")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-intent", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info("Checking the 50 cases against the training corpora...")
    leak = leakage_report()
    logger.info(
        "  verbatim training phrases reused: %d | longest shared word run: %d",
        len(leak["exact_phrase_reuse"]), leak["max_shared_run"],
    )

    extractor, extractor_info = load_extractor()
    logger.info("Extraction: %s", extractor_info["which"])
    logger.info("  artifact: %s", extractor_info["artifact"])
    ex_rows = run_extraction(extractor)
    ex_metrics = extraction_metrics(ex_rows)
    logger.info(
        "  micro-F1 %.3f | field accuracy %.1f%% | %d/%d cases fully correct",
        ex_metrics["micro_f1"], ex_metrics["field_accuracy"] * 100,
        ex_metrics["fully_correct_cases"], ex_metrics["total_cases"],
    )

    sentiment: Optional[Dict[str, Any]] = None
    if args.skip_sentiment:
        sentiment_info = {"which": "SKIPPED (--skip-sentiment)", "artifact": None,
                          "degraded": True, "reason": "Skipped by flag."}
    else:
        head, sentiment_info = load_sentiment()
        logger.info("Sentiment: %s", sentiment_info["which"])
        if head is not None:
            logger.info("  artifact: %s", sentiment_info["artifact"])
            rows = run_classifier(head, "sentiment")
            sentiment = {"rows": rows, "metrics": sentiment_metrics(rows)}
            logger.info(
                "  accuracy %.1f%% | macro-F1 %.3f",
                sentiment["metrics"]["accuracy"] * 100,
                sentiment["metrics"]["macro_f1"],
            )
        else:
            logger.error("  %s", sentiment_info["reason"])

    intent: Optional[Dict[str, Any]] = None
    if args.skip_intent:
        intent_info = {"which": "SKIPPED (--skip-intent)", "artifact": None,
                       "degraded": True, "reason": "Skipped by flag."}
    else:
        intent_head, intent_info = load_intent()
        logger.info("Intent: %s", intent_info["which"])
        if intent_head is not None:
            logger.info("  artifact: %s", intent_info["artifact"])
            rows = run_classifier(intent_head, "intent")
            intent = {"rows": rows, "metrics": intent_metrics(rows)}
            logger.info(
                "  accuracy %.1f%% (majority baseline %.1f%%) | macro-F1 %.3f",
                intent["metrics"]["accuracy"] * 100,
                intent["metrics"]["majority_baseline_accuracy"] * 100,
                intent["metrics"]["macro_f1"],
            )
        else:
            logger.error("  %s", intent_info["reason"])

    result = {
        "leakage": leak,
        "extractor_info": extractor_info,
        "sentiment_info": sentiment_info,
        "intent_info": intent_info,
        "extraction": {"rows": ex_rows, "metrics": ex_metrics},
        "sentiment": sentiment,
        "intent": intent,
    }

    report = render_report(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.out)
    else:
        print()
        print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Raw results written to %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
