"""Stage D — head-to-head: rule-based extractor vs fine-tuned MuRIL.

Scores both systems on the SAME held-out test split with the same entity-level
seqeval metric, so the comparison is apples-to-apples.

Making the rule extractor comparable: it normally returns one collapsed value
per field, which is not a span sequence. Here it is run through the identical
``SpanLocator`` used to bootstrap Stage B pre-labels — its catalogs, synonym
tables and regexes producing character spans — which are then converted to the
same word-level BIO tags the model predicts. Both systems are therefore scored
as span taggers over identical tokens against identical gold tags.

Run:
    python -m nlp.ner.benchmark
    python -m nlp.ner.benchmark --model-dir models_store/ner/muril_ner_v1_...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nlp.extractor import NLPExtractor  # noqa: E402
from nlp.ner.bootstrap_labels import SpanLocator, build_offset_map, map_span_to_raw  # noqa: E402
from nlp.ner.export_dataset import ENTITIES, to_bio  # noqa: E402
from nlp.ner.model import MurilNerPredictor, load_rows  # noqa: E402

logger = logging.getLogger(__name__)

NER_DIR = Path(__file__).resolve().parent
DATASET_DIR = NER_DIR / "dataset"
MODELS_STORE = BACKEND_DIR / "models_store" / "ner"


def resolve_model_dir(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    pointer = MODELS_STORE / "latest.json"
    if pointer.exists():
        return Path(json.loads(pointer.read_text(encoding="utf-8"))["model_dir"])
    candidates = sorted(MODELS_STORE.glob("muril_ner_*"))
    if not candidates:
        raise SystemExit(f"No trained model found in {MODELS_STORE}. Run nlp.ner.train first.")
    return candidates[-1]


def rule_based_tags(extractor: NLPExtractor, text: str) -> List[str]:
    """Rule-extractor spans over raw text, converted to word-level BIO tags."""
    locator = SpanLocator(extractor)
    cleaned_text, index_map = build_offset_map(text)
    spans: List[Dict[str, Any]] = []
    for candidate in locator.locate(cleaned_text):
        raw_start, raw_end = map_span_to_raw(candidate.start, candidate.end, index_map)
        spans.append({"start": raw_start, "end": raw_end, "label": candidate.label})
    _, tags, _ = to_bio(text, spans)
    return tags


def score(
    predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]
) -> Dict[str, Any]:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    report = classification_report(references, predictions, output_dict=True, zero_division=0)
    return {
        "overall_precision": precision_score(references, predictions, zero_division=0),
        "overall_recall": recall_score(references, predictions, zero_division=0),
        "overall_f1": f1_score(references, predictions, zero_division=0),
        "per_entity": {
            key: value
            for key, value in report.items()
            if key not in ("micro avg", "macro avg", "weighted avg", "accuracy")
        },
    }


def entity_f1(metrics: Dict[str, Any], entity: str) -> Tuple[float, float, float, int]:
    scores = metrics["per_entity"].get(entity)
    if not scores:
        return 0.0, 0.0, 0.0, 0
    return (
        scores["precision"],
        scores["recall"],
        scores["f1-score"],
        int(scores["support"]),
    )


def log_systems(systems: Dict[str, Dict[str, Any]], test_size: int) -> None:
    """Per-entity F1 for N systems side by side, plus each one's overall."""
    names = list(systems)
    logger.info("")
    logger.info("=" * 86)
    logger.info("PER-ENTITY F1 — held-out test split (%d transcripts)", test_size)
    logger.info("=" * 86)
    header = f"  {'entity':<16}{'support':>8}" + "".join(f"{n:>14}" for n in names)
    logger.info("%s", header)
    logger.info("  %s", "-" * (24 + 14 * len(names)))

    for entity in ENTITIES:
        support = max(entity_f1(systems[n], entity)[3] for n in names)
        row = "".join(f"{entity_f1(systems[n], entity)[2]:>14.4f}" for n in names)
        logger.info("  %-16s%8d%s", entity, support, row)

    logger.info("  %s", "-" * (24 + 14 * len(names)))
    logger.info(
        "  %-16s%8s%s", "OVERALL (micro)", "",
        "".join(f"{systems[n]['overall_f1']:>14.4f}" for n in names),
    )
    best = max(names, key=lambda n: systems[n]["overall_f1"])
    logger.info("")
    logger.info("BEST OVERALL: %s (F1 %.4f)", best, systems[best]["overall_f1"])


def log_comparison(rules: Dict[str, Any], model: Dict[str, Any], test_size: int) -> None:
    logger.info("")
    logger.info("=" * 86)
    logger.info("NER vs RULES — entity-level F1 on the held-out test split (%d transcripts)",
                test_size)
    logger.info("=" * 86)
    logger.info(
        "  %-16s %8s | %-24s | %-24s",
        "", "support", "RULE-BASED extractor", "MuRIL NER",
    )
    logger.info(
        "  %-16s %8s | %7s %7s %7s | %7s %7s %7s | %8s",
        "entity", "", "P", "R", "F1", "P", "R", "F1", "ΔF1",
    )
    logger.info("  %s", "-" * 84)

    for entity in ENTITIES:
        r_p, r_r, r_f, support = entity_f1(rules, entity)
        m_p, m_r, m_f, m_support = entity_f1(model, entity)
        support = support or m_support
        delta = m_f - r_f
        marker = "  " if abs(delta) < 1e-9 else ("+ " if delta > 0 else "- ")
        logger.info(
            "  %-16s %8d | %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f | %s%+.4f",
            entity, support, r_p, r_r, r_f, m_p, m_r, m_f, marker, delta,
        )

    logger.info("  %s", "-" * 84)
    delta = model["overall_f1"] - rules["overall_f1"]
    logger.info(
        "  %-16s %8s | %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f | %+.4f",
        "OVERALL (micro)", "",
        rules["overall_precision"], rules["overall_recall"], rules["overall_f1"],
        model["overall_precision"], model["overall_recall"], model["overall_f1"],
        delta,
    )
    logger.info("")
    verdict = "BEATS" if delta > 0 else ("TIES" if abs(delta) < 1e-9 else "LOSES TO")
    logger.info("VERDICT: MuRIL NER %s the rule-based extractor "
                "(overall F1 %.4f vs %.4f, %+.4f).",
                verdict, model["overall_f1"], rules["overall_f1"], delta)


def log_hard_case_breakdown(
    rows: Sequence[Dict[str, Any]],
    rule_predictions: Sequence[Sequence[str]],
    model_predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Overall F1 restricted to each hard-case subset of the test split."""
    logger.info("")
    logger.info("=" * 86)
    logger.info("BY HARD-CASE SUBSET (overall micro-F1 within each subset)")
    logger.info("=" * 86)
    logger.info("  %-26s %8s %14s %14s %10s", "subset", "n", "rules F1", "MuRIL F1", "ΔF1")
    logger.info("  %s", "-" * 76)

    subsets: Dict[str, Any] = {}
    definitions = [
        ("all", lambda _: True),
        ("has_negation", lambda r: r["flags"]["has_negation"]),
        ("is_code_mixed", lambda r: r["flags"]["is_code_mixed"]),
        ("is_multi_city", lambda r: r["flags"]["is_multi_city"]),
        ("is_telegraphic", lambda r: r["flags"]["is_telegraphic"]),
        ("is_rare_property_type", lambda r: r["flags"]["is_rare_property_type"]),
        ("no hard flags", lambda r: not any(r["flags"].values())),
    ]

    for name, predicate in definitions:
        indices = [i for i, row in enumerate(rows) if predicate(row)]
        if not indices:
            logger.info("  %-26s %8d %14s %14s %10s", name, 0, "-", "-", "-")
            continue
        rule_subset = score([rule_predictions[i] for i in indices],
                            [references[i] for i in indices])
        model_subset = score([model_predictions[i] for i in indices],
                             [references[i] for i in indices])
        delta = model_subset["overall_f1"] - rule_subset["overall_f1"]
        logger.info("  %-26s %8d %14.4f %14.4f %+10.4f",
                    name, len(indices), rule_subset["overall_f1"],
                    model_subset["overall_f1"], delta)
        subsets[name] = {
            "n": len(indices),
            "rules_f1": rule_subset["overall_f1"],
            "model_f1": model_subset["overall_f1"],
            "delta": delta,
        }
    return subsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Score only hand-corrected transcripts. Required for an honest "
             "comparison: on unreviewed rows the gold labels ARE the rule "
             "extractor's output, so it scores ~1.0 F1 by construction.",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the comparison JSON.")
    parser.add_argument(
        "--with-hybrid",
        action="store_true",
        help="Also score the Stage E hybrid from nlp/predict.py.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    rows = load_rows(DATASET_DIR, args.split, args.reviewed_only)
    if not rows:
        raise SystemExit(f"No rows in split {args.split!r}.")
    logger.info("Held-out %s split: %d transcripts%s", args.split, len(rows),
                "  (human-reviewed only)" if args.reviewed_only else "")

    unreviewed = [r for r in rows if not r.get("reviewed")]
    if unreviewed:
        logger.warning("")
        logger.warning(
            "%d of %d scored transcripts are UNREVIEWED rule pre-labels: their "
            "gold tags are the rule extractor's own output, so its F1 there is "
            "~1.0 by construction and the comparison below is inflated in its "
            "favour. Re-run with --reviewed-only for the honest number.",
            len(unreviewed), len(rows),
        )

    model_dir = resolve_model_dir(args.model_dir)
    logger.info("Model: %s", model_dir)

    extractor = NLPExtractor()
    predictor = MurilNerPredictor(model_dir)

    references: List[List[str]] = []
    rule_predictions: List[List[str]] = []
    model_predictions: List[List[str]] = []

    for row in rows:
        references.append(row["ner_tags"])
        rule_tags = rule_based_tags(extractor, row["text"])
        model_tags = predictor.predict_words(row["tokens"])
        # Both must align with gold or seqeval would silently mis-score.
        assert len(rule_tags) == len(row["ner_tags"]), row["transcript_id"]
        assert len(model_tags) == len(row["ner_tags"]), row["transcript_id"]
        rule_predictions.append(rule_tags)
        model_predictions.append(model_tags)

    rules = score(rule_predictions, references)
    model = score(model_predictions, references)

    hybrid: Optional[Dict[str, Any]] = None
    if args.with_hybrid:
        from nlp.predict import HybridExtractor  # noqa: E402  (optional dependency)

        logger.info("Scoring the Stage E hybrid (nlp/predict.py)...")
        extractor_hybrid = HybridExtractor(model_dir=model_dir, extractor=extractor)
        hybrid_predictions: List[List[str]] = []
        for row in rows:
            spans = [
                {"start": s.start, "end": s.end, "label": s.label}
                for s in extractor_hybrid.merged_spans(row["text"])
            ]
            _, tags, _ = to_bio(row["text"], spans)
            assert len(tags) == len(row["ner_tags"]), row["transcript_id"]
            hybrid_predictions.append(tags)
        hybrid = score(hybrid_predictions, references)

    log_comparison(rules, model, len(rows))
    if hybrid is not None:
        log_systems(
            {"rules": rules, "MuRIL": model, "HYBRID": hybrid}, len(rows)
        )
    subsets = log_hard_case_breakdown(rows, rule_predictions, model_predictions, references)

    payload = {
        "split": args.split,
        "reviewed_only": bool(args.reviewed_only),
        "unreviewed_scored": len(unreviewed),
        "transcripts": len(rows),
        "model_dir": str(model_dir),
        "rule_based": rules,
        "muril_ner": model,
        "hybrid": hybrid,
        "delta_overall_f1": model["overall_f1"] - rules["overall_f1"],
        "hard_case_subsets": subsets,
    }
    output = args.output or (model_dir / "benchmark_vs_rules.json")
    output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    logger.info("")
    logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
