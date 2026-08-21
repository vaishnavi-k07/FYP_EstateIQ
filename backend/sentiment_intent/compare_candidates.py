"""Head-to-head: synthetic-trained vs pretrained vs hybrid, on identical rows.

Three ways to get a 4-tone sentiment head, all scored on the same partition of
the v4 corpus (560 train / 120 seen-idea test / 416 unseen-idea test):

1. **v4 baseline** — DistilBERT, SST-2 warm-start, fine-tuned on our synthetic
   corpus. Strong where it has seen the emotional idea, weak where it has not.
2. **zero-shot pretrained** — a published emotion classifier run as shipped,
   its label space folded onto our four tones.
3. **hybrid** — a published emotion checkpoint's *encoder* with a fresh 4-class
   head fine-tuned on the same 560 transcripts as (1).

The number that decides this is the unseen-idea score, not the headline. The
baseline's known weakness is generalising to emotional expression it has never
met, and a template matcher scores 1.0 on any seen-idea split of a
template-generated corpus, so the seen-idea column is context, not evidence.

Run (after phrase_holdout and pretrained_baseline have produced their reports):
    python -m sentiment_intent.compare_candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MODULE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = MODULE_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logger = logging.getLogger(__name__)

REPORT_PATH = MODULE_DIR / "candidate_comparison.json"
ZEROSHOT_REPORT = MODULE_DIR / "pretrained_zeroshot_report.json"

# Holdout reports to include, in the order they should be presented.
HOLDOUT_REPORTS: Dict[str, str] = {
    "v4 baseline (DistilBERT + SST-2)": "phrase_holdout_report_v4.json",
    "hybrid: emotion-distilroberta": "phrase_holdout_report_v4_hybrid-emodistilroberta.json",
    "hybrid: goemotions-roberta": "phrase_holdout_report_v4_hybrid-goemotions.json",
    "hybrid: xlm-emo (multilingual)": "phrase_holdout_report_v4_hybrid-xlmemo.json",
}


def _polarity(metrics: Dict[str, Any]) -> Dict[str, Any]:
    cm = metrics["confusion_matrix"]
    labels, matrix = cm["labels"], cm["matrix"]
    e, f = labels.index("Enthusiastic"), labels.index("Frustrated")
    e_total, f_total = sum(matrix[e]), sum(matrix[f])
    return {
        "enth_to_frus": matrix[e][f],
        "enth_to_frus_rate": matrix[e][f] / e_total if e_total else None,
        "frus_to_enth": matrix[f][e],
        "frus_to_enth_rate": matrix[f][e] / f_total if f_total else None,
    }


def _row(name: str, kind: str, seen: Dict[str, Any], unseen: Dict[str, Any],
         registers: Optional[Dict[str, Any]], base_model: Optional[str]) -> Dict[str, Any]:
    enth = unseen["per_class"]["Enthusiastic"]
    row = {
        "candidate": name,
        "kind": kind,
        "base_model": base_model,
        "seen_idea_macro_f1": seen["macro_f1"] if seen else None,
        "unseen_idea_macro_f1": unseen["macro_f1"],
        "unseen_idea_accuracy": unseen["accuracy"],
        "gap": (seen["macro_f1"] - unseen["macro_f1"]) if seen else None,
        "enthusiastic_recall": enth["recall"],
        "enthusiastic_precision": enth["precision"],
        "enthusiastic_f1": enth["f1-score"],
        "polarity": _polarity(unseen),
        "per_class_f1": {k: v["f1-score"] for k, v in unseen["per_class"].items()},
    }
    if registers:
        row["english_macro_f1"] = registers["english"]["macro_f1"]
        row["code_mixed_macro_f1"] = registers["code-mixed"]["macro_f1"]
        row["register_delta"] = (
            registers["english"]["macro_f1"] - registers["code-mixed"]["macro_f1"]
        )
    return row


def collect(module_dir: Path = MODULE_DIR) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for name, filename in HOLDOUT_REPORTS.items():
        path = module_dir / filename
        if not path.exists():
            logger.warning("missing %s — skipping %s", filename, name)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))["sentiment"]
        registers = (d.get("register_breakdown") or {}).get("held_out")
        rows.append(_row(
            name,
            "synthetic" if "baseline" in name else "hybrid",
            d["in_distribution_metrics"],
            d["held_out_metrics"],
            registers if registers and "english" in registers else None,
            d.get("base_model"),
        ))

    if ZEROSHOT_REPORT.exists():
        z = json.loads(ZEROSHOT_REPORT.read_text(encoding="utf-8"))
        for key, spec in z.items():
            if not isinstance(spec, dict) or "results" not in spec:
                continue
            for mapping_name, results in spec["results"].items():
                unseen = results["unseen_idea_test"]
                registers = unseen.get("by_register")
                rows.append(_row(
                    f"zero-shot: {key} ({mapping_name} fold)",
                    "zero-shot",
                    results["seen_idea_test"],
                    unseen,
                    registers if registers and "english" in registers else None,
                    spec["model_id"],
                ))
    return rows


def log_table(rows: List[Dict[str, Any]]) -> None:
    header = (f"{'candidate':44s} {'seen':>6s} {'unseen':>7s} {'gap':>7s} "
              f"{'EnthR':>6s} {'EnthP':>6s} {'E>F':>6s} {'F>E':>6s} {'eng':>6s} {'code':>6s}")
    logger.info("")
    logger.info("%s", header)
    logger.info("%s", "-" * len(header))
    for row in sorted(rows, key=lambda r: -r["unseen_idea_macro_f1"]):
        logger.info(
            "%-44s %6.3f %7.3f %7.3f %6.3f %6.3f %5.0f%% %5.0f%% %6s %6s",
            row["candidate"][:44],
            row["seen_idea_macro_f1"] or float("nan"),
            row["unseen_idea_macro_f1"],
            row["gap"] if row["gap"] is not None else float("nan"),
            row["enthusiastic_recall"],
            row["enthusiastic_precision"],
            100 * (row["polarity"]["enth_to_frus_rate"] or 0),
            100 * (row["polarity"]["frus_to_enth_rate"] or 0),
            f"{row['english_macro_f1']:.3f}" if "english_macro_f1" in row else "-",
            f"{row['code_mixed_macro_f1']:.3f}" if "code_mixed_macro_f1" in row else "-",
        )


def verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Does anything beat the baseline where the baseline is weak?"""
    baseline = next((r for r in rows if r["kind"] == "synthetic"), None)
    if baseline is None:
        return {"error": "baseline report not found"}
    challengers = [r for r in rows if r is not baseline]
    better = [
        r for r in challengers
        if r["unseen_idea_macro_f1"] > baseline["unseen_idea_macro_f1"]
    ]
    best = max(challengers, key=lambda r: r["unseen_idea_macro_f1"]) if challengers else None
    return {
        "baseline_unseen_idea_macro_f1": baseline["unseen_idea_macro_f1"],
        "baseline_enthusiastic_recall": baseline["enthusiastic_recall"],
        "candidates_beating_baseline_on_unseen_ideas": [
            {
                "candidate": r["candidate"],
                "unseen_idea_macro_f1": r["unseen_idea_macro_f1"],
                "delta": r["unseen_idea_macro_f1"] - baseline["unseen_idea_macro_f1"],
                "enthusiastic_recall_delta": (
                    r["enthusiastic_recall"] - baseline["enthusiastic_recall"]
                ),
            }
            for r in sorted(better, key=lambda r: -r["unseen_idea_macro_f1"])
        ],
        "best_challenger": best["candidate"] if best else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    rows = collect()
    log_table(rows)
    result = verdict(rows)
    logger.info("")
    logger.info("%s", json.dumps(result, indent=2, default=float))
    args.out.write_text(
        json.dumps({"rows": rows, "verdict": result}, indent=2, default=float),
        encoding="utf-8",
    )
    logger.info("")
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
