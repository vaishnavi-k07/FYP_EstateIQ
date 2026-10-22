"""Scores the promoted intent head on the 60-case balanced, inferred-intent set.

Complements ``run_unseen50``: that set was written for extraction and
sentiment and cannot measure intent (41 of 50 are Buy, no Schedule Visit).
This one is balanced across all six intents and no caller names an intent,
so it measures exactly what the v6 head could not do.

Before scoring, two guards run:

* ``stated_intent_leaks`` must be empty — no test case may name its intent.
* The longest contiguous word run each case shares with the intent training
  corpus is measured and reported, so overlap is a number, not a claim.

Metrics come from ``run_unseen50.classification_metrics`` so both reports
compute accuracy, per-class P/R/F1 and macro averages identically.

Run:
    python -m eval.run_intent_unseen
    python -m eval.run_intent_unseen --out ../docs/intent_unseen_report.md --json-out ../docs/intent_unseen_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
for path in (str(BACKEND_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval.intent_unseen_cases import (  # noqa: E402
    CASES,
    INTENT_CLASSES,
    by_intent,
    stated_intent_leaks,
)
from eval.run_unseen50 import (  # noqa: E402
    _longest_shared_run,
    _ngrams,
    _one_line,
    _pct,
    classification_metrics,
    load_intent,
)

logger = logging.getLogger(__name__)


def training_overlap(corpus: Path, max_n: int = 12) -> Dict[str, int]:
    from sentiment_intent.data import customer_turns  # noqa: PLC0415

    texts = [
        customer_turns(json.loads(line)["text"])
        for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    grams = _ngrams(texts, max_n)
    return {
        case["id"]: _longest_shared_run(customer_turns(case["transcript"]), grams, max_n)
        for case in CASES
    }


def run(head: Any) -> List[Dict[str, Any]]:
    from sentiment_intent.data import customer_turns  # noqa: PLC0415

    rows = []
    for case in CASES:
        text = customer_turns(case["transcript"])
        result = head.predict(text)
        rows.append({
            "id": case["id"],
            "register": case["register"],
            "style": case["style"],
            "transcript": case["transcript"],
            "expected": case["intent"],
            "predicted": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "correct": result["label"] == case["intent"],
        })
    return rows


def render(result: Dict[str, Any]) -> str:
    m = result["metrics"]
    rows = result["rows"]
    overlap = result["overlap"]
    out: List[str] = []
    w = out.append

    w("# EstateIQ — intent on 60 unseen, inferred-intent transcripts\n")
    w(f"Model: `{result['info']['artifact']}`  ")
    w(f"Training corpus checked for overlap: `{result['corpus']}`\n")
    w("## 1. The test set\n")
    w("60 transcripts written for this evaluation before the training data, "
      + ", ".join(f"{k} {v}" for k, v in by_intent().items())
      + ". 7 English and 3 code-mixed per intent. No caller names an intent "
      "(buy / rent / invest / visit / call back / enquiry and their Hindi "
      f"equivalents are banned and checked: **{len(stated_intent_leaks())} leaks**).\n")
    runs = sorted(overlap.values())
    w(f"Longest word run any case shares with the training corpus: "
      f"**{max(runs)} words** (median {runs[len(runs) // 2]}).\n")

    w("## 2. Headline\n")
    w(f"- Accuracy: **{_pct(m['accuracy'])}** ({m['correct']}/{m['total']}) — chance is 16.7%")
    w(f"- Macro precision / recall / F1: {m['macro_precision']:.3f} / "
      f"{m['macro_recall']:.3f} / **{m['macro_f1']:.3f}**\n")

    w("## 3. Per-class precision / recall / F1\n")
    w("| Class | Support | TP | FP | FN | Precision | Recall | F1 |")
    w("|---|---|---|---|---|---|---|---|")
    for label, s in m["per_class"].items():
        w(f"| {label} | {s['support']} | {s['tp']} | {s['fp']} | {s['fn']} | "
          f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |")
    w("")

    w("## 4. Confusion matrix (rows = truth, columns = predicted)\n")
    w("| truth \\ pred | " + " | ".join(INTENT_CLASSES) + " |")
    w("|---" * (len(INTENT_CLASSES) + 1) + "|")
    for gold in INTENT_CLASSES:
        cells = [
            f"**{m['confusion'][gold][p]}**" if gold == p else str(m["confusion"][gold][p])
            for p in INTENT_CLASSES
        ]
        w(f"| **{gold}** | " + " | ".join(cells) + " |")
    w("")

    w("## 5. By subset\n")
    w("| Subset | Cases | Accuracy | Macro-F1 |")
    w("|---|---|---|---|")
    for group in ("by_register", "by_style"):
        for name, s in m[group].items():
            if s.get("cases"):
                w(f"| {name} | {s['cases']} | {_pct(s['accuracy'])} | {s['macro_f1']:.3f} |")
    w("")

    w("## 6. Every miss, in full\n")
    wrong = [r for r in rows if not r["correct"]]
    if not wrong:
        w("None.\n")
    for r in wrong:
        w(f"**{r['id']}** ({r['register']}, {r['style']}) — expected **{r['expected']}**, "
          f"predicted **{r['predicted']}** at {r['confidence']:.2f}")
        w(f"> {_one_line(r['transcript'])}")
        w("- full distribution: "
          + ", ".join(f"{k} {v:.2f}" for k, v in r["probabilities"].items()))
        w("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="intent_v1",
                        help="Training corpus to measure overlap against.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    leaks = stated_intent_leaks()
    if leaks:
        raise SystemExit(f"Test cases name their intent, refusing to score: {leaks}")

    from sentiment_intent.data import resolve_corpus  # noqa: PLC0415

    corpus_path = resolve_corpus(args.corpus)[0]
    overlap = training_overlap(corpus_path)
    logger.info("Longest shared word run with %s: %d", corpus_path.name, max(overlap.values()))

    head, info = load_intent()
    if head is None:
        raise SystemExit(info["reason"])
    logger.info("Intent: %s", info["artifact"])
    rows = run(head)
    metrics = classification_metrics(rows, INTENT_CLASSES)
    metrics.pop("_subset")
    logger.info("accuracy %.1f%% | macro-F1 %.3f", metrics["accuracy"] * 100, metrics["macro_f1"])

    result = {"info": info, "corpus": str(corpus_path), "overlap": overlap,
              "rows": rows, "metrics": metrics}
    report = render(result)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.out)
    else:
        print(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                 encoding="utf-8")
        logger.info("Raw results written to %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
