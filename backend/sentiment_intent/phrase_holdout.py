"""Diagnostic — does the model generalise beyond the generator's templates?

Phase 2's headline scores (~0.92 macro-F1 on both heads) are suspect because
the synthetic transcripts are assembled from a small pool of fixed per-class
phrases: a plain substring matcher over those pools scores *higher* than
DistilBERT. That leaves an open question the standard test split cannot answer
— did the model learn sentiment and intent, or did it memorise 90 phrases?

This holds a slice of each class's phrase pool out of training entirely and
evaluates on transcripts that use only the unseen phrasing:

    in-distribution test  — phrasings the model saw during training
    held-out-phrase test  — phrasings the model has never seen

A model that understood the language should score similarly on both. A model
that memorised templates collapses on the second.

The matcher is re-run restricted to the phrases the *model* could see, so the
comparison is like with like — a matcher that still knew the held-out phrases
would be answering an easier question than the model was asked.

Holdout sizes differ by task because the pools are shaped differently. Each
intent transcript carries exactly one of three intent phrases, so holding one
out per class yields uncontaminated transcripts directly. Sentiment openers and
deflections co-occur in the same transcript, so holding out a single phrase
leaves training-visible cues beside the unseen one; a proportional slice
(3 of 10 openers, 3 of 8 deflections) is held out instead. Either way a
transcript reaches the held-out set only if it contains no training-visible
phrase for its class — mixed transcripts are dropped from both sides.

Run (~25 min per head on CPU):
    python -m sentiment_intent.phrase_holdout
    python -m sentiment_intent.phrase_holdout --task intent
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import (  # noqa: E402
    INTENT_CLASSES,
    MODELS_STORE,
    SEED,
    SENTIMENT_CLASSES,
    TextClassificationDataset,
    _normalize,
    classification_metrics,
    collate,
    load_corpus,
    log_metrics_table,
    majority_baseline,
    template_baseline,
)
from sentiment_intent.domain_adapt import (  # noqa: E402
    WARMSTART_DIR,
    Stage2Config,
    evaluate,
    fine_tune,
)

logger = logging.getLogger(__name__)

HOLDOUT_DIR = MODELS_STORE / "holdout"
REPORT_PATH = Path(__file__).resolve().parent / "phrase_holdout_report.json"

# Fraction of each class's *content fillers* withheld from training.
#
# The unit of holdout is the filler, not the phrase. Since the rewrite, every
# phrase is a shared frame filled with a content word, and one filler yields
# several phrases ("we're looking to buy", "we want to buy", ...). Withholding
# a single phrase would leave its content word plainly visible in a sibling
# phrase, so the model would "generalise" to it trivially and the diagnostic
# would report a pass it had not earned. Withholding the whole filler means the
# held-out transcripts use content the model has genuinely never seen.
HOLD_FRACTION = 1 / 3

TASKS = {"sentiment": SENTIMENT_CLASSES, "intent": INTENT_CLASSES}


def _sample_groups(
    groups: Dict[str, List[str]], fraction: float, rng: random.Random
) -> List[str]:
    """Withholds a fraction of the filler keys, returning all their phrases."""
    keys = sorted(groups)
    count = max(1, round(len(keys) * fraction))
    return [phrase for key in rng.sample(keys, count) for phrase in groups[key]]


def held_out_phrases(task: str, seed: int = SEED) -> Dict[str, List[str]]:
    """Picks the phrases withheld from training, deterministically."""
    from data_generation.phrase_pools import (  # noqa: PLC0415
        DEFLECTION_GROUPS,
        INTENT_PHRASE_GROUPS,
        SENTIMENT_OPENER_GROUPS,
    )

    rng = random.Random(seed)
    held: Dict[str, List[str]] = {}
    if task == "intent":
        for label in INTENT_CLASSES:
            held[label] = _sample_groups(INTENT_PHRASE_GROUPS[label], HOLD_FRACTION, rng)
    else:
        for label in SENTIMENT_CLASSES:
            held[label] = [
                *_sample_groups(SENTIMENT_OPENER_GROUPS[label], HOLD_FRACTION, rng),
                *_sample_groups(DEFLECTION_GROUPS[label], HOLD_FRACTION, rng),
            ]
    return held


def full_pools(task: str) -> Dict[str, List[str]]:
    from data_generation.phrase_pools import (  # noqa: PLC0415
        DEFLECTIONS,
        INTENT_PHRASES,
        SENTIMENT_OPENERS,
    )

    if task == "intent":
        return {label: list(INTENT_PHRASES[label]) for label in INTENT_CLASSES}
    return {
        label: [*SENTIMENT_OPENERS[label], *DEFLECTIONS[label]]
        for label in SENTIMENT_CLASSES
    }


def partition(
    records: Sequence[Dict[str, Any]], task: str, held: Dict[str, List[str]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits the corpus into (train-eligible, held-out-phrase, dropped).

    A transcript is held-out only if it contains a withheld phrase for its own
    class and no training-visible one. Transcripts carrying both are dropped:
    they cannot be trained on without leaking the held-out phrasing, and they
    cannot be tested on without the model recognising the visible half.
    """
    pools = full_pools(task)
    held_norm = {k: [_normalize(p).strip() for p in v] for k, v in held.items()}
    seen_norm = {
        k: [_normalize(p).strip() for p in pools[k] if p not in set(held[k])] for k in pools
    }

    trainable: List[Dict[str, Any]] = []
    heldout: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for record in records:
        haystack = _normalize(record["text"])
        label = record[task]
        hits_held = any(p and p in haystack for p in held_norm[label])
        hits_seen = any(p and p in haystack for p in seen_norm[label])
        if hits_held and not hits_seen:
            heldout.append(record)
        elif hits_held:
            dropped.append(record)
        else:
            # Includes transcripts with no recognisable phrase at all — they
            # carry no template cue either way, so they are safe to train on.
            trainable.append(record)
    return trainable, heldout, dropped


def _has_negated_phrase(text: str) -> bool:
    """True when the transcript states its intent by contrast ("X, not Y")."""
    lowered = text.lower()
    return ", not " in lowered or "don't want to" in lowered


def stratified_split(
    records: Sequence[Dict[str, Any]], task: str, seed: int = SEED
) -> Dict[str, List[Dict[str, Any]]]:
    """70/15/15 within the train-eligible pool, stratified on the task label."""
    buckets: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        buckets[record[task]].append(record)

    rng = random.Random(seed)
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label in sorted(buckets):
        rows = sorted(buckets[label], key=lambda r: r["transcript_id"])
        rng.shuffle(rows)
        n_train = round(len(rows) * 0.70)
        n_val = round(len(rows) * 0.15)
        splits["train"].extend(rows[:n_train])
        splits["validation"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:])
    return splits


@torch.no_grad()
def evaluate_saved_model(
    model_dir: Path,
    records: Sequence[Dict[str, Any]],
    task: str,
    classes: Sequence[str],
    max_length: int,
) -> Dict[str, Any]:
    """Scores a saved artifact over an arbitrary record list."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    loader = DataLoader(
        TextClassificationDataset(records, tokenizer, task, classes, max_length=max_length),
        batch_size=32,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )
    metrics, _ = evaluate(model, loader, classes, device)
    return metrics


def run_task(task: str, epochs: int, seed: int = SEED) -> Dict[str, Any]:
    classes = list(TASKS[task])
    records = load_corpus()
    held = held_out_phrases(task, seed)
    trainable, heldout, dropped = partition(records, task, held)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHRASE HOLDOUT — %s", task)
    logger.info("=" * 70)
    for label in classes:
        logger.info("  held out from %-18s %s", label, held[label])
    logger.info(
        "Partition: %d train-eligible / %d held-out-phrase / %d dropped (mixed)",
        len(trainable), len(heldout), len(dropped),
    )
    logger.info("Held-out class balance: %s",
                dict(collections.Counter(r[task] for r in heldout)))
    if len(heldout) < len(classes) * 5:
        logger.warning("Held-out set is very small — treat its metrics as indicative only.")

    splits = stratified_split(trainable, task, seed)
    config = Stage2Config(
        task=task,
        classes=tuple(classes),
        epochs=epochs,
        seed=seed,
        warmstart_dir=str(WARMSTART_DIR),
    )
    model_dir = fine_tune(
        config,
        output_dir=HOLDOUT_DIR / task,
        splits=splits,
        update_pointer=False,  # never let a handicapped model become the served one
    )

    in_dist = evaluate_saved_model(model_dir, splits["test"], task, classes, config.max_length)
    out_dist = evaluate_saved_model(model_dir, heldout, task, classes, config.max_length)
    log_metrics_table(f"IN-DISTRIBUTION test — {task} (seen phrasings)", in_dist)
    log_metrics_table(f"HELD-OUT-PHRASE test — {task} (unseen phrasings)", out_dist)

    # The matcher is limited to what the model could see, so both are answering
    # the same question.
    baselines = {
        "in_distribution": {
            "majority": majority_baseline(splits["train"], splits["test"], task, classes),
            "template_visible_only": template_baseline(
                splits["test"], task, classes, exclude=held
            ),
        },
        "held_out": {
            "majority": majority_baseline(splits["train"], heldout, task, classes),
            "template_visible_only": template_baseline(heldout, task, classes, exclude=held),
            "template_full_pool": template_baseline(heldout, task, classes),
        },
    }

    # Requirement (2): the Buy/Rent inversion showed the model had learned no
    # negation at all. Which fillers land in the holdout varies with the seed,
    # so negated phrasings are scored explicitly rather than left to chance.
    negation = {}
    for name, pool in (("in_distribution", splits["test"]), ("held_out", heldout)):
        rows = [r for r in pool if _has_negated_phrase(r["text"])]
        negation[name] = {"n": len(rows)}
        if rows:
            negation[name].update(
                evaluate_saved_model(model_dir, rows, task, classes, config.max_length)
            )
    logger.info("")
    logger.info("NEGATED-PHRASING subset — in-dist n=%d acc %s | held-out n=%d acc %s",
                negation["in_distribution"]["n"],
                f"{negation['in_distribution'].get('accuracy', float('nan')):.4f}",
                negation["held_out"]["n"],
                f"{negation['held_out'].get('accuracy', float('nan')):.4f}")

    drop = in_dist["macro_f1"] - out_dist["macro_f1"]
    logger.info("")
    logger.info("GENERALISATION GAP (%s): in-distribution %.4f -> held-out %.4f  (drop %.4f)",
                task, in_dist["macro_f1"], out_dist["macro_f1"], drop)
    logger.info(
        "  matcher on held-out, restricted to training-visible phrases: %.4f",
        baselines["held_out"]["template_visible_only"]["macro_f1"],
    )

    return {
        "task": task,
        "held_out_phrases": held,
        "partition": {
            "train_eligible": len(trainable),
            "held_out": len(heldout),
            "dropped_mixed": len(dropped),
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test_in_distribution": len(splits["test"]),
            "held_out_class_counts": dict(collections.Counter(r[task] for r in heldout)),
        },
        "model_dir": str(model_dir),
        "in_distribution_metrics": in_dist,
        "held_out_metrics": out_dist,
        "generalisation_gap_macro_f1": drop,
        "negated_phrasing_metrics": negation,
        "baselines": baselines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--epochs", type=int, default=Stage2Config.epochs)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    tasks = list(TASKS) if args.task == "both" else [args.task]
    report = {task: run_task(task, args.epochs, args.seed) for task in tasks}

    existing: Dict[str, Any] = {}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    existing.update(report)
    args.out.write_text(json.dumps(existing, indent=2, default=float), encoding="utf-8")
    logger.info("")
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
