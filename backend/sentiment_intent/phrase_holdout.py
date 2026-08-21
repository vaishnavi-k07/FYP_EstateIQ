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
intent transcript carries exactly one intent phrase, built from one content
filler, so withholding fillers yields uncontaminated transcripts directly.
Sentiment openers and deflections co-occur in the same transcript, so holding
out a single phrase leaves training-visible cues beside the unseen one; a
proportional slice (7 of 20 tone clauses, 4 of 12 deflections) is held out
instead. Either way a transcript reaches the held-out set only if it contains
no training-visible phrase for its class — mixed transcripts are dropped from
both sides.

That drop is why the corpus is 1000 rather than 700: roughly a third of
transcripts mix held-out and visible phrasing and are unusable by either side,
so the diagnostic trains on about half the pool. At 700 that left 260
sentiment training transcripts against the production head's 700-odd, and the
diagnostic's absolute scores were depressed for a reason unrelated to the
confound it measures. Widening the phrase pools does not help here — a third
of a wider pool is still a third — so the pool itself was grown.

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
    DEFAULT_POOLS,
    EXCLUDE_TELEGRAPHIC,
    INTENT_CLASSES,
    MODELS_STORE,
    NAMED_CORPORA,
    SEED,
    SENTIMENT_CLASSES,
    TextClassificationDataset,
    _normalize,
    classification_metrics,
    collate,
    load_corpus,
    log_metrics_table,
    majority_baseline,
    pools_module,
    resolve_corpus,
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
    groups: Dict[str, List[str]],
    fraction: float,
    rng: random.Random,
    register_of: Any,
) -> List[str]:
    """Withholds a fraction of the filler keys, stratified by register.

    Sampling uniformly across a class's whole pool withholds the right *number*
    of phrasings but not the right *mix*. On the first v3 run it took all three
    Enthusiastic code-mixed clauses while leaving every other tone two of
    three, so at training time romanised-Hindi markers appeared only under
    rival labels and half the held-out Enthusiastic transcripts went to
    Frustrated. That measures how well the model transfers a register it never
    saw for one class — not whether it memorised templates — and it does so
    asymmetrically, which makes the per-class numbers incomparable.

    Sampling within each register keeps the fraction withheld identical while
    guaranteeing every class keeps some of each. The test stays as hard: a
    third of the code-mixed phrasings and a third of the English ones are
    still unseen.

    ``register_of`` comes from the pools module the corpus was built with, so
    the pilot's declared registers are used for the pilot and v3's sniffed ones
    for v3.
    """
    by_register: Dict[str, List[str]] = collections.defaultdict(list)
    for key in sorted(groups):
        by_register[register_of(key)].append(key)

    held: List[str] = []
    for register in sorted(by_register):
        keys = by_register[register]
        count = max(1, round(len(keys) * fraction))
        held.extend(rng.sample(keys, count))
    return [phrase for key in held for phrase in groups[key]]


def held_out_phrases(
    task: str, seed: int = SEED, pools: str = DEFAULT_POOLS
) -> Dict[str, List[str]]:
    """Picks the phrases withheld from training, deterministically."""
    module = pools_module(pools)
    register_of = module.phrase_register

    rng = random.Random(seed)
    held: Dict[str, List[str]] = {}
    if task == "intent":
        for label in INTENT_CLASSES:
            held[label] = _sample_groups(
                module.INTENT_PHRASE_GROUPS[label], HOLD_FRACTION, rng, register_of
            )
    else:
        for label in SENTIMENT_CLASSES:
            held[label] = [
                *_sample_groups(
                    module.SENTIMENT_OPENER_GROUPS[label], HOLD_FRACTION, rng, register_of
                ),
                *_sample_groups(
                    module.DEFLECTION_GROUPS[label], HOLD_FRACTION, rng, register_of
                ),
            ]
    return held


def full_pools(task: str, pools: str = DEFAULT_POOLS) -> Dict[str, List[str]]:
    module = pools_module(pools)
    if task == "intent":
        return {label: list(module.INTENT_PHRASES[label]) for label in INTENT_CLASSES}
    return {
        label: [*module.SENTIMENT_OPENERS[label], *module.DEFLECTIONS[label]]
        for label in SENTIMENT_CLASSES
    }


def partition(
    records: Sequence[Dict[str, Any]],
    task: str,
    held: Dict[str, List[str]],
    pools: str = DEFAULT_POOLS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits the corpus into (train-eligible, held-out-phrase, dropped).

    A transcript is held-out only if it contains a withheld phrase for its own
    class and no training-visible one. Transcripts carrying both are dropped:
    they cannot be trained on without leaking the held-out phrasing, and they
    cannot be tested on without the model recognising the visible half.
    """
    pool = full_pools(task, pools)
    held_norm = {k: [_normalize(p).strip() for p in v] for k, v in held.items()}
    seen_norm = {
        k: [_normalize(p).strip() for p in pool[k] if p not in set(held[k])] for k in pool
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


def subset_breakdown(
    model_dir: Path,
    pools_by_name: Dict[str, Sequence[Dict[str, Any]]],
    task: str,
    classes: Sequence[str],
    max_length: int,
    field: str = "style",
    values: Sequence[str] = ("clean", "messy"),
) -> Dict[str, Any]:
    """Scores clean and messy transcripts separately within each evaluation set.

    Only meaningful for a corpus that carries a ``style``; the v3 corpus has no
    clean/messy axis, so this returns a note instead of inventing one.

    The question it answers is narrow and worth stating: the pilot deals
    messiness out at exactly 50% within every tone, so messiness *cannot* be a
    cue for a label. What it can still do is make a transcript harder to read.
    A gap between the two here is a difficulty effect, not a leak — and a gap
    of zero would suggest the disfluencies are cosmetic enough that the model
    ignores them.
    """
    first, second = values
    out: Dict[str, Any] = {}
    for set_name, records in pools_by_name.items():
        if {r.get(field) for r in records} == {None}:
            out[set_name] = {"note": f"corpus has no {field} axis"}
            continue
        block: Dict[str, Any] = {}
        for value in values:
            rows = [r for r in records if r.get(field) == value]
            block[value] = {"n": len(rows)}
            if rows:
                block[value].update(
                    evaluate_saved_model(model_dir, rows, task, classes, max_length)
                )
        if block[first].get("n") and block[second].get("n"):
            block[f"macro_f1_delta_{first}_minus_{second}"] = (
                block[first]["macro_f1"] - block[second]["macro_f1"]
            )
            block[f"accuracy_delta_{first}_minus_{second}"] = (
                block[first]["accuracy"] - block[second]["accuracy"]
            )
        out[set_name] = block
    return out


def confusion_pairs(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Off-diagonal confusion cells as counts and as a share of the true class.

    Pulled out because the headline macro-F1 hides the failure that matters
    most to lead scoring: Enthusiastic and Frustrated trading places. A Hot lead
    filed as an escalation, or the reverse, is a worse business error than any
    Neutral/Hesitant slip, so the pair is reported explicitly rather than left
    to be read out of a matrix.
    """
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    cells: Dict[str, Any] = {}
    for i, true_label in enumerate(labels):
        support = sum(matrix[i])
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            cells[f"{true_label}->{pred_label}"] = {
                "n": matrix[i][j],
                "share_of_true_class": round(matrix[i][j] / support, 4) if support else None,
            }
    polarity = [
        cells.get("Enthusiastic->Frustrated", {}).get("n", 0),
        cells.get("Frustrated->Enthusiastic", {}).get("n", 0),
    ]
    total = sum(
        sum(matrix[i])
        for i, label in enumerate(labels)
        if label in ("Enthusiastic", "Frustrated")
    )
    return {
        "cells": cells,
        "polarity_inversion": {
            "enthusiastic_called_frustrated": polarity[0],
            "frustrated_called_enthusiastic": polarity[1],
            "n_enthusiastic_or_frustrated": total,
            "rate": round(sum(polarity) / total, 4) if total else None,
        },
    }


def cap_train(
    splits: Dict[str, List[Dict[str, Any]]], task: str, cap: int, seed: int = SEED
) -> Dict[str, List[Dict[str, Any]]]:
    """Shrinks the training split to ``cap`` rows, stratified on the label.

    Exists so two corpora can be compared at the same training-set size. The
    pilot yields 136 training transcripts against v3's 322, and a model trained
    on 136 rows will score lower whatever the data looks like — so a raw
    pilot-vs-v3 comparison confounds corpus design with corpus size. Capping v3
    to the pilot's size separates the two: what remains is a difference in what
    the transcripts *say*.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for record in splits["train"]:
        buckets[record[task]].append(record)

    rng = random.Random(seed)
    per_class = max(1, cap // max(len(buckets), 1))
    capped: List[Dict[str, Any]] = []
    for label in sorted(buckets):
        rows = sorted(buckets[label], key=lambda r: r["transcript_id"])
        rng.shuffle(rows)
        capped.extend(rows[:per_class])
    return {**splits, "train": capped}


def run_task(
    task: str,
    epochs: int,
    seed: int = SEED,
    corpus: str = "v3",
    train_cap: Optional[int] = None,
    base_model: Optional[str] = None,
    run_label: Optional[str] = None,
) -> Dict[str, Any]:
    classes = list(TASKS[task])
    corpus_path, pools, _ = resolve_corpus(corpus)
    records = load_corpus(corpus_path)
    held = held_out_phrases(task, seed, pools)
    trainable, heldout, dropped = partition(records, task, held, pools)

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
    if train_cap:
        splits = cap_train(splits, task, train_cap, seed)
        logger.info("Training set capped to %d transcripts for size-matched comparison.",
                    len(splits["train"]))
    # The SST-2 warm-start copies ``distilbert.*`` tensors, so it is meaningless
    # for a RoBERTa or XLM-R encoder — a pretrained emotion checkpoint brings its
    # own, better-adapted encoder instead, and its head is discarded exactly as
    # the warm-start's binary head was.
    hybrid = base_model is not None
    config = Stage2Config(
        task=task,
        classes=tuple(classes),
        epochs=epochs,
        seed=seed,
        base_model=base_model or Stage2Config.base_model,
        warmstart_dir=None if hybrid else str(WARMSTART_DIR),
        ignore_mismatched_sizes=hybrid,
        corpus_path=str(corpus_path),
        pools=pools,
    )
    if hybrid:
        logger.info("Hybrid run: encoder from %s, fresh 4-class head.", base_model)
    # One directory per (task, corpus) so a pilot run never overwrites the v3
    # holdout artifact the 0.235 reference gap was measured on.
    run_name = task if corpus == "v3" else f"{task}_{corpus}"
    if train_cap:
        run_name += f"_cap{train_cap}"
    if run_label:
        run_name += f"_{run_label}"
    model_dir = fine_tune(
        config,
        output_dir=HOLDOUT_DIR / run_name,
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
                splits["test"], task, classes, exclude=held, pools=pools
            ),
        },
        "held_out": {
            "majority": majority_baseline(splits["train"], heldout, task, classes),
            "template_visible_only": template_baseline(
                heldout, task, classes, exclude=held, pools=pools
            ),
            "template_full_pool": template_baseline(heldout, task, classes, pools=pools),
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

    # Like-for-like gap. Telegraphic transcripts carry no tone clause and no
    # intent phrase, so they never match a held-out phrase and land wholly on
    # the in-distribution side — 17% of that test set against 0% of the
    # held-out one. They scored at or below chance for both heads, which
    # dragged the in-distribution number down and flattered the gap.
    #
    # They are now excluded from the corpus outright (data.load_corpus), so
    # both sides are tone-bearing and this equals the headline gap. The block
    # is kept, and still computed rather than assumed, so a run with the
    # exclusion turned off is still measured honestly — and so the field the
    # earlier reports were read from does not disappear.
    tone_bearing = [r for r in splits["test"] if not r["is_telegraphic"]]
    comparable: Dict[str, Any] = {
        "telegraphic_excluded_from_corpus": EXCLUDE_TELEGRAPHIC,
        "in_distribution_n": len(splits["test"]),
        "in_distribution_telegraphic_n": len(splits["test"]) - len(tone_bearing),
        "held_out_telegraphic_n": sum(1 for r in heldout if r["is_telegraphic"]),
    }
    if tone_bearing:
        clean = evaluate_saved_model(
            model_dir, tone_bearing, task, classes, config.max_length
        )
        comparable["in_distribution_metrics_excluding_telegraphic"] = clean
        comparable["generalisation_gap_macro_f1"] = clean["macro_f1"] - out_dist["macro_f1"]
        log_metrics_table(
            f"IN-DISTRIBUTION test, tone-bearing only — {task}", clean
        )
        logger.info(
            "COMPARABLE GAP (%s): %.4f -> %.4f  (drop %.4f) over %d transcripts "
            "that actually state the label",
            task, clean["macro_f1"], out_dist["macro_f1"],
            comparable["generalisation_gap_macro_f1"], len(tone_bearing),
        )

    drop = in_dist["macro_f1"] - out_dist["macro_f1"]
    logger.info("")
    logger.info("GENERALISATION GAP (%s): in-distribution %.4f -> held-out %.4f  (drop %.4f)",
                task, in_dist["macro_f1"], out_dist["macro_f1"], drop)
    logger.info(
        "  matcher on held-out, restricted to training-visible phrases: %.4f",
        baselines["held_out"]["template_visible_only"]["macro_f1"],
    )

    evaluation_sets = {"in_distribution": splits["test"], "held_out": heldout}
    styles = subset_breakdown(
        model_dir, evaluation_sets, task, classes, config.max_length,
        field="style", values=("clean", "messy"),
    )
    # English vs code-mixed, so a mostly-English encoder's handling of the
    # Hinglish minority is visible rather than averaged away.
    registers = subset_breakdown(
        model_dir, evaluation_sets, task, classes, config.max_length,
        field="emotion_register", values=("english", "code-mixed"),
    )
    for set_name, block in registers.items():
        if "macro_f1_delta_english_minus_code-mixed" in block:
            logger.info(
                "REGISTER (%s, %s): english n=%d macro-F1 %.4f | code-mixed n=%d "
                "macro-F1 %.4f (english - code-mixed %.4f)",
                task, set_name,
                block["english"]["n"], block["english"]["macro_f1"],
                block["code-mixed"]["n"], block["code-mixed"]["macro_f1"],
                block["macro_f1_delta_english_minus_code-mixed"],
            )
    for set_name, block in styles.items():
        if "macro_f1_delta_clean_minus_messy" in block:
            logger.info(
                "STYLE (%s, %s): clean n=%d macro-F1 %.4f | messy n=%d macro-F1 %.4f "
                "(clean - messy %.4f)",
                task, set_name,
                block["clean"]["n"], block["clean"]["macro_f1"],
                block["messy"]["n"], block["messy"]["macro_f1"],
                block["macro_f1_delta_clean_minus_messy"],
            )

    confusion = {
        "in_distribution": confusion_pairs(in_dist),
        "held_out": confusion_pairs(out_dist),
    }
    if task == "sentiment":
        inversion = confusion["held_out"]["polarity_inversion"]
        logger.info(
            "POLARITY INVERSION on held-out phrasings: E->F %d, F->E %d of %d "
            "Enthusiastic/Frustrated transcripts (%.1f%%)",
            inversion["enthusiastic_called_frustrated"],
            inversion["frustrated_called_enthusiastic"],
            inversion["n_enthusiastic_or_frustrated"],
            100 * (inversion["rate"] or 0.0),
        )

    return {
        "task": task,
        "corpus": str(corpus_path),
        "pools": pools,
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
        "comparable_gap": comparable,
        "negated_phrasing_metrics": negation,
        "confusion_pairs": confusion,
        "style_breakdown": styles,
        "register_breakdown": registers,
        "base_model": config.base_model,
        "baselines": baselines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--epochs", type=int, default=Stage2Config.epochs)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--corpus",
        default="v3",
        help=f"Named corpus ({', '.join(sorted(NAMED_CORPORA))}) or a path. Non-default "
             "corpora write to their own report and their own holdout artifact "
             "directory, so the v3 reference numbers are never overwritten.",
    )
    parser.add_argument(
        "--train-cap",
        type=int,
        default=None,
        help="Subsample the training split to this many transcripts, stratified. Use to "
             "compare two corpora at equal training size.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Encoder to start from instead of DistilBERT + the SST-2 warm-start. "
             "Use a pretrained emotion checkpoint to run the hybrid candidate; its "
             "head is discarded and a fresh 4-class one is trained.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Suffix for the report and artifact directory, so hybrid runs do not "
             "overwrite the baseline's.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        suffix = "" if args.corpus == "v3" else f"_{args.corpus}"
        suffix += "" if not args.train_cap else f"_cap{args.train_cap}"
        suffix += "" if not args.label else f"_{args.label}"
        args.out = (
            REPORT_PATH if not suffix
            else REPORT_PATH.with_name(f"phrase_holdout_report{suffix}.json")
        )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    tasks = list(TASKS) if args.task == "both" else [args.task]
    report = {
        task: run_task(
            task, args.epochs, args.seed, args.corpus, args.train_cap,
            args.base_model, args.label,
        )
        for task in tasks
    }

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
