"""Candidate 2 — off-the-shelf emotion models, zero-shot, mapped to our 4 tones.

The v4 synthetic head reaches 0.840 macro-F1 on emotional ideas it saw in
training and 0.407 on ideas it never saw. That gap is the open question: the
model has learned our *situations*, not tone in general. A model pretrained on
real-world emotional text has, in principle, seen far more ways a human sounds
enthusiastic or fed up, so it is the obvious thing to test against.

Nothing here is fine-tuned. Each model runs as shipped and its own label space
is folded into ours. That means the comparison hinges on the fold, so two are
reported:

*a-priori* — a hand-written mapping, argued below per model. This is the honest
"could I have deployed this today with no data at all" number.

*fitted* — the same frozen model, but each source label is assigned to whichever
of our tones it co-occurs with most on the **training** split. No gradients, no
weight changes; only the 560 training transcripts the other two candidates also
see. Without this the comparison is unfair in v4's favour, because a bad guess
at the mapping would be scored as a weakness of the pretrained model rather than
of my guess.

Evaluation reuses the exact partition of :mod:`phrase_holdout`, so all three
candidates are scored on identical rows: 120 seen-idea test transcripts and 416
whose emotional idea was withheld from training entirely.

Run:
    python -m sentiment_intent.pretrained_baseline
    python -m sentiment_intent.pretrained_baseline --models emotion-distilroberta
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import (  # noqa: E402
    SEED,
    SENTIMENT_CLASSES,
    classification_metrics,
    load_corpus,
    log_metrics_table,
    resolve_corpus,
)

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parent / "pretrained_zeroshot_report.json"

# --------------------------------------------------------------------------- #
# The candidates and their a-priori folds.
#
# Our four tones are not a sentiment axis. Hesitant and Frustrated are both
# "negative" in valence and mean opposite things to a sales team, which is why
# the scheme exists at all (CLAUDE.md, Phase 2). So an *emotion* model with a
# fine-grained label space folds far more naturally than a pos/neg/neutral
# sentiment model, and the registry reflects that: the two sentiment models are
# included mainly to measure how much that distinction costs.
# --------------------------------------------------------------------------- #

PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    "emotion-distilroberta": {
        "model_id": "j-hartmann/emotion-english-distilroberta-base",
        "language": "english",
        "note": "7-way Ekman emotions, DistilRoBERTa. Closest in size to our baseline.",
        "mapping": {
            "joy": "Enthusiastic",
            "surprise": "Enthusiastic",
            "neutral": "Neutral",
            "fear": "Hesitant",
            "sadness": "Hesitant",
            "anger": "Frustrated",
            "disgust": "Frustrated",
        },
    },
    "goemotions-roberta": {
        "model_id": "SamLowe/roberta-base-go_emotions",
        "language": "english",
        "note": "28-way GoEmotions, RoBERTa-base. The finest-grained fold available.",
        "mapping": {
            # Forward-leaning, wants to proceed.
            "excitement": "Enthusiastic",
            "joy": "Enthusiastic",
            "admiration": "Enthusiastic",
            "optimism": "Enthusiastic",
            "desire": "Enthusiastic",
            "love": "Enthusiastic",
            "amusement": "Enthusiastic",
            "gratitude": "Enthusiastic",
            "pride": "Enthusiastic",
            "relief": "Enthusiastic",
            "approval": "Enthusiastic",
            "caring": "Enthusiastic",
            # Affect-free reporting.
            "neutral": "Neutral",
            "realization": "Neutral",
            # Uncommitted, uncertain, holding back — a live lead, not a lost one.
            "confusion": "Hesitant",
            "curiosity": "Hesitant",
            "nervousness": "Hesitant",
            "fear": "Hesitant",
            "embarrassment": "Hesitant",
            "surprise": "Hesitant",
            "sadness": "Hesitant",
            "grief": "Hesitant",
            "remorse": "Hesitant",
            # Aggrieved by us — an escalation signal.
            "anger": "Frustrated",
            "annoyance": "Frustrated",
            "disappointment": "Frustrated",
            "disapproval": "Frustrated",
            "disgust": "Frustrated",
        },
    },
    "xlmr-sentiment": {
        "model_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "language": "multilingual",
        "note": "3-way sentiment, XLM-R. Multilingual, but cannot express the "
                "Hesitant/Frustrated split that the whole tone scheme exists for.",
        "mapping": {
            "positive": "Enthusiastic",
            "neutral": "Neutral",
            "negative": "Frustrated",
        },
    },
    "xlm-emo": {
        "model_id": "MilaNLProc/xlm-emo-t",
        "language": "multilingual",
        "note": "4-way multilingual emotion, XLM-R. The multilingual option that "
                "still has an emotion label space.",
        "mapping": {
            "joy": "Enthusiastic",
            "fear": "Hesitant",
            "sadness": "Hesitant",
            "anger": "Frustrated",
        },
    },
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def source_probabilities(
    model_id: str,
    records: Sequence[Dict[str, Any]],
    text_field: str = "customer_text",
    batch_size: int = 16,
    max_length: int = 512,
) -> Tuple[torch.Tensor, List[str]]:
    """Runs the frozen model and returns (n_records x n_source_labels) probabilities.

    Softmax is used for every model, including the multi-label GoEmotions head.
    The fold sums probability mass across the labels assigned to each tone and
    takes the largest, so what matters is the relative ordering within a
    record — which softmax preserves — not calibration against a 0.5 threshold.
    """
    from transformers import AutoTokenizer

    from sentiment_intent.domain_adapt import load_classifier  # noqa: PLC0415

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_classifier(model_id).to(device)
    model.eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    # Some of these checkpoints cap at 512 positions; our customer turns are far
    # shorter, but clamp anyway so a long messy transcript cannot overflow.
    limit = min(max_length, getattr(model.config, "max_position_embeddings", max_length) - 2)

    out: List[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        chunk = [r[text_field] for r in records[start:start + batch_size]]
        encoded = tokenizer(
            chunk, padding=True, truncation=True, max_length=limit, return_tensors="pt"
        ).to(device)
        logits = model(**encoded).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0, len(labels)), labels


def fold(
    probabilities: torch.Tensor,
    source_labels: Sequence[str],
    mapping: Dict[str, str],
) -> List[str]:
    """Collapses source-label probabilities onto our four tones."""
    columns: Dict[str, List[int]] = {tone: [] for tone in SENTIMENT_CLASSES}
    for index, label in enumerate(source_labels):
        tone = mapping.get(label)
        if tone is not None:
            columns[tone].append(index)

    scores = torch.zeros(probabilities.shape[0], len(SENTIMENT_CLASSES))
    for position, tone in enumerate(SENTIMENT_CLASSES):
        if columns[tone]:
            scores[:, position] = probabilities[:, columns[tone]].sum(dim=1)
        else:
            # A tone no source label maps to can never be predicted. Left at
            # zero rather than dropped, so the confusion matrix shows the
            # blind spot instead of hiding it.
            scores[:, position] = -1.0
    return [SENTIMENT_CLASSES[int(i)] for i in scores.argmax(dim=1)]


def fit_mapping(
    probabilities: torch.Tensor,
    source_labels: Sequence[str],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    """Assigns each source label to the tone it co-occurs with most on train.

    Soft rather than hard: a label's evidence for a tone is the total
    probability mass it puts on transcripts carrying that tone, normalised by
    how many such transcripts there are. Normalising matters because our classes
    are balanced but the source model's priors are not — "neutral" dominates
    most emotion heads and would otherwise win every tone.
    """
    tone_index = {tone: i for i, tone in enumerate(SENTIMENT_CLASSES)}
    totals = torch.zeros(len(source_labels), len(SENTIMENT_CLASSES))
    counts = torch.zeros(len(SENTIMENT_CLASSES))
    for row, record in enumerate(records):
        column = tone_index[record["sentiment"]]
        totals[:, column] += probabilities[row]
        counts[column] += 1
    totals /= counts.clamp(min=1)
    return {
        label: SENTIMENT_CLASSES[int(totals[i].argmax())]
        for i, label in enumerate(source_labels)
    }


def _subset_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    records: Sequence[Dict[str, Any]],
    field: str,
) -> Dict[str, Any]:
    """Metrics per value of a record field (style, emotion_register)."""
    index = {tone: i for i, tone in enumerate(SENTIMENT_CLASSES)}
    out: Dict[str, Any] = {}
    for value in sorted({r.get(field) for r in records if r.get(field)}):
        rows = [i for i, r in enumerate(records) if r.get(field) == value]
        if not rows:
            continue
        out[value] = {
            "n": len(rows),
            **classification_metrics(
                [index[references[i]] for i in rows],
                [index[predictions[i]] for i in rows],
                SENTIMENT_CLASSES,
            ),
        }
    return out


def evaluate_fold(
    probabilities: torch.Tensor,
    source_labels: Sequence[str],
    mapping: Dict[str, str],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    index = {tone: i for i, tone in enumerate(SENTIMENT_CLASSES)}
    predictions = fold(probabilities, source_labels, mapping)
    references = [r["sentiment"] for r in records]
    metrics = classification_metrics(
        [index[t] for t in references], [index[p] for p in predictions], SENTIMENT_CLASSES
    )
    metrics["n"] = len(records)
    metrics["reachable_tones"] = sorted(set(mapping.values()))
    metrics["by_style"] = _subset_metrics(references, predictions, records, "style")
    metrics["by_register"] = _subset_metrics(references, predictions, records, "emotion_register")
    return metrics


def evaluate_model(
    key: str,
    splits: Dict[str, List[Dict[str, Any]]],
    text_field: str = "customer_text",
) -> Dict[str, Any]:
    spec = PRETRAINED_MODELS[key]
    logger.info("")
    logger.info("=" * 72)
    logger.info("ZERO-SHOT — %s (%s)", key, spec["model_id"])
    logger.info("%s", spec["note"])

    cached: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
    for name, records in splits.items():
        logger.info("  scoring %-16s n=%d", name, len(records))
        cached[name] = source_probabilities(spec["model_id"], records, text_field)

    source_labels = cached["train"][1]
    fitted = fit_mapping(cached["train"][0], source_labels, splits["train"])
    logger.info("  fitted mapping: %s", json.dumps(fitted))

    result: Dict[str, Any] = {
        "model_id": spec["model_id"],
        "language": spec["language"],
        "note": spec["note"],
        "source_labels": source_labels,
        "mappings": {"apriori": spec["mapping"], "fitted": fitted},
        "results": {},
    }
    for mapping_name, mapping in (("apriori", spec["mapping"]), ("fitted", fitted)):
        result["results"][mapping_name] = {}
        for name, records in splits.items():
            if name == "train":
                continue
            metrics = evaluate_fold(cached[name][0], source_labels, mapping, records)
            result["results"][mapping_name][name] = metrics
            log_metrics_table(f"{key} / {mapping_name} / {name}", metrics)
    return result


def evaluation_splits(corpus: str = "v4", seed: int = SEED) -> Dict[str, List[Dict[str, Any]]]:
    """The exact rows the phrase-holdout diagnostic trains and scores on.

    Rebuilt rather than cached so the three candidates cannot drift apart: the
    partition is a deterministic function of the corpus and the seed.
    """
    from sentiment_intent.phrase_holdout import (  # noqa: PLC0415
        held_out_phrases,
        partition,
        stratified_split,
    )

    corpus_path, pools, _ = resolve_corpus(corpus)
    records = load_corpus(corpus_path)
    held = held_out_phrases("sentiment", seed, pools)
    trainable, heldout, dropped = partition(records, "sentiment", held, pools)
    splits = stratified_split(trainable, "sentiment", seed)
    logger.info(
        "Partition: %d train / %d seen-idea test / %d unseen-idea / %d dropped",
        len(splits["train"]), len(splits["test"]), len(heldout), len(dropped),
    )
    return {
        "train": splits["train"],
        "seen_idea_test": splits["test"],
        "unseen_idea_test": heldout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="v4")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(PRETRAINED_MODELS),
        choices=list(PRETRAINED_MODELS),
    )
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    splits = evaluation_splits(args.corpus, args.seed)
    report: Dict[str, Any] = {
        "corpus": args.corpus,
        "seed": args.seed,
        "partition": {name: len(rows) for name, rows in splits.items()},
    }
    for key in args.models:
        report[key] = evaluate_model(key, splits)

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
