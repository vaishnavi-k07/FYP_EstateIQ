"""Shared corpus loading, splitting, encoding and scoring for Phase 2.

Both heads (4-class sentiment, 6-class intent) train on the same 1000 synthetic
transcripts and are evaluated the same way, so everything they agree on lives
here and the two ``train_*.py`` scripts stay thin.

Two deliberate choices are implemented here rather than in the trainers:

*Customer turns only.* The model is fed the customer's words, not the full
dialogue. The agent's lines in the synthetic data were written by our own
generator, while in production they will come from Vapi's system prompt — a
different distribution. Training on them would let the model key on phrasing
that never appears in a real call. See ``customer_turns``.

*One shared split.* A single (sentiment, intent)-stratified assignment is
saved to ``split.json`` and reused by both heads, so a transcript held out
from one model is held out from the other.

*No telegraphic transcripts.* See :func:`load_corpus` — their sentiment and
intent labels describe a caller their own text never characterises, so they
are unlearnable input paired with a confident label.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

SYNTHETIC_PATH = REPO_ROOT / "data_generation" / "output" / "synthetic_transcripts.jsonl"
# The Phase 2b pilot corpus: a separate few-hundred-transcript set built by
# data_generation/sentiment_pilot.py to test whether situational (rather than
# named) emotion closes the sentiment generalisation gap. Deliberately a
# different file from the one above, which the Phase 3 NER work reads.
SENTIMENT_PILOT_PATH = REPO_ROOT / "data_generation" / "output" / "sentiment_pilot.jsonl"
# The widened corpus: the same design over 38 emotional ideas per tone instead
# of 18, scaled to 1216 transcripts. The pilot showed the design removes the
# affect-word shortcut but that 288 transcripts over 18 ideas leave the model at
# chance, so breadth was increased before volume.
SENTIMENT_V4_PATH = REPO_ROOT / "data_generation" / "output" / "sentiment_v4.jsonl"
# v5 = v4 plus 96 Neutral transcripts carrying no emotional expression at all.
# The spot check found the v4-trained hybrid reading a bare requirement line as
# Enthusiastic at 0.886: nothing in v4 taught it that absence of affect is
# itself Neutral, because every v4 transcript carries an emotional idea.
SENTIMENT_V5_PATH = REPO_ROOT / "data_generation" / "output" / "sentiment_v5.jsonl"
# v6 = v5 with 6 more ideas per tone. The demo set found that past-tense
# recounting had become a Frustrated cue, because Enthusiastic carried few
# backstory-driven ideas: "second time buyers with you, the first went so well"
# inverted to Frustrated at 0.99. v6 adds backstory enthusiasm, and matching
# breadth to the other three tones so the holdout stays balanced.
SENTIMENT_V6_PATH = REPO_ROOT / "data_generation" / "output" / "sentiment_v6.jsonl"
# Intent-only corpus built by data_generation/intent_context.py. The v6-trained
# intent head memorised stock intent phrases (1.000 on its test split, 12% on
# the unseen-50 set), so intent now trains on transcripts where the intent is
# implied by situation and never named. Sentiment keeps training on v6.
INTENT_V1_PATH = REPO_ROOT / "data_generation" / "output" / "intent_v1.jsonl"
# v2 = v1 with Buy vs Investment separated by end use (own use vs returns). The
# v1 holdout read 60% of unseen Buy ideas as Investment, because v1's Buy money
# ideas never said who would live in the place.
INTENT_V2_PATH = REPO_ROOT / "data_generation" / "output" / "intent_v2.jsonl"
MODULE_DIR = Path(__file__).resolve().parent
SPLIT_PATH = MODULE_DIR / "split.json"
# Its own split file: build_split rewrites whatever path it is given whenever
# the corpus fingerprint changes, so pointing the pilot at split.json would
# destroy the assignment the promoted intent head was trained under.
PILOT_SPLIT_PATH = MODULE_DIR / "split_pilot.json"
V4_SPLIT_PATH = MODULE_DIR / "split_v4.json"
V5_SPLIT_PATH = MODULE_DIR / "split_v5.json"
V6_SPLIT_PATH = MODULE_DIR / "split_v6.json"
INTENT_V1_SPLIT_PATH = MODULE_DIR / "split_intent_v1.json"
INTENT_V2_SPLIT_PATH = MODULE_DIR / "split_intent_v2.json"
MODELS_STORE = BACKEND_DIR / "models_store" / "sentiment_intent"

# Which phrase pools a corpus was generated from. The template baseline and the
# phrase-holdout diagnostic both need the generator's own pools to do their job,
# and the pilot corpus was built from a different set, so the choice travels
# with the corpus rather than being hardcoded.
POOL_MODULES: Dict[str, str] = {
    "v3": "data_generation.phrase_pools",
    "pilot": "data_generation.pilot_emotion_pools",
    "intent": "data_generation.intent_context_pools",
}
DEFAULT_POOLS = "v3"

# Corpora shipped with the project, so callers can say --corpus pilot instead
# of spelling out a path, and get the matching pools and split file for free.
NAMED_CORPORA: Dict[str, Dict[str, Any]] = {
    "v3": {"path": SYNTHETIC_PATH, "pools": "v3", "split": SPLIT_PATH},
    "pilot": {"path": SENTIMENT_PILOT_PATH, "pools": "pilot", "split": PILOT_SPLIT_PATH},
    "v4": {"path": SENTIMENT_V4_PATH, "pools": "pilot", "split": V4_SPLIT_PATH},
    "v5": {"path": SENTIMENT_V5_PATH, "pools": "pilot", "split": V5_SPLIT_PATH},
    "v6": {"path": SENTIMENT_V6_PATH, "pools": "pilot", "split": V6_SPLIT_PATH},
    "intent_v1": {"path": INTENT_V1_PATH, "pools": "intent", "split": INTENT_V1_SPLIT_PATH},
    "intent_v2": {"path": INTENT_V2_PATH, "pools": "intent", "split": INTENT_V2_SPLIT_PATH},
}

# The agreed Phase 2 label spaces (see CLAUDE.md section 8, Phase 2). Order is
# fixed because it defines the model's output indices.
SENTIMENT_CLASSES: Tuple[str, ...] = ("Enthusiastic", "Neutral", "Hesitant", "Frustrated")
INTENT_CLASSES: Tuple[str, ...] = (
    "Buy",
    "Rent",
    "Inquiry",
    "Schedule Visit",
    "Investment",
    "Request Callback",
)

SEED = 42
DEFAULT_MAX_LENGTH = 256  # customer turns top out at 78 words; 256 never truncates

# Telegraphic transcripts are excluded from both heads by default — see
# :func:`load_corpus`. Set False only to reproduce a pre-exclusion measurement.
EXCLUDE_TELEGRAPHIC = True

CUSTOMER_TURN = re.compile(r"^Customer:\s*(.*)$", re.MULTILINE)


def pools_module(name: str = DEFAULT_POOLS) -> Any:
    """Imports the phrase-pool module a corpus was generated from."""
    import importlib  # noqa: PLC0415

    if name not in POOL_MODULES:
        raise SystemExit(f"Unknown pools {name!r}; expected one of {sorted(POOL_MODULES)}.")
    return importlib.import_module(POOL_MODULES[name])


def resolve_corpus(name_or_path: str) -> Tuple[Path, str, Path]:
    """Returns (corpus path, pools name, split path) for a name or a raw path.

    An unrecognised value is treated as a path to a JSONL corpus and defaults to
    the v3 pools, which is what a hand-made corpus in the original format wants.
    """
    named = NAMED_CORPORA.get(name_or_path)
    if named:
        return named["path"], named["pools"], named["split"]
    path = Path(name_or_path)
    return path, DEFAULT_POOLS, path.with_suffix(".split.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def customer_turns(transcript: str) -> str:
    """Returns only what the customer said, agent lines dropped.

    Falls back to the raw text when no ``Customer:`` prefixes are present, so a
    bare utterance passed straight to ``predict`` still works.
    """
    turns = CUSTOMER_TURN.findall(transcript)
    return " ".join(t.strip() for t in turns if t.strip()) or transcript.strip()


def load_corpus(
    path: Path = SYNTHETIC_PATH, exclude_telegraphic: bool = EXCLUDE_TELEGRAPHIC
) -> List[Dict[str, Any]]:
    """Loads the synthetic transcripts with their sentiment/intent labels.

    Telegraphic transcripts are dropped by default. A telegraphic caller says
    "Pune. 2 BHK. 60 lakhs." — bare requirement fragments, no tone clause and
    no intent phrase. The generator still stamps each one with a sentiment and
    an intent, but nothing in the customer's words expresses either, so the
    pair is an unlearnable input with a confident label: noise to train on and
    noise to score against. They are ~11% of the corpus and, because they can
    never match a held-out phrase, they concentrate on the in-distribution
    side of the phrase-holdout diagnostic (17% of that test set against 0% of
    the held-out one), which depressed the in-distribution number and made the
    generalisation gap look smaller than it was.

    Dropping them here rather than editing the transcripts leaves the text
    untouched, so the Phase 3 NER dataset — which *does* want telegraphic
    input, since the requirement fields are all still there to extract — reads
    the same file unchanged.
    """
    if not path.exists():
        raise SystemExit(f"Synthetic transcripts not found at {path}. Run data_generation first.")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row["metadata"]
            records.append(
                {
                    "transcript_id": row["transcript_id"],
                    "text": row["text"],
                    "customer_text": customer_turns(row["text"]),
                    "sentiment": meta["sentiment"],
                    "intent": meta["intent"],
                    "completeness": meta["completeness"],
                    # Kept on every record even when the exclusion is on, so
                    # the scorers can still report a telegraphic count rather
                    # than assume one.
                    "is_telegraphic": bool(meta.get("flags", {}).get("is_telegraphic")),
                    # Pilot-corpus fields. Absent from the v3 corpus, which has
                    # no clean/messy axis, so they default rather than KeyError
                    # and the scorers can test for them.
                    "style": meta.get("style"),
                    "emotion_idea": meta.get("emotion_idea"),
                    "emotion_family": meta.get("emotion_family"),
                    "emotion_register": meta.get("emotion_register"),
                }
            )

    for field, expected in (("sentiment", SENTIMENT_CLASSES), ("intent", INTENT_CLASSES)):
        seen = {r[field] for r in records}
        if seen - set(expected):
            raise SystemExit(f"Corpus has unexpected {field} labels: {sorted(seen - set(expected))}")

    total = len(records)
    if exclude_telegraphic:
        records = [r for r in records if not r["is_telegraphic"]]
        logger.info(
            "Loaded %d of %d transcripts from %s (%d telegraphic excluded: no tone "
            "clause and no intent phrase, so their labels are unlearnable).",
            len(records), total, path.name, total - len(records),
        )
    else:
        logger.warning(
            "Loaded all %d transcripts from %s including %d telegraphic ones — "
            "their sentiment/intent labels are noise.",
            total, path.name, sum(1 for r in records if r["is_telegraphic"]),
        )
    return records


def _corpus_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    """Stable digest of ids and labels, used to invalidate a stale split."""
    payload = "\n".join(
        f"{r['transcript_id']}|{r['sentiment']}|{r['intent']}"
        for r in sorted(records, key=lambda r: r["transcript_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_split(
    records: Sequence[Dict[str, Any]],
    seed: int = SEED,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    path: Path = SPLIT_PATH,
) -> Dict[str, str]:
    """Creates (or loads) a transcript-id -> split assignment.

    Stratified on the joint (sentiment, intent) key so both heads get balanced
    splits from one file. Written once and reused: regenerating per run would
    silently move transcripts between train and test across experiments.
    """
    fingerprint = _corpus_fingerprint(records)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            saved.get("seed") == seed
            and saved.get("n") == len(records)
            and saved.get("fingerprint") == fingerprint
        ):
            logger.info("Reusing saved split from %s", path.name)
            return saved["assignment"]
        # Size alone is not enough: regenerating the corpus reuses the same ids
        # while changing what each one says and how it is labelled, so a stale split
        # would silently stratify on labels the transcripts no longer carry.
        logger.warning("Saved split doesn't match this corpus; rebuilding.")

    buckets: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    for record in records:
        buckets[(record["sentiment"], record["intent"])].append(record["transcript_id"])

    rng = random.Random(seed)
    assignment: Dict[str, str] = {}
    train_ratio, val_ratio, _ = ratios
    for key in sorted(buckets):
        ids = sorted(buckets[key])
        rng.shuffle(ids)
        # Round per stratum so every (sentiment, intent) cell contributes to all
        # three splits rather than the remainder piling into one of them.
        n_train = round(len(ids) * train_ratio)
        n_val = round(len(ids) * val_ratio)
        for index, transcript_id in enumerate(ids):
            if index < n_train:
                assignment[transcript_id] = "train"
            elif index < n_train + n_val:
                assignment[transcript_id] = "validation"
            else:
                assignment[transcript_id] = "test"

    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n": len(records),
                "fingerprint": fingerprint,
                "ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]},
                "stratified_on": ["sentiment", "intent"],
                "sizes": dict(collections.Counter(assignment.values())),
                "assignment": assignment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote split to %s: %s", path.name, dict(collections.Counter(assignment.values())))
    return assignment


def split_records(
    records: Sequence[Dict[str, Any]], assignment: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for record in records:
        out[assignment[record["transcript_id"]]].append(record)
    return out


class TextClassificationDataset(Dataset):
    """Tokenizes on access; padded per batch by :func:`collate`."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        label_field: str,
        classes: Sequence[str],
        text_field: str = "customer_text",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.label_field = label_field
        self.label2id = {label: index for index, label in enumerate(classes)}
        self.text_field = text_field
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        encoding = self.tokenizer(
            record[self.text_field], truncation=True, max_length=self.max_length
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": self.label2id[record[self.label_field]],
        }


def collate(batch: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pads to the longest sequence in the batch rather than to ``max_length``."""
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention = [], []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention.append(item["attention_mask"] + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def count_truncated(
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    text_field: str = "customer_text",
    max_length: int = DEFAULT_MAX_LENGTH,
) -> List[str]:
    """Transcript ids whose text does not fit — their tail would be dropped."""
    truncated: List[str] = []
    for record in records:
        if len(tokenizer(record[text_field], truncation=False)["input_ids"]) > max_length:
            truncated.append(record["transcript_id"])
    return truncated


def classification_metrics(
    references: Sequence[int], predictions: Sequence[int], classes: Sequence[str]
) -> Dict[str, Any]:
    """Accuracy, macro-F1 and per-class P/R/F1 plus a confusion matrix.

    Macro-F1 is the headline: with near-uniform classes, accuracy hides a head
    that has quietly stopped predicting one of them.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(range(len(classes)))
    report = classification_report(
        references,
        predictions,
        labels=labels,
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(f1_score(references, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(references, predictions, average="weighted", zero_division=0)
        ),
        "per_class": {name: report[name] for name in classes},
        "confusion_matrix": {
            "labels": list(classes),
            "rows_are_true": True,
            "matrix": confusion_matrix(references, predictions, labels=labels).tolist(),
        },
    }


def log_metrics_table(title: str, metrics: Dict[str, Any]) -> None:
    logger.info("")
    logger.info("%s", title)
    logger.info("  %-18s %10s %10s %10s %9s", "class", "precision", "recall", "f1", "support")
    logger.info("  %s", "-" * 60)
    for name, scores in metrics["per_class"].items():
        logger.info(
            "  %-18s %10.4f %10.4f %10.4f %9d",
            name, scores["precision"], scores["recall"], scores["f1-score"], int(scores["support"]),
        )
    logger.info("  %s", "-" * 60)
    logger.info("  accuracy %.4f   macro-F1 %.4f", metrics["accuracy"], metrics["macro_f1"])


# --------------------------------------------------------------------------
# Baselines
#
# The exit criterion is "beats a trivial baseline by a clear margin". Majority
# class is the trivial one. The template baseline below is the honest one: the
# synthetic generator builds each transcript from a small fixed pool of
# per-class phrases, so a plain substring matcher over those pools measures how
# much of the task is memorisation rather than language understanding. If it
# scores near the neural model, the headline metric is not evidence of
# generalisation. Phrase pools are imported from the generator, never copied.
# --------------------------------------------------------------------------

def _phrase_pools(pools: str = DEFAULT_POOLS) -> Dict[str, Dict[str, List[str]]]:
    module = pools_module(pools)
    sentiment_pool = {
        label: [*module.SENTIMENT_OPENERS[label], *module.DEFLECTIONS[label]]
        for label in SENTIMENT_CLASSES
    }
    return {"sentiment": sentiment_pool, "intent": dict(module.INTENT_PHRASES)}


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumerics so template matching ignores the
    generator's decorative punctuation and emoji."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def template_baseline(
    records: Sequence[Dict[str, Any]],
    label_field: str,
    classes: Sequence[str],
    exclude: Optional[Dict[str, Sequence[str]]] = None,
    pools: str = DEFAULT_POOLS,
) -> Dict[str, Any]:
    """Scores a substring matcher built from the generator's own phrase pools.

    ``exclude`` removes phrases from the matcher's vocabulary. The phrase-
    holdout diagnostic passes the held-out phrases here so the matcher knows
    exactly what the model saw in training — otherwise it would "recognise"
    phrasings the model was never shown, which is not a comparison of like
    with like.
    """
    pool = _phrase_pools(pools)[label_field]
    if exclude:
        pool = {
            label: [p for p in phrases if p not in set(exclude.get(label, ()))]
            for label, phrases in pool.items()
        }
    normalized_pools = {
        label: [_normalize(p) for p in phrases] for label, phrases in pool.items()
    }
    label2id = {label: index for index, label in enumerate(classes)}

    references, predictions, matched = [], [], 0
    for record in records:
        haystack = _normalize(record["text"])
        scores = {
            label: sum(1 for phrase in phrases if phrase.strip() and phrase.strip() in haystack)
            for label, phrases in normalized_pools.items()
        }
        best = max(scores, key=lambda label: scores[label])
        if scores[best] == 0:
            # No template hit: fall back to the majority class, as a naive
            # keyword system would.
            best = collections.Counter(r[label_field] for r in records).most_common(1)[0][0]
        else:
            matched += 1
        references.append(label2id[record[label_field]])
        predictions.append(label2id[best])

    metrics = classification_metrics(references, predictions, classes)
    metrics["template_hit_rate"] = round(matched / max(len(records), 1), 4)
    return metrics


def majority_baseline(
    train_records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]],
    label_field: str,
    classes: Sequence[str],
) -> Dict[str, Any]:
    """Always predicts the most frequent training class."""
    label2id = {label: index for index, label in enumerate(classes)}
    majority = collections.Counter(r[label_field] for r in train_records).most_common(1)[0][0]
    references = [label2id[r[label_field]] for r in eval_records]
    predictions = [label2id[majority]] * len(eval_records)
    metrics = classification_metrics(references, predictions, classes)
    metrics["majority_class"] = majority
    return metrics


def resolve_latest(task: str, models_store: Path = MODELS_STORE) -> Optional[Path]:
    """Path of the most recent artifact for ``task`` ('sentiment' or 'intent')."""
    pointer = models_store / f"{task}_latest.json"
    if pointer.exists():
        candidate = Path(json.loads(pointer.read_text(encoding="utf-8"))["model_dir"])
        if candidate.exists():
            return candidate
    candidates = sorted(models_store.glob(f"distilbert_{task}_*"))
    return candidates[-1] if candidates else None
