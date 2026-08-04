"""Shared encoding and inference for the MuRIL token-classification model.

Used by both ``train.py`` and ``benchmark.py`` so the two agree exactly on how
words are encoded and how sub-token predictions collapse back to word level.

Sub-token labeling policy (finalizing the open question in schema.md section 4):
the first sub-token of each word carries the word's BIO tag; continuation
sub-tokens are masked with ``-100`` so they contribute no loss and are ignored
at prediction time. Predictions are read off the first sub-token of each word.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_MAX_LENGTH = 512


@dataclass
class Example:
    transcript_id: str
    tokens: List[str]
    tags: List[str]


def load_label_list(dataset_dir: Path) -> List[str]:
    return json.loads((dataset_dir / "label_list.json").read_text(encoding="utf-8"))


def load_rows(
    dataset_dir: Path, split: Optional[str] = None, reviewed_only: bool = False
) -> List[Dict[str, Any]]:
    """Loads dataset rows, optionally restricted to human-reviewed transcripts.

    ``reviewed_only`` drops rows whose annotation is still the untouched Stage B
    rule pre-label. Scoring the rule extractor against those is circular, so any
    honest rules-vs-model comparison must exclude them.
    """
    path = dataset_dir / "ner_dataset.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    if reviewed_only:
        rows = [r for r in rows if r.get("reviewed")]
    return rows


def encode(
    tokenizer: Any,
    tokens: Sequence[str],
    tags: Optional[Sequence[str]],
    label2id: Dict[str, int],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Any]:
    """Encodes pre-split words, aligning BIO tags onto first sub-tokens."""
    encoding = tokenizer(
        list(tokens),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    word_ids = encoding.word_ids()

    labels: List[int] = []
    previous_word: Optional[int] = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(label2id[tags[word_id]] if tags is not None else IGNORE_INDEX)
        previous_word = word_id

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "word_ids": word_ids,
    }


def count_truncated(
    tokenizer: Any, rows: Sequence[Dict[str, Any]], max_length: int = DEFAULT_MAX_LENGTH
) -> List[str]:
    """Transcripts whose words don't all fit — their tail labels would be lost."""
    truncated: List[str] = []
    for row in rows:
        encoding = tokenizer(row["tokens"], is_split_into_words=True, truncation=False)
        if len(encoding["input_ids"]) > max_length:
            truncated.append(row["transcript_id"])
    return truncated


class NerDataset(Dataset):
    """Token-classification examples, padded per batch by :func:`collate`."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        label2id: Dict[str, int],
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        encoded = encode(
            self.tokenizer, row["tokens"], row["ner_tags"], self.label2id, self.max_length
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
        }


def collate(batch: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pads to the longest sequence in the batch rather than to ``max_length``."""
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention, labels = [], [], []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [IGNORE_INDEX] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class MurilNerPredictor:
    """Loads a saved artifact and predicts word-level BIO tags."""

    def __init__(self, model_dir: Path, device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    @torch.no_grad()
    def predict_words(
        self, tokens: Sequence[str], max_length: int = DEFAULT_MAX_LENGTH
    ) -> List[str]:
        """Returns one BIO tag per input word (always ``len(tokens)`` long)."""
        if not tokens:
            return []
        encoding = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        logits = self.model(**inputs).logits[0]
        predictions = logits.argmax(dim=-1).tolist()

        # Words past the truncation point keep "O" rather than going missing,
        # so the returned sequence always aligns with the gold tag sequence.
        tags = ["O"] * len(tokens)
        previous_word: Optional[int] = None
        for position, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word:
                tags[word_id] = self.id2label[predictions[position]]
            previous_word = word_id
        return tags
