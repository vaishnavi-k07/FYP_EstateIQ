"""Stage D — fine-tune MuRIL for real-estate entity extraction.

Trains ``google/muril-base-cased`` with ``AutoModelForTokenClassification`` on
the BIO dataset produced by ``export_dataset.py``, evaluates each epoch on the
validation split with entity-level seqeval metrics, and saves a versioned
artifact plus its metrics to ``models_store/ner/``.

Reproducible by construction: fixed seeds, a saved split, and a metrics file
written alongside every artifact (per CLAUDE.md section 7).

Run:
    python -m nlp.ner.train
    python -m nlp.ner.train --epochs 8 --batch-size 8 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nlp.ner.model import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
    NerDataset,
    collate,
    count_truncated,
    load_label_list,
    load_rows,
)

logger = logging.getLogger(__name__)

NER_DIR = Path(__file__).resolve().parent
DATASET_DIR = NER_DIR / "dataset"
MODELS_STORE = BACKEND_DIR / "models_store" / "ner"

SEED = 42

# Prefer the locally materialized checkpoint (see prepare_base_model.py) so a
# training run never depends on the Hub being reachable; fall back to the Hub id.
_LOCAL_BASE = BACKEND_DIR / "models_store" / "base" / "muril-base-cased"
BASE_MODEL = str(_LOCAL_BASE) if (_LOCAL_BASE / "model.safetensors").exists() \
    else "google/muril-base-cased"


@dataclass
class TrainingConfig:
    base_model: str = BASE_MODEL
    epochs: int = 40
    batch_size: int = 4
    eval_batch_size: int = 16
    learning_rate: float = 5e-5
    # Epoch count alone silently under-trains on a small split: 116 reviewed
    # transcripts at batch 8 is only 15 steps/epoch, and 10 epochs of that
    # (150 steps) leaves the model at the all-"O" majority baseline. Fine-tuning
    # a 237M-parameter encoder needs ~1k steps, so epochs are raised
    # automatically whenever the split is too small to reach this floor.
    min_train_steps: int = 1000
    # The MuRIL embedding matrix is 197k x 768 = 151M of the model's 237M
    # parameters. With only a few hundred training transcripts it cannot be
    # estimated meaningfully, and updating it dominates CPU step time.
    freeze_embeddings: bool = False
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = SEED
    # Train only on hand-corrected transcripts. Slower to reach good F1 with the
    # smaller set, but the labels are real rather than the rule extractor's own
    # output, so the downstream rules-vs-model benchmark stays meaningful.
    reviewed_only: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def entity_metrics(
    predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]
) -> Dict[str, Any]:
    """Entity-level precision/recall/F1 via seqeval."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    report = classification_report(
        references, predictions, output_dict=True, zero_division=0
    )
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


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
) -> Tuple[Dict[str, Any], float]:
    """Runs the model over a loader and scores it at entity level."""
    model.eval()
    all_predictions: List[List[str]] = []
    all_references: List[List[str]] = []
    total_loss = 0.0
    batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += float(outputs.loss)
        batches += 1
        predicted = outputs.logits.argmax(dim=-1)

        for prediction_row, label_row in zip(predicted, batch["labels"]):
            predictions, references = [], []
            for prediction, label in zip(prediction_row.tolist(), label_row.tolist()):
                if label == IGNORE_INDEX:
                    continue
                predictions.append(id2label[prediction])
                references.append(id2label[label])
            all_predictions.append(predictions)
            all_references.append(references)

    return entity_metrics(all_predictions, all_references), total_loss / max(batches, 1)


def log_entity_table(title: str, metrics: Dict[str, Any]) -> None:
    logger.info("")
    logger.info("%s", title)
    logger.info("  %-16s %10s %10s %10s %9s", "entity", "precision", "recall", "f1", "support")
    logger.info("  %s", "-" * 58)
    for entity, scores in sorted(metrics["per_entity"].items()):
        logger.info(
            "  %-16s %10.4f %10.4f %10.4f %9d",
            entity,
            scores["precision"],
            scores["recall"],
            scores["f1-score"],
            int(scores["support"]),
        )
    logger.info("  %s", "-" * 58)
    logger.info(
        "  %-16s %10.4f %10.4f %10.4f",
        "OVERALL (micro)",
        metrics["overall_precision"],
        metrics["overall_recall"],
        metrics["overall_f1"],
    )


def train(config: TrainingConfig, output_dir: Optional[Path] = None) -> Path:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    label_list = load_label_list(DATASET_DIR)
    label2id = {label: index for index, label in enumerate(label_list)}
    id2label = {index: label for label, index in label2id.items()}

    train_rows = load_rows(DATASET_DIR, "train", config.reviewed_only)
    val_rows = load_rows(DATASET_DIR, "validation", config.reviewed_only)
    test_rows = load_rows(DATASET_DIR, "test", config.reviewed_only)
    logger.info(
        "Dataset: %d train / %d validation / %d test transcripts%s",
        len(train_rows), len(val_rows), len(test_rows),
        "  (human-reviewed only)" if config.reviewed_only else "",
    )
    if not config.reviewed_only:
        unreviewed = sum(
            1 for r in train_rows + val_rows + test_rows if not r.get("reviewed")
        )
        if unreviewed:
            logger.warning(
                "%d of these transcripts still carry unreviewed rule pre-labels. "
                "Metrics computed over them are not an independent check on the "
                "rule extractor — see --reviewed-only.", unreviewed,
            )

    logger.info("Loading tokenizer and model: %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    if config.freeze_embeddings:
        frozen = 0
        for name, parameter in model.named_parameters():
            if name.startswith("bert.embeddings."):
                parameter.requires_grad = False
                frozen += parameter.numel()
        logger.info("Froze embeddings: %.1fM parameters held fixed.", frozen / 1e6)
    model.to(device)

    truncated = count_truncated(tokenizer, train_rows + val_rows + test_rows, config.max_length)
    if truncated:
        logger.warning(
            "%d transcript(s) exceed max_length=%d and will lose tail labels: %s",
            len(truncated), config.max_length, ", ".join(truncated[:10]),
        )
    else:
        logger.info("All transcripts fit within max_length=%d.", config.max_length)

    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        NerDataset(train_rows, tokenizer, label2id, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        NerDataset(val_rows, tokenizer, label2id, config.max_length),
        batch_size=config.eval_batch_size,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        NerDataset(test_rows, tokenizer, label2id, config.max_length),
        batch_size=config.eval_batch_size,
        collate_fn=collate_fn,
    )

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay_params = [p for n, p in trainable
                    if not any(s in n for s in ("bias", "LayerNorm.weight"))]
    no_decay_params = [p for n, p in trainable
                       if any(s in n for s in ("bias", "LayerNorm.weight"))]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )
    steps_per_epoch = len(train_loader)
    epochs = config.epochs
    if steps_per_epoch * epochs < config.min_train_steps:
        needed = -(-config.min_train_steps // max(steps_per_epoch, 1))  # ceil
        logger.warning(
            "%d epochs x %d steps/epoch = %d steps, below the %d-step floor "
            "needed to fine-tune this encoder; raising to %d epochs.",
            epochs, steps_per_epoch, steps_per_epoch * epochs,
            config.min_train_steps, needed,
        )
        epochs = needed

    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )

    logger.info("Training: %d epochs, %d steps/epoch, %d total steps (lr %.0e)",
                epochs, steps_per_epoch, total_steps, config.learning_rate)

    history: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += float(outputs.loss)
            if step % 10 == 0:
                logger.info("  epoch %d  step %d/%d  loss %.4f",
                            epoch, step, len(train_loader), epoch_loss / step)

        val_metrics, val_loss = evaluate(model, val_loader, id2label, device)
        logger.info(
            "epoch %d  train_loss %.4f  val_loss %.4f  val_f1 %.4f",
            epoch, epoch_loss / len(train_loader), val_loss, val_metrics["overall_f1"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(train_loader),
                "val_loss": val_loss,
                "val_f1": val_metrics["overall_f1"],
                "val_precision": val_metrics["overall_precision"],
                "val_recall": val_metrics["overall_recall"],
            }
        )

        if val_metrics["overall_f1"] > best_f1:
            best_f1 = val_metrics["overall_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        logger.info("Restoring best checkpoint (epoch %d, val_f1 %.4f).", best_epoch, best_f1)
        model.load_state_dict(best_state)
        model.to(device)

    val_metrics, _ = evaluate(model, val_loader, id2label, device)
    test_metrics, _ = evaluate(model, test_loader, id2label, device)
    log_entity_table("VALIDATION — entity-level (seqeval)", val_metrics)
    log_entity_table("TEST — entity-level (seqeval)", test_metrics)

    suffix = "_reviewed" if config.reviewed_only else "_all"
    version = datetime.now(timezone.utc).strftime("v1_%Y%m%d_%H%M%S") + suffix
    target = output_dir or (MODELS_STORE / f"muril_ner_{version}")
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))

    metrics_payload = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": config.base_model,
        "config": asdict(config),
        "epochs_run": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "dataset": {
            "dir": str(DATASET_DIR),
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
            "labels": label_list,
            "truncated_transcripts": truncated,
        },
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (target / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=float), encoding="utf-8"
    )
    (MODELS_STORE / "latest.json").write_text(
        json.dumps({"model_dir": str(target), "version": version, "test_f1":
                    test_metrics["overall_f1"]}, indent=2, default=float),
        encoding="utf-8",
    )
    logger.info("")
    logger.info("Saved model + metrics to %s", target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--max-length", type=int, default=TrainingConfig.max_length)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--base-model", default=TrainingConfig.base_model)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Use only hand-corrected transcripts (excludes raw rule pre-labels).",
    )
    parser.add_argument("--min-train-steps", type=int, default=TrainingConfig.min_train_steps)
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Hold MuRIL's 151M-parameter embedding matrix fixed (faster on CPU).",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    config = TrainingConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        reviewed_only=args.reviewed_only,
        min_train_steps=args.min_train_steps,
        freeze_embeddings=args.freeze_embeddings,
    )
    train(config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
