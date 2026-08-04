"""The two-stage transfer machinery both Phase 2 heads share.

Stage 1 (:func:`stage1_warmstart`) fine-tunes DistilBERT on SST-2 and keeps
only the resulting encoder. Stage 2 (:func:`fine_tune`) starts from that
encoder, attaches a fresh head sized for the target label space, and fine-tunes
on the labeled real-estate transcripts.

Only the encoder crosses the boundary. SST-2 is binary and the domain schemes
are 4-way and 6-way, so the stage-1 head is not merely unhelpful but wrongly
shaped; :func:`load_warmstart_encoder` copies ``distilbert.*`` tensors and
nothing else, which is what makes the label-space mismatch a non-issue.

Run stage 1 once, then reuse it for both heads:
    python -m sentiment_intent.domain_adapt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    MODELS_STORE,
    SEED,
    TextClassificationDataset,
    build_split,
    classification_metrics,
    collate,
    count_truncated,
    load_corpus,
    log_metrics_table,
    majority_baseline,
    set_seed,
    split_records,
    template_baseline,
)

logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
GENERAL_CORPUS = "stanfordnlp/sst2"
WARMSTART_DIR = MODELS_STORE / "encoder_warmstart"

# Prefix of the weights that transfer. Everything else (pre_classifier,
# classifier) is task-specific and is left freshly initialized.
ENCODER_PREFIX = "distilbert."


@dataclass
class Stage1Config:
    base_model: str = BASE_MODEL
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-5
    # SST-2's 67k rows are far more than a warm-start needs and cost ~4x the
    # runtime on CPU for a fraction of a point. Raise for a stronger encoder.
    max_train_samples: Optional[int] = 20000
    max_length: int = 128  # SST-2 sentences: mean 9 words, max 52
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = SEED


@dataclass
class Stage2Config:
    task: str = "sentiment"
    epochs: int = 8
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    max_length: int = DEFAULT_MAX_LENGTH
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = SEED
    text_field: str = "customer_text"
    base_model: str = BASE_MODEL
    warmstart_dir: Optional[str] = None
    classes: Tuple[str, ...] = field(default_factory=tuple)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optimizer(model: Any, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for n, p in trainable if not any(s in n for s in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in trainable if any(s in n for s in ("bias", "LayerNorm.weight"))]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


@torch.no_grad()
def evaluate(
    model: Any, loader: DataLoader, classes: Sequence[str], device: torch.device
) -> Tuple[Dict[str, Any], float]:
    model.eval()
    references: List[int] = []
    predictions: List[int] = []
    total_loss, batches = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += float(outputs.loss)
        batches += 1
        predictions.extend(outputs.logits.argmax(dim=-1).tolist())
        references.extend(batch["labels"].tolist())

    return classification_metrics(references, predictions, classes), total_loss / max(batches, 1)


def _run_epochs(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: Sequence[str],
    device: torch.device,
    epochs: int,
    config: Any,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, torch.Tensor]], float, int]:
    """Trains, tracking the best epoch by validation macro-F1."""
    optimizer = _optimizer(model, config.learning_rate, config.weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    logger.info(
        "Training: %d epochs, %d steps/epoch, %d total steps (lr %.0e)",
        epochs, len(train_loader), total_steps, config.learning_rate,
    )

    history: List[Dict[str, Any]] = []
    best_f1, best_epoch = -1.0, -1
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
            if step % 50 == 0:
                logger.info("  epoch %d  step %d/%d  loss %.4f",
                            epoch, step, len(train_loader), epoch_loss / step)

        val_metrics, val_loss = evaluate(model, val_loader, classes, device)
        logger.info(
            "epoch %d  train_loss %.4f  val_loss %.4f  val_acc %.4f  val_macro_f1 %.4f",
            epoch, epoch_loss / len(train_loader), val_loss,
            val_metrics["accuracy"], val_metrics["macro_f1"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(train_loader),
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return history, best_state, best_f1, best_epoch


def stage1_warmstart(config: Stage1Config, output_dir: Optional[Path] = None) -> Path:
    """Fine-tunes DistilBERT on SST-2 and saves the adapted encoder."""
    from datasets import load_dataset

    set_seed(config.seed)
    device = _device()
    logger.info("Stage 1 — general sentiment warm-start on %s (device: %s)",
                GENERAL_CORPUS, device)

    dataset = load_dataset(GENERAL_CORPUS)
    train_split = dataset["train"].shuffle(seed=config.seed)
    if config.max_train_samples:
        train_split = train_split.select(range(min(config.max_train_samples, len(train_split))))
    # SST-2's test split is unlabeled (all -1), so validation is the only
    # usable held-out set here.
    val_split = dataset["validation"]
    class_names: List[str] = dataset["train"].features["label"].names
    logger.info("SST-2: %d train (of %d) / %d validation",
                len(train_split), len(dataset["train"]), len(val_split))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=len(class_names)
    ).to(device)

    def to_records(split: Any) -> List[Dict[str, Any]]:
        return [
            {"transcript_id": str(i), "customer_text": s, "label_name": class_names[y]}
            for i, (s, y) in enumerate(zip(split["sentence"], split["label"]))
        ]

    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        TextClassificationDataset(
            to_records(train_split), tokenizer, "label_name", class_names,
            max_length=config.max_length,
        ),
        batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, generator=generator,
    )
    val_loader = DataLoader(
        TextClassificationDataset(
            to_records(val_split), tokenizer, "label_name", class_names,
            max_length=config.max_length,
        ),
        batch_size=config.batch_size * 2, collate_fn=collate_fn,
    )

    history, best_state, best_f1, best_epoch = _run_epochs(
        model, train_loader, val_loader, class_names, device, config.epochs, config
    )
    if best_state is not None:
        logger.info("Restoring best stage-1 checkpoint (epoch %d, val macro-F1 %.4f).",
                    best_epoch, best_f1)
        model.load_state_dict(best_state)
        model.to(device)

    val_metrics, _ = evaluate(model, val_loader, class_names, device)
    log_metrics_table("STAGE 1 — SST-2 validation", val_metrics)

    target = output_dir or WARMSTART_DIR
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    (target / "metrics.json").write_text(
        json.dumps(
            {
                "stage": 1,
                "role": "encoder warm-start only; the binary head is discarded",
                "corpus": GENERAL_CORPUS,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config),
                "class_names": class_names,
                "best_epoch": best_epoch,
                "history": history,
                "validation_metrics": val_metrics,
            },
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    logger.info("Saved warm-start encoder to %s", target)
    return target


def load_warmstart_encoder(model: Any, warmstart_dir: Path) -> int:
    """Copies ``distilbert.*`` weights from a stage-1 checkpoint into ``model``.

    Returns the number of tensors transferred. The classification head is
    deliberately left untouched — it is a different shape and a different task.
    """
    from safetensors.torch import load_file

    weights_path = warmstart_dir / "model.safetensors"
    if weights_path.exists():
        state = load_file(str(weights_path))
    else:
        state = torch.load(str(warmstart_dir / "pytorch_model.bin"), map_location="cpu")

    target_state = model.state_dict()
    transferred = {
        key: value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
        and key in target_state
        and target_state[key].shape == value.shape
    }
    missing = [k for k in state if k.startswith(ENCODER_PREFIX) and k not in transferred]
    if missing:
        logger.warning("%d encoder tensors did not match and were skipped: %s",
                       len(missing), missing[:5])

    model.load_state_dict(transferred, strict=False)
    logger.info("Warm-start: transferred %d encoder tensors from %s (head reinitialized).",
                len(transferred), warmstart_dir.name)
    return len(transferred)


def fine_tune(
    config: Stage2Config,
    output_dir: Optional[Path] = None,
    splits: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    update_pointer: bool = True,
) -> Path:
    """Stage 2 — fine-tunes a fresh head on the labeled transcripts.

    ``splits`` overrides the standard corpus split; the phrase-holdout
    diagnostic uses it to train on only the transcripts whose phrasing it wants
    the model to have seen. Such a run must pass ``update_pointer=False`` so a
    deliberately handicapped model never becomes what ``predict.py`` loads.
    """
    if not config.classes:
        raise ValueError("Stage2Config.classes must name the label space.")

    set_seed(config.seed)
    device = _device()
    classes = list(config.classes)
    logger.info("Stage 2 — %s (%d classes) on device %s", config.task, len(classes), device)

    if splits is None:
        records = load_corpus()
        assignment = build_split(records, seed=config.seed)
        splits = split_records(records, assignment)
    else:
        records = [r for split in splits.values() for r in split]
        logger.info("Using caller-supplied splits (%d transcripts).", len(records))
    logger.info("Split: %d train / %d validation / %d test",
                len(splits["train"]), len(splits["validation"]), len(splits["test"]))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(classes),
        id2label={index: name for index, name in enumerate(classes)},
        label2id={name: index for index, name in enumerate(classes)},
    )

    warmstart_used: Optional[str] = None
    if config.warmstart_dir:
        warmstart_path = Path(config.warmstart_dir)
        if not warmstart_path.exists():
            raise SystemExit(
                f"Warm-start encoder not found at {warmstart_path}. "
                "Run `python -m sentiment_intent.domain_adapt` first, or pass --no-warmstart."
            )
        load_warmstart_encoder(model, warmstart_path)
        warmstart_used = str(warmstart_path)
    else:
        logger.warning("No warm-start encoder: starting from stock %s.", config.base_model)
    model.to(device)

    truncated = count_truncated(
        tokenizer, records, config.text_field, config.max_length
    )
    if truncated:
        logger.warning("%d transcript(s) exceed max_length=%d: %s",
                       len(truncated), config.max_length, ", ".join(truncated[:10]))
    else:
        logger.info("All transcripts fit within max_length=%d.", config.max_length)

    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    def loader(name: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            TextClassificationDataset(
                splits[name], tokenizer, config.task, classes,
                text_field=config.text_field, max_length=config.max_length,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            generator=generator if shuffle else None,
        )

    train_loader = loader("train", config.batch_size, shuffle=True)
    val_loader = loader("validation", config.eval_batch_size)
    test_loader = loader("test", config.eval_batch_size)

    history, best_state, best_f1, best_epoch = _run_epochs(
        model, train_loader, val_loader, classes, device, config.epochs, config
    )
    if best_state is not None:
        logger.info("Restoring best checkpoint (epoch %d, val macro-F1 %.4f).", best_epoch, best_f1)
        model.load_state_dict(best_state)
        model.to(device)

    val_metrics, _ = evaluate(model, val_loader, classes, device)
    test_metrics, _ = evaluate(model, test_loader, classes, device)
    log_metrics_table(f"VALIDATION — {config.task}", val_metrics)
    log_metrics_table(f"TEST — {config.task}", test_metrics)

    baselines = {
        "majority": majority_baseline(splits["train"], splits["test"], config.task, classes),
        "template_match": template_baseline(splits["test"], config.task, classes),
    }
    logger.info("")
    logger.info("TEST baselines — majority acc %.4f / macro-F1 %.4f | template acc %.4f / "
                "macro-F1 %.4f (hit rate %.2f)",
                baselines["majority"]["accuracy"], baselines["majority"]["macro_f1"],
                baselines["template_match"]["accuracy"], baselines["template_match"]["macro_f1"],
                baselines["template_match"]["template_hit_rate"])
    if baselines["template_match"]["macro_f1"] >= test_metrics["macro_f1"] - 0.02:
        logger.warning(
            "A plain substring matcher over the generator's phrase pools matches the "
            "model (%.4f vs %.4f macro-F1). The synthetic transcripts are built from "
            "fixed per-class phrases, so this score reflects template memorisation, "
            "not language understanding, and will not transfer to real calls.",
            baselines["template_match"]["macro_f1"], test_metrics["macro_f1"],
        )

    version = datetime.now(timezone.utc).strftime("v1_%Y%m%d_%H%M%S")
    target = output_dir or (MODELS_STORE / f"distilbert_{config.task}_{version}")
    target.mkdir(parents=True, exist_ok=True)
    # The pointer always lives in the store, even when --output-dir sends the
    # artifact elsewhere, so the store may not exist yet.
    MODELS_STORE.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))

    (target / "metrics.json").write_text(
        json.dumps(
            {
                "stage": 2,
                "task": config.task,
                "version": version,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config),
                "warmstart_encoder": warmstart_used,
                "classes": classes,
                "dataset": {
                    "source": "data_generation/output/synthetic_transcripts.jsonl",
                    "text_field": config.text_field,
                    "train": len(splits["train"]),
                    "validation": len(splits["validation"]),
                    "test": len(splits["test"]),
                    "truncated": truncated,
                },
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "history": history,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "baselines": baselines,
            },
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    if update_pointer:
        (MODELS_STORE / f"{config.task}_latest.json").write_text(
            json.dumps(
                {
                    "model_dir": str(target),
                    "version": version,
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                },
                indent=2, default=float,
            ),
            encoding="utf-8",
        )
    else:
        logger.info("Pointer not updated — this artifact is a diagnostic, not the served model.")
    logger.info("")
    logger.info("Saved %s model + metrics to %s", config.task, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=Stage1Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Stage1Config.batch_size)
    parser.add_argument("--lr", type=float, default=Stage1Config.learning_rate)
    parser.add_argument("--max-train-samples", type=int, default=Stage1Config.max_train_samples,
                        help="0 uses all 67k SST-2 rows.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    stage1_warmstart(
        Stage1Config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_train_samples=args.max_train_samples or None,
            seed=args.seed,
        ),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
