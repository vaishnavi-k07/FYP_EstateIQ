"""Trains the 4-class sentiment head (Enthusiastic/Neutral/Hesitant/Frustrated).

Stage 2 of the two-stage scheme: takes the SST-2 warm-started encoder from
``domain_adapt.stage1_warmstart`` and fine-tunes it with a fresh 4-way head on
the labeled synthetic transcripts.

Run:
    python -m sentiment_intent.domain_adapt          # stage 1, once
    python -m sentiment_intent.train_sentiment       # stage 2
    python -m sentiment_intent.train_sentiment --no-warmstart   # ablation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import SEED, SENTIMENT_CLASSES  # noqa: E402
from sentiment_intent.domain_adapt import (  # noqa: E402
    WARMSTART_DIR,
    Stage2Config,
    fine_tune,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=Stage2Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Stage2Config.batch_size)
    parser.add_argument("--lr", type=float, default=Stage2Config.learning_rate)
    parser.add_argument("--max-length", type=int, default=Stage2Config.max_length)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--text-field",
        choices=["customer_text", "text"],
        default=Stage2Config.text_field,
        help="'customer_text' (default) drops agent turns; 'text' keeps the full dialogue.",
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Skip stage 1 and fine-tune stock DistilBERT — the transfer ablation.",
    )
    parser.add_argument("--warmstart-dir", type=Path, default=WARMSTART_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    config = Stage2Config(
        task="sentiment",
        classes=SENTIMENT_CLASSES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        text_field=args.text_field,
        warmstart_dir=None if args.no_warmstart else str(args.warmstart_dir),
    )
    fine_tune(config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
