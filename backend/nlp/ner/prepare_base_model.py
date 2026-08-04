"""Stage D — materialize the MuRIL base checkpoint locally.

``google/muril-base-cased`` publishes only ``pytorch_model.bin`` (no
safetensors), and resolving it straight from the Hub is fragile on a slow or
flaky connection. This fetches each file with resume + retry, then re-saves the
checkpoint as safetensors in a local directory so training loads from disk and
never depends on the network again.

Run:
    python -m nlp.ner.prepare_base_model
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logger = logging.getLogger(__name__)

REPO_ID = "google/muril-base-cased"
TARGET_DIR = BACKEND_DIR / "models_store" / "base" / "muril-base-cased"
REQUIRED_FILES: List[str] = [
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin",
]


def download_with_retry(repo_id: str, filename: str, attempts: int = 8) -> Path:
    """hf_hub_download already resumes partial files; this adds outer retries."""
    from huggingface_hub import hf_hub_download

    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            logger.info("  [%d/%d] %s", attempt, attempts, filename)
            return Path(hf_hub_download(repo_id=repo_id, filename=filename))
        except Exception as error:  # noqa: BLE001 - retry on any transport failure
            last_error = error
            wait = min(30, 3 * attempt)
            logger.warning("      failed (%s); retrying in %ds", type(error).__name__, wait)
            time.sleep(wait)
    raise SystemExit(f"Could not download {filename} after {attempts} attempts: {last_error}")


def convert_to_safetensors(target: Path) -> None:
    """Re-saves the .bin state dict as safetensors, dropping the unused MLM head."""
    import torch
    from safetensors.torch import save_file

    bin_path = target / "pytorch_model.bin"
    logger.info("Loading %s", bin_path)
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    # Token classification only needs the encoder; the pretraining heads would
    # otherwise trip safetensors' shared-tensor check via tied embeddings.
    cleaned = {
        key: value.contiguous()
        for key, value in state_dict.items()
        if not key.startswith(("cls.", "seq_relationship"))
    }
    dropped = len(state_dict) - len(cleaned)

    out_path = target / "model.safetensors"
    save_file(cleaned, str(out_path), metadata={"format": "pt"})
    logger.info("Wrote %s (%d tensors, dropped %d pretraining-head tensors)",
                out_path, len(cleaned), dropped)
    bin_path.unlink()
    logger.info("Removed %s", bin_path)


def verify(target: Path) -> bool:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    logger.info("Verifying local checkpoint loads...")
    tokenizer = AutoTokenizer.from_pretrained(str(target))
    model = AutoModelForTokenClassification.from_pretrained(str(target), num_labels=15)
    encoded = tokenizer(["Baner", "Pune"], is_split_into_words=True, return_tensors="pt")
    logits = model(**encoded).logits
    logger.info("  tokenizer vocab: %d", tokenizer.vocab_size)
    logger.info("  parameters     : %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("  forward pass   : logits %s", tuple(logits.shape))
    return logits.shape[-1] == 15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--target", type=Path, default=TARGET_DIR)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    target: Path = args.target
    if (target / "model.safetensors").exists():
        logger.info("Local checkpoint already present at %s", target)
        return 0 if verify(target) else 1

    target.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching %s -> %s", args.repo_id, target)
    for filename in REQUIRED_FILES:
        cached = download_with_retry(args.repo_id, filename)
        shutil.copyfile(cached, target / filename)

    convert_to_safetensors(target)
    ok = verify(target)
    logger.info("")
    logger.info("Base model ready: %s", target)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
