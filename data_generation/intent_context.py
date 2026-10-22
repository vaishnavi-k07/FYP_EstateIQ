"""Intent corpus — intent implied by situation, never named by a stock phrase.

Separate from every other corpus: its own pools (:mod:`intent_context_pools`),
its own output file, its own split. The v6 sentiment corpus, the NER corpus and
both of their generators are untouched; only the intent head trains on this.

What this corpus fixes
----------------------
The v6-trained intent head scored 1.000 on its test split and 12% on the
unseen-50 set — 57% where the caller named an intent, 2.6% where the intent
had to be read from context. v6 carried a stock intent phrase ("we want to
buy", "rent par lena") in 1456 of 1504 transcripts, so the head learned the
phrases. This corpus applies the design that fixed the same failure in
sentiment:

*Intent is situational.* Every transcript expresses its intent through one
idea from :mod:`intent_context_pools` — a sanctioned loan, an advance
question, "nobody from my family will stay in it" — and never through the
words buy / rent / invest / visit / call back / enquiry.

*One idea per transcript*, one or two surface forms of it, so the
phrase-holdout diagnostic can withhold whole ideas with nothing dropped.

*The surroundings carry no label.* Emotion lines (from the sentiment pools,
screened for intent content), shared framing sentences, messiness, requirement
completeness, and where in the turn the intent sentence sits are all dealt in
exactly the same proportions for every intent.

*Requirements match the intent where a real caller's would.* A Rent caller
states a monthly figure, not "85 lakh"; this corpus swaps the purchase budget
for a monthly one on Rent calls. Monthly figures are not left as a free Rent
cue: EMI and tenant-income ideas put them under Buy and Investment too.

Run:
    python -m data_generation.intent_context
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_generation import intent_context_pools as _pools  # noqa: E402
from data_generation.sentiment_pilot import (  # noqa: E402
    AGENT_FOLLOWUPS,
    AGENT_OPENERS,
    Segment,
    _answer_for,
    _normalize,
    build_requirement,
    make_messy,
)
from data_generation.synthetic_transcripts import LookupTables  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SEED = 42

# 240 transcripts per intent, 1440 in all: 24 ideas x 10. The count is fixed
# per *intent* and divided across its ideas, so an intent given more ideas
# (as intent_v2 did for Buy and Investment) cannot dominate the prior. Fixing it per idea instead would
# let the class with more ideas dominate the prior. Each idea's share must be
# even so it splits 50/50 clean/messy and 50/50 with/without an emotion line.
TRANSCRIPTS_PER_INTENT = 240

INTENT_CLASSES = _pools.INTENT_CLASSES

# The v6 closers depend on the intent ("someone will call you back"). Training
# strips agent turns so they never reached the model, but a corpus meant to be
# free of intent templates should not contain them at all.
AGENT_CLOSERS: Tuple[str, ...] = (
    "Agent: Noted, I've got all of that down.",
    "Agent: Understood. I'll take this forward.",
    "Agent: Thanks, that's everything I need for now.",
)

MONTHLY_FRAMES: Tuple[str, ...] = (
    "Around {amount} a month.",
    "Up to {amount} per month.",
    "{amount} a month is the ceiling.",
)
MONTHLY_AMOUNTS: Tuple[str, ...] = (
    "12,000", "15,000", "18,000", "20k", "22,000", "25,000", "28k", "30,000",
    "35,000", "40,000", "fifteen thousand", "twenty thousand",
)


def _tone_pool() -> List[Tuple[str, str, str]]:
    """(tone, idea, expression) for every emotion opener with no intent content."""
    pool = []
    for tone, ideas in _pools.OPENER_IDEAS.items():
        for idea, forms in ideas.items():
            if idea in _pools.TONE_IDEAS_EXCLUDED:
                continue
            # Screened per phrase too: a few otherwise intent-free ideas have
            # one form that names an intent ("I requested a callback", "a
            # place we haven't bought").
            pool.extend(
                (tone, idea, form) for form in forms
                if not any(p.search(form.lower()) for p in _STOCK)
            )
    return pool


def _rent_budget(rng, segments: List[Segment]) -> List[Segment]:
    """Replaces the purchase budget with a monthly figure (or drops it)."""
    out = [s for s in segments if s.kind != "budget"]
    had_budget = len(out) != len(segments)
    if had_budget and rng.random() < 0.8:
        amount = str(rng.choice(MONTHLY_AMOUNTS))
        out.insert(1, Segment(str(rng.choice(MONTHLY_FRAMES)).format(amount=amount),
                              kind="budget"))
    return out


def build_transcript(
    rng,
    lookups: LookupTables,
    intent: str,
    idea: str,
    messy: bool,
    with_tone: bool,
    completeness: str,
    tone_pool: Sequence[Tuple[str, str, str]],
) -> Dict[str, Any]:
    register = _pools.phrase_register(idea)
    forms = list(_pools.INTENT_IDEAS[intent][idea])
    rng.shuffle(forms)
    expressions = forms[: 1 if rng.random() < 0.6 else 2]

    req, ground_truth, corrections = build_requirement(rng, lookups, completeness)
    if intent == "Rent":
        req = _rent_budget(rng, req)
        ground_truth = {**ground_truth, "budget_lakhs": None, "budget_text": None}
    # A self-correction must only rewrite text that is still present.
    present = " ".join(s.text for s in req)
    corrections = [(right, wrong) for right, wrong in corrections if right in present]

    first: List[Segment] = []
    tone, tone_idea = "Neutral", None
    if with_tone:
        tone, tone_idea, tone_text = tone_pool[int(rng.integers(len(tone_pool)))]
        first.append(Segment(tone_text, protected=True, kind="emotion"))
    if rng.random() < 0.3:
        frames = _pools.SHARED_FRAMES_CM if register == "code-mixed" else _pools.SHARED_FRAMES_EN
        first.append(Segment(str(rng.choice(frames)), protected=True, kind="frame"))

    intent_seg = Segment(expressions[0], protected=True, kind="intent")
    # Where the intent sits is dealt, so position cannot become a cue.
    if rng.random() < 0.5:
        first.append(intent_seg)
        first.extend(req)
    else:
        first.extend(req)
        first.append(intent_seg)

    later: List[Segment] = []
    followup_field = str(rng.choice(list(AGENT_FOLLOWUPS)))
    if len(expressions) > 1:
        later.append(Segment(expressions[1], protected=True, kind="intent"))
    else:
        answer = _answer_for(rng, followup_field, ground_truth)
        later.append(answer or Segment("That's about it, really.", kind="answer"))

    turns = [first, later]
    applied: List[str] = []
    if messy:
        applied = make_messy(rng, turns, corrections)

    text = "\n".join([
        str(rng.choice(AGENT_OPENERS)),
        "Customer: " + " ".join(s.text for s in turns[0]),
        str(rng.choice(AGENT_FOLLOWUPS[followup_field])),
        "Customer: " + " ".join(s.text for s in turns[1]),
        str(rng.choice(AGENT_CLOSERS)),
    ])

    normalized = _normalize(text)
    for expression in expressions:
        if _normalize(expression).strip() not in normalized:
            raise AssertionError(f"messiness corrupted an intent expression: {expression!r}")

    return {
        "text": text,
        "ground_truth": ground_truth,
        "metadata": {
            "intent": intent,
            "sentiment": tone,
            "completeness": completeness,
            "style": "messy" if messy else "clean",
            "intent_idea": idea,
            "intent_register": register,
            "intent_expressions": expressions,
            "emotion_idea": tone_idea,
            "messy_ops": applied,
            "flags": {"is_telegraphic": False, "is_code_mixed": register == "code-mixed"},
        },
    }


def _balanced_flags(rng, n: int) -> List[Tuple[bool, bool]]:
    """(messy, with_tone) for one idea: 50/50 on each, crossed as evenly as n allows."""
    half = n // 2
    messy = [True] * half + [False] * (n - half)
    tone = ([True, False] * n)[:n]
    rng.shuffle(tone)
    return list(zip(messy, tone))


def generate(seed: int = SEED, per_intent: int = TRANSCRIPTS_PER_INTENT) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    lookups = LookupTables()
    tone_pool = _tone_pool()

    plan: List[Tuple[str, str, bool, bool]] = []
    for intent in INTENT_CLASSES:
        ideas = _pools.INTENT_IDEAS[intent]
        per_idea, remainder = divmod(per_intent, len(ideas))
        if remainder or per_idea % 2:
            raise SystemExit(
                f"{intent}: {per_intent} transcripts over {len(ideas)} ideas must give an "
                f"even whole number per idea; got {per_intent / len(ideas):g}."
            )
        for idea in ideas:
            for messy, with_tone in _balanced_flags(rng, per_idea):
                plan.append((intent, idea, messy, with_tone))
    rng.shuffle(plan)
    completeness = ["full", "partial", "vague"] * (len(plan) // 3 + 1)
    completeness = completeness[: len(plan)]
    rng.shuffle(completeness)

    records = []
    for index, ((intent, idea, messy, with_tone), comp) in enumerate(zip(plan, completeness), 1):
        record = build_transcript(rng, lookups, intent, idea, messy, with_tone, comp, tone_pool)
        records.append({"transcript_id": f"ic_{index:05d}", **record})
    return records


_STOCK = [re.compile(p) for p in _pools.STOCK_INTENT_PATTERNS]


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    meta = [r["metadata"] for r in records]

    def rate(key_fn, pred) -> Dict[str, float]:
        totals, hits = collections.Counter(), collections.Counter()
        for m in meta:
            totals[key_fn(m)] += 1
            hits[key_fn(m)] += pred(m)
        return {k: round(hits[k] / totals[k], 3) for k in sorted(totals)}

    customer = [
        " ".join(l.split(":", 1)[1] for l in r["text"].splitlines() if l.startswith("Customer:")).lower()
        for r in records
    ]
    stock_hits = sum(1 for text in customer if any(p.search(text) for p in _STOCK))
    return {
        "total": len(records),
        "intent_counts": dict(collections.Counter(m["intent"] for m in meta)),
        "sentiment_counts": dict(collections.Counter(m["sentiment"] for m in meta)),
        "messy_rate_by_intent": rate(lambda m: m["intent"], lambda m: m["style"] == "messy"),
        "tone_line_rate_by_intent": rate(lambda m: m["intent"], lambda m: m["emotion_idea"] is not None),
        "code_mixed_rate_by_intent": rate(lambda m: m["intent"], lambda m: m["intent_register"] == "code-mixed"),
        "transcripts_with_stock_intent_word": stock_hits,
        "pool_checks": {
            "stated_intent_leaks": _pools.stated_intent_leaks(),
            "lopsided_tokens": _pools.lopsided_tokens(),
        },
        "words": {
            "mean": round(float(np.mean([len(r["text"].split()) for r in records])), 1),
        },
    }


def write(records: Sequence[Dict[str, Any]], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records)
    (output_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %d transcripts to %s", len(records), path)
    for key in ("intent_counts", "sentiment_counts", "messy_rate_by_intent",
                "tone_line_rate_by_intent", "code_mixed_rate_by_intent",
                "transcripts_with_stock_intent_word"):
        logger.info("%s: %s", key, summary[key])
    logger.info("Stated-intent leaks in pools: %d | lopsided tokens: %s",
                len(summary["pool_checks"]["stated_intent_leaks"]),
                sorted(summary["pool_checks"]["lopsided_tokens"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--transcripts-per-intent", type=int, default=TRANSCRIPTS_PER_INTENT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--name", default="intent_v1")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    write(generate(args.seed, args.transcripts_per_intent), args.output_dir, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
