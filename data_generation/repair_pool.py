"""One-time surgical repair of the generated transcript pool.

The pool in output/synthetic_transcripts.jsonl was reviewed and accepted,
so it is repaired in place rather than regenerated from scratch — a fresh
`synthetic_transcripts.py` run would reshuffle every transcript and throw
away reviewed hard-case content (negation, code-mixing, multi-city).

Two defects are fixed:

1. Studio/BHK contradiction. "Studio apartment, 2 BHK" is impossible — a
   studio has no bedroom count. Affected records are rebuilt with the fixed
   generator, holding city/intent/sentiment/hard-case flags constant so the
   pool's composition does not move.

2. Verbatim boilerplate repetition. Openers, follow-up questions, customer
   non-answers, and closers were drawn from banks of 2-6 strings across 700
   transcripts, so each one recurred ~50-200 times and the same non-answer
   could appear twice in one call. Those lines — and only those lines — are
   resampled from the expanded banks in `synthetic_transcripts.py`. Every
   entity-bearing span in the transcript is left byte-identical.

Run: python data_generation/repair_pool.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from numpy.random import default_rng

from synthetic_transcripts import (
    CLOSERS_CALLBACK,
    CLOSERS_FRUSTRATED,
    CLOSERS_GENERIC,
    CLOSERS_SCHEDULE_VISIT,
    DEFLECTIONS,
    FOLLOWUP_QUESTIONS,
    OPENERS,
    SENTIMENT_OPENERS,
    LookupTables,
    build_transcript,
    summarize,
    write_pool,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
REPAIR_SEED = 4242

# --------------------------------------------------------------------------
# Legacy banks — verbatim copies of the pre-repair phrasing banks (commit
# 4f967a9). Needed to recognise which lines in the existing pool are
# boilerplate, and which field/closer bank each one came from. Kept here so
# the repair is reproducible without checking out the old generator.
# --------------------------------------------------------------------------

LEGACY_OPENERS = [
    "Agent: Hi, thanks for calling in! What kind of property are you looking for today?",
    "Agent: Hello, thank you for reaching out. Could you tell me a bit about what you're looking for?",
    "Agent: Hi there! What brings you in today — are you looking to buy, rent, or just exploring?",
    "Agent: Good afternoon, thanks for calling. What can I help you find today?",
    "Agent: Hello! So, tell me — what kind of place are you hoping to find?",
    "Agent: Hi, welcome! Let's start with the basics — what are you looking for?",
]

LEGACY_SENTIMENT_OPENERS = {
    "Enthusiastic": [
        "Hi! We're really excited to finally look into this —",
        "Hey there, thanks for picking up! So,",
        "Hi, great, this is perfect timing —",
    ],
    "Neutral": [
        "Hi, so basically,",
        "Hello, yes so",
        "Hi, okay so",
    ],
    "Hesitant": [
        "Um, hi, we're not entirely sure yet, but",
        "Hi... we're still exploring options, but roughly,",
        "Hello, we're just starting to look around, so",
    ],
    "Frustrated": [
        "Hi, I've called before about this and no one got back to me, but anyway,",
        "Yeah hi, look, I don't have much time, but",
        "Hi, this is actually my second time calling, so quickly —",
    ],
}

LEGACY_FOLLOWUP_QUESTIONS = {
    "area": [
        "Agent: Any particular area you're targeting?",
        "Agent: Do you have a specific locality in mind?",
        "Agent: Is there a neighbourhood you're leaning towards?",
    ],
    "property_type": [
        "Agent: What kind of property did you have in mind?",
        "Agent: And what type of property are we talking about?",
    ],
    "bhk": [
        "Agent: How many bedrooms are you thinking — a 2 BHK, 3 BHK?",
        "Agent: What configuration works for you, BHK-wise?",
        "Agent: And how many bedrooms do you need?",
    ],
    "budget": [
        "Agent: What budget range did you have in mind?",
        "Agent: Do you have a budget figure in mind?",
        "Agent: And roughly what's your budget looking like?",
    ],
    "amenities": [
        "Agent: Any specific amenities you're looking for?",
        "Agent: Is there anything particular you need, like parking or a gym?",
        "Agent: Any must-have facilities?",
    ],
    "furnishing": [
        "Agent: Would you prefer it furnished or unfurnished?",
        "Agent: And furnishing-wise, any preference?",
    ],
}

LEGACY_DEFLECTIONS = {
    "Enthusiastic": [
        "Customer: Oh, haven't decided that part yet, but I'll figure it out soon!",
        "Customer: Not sure yet, honestly, but open to suggestions!",
    ],
    "Neutral": [
        "Customer: Not decided yet, we'll figure that out later.",
        "Customer: No preference there, whatever works.",
    ],
    "Hesitant": [
        "Customer: Um, we haven't really thought about that yet.",
        "Customer: Not sure, we're still figuring things out.",
    ],
    "Frustrated": [
        "Customer: I don't know, can we just move on?",
        "Customer: Not decided, look, can someone just call me back about this?",
    ],
}

LEGACY_CLOSERS = {
    "generic": [
        "Agent: Thanks for sharing all that, our team will get back to you shortly.",
        "Agent: Perfect, I've noted everything down. Someone will follow up with you soon.",
        "Agent: Got it, thank you. We'll be in touch shortly with the next steps.",
    ],
    "schedule_visit": [
        "Agent: Sure, I'll get someone to set up a site visit and confirm a time with you.",
        "Agent: Great, I'll pass this along so our team can arrange a visit at a convenient time.",
    ],
    "callback": [
        "Agent: No problem, I'll make sure someone calls you back at a convenient time.",
        "Agent: Understood, we'll have someone reach out to you soon.",
    ],
    "frustrated": [
        "Agent: I understand, I'm sorry for the trouble — I'll flag this so someone follows up personally.",
        "Agent: I hear you, let me make sure this gets prioritized and someone calls you back soon.",
    ],
}

CLOSER_BANKS = {
    "generic": CLOSERS_GENERIC,
    "schedule_visit": CLOSERS_SCHEDULE_VISIT,
    "callback": CLOSERS_CALLBACK,
    "frustrated": CLOSERS_FRUSTRATED,
}


# --------------------------------------------------------------------------
# Defect 1 — studio/BHK contradiction
# --------------------------------------------------------------------------

def has_studio_bhk_conflict(record: Dict[str, Any]) -> bool:
    """True when the record pairs studio-ness with a bedroom count.

    Two shapes occur in the reviewed pool:
      - property_type "Studio Apartment" with a numeric bhk ("studio
        apartment, 2 BHK");
      - the string "Studio" used as a bhk value, which renders next to an
        unrelated type ("a studio gated community villa").
    Both are contradictions; a studio is carried by PROPERTY_TYPE alone.
    """
    gt = record["ground_truth"]
    if gt["property_type"] == "Studio Apartment" and gt["bhk"] is not None:
        return True
    return gt["bhk"] == "Studio"


def rebuild_record(rng, lookups: LookupTables, record: Dict[str, Any], max_tries: int = 4000) -> Dict[str, Any]:
    """Rebuilds one transcript with the fixed generator, rejection-sampling
    until city, intent and sentiment match the original. Hard-case flags are
    forced, so the repaired pool's composition is unchanged except for the
    defect itself.
    """
    old_gt = record["ground_truth"]
    old_meta = record["metadata"]
    flags = old_meta["flags"]

    forced_type: Optional[str] = old_gt["property_type"]  # None if it never surfaced in the text

    for _ in range(max_tries):
        candidate = build_transcript(
            rng,
            lookups,
            forced_property_type=forced_type,
            force_negation=flags["has_negation"],
            force_code_mixed=flags["is_code_mixed"],
            force_multi_city=flags["is_multi_city"],
            force_telegraphic=flags["is_telegraphic"],
        )
        meta = candidate["metadata"]
        if (
            candidate["ground_truth"]["city"] == old_gt["city"]
            and meta["intent"] == old_meta["intent"]
            and meta["sentiment"] == old_meta["sentiment"]
        ):
            return candidate

    logger.warning(
        "Could not match city/intent/sentiment for %s within %d tries; keeping last candidate",
        record["transcript_id"], max_tries,
    )
    return candidate


# --------------------------------------------------------------------------
# Defect 2 — boilerplate repetition
# --------------------------------------------------------------------------

def _legacy_field_for_question(line: str) -> Optional[str]:
    for field, bank in LEGACY_FOLLOWUP_QUESTIONS.items():
        if line in bank:
            return field
    return None


def _legacy_closer_bank(line: str) -> Optional[str]:
    for bank_name, bank in LEGACY_CLOSERS.items():
        if line in bank:
            return bank_name
    return None


def refresh_boilerplate(rng, record: Dict[str, Any]) -> int:
    """Resamples this transcript's framing lines and non-answers from the
    expanded banks. Returns the number of lines changed.

    Only lines that exactly match a legacy bank entry are touched, so any
    line carrying entity content is left alone by construction. Non-answers
    are drawn without replacement within the transcript.
    """
    sentiment = record["metadata"]["sentiment"]
    lines = record["text"].split("\n")
    changed = 0

    deflection_pool = list(DEFLECTIONS[sentiment])
    rng.shuffle(deflection_pool)

    for i, line in enumerate(lines):
        if line in LEGACY_OPENERS:
            lines[i] = str(rng.choice(OPENERS))
            changed += 1
            continue

        if line in LEGACY_DEFLECTIONS[sentiment]:
            lines[i] = str(deflection_pool.pop())
            changed += 1
            continue

        field = _legacy_field_for_question(line)
        if field is not None:
            lines[i] = str(rng.choice(FOLLOWUP_QUESTIONS[field]))
            changed += 1
            continue

        bank_name = _legacy_closer_bank(line)
        if bank_name is not None:
            lines[i] = str(rng.choice(CLOSER_BANKS[bank_name]))
            changed += 1
            continue

        # Customer's opening framing sits as a prefix on the requirement
        # line, so it is swapped in place and the requirement text after it
        # is preserved exactly. Telegraphic lines have no such prefix.
        if line.startswith("Customer: "):
            body = line[len("Customer: "):]
            for legacy in LEGACY_SENTIMENT_OPENERS[sentiment]:
                if body.startswith(legacy):
                    remainder = body[len(legacy):]
                    lines[i] = "Customer: " + str(rng.choice(SENTIMENT_OPENERS[sentiment])) + remainder
                    changed += 1
                    break

    record["text"] = "\n".join(lines)
    return changed


DOUBLE_ARTICLE_RE = re.compile(r"\ba a\b")


def fix_double_article(record: Dict[str, Any]) -> bool:
    """Collapses "a a 1RK builder floor" -> "a 1RK builder floor".

    The old generator's BHK phrases carried their own article ("a 1RK")
    while the sentence template supplied one too. The generator now emits
    article-free BHK phrases; this cleans up the rendered text left behind
    in the reviewed pool. Entity spans are unaffected — only a duplicated
    determiner in front of them is removed.
    """
    fixed = DOUBLE_ARTICLE_RE.sub("a", record["text"])
    if fixed != record["text"]:
        record["text"] = fixed
        return True
    return False


# --------------------------------------------------------------------------

def load_pool(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def repair(pool: List[Dict[str, Any]], seed: int = REPAIR_SEED) -> Dict[str, int]:
    rng = default_rng(seed)
    lookups = LookupTables()

    rebuilt = 0
    for idx, record in enumerate(pool):
        if has_studio_bhk_conflict(record):
            new = rebuild_record(rng, lookups, record)
            pool[idx] = {
                "transcript_id": record["transcript_id"],
                "text": new["text"],
                "ground_truth": new["ground_truth"],
                "metadata": new["metadata"],
            }
            rebuilt += 1
            logger.info("Rebuilt %s (studio/BHK conflict)", record["transcript_id"])

    touched = 0
    articles_fixed = 0
    for record in pool:
        if refresh_boilerplate(rng, record):
            touched += 1
        if fix_double_article(record):
            articles_fixed += 1

    return {"rebuilt": rebuilt, "boilerplate_refreshed": touched, "double_articles_fixed": articles_fixed}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    jsonl_path = OUTPUT_DIR / "synthetic_transcripts.jsonl"
    pool = load_pool(jsonl_path)
    logger.info("Loaded %d transcripts from %s", len(pool), jsonl_path)

    stats = repair(pool)
    logger.info("Rebuilt %d transcripts; refreshed boilerplate in %d; fixed %d double articles",
                stats["rebuilt"], stats["boilerplate_refreshed"], stats["double_articles_fixed"])

    # write_pool re-derives transcript_ids from position; the pool order is
    # unchanged by the repair, so ids are stable.
    write_pool([{"text": r["text"], "ground_truth": r["ground_truth"], "metadata": r["metadata"]} for r in pool])

    summary = summarize(pool)
    logger.info("Repaired pool: %d transcripts, cities=%s", summary["total"], summary["city_counts"])


if __name__ == "__main__":
    main()
