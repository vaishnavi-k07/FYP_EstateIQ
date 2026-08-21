"""Sentiment corpus — 1216 transcripts, emotion-rich and half of them messy.

Separate from ``output/synthetic_transcripts.jsonl`` in every respect: separate
pools (:mod:`pilot_emotion_pools`), separate output file, separate split. The
Phase 3 NER dataset and the promoted intent head read the old file and are not
touched by anything here.

What this corpus is for
-----------------------
The v3 sentiment head scores 0.713 macro-F1 on phrasings it saw and 0.478 on
phrasings it did not, and misfiles 42% of held-out Enthusiastic transcripts as
Frustrated. The cause is that v3 *names* the tone, so the model memorises a
handful of affect adjectives and has nothing left when they are withheld.

A 288-transcript pilot removed the affect vocabulary and could not answer
whether that helps: it trained on 136 transcripts and never left the chance
floor (0.245 in-distribution against a 0.25 baseline), which makes its 0.049
gap meaningless. The diagnosis was breadth, not volume — the holdout asks the
model to read a tone in an idea it has never met, and that takes exposure to
many different ideas. So the idea pool was widened first, from 18 to 38 ideas
per tone, and only then was the corpus scaled to 8 transcripts per idea.

Four design properties, each one a lever on a specific failure:

*Emotion is situational.* Tone comes from ``pilot_emotion_pools``, where no
phrase contains an affect word. See that module for why.

*One idea per transcript.* Every tone-bearing sentence in a transcript is a
surface form of the same idea. That is what makes the holdout clean: a
transcript either uses a withheld idea or it does not, so none are dropped as
"mixed". In the v3 run 255 of 1000 transcripts were unusable for that reason,
which is why its diagnostic trained on half the corpus.

*Half messy, and messy in the same proportion for every tone.* Disfluencies,
false starts, self-corrections, rambles, trailing off, mid-sentence code
switching and dropped grammar. Exactly half the transcripts built from each
idea are messy, so the messy rate is 50% within every tone, every idea and
every register by construction rather than by sampling — messiness cannot
become a cue for any tone. :func:`summarize` reports the contingency table so
this is checked rather than asserted.

*Messiness never rewrites a tone-bearing sentence.* Every disfluency lands on
requirement text, on the intent phrase, or between sentences. Two reasons: the
holdout diagnostic locates the emotion by substring match, which an inserted
"um" in the middle would defeat; and a messy transcript should test whether the
model can find the tone in noisier surroundings, not whether it can read a
corrupted expression of it. The invariant is asserted for every transcript
before it is written.

Requirement content (city, area, property type, BHK, budget, amenities,
furnishing) reuses the CSV-backed helpers of :mod:`synthetic_transcripts`, so
no location or property value is hardcoded here either. Intent phrasing is the
shared ``phrase_pools`` intent pool, dealt independently of tone so the two
labels stay uncorrelated.

Run:
    python -m data_generation.sentiment_pilot
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_generation import pilot_emotion_pools as _pools  # noqa: E402
from data_generation.synthetic_transcripts import (  # noqa: E402
    CITY_WEIGHTS,
    FURNISHING_OPTIONS,
    NO_BHK_CATEGORIES,
    NO_BHK_PROPERTY_TYPES,
    TARGET_CITIES,
    LookupTables,
    build_area_weights,
    cap_first,
    generate_bhk,
    generate_budget,
    indefinite_article,
    join_amenities,
    pick_amenities,
    render_property_type,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42

# 4 tones x 38 ideas x 8 transcripts = 1216, exactly half of each idea messy.
#
# Eight per idea rather than four is the *scale* half of the widen-then-scale
# fix; the widening from 18 to 38 ideas per tone is the half that matters more.
# The first pilot ran 4 per idea over 18 ideas, its diagnostic trained on 136
# transcripts, and it never left the chance floor. This yields ~800
# train-eligible transcripts and ~560 training rows.
TRANSCRIPTS_PER_IDEA = 8

# The 288-transcript pilot, kept reproducible so its numbers can be regenerated.
PILOT_PER_IDEA = 4

# Neutral transcripts carrying no emotional expression at all — see
# :func:`build_tone_free_transcript`. Sized at 96 against 1216 tone-bearing
# rows (~7%): enough that "no affect stated" is a trained signal rather than a
# gap, small enough that Neutral does not dominate the corpus. It does make the
# corpus mildly Neutral-heavy (400 vs 304), which macro-F1 and the recomputed
# majority baseline both account for.
TONE_FREE_NEUTRAL = 96

SENTIMENT_CLASSES = _pools.SENTIMENT_CLASSES
INTENT_CLASSES = _pools.INTENT_CLASSES


# --------------------------------------------------------------------------- #
# Agent side. Nothing here varies with tone: a tone-dependent agent line would
# put the label in the transcript for the template baseline to read off, and
# in production the agent's words come from Vapi's prompt, not from us.
# --------------------------------------------------------------------------- #

AGENT_OPENERS: Tuple[str, ...] = (
    "Agent: Hi, thanks for calling in. What kind of property are you looking for?",
    "Agent: Hello, you've reached the sales desk. How can I help today?",
    "Agent: Hi there, thanks for getting in touch. What can I find for you?",
    "Agent: Good afternoon. Tell me what you're after and I'll note it down.",
    "Agent: Hello. Let's start with what you're looking for.",
    "Agent: Hi, thanks for your time. Walk me through the requirement.",
)

AGENT_FOLLOWUPS: Dict[str, Tuple[str, ...]] = {
    "area": (
        "Agent: Any particular area you're targeting?",
        "Agent: Do you have a locality in mind?",
        "Agent: Which part of the city suits you?",
    ),
    "bhk": (
        "Agent: And how many bedrooms are we looking at?",
        "Agent: What configuration did you have in mind?",
        "Agent: How big does it need to be?",
    ),
    "budget": (
        "Agent: What budget are we working with?",
        "Agent: And what range are you comfortable in?",
        "Agent: Do you have a figure in mind?",
    ),
    "amenities": (
        "Agent: Anything specific you'd want in the building?",
        "Agent: Any amenities that matter to you?",
        "Agent: Is there anything the complex must have?",
    ),
    "furnishing": (
        "Agent: Furnished or unfurnished?",
        "Agent: Would you want it furnished?",
        "Agent: Any preference on furnishing?",
    ),
    "possession": (
        "Agent: When would you want possession?",
        "Agent: Any timeline on your side?",
        "Agent: When are you hoping to move?",
    ),
}

AGENT_CLOSERS: Dict[str, Tuple[str, ...]] = {
    "Schedule Visit": (
        "Agent: Noted. I'll set up a site visit and confirm the slot.",
        "Agent: Understood, I'll arrange the viewing and text you the details.",
    ),
    "Request Callback": (
        "Agent: Noted, someone from our side will call you back.",
        "Agent: Understood, I'll have the team ring you.",
    ),
    "_default": (
        "Agent: Noted, I've got all of that down. I'll follow up with options.",
        "Agent: Understood. I'll take this forward and be in touch.",
        "Agent: Thanks, that's everything I need for now.",
    ),
}


# --------------------------------------------------------------------------- #
# Requirement phrasing. Deliberately plain and shared across tones: the tone
# lives in the emotion expression, and any tone colour here would confound it.
# --------------------------------------------------------------------------- #

REQUIREMENT_FRAMES: Tuple[str, ...] = (
    "We're looking for {det} {bhk}{ptype} in {where}.",
    "We need {det} {bhk}{ptype} in {where}.",
    "It's {det} {bhk}{ptype} in {where} that we're after.",
    "The requirement is {det} {bhk}{ptype} in {where}.",
)
REQUIREMENT_FRAMES_NO_BHK: Tuple[str, ...] = (
    "We're looking for {det} {ptype} in {where}.",
    "We need {det} {ptype} in {where}.",
    "It's {det} {ptype} in {where} that we're after.",
    "The requirement is {det} {ptype} in {where}.",
)
BUDGET_FRAMES: Tuple[str, ...] = (
    "Budget is around {budget}.",
    "We're working with about {budget}.",
    "The figure we have in mind is {budget}.",
    "Somewhere near {budget} is the range.",
)
AMENITY_FRAMES: Tuple[str, ...] = (
    "It should have {amenities}.",
    "We'd want {amenities}.",
    "{amenities_cap} would matter to us.",
)
# Administrative asks that carry no affect at all. Used only by the tone-free
# Neutral rows, and deliberately flat: a caller can request a brochure without
# being pleased or annoyed about it, and the model needs examples of that.
ADMIN_LINES: Tuple[str, ...] = (
    "Can you send me the details over email.",
    "Please share the brochure when you get a chance.",
    "What documents would you need from me?",
    "I'd like the price sheet and the floor plan.",
    "Send across whatever paperwork is involved.",
    "Just noting down options at this stage.",
    "I'm collecting details for a shortlist.",
    "Put me on the list for updates.",
)

# Plain factual non-answers for the tone-free rows. The tone-bearing
# deflections live in the pools; these must stay affect-free.
FACTUAL_NON_ANSWERS: Tuple[str, ...] = (
    "That's all I have at the moment.",
    "Nothing further to add on that.",
    "That covers it.",
    "No other requirements to note.",
)

FURNISHING_FRAMES: Tuple[str, ...] = (
    "{furnishing_cap} would suit us.",
    "We'd want it {furnishing_lower}.",
    "Preferably {furnishing_lower}.",
)


# --------------------------------------------------------------------------- #
# Messiness. Every operation here is tone-blind: it never reads the label, and
# the caller deals it out in a fixed 50% ratio per idea.
# --------------------------------------------------------------------------- #

FILLERS: Tuple[str, ...] = (
    "Um,", "Uh,", "Erm,", "So, um,", "Yeah so", "Right, um,", "Hmm,", "Like,",
)
ASIDES: Tuple[str, ...] = (
    "Sorry, the signal's a bit patchy here.",
    "One second, someone's at the door.",
    "I'm walking to the car, hope you can hear me.",
    "Hang on, let me move somewhere quieter.",
    "Sorry, that was my phone buzzing.",
    "Can you hear me alright?",
    "There's roadworks outside, apologies for the noise.",
)
TRAIL_OFFS: Tuple[str, ...] = (
    "and, yeah...", "or something like that...", "so... yeah.",
    "anyway...", "you know how it is...", "that kind of thing...",
)
SWITCH_TOKENS: Tuple[str, ...] = (
    "matlab", "yaani", "haan", "achha", "arre", "bas",
)
# Words a caller drops or slurs in casual speech. Applied to requirement text
# only, so no emotion expression is ever rewritten.
CASUAL_SUBS: Tuple[Tuple[str, str], ...] = (
    (r"\bWe're looking for\b", "We want"),
    (r"\bWe need\b", "We want"),
    (r"\bwe're after\b", "we want"),
    (r"\bIt should have\b", "Should have"),
    (r"\bWe'd want\b", "Want"),
    (r"\bBudget is around\b", "Budget around"),
    (r"\bgoing to\b", "gonna"),
    (r"\bwant to\b", "wanna"),
    (r"\bThe requirement is\b", "Requirement is"),
)

MESSY_OPS: Tuple[str, ...] = (
    "filler", "false_start", "self_correction", "ramble",
    "trail_off", "code_switch", "casual_grammar",
)
MESSY_OPS_MIN = 3
MESSY_OPS_MAX = 5


class Segment:
    """A piece of a customer turn, flagged for whether messiness may touch it.

    ``protected`` marks the tone-bearing sentences and the intent phrase. Those
    must survive verbatim: the holdout diagnostic finds them by substring match,
    and a rewritten expression would silently disappear from the partition.
    """

    __slots__ = ("text", "protected", "kind")

    def __init__(self, text: str, protected: bool = False, kind: str = "requirement") -> None:
        self.text = text
        self.protected = protected
        self.kind = kind

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Segment({self.kind!r}, protected={self.protected}, {self.text!r})"


def _apply_filler(rng, segments: List[Segment]) -> None:
    targets = [s for s in segments if not s.protected]
    if not targets:
        return
    seg = targets[int(rng.integers(len(targets)))]
    seg.text = f"{rng.choice(FILLERS)} {seg.text[0].lower() + seg.text[1:]}"


def _apply_false_start(rng, segments: List[Segment]) -> None:
    targets = [s for s in segments if not s.protected and len(s.text.split()) > 3]
    if not targets:
        return
    seg = targets[int(rng.integers(len(targets)))]
    words = seg.text.split()
    stem = " ".join(words[: 1 if rng.random() < 0.5 else 2])
    seg.text = f"{stem}— {stem[0].lower() + stem[1:]} {' '.join(words[len(stem.split()):])}"


def _apply_self_correction(rng, segments: List[Segment], corrections: Sequence[Tuple[str, str]]) -> None:
    """Rewrites a stated value as a correction to a wrong one: "2 BHK, no wait, 3 BHK".

    The *final* value is always the ground truth, so a self-corrected transcript
    stays honestly labelled for the NLP module even though this corpus is only
    used by the sentiment head.
    """
    usable = [
        (seg, right, wrong)
        for seg in segments
        if not seg.protected
        for right, wrong in corrections
        if right and right in seg.text
    ]
    if not usable:
        return
    seg, right, wrong = usable[int(rng.integers(len(usable)))]
    phrasing = str(rng.choice(["no wait,", "sorry, no,", "actually no,", "no, hang on,"]))
    seg.text = seg.text.replace(right, f"{wrong}, {phrasing} {right}", 1)


def _apply_ramble(rng, segments: List[Segment]) -> None:
    aside = Segment(str(rng.choice(ASIDES)), protected=False, kind="aside")
    positions = [i for i, s in enumerate(segments) if not s.protected]
    index = positions[int(rng.integers(len(positions)))] if positions else len(segments)
    segments.insert(index, aside)


def _apply_trail_off(rng, segments: List[Segment]) -> None:
    targets = [s for s in segments if not s.protected]
    if not targets:
        return
    seg = targets[int(rng.integers(len(targets)))]
    seg.text = f"{seg.text.rstrip('.')}, {rng.choice(TRAIL_OFFS)}"


def _apply_code_switch(rng, segments: List[Segment]) -> None:
    """Drops a discourse particle mid-sentence, the way a bilingual caller does.

    Neutral by construction: the particles carry no affect and the operation is
    dealt to all four tones at the same rate, so romanised-Hindi tokens in a
    messy transcript say nothing about its label.
    """
    targets = [s for s in segments if not s.protected and len(s.text.split()) > 4]
    if not targets:
        return
    seg = targets[int(rng.integers(len(targets)))]
    words = seg.text.split()
    at = int(rng.integers(2, len(words) - 1))
    words.insert(at, str(rng.choice(SWITCH_TOKENS)))
    seg.text = " ".join(words)


def _apply_casual_grammar(rng, segments: List[Segment]) -> None:
    for seg in segments:
        if seg.protected:
            continue
        for pattern, replacement in CASUAL_SUBS:
            if re.search(pattern, seg.text):
                seg.text = re.sub(pattern, replacement, seg.text, count=1)
                break


def make_messy(
    rng, turns: List[List[Segment]], corrections: Sequence[Tuple[str, str]]
) -> List[str]:
    """Applies 3-5 distinct disfluency operations across the customer's turns.

    The operation set, its size and the turn it lands on are all drawn without
    reference to the tone, which is what makes "messy" uninformative about the
    label.
    """
    flat = [seg for turn in turns for seg in turn]
    n_ops = int(rng.integers(MESSY_OPS_MIN, MESSY_OPS_MAX + 1))
    ops = list(rng.choice(MESSY_OPS, size=n_ops, replace=False))
    applied: List[str] = []
    for op in ops:
        # Operate on the whole transcript's unprotected segments; _apply_ramble
        # needs the owning turn, so it is dispatched per turn instead.
        if op == "ramble":
            turn = turns[int(rng.integers(len(turns)))]
            _apply_ramble(rng, turn)
        elif op == "filler":
            _apply_filler(rng, flat)
        elif op == "false_start":
            _apply_false_start(rng, flat)
        elif op == "self_correction":
            _apply_self_correction(rng, flat, corrections)
        elif op == "trail_off":
            _apply_trail_off(rng, flat)
        elif op == "code_switch":
            _apply_code_switch(rng, flat)
        elif op == "casual_grammar":
            _apply_casual_grammar(rng, flat)
        applied.append(str(op))
    return applied


# --------------------------------------------------------------------------- #
# Requirement content
# --------------------------------------------------------------------------- #

def _wrong_value(rng, right: str, pool: Sequence[str]) -> str:
    """A plausible wrong value for a self-correction, never equal to the right one."""
    options = [p for p in pool if p != right]
    return str(rng.choice(options)) if options else right


def build_requirement(
    rng, lookups: LookupTables, completeness: str
) -> Tuple[List[Segment], Dict[str, Any], List[Tuple[str, str]]]:
    """Returns (segments, ground_truth, self-correction candidates)."""
    city = str(rng.choice(TARGET_CITIES, p=CITY_WEIGHTS))
    areas = lookups.areas_by_city[city]
    weights = build_area_weights(rng, areas)
    area = str(rng.choice(areas, p=[w / sum(weights) for w in weights]))
    property_type = str(rng.choice(lookups.non_rare_property_types))
    category = lookups.property_type_category[property_type]
    surface = render_property_type(rng, lookups, property_type)

    if completeness == "full":
        reveal = {"area": True, "budget": True, "amenities": True, "furnishing": rng.random() < 0.7}
    elif completeness == "partial":
        reveal = {
            "area": rng.random() < 0.7,
            "budget": rng.random() < 0.7,
            "amenities": rng.random() < 0.5,
            "furnishing": rng.random() < 0.3,
        }
    else:
        reveal = {
            "area": rng.random() < 0.4,
            "budget": rng.random() < 0.3,
            "amenities": False,
            "furnishing": False,
        }

    wants_bhk = not (
        category in NO_BHK_CATEGORIES
        or property_type in NO_BHK_PROPERTY_TYPES
        or "Plot" in property_type
        or "Land" in property_type
    )
    bhk_value = bhk_phrase = None
    if wants_bhk:
        bhk_value, bhk_phrase, _ = generate_bhk(rng)

    where = f"{area}, {city}" if reveal["area"] else city
    if bhk_phrase:
        frame = str(rng.choice(REQUIREMENT_FRAMES))
        head = f"{bhk_phrase} {surface}"
        text = frame.format(det=indefinite_article(bhk_phrase), bhk=f"{bhk_phrase} ", ptype=surface, where=where)
    else:
        frame = str(rng.choice(REQUIREMENT_FRAMES_NO_BHK))
        head = surface
        text = frame.format(det=indefinite_article(surface), ptype=surface, where=where)
    segments = [Segment(text, kind="requirement")]
    corrections: List[Tuple[str, str]] = []
    if bhk_phrase:
        corrections.append((bhk_phrase, _wrong_value(rng, bhk_phrase, ["1 BHK", "2 BHK", "3 BHK", "4 BHK"])))
    if reveal["area"]:
        corrections.append((area, _wrong_value(rng, area, areas)))

    budget_lakhs = budget_text = None
    if reveal["budget"]:
        budget_lakhs, budget_text, _ = generate_budget(rng, category)
        segments.append(
            Segment(str(rng.choice(BUDGET_FRAMES)).format(budget=budget_text), kind="budget")
        )

    amenities: List[str] = []
    if reveal["amenities"]:
        amenities = pick_amenities(rng, lookups, completeness)
        if amenities:
            joined = join_amenities(rng, amenities)
            segments.append(
                Segment(
                    str(rng.choice(AMENITY_FRAMES)).format(
                        amenities=joined, amenities_cap=cap_first(joined)
                    ),
                    kind="amenities",
                )
            )

    furnishing = None
    if reveal["furnishing"]:
        furnishing = str(rng.choice(FURNISHING_OPTIONS))
        segments.append(
            Segment(
                str(rng.choice(FURNISHING_FRAMES)).format(
                    furnishing_cap=furnishing, furnishing_lower=furnishing.lower()
                ),
                kind="furnishing",
            )
        )

    ground_truth = {
        "city": city,
        "area": area if reveal["area"] else None,
        "property_type": property_type,
        "category": category,
        "bhk": bhk_value,
        "budget_lakhs": round(budget_lakhs, 2) if budget_lakhs is not None else None,
        "budget_text": budget_text,
        "amenities": amenities,
        "furnishing": furnishing,
    }
    _ = head
    return segments, ground_truth, corrections


def _answer_for(rng, field: str, ground_truth: Dict[str, Any]) -> Optional[Segment]:
    """A plain answer to an agent follow-up, used when the tone is not a deflection."""
    if field == "possession":
        return Segment(
            str(rng.choice([
                "Within the next few months.",
                "No fixed date on that.",
                "Sometime after the monsoon.",
                "Whenever it's ready.",
            ])),
            kind="answer",
        )
    if field == "furnishing" and ground_truth["furnishing"]:
        return Segment(f"{ground_truth['furnishing']}, ideally.", kind="answer")
    if field == "amenities" and ground_truth["amenities"]:
        return Segment(f"{cap_first(join_amenities(rng, ground_truth['amenities']))}, mainly.", kind="answer")
    if field == "budget" and ground_truth["budget_text"]:
        return Segment(f"About {ground_truth['budget_text']}.", kind="answer")
    if field == "area" and ground_truth["area"]:
        return Segment(f"{ground_truth['area']}, ideally.", kind="answer")
    if field == "bhk" and ground_truth["bhk"]:
        return Segment(f"{ground_truth['bhk']} BHK.", kind="answer")
    return None


# --------------------------------------------------------------------------- #
# Transcript assembly
# --------------------------------------------------------------------------- #

def build_transcript(
    rng,
    lookups: LookupTables,
    tone: str,
    idea: str,
    intent: str,
    completeness: str,
    messy: bool,
) -> Dict[str, Any]:
    family = _pools.idea_family(idea)
    variants = list(_pools.ALL_IDEAS[tone][idea])
    rng.shuffle(variants)
    n_expressions = 1 if rng.random() < 0.5 else 2
    expressions = variants[:n_expressions]

    intent_phrase = str(rng.choice(_pools.INTENT_PHRASES[intent]))
    req_segments, ground_truth, corrections = build_requirement(rng, lookups, completeness)

    first_turn: List[Segment] = []
    later_turn: List[Segment] = []

    if family == "opener":
        first_turn.append(Segment(expressions[0], protected=True, kind="emotion"))
        first_turn.extend(req_segments)
        first_turn.append(Segment(cap_first(intent_phrase) + ".", protected=True, kind="intent"))
        if len(expressions) > 1:
            later_turn.append(Segment(expressions[1], protected=True, kind="emotion"))
    else:
        # Deflection ideas carry no tone in the opening turn at all — the
        # emotion only surfaces when the agent asks something. Harder, and dealt
        # to every tone in the same proportion.
        first_turn.extend(req_segments)
        first_turn.append(Segment(cap_first(intent_phrase) + ".", protected=True, kind="intent"))
        later_turn.append(Segment(expressions[0], protected=True, kind="emotion"))
        if len(expressions) > 1:
            later_turn.append(Segment(expressions[1], protected=True, kind="emotion"))

    followup_field = str(rng.choice(list(AGENT_FOLLOWUPS)))
    if family == "opener" and not later_turn:
        answer = _answer_for(rng, followup_field, ground_truth)
        later_turn.append(answer or Segment("That's about it, really.", kind="answer"))

    turns = [first_turn, later_turn]
    applied_ops: List[str] = []
    if messy:
        applied_ops = make_messy(rng, turns, corrections)

    lines = [
        str(rng.choice(AGENT_OPENERS)),
        "Customer: " + " ".join(s.text for s in turns[0]),
        str(rng.choice(AGENT_FOLLOWUPS[followup_field])),
        "Customer: " + " ".join(s.text for s in turns[1]),
        str(rng.choice(AGENT_CLOSERS.get(intent, AGENT_CLOSERS["_default"]))),
    ]
    text = "\n".join(lines)

    # The invariant the diagnostic depends on: every emotion expression is still
    # findable verbatim. If a messy operation ever leaks onto a protected
    # segment this fires here rather than silently skewing the partition.
    normalized = _normalize(text)
    for expression in expressions:
        if _normalize(expression).strip() not in normalized:
            raise AssertionError(f"messiness corrupted a tone expression: {expression!r}")

    return {
        "text": text,
        "ground_truth": ground_truth,
        "metadata": {
            "intent": intent,
            "sentiment": tone,
            "completeness": completeness,
            "style": "messy" if messy else "clean",
            "emotion_idea": idea,
            "emotion_family": family,
            "emotion_register": _pools.phrase_register(idea),
            "emotion_expressions": expressions,
            "messy_ops": applied_ops,
            "flags": {
                # The pilot has no telegraphic transcripts by design: a caller
                # who says only "Pune. 2 BHK." expresses no tone, so the row
                # would be an unlearnable input with a confident label — the
                # reason data.load_corpus filters them out of the v3 corpus.
                "is_telegraphic": False,
                "is_code_mixed": _pools.phrase_register(idea) == "code-mixed",
            },
        },
    }


def build_tone_free_transcript(
    rng,
    lookups: LookupTables,
    intent: str,
    completeness: str,
    messy: bool,
) -> Dict[str, Any]:
    """A Neutral transcript with no emotional expression anywhere in it.

    Every other transcript in this corpus carries exactly one emotional idea, so
    "requirement text and nothing else" was a shape the model had never seen.
    The spot check caught what that costs: the full-corpus hybrid read the plain
    factual line *"we're looking for a 3 BHK in Baner, budget around 90 lakhs"*
    as **Enthusiastic at 0.886**. Nothing in the training data said that absence
    of affect is itself Neutral, so the decision boundary over unadorned
    requirement text was unconstrained.

    These rows constrain it. They are correctly labelled rather than noise —
    which is what separates them from the v3 telegraphic transcripts that
    ``data.load_corpus`` discards. A telegraphic row was stamped with a random
    tone its own words never expressed, so its label was unlearnable. Here the
    label follows from the text: no affect stated, therefore Neutral.

    Two consequences worth knowing:

    *They are never held out.* The phrase-holdout partition works by finding a
    withheld phrase in the text, and there is no phrase here to withhold. They
    are always training-eligible and never reach the unseen-idea test set —
    correctly, since they express no idea to be unseen.

    *They carry ``emotion_register: "none"``.* The English/code-mixed breakdown
    reports on the two real registers, and folding tone-free rows into
    "english" would inflate that side with rows that contain no tone at all.
    """
    intent_phrase = str(rng.choice(_pools.INTENT_PHRASES[intent]))
    req_segments, ground_truth, corrections = build_requirement(rng, lookups, completeness)

    first_turn: List[Segment] = list(req_segments)
    if rng.random() < 0.35:
        # An administrative request instead of a bare requirement. Same absence
        # of affect, different sentence shape, so the model does not learn
        # "Neutral" as a single template.
        first_turn.insert(0, Segment(str(rng.choice(ADMIN_LINES)), kind="admin"))
    first_turn.append(Segment(cap_first(intent_phrase) + ".", protected=True, kind="intent"))

    followup_field = str(rng.choice(list(AGENT_FOLLOWUPS)))
    answer = _answer_for(rng, followup_field, ground_truth)
    later_turn: List[Segment] = [
        answer or Segment(str(rng.choice(FACTUAL_NON_ANSWERS)), kind="answer")
    ]

    turns = [first_turn, later_turn]
    applied_ops: List[str] = []
    if messy:
        applied_ops = make_messy(rng, turns, corrections)

    lines = [
        str(rng.choice(AGENT_OPENERS)),
        "Customer: " + " ".join(s.text for s in turns[0]),
        str(rng.choice(AGENT_FOLLOWUPS[followup_field])),
        "Customer: " + " ".join(s.text for s in turns[1]),
        str(rng.choice(AGENT_CLOSERS.get(intent, AGENT_CLOSERS["_default"]))),
    ]

    return {
        "text": "\n".join(lines),
        "ground_truth": ground_truth,
        "metadata": {
            "intent": intent,
            "sentiment": "Neutral",
            "completeness": completeness,
            "style": "messy" if messy else "clean",
            "emotion_idea": None,
            "emotion_family": "none",
            "emotion_register": "none",
            "emotion_expressions": [],
            "messy_ops": applied_ops,
            "flags": {
                "is_telegraphic": False,
                "is_code_mixed": False,
                "is_tone_free": True,
            },
        },
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #

def _plan(rng, per_idea: int = TRANSCRIPTS_PER_IDEA) -> List[Tuple[str, str, bool]]:
    """(tone, idea, messy) for every transcript, balanced by construction.

    Each idea gets ``TRANSCRIPTS_PER_IDEA`` transcripts of which exactly
    ``MESSY_PER_IDEA`` are messy. Because every tone owns the same number of
    ideas, this makes the corpus exactly balanced on tone, exactly 50% messy,
    and 50% messy *within* every tone, family and register at once — no
    sampling, so no drift for the diagnostic to mistake for signal.
    """
    if per_idea % 2:
        raise SystemExit(
            "transcripts-per-idea must be even so every idea splits 50/50 "
            f"clean/messy; got {per_idea}."
        )
    plan: List[Tuple[str, str, bool]] = []
    for tone in SENTIMENT_CLASSES:
        for idea in _pools.ALL_IDEAS[tone]:
            flags = [True] * (per_idea // 2) + [False] * (per_idea // 2)
            plan.extend((tone, idea, messy) for messy in flags)
    rng.shuffle(plan)
    return plan


def _dealt(rng, values: Sequence[str], n: int) -> List[str]:
    """A shuffled deck that uses every value equally, cycling to length ``n``.

    Dealt independently of tone, so intent and completeness stay uncorrelated
    with the label under test.
    """
    deck = [values[i % len(values)] for i in range(n)]
    rng.shuffle(deck)
    return deck


def generate(
    seed: int = SEED,
    per_idea: int = TRANSCRIPTS_PER_IDEA,
    tone_free: int = TONE_FREE_NEUTRAL,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    lookups = LookupTables()
    plan = _plan(rng, per_idea)
    intents = _dealt(rng, list(INTENT_CLASSES), len(plan))
    completeness = _dealt(rng, ["full", "partial", "vague"], len(plan))

    records: List[Dict[str, Any]] = []
    for index, ((tone, idea, messy), intent, comp) in enumerate(zip(plan, intents, completeness), 1):
        record = build_transcript(rng, lookups, tone, idea, intent, comp, messy)
        record = {"transcript_id": f"sp_{index:05d}", **record}
        records.append(record)

    if tone_free:
        if tone_free % 2:
            raise SystemExit(f"tone-free count must be even; got {tone_free}.")
        # Half messy, like every other slice of this corpus, so the clean/messy
        # axis stays uninformative about the label even after the addition.
        messy_flags = [True] * (tone_free // 2) + [False] * (tone_free // 2)
        rng.shuffle(messy_flags)
        tf_intents = _dealt(rng, list(INTENT_CLASSES), tone_free)
        tf_completeness = _dealt(rng, ["full", "partial", "vague"], tone_free)
        start = len(records)
        for offset, (messy, intent, comp) in enumerate(
            zip(messy_flags, tf_intents, tf_completeness), start=1
        ):
            record = build_tone_free_transcript(rng, lookups, intent, comp, messy)
            records.append({"transcript_id": f"sp_{start + offset:05d}", **record})
    return records


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts, plus the checks that make the pilot's claims verifiable."""
    meta = [r["metadata"] for r in records]
    by_tone = collections.Counter(m["sentiment"] for m in meta)
    messy_by_tone = collections.Counter(m["sentiment"] for m in meta if m["style"] == "messy")
    tone_intent = collections.Counter((m["sentiment"], m["intent"]) for m in meta)

    lengths = [len(r["text"].split()) for r in records]
    return {
        "n": len(records),
        "sentiment_counts": dict(sorted(by_tone.items())),
        "intent_counts": dict(sorted(collections.Counter(m["intent"] for m in meta).items())),
        "style_counts": dict(sorted(collections.Counter(m["style"] for m in meta).items())),
        # The property the whole clean/messy question rests on. Every tone must
        # read exactly 0.5; anything else and "messy" is a cue for a label.
        "messy_rate_by_tone": {
            tone: round(messy_by_tone[tone] / by_tone[tone], 4) for tone in sorted(by_tone)
        },
        "messy_rate_by_family": {
            family: round(
                sum(1 for m in meta if m["emotion_family"] == family and m["style"] == "messy")
                / max(sum(1 for m in meta if m["emotion_family"] == family), 1),
                4,
            )
            for family in ("opener", "deflection")
        },
        "messy_rate_by_register": {
            register: round(
                sum(1 for m in meta if m["emotion_register"] == register and m["style"] == "messy")
                / max(sum(1 for m in meta if m["emotion_register"] == register), 1),
                4,
            )
            for register in ("english", "code-mixed")
        },
        "tone_x_intent": {f"{t}|{i}": c for (t, i), c in sorted(tone_intent.items())},
        "family_counts": dict(
            sorted(collections.Counter(m["emotion_family"] for m in meta).items())
        ),
        "register_counts": dict(
            sorted(collections.Counter(m["emotion_register"] for m in meta).items())
        ),
        "tone_free_neutral": sum(
            1 for m in meta if m["flags"].get("is_tone_free")
        ),
        "ideas_used": len({(m["sentiment"], m["emotion_idea"]) for m in meta}),
        "transcripts_per_idea": dict(
            sorted(
                collections.Counter(
                    collections.Counter(
                        (m["sentiment"], m["emotion_idea"]) for m in meta
                    ).values()
                ).items()
            )
        ),
        "messy_op_counts": dict(
            sorted(collections.Counter(op for m in meta for op in m["messy_ops"]).items())
        ),
        "words": {
            "mean": round(sum(lengths) / len(lengths), 1),
            "min": min(lengths),
            "max": max(lengths),
        },
        "pool_checks": _pools.pool_summary(),
    }


def write(
    records: Sequence[Dict[str, Any]],
    output_dir: Path = OUTPUT_DIR,
    name: str = "sentiment_v5",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / f"{name}.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records)
    (output_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %d transcripts to %s", len(records), corpus_path)
    logger.info("Sentiment: %s", summary["sentiment_counts"])
    logger.info("Messy rate by tone: %s", summary["messy_rate_by_tone"])
    logger.info("Words: %s", summary["words"])
    checks = summary["pool_checks"]
    logger.info(
        "Pool checks — affect leaks %d, strandable tokens %d, substring collisions %d",
        len(checks["affect_word_leaks"]),
        len(checks["strandable_tokens"]),
        len(checks["substring_collisions"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--transcripts-per-idea",
        type=int,
        default=TRANSCRIPTS_PER_IDEA,
        help="Must be even, so each idea splits exactly 50/50 clean/messy. "
             f"{TRANSCRIPTS_PER_IDEA} gives the full corpus; {PILOT_PER_IDEA} "
             "reproduces the original 288-transcript pilot.",
    )
    parser.add_argument(
        "--tone-free",
        type=int,
        default=TONE_FREE_NEUTRAL,
        help="Neutral transcripts with no emotional expression at all. Must be even. "
             "0 reproduces the v4 corpus, which lacked them.",
    )
    parser.add_argument(
        "--name",
        default="sentiment_v5",
        help="Basename for <name>.jsonl and <name>_summary.json in the output dir.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    write(
        generate(args.seed, args.transcripts_per_idea, args.tone_free),
        args.output_dir,
        args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
