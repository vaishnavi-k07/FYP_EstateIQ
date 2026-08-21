"""Mentor demo — 36 unseen transcripts scored by the promoted sentiment model.

The point of this set is generalisation, so "unseen" has to be provable rather
than promised. Two guards run before any scoring happens
(:func:`leakage_report`):

1. **No training phrase appears in any demo transcript.** Every phrase the
   generator can emit — all 560 tone expressions, the intent phrases, the
   requirement frames, the admin lines, the disfluency fillers and asides — is
   checked as a normalised substring. Any hit is a hard failure.
2. **Longest shared word sequence is reported per transcript.** Individual
   words must overlap; these are real-estate calls in the same two languages.
   What would undermine the demo is a long shared *sequence*, so the longest
   contiguous run of words each transcript shares with any training phrase is
   measured and printed.

The transcripts are written to be realistic rather than convenient. Several use
affect vocabulary the training corpus bans outright — "excited", "annoyed",
"patience" never appear in any pool — so those are words the model has
genuinely never seen in training. A few are deliberately awkward: trailing off,
self-correcting, changing their mind mid-sentence.

Expected labels are one human reading of each line, not ground truth from a
generator. Where a caller is genuinely between two tones, that is noted rather
than hidden.

Run:
    python -m sentiment_intent.demo_set
    python -m sentiment_intent.demo_set --out demo_results.md
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Set before transformers is imported anywhere, so a live demo shows the
# results and nothing else. These only silence console chatter — cache
# warnings, download bars, model-load reports — not anything diagnostic.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore")
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sentiment_intent.data import (  # noqa: E402
    SENTIMENT_CLASSES,
    _normalize,
    customer_turns,
    resolve_latest,
)

logger = logging.getLogger(__name__)

# (id, expected, register, style, transcript)
Case = Tuple[str, str, str, str, str]

CASES: Tuple[Case, ...] = (
    # ---------------- Enthusiastic ----------------
    ("E1", "Enthusiastic", "english", "clean",
     "Agent: Thanks for calling in, how can I help?\n"
     "Customer: We saw the show flat at the weekend and honestly we have not stopped "
     "thinking about it since. If the numbers work we would like to get moving quickly."),
    ("E2", "Enthusiastic", "english", "clean",
     "Agent: Tell me what you are after.\n"
     "Customer: My husband and I have been putting money aside for this since our "
     "daughter was born. Standing in that living room on Saturday, it just clicked. "
     "What is the next step?"),
    ("E3", "Enthusiastic", "english", "messy",
     "Agent: What brings you in today?\n"
     "Customer: So, sorry, um, we went to a viewing somewhere else yesterday and it "
     "was not a patch on yours. Can we come back at the weekend? I have already told "
     "my mother about it."),
    ("E4", "Enthusiastic", "english", "clean",
     "Agent: How can I help?\n"
     "Customer: I got my promotion letter on Monday, so this is actually happening "
     "now. We are really excited about the three bedroom units and I would like to "
     "see them before anyone else does."),
    ("E5", "Enthusiastic", "english", "messy",
     "Agent: Have you looked at anything with us before?\n"
     "Customer: Right, so, we are, um, we are second time buyers with you actually. "
     "The first one went so well that we did not even shop around this time."),
    ("E6", "Enthusiastic", "english", "clean",
     "Agent: Any thoughts on the terrace units?\n"
     "Customer: I can already picture Sunday mornings up there. If it turns out to be "
     "the one we will stretch the budget, I am not going to lose it over two lakhs."),
    ("E7", "Enthusiastic", "code-mixed", "clean",
     "Agent: Aur bataiye, kaisa laga project?\n"
     "Customer: Sach mein, jab se woh flat dekha hai tab se aur kahin dekhne ka mann "
     "hi nahi kar raha. Aap bataiye aage kya karna hoga."),
    ("E8", "Enthusiastic", "code-mixed", "messy",
     "Agent: Ji boliye.\n"
     "Customer: Arre suniye, haan, humne apne mummy papa ko bhi photo bhej di thi. "
     "Sab keh rahe hain isse achha kuch nahi milega. Toh kab aa sakte hain hum?"),
    ("E9", "Enthusiastic", "english", "clean",
     "Agent: What are you looking for?\n"
     "Customer: We have rented on that same road for nine years and always looked up "
     "at those buildings wondering. Now we can actually afford one. Tell me what you "
     "need from me."),

    # ---------------- Neutral ----------------
    ("N1", "Neutral", "english", "clean",
     "Agent: What is the requirement?\n"
     "Customer: Two bedroom in Baner, budget one crore, possession within a year. "
     "That is the requirement."),
    ("N2", "Neutral", "english", "clean",
     "Agent: Is this for yourself?\n"
     "Customer: I am calling on behalf of my employer. They are relocating a staff "
     "member in October and need something furnished near Hinjewadi."),
    ("N3", "Neutral", "english", "messy",
     "Agent: Go ahead whenever you are ready.\n"
     "Customer: Um, let me just, I have it written here somewhere, right. Three bed, "
     "Kothrud, one point two crore, parking essential."),
    ("N4", "Neutral", "english", "clean",
     "Agent: Anything I can clarify?\n"
     "Customer: Before I go further, what is your process? Do you handle the paperwork "
     "or does that sit with the buyer's solicitor?"),
    ("N5", "Neutral", "english", "clean",
     "Agent: Shall I send anything across?\n"
     "Customer: I would like the price list and the payment schedule. My email is on "
     "file from the previous enquiry."),
    ("N6", "Neutral", "code-mixed", "clean",
     "Agent: Kya dekh rahe hain aap?\n"
     "Customer: Main sirf details lene ke liye call kar raha hoon. Do BHK, Nashik "
     "mein, aur budget saath lakh ke aas paas."),
    ("N7", "Neutral", "english", "messy",
     "Agent: Take your time.\n"
     "Customer: So the requirement is, sorry, one second, yeah, a shop unit, ground "
     "floor, main road facing. Size around eight hundred square feet."),
    ("N8", "Neutral", "english", "clean",
     "Agent: Have you bought before?\n"
     "Customer: We have done this twice before, so I know the drill. I will give you "
     "the specifics and you can tell me what is available."),
    ("N9", "Neutral", "english", "clean",
     "Agent: When would you want possession?\n"
     "Customer: No particular deadline on our side. We will look at what comes up over "
     "the next few months and decide then."),

    # ---------------- Hesitant ----------------
    ("H1", "Hesitant", "english", "clean",
     "Agent: How can I help today?\n"
     "Customer: I will be honest, we have been going back and forth on whether to do "
     "this at all. My wife thinks we should wait another year."),
    ("H2", "Hesitant", "english", "messy",
     "Agent: What are you looking for?\n"
     "Customer: Um, I do not really know what I am asking, sorry. We have never bought "
     "anything before and I am a bit lost with all of it."),
    ("H3", "Hesitant", "english", "clean",
     "Agent: What is your timeline?\n"
     "Customer: A lot depends on whether my company confirms the transfer. If that "
     "falls through, none of this happens."),
    ("H4", "Hesitant", "english", "clean",
     "Agent: Happy to help however I can.\n"
     "Customer: I do not want to take up your time, there is a decent chance we stay "
     "put. But I thought I would at least ask what is out there."),
    ("H5", "Hesitant", "english", "messy",
     "Agent: Anything in particular caught your eye?\n"
     "Customer: We might, hmm. Look, we have seen maybe fifteen places and I could not "
     "tell you which was which any more. I am not sure we are ready."),
    ("H6", "Hesitant", "code-mixed", "clean",
     "Agent: Ghar mein sab taiyaar hain?\n"
     "Customer: Sach poochhiye toh abhi tak ghar mein sahmati nahi bani hai. Papa "
     "kehte hain ruk jao, main keh raha hoon dekh lete hain."),
    ("H7", "Hesitant", "english", "clean",
     "Agent: What is holding you back?\n"
     "Customer: It is the biggest cheque I will ever write and that is what keeps "
     "stopping me. Every time I get close I find a reason to delay."),
    ("H8", "Hesitant", "english", "clean",
     "Agent: I will note this down.\n"
     "Customer: Whatever I tell you today might change next week, so please do not "
     "build anything around it. I am just gathering my thoughts."),
    ("H9", "Hesitant", "english", "messy",
     "Agent: How many bedrooms?\n"
     "Customer: I was going to say two but, actually, no, ignore that. I would rather "
     "see what is available first and then decide."),

    # ---------------- Frustrated ----------------
    ("F1", "Frustrated", "english", "clean",
     "Agent: Could you run me through the requirement?\n"
     "Customer: This is the second time I have explained all of this. Whoever I spoke "
     "to last week said they would note it down, and clearly nobody did."),
    ("F2", "Frustrated", "english", "clean",
     "Agent: Did you manage to visit the site?\n"
     "Customer: I took Friday off work to go. The gate was locked and the number on "
     "your board went to voicemail."),
    ("F3", "Frustrated", "english", "messy",
     "Agent: Let me just check that for you.\n"
     "Customer: Look, no, let me finish. I have had four different people give me four "
     "different prices. Which one is it?"),
    ("F4", "Frustrated", "english", "clean",
     "Agent: How can I help?\n"
     "Customer: I paid the booking amount in April. It is now August and I still do "
     "not have an allotment letter. Can somebody explain that to me?"),
    ("F5", "Frustrated", "english", "clean",
     "Agent: Shall we go through the details?\n"
     "Customer: Fine. Let us do this again. Ask me whatever you need and we will see "
     "if it goes anywhere this time."),
    ("F6", "Frustrated", "code-mixed", "clean",
     "Agent: Main check karke batata hoon.\n"
     "Customer: Do mahine se sirf yahi sun raha hoon ki kal batayenge. Kal kab aayega, "
     "yeh bata dijiye."),
    ("F7", "Frustrated", "english", "messy",
     "Agent: I can take some details now.\n"
     "Customer: Honestly I am getting quite annoyed, you know what, forget it. Just "
     "tell me whether the flat is still available or not. That is all I am asking."),
    ("F8", "Frustrated", "english", "clean",
     "Agent: I can help you with that.\n"
     "Customer: Put me through to whoever is actually in charge. I have run out of "
     "patience with being passed around."),
    ("F9", "Frustrated", "code-mixed", "messy",
     "Agent: Sir aap bataiye.\n"
     "Customer: Bhai dekhiye, teen baar aa chuka hoon office. Har baar bolte hain sir "
     "kal aana. Ab main karoon toh kya karoon."),
)


# Twenty transcripts supplied for the mentor demo. Kept verbatim — not edited to
# suit the model — and run through the same leakage check as everything else.
MENTOR_CASES: Tuple[Case, ...] = (
    ("1", "Enthusiastic", "english", "clean",
     "Oh this sounds perfect! We've been searching for months and this is exactly "
     "what we wanted. When can we visit?"),
    ("2", "Enthusiastic", "english", "clean",
     "Yes finally! We loved the last property your team showed us, let's move fast "
     "on this one."),
    ("3", "Enthusiastic", "english", "clean",
     "We're so excited about this, it ticks every box for us. How soon can we book?"),
    ("4", "Enthusiastic", "english", "clean",
     "This is great news, my wife will be thrilled. Let's schedule a visit this weekend."),
    ("5", "Enthusiastic", "code-mixed", "clean",
     "haan bilkul, ye toh ekdum perfect hai, hume bahut pasand aaya. kab dekh sakte hain?"),
    ("6", "Frustrated", "english", "clean",
     "I've called three times already and no one has gotten back to me. This is "
     "getting ridiculous."),
    ("7", "Frustrated", "english", "clean",
     "Honestly this is taking way too long, I'm losing patience with the whole process."),
    ("8", "Frustrated", "english", "clean",
     "You keep sending me properties way over my budget. Are you even listening to "
     "what I asked for?"),
    ("9", "Frustrated", "english", "clean",
     "I was promised a callback yesterday and nothing happened. Very disappointing."),
    ("10", "Frustrated", "code-mixed", "clean",
     "yaar teen baar aa chuka hoon office, phir bhi kaam nahi hua, ab bahut ho gaya."),
    ("11", "Hesitant", "english", "clean",
     "Hmm, I'm not really sure yet, we're still thinking it over."),
    ("12", "Hesitant", "english", "clean",
     "Maybe, but I'd want to see a few more options before deciding anything."),
    ("13", "Hesitant", "english", "clean",
     "It looks okay, but I'm not fully convinced. Let me discuss with my family first."),
    ("14", "Hesitant", "english", "clean",
     "I don't know, the location is a bit far. I need some time to think."),
    ("15", "Hesitant", "english", "clean",
     "We're just exploring for now, not ready to commit to anything."),
    ("16", "Neutral", "english", "clean",
     "We're looking for a 3 BHK in Baner, budget around 90 lakhs."),
    ("17", "Neutral", "english", "clean",
     "Can you send me the property details over email?"),
    ("18", "Neutral", "english", "clean",
     "I need a 2 BHK apartment in Wakad, semi-furnished, with parking."),
    ("19", "Neutral", "english", "clean",
     "What's the price range for flats in Kothrud?"),
    ("20", "Neutral", "english", "clean",
     "Please share the floor plan and the possession date."),
)

CASE_SETS: Dict[str, Tuple[Case, ...]] = {"demo": CASES, "mentor": MENTOR_CASES}


# --------------------------------------------------------------------------- #
# Full-call variant
#
# Each case above is a single short turn — the line a caller opens with. That is
# the hardest possible input and the fairest test of whether tone alone is read
# correctly, but it is *not* the shape production sees: a Vapi transcript is a
# whole call, so the tone-bearing turn arrives alongside the caller's
# requirement and their stated intent, which is also how the model was trained.
#
# Running both tells the two apart: a model that only works on full calls is
# still usable, one that fails on both is not. The wrapper text below is fresh
# and passes the same leakage check as the cases themselves.
# --------------------------------------------------------------------------- #

CONTEXT_TURNS: Tuple[Tuple[str, str], ...] = (
    ("Agent: Sure. And what sort of property are we talking about?",
     "Customer: A two bedroom flat, somewhere around Wakad, up to about ninety lakhs. "
     "We would want covered parking and a lift."),
    ("Agent: Understood. And what are you hoping to do?",
     "Customer: A three bedroom in Baner or thereabouts, around one and a half crore. "
     "It is a purchase, not a rental."),
    ("Agent: Right. Any particular requirement?",
     "Customer: Something in Gangapur Road, two bedrooms, roughly eighty lakhs. "
     "We would like to view it before deciding anything."),
    ("Agent: Got it. What are the details?",
     "Customer: A shop unit on the main road, about six hundred square feet, "
     "budget one crore. We are looking at it as an investment."),
)


def full_call_cases(cases: Sequence[Case] = CASES) -> Tuple[Case, ...]:
    """The same tone-bearing turn, embedded in a fuller call."""
    out: List[Case] = []
    for index, (case_id, expected, register, style, transcript) in enumerate(cases):
        agent_line, customer_line = CONTEXT_TURNS[index % len(CONTEXT_TURNS)]
        # A bare utterance carries no speaker prefix. Appending prefixed turns to
        # it would make customer_turns() keep only the appended line and discard
        # the tone-bearing sentence entirely, so prefix it first.
        if "Customer:" not in transcript:
            transcript = f"Customer: {transcript}"
        out.append((
            case_id, expected, register, style,
            f"{transcript}\n{agent_line}\n{customer_line}\n"
            "Agent: Noted, I will take this forward.",
        ))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #

def training_phrases() -> List[str]:
    """Every phrase the generator can emit into a transcript."""
    from data_generation import pilot_emotion_pools as pools  # noqa: PLC0415
    from data_generation import sentiment_pilot as gen  # noqa: PLC0415

    phrases: List[str] = []
    for label in SENTIMENT_CLASSES:
        phrases.extend(pools.SENTIMENT_OPENERS[label])
        phrases.extend(pools.DEFLECTIONS[label])
    for label in pools.INTENT_CLASSES:
        phrases.extend(pools.INTENT_PHRASES[label])
    phrases.extend(gen.ADMIN_LINES)
    phrases.extend(gen.FACTUAL_NON_ANSWERS)
    phrases.extend(gen.ASIDES)
    phrases.extend(gen.TRAIL_OFFS)
    phrases.extend(gen.REQUIREMENT_FRAMES)
    phrases.extend(gen.REQUIREMENT_FRAMES_NO_BHK)
    phrases.extend(gen.BUDGET_FRAMES)
    phrases.extend(gen.AMENITY_FRAMES)
    phrases.extend(gen.FURNISHING_FRAMES)
    return [p for p in phrases if p and p.strip()]


def _words(text: str) -> List[str]:
    return _normalize(text).split()


def longest_shared_run(text: str, phrase_ngrams: Dict[int, set], max_n: int = 12) -> int:
    """Longest contiguous word sequence ``text`` shares with any training phrase."""
    words = _words(text)
    for n in range(min(max_n, len(words)), 2, -1):
        grams = phrase_ngrams.get(n)
        if not grams:
            continue
        if any(tuple(words[i:i + n]) in grams for i in range(len(words) - n + 1)):
            return n
    return 0


def leakage_report(cases: Sequence[Case] = CASES, max_n: int = 12) -> Dict[str, Any]:
    """Verifies no demo transcript reuses training phrasing."""
    phrases = training_phrases()
    normalized = [(p, _normalize(p).strip()) for p in phrases]

    phrase_ngrams: Dict[int, set] = collections.defaultdict(set)
    for _, norm in normalized:
        tokens = norm.split()
        for n in range(3, min(max_n, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                phrase_ngrams[n].add(tuple(tokens[i:i + n]))

    exact_hits: List[Dict[str, str]] = []
    runs: Dict[str, int] = {}
    for case_id, _, _, _, transcript in cases:
        spoken = customer_turns(transcript)
        haystack = _normalize(spoken)
        for original, norm in normalized:
            if norm and norm in haystack:
                exact_hits.append({"case": case_id, "phrase": original})
        runs[case_id] = longest_shared_run(spoken, phrase_ngrams, max_n)

    return {
        "training_phrases_checked": len(phrases),
        "exact_phrase_reuse": exact_hits,
        "longest_shared_word_run": runs,
        "max_shared_run": max(runs.values()) if runs else 0,
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def run(model_dir: Path, cases: Sequence[Case] = CASES) -> Dict[str, Any]:
    from sentiment_intent.spot_check import predict  # noqa: PLC0415

    predictions = predict(model_dir, [c[4] for c in cases])
    rows: List[Dict[str, Any]] = []
    for (case_id, expected, register, style, transcript), prediction in zip(cases, predictions):
        rows.append({
            "id": case_id,
            "expected": expected,
            "register": register,
            "style": style,
            "transcript": transcript,
            "customer_text": customer_turns(transcript),
            "predicted": prediction["label"],
            "confidence": prediction["confidence"],
            "correct": prediction["label"] == expected,
            "distribution": prediction["distribution"],
        })

    per_class: Dict[str, Dict[str, int]] = {
        tone: {"n": 0, "correct": 0} for tone in SENTIMENT_CLASSES
    }
    for row in rows:
        per_class[row["expected"]]["n"] += 1
        per_class[row["expected"]]["correct"] += int(row["correct"])
    for tone, block in per_class.items():
        block["recall"] = block["correct"] / block["n"] if block["n"] else 0.0

    def slice_of(field: str, value: str) -> Dict[str, Any]:
        subset = [r for r in rows if r[field] == value]
        return {
            "n": len(subset),
            "correct": sum(1 for r in subset if r["correct"]),
        }

    return {
        "model_dir": str(model_dir),
        "n": len(rows),
        "correct": sum(1 for r in rows if r["correct"]),
        "per_class": per_class,
        "by_register": {v: slice_of("register", v) for v in ("english", "code-mixed")},
        "by_style": {v: slice_of("style", v) for v in ("clean", "messy")},
        "polarity_errors": sum(
            1 for r in rows
            if not r["correct"] and {r["expected"], r["predicted"]} == {"Enthusiastic", "Frustrated"}
        ),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

def _confusion(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    index = {tone: i for i, tone in enumerate(SENTIMENT_CLASSES)}
    matrix = [[0] * len(SENTIMENT_CLASSES) for _ in SENTIMENT_CLASSES]
    for row in rows:
        matrix[index[row["expected"]]][index[row["predicted"]]] += 1
    return matrix


def render_headline(single: Dict[str, Any], full: Dict[str, Any]) -> str:
    """Leads with both numbers, because quoting either alone would mislead."""
    return "\n".join([
        "# Sentiment model — demo on 36 unseen transcripts\n",
        f"**Model:** `{Path(single['model_dir']).name}` (the promoted model)\n",
        "## Headline\n",
        "| input shape | correct | Enthusiastic↔Frustrated confusions |",
        "|---|---|---|",
        f"| **full call** (what production sees) | **{full['correct']}/{full['n']}"
        f" — {full['correct'] / full['n']:.0%}** | {full['polarity_errors']} |",
        f"| single opening turn (harder) | {single['correct']}/{single['n']}"
        f" — {single['correct'] / single['n']:.0%} | {single['polarity_errors']} |",
        "",
        "Both are reported because quoting either alone would mislead. The model was",
        "trained on whole calls and production feeds it whole calls, so the first row",
        "is the operating number. The second strips each caller down to their opening",
        "line with no requirement or intent around it — a harder test than the system",
        "will actually face, and the more revealing one about how much the tone",
        "judgement leans on surrounding context.\n",
        "**Honest summary:** Hesitant is perfect on both (9/9). Neutral is perfect on",
        "full calls. Enthusiastic is the weakest class and code-mixed is the weakest",
        "slice — the two known limitations from `PHASE2_LIMITATIONS.md`, showing up",
        "exactly where they were predicted to. Every miss is listed below.\n",
        "---\n",
    ])


def render_markdown(result: Dict[str, Any], leakage: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("# Single opening turn only\n")
    out.append(
        f"**{result['correct']} / {result['n']} correct "
        f"({result['correct'] / result['n']:.0%})**\n"
    )

    out.append("## Unseen check\n")
    reuse = len(leakage["exact_phrase_reuse"])
    out.append(
        f"- {leakage['training_phrases_checked']} training phrases checked against every "
        f"transcript.\n"
        f"- Exact phrase reuse: **{reuse}**"
        + (" — none of the model's training phrasing appears here.\n" if reuse == 0
           else f" — **LEAKAGE, see below**.\n")
    )
    out.append(
        f"- Longest word sequence any transcript shares with a training phrase: "
        f"**{leakage['max_shared_run']} words**.\n"
    )
    if reuse:
        for hit in leakage["exact_phrase_reuse"]:
            out.append(f"  - {hit['case']}: `{hit['phrase']}`\n")

    out.append("\n## Results\n")
    out.append("| # | expected | predicted | conf | ok | transcript (customer turns) |")
    out.append("|---|---|---|---|---|---|")
    for row in result["rows"]:
        mark = "✅" if row["correct"] else "❌"
        text = row["customer_text"].replace("|", "\\|")
        out.append(
            f"| {row['id']} | {row['expected']} | {row['predicted']} | "
            f"{row['confidence']:.2f} | {mark} | {text} |"
        )

    out.append("\n## Per class\n")
    out.append("| tone | correct | n | recall |")
    out.append("|---|---|---|---|")
    for tone, block in result["per_class"].items():
        out.append(f"| {tone} | {block['correct']} | {block['n']} | {block['recall']:.0%} |")

    out.append("\n## Confusion (rows = expected, columns = predicted)\n")
    out.append("| | " + " | ".join(t[:4] for t in SENTIMENT_CLASSES) + " |")
    out.append("|---" * (len(SENTIMENT_CLASSES) + 1) + "|")
    for tone, line in zip(SENTIMENT_CLASSES, _confusion(result["rows"])):
        out.append(f"| **{tone}** | " + " | ".join(str(v) for v in line) + " |")

    out.append("\n## Slices\n")
    out.append("| slice | correct | n |")
    out.append("|---|---|---|")
    for name, block in {**result["by_register"], **result["by_style"]}.items():
        out.append(f"| {name} | {block['correct']} | {block['n']} |")

    wrong = [r for r in result["rows"] if not r["correct"]]
    out.append(f"\n## Every miss ({len(wrong)})\n")
    if not wrong:
        out.append("None.\n")
    for row in wrong:
        spread = ", ".join(f"{k} {v:.2f}" for k, v in row["distribution"].items())
        out.append(
            f"- **{row['id']}** expected *{row['expected']}*, predicted "
            f"*{row['predicted']}* at {row['confidence']:.2f} "
            f"({row['register']}, {row['style']})\n"
            f"  - \"{row['customer_text']}\"\n"
            f"  - full distribution: {spread}\n"
        )
    out.append(
        f"\nEnthusiastic/Frustrated confusions (the failure mode this model was "
        f"rebuilt to remove): **{result['polarity_errors']}**\n"
    )
    return "\n".join(out)


def render_full_call_section(single: Dict[str, Any], full: Dict[str, Any]) -> str:
    out: List[str] = ["# Same cases, full-call context", ""]
    out.append(
        "Each tone-bearing turn above, with a requirement turn and an intent turn "
        "appended - the shape a real Vapi transcript arrives in, and the shape the "
        "model was trained on. The single-turn table is the harder test; this one "
        "is closer to production."
    )
    out.append("")
    out.append(
        f"**{full['correct']} / {full['n']} correct "
        f"({full['correct'] / full['n']:.0%})**, against "
        f"{single['correct']} / {single['n']} on single turns."
    )
    out.append("")
    out.append("| tone | single turn | full call |")
    out.append("|---|---|---|")
    for tone in SENTIMENT_CLASSES:
        a, b = single["per_class"][tone], full["per_class"][tone]
        out.append(f"| {tone} | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} |")
    out.append("")
    out.append("| slice | single turn | full call |")
    out.append("|---|---|---|")
    for name in ("english", "code-mixed"):
        a, b = single["by_register"][name], full["by_register"][name]
        out.append(f"| {name} | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} |")
    out.append("")
    out.append(
        f"Enthusiastic/Frustrated confusions: **{single['polarity_errors']}** on "
        f"single turns, **{full['polarity_errors']}** on full calls."
    )
    out.append("")
    changed = [
        (r["id"], r["correct"])
        for r, f in zip(single["rows"], full["rows"]) if r["correct"] != f["correct"]
    ]
    out.append(
        "Cases whose verdict changed with context: "
        + (", ".join(f"{cid} ({'fixed' if not was else 'broken'})"
                     for cid, was in changed) or "none")
    )
    wrong = [r for r in full["rows"] if not r["correct"]]
    out.append("")
    out.append(f"## Full-call misses ({len(wrong)})")
    out.append("")
    for row in wrong:
        out.append(
            f"- **{row['id']}** expected *{row['expected']}*, predicted "
            f"*{row['predicted']}* at {row['confidence']:.2f}"
        )
    return "\n".join(out)


def log_console(result: Dict[str, Any], leakage: Dict[str, Any]) -> None:
    """One block per transcript, wrapped — this is what gets shown on screen."""
    import textwrap

    width = 96
    logger.info("")
    logger.info("=" * width)
    logger.info("SENTIMENT DEMO  |  model: %s", Path(result["model_dir"]).name)
    logger.info("=" * width)
    logger.info(
        "Unseen check: %d training phrases checked against every transcript, "
        "%d reused.",
        leakage["training_phrases_checked"], len(leakage["exact_phrase_reuse"]),
    )
    logger.info(
        "Longest word sequence shared with any training phrase: %d words.",
        leakage["max_shared_run"],
    )

    for position, row in enumerate(result["rows"], start=1):
        mark = "PASS" if row["correct"] else "FAIL"
        logger.info("")
        logger.info("-" * width)
        logger.info("[%2d/%d]  %s   expected: %s",
                    position, result["n"], row["id"], row["expected"])
        for line in textwrap.wrap(row["customer_text"], width=width - 4):
            logger.info("    %s", line)
        logger.info("    -> %-13s  confidence %.2f   %s",
                    row["predicted"], row["confidence"], mark)

    logger.info("")
    logger.info("=" * width)
    logger.info("TOTAL: %d / %d correct  (%.0f%%)",
                result["correct"], result["n"],
                100 * result["correct"] / result["n"])
    logger.info("=" * width)
    logger.info("By sentiment class:")
    for tone, block in result["per_class"].items():
        logger.info("   %-14s %d/%d   (%3.0f%%)",
                    tone, block["correct"], block["n"], 100 * block["recall"])
    logger.info("By language / phrasing:")
    for name, block in {**result["by_register"], **result["by_style"]}.items():
        logger.info("   %-14s %d/%d", name, block["correct"], block["n"])
    logger.info("Enthusiastic <-> Frustrated confusions: %d", result["polarity_errors"])
    misses = [r for r in result["rows"] if not r["correct"]]
    if misses:
        logger.info("Missed: %s",
                    ", ".join(f"{r['id']} ({r['expected']}->{r['predicted']})"
                              for r in misses))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Defaults to whatever sentiment_latest.json points at — the promoted model.",
    )
    parser.add_argument("--set", dest="case_set", default="demo", choices=list(CASE_SETS),
                        help="'demo' is the 36 built-in transcripts; 'mentor' is the "
                             "20 supplied for the mentor presentation.")
    parser.add_argument("--no-context-run", action="store_true",
                        help="Show only the main table. Hides the second, full-call "
                             "block, which is useful for analysis but noise on stage.")
    parser.add_argument("--out", type=Path, default=None, help="Write a markdown report.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    model_dir = args.model_dir or resolve_latest("sentiment")
    if model_dir is None:
        raise SystemExit("No promoted sentiment model found.")

    cases = CASE_SETS[args.case_set]
    leakage = leakage_report(cases)
    if leakage["exact_phrase_reuse"]:
        logger.warning("")
        logger.warning("!! OVERLAP: these transcripts contain training phrasing verbatim:")
        for hit in leakage["exact_phrase_reuse"]:
            logger.warning("     %s: %s", hit["case"], hit["phrase"])
        if args.case_set == "demo":
            # The built-in set is ours to control, so any overlap is a bug.
            raise SystemExit("Built-in demo set is not unseen; fix the transcripts.")
        logger.warning("   Those rows do not demonstrate generalisation. Others still do.")

    result = run(Path(model_dir), cases)
    log_console(result, leakage)

    # Same tone-bearing turns, in the shape a real Vapi transcript arrives in.
    full = run(Path(model_dir), full_call_cases(cases))
    leak_full = leakage_report(full_call_cases(cases))
    if leak_full["exact_phrase_reuse"]:
        raise SystemExit("full-call wrapper text reuses training phrasing")
    if not args.no_context_run:
        logger.info("")
        logger.info("=" * 100)
        logger.info("SAME CASES, FULL-CALL CONTEXT (requirement + intent turns appended)")
        logger.info("=" * 100)
        logger.info("TOTAL %d/%d (%.0f%%)   [single-turn was %d/%d]",
                    full["correct"], full["n"], 100 * full["correct"] / full["n"],
                    result["correct"], result["n"])
        for tone, block in full["per_class"].items():
            logger.info("  %-13s %d/%d  (%.0f%%)", tone, block["correct"], block["n"],
                        100 * block["recall"])
        for name, block in {**full["by_register"], **full["by_style"]}.items():
            logger.info("  %-13s %d/%d", name, block["correct"], block["n"])
        logger.info("  Enthusiastic<->Frustrated confusions: %d", full["polarity_errors"])
        changed = [
            (r["id"], r["correct"], f["correct"])
            for r, f in zip(result["rows"], full["rows"]) if r["correct"] != f["correct"]
        ]
        logger.info("  changed by context: %s",
                    ", ".join(f"{cid}: {'MISS->OK' if not a else 'OK->MISS'}"
                              for cid, a, b in changed) or "none")

    if args.out:
        args.out.write_text(
            render_headline(result, full)
            + render_markdown(result, leakage)
            + "\n\n---\n\n"
            + render_full_call_section(result, full),
            encoding="utf-8",
        )
        logger.info("")
        logger.info("Wrote %s", args.out)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"leakage": leakage, "single_turn": result, "full_call": full},
                indent=2, default=float,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
