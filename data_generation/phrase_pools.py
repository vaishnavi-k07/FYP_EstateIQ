"""Class-carrying phrase pools, built from frames shared across every class.

The phrase-holdout diagnostic found that the first version of these pools made
the *sentence frame* a perfect predictor of the label. With three phrases per
intent, "we're looking to ___" only ever appeared as Rent and "this is for
___, not ___" only ever appeared as Buy, so a model could score 0.92 without
ever reading the content word. On unseen phrasings Buy and Rent inverted
completely: 32/32 and 31/31 wrong.

The fix is structural rather than a matter of volume. Every frame here is
instantiated for **every** applicable class, so the frame carries no label
signal at all and only the content word does:

    "we're looking to buy"      (Buy)
    "we're looking to rent"     (Rent)
    "we're looking to invest"   (Investment)   ... and so on for all six

Negation is covered symmetrically in both directions — "for buying, not
renting" and "for renting, not buying" both exist — because the inversion
proved the model had learned no negation handling whatsoever.

The same principle applies to sentiment: greetings are shared across all four
tones (including "Um, hi." and "Yeah, hi.", which previously leaked Hesitant
and Frustrated on their own), and only the tone clause carries the label.

Sentiment needed a second pass. Sharing the *greeting* left the confound one
level down, at the token level: with eight clauses per tone and a one-third
holdout, a token could be erased from its own class's training clauses while
surviving in another class's. "bahut" was the clearest case — it appeared in
exactly one Enthusiastic clause ("bahut excited hain hum") and one Frustrated
one ("bahut baar bataya hai yeh"). The holdout withheld the Enthusiastic one,
so at training time "bahut" was a pure Frustrated cue, and every held-out
"Bahut excited hain hum" transcript was classified Frustrated. Enthusiastic
and Frustrated inverted almost exactly (17 each way).

The fix is the same shared-exemplar principle applied per token rather than
per frame, enforced by ``shared_token_confounds``: any content token used by
more than one tone must appear in at least two clauses of *every* tone that
uses it, so no single withheld clause can hand a token to a rival class. Tone
is carried by the composition of the clause, not by any one word — "excited",
"keen", "decided", "sure", "call", "bahut" and "abhi" all appear under several
tones, resolved by what surrounds them.

``*_GROUPS`` maps each class to {content filler: phrases built from it}. The
holdout diagnostic withholds whole fillers rather than individual phrases —
withholding one phrase would leave its content word visible in a sibling
phrase and make the generalisation test trivially easy.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

INTENT_CLASSES: Tuple[str, ...] = (
    "Buy",
    "Rent",
    "Inquiry",
    "Schedule Visit",
    "Investment",
    "Request Callback",
)

SENTIMENT_CLASSES: Tuple[str, ...] = ("Enthusiastic", "Neutral", "Hesitant", "Frustrated")


# --------------------------------------------------------------------------- #
# Intent
#
# Every frame below is filled for every class. `inf` slots take a bare
# infinitive, `ger` slots a gerund or noun phrase, `cm` slots a code-mixed
# "<verb> karna" form.
# --------------------------------------------------------------------------- #

INTENT_FRAMES_INF: Tuple[str, ...] = (
    "we're looking to {x}",
    "we want to {x}",
    "the plan is to {x}",
    "we're hoping to {x}",
    "we'd like to {x}",
    "we intend to {x}",
)

INTENT_FRAMES_GER: Tuple[str, ...] = (
    "we're interested in {x}",
    "this is about {x}",
    "our focus is {x}",
    "we're thinking about {x}",
)

# Both orderings of every pair are produced, so no negated construction is
# tied to one label.
INTENT_FRAMES_NEG: Tuple[str, ...] = (
    "this is for {x}, not {y}",
    "we want to {xi}, not {yi}",
    "we don't want to {yi}, we want to {xi}",
    "it's {x}, definitely not {y}",
)

INTENT_FRAMES_CM: Tuple[str, ...] = (
    "humko {x} hai",
    "hum {x} chahte hain",
    "basically {x} hai",
)

INTENT_FILLERS: Dict[str, Dict[str, List[str]]] = {
    "Buy": {
        "inf": ["buy", "purchase it outright", "buy a place"],
        "ger": ["buying", "a purchase"],
        "cm": ["buy karna", "purchase karna"],
    },
    "Rent": {
        "inf": ["rent", "take it on rent", "rent a place"],
        "ger": ["renting", "a rental"],
        "cm": ["rent par lena", "rent karna"],
    },
    "Inquiry": {
        "inf": ["just get some information", "understand the options", "just explore"],
        "ger": ["gathering information", "an enquiry"],
        "cm": ["information lena", "sirf inquiry karna"],
    },
    "Schedule Visit": {
        "inf": ["see the place", "schedule a site visit", "go and view it"],
        "ger": ["a site visit", "seeing the place"],
        "cm": ["site visit karna", "property dekhna"],
    },
    "Investment": {
        "inf": ["invest", "invest in this", "put money into this"],
        "ger": ["investing", "an investment"],
        "cm": ["invest karna", "investment karna"],
    },
    "Request Callback": {
        "inf": ["get a callback", "have someone call me", "be called back"],
        "ger": ["a callback", "being called back"],
        "cm": ["callback lena", "call back karwana"],
    },
}


def _build_intent_pools() -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    groups: Dict[str, Dict[str, List[str]]] = {c: {} for c in INTENT_CLASSES}

    for label in INTENT_CLASSES:
        fillers = INTENT_FILLERS[label]
        others = [c for c in INTENT_CLASSES if c != label]

        for filler in fillers["inf"][:2]:
            groups[label].setdefault(filler, []).extend(
                frame.format(x=filler) for frame in INTENT_FRAMES_INF
            )
        for filler in fillers["ger"][:2]:
            groups[label].setdefault(filler, []).extend(
                frame.format(x=filler) for frame in INTENT_FRAMES_GER
            )

        # Contrast against every other class at least once, which makes the
        # set of negated phrases symmetric across the whole taxonomy.
        own_ger, own_inf = fillers["ger"][0], fillers["inf"][0]
        for index, other in enumerate(others):
            frame = INTENT_FRAMES_NEG[index % len(INTENT_FRAMES_NEG)]
            phrase = frame.format(
                x=own_ger,
                y=INTENT_FILLERS[other]["ger"][0],
                xi=own_inf,
                yi=INTENT_FILLERS[other]["inf"][0],
            )
            groups[label].setdefault(own_ger, []).append(phrase)

        for filler in fillers["cm"][:2]:
            groups[label].setdefault(filler, []).extend(
                frame.format(x=filler) for frame in INTENT_FRAMES_CM
            )

    phrases = {label: [p for group in groups[label].values() for p in group]
               for label in INTENT_CLASSES}
    return phrases, groups


# --------------------------------------------------------------------------- #
# Sentiment — openers
#
# Greetings are shared by all four tones. Markers that used to give a tone away
# on their own ("Um,", "Yeah,", "Look,") now appear with every label, so the
# tone clause is the only thing that carries the signal.
# --------------------------------------------------------------------------- #

GREETINGS: Tuple[str, ...] = (
    "Hi,",
    "Hello,",
    "Hi there,",
    "Hey there,",
    "Yeah, hi.",
    "Um, hi.",
    "Hello! Yes,",
    "Oh, hi.",
    "Hi, thanks for picking up.",
    "Haan, hello.",
)

# Twenty clauses per tone, three of them code-mixed. The width matters for two
# reasons: a one-third holdout still leaves thirteen clauses to train on, and
# no single clause is the only carrier of its vocabulary. Openers that used to
# belong to one tone outright ("I don't have much time", "let me lay it out",
# "this is my second time calling", "we're still exploring options") now open
# clauses under several tones, with the tail deciding the label.
TONE_CLAUSES: Dict[str, List[str]] = {
    "Enthusiastic": [
        "we're really excited about this, so",
        "this is perfect timing, we've been waiting for this, so",
        "we've been looking forward to this call, so",
        "we're quite keen to get going, so",
        "it's great to finally be doing this, so",
        "we're thrilled to be starting, so",
        "we can't wait to get moving, so",
        "we've finally decided to go ahead, so",
        "we're completely sure about what we want, so",
        "I've been waiting to make this call for months and I'm delighted, so",
        "quickly, because we're very keen to get started,",
        "honestly this is the best news we've had all week, so",
        "great to be talking again, we loved what we saw last time, so",
        "I don't have much time today, so quickly, though I'm thrilled we're finally doing this,",
        "we've been round a few places and this one felt right, so",
        "let me lay it out, we're delighted to be starting,",
        "we still have a couple of details to decide, but we're excited either way, so",
        "matlab bahut excited hain hum, thoda jaldi shuru karna hai, so",
        "abhi matlab pakka decide kar liya hai, bahut khush hain, so",
        "abhi thoda bahut khush hain hum, matlab pakka timing perfect hai, so",
    ],
    "Neutral": [
        "so basically,",
        "the requirement is simply,",
        "to give you the details,",
        "let me lay it out,",
        "so essentially,",
        "here's what we need,",
        "just to state it plainly,",
        "we're keen but I'll keep this call short and factual, so",
        "I don't have much time, so quickly, let me just list it out,",
        "we've decided the basics and the rest is open, so",
        "quickly, here's the requirement,",
        "honestly there isn't much to it,",
        "sure, here's what we're after,",
        "we're in no rush, so I'll just give you the facts,",
        "this is my second time calling about this, so to recap,",
        "we're keen enough to be calling, but I'll stick to the details,",
        "to be clear about what we need,",
        "matlab dekhiye, requirement thoda simple hai,",
        "seedha seedha bataata hoon, matlab abhi bahut clear hai,",
        "abhi bahut short rakhta hoon, dekhiye thoda seedha bata deta hoon,",
    ],
    "Hesitant": [
        "we're not entirely sure yet, but",
        "we're still exploring options, but",
        "this is all quite new to us, but",
        "we haven't fully decided, but",
        "we're a bit unclear ourselves, but",
        "we're only at the thinking stage and probably waiting a bit, but",
        "we're still weighing things up, but",
        "we're keen in principle, just not certain yet, so",
        "I almost didn't make this call, honestly, but",
        "we've been going round in circles in our own heads, so",
        "I'm not sure this is the right time to be calling, but",
        "quickly — we're excited, though I may well change my mind,",
        "we might be waiting a while before we decide anything, so",
        "this is my second time thinking about calling, so",
        "we're excited and quite nervous about it in equal measure, so",
        "let me lay it out, though nothing is settled,",
        "quickly, we're keen but we've not decided, so",
        "abhi pakka decide nahi kiya hai, hum thoda unsure hain, but",
        "thoda confusion hai, matlab abhi bahut clear nahi hai, but",
        "bahut soch rahe hain, matlab pakka kuch nahi hai abhi, thoda time chahiye, so",
    ],
    "Frustrated": [
        "I don't have much time, so quickly,",
        "this is my second time calling, so",
        "let me repeat what I told the last person I was waiting on,",
        "I've been going round in circles, so",
        "frankly this has taken far too long, so",
        "nobody got back to me last time I was calling, so",
        "I'm losing patience here, so quickly,",
        "I was genuinely keen and excited about this and then nothing happened, so",
        "I've been waiting a week for someone to call, so",
        "honestly I'm not sure why I'm still bothering, but",
        "we decided months ago and nothing has moved, so",
        "I'll keep this short because I've said it all before,",
        "I was excited and quite keen at the start and much less so now, so",
        "let me lay it out again, for the third time,",
        "this is all quite new to whoever I speak to each time, apparently, so",
        "we're still exploring options only because nobody here has helped, so",
        "to give you the details yet again,",
        "bahut baar bataya hai yeh, matlab abhi tak koi reply nahi aaya, khush nahi hoon bilkul, so",
        "matlab dekhiye, thoda bhi khush nahi hoon main, patience khatam ho raha hai, so",
        "abhi tak kuch nahi hua, bahut time waste ho gaya, main thak gaya hoon, dekhiye thoda serious lijiye, so",
    ],
}


def _join_greeting(greeting: str, clause: str) -> str:
    """Capitalises the clause when the greeting closed the sentence."""
    if greeting.rstrip().endswith((".", "!", "?")):
        clause = clause[0].upper() + clause[1:]
    return f"{greeting} {clause}"


def _build_opener_pools() -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    groups = {
        label: {clause: [_join_greeting(greeting, clause) for greeting in GREETINGS]
                for clause in clauses}
        for label, clauses in TONE_CLAUSES.items()
    }
    phrases = {label: [p for group in groups[label].values() for p in group]
               for label in SENTIMENT_CLASSES}
    return phrases, groups


# --------------------------------------------------------------------------- #
# Sentiment — deflections (customer non-answers)
# --------------------------------------------------------------------------- #

DEFLECTION_LEADINS: Tuple[str, ...] = ("", "Honestly, ", "Hmm, ", "Look, ", "Well, ")

# Twelve per tone. The first six of each are the originals; the rest exist to
# share openings across tones. Previously "no idea", "haven't decided" and
# "can someone call me back" each belonged to one tone, and every Enthusiastic
# non-answer was the only one to end in an exclamation mark — all four are
# now spread across the taxonomy so the tail has to be read.
DEFLECTION_CORES: Dict[str, List[str]] = {
    "Enthusiastic": [
        "haven't decided that part yet, but I'll figure it out soon!",
        "no idea yet, but I'm totally open to suggestions!",
        "haven't got that far, but tell me what people usually go for!",
        "we're easy on that one, surprise us!",
        "still working that bit out, but nothing's ruled out!",
        "that one's up in the air, but we'll sort it out!",
        "not sure yet, and honestly that's half the fun of it!",
        "I'd have to check with my family, but they'll be just as keen as I am.",
        "haven't thought about that one, so I'd love a recommendation.",
        "can someone call me back on that? I'd really enjoy talking it through.",
        "no strong view yet, whatever gets us moving fastest works for me.",
        "I've already explained most of what matters, and I'm happy to go over it again!",
    ],
    "Neutral": [
        "not decided yet, we'll figure that out later.",
        "no preference there, whatever works.",
        "haven't thought about that one.",
        "that's flexible, we can decide later.",
        "no strong view on that.",
        "we'll come back to that once the rest is clear.",
        "honestly not sure at this point, but it isn't a blocker.",
        "I'd have to check with my family, then I'll confirm it.",
        "can someone call me back about that once I've checked? It isn't urgent.",
        "I've already explained most of it, so let's move on to the rest.",
        "still working that out, I'll send it across later.",
        "no idea on that yet, we'll decide when we get there.",
    ],
    "Hesitant": [
        "we haven't really thought about that yet.",
        "not sure, we're still figuring things out.",
        "I'd have to check with my family on that one.",
        "I really can't say at this point.",
        "I don't want to say something wrong, so let me get back to you.",
        "we might change our mind on that, so I'd rather not commit.",
        "haven't decided, and I'd rather not guess at it.",
        "no idea yet, and that worries me a little honestly.",
        "can someone call me back about that? I'd like to think it over first.",
        "no strong view, though maybe we should have one by now.",
        "still working that out, and it's taking us a while.",
        "I've already explained as much as I know, which isn't much.",
    ],
    "Frustrated": [
        "not decided, can someone just call me back about this?",
        "I'd rather sort that out with whoever actually calls me.",
        "haven't decided, and I'm not deciding on this call!",
        "I've already explained most of this once.",
        "does this really need answering right now?",
        "I don't see why that matters at this stage.",
        "not sure, and honestly I shouldn't have to be.",
        "no idea, because nobody has explained any of it to me.",
        "I'd have to check with my family, which I'd have done if anyone had called!",
        "haven't thought about it, I've been too busy chasing your team.",
        "no strong view, other than wanting this finished.",
        "still working that out, same as the last time somebody asked.",
    ],
}


def _build_deflection_pools() -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    groups: Dict[str, Dict[str, List[str]]] = {}
    for label, cores in DEFLECTION_CORES.items():
        groups[label] = {}
        for core in cores:
            variants = []
            for leadin in DEFLECTION_LEADINS:
                body = core[0].upper() + core[1:] if not leadin else core
                variants.append(f"Customer: {leadin}{body}")
            groups[label][core] = variants
    phrases = {label: [p for group in groups[label].values() for p in group]
               for label in SENTIMENT_CLASSES}
    return phrases, groups


INTENT_PHRASES, INTENT_PHRASE_GROUPS = _build_intent_pools()
SENTIMENT_OPENERS, SENTIMENT_OPENER_GROUPS = _build_opener_pools()
DEFLECTIONS, DEFLECTION_GROUPS = _build_deflection_pools()


def _all_intent_fillers() -> List[str]:
    """Every content filler, longest first so "buying" is masked before "buy"."""
    fillers = {
        filler
        for slots in INTENT_FILLERS.values()
        for values in slots.values()
        for filler in values
    }
    return sorted(fillers, key=len, reverse=True)


def intent_frame_of(phrase: str) -> str:
    """Strips every content word, leaving the bare frame.

    Both slots of a negated phrase must be masked: "we want to buy, not rent"
    and "we want to rent, not buying" are the same frame, and treating them as
    different would hide exactly the confound this guards against.
    """
    frame = phrase
    for filler in _all_intent_fillers():
        frame = frame.replace(filler, "{}")
    return frame


_TOKEN_RE = re.compile(r"[a-z']+")

# The tokens the guard below polices. Scope is deliberate: it is *not* every
# content word. Incidental vocabulary ("family", "week", "circles") appearing
# once in two tones is harmless — it is weak in both, so withholding either
# side strands nothing. What inverted Enthusiastic and Frustrated was a token
# that reads as a tone marker on its own and was carried by a single clause,
# so the holdout could delete it from one tone while a rival kept it.
#
# Two groups, both learned the hard way:
#   - the romanised-Hindi markers, because code-mixed clauses were previously
#     one-per-tone and so were *all* strandable ("bahut" is the case that fired)
#   - the English words that name a tone outright, which a model can and did
#     use as a label shortcut
TONE_MARKER_TOKENS: frozenset = frozenset("""
bahut abhi matlab hai hain hoon nahi pakka thoda seedha dekhiye khush
excited thrilled delighted keen enjoy loved
decided decide deciding sure certain settled
unclear nervous worries guess
patience losing chasing bothering
waiting wait urgent rush quickly
honestly frankly again calling call time timing
""".split())


# Romanised-Hindi tokens, used to tell a code-mixed phrase from an English one.
# Covers both pools: the sentiment tone clauses and the intent ``cm`` fillers
# ("buy karna", "property dekhna", ...).
ROMANISED_HINDI_MARKERS: frozenset = frozenset("""
bahut abhi matlab hai hain hoon nahi pakka thoda seedha dekhiye khush hum
kar karna karwana lena dekhna dekhna sirf par baar yeh main bhi koi tak kiya
kuch soch rahe jaldi shuru liya bata bataata bataya deta rakhta hua gaya thak
khatam raha lijiye chahiye confusion unsure aaya bilkul
""".split())


def phrase_register(text: str) -> str:
    """"code-mixed" when the phrase carries romanised-Hindi markers, else "english".

    The holdout samples within each register separately. Drawing uniformly
    across a class's whole pool let one class lose *every* code-mixed phrasing
    while its rivals kept theirs, which recreates the confound the pools were
    rewritten to remove — the model sees Hindi markers only under other labels,
    so a held-out code-mixed transcript is pushed to whichever label kept them.
    That is what happened to Enthusiastic on the first v3 run: all three of its
    code-mixed clauses were withheld and it lost 25 of 48 held-out transcripts
    to Frustrated.
    """
    tokens = set(_TOKEN_RE.findall(text.lower()))
    return "code-mixed" if tokens & ROMANISED_HINDI_MARKERS else "english"


def _tone_clause_sets() -> Dict[str, List[str]]:
    """Every tone-bearing clause per class: openers plus deflection cores."""
    return {
        label: [*TONE_CLAUSES[label], *DEFLECTION_CORES[label]]
        for label in SENTIMENT_CLASSES
    }


def marker_token_counts() -> Dict[str, Dict[str, int]]:
    """{marker token: {tone: how many of that tone's clauses contain it}}."""
    counts: Dict[str, Dict[str, int]] = {}
    for label, clauses in _tone_clause_sets().items():
        for clause in clauses:
            for token in set(_TOKEN_RE.findall(clause.lower())) & TONE_MARKER_TOKENS:
                per_label = counts.setdefault(token, {})
                per_label[label] = per_label.get(label, 0) + 1
    return counts


def shared_token_confounds(min_clauses: int = 2) -> Dict[str, Dict[str, int]]:
    """Marker tokens a holdout could hand to a rival tone — always empty.

    The sentiment analogue of :func:`frame_label_confounds`, and the property
    the second sentiment rewrite exists to establish.

    A marker used by more than one tone is dangerous when one of those tones
    carries it in a single clause: withhold that clause and the marker becomes
    a pure cue for the *other* tone, so held-out transcripts containing it
    invert. That is exactly what "bahut" did — one Enthusiastic clause, one
    Frustrated clause, and the holdout took the Enthusiastic one. Requiring
    ``min_clauses`` carriers in every tone that uses a marker means no single
    withheld clause can strand it.

    Returns {token: {tone: clause count}} per violation, naming both the word
    and the tone that is one clause away from losing it.
    """
    return {
        token: dict(sorted(per_label.items()))
        for token, per_label in marker_token_counts().items()
        if len(per_label) > 1 and min(per_label.values()) < min_clauses
    }


def frame_label_confounds() -> Dict[str, List[str]]:
    """Frames that still occur with only one class — should always be empty.

    Guards the property the whole rewrite exists to establish. Called by the
    test suite; a non-empty result means a confound has crept back in and the
    model can once again read the label off the sentence shape.
    """
    owners: Dict[str, set] = {}
    for label in INTENT_CLASSES:
        for phrase in INTENT_PHRASES[label]:
            owners.setdefault(intent_frame_of(phrase), set()).add(label)
    return {frame: sorted(labels) for frame, labels in owners.items() if len(labels) < 2}
