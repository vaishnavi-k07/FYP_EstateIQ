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

``*_GROUPS`` maps each class to {content filler: phrases built from it}. The
holdout diagnostic withholds whole fillers rather than individual phrases —
withholding one phrase would leave its content word visible in a sibling
phrase and make the generalisation test trivially easy.
"""

from __future__ import annotations

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

TONE_CLAUSES: Dict[str, List[str]] = {
    "Enthusiastic": [
        "we're really excited about this, so",
        "this is perfect timing, so",
        "we've been looking forward to this call, so",
        "we're quite keen to get going, so",
        "it's great to finally be doing this, so",
        "we're thrilled to be starting, so",
        "bahut excited hain hum, so",
        "we can't wait to get moving, so",
    ],
    "Neutral": [
        "so basically,",
        "the requirement is simply,",
        "to give you the details,",
        "let me lay it out,",
        "so essentially,",
        "here's what we need,",
        "seedha seedha bataata hoon,",
        "just to state it plainly,",
    ],
    "Hesitant": [
        "we're not entirely sure yet, but",
        "we're still exploring options, but",
        "this is all quite new to us, but",
        "we haven't fully decided, but",
        "we're a bit unclear ourselves, but",
        "we're only at the thinking stage, but",
        "abhi pakka decide nahi kiya hai, but",
        "we're still weighing things up, but",
    ],
    "Frustrated": [
        "I don't have much time, so quickly,",
        "this is my second time calling, so",
        "let me repeat what I told the last person,",
        "I've been going round in circles, so",
        "frankly this has taken far too long, so",
        "nobody got back to me last time, so",
        "bahut baar bataya hai yeh, so",
        "I'm losing patience here, so",
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

DEFLECTION_CORES: Dict[str, List[str]] = {
    "Enthusiastic": [
        "haven't decided that part yet, but I'll figure it out soon!",
        "no idea yet, but I'm totally open to suggestions!",
        "haven't got that far, but tell me what people usually go for!",
        "we're easy on that one, surprise us!",
        "still working that bit out, but nothing's ruled out!",
        "that one's up in the air, but we'll sort it out!",
    ],
    "Neutral": [
        "not decided yet, we'll figure that out later.",
        "no preference there, whatever works.",
        "haven't thought about that one.",
        "that's flexible, we can decide later.",
        "no strong view on that.",
        "we'll come back to that once the rest is clear.",
    ],
    "Hesitant": [
        "we haven't really thought about that yet.",
        "not sure, we're still figuring things out.",
        "I'd have to check with my family on that one.",
        "I really can't say at this point.",
        "I don't want to say something wrong, so let me get back to you.",
        "we might change our mind on that, so I'd rather not commit.",
    ],
    "Frustrated": [
        "not decided, can someone just call me back about this?",
        "I'd rather sort that out with whoever actually calls me.",
        "haven't decided, and I'm not deciding on this call.",
        "I've already explained most of this once.",
        "does this really need answering right now?",
        "I don't see why that matters at this stage.",
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
