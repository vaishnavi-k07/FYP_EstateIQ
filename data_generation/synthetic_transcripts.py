"""Synthetic Agent/Customer call-transcript generator for EstateIQ.

Produces a pool of realistic multi-turn real-estate call transcripts, used
downstream as (a) Label Studio pre-annotation input for the MuRIL NER model
(nlp/ner/schema.md, Stage B) and (b) later domain-adaptation data for the
sentiment/intent models (Phase 2).

City/area/property-type/amenity values are always drawn from the CSV lookup
tables in backend/dataset/ (never hardcoded), per CLAUDE.md section 7.
Sentence templates, connector words, and phrasing banks below are generator
logic, not lookup values, so they live in this file.

Run: python data_generation/synthetic_transcripts.py
"""

import argparse
import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from numpy.random import default_rng

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).resolve().parent.parent / "backend" / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42
INITIAL_POOL_SIZE = 700
MAX_POOL_SIZE = 800

TARGET_CITIES = ["Nashik", "Pune"]
CITY_WEIGHTS = [0.6, 0.4]  # 60% Nashik / 40% Pune

INTENTS = ["Buy", "Rent", "Investment", "Inquiry", "Schedule Visit", "Request Callback"]
SENTIMENTS = ["Enthusiastic", "Neutral", "Hesitant", "Frustrated"]

RARE_TYPES = ["Warehouse", "Farm House", "Office Space", "Commercial Shop", "Residential Plot"]
RARE_TYPE_MIN_EACH = 45  # exceeds the >40 requirement

NEGATION_MIN = 130
CODE_MIX_MIN = 130
MULTI_CITY_MIN = 70
TELEGRAPHIC_MIN = 65

P_NEGATION = 0.20
P_CODE_MIX = 0.20
P_MULTI_CITY = 0.12
P_TELEGRAPHIC = 0.10

FURNISHING_OPTIONS = ["Furnished", "Semi Furnished", "Fully Furnished", "Unfurnished"]
NO_BHK_CATEGORIES = {"Commercial", "Industrial", "Agricultural", "Mixed Use"}

# Residential types that are self-describing about room count, so pairing
# them with a BHK number is a contradiction ("studio apartment, 2 BHK").
# Studio-ness is carried by PROPERTY_TYPE alone; BHK stays null. This is
# also what nlp/ner/schema.md requires — BHK means *bedroom count*, and a
# studio has none, so "studio" is never a BHK surface form.
NO_BHK_PROPERTY_TYPES = {"Studio Apartment"}

# Surface-form phrasing variety for types with no entries in
# property_type_synonyms.csv (the 5 rare types, plus Studio Apartment which
# absorbed the studio phrasings that used to live in generate_bhk). These
# affect only how the sentence reads — ground_truth.property_type always
# stores the canonical CSV name.
EXTRA_SURFACE_FORMS: Dict[str, List[str]] = {
    "Warehouse": ["warehouse", "godown-type space", "warehouse unit"],
    "Farm House": ["farm house", "farmhouse", "weekend farm house"],
    "Office Space": ["office space", "office", "small office"],
    "Commercial Shop": ["commercial shop", "shop", "retail shop"],
    "Residential Plot": ["residential plot", "plot", "piece of land"],
    "Studio Apartment": ["studio apartment", "studio", "compact studio", "studio flat"],
}

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
HINDI_NUM_WORDS = {1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch"}

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def indefinite_article(phrase: str) -> str:
    """'a' or 'an' based on the first word's leading letter (e.g. 'apartment'
    -> 'an', 'villa' -> 'a'). Approximate (letter-based, not phonetic) but
    correct for every property-type surface form used in this generator.
    """
    first_word = phrase.strip().split()[0] if phrase.strip() else ""
    return "an" if first_word[:1].lower() in "aeiou" else "a"


def cap_first(s: str) -> str:
    """Capitalizes only the first character, unlike str.capitalize() which
    also lowercases the rest of the string (would mangle 'Rs', 'BHK', etc.
    embedded mid-sentence).
    """
    return s[0].upper() + s[1:] if s else s


def spell_number(n: int) -> str:
    """Spells out an integer 1-999 in English words (e.g. 60 -> 'sixty')."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[rem]}" if rem else "")
    hundreds, rem = divmod(n, 100)
    prefix = f"{_ONES[hundreds]} hundred"
    return prefix + (f" {spell_number(rem)}" if rem else "")


# --------------------------------------------------------------------------
# Phrasing banks (generator logic, not CSV lookup values)
# --------------------------------------------------------------------------

OPENERS = [
    "Agent: Hi, thanks for calling in! What kind of property are you looking for today?",
    "Agent: Hello, thank you for reaching out. Could you tell me a bit about what you're looking for?",
    "Agent: Hi there! What brings you in today — are you looking to buy, rent, or just exploring?",
    "Agent: Good afternoon, thanks for calling. What can I help you find today?",
    "Agent: Hello! So, tell me — what kind of place are you hoping to find?",
    "Agent: Hi, welcome! Let's start with the basics — what are you looking for?",
    "Agent: Hi, good morning! How can I help you with your property search?",
    "Agent: Hello, you've reached the sales desk. What sort of property did you have in mind?",
    "Agent: Hi! Before we get into details — what are you hoping to find?",
    "Agent: Thanks for getting in touch. Tell me what you're after and I'll take it from there.",
    "Agent: Hello there. Let's see what we can find for you — what's the requirement?",
    "Agent: Hi, thanks for your time. Walk me through what you're looking for.",
    "Agent: Good to hear from you! What kind of place are we hunting for?",
    "Agent: Hello, how can I help today — buying, renting, or still deciding?",
    "Agent: Hi! Give me a quick idea of what you need and I'll note it all down.",
    "Agent: Hello, thanks for the call. What's on your wishlist?",
]

# Customer's opening framing. Each ends on a comma, dash, or 'so' because
# the main requirement clause is appended capitalized right after it.
SENTIMENT_OPENERS = {
    "Enthusiastic": [
        "Hi! We're really excited to finally look into this —",
        "Hey there, thanks for picking up! So,",
        "Hi, great, this is perfect timing —",
        "Hello! Yes, we've been waiting to start this, so",
        "Hi, so glad I got through —",
        "Hey! Right, so here's the plan —",
        "Hi there, we've been looking forward to this call, so",
        "Oh hi, brilliant — okay so,",
        "Hello! We're quite keen on this, so",
        "Hi, wonderful, let me tell you what we want —",
    ],
    "Neutral": [
        "Hi, so basically,",
        "Hello, yes so",
        "Hi, okay so",
        "Hello. Right, so",
        "Hi there. So, the requirement is,",
        "Yes, hello — so",
        "Hi, sure. Basically,",
        "Hello, okay. To give you the details,",
        "Hi. Let me just lay it out —",
        "Yeah, hi. So essentially,",
    ],
    "Hesitant": [
        "Um, hi, we're not entirely sure yet, but",
        "Hi... we're still exploring options, but roughly,",
        "Hello, we're just starting to look around, so",
        "Hi, um, this is all quite new to us, but",
        "Hello... we haven't fully decided, but broadly,",
        "Hi, sorry, we're a bit unclear ourselves, but",
        "Um, hello. We might be jumping the gun here, but",
        "Hi, we're only at the thinking stage really, so",
        "Hello, I hope this isn't premature, but",
        "Hi... give me a second. Okay, so roughly,",
    ],
    "Frustrated": [
        "Hi, I've called before about this and no one got back to me, but anyway,",
        "Yeah hi, look, I don't have much time, but",
        "Hi, this is actually my second time calling, so quickly —",
        "Hello — right, let's not go round in circles again.",
        "Yeah, hi. I'll keep this short because I've explained it already —",
        "Hi, honestly I've been passed around a fair bit, so just noting it once:",
        "Look, hi. Third call now, so briefly —",
        "Yeah hello. I'm short on patience today, so straight to it —",
        "Hi. I did fill in your form, but clearly that went nowhere, so",
        "Right, hi. Let me repeat what I told the last person —",
    ],
}

FOLLOWUP_QUESTIONS = {
    "area": [
        "Agent: Any particular area you're targeting?",
        "Agent: Do you have a specific locality in mind?",
        "Agent: Is there a neighbourhood you're leaning towards?",
        "Agent: Which part of the city suits you best?",
        "Agent: Any locality preference at all, even a rough one?",
        "Agent: Are you fixed on a particular pocket, or open across the city?",
    ],
    "property_type": [
        "Agent: What kind of property did you have in mind?",
        "Agent: And what type of property are we talking about?",
        "Agent: Is this for a flat, a house, something else?",
        "Agent: What sort of property should I be looking at for you?",
        "Agent: And the property type — any preference there?",
    ],
    "bhk": [
        "Agent: How many bedrooms are you thinking — a 2 BHK, 3 BHK?",
        "Agent: What configuration works for you, BHK-wise?",
        "Agent: And how many bedrooms do you need?",
        "Agent: Size-wise, what configuration are you after?",
        "Agent: Do you know roughly how many rooms you'd need?",
        "Agent: Any thoughts on the layout — how many bedrooms?",
    ],
    "budget": [
        "Agent: What budget range did you have in mind?",
        "Agent: Do you have a budget figure in mind?",
        "Agent: And roughly what's your budget looking like?",
        "Agent: What sort of price bracket should I work within?",
        "Agent: Have you set a ceiling on the budget?",
        "Agent: Ballpark figure on budget, if you have one?",
    ],
    "amenities": [
        "Agent: Any specific amenities you're looking for?",
        "Agent: Is there anything particular you need, like parking or a gym?",
        "Agent: Any must-have facilities?",
        "Agent: Anything on the amenities side that matters to you?",
        "Agent: Are there facilities you'd consider non-negotiable?",
        "Agent: Anything specific you'd want in the building itself?",
    ],
    "furnishing": [
        "Agent: Would you prefer it furnished or unfurnished?",
        "Agent: And furnishing-wise, any preference?",
        "Agent: Do you want it move-in ready or bare?",
        "Agent: Should I look at furnished options as well?",
        "Agent: Any view on furnishing?",
    ],
}

# Customer non-answers. Sampled without replacement within a single
# transcript so the same non-answer never appears twice in one call.
DEFLECTIONS = {
    "Enthusiastic": [
        "Customer: Oh, haven't decided that part yet, but I'll figure it out soon!",
        "Customer: Not sure yet, honestly, but open to suggestions!",
        "Customer: Ooh, good question — no idea yet, but I'm flexible!",
        "Customer: Haven't got that far, but tell me what people usually go for!",
        "Customer: We're easy on that one, surprise us!",
        "Customer: Still working that bit out, but nothing's ruled out!",
        "Customer: Honestly no clue there yet — happy to be guided!",
        "Customer: That one's still up in the air, but we'll sort it out!",
    ],
    "Neutral": [
        "Customer: Not decided yet, we'll figure that out later.",
        "Customer: No preference there, whatever works.",
        "Customer: Haven't thought about that one, to be honest.",
        "Customer: That's flexible, we can decide later.",
        "Customer: No strong view on that.",
        "Customer: We'll come back to that once the rest is clear.",
        "Customer: Nothing fixed there yet.",
        "Customer: Leave that open for now.",
    ],
    "Hesitant": [
        "Customer: Um, we haven't really thought about that yet.",
        "Customer: Not sure, we're still figuring things out.",
        "Customer: Hmm... I'd have to check with my family on that one.",
        "Customer: I don't want to say something wrong, so let me get back to you.",
        "Customer: That's... honestly, I'm not sure yet.",
        "Customer: We haven't sat down and discussed that part properly.",
        "Customer: Sorry, I really can't say at this point.",
        "Customer: Maybe? I'd rather not commit to that yet.",
    ],
    "Frustrated": [
        "Customer: I don't know, can we just move on?",
        "Customer: Not decided, look, can someone just call me back about this?",
        "Customer: I've no answer for that right now.",
        "Customer: Does it matter at this stage? Let's skip it.",
        "Customer: No idea. Next question.",
        "Customer: I'd rather sort that out with whoever actually calls me.",
        "Customer: Haven't decided, and I'm not deciding on this call.",
        "Customer: Can we park that one, please?",
    ],
}

CLOSERS_GENERIC = [
    "Agent: Thanks for sharing all that, our team will get back to you shortly.",
    "Agent: Perfect, I've noted everything down. Someone will follow up with you soon.",
    "Agent: Got it, thank you. We'll be in touch shortly with the next steps.",
    "Agent: That's all noted. Someone from the team will pick this up soon.",
    "Agent: Lovely, I have what I need. We'll revert shortly.",
    "Agent: Right, that's logged. Expect to hear from us soon.",
    "Agent: Thank you, this is all captured. We'll be in touch.",
    "Agent: Appreciate the details — I'll pass them to the right person.",
]
CLOSERS_SCHEDULE_VISIT = [
    "Agent: Sure, I'll get someone to set up a site visit and confirm a time with you.",
    "Agent: Great, I'll pass this along so our team can arrange a visit at a convenient time.",
    "Agent: Noted, I'll have the team block a slot and confirm the timing with you.",
    "Agent: Understood, someone will coordinate the site visit directly with you.",
    "Agent: Right, I'll arrange for a visit to be scheduled and you'll get a confirmation.",
    "Agent: Perfect, I'll flag this for a viewing and we'll fix a time shortly.",
]
CLOSERS_CALLBACK = [
    "Agent: No problem, I'll make sure someone calls you back at a convenient time.",
    "Agent: Understood, we'll have someone reach out to you soon.",
    "Agent: Sure, I've marked this for a callback — someone will ring you.",
    "Agent: That's fine, I'll get the team to phone you back shortly.",
    "Agent: Noted, expect a call from our side soon.",
    "Agent: Absolutely, someone will get back to you on this number.",
]
CLOSERS_FRUSTRATED = [
    "Agent: I understand, I'm sorry for the trouble — I'll flag this so someone follows up personally.",
    "Agent: I hear you, let me make sure this gets prioritized and someone calls you back soon.",
    "Agent: That's fair, and I apologise. I'm escalating this so it doesn't get lost again.",
    "Agent: Sorry you've had to chase — I'm marking this urgent for follow-up.",
    "Agent: Completely understand the frustration. I'll personally see that someone responds.",
    "Agent: Apologies for the run-around. I'm putting this at the top of the list.",
]

# {type_bhk} placeholder = the property-type+BHK clause text; filled separately.
LOCATION_PHRASES = [
    "in {area}, {city}",
    "somewhere around {area}, in {city}",
    "in the {area} area of {city}",
    "near {area}, {city}",
]
LOCATION_PHRASES_CITY_ONLY = [
    "in {city}",
    "somewhere in {city}",
    "in the {city} market",
]

CODE_MIX_LOCATION = [
    "{area} mein, {city}",
    "{city} ke {area} side",
    "{area} area {city} mein",
]
CODE_MIX_LOCATION_CITY_ONLY = [
    "{city} mein",
    "{city} side",
]

TYPE_BHK_TEMPLATES = [
    "We're looking for a {bhk_phrase} {property_type}",
    "I want a {bhk_phrase} {property_type}",
    "We need {article} {property_type}, {bhk_phrase}",
    "Looking for {article} {property_type}, {bhk_phrase} configuration",
]
TYPE_ONLY_TEMPLATES = [
    "We're looking for {article} {property_type}",
    "I want {article} {property_type}",
    "Looking for {article} {property_type}",
]
BHK_ONLY_TEMPLATES = [
    "We're looking for a {bhk_phrase} place",
    "I want something {bhk_phrase}",
]

CODE_MIX_TYPE_BHK_TEMPLATES = [
    "Mujhe {bhk_phrase} {property_type} chahiye",
    "Hume {property_type} dekhna hai, {bhk_phrase} ka",
    "Basically hume {bhk_phrase} {property_type} chahiye",
    "Yaar {property_type} dekh rahe hain, {bhk_phrase}",
]
CODE_MIX_TYPE_ONLY_TEMPLATES = [
    "Mujhe {property_type} chahiye",
    "Hume {property_type} dekhna hai",
]

BUDGET_PHRASES = [
    "our budget is {budget_phrase}",
    "budget wise we're thinking {budget_phrase}",
    "we're looking at a budget of {budget_phrase}",
    "budget is around {budget_phrase}",
]
CODE_MIX_BUDGET_PHRASES = [
    "budget {budget_phrase} hai",
    "budget {budget_phrase} ke aas paas rakha hai",
    "budget {budget_phrase} tak hi hai",
]

AMENITY_PHRASES = [
    "we'd also like {amenities}",
    "it should have {amenities}",
    "we definitely want {amenities}",
]
CODE_MIX_AMENITY_PHRASES = [
    "{amenities} bhi zaroor chahiye",
    "{amenities} hona zaroori hai",
]

FURNISHING_PHRASES = [
    "and it should be {furnishing_lower}",
    "preferably {furnishing_lower}",
    "we'd prefer it {furnishing_lower}",
]

INTENT_PHRASES = {
    "Buy": ["we're looking to buy", "we want to purchase this outright", "this is for buying, not renting"],
    "Rent": ["we're looking to rent", "this would be on rent", "just for rent, not buying"],
    "Investment": ["this is purely as an investment", "we're buying this as an investment property", "looking at this from an investment angle"],
    "Inquiry": ["just gathering some information for now", "we're only inquiring at this stage", "just exploring options right now"],
    "Schedule Visit": ["we'd like to schedule a site visit", "can we set up a visit to see the place", "we want to actually go see a property"],
    "Request Callback": ["could someone call me back about this", "please have someone get back to me", "I'd prefer a callback to discuss this"],
}

NEGATION_PROPERTY_TEMPLATES = [
    "Not {neg_article} {neg_type}, more like {pos_article} {pos_type}.",
    "{neg_type_cap} is not really for us.",
    "Anything but {neg_article} {neg_type}, honestly.",
    "No {neg_type_lower}s please, we're not interested in that.",
    "We don't want {neg_article} {neg_type}.",
]
CODE_MIX_NEGATION_PROPERTY_TEMPLATES = [
    "{neg_type} nahi chahiye, {pos_type} dekh lo.",
    "{neg_type} bilkul nahi, humein {pos_type} chahiye.",
    "{neg_type} wala mat batana, {pos_type} chahiye bas.",
]

NEGATION_AMENITY_TEMPLATES = [
    "I don't need a {neg_amenity}.",
    "{neg_amenity_cap} is not necessary for us.",
    "We can skip the {neg_amenity}, not bothered about that.",
    "No need for {neg_amenity}, honestly.",
]
CODE_MIX_NEGATION_AMENITY_TEMPLATES = [
    "{neg_amenity} ki zaroorat nahi hai.",
    "{neg_amenity} nahi chahiye bas.",
]

NEGATION_BHK_TEMPLATES = [
    "Not {neg_bhk}, {pos_bhk} is what we need.",
    "We're not looking at {neg_bhk}, {pos_bhk} works better.",
    "{neg_bhk_cap} won't work for us, {pos_bhk} is what we want.",
]
CODE_MIX_NEGATION_BHK_TEMPLATES = [
    "{neg_bhk} nahi, {pos_bhk} chahiye bas.",
    "{neg_bhk} mat dikhaiye, {pos_bhk} hi chahiye.",
]

NEGATION_CITY_TEMPLATES = [
    "Not in {neg_city}, only {pos_city} works for us.",
    "We're not considering {neg_city} at all, just {pos_city}.",
    "{neg_city_cap} is out of the question, {pos_city} is what we want.",
]
CODE_MIX_NEGATION_CITY_TEMPLATES = [
    "{neg_city} mein nahi, sirf {pos_city} mein dekhna hai.",
    "{neg_city} bilkul nahi, {pos_city} hi chahiye.",
]

MULTI_CITY_TEMPLATES = [
    "I'm currently based in {distractor_city} but I'm looking at options in {target_city}.",
    "We live in {distractor_city} right now, though the property should be in {target_city}.",
    "{distractor_city} is where I am at the moment, {target_city} is what I'm focused on.",
    "I work in {distractor_city}, but I want something in {target_city}.",
]
CODE_MIX_MULTI_CITY_TEMPLATES = [
    "Hum {distractor_city} mein rehte hain abhi, par {target_city} mein dekhna hai.",
    "{distractor_city} se hoon main, lekin {target_city} mein dekhna hai.",
    "Abhi {distractor_city} mein hoon, par property {target_city} mein chahiye.",
]


def build_area_weights(rng, areas: List[str]) -> List[float]:
    """Upweights areas whose name embeds a target city word (e.g. 'Nashik
    Road', 'Nashik Pune Road', 'Chandan Nagar Pune'), so the NER training
    set has good coverage of the "area contains city substring" hard case
    the model must learn to tag as one whole AREA span, not split on the
    embedded city word.
    """
    weights = []
    for area in areas:
        if is_collision_area(area):
            weights.append(2.0)
        else:
            weights.append(1.0)
    return weights


def is_collision_area(area: str) -> bool:
    return any(re.search(rf"\b{re.escape(c)}\b", area, re.IGNORECASE) for c in TARGET_CITIES)


class LookupTables:
    """Loads all CSV lookup values. No entity value in this generator is
    ever hardcoded outside of this class' CSV reads, per CLAUDE.md section 7.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR) -> None:
        self.locations = pd.read_csv(dataset_dir / "locations.csv")
        self.property_types = pd.read_csv(dataset_dir / "property_type.csv")
        self.amenities = pd.read_csv(dataset_dir / "amenities.csv")
        self.distractor_cities = pd.read_csv(dataset_dir / "distractor_cities.csv")
        self.property_synonyms = pd.read_csv(dataset_dir / "property_type_synonyms.csv")

        self.areas_by_city: Dict[str, List[str]] = {
            city: self.locations.loc[self.locations.city == city, "area"].tolist()
            for city in TARGET_CITIES
        }
        self.property_type_names: List[str] = self.property_types["property_type_name"].tolist()
        self.property_type_category: Dict[str, str] = dict(
            zip(self.property_types["property_type_name"], self.property_types["category"])
        )
        self.amenity_names: List[str] = self.amenities["amenity_name"].tolist()
        self.distractor_city_names: List[str] = self.distractor_cities["city"].tolist()

        self.canonical_to_synonyms: Dict[str, List[str]] = {}
        for _, row in self.property_synonyms.iterrows():
            self.canonical_to_synonyms.setdefault(row["property_type_name"], []).append(row["synonym"])

        self.non_rare_property_types: List[str] = [
            p for p in self.property_type_names if p not in RARE_TYPES
        ]

        logger.info(
            "Loaded lookups: %d Nashik areas, %d Pune areas, %d property types, "
            "%d amenities, %d distractor cities",
            len(self.areas_by_city["Nashik"]),
            len(self.areas_by_city["Pune"]),
            len(self.property_type_names),
            len(self.amenity_names),
            len(self.distractor_city_names),
        )


def render_property_type(rng, lookups: LookupTables, canonical: str) -> str:
    """Returns a surface phrase for a property type. ground_truth always
    stores `canonical`, regardless of which surface form is chosen here.
    """
    if canonical in EXTRA_SURFACE_FORMS:
        return rng.choice(EXTRA_SURFACE_FORMS[canonical])
    options = [canonical.lower()] + lookups.canonical_to_synonyms.get(canonical, [])
    return rng.choice(options)


def generate_bhk(rng) -> Tuple[Any, str, str]:
    """Returns (ground_truth_bhk, phrase, form_tag).

    Phrases are article-free ("1 RK", not "a 1 RK") because every calling
    template already supplies its own article — "looking for a {bhk_phrase}
    {property_type}" would otherwise render "a a 1RK".

    There is no "Studio" BHK value: a studio has no bedroom count, so it is
    represented as the property type "Studio Apartment" instead (see
    NO_BHK_PROPERTY_TYPES).
    """
    roll = rng.random()
    if roll < 0.10:
        phrase = rng.choice(["1RK", "1 RK", "1RK setup", "1 RK unit"])
        return "1RK", phrase, "1rk"

    n = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.40, 0.35, 0.12, 0.03]))
    subform = rng.choice(["spaced", "bare", "worded"], p=[0.50, 0.35, 0.15])
    if subform == "spaced":
        phrase = f"{n} BHK"
    elif subform == "bare":
        phrase = f"{n}bhk"
    else:
        if rng.random() < 0.5:
            phrase = f"{NUM_WORDS[n]} BHK"
        else:
            unit = "bedroom" if n == 1 else "bedrooms"
            phrase = f"{n} {unit}"
    return n, phrase, subform


def generate_code_mixed_bhk_phrase(rng, bhk_value: Any) -> str:
    if bhk_value == "1RK":
        return rng.choice(["1RK", "ek RK"])
    n = bhk_value
    if rng.random() < 0.4:
        return f"{HINDI_NUM_WORDS[n]} BHK"
    return f"{n} BHK"


def generate_budget(rng, category: str) -> Tuple[float, str, str]:
    """Returns (ground_truth_budget_lakhs, phrase, form_tag)."""
    if category in ("Commercial", "Industrial", "Mixed Use"):
        low, high = 40, 500
    else:
        low, high = 15, 200

    amount_lakhs = float(rng.uniform(low, high))
    form = rng.choice(
        ["lakh", "crore", "cr", "exact", "spelled_out", "range"],
        p=[0.30, 0.10, 0.15, 0.15, 0.15, 0.15],
    )

    if form in ("crore", "cr") and amount_lakhs < 50:
        amount_lakhs = float(rng.uniform(50, min(high, 500)))

    if form == "lakh":
        val = max(5, round(amount_lakhs / 5) * 5)
        unit = rng.choice(["lakh", "lakhs"])
        return float(val), f"{val} {unit}", form

    if form in ("crore", "cr"):
        crore_val = round(amount_lakhs / 100, 2)
        if form == "crore":
            unit = rng.choice(["crore", "crores"])
            return crore_val * 100, f"{crore_val} {unit}", form
        return crore_val * 100, f"{crore_val}cr", form

    if form == "exact":
        exact_rupees = int(round(amount_lakhs * 100000, -4))
        style = rng.choice(["plain", "rupee_symbol", "exactly"])
        if style == "plain":
            phrase = f"{exact_rupees} rupees"
        elif style == "rupee_symbol":
            phrase = f"Rs {exact_rupees}"
        else:
            phrase = f"exactly {exact_rupees}"
        return exact_rupees / 100000, phrase, form

    if form == "spelled_out":
        val = max(5, round(amount_lakhs / 5) * 5)
        return float(val), f"{spell_number(val)} lakhs", form

    # range
    low_val = max(5, round(amount_lakhs / 5) * 5)
    high_val = low_val + int(rng.choice([5, 10, 15, 20]))
    midpoint = (low_val + high_val) / 2
    return midpoint, f"{low_val} to {high_val} lakhs", form


def pick_amenities(rng, lookups: LookupTables, completeness: str) -> List[str]:
    if completeness == "full":
        k = int(rng.choice([2, 3, 4]))
    elif completeness == "partial":
        k = int(rng.choice([0, 1, 2], p=[0.35, 0.35, 0.30]))
    else:
        k = int(rng.choice([0, 1], p=[0.75, 0.25]))
    if k == 0:
        return []
    return list(rng.choice(lookups.amenity_names, size=k, replace=False))


def join_amenities(rng, amenities: List[str]) -> str:
    lowered = [a.lower() for a in amenities]
    if len(lowered) == 1:
        return lowered[0]
    return ", ".join(lowered[:-1]) + f" and {lowered[-1]}"


def join_amenities_code_mixed(rng, amenities: List[str]) -> str:
    lowered = [a.lower() for a in amenities]
    if len(lowered) == 1:
        return lowered[0]
    return " aur ".join([", ".join(lowered[:-1]), lowered[-1]]) if len(lowered) > 2 else " aur ".join(lowered)


def decide_revealed_fields(rng, completeness: str) -> Dict[str, bool]:
    always = {"city": True}
    if completeness == "full":
        chosen = {
            "area": rng.random() < 0.85,
            "property_type": True,
            "bhk": rng.random() < 0.9,
            "budget": True,
            "amenities": rng.random() < 0.8,
            "furnishing": rng.random() < 0.75,
        }
    elif completeness == "partial":
        candidates = ["area", "property_type", "bhk", "budget", "amenities", "furnishing"]
        n_reveal = int(rng.choice([2, 3, 4]))
        chosen_keys = set(rng.choice(candidates, size=min(n_reveal, len(candidates)), replace=False))
        chosen = {k: (k in chosen_keys) for k in candidates}
    else:  # vague
        always["city"] = rng.random() < 0.9
        candidates = ["area", "property_type", "bhk", "budget", "amenities", "furnishing"]
        n_reveal = int(rng.choice([0, 1], p=[0.4, 0.6]))
        chosen_keys = set(rng.choice(candidates, size=n_reveal, replace=False)) if n_reveal else set()
        chosen = {k: (k in chosen_keys) for k in candidates}

    chosen.update(always)
    return chosen


def build_transcript(
    rng,
    lookups: LookupTables,
    forced_property_type: Optional[str] = None,
    force_negation: Optional[bool] = None,
    force_code_mixed: Optional[bool] = None,
    force_multi_city: Optional[bool] = None,
    force_telegraphic: Optional[bool] = None,
) -> Dict[str, Any]:
    city = str(rng.choice(TARGET_CITIES, p=CITY_WEIGHTS))
    areas = lookups.areas_by_city[city]
    area_weights = build_area_weights(rng, areas)
    area = str(rng.choice(areas, p=[w / sum(area_weights) for w in area_weights]))
    area_collision = is_collision_area(area)

    if forced_property_type is not None:
        property_type = forced_property_type
    else:
        property_type = str(rng.choice(lookups.non_rare_property_types))
    category = lookups.property_type_category[property_type]
    is_rare = property_type in RARE_TYPES

    intent = str(rng.choice(INTENTS))
    sentiment = str(rng.choice(SENTIMENTS))
    completeness = str(rng.choice(["full", "partial", "vague"], p=[0.35, 0.40, 0.25]))

    is_code_mixed = bool(rng.random() < P_CODE_MIX) if force_code_mixed is None else force_code_mixed
    is_telegraphic = bool(rng.random() < P_TELEGRAPHIC) if force_telegraphic is None else force_telegraphic
    if is_telegraphic and completeness == "vague":
        completeness = str(rng.choice(["full", "partial"]))
    has_negation = bool(rng.random() < P_NEGATION) if force_negation is None else force_negation
    is_multi_city = bool(rng.random() < P_MULTI_CITY) if force_multi_city is None else force_multi_city

    revealed = decide_revealed_fields(rng, completeness)
    if forced_property_type is not None:
        # Reserved rare-type transcripts must always state the type in
        # text, otherwise the reservation wouldn't achieve its purpose
        # (guaranteeing NER training examples that actually mention it).
        revealed["property_type"] = True
    if is_multi_city:
        # The multi-city clause always names the target city by name
        # ("...but I want something in {target_city}"), so the city is
        # revealed in text regardless of the independent reveal roll —
        # otherwise ground_truth.city would be None while the city
        # literally appears in the transcript.
        revealed["city"] = True
    if category in NO_BHK_CATEGORIES or "Plot" in property_type or "Land" in property_type:
        # BHK (bedroom count) doesn't apply to non-residential property —
        # a warehouse, shop, or plot doesn't get described in bedrooms.
        revealed["bhk"] = False
    if property_type in NO_BHK_PROPERTY_TYPES and revealed["property_type"]:
        # "Studio apartment, 2 BHK" is self-contradictory. Suppressed only
        # when the type is actually stated: if it never surfaces in the
        # text, ground_truth.property_type is None too, so a BHK number
        # contradicts nothing.
        revealed["bhk"] = False

    bhk_value = bhk_phrase = bhk_form = None
    if revealed["bhk"]:
        bhk_value, bhk_phrase, bhk_form = generate_bhk(rng)

    budget_lakhs = budget_phrase = budget_form = None
    if revealed["budget"]:
        budget_lakhs, budget_phrase, budget_form = generate_budget(rng, category)

    amenities: List[str] = []
    if revealed["amenities"]:
        amenities = pick_amenities(rng, lookups, completeness)
        revealed["amenities"] = bool(amenities)

    furnishing = str(rng.choice(FURNISHING_OPTIONS)) if revealed["furnishing"] else None

    negation_target = None
    if has_negation:
        # Only negate a field that's actually revealed elsewhere in the
        # transcript — otherwise the negation clause states a positive
        # value ("more like an apartment", "2 BHK chahiye") for a field
        # ground_truth records as never mentioned, and (for bhk/city) can
        # even collide with the negated value itself since there's no real
        # positive value to contrast it with.
        candidates = ["amenity"]
        if revealed["property_type"]:
            candidates.append("property_type")
        if revealed["bhk"]:
            candidates.append("bhk")
        if revealed.get("city", True):
            candidates.append("city")
        negation_target = str(rng.choice(candidates))

    distractor_city = str(rng.choice(lookups.distractor_city_names)) if is_multi_city else None

    text = render_dialogue(
        rng=rng,
        lookups=lookups,
        city=city,
        area=area,
        property_type=property_type,
        revealed=revealed,
        bhk_value=bhk_value,
        bhk_phrase=bhk_phrase,
        budget_phrase=budget_phrase,
        amenities=amenities,
        furnishing=furnishing,
        intent=intent,
        sentiment=sentiment,
        completeness=completeness,
        is_code_mixed=is_code_mixed,
        is_telegraphic=is_telegraphic,
        has_negation=has_negation,
        negation_target=negation_target,
        is_multi_city=is_multi_city,
        distractor_city=distractor_city,
    )

    ground_truth = {
        "city": city if revealed.get("city", True) else None,
        "area": area if revealed.get("area") else None,
        "property_type": property_type if revealed["property_type"] else None,
        "category": category if revealed["property_type"] else None,
        "bhk": bhk_value if revealed["bhk"] else None,
        "budget_lakhs": round(budget_lakhs, 2) if revealed["budget"] and budget_lakhs is not None else None,
        "budget_text": budget_phrase if revealed["budget"] else None,
        "amenities": amenities,
        "furnishing": furnishing,
    }

    metadata = {
        "intent": intent,
        "sentiment": sentiment,
        "completeness": completeness,
        "flags": {
            "has_negation": has_negation,
            "negation_target": negation_target,
            "is_code_mixed": is_code_mixed,
            "is_multi_city": is_multi_city,
            "distractor_city": distractor_city,
            "is_telegraphic": is_telegraphic,
            "is_rare_property_type": is_rare,
            "rare_property_type_name": property_type if is_rare else None,
            "budget_form": budget_form,
            "bhk_form": bhk_form,
            "area_contains_city_substring": area_collision,
        },
    }

    return {"text": text, "ground_truth": ground_truth, "metadata": metadata}


def render_dialogue(
    rng,
    lookups: LookupTables,
    city: str,
    area: str,
    property_type: str,
    revealed: Dict[str, bool],
    bhk_value: Any,
    bhk_phrase: Optional[str],
    budget_phrase: Optional[str],
    amenities: List[str],
    furnishing: Optional[str],
    intent: str,
    sentiment: str,
    completeness: str,
    is_code_mixed: bool,
    is_telegraphic: bool,
    has_negation: bool,
    negation_target: Optional[str],
    is_multi_city: bool,
    distractor_city: Optional[str],
) -> str:
    turns = [rng.choice(OPENERS)]

    property_surface = render_property_type(rng, lookups, property_type) if revealed["property_type"] else None
    cm_bhk_phrase = generate_code_mixed_bhk_phrase(rng, bhk_value) if (is_code_mixed and revealed["bhk"]) else bhk_phrase

    if is_telegraphic:
        customer_line = build_telegraphic_line(
            rng, city, area, revealed, property_surface, cm_bhk_phrase, budget_phrase,
            amenities, furnishing, is_code_mixed,
        )
    else:
        customer_line = build_flowing_line(
            rng, lookups, city, area, revealed, property_surface, cm_bhk_phrase, budget_phrase,
            amenities, furnishing, intent, sentiment, is_code_mixed,
        )

    if has_negation:
        customer_line += " " + build_negation_clause(rng, lookups, negation_target, property_type, property_surface,
                                                       bhk_value, bhk_phrase, city, amenities, is_code_mixed)

    if is_multi_city:
        bank = CODE_MIX_MULTI_CITY_TEMPLATES if is_code_mixed else MULTI_CITY_TEMPLATES
        customer_line += " " + str(rng.choice(bank)).format(distractor_city=distractor_city, target_city=city)

    turns.append(f"Customer: {customer_line}")

    missing_fields = [f for f in ("area", "property_type", "bhk", "budget", "amenities", "furnishing") if not revealed[f]]
    n_followups = 0 if is_telegraphic else int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
    rng.shuffle(missing_fields) if missing_fields else None
    # Drawn without replacement so a caller never gives the identical
    # non-answer twice in one call.
    deflection_pool = list(DEFLECTIONS[sentiment])
    rng.shuffle(deflection_pool)
    for field in missing_fields[:n_followups]:
        turns.append(str(rng.choice(FOLLOWUP_QUESTIONS[field])))
        turns.append(str(deflection_pool.pop()))

    if intent == "Schedule Visit":
        closer_bank = CLOSERS_SCHEDULE_VISIT
    elif intent == "Request Callback":
        closer_bank = CLOSERS_CALLBACK
    elif sentiment == "Frustrated":
        closer_bank = CLOSERS_FRUSTRATED
    else:
        closer_bank = CLOSERS_GENERIC
    turns.append(str(rng.choice(closer_bank)))

    return "\n".join(turns)


def build_flowing_line(
    rng, lookups, city, area, revealed, property_surface, bhk_phrase, budget_phrase,
    amenities, furnishing, intent, sentiment, is_code_mixed,
) -> str:
    opener = str(rng.choice(SENTIMENT_OPENERS[sentiment]))

    main_clause = ""
    if revealed["property_type"] and revealed["bhk"]:
        bank = CODE_MIX_TYPE_BHK_TEMPLATES if is_code_mixed else TYPE_BHK_TEMPLATES
        main_clause = str(rng.choice(bank)).format(
            bhk_phrase=bhk_phrase, property_type=property_surface,
            article=indefinite_article(property_surface),
        )
    elif revealed["property_type"]:
        bank = CODE_MIX_TYPE_ONLY_TEMPLATES if is_code_mixed else TYPE_ONLY_TEMPLATES
        main_clause = str(rng.choice(bank)).format(
            property_type=property_surface, article=indefinite_article(property_surface)
        )
    elif revealed["bhk"]:
        main_clause = str(rng.choice(BHK_ONLY_TEMPLATES)).format(bhk_phrase=bhk_phrase)

    location_clause = ""
    if revealed.get("area"):
        bank = CODE_MIX_LOCATION if is_code_mixed else LOCATION_PHRASES
        location_clause = str(rng.choice(bank)).format(area=area, city=city)
    elif revealed.get("city", True):
        bank = CODE_MIX_LOCATION_CITY_ONLY if is_code_mixed else LOCATION_PHRASES_CITY_ONLY
        location_clause = str(rng.choice(bank)).format(city=city)

    parts = [p for p in (main_clause, location_clause) if p]
    line = opener
    if parts:
        line += " " + " ".join(parts) + "."

    if revealed["budget"]:
        bank = CODE_MIX_BUDGET_PHRASES if is_code_mixed else BUDGET_PHRASES
        line += " " + cap_first(str(rng.choice(bank)).format(budget_phrase=budget_phrase)) + "."

    if revealed["amenities"] and amenities:
        joined = join_amenities_code_mixed(rng, amenities) if is_code_mixed else join_amenities(rng, amenities)
        bank = CODE_MIX_AMENITY_PHRASES if is_code_mixed else AMENITY_PHRASES
        line += " " + cap_first(str(rng.choice(bank)).format(amenities=joined)) + "."

    if revealed["furnishing"] and furnishing:
        line += " " + cap_first(str(rng.choice(FURNISHING_PHRASES)).format(furnishing_lower=furnishing.lower())) + "."

    line += " " + cap_first(str(rng.choice(INTENT_PHRASES[intent]))) + "."

    return re.sub(r"\s+", " ", line).strip()


def build_telegraphic_line(
    rng, city, area, revealed, property_surface, bhk_phrase, budget_phrase,
    amenities, furnishing, is_code_mixed,
) -> str:
    fragments = []
    if revealed.get("area"):
        fragments.append(f"{area}, {city}.")
    elif revealed.get("city", True):
        fragments.append(f"{city}.")
    if revealed["bhk"] and bhk_phrase:
        fragments.append(f"{bhk_phrase}.")
    if revealed["property_type"] and property_surface:
        fragments.append(f"{cap_first(property_surface)}.")
    if revealed["budget"] and budget_phrase:
        fragments.append(f"{cap_first(budget_phrase)}.")
    if revealed["amenities"] and amenities:
        joined = join_amenities_code_mixed(rng, amenities) if is_code_mixed else join_amenities(rng, amenities)
        suffix = " chahiye." if is_code_mixed else " needed."
        fragments.append(f"{cap_first(joined)}{suffix}")
    if revealed["furnishing"] and furnishing:
        fragments.append(f"{furnishing}.")
    if not fragments:
        fragments.append(f"{city}.")
    return " ".join(fragments)


def build_negation_clause(rng, lookups, target, property_type, property_surface, bhk_value, bhk_phrase, city, amenities, is_code_mixed) -> str:
    if target == "property_type":
        others = [p for p in lookups.property_type_names if p != property_type]
        neg_type = str(rng.choice(others))
        pos_type = property_surface or property_type.lower()
        bank = CODE_MIX_NEGATION_PROPERTY_TEMPLATES if is_code_mixed else NEGATION_PROPERTY_TEMPLATES
        return str(rng.choice(bank)).format(
            neg_type=neg_type.lower(), neg_type_cap=neg_type, neg_type_lower=neg_type.lower(),
            pos_type=pos_type,
            neg_article=indefinite_article(neg_type), pos_article=indefinite_article(pos_type),
        )

    if target == "amenity":
        pool = [a for a in lookups.amenity_names if a not in amenities]
        neg_amenity = str(rng.choice(pool))
        bank = CODE_MIX_NEGATION_AMENITY_TEMPLATES if is_code_mixed else NEGATION_AMENITY_TEMPLATES
        return str(rng.choice(bank)).format(neg_amenity=neg_amenity.lower(), neg_amenity_cap=neg_amenity)

    if target == "bhk":
        pos_bhk_phrase = bhk_phrase or "2 BHK"
        candidates = [n for n in (1, 2, 3, 4, 5) if n != bhk_value]
        neg_bhk_phrase = f"{rng.choice(candidates)} BHK"
        bank = CODE_MIX_NEGATION_BHK_TEMPLATES if is_code_mixed else NEGATION_BHK_TEMPLATES
        return str(rng.choice(bank)).format(
            neg_bhk=neg_bhk_phrase, neg_bhk_cap=neg_bhk_phrase, pos_bhk=pos_bhk_phrase,
        )

    # target == "city"
    neg_city = [c for c in TARGET_CITIES if c != city][0]
    bank = CODE_MIX_NEGATION_CITY_TEMPLATES if is_code_mixed else NEGATION_CITY_TEMPLATES
    return str(rng.choice(bank)).format(neg_city=neg_city, neg_city_cap=neg_city, pos_city=city)


# --------------------------------------------------------------------------
# Pool assembly
# --------------------------------------------------------------------------

def flag_counter(pool: List[Dict[str, Any]], key: str) -> int:
    return sum(1 for t in pool if t["metadata"]["flags"][key])


def top_up(pool, rng, lookups, flag_key, minimum, force_kwarg, id_gen):
    count = flag_counter(pool, flag_key)
    while count < minimum and len(pool) < MAX_POOL_SIZE:
        pool.append(build_transcript(rng, lookups, **{force_kwarg: True}))
        count += 1
    if count < minimum:
        logger.warning("Could not reach %s minimum of %d (got %d) within MAX_POOL_SIZE=%d",
                        flag_key, minimum, count, MAX_POOL_SIZE)
    else:
        logger.info("%s: %d (minimum %d)", flag_key, count, minimum)


def generate_pool(seed: int = SEED, initial_size: int = INITIAL_POOL_SIZE) -> List[Dict[str, Any]]:
    rng = default_rng(seed)
    lookups = LookupTables()

    pool: List[Dict[str, Any]] = []

    for rare_type in RARE_TYPES:
        for _ in range(RARE_TYPE_MIN_EACH):
            pool.append(build_transcript(rng, lookups, forced_property_type=rare_type))
    logger.info("Reserved %d rare-type transcripts (%d each x %d types)",
                len(pool), RARE_TYPE_MIN_EACH, len(RARE_TYPES))

    remaining = max(0, initial_size - len(pool))
    for _ in range(remaining):
        pool.append(build_transcript(rng, lookups))
    logger.info("Generated %d general-pool transcripts, total pool = %d", remaining, len(pool))

    top_up(pool, rng, lookups, "has_negation", NEGATION_MIN, "force_negation", None)
    top_up(pool, rng, lookups, "is_code_mixed", CODE_MIX_MIN, "force_code_mixed", None)
    top_up(pool, rng, lookups, "is_multi_city", MULTI_CITY_MIN, "force_multi_city", None)
    top_up(pool, rng, lookups, "is_telegraphic", TELEGRAPHIC_MIN, "force_telegraphic", None)

    rng.shuffle(pool)
    return pool


def summarize(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(pool)
    city_counts: Dict[str, int] = {}
    rare_counts: Dict[str, int] = {t: 0 for t in RARE_TYPES}
    budget_form_counts: Dict[str, int] = {}
    bhk_form_counts: Dict[str, int] = {}

    for t in pool:
        city = t["ground_truth"]["city"]
        if city:
            city_counts[city] = city_counts.get(city, 0) + 1
        flags = t["metadata"]["flags"]
        if flags["rare_property_type_name"]:
            rare_counts[flags["rare_property_type_name"]] += 1
        if flags["budget_form"]:
            budget_form_counts[flags["budget_form"]] = budget_form_counts.get(flags["budget_form"], 0) + 1
        if flags["bhk_form"]:
            bhk_form_counts[flags["bhk_form"]] = bhk_form_counts.get(flags["bhk_form"], 0) + 1

    return {
        "total": total,
        "city_counts": city_counts,
        "negation_count": flag_counter(pool, "has_negation"),
        "code_mixed_count": flag_counter(pool, "is_code_mixed"),
        "multi_city_count": flag_counter(pool, "is_multi_city"),
        "telegraphic_count": flag_counter(pool, "is_telegraphic"),
        "rare_type_counts": rare_counts,
        "budget_form_counts": budget_form_counts,
        "bhk_form_counts": bhk_form_counts,
    }


def write_pool(pool: List[Dict[str, Any]], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "synthetic_transcripts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, transcript in enumerate(pool, start=1):
            record = {
                "transcript_id": f"tr_{idx:04d}",
                "text": transcript["text"],
                "ground_truth": transcript["ground_truth"],
                "metadata": transcript["metadata"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d transcripts to %s", len(pool), jsonl_path)

    summary = summarize(pool)
    summary_path = output_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Wrote summary to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EstateIQ call transcripts.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--initial-size", type=int, default=INITIAL_POOL_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pool = generate_pool(seed=args.seed, initial_size=args.initial_size)
    write_pool(pool)


if __name__ == "__main__":
    main()
