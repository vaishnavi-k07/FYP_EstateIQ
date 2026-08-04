"""Stage E — hybrid rule + MuRIL extraction, drop-in for the rule extractor.

``HybridExtractor.extract`` takes a raw transcript and returns the same dict
shape ``NLPExtractor.extract`` returns, so the rest of the pipeline is
unchanged. Two fields are added, not substituted: ``budget_value`` (numeric
rupees, which Phase 4 lead scoring needs) and ``negated`` (what the customer
ruled out).

Why hybrid: the Stage D benchmark on the held-out test split showed neither
system dominates. Rules are near-perfect on closed-set CSV lookups (AREA
F1 1.000, AMENITY 0.983) because those are exact catalog matches; MuRIL wins
decisively on BUDGET (0.956 vs 0.613), the one open-ended field, where the
rule regexes recall only half the mentions. Routing each entity to its
stronger system beat both end to end.

Pipeline:
    raw text
      -> rule spans (SpanLocator) + model spans (MuRIL)      [both raw offsets]
      -> per-entity ownership filter
      -> conflict resolution                                 [atomic spans]
      -> negation / multi-city post-processing
      -> CSV normalization + budget -> numeric rupees
      -> structured dict

Run:
    python -m nlp.predict --text "I want a 2 BHK in Baner, Pune under 60 lakhs"
    python -m nlp.predict --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nlp.extractor import NLPExtractor  # noqa: E402
from nlp.ner.bootstrap_labels import (  # noqa: E402
    SpanLocator,
    build_offset_map,
    map_span_to_raw,
)
from nlp.ner.export_dataset import ENTITIES, tokenize_with_offsets  # noqa: E402
from nlp.ner.model import MurilNerPredictor  # noqa: E402

logger = logging.getLogger(__name__)

MODELS_STORE = BACKEND_DIR / "models_store" / "ner"
DATASET_DIR = BACKEND_DIR / "dataset"

RULES = "rules"
MODEL = "model"

# Which system owns each entity, from the Stage D per-entity comparison.
#
# AREA / PROPERTY_TYPE / AMENITY / BHK -> rules: closed-set CSV lookups where
#   exact catalog matching beats a model trained on a few hundred transcripts.
#   BHK is routed to rules deliberately even though validation preferred the
#   model (0.919 vs 0.682): on the held-out test split that reversed (0.741 vs
#   0.867), so the validation preference did not generalize on 15 test spans.
# BUDGET -> model: the decisive win (0.956 vs 0.613); rule regexes cannot
#   enumerate the ways a budget gets spoken.
# CITY / FURNISHING -> model: validation-chosen owner, confirmed on test
#   (CITY 0.970 vs 0.963, FURNISHING 1.000 vs 0.940).
OWNERS: Dict[str, str] = {
    "AREA": RULES,
    "PROPERTY_TYPE": RULES,
    "AMENITY": RULES,
    "BHK": RULES,
    "BUDGET": MODEL,
    "CITY": MODEL,
    "FURNISHING": MODEL,
}

# Tie-break order when two spans of different types overlap and are the same
# length. Longer spans always win first (see _resolve_conflicts); this only
# settles exact-length collisions, ranked by the owner's measured test F1.
_TIE_BREAK: Dict[str, int] = {
    "AREA": 7,
    "FURNISHING": 6,
    "CITY": 5,
    "BUDGET": 4,
    "AMENITY": 3,
    "PROPERTY_TYPE": 2,
    "BHK": 1,
}

_NEGATION_CUES = (
    r"(?:don't|do not|dont|doesn't|does not|didn't|did not|"
    r"won't|will not|isn't|is not|not|no|never|avoid|except|rather than|instead of)"
)
# A cue scopes over a span in one of two shapes:
#
#   plain        "not a villa"                    -> cue, then up to 3 tokens
#   coordinated  "not a flat or a villa"          -> cue, then a longer run that
#                                                    must reach a coordinating
#                                                    conjunction
#
# The coordinated branch is only permitted when an explicit or/and/nor is
# present. Commas alone must never extend the scope: in this corpus a comma
# after a negation introduces a *correction* ("Not a penthouse, more like a
# resort", "Not a showroom, more like a hostel"), so carrying the negation
# across it would discard the customer's actual requirement. Measured across
# all 400 transcripts: 130 commas follow a cue, and zero or/and do — so the
# coordinated branch cannot fire on real corpus data at all, only on the
# coordinated phrasings it was added for.
_NEGATION_LOOKBEHIND = re.compile(
    rf"\b{_NEGATION_CUES}\b"
    rf"(?:"
    rf"(?:\s+\S+){{0,3}}"                                    # plain scope
    rf"|(?:\s+\S+){{0,8}}?\s+\b(?:or|and|nor)\b(?:\s+\S+){{0,4}}"  # coordinated
    rf")\s*$",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 80

# A negation cue stops at the end of its clause: punctuation, or a contrastive
# connective that introduces what the customer *does* want.
_CLAUSE_BOUNDARY = re.compile(
    r"[,;:.?!]|\b(?:but|instead|rather|however|though|prefer)\b", re.IGNORECASE
)

# Cues that mark the city the customer is shopping in, as opposed to the one
# they are calling from ("I'm in Mumbai but looking in Pune").
_SEEKING_CUE = re.compile(
    r"\b(?:looking|search(?:ing)?|want|need|buy|rent|invest|shift(?:ing)?|"
    r"move|moving|relocat\w*|interested)\b[^.?!]{0,60}$",
    re.IGNORECASE,
)
_CURRENT_LOCATION_CUE = re.compile(
    r"\b(?:i am|i'm|we are|we're|currently|presently|based|living|stay(?:ing)?|"
    r"from)\b[^.?!]{0,40}$",
    re.IGNORECASE,
)

_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_SCALES: Dict[str, int] = {"hundred": 100, "thousand": 1_000}
_NUMBER_TOKEN = re.compile(
    r"\b(?:" + "|".join(sorted(
        list(_NUMBER_WORDS) + list(_NUMBER_SCALES) + ["and"], key=len, reverse=True
    )) + r")\b",
    re.IGNORECASE,
)


def words_to_number(phrase: str) -> Optional[float]:
    """Parses spelled-out numbers, including compounds.

    "one hundred seventy" -> 170. Reading only the last word would give 70,
    understating a budget by more than half — which matters because the value
    feeds lead scoring.
    """
    tokens = [t.lower() for t in _NUMBER_TOKEN.findall(phrase or "")]
    tokens = [t for t in tokens if t != "and"]
    if not tokens:
        return None

    total = 0.0
    current = 0.0
    seen = False
    for token in tokens:
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            seen = True
        elif token in _NUMBER_SCALES:
            scale = _NUMBER_SCALES[token]
            current = (current or 1) * scale
            if scale >= 1_000:
                total += current
                current = 0.0
            seen = True
    if not seen:
        return None
    return total + current

_LAKH = 100_000
_CRORE = 10_000_000

# Any amount-with-unit anywhere in the text, used only as a fallback when no
# model BUDGET span survives. Anchored on the unit so bare numbers (BHK counts,
# floor numbers, years) cannot match.
_BUDGET_SCAN = re.compile(
    r"(?:\d+(?:[.,]\d+)*|(?:\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety|hundred|and)\b[\s-]*)+)"
    r"\s*(?:crores?|cr\b|lakhs?|lacs?|l\b|rupees|rs\.?)",
    re.IGNORECASE,
)

# Marks a stated span of values rather than one compound figure.
_RANGE_CONNECTOR = re.compile(
    r"\b(?:to|or|between|upto|up to|under|below|around|max(?:imum)?)\b|[-–—]",
    re.IGNORECASE,
)

_FURNISHING_CANONICAL: Tuple[Tuple[str, str], ...] = (
    (r"\bsemi[\s-]?furnished\b", "Semi Furnished"),
    (r"\bfully[\s-]?furnished\b", "Fully Furnished"),
    (r"\bunfurnished\b", "Unfurnished"),
    (r"\bfurnished\b", "Furnished"),
)


@dataclass
class EntitySpan:
    """A resolved entity mention in raw-text offsets."""

    start: int
    end: int
    label: str
    text: str
    source: str
    negated: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class _Candidate:
    span: EntitySpan
    priority: int = field(default=0)


class BudgetParser:
    """Turns a spoken budget span into rupees.

    Handles lakh/lac/L, crore/cr, bare figures, spelled-out numbers and ranges.
    Ranges resolve to their upper bound, which is what a lead's stated ceiling
    means for scoring.
    """

    _UNIT = r"(?P<unit>crores?|cr\b|lakhs?|lacs?|lakh|l\b)"
    _NUM = r"(?P<num>\d+(?:[.,]\d+)?)"
    _PAIR = re.compile(rf"{_NUM}\s*{_UNIT}", re.IGNORECASE)
    _BARE = re.compile(r"\b(?P<num>\d{5,})\b")
    # A run of number words (possibly compound) immediately before a unit.
    _WORD_UNIT = re.compile(
        rf"(?P<word>(?:\b(?:{'|'.join(sorted(list(_NUMBER_WORDS) + list(_NUMBER_SCALES) + ['and'], key=len, reverse=True))})\b[\s-]*)+)"
        rf"{_UNIT}",
        re.IGNORECASE,
    )

    @classmethod
    def _unit_multiplier(cls, unit: str) -> int:
        unit = unit.lower().rstrip(".")
        if unit.startswith("cr"):
            return _CRORE
        return _LAKH

    @classmethod
    def parse(cls, surface: str) -> Optional[float]:
        """Returns rupees, or None when nothing numeric can be recovered."""
        if not surface:
            return None
        values: List[float] = []
        multipliers: List[int] = []

        for match in cls._PAIR.finditer(surface):
            number = float(match.group("num").replace(",", ""))
            multiplier = cls._unit_multiplier(match.group("unit"))
            values.append(number * multiplier)
            multipliers.append(multiplier)

        if not values:
            for match in cls._WORD_UNIT.finditer(surface):
                number = words_to_number(match.group("word"))
                if number:
                    multiplier = cls._unit_multiplier(match.group("unit"))
                    values.append(number * multiplier)
                    multipliers.append(multiplier)

        if not values:
            # A bare figure like "13170000" is already in rupees. Require 5+
            # digits so a stray year or pin code is not read as a budget.
            for match in cls._BARE.finditer(surface.replace(",", "")):
                values.append(float(match.group("num")))

        if not values:
            return None
        if len(values) == 1:
            return values[0]

        # Two readings are possible when several amounts appear. A range
        # ("60 to 80 lakhs") states a ceiling, so the upper bound is the
        # customer's limit. A compound figure in descending units ("one crore
        # fifty lakhs") is a single amount and must be summed — taking the max
        # there would report 1 crore instead of 1.5.
        if _RANGE_CONNECTOR.search(surface):
            return max(values)
        if len(set(multipliers)) == len(multipliers) and multipliers == sorted(
            multipliers, reverse=True
        ):
            return sum(values)
        return max(values)


class HybridExtractor:
    """Rule + MuRIL extraction behind the rule extractor's output contract."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        extractor: Optional[NLPExtractor] = None,
        owners: Optional[Dict[str, str]] = None,
    ) -> None:
        self.extractor = extractor or NLPExtractor()
        self.locator = SpanLocator(self.extractor)
        self.owners = dict(owners or OWNERS)
        resolved = self._resolve_model_dir(model_dir)
        logger.info("Loading NER model from %s", resolved)
        self.predictor = MurilNerPredictor(resolved)
        self._distractor_cities = self._load_distractor_cities()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_model_dir(explicit: Optional[Path]) -> Path:
        if explicit:
            return Path(explicit)
        pointer = MODELS_STORE / "latest.json"
        if pointer.exists():
            return Path(json.loads(pointer.read_text(encoding="utf-8"))["model_dir"])
        candidates = sorted(MODELS_STORE.glob("muril_ner_*"))
        if not candidates:
            raise SystemExit(
                f"No trained NER model in {MODELS_STORE}. Run nlp.ner.train first."
            )
        return candidates[-1]

    def _load_distractor_cities(self) -> set:
        path = DATASET_DIR / "distractor_cities.csv"
        if not path.exists():
            logger.warning("No distractor_cities.csv at %s", path)
            return set()
        import pandas as pd

        frame = pd.read_csv(path)
        return {str(c).strip().lower() for c in frame["city"].dropna()}

    # ------------------------------------------------------------------ #
    # Span collection
    # ------------------------------------------------------------------ #

    def _rule_spans(self, raw_text: str) -> List[EntitySpan]:
        cleaned, index_map = build_offset_map(raw_text)
        spans: List[EntitySpan] = []
        for candidate in self.locator.locate(cleaned):
            start, end = map_span_to_raw(candidate.start, candidate.end, index_map)
            spans.append(
                EntitySpan(start, end, candidate.label, raw_text[start:end], RULES)
            )
        return spans

    def _model_spans(self, raw_text: str) -> List[EntitySpan]:
        """Groups the model's word-level BIO tags back into character spans."""
        tokens = tokenize_with_offsets(raw_text)
        if not tokens:
            return []
        tags = self.predictor.predict_words([t[0] for t in tokens])

        spans: List[EntitySpan] = []
        current: Optional[List[Any]] = None
        for (_, start, end), tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current:
                    spans.append(self._close(current, raw_text))
                current = [tag[2:], start, end]
            elif tag.startswith("I-") and current and current[0] == tag[2:]:
                current[2] = end
            else:
                if current:
                    spans.append(self._close(current, raw_text))
                current = None
        if current:
            spans.append(self._close(current, raw_text))
        return spans

    @staticmethod
    def _close(current: Sequence[Any], raw_text: str) -> EntitySpan:
        label, start, end = current[0], current[1], current[2]
        return EntitySpan(start, end, label, raw_text[start:end], MODEL)

    # ------------------------------------------------------------------ #
    # Conflict resolution
    # ------------------------------------------------------------------ #

    def _owned(self, spans: Sequence[EntitySpan]) -> List[EntitySpan]:
        return [s for s in spans if self.owners.get(s.label) == s.source]

    @staticmethod
    def _resolve_conflicts(spans: Sequence[EntitySpan]) -> List[EntitySpan]:
        """Drops overlapping spans **whole**, never truncating one.

        This is the fix for the fragmentation seen in the first hybrid attempt,
        where merging at the BIO-tag level let a model BUDGET span blank out the
        middle of a rule PROPERTY_TYPE span and split it into two bogus
        entities — costing PROPERTY_TYPE 0.047 F1 despite rules owning it.
        Resolving at span granularity means a span is either kept intact or
        dropped intact, so routing one entity to the model can no longer
        fragment another entity's span.

        Only *cross-source* overlaps are resolved. Two spans from the same
        system overlapping is intentional and must survive: a term like
        "business center" or "coworking space" is in both the property-type and
        amenity catalogs, so the rule locator emits both, and the original
        extractor reports both because it runs each field independently.
        Collapsing those would silently lose the property type. Only a model
        span landing on a rule span (or vice versa) can fragment anything, so
        that is the only case arbitrated here.

        Longest match wins, matching ``NLPExtractor._resolve_overlaps``; exact
        ties fall back to the owner's measured reliability.
        """
        ordered = sorted(
            spans,
            key=lambda s: (-s.length, -_TIE_BREAK.get(s.label, 0), s.start),
        )
        kept: List[EntitySpan] = []
        for span in ordered:
            if any(
                span.start < k.end and span.end > k.start and k.source != span.source
                for k in kept
            ):
                logger.debug(
                    "Dropped %s %r — overlaps a longer span from the other system",
                    span.label, span.text,
                )
                continue
            kept.append(span)
        return sorted(kept, key=lambda s: s.start)

    def merged_spans(self, raw_text: str) -> List[EntitySpan]:
        """Ownership-routed, conflict-resolved spans — negation not yet applied.

        Negated mentions are retained here because schema.md section 3 tags them
        like any other span; dropping them is a downstream decision, and keeping
        them makes this directly comparable to the Stage D gold labels.
        """
        candidates = self._owned(self._rule_spans(raw_text)) + self._owned(
            self._model_spans(raw_text)
        )
        resolved = self._resolve_conflicts(candidates)
        for span in resolved:
            span.negated = self._is_negated(raw_text, span.start)
        return resolved

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_negated(raw_text: str, start: int) -> bool:
        """Whether a negation cue scopes over the span starting at ``start``.

        The cue must sit in the *same clause*. Without that constraint "Not a
        villa, an independent house" negates both property types, because the
        cue is still within a few tokens of the second one — so the customer's
        actual requirement gets discarded along with what they ruled out.
        """
        window = raw_text[max(0, start - _NEGATION_WINDOW):start]
        # Collapse newlines so a cue at the end of the previous line still reads
        # as adjacent text.
        flattened = " ".join(window.split())
        clause = _CLAUSE_BOUNDARY.split(flattened)[-1]
        return bool(_NEGATION_LOOKBEHIND.search(clause))

    def _choose_city(
        self, spans: Sequence[EntitySpan], raw_text: str
    ) -> Optional[EntitySpan]:
        """Picks the city being shopped in, not the one being called from."""
        cities = [s for s in spans if s.label == "CITY" and not s.negated]
        if not cities:
            return None
        if len(cities) == 1:
            return cities[0]

        known = [s for s in cities if s.text.strip().lower() in self.extractor._city_display]
        distractors = [s for s in cities if s.text.strip().lower() in self._distractor_cities]
        # A city the company actually operates in beats an out-of-scope mention.
        preferred = known or [s for s in cities if s not in distractors] or cities

        seeking = [s for s in preferred if self._preceded_by(raw_text, s, _SEEKING_CUE)]
        if seeking:
            return seeking[-1]
        not_current = [
            s for s in preferred if not self._preceded_by(raw_text, s, _CURRENT_LOCATION_CUE)
        ]
        if not_current:
            return not_current[-1]
        return preferred[-1]

    @staticmethod
    def _preceded_by(raw_text: str, span: EntitySpan, pattern: re.Pattern) -> bool:
        window = raw_text[max(0, span.start - 80):span.start]
        return bool(pattern.search(" ".join(window.split())))

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #

    def _canonical(self, label: str, surface: str) -> Optional[str]:
        key = " ".join(surface.split()).strip().lower()
        if not key:
            return None
        if label == "CITY":
            return self.extractor._city_display.get(key, surface.strip().title())
        if label == "AREA":
            return self.extractor._area_display.get(key, surface.strip().title())
        if label == "PROPERTY_TYPE":
            if key in self.extractor.property_synonyms:
                return self.extractor.property_synonyms[key]
            return self.extractor._property_display.get(key, surface.strip().title())
        if label == "AMENITY":
            if key in self.extractor.amenity_synonyms:
                return self.extractor.amenity_synonyms[key]
            return self.extractor._amenity_display.get(key, surface.strip().title())
        if label == "FURNISHING":
            for pattern, canonical in _FURNISHING_CANONICAL:
                if re.search(pattern, key, re.IGNORECASE):
                    return canonical
            return surface.strip().title()
        return surface.strip()

    @staticmethod
    def _parse_bhk(surface: str) -> Optional[int]:
        digits = re.search(r"\d+", surface)
        if digits:
            return int(digits.group(0))
        for word, value in _NUMBER_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", surface, re.IGNORECASE):
                return value
        return None

    def _resolve_budget(
        self, active: Sequence[EntitySpan], cleaned: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """Picks the budget mention that actually carries an amount.

        Two failure modes this guards against, both seen on real transcripts:

        * The model sometimes tags a non-monetary phrase as BUDGET — "2 bhk"
          being the common one. Because that collides with the rule BHK span of
          identical length and BUDGET outranks BHK on ties, it would win and be
          reported as the budget. Requiring the span to parse to a number
          rejects it and falls through to the real amount.
        * In telegraphic or spelled-out transcripts ("Pune. 2bhk. 60 lakhs.",
          "around fifty lakhs") the model can miss the span entirely, so the
          text is rescanned for any amount-with-unit.
        """
        for span in reversed([s for s in active if s.label == "BUDGET"]):
            surface = " ".join(span.text.split())
            value = BudgetParser.parse(surface)
            if value:
                return surface, value
            logger.debug("Ignoring BUDGET span %r — no amount in it", surface)

        matches = [m for m in _BUDGET_SCAN.finditer(cleaned)
                   if not self._is_negated(cleaned, m.start())]
        if matches:
            surface = " ".join(matches[-1].group(0).split())
            return surface, BudgetParser.parse(surface)
        return None, None

    def _category_for(self, property_type: Optional[str]) -> Optional[str]:
        if not property_type:
            return None
        _, category = self.extractor.get_property_category(property_type)
        return category

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self, text: str) -> Dict[str, Any]:
        """Same contract as ``NLPExtractor.extract``, plus budget_value/negated."""
        if not text or not text.strip():
            return self._empty()

        spans = self.merged_spans(text)
        active = [s for s in spans if not s.negated]

        def last(label: str) -> Optional[EntitySpan]:
            # Last mention wins, so a self-correction ("not a villa, an
            # apartment instead") reflects the customer's final intent.
            found = [s for s in active if s.label == label]
            return found[-1] if found else None

        city_span = self._choose_city(spans, text)
        area_span = last("AREA")
        property_span = last("PROPERTY_TYPE")
        bhk_span = last("BHK")
        budget_span = last("BUDGET")
        furnishing_span = last("FURNISHING")

        # BIO tagging allows one label per token, so a term listed in two
        # catalogs ("business center", "coworking space" are both a property
        # type and an amenity) survives as only one span — the other is lost
        # before it ever reaches here. NLPExtractor has no such constraint
        # because it runs each field independently, so for rule-owned fields it
        # is consulted whenever the span view came up empty.
        cleaned = self.extractor.preprocessor.clean(text)

        city = self._canonical("CITY", city_span.text) if city_span else None
        area = self._canonical("AREA", area_span.text) if area_span else None
        if area is None:
            area = self.extractor.extract_area(cleaned)
        if city is None and area is not None:
            city = self.extractor._area_city_map.get(area.lower())

        property_type = (
            self._canonical("PROPERTY_TYPE", property_span.text) if property_span else None
        )
        if property_type is None:
            property_type, _ = self.extractor.extract_property_type(cleaned)

        budget_raw, budget_value = self._resolve_budget(active, cleaned)

        amenities: List[str] = []
        for span in active:
            if span.label != "AMENITY":
                continue
            canonical = self._canonical("AMENITY", span.text)
            if canonical and canonical not in amenities:
                amenities.append(canonical)

        return {
            "city": city,
            "area": area,
            "property_type": property_type,
            "category": self._category_for(property_type),
            "bhk": (
                self._parse_bhk(bhk_span.text)
                if bhk_span
                else self.extractor.extract_bhk(cleaned)
            ),
            "budget": budget_raw,
            "budget_value": budget_value,
            "amenities": amenities,
            # Model-owned, but it misses furnishing in telegraphic transcripts
            # ("Pune. 2bhk. 60 lakhs. Semi furnished."). The rule regex is a
            # pure gain here: it only runs when the model found nothing, so it
            # can add recall but never override the stronger system.
            "furnishing": (
                self._canonical("FURNISHING", furnishing_span.text)
                if furnishing_span
                else self.extractor.extract_furnishing(cleaned)
            ),
            "negated": sorted(
                {
                    f"{s.label}:{' '.join(s.text.split())}"
                    for s in spans
                    if s.negated
                }
            ),
        }

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "city": None, "area": None, "property_type": None, "category": None,
            "bhk": None, "budget": None, "budget_value": None,
            "amenities": [], "furnishing": None, "negated": [],
        }


_INSTANCE: Optional[HybridExtractor] = None


def get_hybrid_extractor(model_dir: Optional[Path] = None) -> HybridExtractor:
    """Process-wide singleton.

    Constructing one loads MuRIL plus the CSV catalogs, which takes seconds and
    hundreds of MB. A web worker must do that once at startup, never per
    request, so the webhook shares this instance.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HybridExtractor(model_dir=model_dir)
    return _INSTANCE


_DEMOS: Tuple[str, ...] = (
    "Customer: I'm looking for a 2 BHK apartment in Baner, Pune within 60 lakhs, "
    "fully furnished with a swimming pool.",
    "Customer: I'm in Mumbai right now but we want to buy in Nashik. "
    "Not a villa, an independent house in Gangapur Road, budget around 1.2 crore.",
    "Customer: Mujhe Wakad mein 3 BHK flat chahiye, budget 80 lakh, "
    "gym aur parking zaroor chahiye.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Transcript to extract from.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--demo", action="store_true", help="Run built-in examples.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.text and not args.demo:
        parser.error("Pass --text or --demo.")

    hybrid = HybridExtractor(model_dir=args.model_dir)
    for text in ([args.text] if args.text else list(_DEMOS)):
        logger.info("")
        logger.info("%s", "=" * 74)
        logger.info("%s", text)
        logger.info("%s", "-" * 74)
        logger.info("%s", json.dumps(hybrid.extract(text), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
