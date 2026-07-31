import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nlp.preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Tolerates disfluencies ("umm", "uh", "like", ...) and stray punctuation/whitespace
# sitting between a requirement word and its value (e.g. "3, umm, bhk").
_FILLER = r"[\s,]*(?:(?:um+|uh+h?|ah+h?|like|you know|i mean|actually)[\s,]*)*"

_BHK_WORD_PATTERN = "|".join(_NUMBER_WORDS)
_BHK_PATTERN = re.compile(
    rf"\b(?:(?P<num>\d+)|(?P<word>{_BHK_WORD_PATTERN})\b){_FILLER}(?:bhk|bed\s*rooms?)\b",
    re.IGNORECASE,
)
_BUDGET_PATTERN = re.compile(
    rf"(\d+(?:\.\d+)?){_FILLER}(lakhs?|lacs?|crores?|cr\.?s?|l)\b",
    re.IGNORECASE,
)

# Negation cues that, when found shortly before a requirement mention, mean the
# speaker is ruling that value out rather than asking for it
# (e.g. "I don't want a villa", "not 3 bhk").
_NEGATION_CUES = (
    r"(?:don't|do not|dont|doesn't|does not|didn't|did not|"
    r"won't|will not|isn't|is not|not|no|never)"
)
_NEGATION_LOOKBEHIND = re.compile(
    rf"\b{_NEGATION_CUES}\b(?:\s+\S+){{0,3}}\s*$", re.IGNORECASE
)


class NLPExtractor:
    def __init__(self) -> None:
        self.locations_df = self._load_csv("locations.csv")
        self.property_df = self._load_csv("property_type.csv")
        self.amenities_df = self._load_csv("amenities.csv")
        self.property_synonyms_df = self._load_csv("property_type_synonyms.csv")
        self.amenity_synonyms_df = self._load_csv("amenity_synonyms.csv")

        self.cities: List[str] = self._unique_lower(self.locations_df, "city")
        self.areas: List[str] = self._unique_lower(self.locations_df, "area")
        self.property_types: List[str] = self._unique_lower(self.property_df, "property_type_name")
        self.amenities: List[str] = self._unique_lower(self.amenities_df, "amenity_name")

        self.property_synonyms: Dict[str, str] = self._load_synonym_map(
            self.property_synonyms_df, "synonym", "property_type_name"
        )
        self.amenity_synonyms: Dict[str, str] = self._load_synonym_map(
            self.amenity_synonyms_df, "synonym", "amenity_name"
        )

        # lower -> original casing, so display output doesn't rely on str.title()
        # (which mangles names like "NIBM Road" -> "Nibm Road" or "24x7 Security" -> "24X7 Security")
        self._city_display = self._display_map(self.locations_df, "city")
        self._area_display = self._display_map(self.locations_df, "area")
        self._property_display = self._display_map(self.property_df, "property_type_name")
        self._amenity_display = self._display_map(self.amenities_df, "amenity_name")

        self._area_city_map = self._build_area_city_map()

        self.preprocessor = TextPreprocessor()

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = DATASET_DIR / filename
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            logger.error("Required lookup file not found: %s", path)
            raise

    @staticmethod
    def _unique_lower(df: pd.DataFrame, column: str) -> List[str]:
        return df[column].dropna().astype(str).str.lower().unique().tolist()

    @staticmethod
    def _display_map(df: pd.DataFrame, column: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for value in df[column].dropna().astype(str):
            result[value.strip().lower()] = value.strip()
        return result

    @staticmethod
    def _load_synonym_map(df: pd.DataFrame, key_col: str, value_col: str) -> Dict[str, str]:
        return {
            str(row[key_col]).strip().lower(): str(row[value_col]).strip()
            for _, row in df.iterrows()
        }

    def _build_area_city_map(self) -> Dict[str, str]:
        """Maps area -> city, but only for areas that belong to exactly one city."""
        grouped: Dict[str, set] = {}
        subset = self.locations_df.dropna(subset=["area", "city"])
        for _, row in subset.iterrows():
            area_key = str(row["area"]).strip().lower()
            grouped.setdefault(area_key, set()).add(str(row["city"]).strip())
        return {area: next(iter(cities)) for area, cities in grouped.items() if len(cities) == 1}

    @staticmethod
    def _is_negated(text: str, match_start: int) -> bool:
        window = text[max(0, match_start - 30):match_start]
        return bool(_NEGATION_LOOKBEHIND.search(window))

    def _longest_match(
        self, text: str, candidates: List[str], display_map: Dict[str, str]
    ) -> Optional[str]:
        best: Optional[str] = None
        best_len = -1
        for candidate in candidates:
            if len(candidate) <= best_len:
                continue
            if re.search(rf"\b{re.escape(candidate)}\b", text, re.IGNORECASE):
                best = candidate
                best_len = len(candidate)
        return display_map.get(best) if best else None

    @staticmethod
    def _resolve_overlaps(
        candidates: List[Tuple[int, int, str, bool]]
    ) -> List[Tuple[int, str]]:
        """Keeps the most specific (longest) match at each text span, discarding
        shorter matches fully contained in a longer one (e.g. "house" inside
        "farm house", or "spa" inside "space" would already be excluded by the
        caller's word-boundary matching). On an exact-span tie, a synonym match
        wins over a plain catalog match, since synonyms encode an explicit
        normalization rule (e.g. "flat" -> "Apartment").
        """
        ordered = sorted(
            candidates, key=lambda c: (-(c[1] - c[0]), 0 if c[3] else 1, c[0])
        )
        occupied: List[Tuple[int, int]] = []
        resolved: List[Tuple[int, str]] = []
        for start, end, name, _is_synonym in ordered:
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue
            occupied.append((start, end))
            resolved.append((start, name))
        return resolved

    def extract_area(self, text: str) -> Optional[str]:
        area = self._longest_match(text, self.areas, self._area_display)
        return area

    def extract_city(self, text: str) -> Optional[str]:
        return self._longest_match(text, self.cities, self._city_display)

    def extract_property_type(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        candidates: List[Tuple[int, int, str, bool]] = []

        for prop in self.property_types:
            for m in re.finditer(rf"\b{re.escape(prop)}\b", text, re.IGNORECASE):
                candidates.append((m.start(), m.end(), self._property_display[prop], False))

        for synonym, canonical_name in self.property_synonyms.items():
            for m in re.finditer(rf"\b{re.escape(synonym)}\b", text, re.IGNORECASE):
                candidates.append((m.start(), m.end(), canonical_name, True))

        resolved = self._resolve_overlaps(candidates)
        non_negated = [
            (start, name) for start, name in resolved if not self._is_negated(text, start)
        ]
        if not non_negated:
            return None, None

        # Last non-negated mention wins, so a self-correction ("not a villa,
        # an apartment instead") reflects the customer's actual intent.
        non_negated.sort(key=lambda x: x[0])
        _, chosen = non_negated[-1]
        return self.get_property_category(chosen)

    def get_property_category(self, property_name: str) -> Tuple[Optional[str], Optional[str]]:
        row = self.property_df[
            self.property_df["property_type_name"].str.lower() == property_name.lower()
        ]

        if not row.empty:
            return property_name, row.iloc[0]["category"]

        return property_name, None

    def extract_amenities(self, text: str) -> List[str]:
        candidates: List[Tuple[int, int, str, bool]] = []

        for amenity in self.amenities:
            for m in re.finditer(rf"\b{re.escape(amenity)}\b", text, re.IGNORECASE):
                candidates.append((m.start(), m.end(), self._amenity_display[amenity], False))

        for synonym, canonical_name in self.amenity_synonyms.items():
            for m in re.finditer(rf"\b{re.escape(synonym)}\b", text, re.IGNORECASE):
                candidates.append((m.start(), m.end(), canonical_name, True))

        resolved = self._resolve_overlaps(candidates)
        resolved.sort(key=lambda x: x[0])
        return list(dict.fromkeys(name for _, name in resolved))

    def extract_bhk(self, text: str) -> Optional[int]:
        candidates: List[int] = []
        for match in _BHK_PATTERN.finditer(text):
            if self._is_negated(text, match.start()):
                continue
            if match.group("num"):
                candidates.append(int(match.group("num")))
            else:
                candidates.append(_NUMBER_WORDS[match.group("word").lower()])

        # Last non-negated mention wins (handles self-corrections like
        # "I meant 2 bhk not 3 bhk").
        return candidates[-1] if candidates else None

    def extract_budget(self, text: str) -> Optional[str]:
        matches = list(_BUDGET_PATTERN.finditer(text))
        if not matches:
            return None

        match = matches[-1]
        amount = match.group(1)
        unit = match.group(2).lower()
        normalized_unit = "crore" if unit.startswith("cr") else "lakh"
        return f"{amount} {normalized_unit}"

    def extract_furnishing(self, text: str) -> Optional[str]:
        if re.search(r"\bsemi[\s-]?furnished\b", text, re.IGNORECASE):
            return "Semi Furnished"
        if re.search(r"\bfully[\s-]?furnished\b", text, re.IGNORECASE):
            return "Fully Furnished"
        if re.search(r"\bunfurnished\b", text, re.IGNORECASE):
            return "Unfurnished"
        if re.search(r"\bfurnished\b", text, re.IGNORECASE):
            return "Furnished"

        return None

    def extract(self, text: str) -> Dict[str, Any]:
        cleaned_text = self.preprocessor.clean(text)

        property_type, category = self.extract_property_type(cleaned_text)

        city = self.extract_city(cleaned_text)
        area = self.extract_area(cleaned_text)
        if city is None and area is not None:
            city = self._area_city_map.get(area.lower())

        return {
            "city": city,
            "area": area,
            "property_type": property_type,
            "category": category,
            "bhk": self.extract_bhk(cleaned_text),
            "budget": self.extract_budget(cleaned_text),
            "amenities": self.extract_amenities(cleaned_text),
            "furnishing": self.extract_furnishing(cleaned_text),
        }
