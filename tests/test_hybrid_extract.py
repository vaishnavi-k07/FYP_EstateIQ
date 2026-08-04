"""End-to-end extraction against the real hybrid model.

Marked ``slow`` — loads MuRIL. Run the fast suite with:
    pytest -m "not slow"
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

LAKH = 100_000
CRORE = 10_000_000

RULE_CONTRACT_KEYS = {
    "city", "area", "property_type", "category", "bhk", "budget",
    "amenities", "furnishing",
}


def test_output_is_a_superset_of_the_rule_contract(hybrid, rule_extractor):
    """Drop-in means every key the pipeline already reads is still present."""
    text = "I want a 3 BHK flat in Wakad Pune for 75 lakhs, semi furnished with gym."
    result = hybrid.extract(text)
    assert RULE_CONTRACT_KEYS.issubset(result)
    assert set(rule_extractor.extract(text)).issubset(result)
    # Additions for Phase 4 scoring and Stage E explainability.
    assert "budget_value" in result
    assert "negated" in result


def test_straightforward_transcript(hybrid):
    result = hybrid.extract(
        "Customer: I'm looking for a 2 BHK apartment in Baner, Pune "
        "within 60 lakhs, fully furnished with a swimming pool."
    )
    assert result["city"] == "Pune"
    assert result["area"] == "Baner"
    assert result["property_type"] == "Apartment"
    assert result["category"] == "Residential"
    assert result["bhk"] == 2
    assert result["budget_value"] == 60 * LAKH
    assert result["furnishing"] == "Fully Furnished"
    assert "Swimming Pool" in result["amenities"]


def test_negation_keeps_the_requirement_and_drops_the_rejection(hybrid):
    result = hybrid.extract(
        "Not a villa, an independent house in Gangapur Road, Nashik, "
        "budget around 1.2 crore."
    )
    assert result["property_type"] == "Independent House"
    assert result["budget_value"] == 1.2 * CRORE
    assert any("villa" in n.lower() for n in result["negated"])


def test_multi_city_prefers_the_target_over_current_location(hybrid):
    result = hybrid.extract(
        "I'm in Mumbai right now but we want to buy in Nashik, 3 BHK, 90 lakhs."
    )
    assert result["city"] == "Nashik"


def test_code_mixed_transcript(hybrid):
    result = hybrid.extract(
        "Mujhe Wakad mein 3 BHK flat chahiye, budget 80 lakh, "
        "gym aur parking zaroor chahiye."
    )
    assert result["area"] == "Wakad"
    assert result["bhk"] == 3
    assert result["budget_value"] == 80 * LAKH
    assert result["amenities"]


def test_area_implies_city(hybrid):
    """An area that belongs to exactly one city fills the city in."""
    result = hybrid.extract("Looking for a flat in Gangapur Road, 50 lakhs.")
    assert result["city"] == "Nashik"


def test_dual_catalog_term_still_yields_a_property_type(hybrid):
    """"Business center" is in both the property-type and amenity catalogs.

    BIO tagging can only carry one label per token, so the span view loses one
    of them; the rule fallback must still produce the property type.
    """
    result = hybrid.extract(
        "We need a business center in Nashik, budget one hundred seventy lakhs."
    )
    assert result["property_type"] is not None
    assert result["budget_value"] == 170 * LAKH


def test_vague_transcript_yields_nulls_not_guesses(hybrid):
    result = hybrid.extract(
        "Customer: We haven't really decided anything yet, just looking around."
    )
    assert result["bhk"] is None
    assert result["budget_value"] is None
    assert result["amenities"] == []


def test_empty_input(hybrid):
    for value in ("", "   ", None):
        result = hybrid.extract(value)
        assert result["city"] is None
        assert result["amenities"] == []
        assert result["budget_value"] is None


def test_extract_is_deterministic(hybrid):
    text = "2 BHK in Baner Pune, 60 lakhs, semi furnished."
    assert hybrid.extract(text) == hybrid.extract(text)
