"""Adversarial transcripts — the collision traps and awkward phrasings.

Converted from the ad-hoc ``backend/test.py`` script, which printed results for
eyeballing but asserted nothing. Each case here is either a pinned expectation
or an ``xfail`` documenting a real limitation, so nothing silently regresses and
the known gaps stay visible.

Marked ``slow`` — loads the real model. Run the fast suite with:
    pytest -m "not slow"
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

pytestmark = pytest.mark.slow

LAKH = 100_000


def assert_fields(result: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """Checks only the named fields, so unrelated drift doesn't fail the test."""
    for field, value in expected.items():
        assert result[field] == value, (
            f"{field}: expected {value!r}, got {result[field]!r}"
        )


# --------------------------------------------------------------------------- #
# Cases that must extract correctly
# --------------------------------------------------------------------------- #

STRESS_CASES = [
    pytest.param(
        "haan so basically mujhe ek 2bhk chahiye Pune mein, budget around 55 lakh hai",
        {"city": "Pune", "bhk": 2, "budget_value": 55 * LAKH},
        id="hinglish-code-mixed",
    ),
    pytest.param(
        "budget is exactly 8500000 rupees, looking for a 3 bhk in Nashik",
        {"city": "Nashik", "bhk": 3, "budget_value": 8_500_000},
        id="exact-rupee-figure",
    ),
    pytest.param(
        "I want something under 50 lakhs, 2 bhk apartment, doesn't matter which area",
        {"property_type": "Apartment", "bhk": 2, "budget_value": 50 * LAKH},
        id="under-x-framing",
    ),
    pytest.param(
        "I was considering either a 2bhk or a 3bhk, but I think we'll go with "
        "3bhk finally, in Pune",
        {"city": "Pune", "bhk": 3},
        id="self-correction-last-wins",
    ),
    pytest.param(
        "looking for something in Wagholi, budget 40 lakhs, 2 bhk",
        {"city": "Pune", "area": "Wagholi", "bhk": 2, "budget_value": 40 * LAKH},
        id="area-implies-city",
    ),
    pytest.param(
        "want it semifurnished, 2bhk, somewhere in Pune",
        {"city": "Pune", "bhk": 2, "furnishing": "Semi Furnished"},
        id="furnishing-casual-spelling",
    ),
    pytest.param(
        "no specific amenities needed, just a basic 1 bhk flat, budget 25 lakh, Nashik",
        {"city": "Nashik", "property_type": "Apartment", "bhk": 1,
         "budget_value": 25 * LAKH, "amenities": []},
        id="no-amenities-needed",
    ),
    pytest.param(
        "I'm currently based in Mumbai but I want to buy a property in Pune, "
        "3bhk, budget 90 lakhs",
        {"city": "Pune", "bhk": 3, "budget_value": 90 * LAKH},
        id="multi-city-current-vs-target",
    ),
    pytest.param(
        "we're a startup looking for an office space, around Nashik, nothing fancy",
        {"city": "Nashik", "property_type": "Office Space", "category": "Commercial"},
        id="commercial-category",
    ),
    pytest.param(
        "Pune. 2bhk. 60 lakhs. Semi furnished. Parking and gym.",
        {"city": "Pune", "bhk": 2, "budget_value": 60 * LAKH,
         "furnishing": "Semi Furnished"},
        id="telegraphic-answers",
    ),
    pytest.param(
        "budget is around fifty lakhs, looking for a 2 bhk apartment",
        {"property_type": "Apartment", "bhk": 2, "budget_value": 50 * LAKH},
        id="budget-spelled-out",
    ),
    pytest.param(
        "not looking for a flat or a villa, just a plain residential plot, in Nashik",
        {"city": "Nashik", "property_type": "Residential Plot"},
        id="negation-then-requirement",
    ),
    pytest.param(
        "want a gated society, garden, and power backup, looking for an apartment "
        "not a bungalow",
        {"property_type": "Apartment", "category": "Residential"},
        id="amenity-list-with-negated-property",
    ),
]


@pytest.mark.parametrize("text, expected", STRESS_CASES)
def test_stress_case(hybrid, text, expected):
    assert_fields(hybrid.extract(text), expected)


# --------------------------------------------------------------------------- #
# Negative cases — must NOT invent values
# --------------------------------------------------------------------------- #


def test_budget_not_a_constraint_extracts_no_number(hybrid):
    """"Budget is not a constraint" states the absence of a figure."""
    result = hybrid.extract(
        "budget is not really a constraint, we just want the right 4 bhk villa in Pune"
    )
    assert result["budget"] is None
    assert result["budget_value"] is None
    assert result["bhk"] == 4
    assert result["property_type"] == "Villa"


def test_tire_kicker_yields_nothing(hybrid):
    """A caller who commits to nothing must not produce invented requirements."""
    result = hybrid.extract(
        "umm I don't know, just checking what's available, maybe something small, "
        "not sure about budget honestly"
    )
    assert result["bhk"] is None
    assert result["budget_value"] is None
    assert result["city"] is None
    assert result["amenities"] == []


def test_bhk_count_never_read_as_budget(hybrid):
    """A BHK count must never be reported as the budget.

    The model does sometimes tag "2 bhk" as BUDGET; because that collides with
    the rule BHK span of equal length and BUDGET outranks BHK on ties, it would
    surface as the budget. Guarded by requiring a parseable amount.
    """
    for text in (
        "I want something under 50 lakhs, 2 bhk apartment",
        "looking for something in Wagholi, budget 40 lakhs, 2 bhk",
    ):
        result = hybrid.extract(text)
        assert "bhk" not in (result["budget"] or "").lower()
        assert result["budget_value"] >= 40 * LAKH


def test_long_rambling_call(hybrid):
    """A realistic, filler-heavy transcript still yields the full requirement."""
    result = hybrid.extract(
        "hello yes hi, so my name is Priya, um actually my husband and I have been "
        "looking for a while now, we currently live on rent in Pune, Kothrud side, "
        "and we want to finally buy something, probably 2 bhk, maybe 3 if budget allows, "
        "budget wise we are thinking 65 to 75 lakhs, we'd like a lift and parking at "
        "least, gym would be nice but not necessary, and ideally semi furnished so we "
        "don't have to spend more on interiors"
    )
    assert result["city"] == "Pune"
    assert result["area"] == "Kothrud"
    # A stated range resolves to its ceiling.
    assert result["budget_value"] == 75 * LAKH
    assert result["furnishing"] == "Semi Furnished"
    assert {"Lift", "Parking"}.issubset(set(result["amenities"]))


def test_amenity_negation_is_recorded(hybrid):
    result = hybrid.extract(
        "I don't need a swimming pool or clubhouse, just basic parking and "
        "security is enough"
    )
    assert any("swimming pool" in n.lower() for n in result["negated"])
    assert "Parking" in result["amenities"]


# --------------------------------------------------------------------------- #
# Known limitations — xfail so they stay visible instead of silently passing
# --------------------------------------------------------------------------- #


def test_coordinated_negation_covers_both_items(hybrid):
    """A single cue scopes over every item joined by a coordinating conjunction."""
    result = hybrid.extract(
        "I don't need a swimming pool or clubhouse, just basic parking and "
        "security is enough"
    )
    assert "Club House" not in result["amenities"]
    assert "Swimming Pool" not in result["amenities"]
    # What the customer *does* want survives the wider scope.
    assert "Parking" in result["amenities"]


def test_coordinated_property_negation(hybrid):
    result = hybrid.extract(
        "not looking for a flat or a villa, just a plain residential plot, in Nashik"
    )
    negated = " ".join(result["negated"]).lower()
    assert "villa" in negated
    assert "flat" in negated
    assert result["property_type"] == "Residential Plot"


@pytest.mark.parametrize(
    "text, kept",
    [
        # A comma after a negation introduces a correction, not another negated
        # item — the scope must stop there.
        ("Not a villa, an independent house in Gangapur Road, Nashik",
         "Independent House"),
        ("Not a penthouse, more like a resort in Nashik", None),
        ("I don't want a flat, a row house instead in Nashik", "Row House"),
    ],
)
def test_comma_correction_is_not_swallowed(hybrid, text, kept):
    """Widening across conjunctions must not widen across commas."""
    result = hybrid.extract(text)
    assert result["property_type"] is not None
    if kept:
        assert result["property_type"] == kept


@pytest.mark.xfail(
    reason="'1RK' is not in the BHK vocabulary — neither the rule regex nor the "
           "model's training data covers the RK (room-kitchen) format.",
    strict=True,
)
def test_1rk_is_recognised(hybrid):
    result = hybrid.extract(
        "just need a 1RK for now, budget around 15 lakh, in Nashik"
    )
    assert result["bhk"] == 1


@pytest.mark.xfail(
    reason="Collision trap: the invented word 'punery' is tagged CITY by the "
           "model, which generalises city-shaped tokens. The real mention of "
           "Pune in the same sentence is correctly negated, so the false "
           "positive wins.",
    strict=True,
)
def test_invented_word_is_not_a_city(hybrid):
    result = hybrid.extract(
        "I run a punery... I mean a bakery business, need commercial space, "
        "not related to Pune though"
    )
    assert result["city"] != "Punery"
