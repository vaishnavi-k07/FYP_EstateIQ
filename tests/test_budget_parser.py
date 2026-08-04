"""Budget parsing — the numeric value Phase 4 lead scoring consumes.

Getting this wrong is not a rounding error: reading "one hundred seventy lakhs"
as 70 lakh understates a lead's budget by 60%.
"""

from __future__ import annotations

import pytest

from nlp.predict import BudgetParser, words_to_number

LAKH = 100_000
CRORE = 10_000_000


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("one", 1),
        ("seventy", 70),
        ("ninety five", 95),
        ("one hundred", 100),
        ("one hundred seventy", 170),
        ("one hundred and fifty", 150),
        ("two hundred", 200),
        ("nothing numeric", None),
        ("", None),
    ],
)
def test_words_to_number(phrase, expected):
    assert words_to_number(phrase) == expected


@pytest.mark.parametrize(
    "surface, expected",
    [
        # digits + unit
        ("60 lakhs", 60 * LAKH),
        ("75 lakh", 75 * LAKH),
        ("1.2 crore", 1.2 * CRORE),
        ("2 cr", 2 * CRORE),
        ("80L", 80 * LAKH),
        # bare rupee figures
        ("13170000", 13_170_000),
        ("1,20,00,000", 12_000_000),
        # spelled out, including compounds
        ("sixty lakh", 60 * LAKH),
        ("ninety five lakhs", 95 * LAKH),
        ("one hundred seventy lakhs", 170 * LAKH),
        # ranges resolve to the ceiling
        ("60 to 80 lakhs", 80 * LAKH),
        ("60-80 lakhs", 80 * LAKH),
        ("between 50 and 70 lakhs", 70 * LAKH),
        ("under 75 lakhs", 75 * LAKH),
        # compound descending units are one amount, so they sum
        ("one crore fifty lakhs", 1.5 * CRORE),
        ("1 crore 50 lakhs", 1.5 * CRORE),
        # nothing to parse
        ("budget is not a constraint", None),
        ("", None),
        (None, None),
    ],
)
def test_budget_parser(surface, expected):
    assert BudgetParser.parse(surface) == expected


def test_range_takes_ceiling_not_sum():
    """A range must not be summed — "60 to 80 lakhs" is 80, never 140."""
    assert BudgetParser.parse("60 to 80 lakhs") == 80 * LAKH


def test_compound_sums_not_maxed():
    """A compound figure must not be maxed — 1cr 50L is 1.5cr, not 1cr."""
    assert BudgetParser.parse("one crore fifty lakhs") == 1.5 * CRORE


def test_short_number_is_not_a_budget():
    """A stray 4-digit number (a year, a door number) is not a rupee figure."""
    assert BudgetParser.parse("built in 2019") is None
