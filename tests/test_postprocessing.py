"""Negation scoping and cross-source conflict resolution.

Both are static/pure logic on HybridExtractor, so these run without loading the
neural model.
"""

from __future__ import annotations

import pytest

from nlp.predict import EntitySpan, HybridExtractor

MODEL = "model"
RULES = "rules"


def negated_at(text: str, needle: str) -> bool:
    return HybridExtractor._is_negated(text, text.index(needle))


# --------------------------------------------------------------------------- #
# Negation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, needle",
    [
        ("I don't want a villa", "villa"),
        ("not a villa", "villa"),
        ("we do not need a swimming pool", "swimming pool"),
        ("no parking required", "parking"),
        ("I don't want an unfurnished place", "unfurnished"),
    ],
)
def test_negation_detected(text, needle):
    assert negated_at(text, needle) is True


@pytest.mark.parametrize(
    "text, needle",
    [
        ("I want a villa", "villa"),
        ("looking for a 2 BHK apartment in Baner", "apartment"),
        ("we need a swimming pool", "swimming pool"),
    ],
)
def test_plain_mentions_not_negated(text, needle):
    assert negated_at(text, needle) is False


@pytest.mark.parametrize(
    "text, negated_term, wanted_term",
    [
        ("Not a villa, an independent house", "villa", "independent house"),
        ("I don't want a flat, a row house instead", "flat", "row house"),
        ("no gym, but a swimming pool is a must", "gym", "swimming pool"),
    ],
)
def test_negation_stops_at_clause_boundary(text, negated_term, wanted_term):
    """A cue must not reach past its clause into what the customer *does* want.

    Without this, "Not a villa, an independent house" negates both and the
    actual requirement is thrown away with the rejection.
    """
    assert negated_at(text, negated_term) is True
    assert negated_at(text, wanted_term) is False


def test_negation_does_not_cross_sentences():
    text = "We decided against a villa. An apartment in Baner works."
    assert negated_at(text, "apartment") is False


def test_negation_window_is_bounded():
    """A cue far away must not scope over a later mention."""
    text = "I don't like the paperwork involved in all of this honestly, " \
           "anyway we want a villa"
    assert negated_at(text, "villa") is False


# --------------------------------------------------------------------------- #
# Conflict resolution
# --------------------------------------------------------------------------- #


def span(start, end, label, source, text="x"):
    return EntitySpan(start, end, label, text, source)


def test_same_source_overlap_is_preserved():
    """A term in two catalogs ("business center") is both — keep both spans.

    Dropping one loses the property type entirely, which is what made
    tr_0353 regress during development.
    """
    spans = [
        span(0, 15, "PROPERTY_TYPE", RULES, "business center"),
        span(0, 15, "AMENITY", RULES, "business center"),
    ]
    kept = HybridExtractor._resolve_conflicts(spans)
    assert {s.label for s in kept} == {"PROPERTY_TYPE", "AMENITY"}


def test_cross_source_overlap_is_resolved_to_longest():
    """A model span overlapping a rule span must not fragment it."""
    spans = [
        span(0, 20, "PROPERTY_TYPE", RULES, "independent house"),
        span(5, 12, "BUDGET", MODEL, "house b"),
    ]
    kept = HybridExtractor._resolve_conflicts(spans)
    assert len(kept) == 1
    assert kept[0].label == "PROPERTY_TYPE"
    # The surviving span is intact, not truncated.
    assert (kept[0].start, kept[0].end) == (0, 20)


def test_spans_are_atomic_never_truncated():
    """Whatever survives keeps its original boundaries."""
    original = span(10, 40, "AMENITY", RULES, "central air conditioning")
    kept = HybridExtractor._resolve_conflicts([original, span(20, 25, "BUDGET", MODEL)])
    assert [(s.start, s.end) for s in kept] == [(10, 40)]


def test_non_overlapping_spans_all_survive():
    spans = [
        span(0, 5, "CITY", MODEL),
        span(10, 15, "AREA", RULES),
        span(20, 30, "BUDGET", MODEL),
    ]
    kept = HybridExtractor._resolve_conflicts(spans)
    assert len(kept) == 3
    assert [s.start for s in kept] == [0, 10, 20]


def test_output_is_sorted_by_position():
    spans = [span(20, 30, "BUDGET", MODEL), span(0, 5, "CITY", MODEL)]
    kept = HybridExtractor._resolve_conflicts(spans)
    assert [s.start for s in kept] == [0, 20]
