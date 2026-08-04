"""Phase 2 data handling, splitting, metrics and baselines.

Covers the pure logic only — no DistilBERT checkpoint is loaded, so these run
in milliseconds. Model-dependent behaviour belongs with the Phase 8 integration
tests, once artifacts are a build dependency.
"""

from __future__ import annotations

import collections
import json

import pytest

from sentiment_intent.data import (
    INTENT_CLASSES,
    SENTIMENT_CLASSES,
    build_split,
    classification_metrics,
    customer_turns,
    load_corpus,
    majority_baseline,
    resolve_latest,
    split_records,
    template_baseline,
)


# --------------------------------------------------------------------------- #
# Customer-turn extraction
# --------------------------------------------------------------------------- #


def test_customer_turns_drops_agent_lines():
    transcript = (
        "Agent: Good afternoon, how can I help?\n"
        "Customer: Looking for a 2 BHK in Pune.\n"
        "Agent: Any budget in mind?\n"
        "Customer: Around 60 lakhs."
    )
    assert customer_turns(transcript) == "Looking for a 2 BHK in Pune. Around 60 lakhs."


def test_customer_turns_falls_back_to_raw_text():
    """A bare utterance with no speaker prefixes is passed through unchanged."""
    assert customer_turns("just looking around for now") == "just looking around for now"


def test_customer_turns_on_agent_only_transcript_falls_back():
    # No customer speech at all — returning "" would silently score an empty
    # string, so the raw text is returned and the caller can decide.
    transcript = "Agent: Hello? Anyone there?"
    assert customer_turns(transcript) == transcript


def test_customer_turns_handles_empty():
    assert customer_turns("") == ""


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_corpus_labels_are_in_the_agreed_schemes(corpus):
    assert {r["sentiment"] for r in corpus} <= set(SENTIMENT_CLASSES)
    assert {r["intent"] for r in corpus} <= set(INTENT_CLASSES)


def test_corpus_has_no_commercial_intent(corpus):
    """Commercial is a property attribute, not an intent (CLAUDE.md Phase 2)."""
    assert "Commercial" not in {r["intent"] for r in corpus}


def test_corpus_customer_text_is_shorter_than_full_transcript(corpus):
    record = corpus[0]
    assert len(record["customer_text"]) < len(record["text"])


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


@pytest.fixture
def assignment(corpus, tmp_path):
    return build_split(corpus, path=tmp_path / "split.json")


def test_split_covers_every_transcript_exactly_once(corpus, assignment):
    assert set(assignment) == {r["transcript_id"] for r in corpus}
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_split_is_roughly_70_15_15(corpus, assignment):
    sizes = collections.Counter(assignment.values())
    assert sizes["train"] == pytest.approx(0.70 * len(corpus), abs=10)
    assert sizes["validation"] == pytest.approx(0.15 * len(corpus), abs=10)


def test_split_is_deterministic(corpus, tmp_path):
    first = build_split(corpus, path=tmp_path / "a.json")
    second = build_split(corpus, path=tmp_path / "b.json")
    assert first == second


def test_split_is_reused_when_seed_and_size_match(corpus, tmp_path):
    path = tmp_path / "split.json"
    build_split(corpus, path=path)
    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["assignment"]["tr_0001"] = "sentinel"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    # A saved split must be honoured, not silently regenerated — otherwise
    # transcripts drift between train and test across runs.
    assert build_split(corpus, path=path)["tr_0001"] == "sentinel"


def test_split_is_stratified_on_both_labels(corpus, assignment):
    """Every class of both schemes must appear in all three splits."""
    splits = split_records(corpus, assignment)
    for name, records in splits.items():
        assert {r["sentiment"] for r in records} == set(SENTIMENT_CLASSES), name
        assert {r["intent"] for r in records} == set(INTENT_CLASSES), name


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_classification_metrics_on_perfect_predictions():
    metrics = classification_metrics([0, 1, 2], [0, 1, 2], ["a", "b", "c"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_classification_metrics_confusion_matrix_orientation():
    """Rows are true labels, columns predicted."""
    metrics = classification_metrics([0, 0], [0, 1], ["a", "b"])
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [0, 0]]
    assert metrics["confusion_matrix"]["rows_are_true"] is True


def test_macro_f1_punishes_a_collapsed_class():
    """Accuracy stays high while macro-F1 exposes a never-predicted class."""
    references = [0] * 9 + [1]
    predictions = [0] * 10
    metrics = classification_metrics(references, predictions, ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["macro_f1"] < 0.5


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def test_majority_baseline_predicts_the_commonest_training_class(corpus, assignment):
    splits = split_records(corpus, assignment)
    metrics = majority_baseline(splits["train"], splits["test"], "intent", INTENT_CLASSES)
    assert metrics["majority_class"] in INTENT_CLASSES
    # Six near-balanced classes: guessing one of them lands near 1/6.
    assert 0.10 < metrics["accuracy"] < 0.30


def test_template_baseline_exposes_generator_leakage(corpus, assignment):
    """The synthetic intents are built from three fixed phrases per class, so a
    substring matcher nearly solves the task. This test pins that fact: if it
    ever fails, the generator has become less templated and the neural metrics
    have become more meaningful."""
    splits = split_records(corpus, assignment)
    metrics = template_baseline(splits["test"], "intent", INTENT_CLASSES)
    assert metrics["macro_f1"] > 0.80
    assert metrics["template_hit_rate"] > 0.80


def test_template_baseline_returns_full_metrics(corpus, assignment):
    splits = split_records(corpus, assignment)
    metrics = template_baseline(splits["test"], "sentiment", SENTIMENT_CLASSES)
    assert set(metrics["per_class"]) == set(SENTIMENT_CLASSES)
    assert 0.0 <= metrics["accuracy"] <= 1.0


# --------------------------------------------------------------------------- #
# Artifact resolution
# --------------------------------------------------------------------------- #


def test_resolve_latest_returns_none_for_empty_store(tmp_path):
    assert resolve_latest("sentiment", models_store=tmp_path) is None


def test_resolve_latest_prefers_the_pointer_file(tmp_path):
    model_dir = tmp_path / "distilbert_sentiment_v1_20260101_000000"
    model_dir.mkdir()
    (tmp_path / "sentiment_latest.json").write_text(
        json.dumps({"model_dir": str(model_dir)}), encoding="utf-8"
    )
    assert resolve_latest("sentiment", models_store=tmp_path) == model_dir


def test_resolve_latest_ignores_a_pointer_to_a_deleted_model(tmp_path):
    (tmp_path / "intent_latest.json").write_text(
        json.dumps({"model_dir": str(tmp_path / "gone")}), encoding="utf-8"
    )
    assert resolve_latest("intent", models_store=tmp_path) is None


# --------------------------------------------------------------------------- #
# Generator phrase pools — the frame/label confound guard
#
# The phrase-holdout diagnostic found the original pools let the sentence frame
# predict the label outright ("we're looking to ___" was only ever Rent), and
# Buy/Rent inverted 63/63 on unseen phrasings. These pin the property that fix
# depends on, so a future edit to the pools cannot quietly reintroduce it.
# --------------------------------------------------------------------------- #


def test_no_intent_frame_is_exclusive_to_one_class():
    from data_generation.phrase_pools import frame_label_confounds

    assert frame_label_confounds() == {}


def test_every_intent_frame_is_shared_by_all_classes():
    import collections

    from data_generation.phrase_pools import (
        INTENT_CLASSES,
        INTENT_PHRASES,
        intent_frame_of,
    )

    owners = collections.defaultdict(set)
    for label in INTENT_CLASSES:
        for phrase in INTENT_PHRASES[label]:
            owners[intent_frame_of(phrase)].add(label)
    assert owners, "no frames found"
    assert all(len(labels) == len(INTENT_CLASSES) for labels in owners.values())


def test_negation_is_covered_in_both_directions():
    """"for buying, not renting" and "for renting, not buying" must both exist."""
    from data_generation.phrase_pools import INTENT_PHRASES

    assert any("not renting" in p for p in INTENT_PHRASES["Buy"])
    assert any("not buying" in p for p in INTENT_PHRASES["Rent"])


def test_pools_are_wide_enough():
    from data_generation.phrase_pools import INTENT_CLASSES, INTENT_PHRASES

    for label in INTENT_CLASSES:
        assert len(INTENT_PHRASES[label]) >= 20, label


def test_greetings_are_shared_across_all_sentiments():
    """A greeting must never signal a tone on its own — "Um, hi." previously
    appeared only with Hesitant."""
    from data_generation.phrase_pools import GREETINGS, SENTIMENT_CLASSES, SENTIMENT_OPENERS

    for greeting in GREETINGS:
        carriers = {
            label for label in SENTIMENT_CLASSES
            if any(p.startswith(greeting) for p in SENTIMENT_OPENERS[label])
        }
        assert carriers == set(SENTIMENT_CLASSES), (greeting, carriers)
