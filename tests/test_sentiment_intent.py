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


def test_corpus_excludes_telegraphic_transcripts(corpus):
    """A telegraphic caller ("Pune. 2 BHK. 60 lakhs.") states neither a tone nor
    an intent, so its labels are noise for both heads. They stay in the file for
    Phase 3's NER data and are filtered on load here."""
    assert not any(r["is_telegraphic"] for r in corpus)


def test_telegraphic_transcripts_are_filtered_not_deleted():
    """The exclusion must be a load-time filter — the transcripts themselves are
    Phase 3 training data and must remain in the .jsonl untouched."""
    with_telegraphic = load_corpus(exclude_telegraphic=False)
    telegraphic = [r for r in with_telegraphic if r["is_telegraphic"]]
    assert telegraphic, "no telegraphic transcripts left in the corpus file"
    assert len(load_corpus()) == len(with_telegraphic) - len(telegraphic)


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


# --------------------------------------------------------------------------- #
# Generator phrase pools — the sentiment token confound guard
#
# Sharing the greeting fixed sentiment only down to the clause. One level
# below, "bahut" sat in a single Enthusiastic clause and a single Frustrated
# one; the holdout withheld the Enthusiastic clause, so at training time the
# word was pure Frustrated and the two tones inverted 17 each way. These pin
# the token-level property that fix depends on.
# --------------------------------------------------------------------------- #


def test_no_tone_marker_is_one_clause_from_being_stranded():
    from data_generation.phrase_pools import shared_token_confounds

    assert shared_token_confounds() == {}


def test_bahut_is_carried_by_at_least_two_clauses_in_every_tone_that_uses_it():
    """The specific token that caused the inversion, pinned by name."""
    from data_generation.phrase_pools import marker_token_counts

    for token in ("bahut", "abhi", "matlab", "excited", "keen", "time"):
        counts = marker_token_counts()[token]
        assert len(counts) >= 2, f"{token} is exclusive to {list(counts)}"
        assert min(counts.values()) >= 2, f"{token} {counts}"


def test_every_tone_has_several_code_mixed_clauses():
    """One code-mixed clause per tone meant a holdout could remove all of it."""
    from data_generation.phrase_pools import (
        SENTIMENT_CLASSES,
        TONE_CLAUSES,
        TONE_MARKER_TOKENS,
    )

    hindi = {"bahut", "abhi", "matlab", "hai", "hain", "hoon", "nahi", "pakka",
             "thoda", "seedha", "dekhiye", "khush"}
    assert hindi <= TONE_MARKER_TOKENS
    for label in SENTIMENT_CLASSES:
        code_mixed = [c for c in TONE_CLAUSES[label]
                      if hindi & set(c.lower().replace(",", " ").split())]
        assert len(code_mixed) >= 3, f"{label}: {code_mixed}"


def test_no_tone_loses_its_whole_code_mixed_register_to_the_holdout():
    """The end-to-end version of the guard, run against the real sampler.

    Pool design alone is not enough: the first v3 run withheld all three
    Enthusiastic code-mixed clauses and half its held-out transcripts went to
    Frustrated. Every tone must keep some code-mixed phrasing in training, or
    romanised Hindi becomes a cue for whichever tones kept theirs.
    """
    from sentiment_intent.phrase_holdout import held_out_phrases

    from data_generation.phrase_pools import (
        SENTIMENT_CLASSES,
        TONE_CLAUSES,
        phrase_register,
    )

    held = held_out_phrases("sentiment")
    for label in SENTIMENT_CLASSES:
        code_mixed = [c for c in TONE_CLAUSES[label]
                      if phrase_register(c) == "code-mixed"]
        survivors = [c for c in code_mixed
                     if not any(c in phrase for phrase in held[label])]
        assert survivors, f"{label} keeps no code-mixed clause in training"


def test_tone_clause_pools_are_wide_enough():
    """A third of a pool of eight left too little to train on."""
    from data_generation.phrase_pools import (
        DEFLECTION_CORES,
        SENTIMENT_CLASSES,
        TONE_CLAUSES,
    )

    for label in SENTIMENT_CLASSES:
        assert len(TONE_CLAUSES[label]) >= 20, label
        assert len(DEFLECTION_CORES[label]) >= 12, label


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


# --------------------------------------------------------------------------- #
# Pilot corpus (Phase 2b) — the properties its conclusions rest on
#
# The pilot exists to test whether situational emotion closes the sentiment
# generalisation gap. Every one of its claims depends on a structural property
# of the corpus rather than on the model, so each is pinned here: if one breaks,
# the pilot's numbers stop meaning what the report says they mean.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pilot_corpus():
    """The widened corpus. Named for the pilot because it is the same design.

    The 288-transcript pilot is still on disk for comparison, but the
    properties below are asserted against the corpus that is actually trained
    on, since that is the one whose numbers get reported.
    """
    from sentiment_intent.data import SENTIMENT_V6_PATH

    if not SENTIMENT_V6_PATH.exists():
        pytest.skip("v6 corpus not generated; run data_generation.sentiment_pilot")
    return load_corpus(SENTIMENT_V6_PATH)


def test_pilot_pools_contain_no_affect_vocabulary():
    """The pilot's premise: no phrase names the feeling it is meant to convey."""
    from data_generation.pilot_emotion_pools import affect_word_leaks

    assert affect_word_leaks() == {}


def test_pilot_pools_have_no_strandable_shortcut_token():
    from data_generation.pilot_emotion_pools import strandable_tokens

    assert strandable_tokens() == {}


def test_strandable_token_guard_is_not_vacuous():
    """The guard exempts closed-class words, so check it still polices plenty.

    A stopword list wide enough to make ``strandable_tokens()`` pass by
    exempting everything would be worse than no guard at all, because it would
    read as green. This pins that most of the vocabulary is still in scope and
    that a good share of it is genuinely shared between tones.
    """
    from data_generation.pilot_emotion_pools import token_idea_counts

    counts = token_idea_counts()
    assert len(counts) > 600, len(counts)
    shared = [t for t, per_label in counts.items() if len(per_label) > 1]
    assert len(shared) > 150, len(shared)


def test_pool_breadth_is_uniform_across_tones():
    """Breadth was the pilot's binding constraint, so the count is pinned.

    Unequal idea counts would also make the register-stratified holdout
    withhold different numbers of ideas per tone, which is what made the very
    first v3 run's per-class numbers incomparable.
    """
    from data_generation.pilot_emotion_pools import (
        DEFLECTION_IDEAS,
        OPENER_IDEAS,
        SENTIMENT_CLASSES as PILOT_CLASSES,
        phrase_register,
    )

    shapes = set()
    for label in PILOT_CLASSES:
        openers, deflections = OPENER_IDEAS[label], DEFLECTION_IDEAS[label]
        assert len(openers) + len(deflections) >= 35, label
        shapes.add((
            len(openers),
            len(deflections),
            sum(1 for k in openers if phrase_register(k) == "code-mixed"),
            sum(1 for k in deflections if phrase_register(k) == "code-mixed"),
        ))
    assert len(shapes) == 1, shapes


def test_idea_keys_declare_their_register():
    """Registers are declared by the ``cm-`` prefix, not sniffed off the key.

    Sniffing romanised-Hindi markers in a slug misreads English keys whose
    fragments collide with short particles ("se", "ka", "do"), which would put
    an English idea in the code-mixed stratum and distort the holdout.
    """
    from data_generation.pilot_emotion_pools import (
        ALL_IDEAS,
        SENTIMENT_CLASSES as PILOT_CLASSES,
        phrase_register,
    )

    for label in PILOT_CLASSES:
        for idea in ALL_IDEAS[label]:
            expected = "code-mixed" if idea.startswith("cm-") else "english"
            assert phrase_register(idea) == expected, (label, idea)


def test_pilot_ideas_do_not_contain_one_another():
    """Substring collisions would misfile transcripts in the holdout partition."""
    from data_generation.pilot_emotion_pools import substring_collisions

    assert substring_collisions() == {}


def test_pilot_every_tone_keeps_code_mixed_ideas_in_training():
    from sentiment_intent.phrase_holdout import held_out_phrases

    from data_generation.pilot_emotion_pools import (
        ALL_IDEAS,
        SENTIMENT_CLASSES as PILOT_CLASSES,
        phrase_register,
    )

    held = held_out_phrases("sentiment", pools="pilot")
    for label in PILOT_CLASSES:
        code_mixed = [i for i in ALL_IDEAS[label] if phrase_register(i) == "code-mixed"]
        assert len(code_mixed) >= 4, f"{label} has only {len(code_mixed)} code-mixed ideas"
        survivors = [
            idea for idea in code_mixed
            if not any(p in held[label] for p in ALL_IDEAS[label][idea])
        ]
        assert survivors, f"{label} keeps no code-mixed idea in training"


def test_pilot_corpus_is_balanced_on_tone_among_tone_bearing_rows(pilot_corpus):
    """The four tones are equal once the tone-free Neutral rows are set aside.

    Those rows are deliberately Neutral-only — there is no such thing as a
    tone-free Frustrated transcript — so the corpus is mildly Neutral-heavy by
    design. What must stay exactly balanced is the tone-bearing part, since
    that is what the holdout partitions.
    """
    bearing = [r for r in pilot_corpus if r["emotion_family"] != "none"]
    counts = collections.Counter(r["sentiment"] for r in bearing)
    assert set(counts) == set(SENTIMENT_CLASSES)
    assert len(set(counts.values())) == 1, counts


def test_tone_free_rows_are_all_neutral(pilot_corpus):
    """"No affect stated" must mean Neutral, or the label is not learnable."""
    tone_free = [r for r in pilot_corpus if r["emotion_family"] == "none"]
    assert tone_free, "no tone-free rows; the #9 spot-check fix is missing"
    assert {r["sentiment"] for r in tone_free} == {"Neutral"}
    assert all(r["emotion_register"] == "none" for r in tone_free)


def test_tone_free_rows_are_half_messy(pilot_corpus):
    tone_free = [r for r in pilot_corpus if r["emotion_family"] == "none"]
    messy = sum(1 for r in tone_free if r["style"] == "messy")
    assert messy / len(tone_free) == 0.5, (messy, len(tone_free))


def test_tone_free_rows_never_reach_the_held_out_set(pilot_corpus):
    """They carry no phrasing, so there is nothing for the holdout to withhold.

    Pinned because it is the honest caveat on the diagnostic: tone-free rows
    are always training-eligible, so they can only ever flatter the seen-idea
    side, never the unseen-idea side the recommendation rests on.
    """
    from sentiment_intent.phrase_holdout import held_out_phrases, partition

    held = held_out_phrases("sentiment", pools="pilot")
    _, heldout, _ = partition(pilot_corpus, "sentiment", held, pools="pilot")
    assert all(r["emotion_family"] != "none" for r in heldout)


def test_pilot_messiness_is_identical_across_tones(pilot_corpus):
    """The load-bearing property of the clean-vs-messy result.

    If messy transcripts were even slightly over-represented in one tone, the
    model could read the label off the disfluencies and the whole pilot would
    measure the wrong thing.
    """
    by_tone = collections.Counter(r["sentiment"] for r in pilot_corpus)
    messy = collections.Counter(
        r["sentiment"] for r in pilot_corpus if r["style"] == "messy"
    )
    for tone in SENTIMENT_CLASSES:
        assert messy[tone] / by_tone[tone] == 0.5, f"{tone}: {messy[tone]}/{by_tone[tone]}"


def test_pilot_messiness_is_identical_across_families_and_registers(pilot_corpus):
    for field in ("emotion_family", "emotion_register"):
        groups = collections.Counter(r[field] for r in pilot_corpus)
        messy = collections.Counter(
            r[field] for r in pilot_corpus if r["style"] == "messy"
        )
        for key, total in groups.items():
            assert messy[key] / total == 0.5, f"{field}={key}: {messy[key]}/{total}"


def test_named_corpora_all_resolve_to_existing_pools():
    from sentiment_intent.data import NAMED_CORPORA, pools_module

    for name, entry in NAMED_CORPORA.items():
        assert pools_module(entry["pools"]) is not None, name


def test_pilot_intent_is_uncorrelated_with_tone(pilot_corpus):
    """Intent is dealt independently, so it cannot stand in for the tone label."""
    cells = collections.Counter((r["sentiment"], r["intent"]) for r in pilot_corpus)
    expected = len(pilot_corpus) / (len(SENTIMENT_CLASSES) * len(INTENT_CLASSES))
    for (tone, intent), count in cells.items():
        assert abs(count - expected) <= expected, f"{tone}/{intent}: {count} vs ~{expected}"


def test_pilot_holdout_drops_no_transcript_as_mixed(pilot_corpus):
    """One idea per transcript is what makes the pilot's holdout lossless.

    The v3 diagnostic discarded 255 of 1000 transcripts because they mixed
    held-out and visible phrasing, which halved its training set for reasons
    unrelated to the confound it was measuring.
    """
    from sentiment_intent.phrase_holdout import held_out_phrases, partition

    held = held_out_phrases("sentiment", pools="pilot")
    trainable, heldout, dropped = partition(pilot_corpus, "sentiment", held, pools="pilot")
    assert dropped == []
    assert trainable and heldout
    balance = collections.Counter(r["sentiment"] for r in heldout)
    assert len(set(balance.values())) == 1, balance


def test_pilot_messiness_never_rewrites_a_tone_expression(pilot_corpus):
    """Every emotion expression must survive verbatim, in messy rows too.

    The holdout finds the tone by substring match. An "um" inserted mid-phrase
    would make a held-out transcript look like it carried no withheld phrasing,
    and it would be silently filed as trainable.
    """
    import json

    from sentiment_intent.data import SENTIMENT_V6_PATH, _normalize

    with SENTIMENT_V6_PATH.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    messy = [r for r in rows if r["metadata"]["style"] == "messy"]
    assert messy, "no messy transcripts to check"
    for row in rows:
        haystack = _normalize(row["text"])
        for expression in row["metadata"]["emotion_expressions"]:
            assert _normalize(expression).strip() in haystack, row["transcript_id"]


def test_pilot_uses_its_own_split_file_not_the_v3_one():
    """A shared split file would be rewritten and destroy the v3 assignment."""
    from sentiment_intent.data import PILOT_SPLIT_PATH, SPLIT_PATH, resolve_corpus

    assert resolve_corpus("pilot")[2] == PILOT_SPLIT_PATH
    assert resolve_corpus("v3")[2] == SPLIT_PATH
    assert PILOT_SPLIT_PATH != SPLIT_PATH


def test_pilot_corpus_is_a_separate_file_from_the_ner_corpus():
    from sentiment_intent.data import (
        SENTIMENT_PILOT_PATH,
        SENTIMENT_V4_PATH,
        SENTIMENT_V6_PATH,
        SYNTHETIC_PATH,
    )

    corpora = {SENTIMENT_PILOT_PATH, SENTIMENT_V4_PATH, SENTIMENT_V6_PATH}
    assert SYNTHETIC_PATH not in corpora
    assert len(corpora) == 3


def test_every_corpus_generation_is_kept_for_comparison():
    """Superseded corpora stay on disk so a regression can be bisected.

    Each round fixed something the previous one measured, and the reports quote
    numbers from each; deleting a corpus would strand those numbers.
    """
    from sentiment_intent.data import NAMED_CORPORA

    assert {"v3", "pilot", "v4", "v5", "v6"} <= set(NAMED_CORPORA)


def test_every_named_corpus_has_its_own_split_file():
    """Two corpora sharing a split file would silently overwrite each other.

    ``build_split`` rewrites its target whenever the corpus fingerprint
    changes, so a shared path means training on one corpus destroys the
    assignment the other was evaluated under.
    """
    from sentiment_intent.data import NAMED_CORPORA

    splits = [str(entry["split"]) for entry in NAMED_CORPORA.values()]
    assert len(set(splits)) == len(splits), splits
    paths = [str(entry["path"]) for entry in NAMED_CORPORA.values()]
    assert len(set(paths)) == len(paths), paths
