# Lead record schema — Phase 4 design document

**Status: draft for review. No modelling code should be written against this
until it is signed off.** This is the shape on paper: what a lead record holds
end to end, where each field comes from, and which fields exist today versus
which only become real once sales agents start marking outcomes.

Scope note: EstateIQ is a lead-intelligence system. This schema stores
requirements, tone, a score and an explanation. It stores nothing that
constitutes a property recommendation, and no field here is ever surfaced to
the caller (CLAUDE.md §1).

---

## 1. The pipeline this schema has to hold

```
Vapi call ends
   └─ webhook  ──►  transcripts        (raw text + call metadata)
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               │
   nlp/predict    sentiment_intent/     │
   (entities)     predict (tone+intent) │
        │               │               │
        └───────┬───────┘               │
                ▼                        │
        lead_features   ◄────────────────┘   (one row per scoring attempt)
                │
                ▼
        lead_scores      (score + category + SHAP explanation)
                │
                ▼
        leads            (the durable business object agents act on)
                │
                ▼
        lead_status_events   ◄── the real training signal, Phase 8+
```

Five tables rather than one wide row, for three reasons: the extraction output
is versioned and will be re-run when the NER model improves; a lead is scored
more than once over its life; and the status history is an append-only audit
trail, not a mutable column.

---

## 2. `transcripts`

Raw capture. Never edited after insert.

| field | type | source | notes |
|---|---|---|---|
| `transcript_id` | uuid PK | generated | |
| `vapi_call_id` | text unique | Vapi webhook | idempotency key — a redelivered webhook must not create a second lead |
| `caller_phone` | text | Vapi | **PII**, see §8 |
| `started_at` / `ended_at` | timestamptz | Vapi | |
| `duration_seconds` | int | derived | very short calls are a data-quality flag |
| `raw_text` | text | Vapi | full dialogue, `Agent:` / `Customer:` prefixed |
| `customer_text` | text | derived | customer turns only — what both models score. Stored, not recomputed, so a change to `customer_turns()` cannot silently alter what a past score was based on |
| `language_hint` | text | Vapi / detector | `en`, `hi`, `mixed` — needed for the known Hinglish weakness (§7) |
| `received_at` | timestamptz | backend | |

---

## 3. `lead_features` — one row per extraction run

Model outputs, kept separate from the lead so re-running an improved extractor
appends rather than overwrites.

**Requirement fields** (from `nlp/predict.py`, CSV-normalised):

| field | type | notes |
|---|---|---|
| `city` | text | FK-ish to `locations.city`; null when never stated |
| `area` | text | null when never stated |
| `property_type` | text | canonical name from `property_type.csv` |
| `category` | text | Residential / Commercial / … — derived from property type, **not** an intent (CLAUDE.md Phase 2) |
| `bhk` | text | `1`–`5` or `1RK`; null for plots, commercial, studio |
| `budget_lakhs` | numeric | normalised to lakhs; `1.14 crore` → `114.0` |
| `budget_text` | text | the surface form, kept for the dashboard and for auditing the parser |
| `amenities` | text[] | canonical amenity names |
| `furnishing` | text | Furnished / Semi Furnished / Fully Furnished / Unfurnished |

**Tone and intent** (from `sentiment_intent/predict.py`, the contract it
already returns):

| field | type | notes |
|---|---|---|
| `sentiment` | text | Enthusiastic / Neutral / Hesitant / Frustrated |
| `sentiment_confidence` | numeric | **must be stored** — see §7, low-confidence predictions should not drive action |
| `intent` | text | Buy / Rent / Inquiry / Schedule Visit / Investment / Request Callback |
| `intent_confidence` | numeric | |

**Derived completeness signals** (computed here, not by either model):

| field | type | notes |
|---|---|---|
| `fields_stated` | int | how many of the nine requirement fields are non-null |
| `has_budget` / `has_bhk` / `has_area` | bool | the individually predictive ones |
| `completeness` | text | full / partial / vague, bucketed from `fields_stated` |

**Provenance** — the part that makes this reproducible:

| field | type | notes |
|---|---|---|
| `nlp_model_version` | text | e.g. `muril_ner_v1_2026…` or `rules_v2` |
| `sentiment_model_version` | text | e.g. `distilbert_sentiment_v1_20260807_…` |
| `intent_model_version` | text | |
| `extracted_at` | timestamptz | |

> Without these three version columns a score cannot be explained six weeks
> later, and the Phase 8 evaluation cannot separate "the model got better" from
> "the leads got better".

---

## 4. `lead_scores` — one row per scoring run

| field | type | notes |
|---|---|---|
| `score` | int 0–100 | |
| `category` | text | Hot / Warm / Cold |
| `thresholds` | jsonb | the cut points used **at the time**, so a later threshold change does not silently re-categorise history |
| `explanation` | jsonb | ordered SHAP contributions, human-readable |
| `scoring_model_version` | text | |
| `label_source` | text | **`heuristic_v1` or `agent_outcome`** — see §6 |
| `scored_at` | timestamptz | |

`explanation` shape — human-readable feature names, never raw indices
(CLAUDE.md Phase 4 exit criteria):

```json
[
  {"feature": "Intent: Buy",        "contribution":  12.4, "direction": "up"},
  {"feature": "Budget stated",      "contribution":   8.1, "direction": "up"},
  {"feature": "Sentiment: Hesitant","contribution":  -5.2, "direction": "down"},
  {"feature": "No area given",      "contribution":  -3.0, "direction": "down"}
]
```

---

## 5. `leads` — the object the sales team acts on

| field | type | notes |
|---|---|---|
| `lead_id` | uuid PK | |
| `transcript_id` | uuid FK | |
| `current_features_id` | uuid FK | latest `lead_features` |
| `current_score_id` | uuid FK | latest `lead_scores` |
| `status` | text | New → Contacted → Qualified → Converted / Lost |
| `assigned_to` | uuid FK → users | nullable |
| `first_contacted_at` | timestamptz | null until an agent acts — drives the missed-lead SLA |
| `closed_at` | timestamptz | set when Converted or Lost |
| `created_at` / `updated_at` | timestamptz | |

**Missed lead** is a derived condition, not a stored flag:
`status = 'New' AND now() - created_at > SLA` (48h per CLAUDE.md Phase 6).
Storing it would go stale the moment the SLA is renegotiated.

---

## 6. `lead_status_events` — the future training signal

Append-only. **This is the table Phase 4 exists to set up.**

| field | type | notes |
|---|---|---|
| `event_id` | uuid PK | |
| `lead_id` | uuid FK | |
| `from_status` / `to_status` | text | |
| `changed_by` | uuid FK → users | |
| `changed_at` | timestamptz | |
| `note` | text | free text, agent's own words |
| `lost_reason` | text | nullable, controlled vocabulary — budget / timing / location / went elsewhere / unreachable / not serious |

### Why this matters more than the model

At project start there is no conversion history, so the initial lead-scoring
ground truth is **a documented heuristic over synthetic data, not business
truth** (CLAUDE.md §9). That heuristic is a bootstrap: it encodes what we
*assume* makes a lead good (clear intent, stated budget, positive tone) and can
only ever reproduce those assumptions back to us.

Every row an agent writes here is a real label. Once enough accumulate, the
target changes from "heuristic score" to "did this lead actually convert", and
the model can learn things the heuristic cannot — for instance whether
Hesitant-but-complete leads convert better than Enthusiastic-but-vague ones,
which is exactly the question the 4-class tone scheme was designed to make
askable.

Two schema decisions follow directly, and both are cheap now and expensive later:

1. **`label_source` on every score.** When retraining begins, the training set
   must be filterable to agent-labelled leads only. Without this column,
   heuristic-scored and outcome-scored leads are indistinguishable and the real
   signal is diluted by the assumptions it was meant to replace.
2. **`lost_reason` as a controlled vocabulary.** "Lost — budget" and "Lost —
   unreachable" are not the same outcome. The first is a genuine negative; the
   second is mostly a data-quality problem and should probably be excluded from
   training rather than taught as a negative.

### Readiness gate (proposed, for review)

Do not retrain on real outcomes until: **≥ 200 closed leads**, **≥ 50 in the
minority class**, spanning **≥ 2 months**. Below that the model will fit one
quarter's market conditions. Until the gate is met, the heuristic model stays
in service and the outcomes only accumulate.

---

## 7. Fields that exist because of measured model limitations

Not defensive padding — each of these is here because Phase 2 measured a
specific failure.

- **`sentiment_confidence`.** The spot check produced genuine near-ties (0.52,
  0.51) on utterances that really are ambiguous. The dashboard should show tone
  as uncertain below a threshold rather than asserting it, and lead scoring
  should down-weight it.
- **`language_hint`.** The promoted sentiment model is English-strong and
  weaker on the code-mixed minority. Storing the language of the call means the
  limitation can be *measured in production* per segment, and lets code-mixed
  calls be routed to a different model later without a schema change.
- **`nlp_model_version` on features.** Phase 3 will replace the rule extractor
  with MuRIL NER. Re-extraction must be a new row, so before/after can be
  compared on the same calls.
- **`thresholds` on scores.** Hot/Warm/Cold cut points will be tuned once real
  conversion data arrives; history must not silently re-categorise.

---

## 8. PII and retention

- `caller_phone` and `raw_text` are personal data. Access restricted to
  authenticated Admin / Sales Agent roles (CLAUDE.md Phase 6); no unauthenticated
  endpoint may return either.
- Analytics endpoints must aggregate — never return `caller_phone` in a
  breakdown response.
- Proposed retention: raw transcripts 24 months, then delete `raw_text` and
  `caller_phone` while keeping `lead_features` and `lead_scores`, which are
  non-identifying and are what the model needs.
- Deletion must cascade from `transcripts` to nothing else — features and
  scores are retained deliberately.

---

## 9. Open questions for review

1. **Re-scoring policy.** When a model is promoted, do we re-score historical
   leads (comparable metrics, but changes what agents already saw) or only new
   ones (stable history, mixed-version dashboard)? My inclination: re-score into
   *new* rows and let the dashboard show the current one, since the schema
   already supports it.
2. **Multiple calls, one caller.** Same phone number calling twice — one lead
   updated, or two leads linked by a `caller_id`? Affects the missed-lead metric
   materially.
3. **Is `lost_reason` mandatory on Lost?** Enforcing it improves the training
   signal but adds friction that may push agents to pick a default.
4. **Score range.** 0–100 is in the plan; confirm it is a calibrated probability
   ×100 rather than an arbitrary scale, since SHAP contributions are easier to
   explain against a probability.
5. **Readiness gate numbers** in §6 — are 200 / 50 / 2 months right for the
   expected call volume?

---

## 10. What Phase 4 builds against this (not yet started)

1. `lead_scoring/feature_builder.py` — `lead_features` row → model vector.
2. Documented heuristic labeller over the synthetic corpus, written so it is
   obviously replaceable.
3. `lead_scoring/train.py` — XGBoost on the heuristic labels.
4. `lead_scoring/explainability.py` — SHAP → the `explanation` JSON above.
5. `lead_scoring/score.py` — returns `{"score", "category", "explanation"}`.

Nothing above is written until this document is approved.
