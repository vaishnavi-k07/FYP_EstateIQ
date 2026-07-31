# CLAUDE.md — EstateIQ

This file is the operating guide for Claude Code (and any collaborator) working on this repository. It defines scope, architecture, coding standards, and a phase-by-phase build plan. Follow it strictly — do not introduce features outside this scope (see "Explicitly Out of Scope").

---

## 1. Project Summary

EstateIQ is an AI-powered real estate **lead intelligence system**. A customer calls a live AI voice agent from the company website. The agent has a natural conversation, asking the questions a real sales agent would ask. The resulting transcript is run through three AI modules — NLP feature extraction, sentiment/intent analysis, and ML lead scoring — and the results are surfaced to the real estate sales team through an admin dashboard so they know exactly who to call back first and why.

**This is a lead-intelligence and lead-prioritization system. It is NOT a property recommendation engine and must never suggest specific properties or "next best action" content to the lead.** Its only output to the business is: structured requirements, sentiment/intent, a lead score, a lead category (Hot/Warm/Cold), and an explanation of that score.

---

## 2. Two Interfaces

### 2.1 User Interface (public-facing)
- A "Talk with Agent" button/widget embedded on the company website.
- Launches a live voice session with the AI agent (Vapi AI).
- The agent asks questions to naturally uncover: city, area, property type, BHK, budget, amenities, furnishing, buy/rent/invest intent, timeline.
- No login required for the user. No property listings or recommendations are shown to the user during or after the call — the call is purely conversational data capture.

### 2.2 Admin Interface (internal, sales team)
- Role-based authentication (Admin / Sales Agent roles).
- Dashboard of all leads with: transcript, extracted features, intent, sentiment, lead score, lead category, SHAP explanation.
- Aggregate analytics: total inquiries, hot/warm/cold lead counts, missed leads (no follow-up within SLA), completed buyers (manually marked converted), trends over time, city/area breakdown.
- Sales agents can update a lead's status (New → Contacted → Qualified → Converted / Lost) — this status becomes future training/feedback data.

---

## 3. Explicitly Out of Scope

Do not build any of the following, even if convenient given the codebase:
- Property recommendation or matching engine
- Suggesting "next best action" or follow-up scripts to the lead
- Property listing browsing/search UI for the end user
- Payment or transaction processing

---

## 4. Architecture & Data Flow

```
Customer clicks "Talk with Agent" (website widget)
        │
        ▼
   Vapi AI (voice call, speech-to-text, live Q&A)
        │  webhook (call ended → transcript + metadata)
        ▼
   Flask Backend (webhook receiver)
        │
        ▼
   Transcript stored in DB (raw)
        │
        ▼
 ┌──────────────┬────────────────────┬─────────────────────┐
 │ NLP Extractor│ Sentiment/Intent    │ (waits for both)    │
 │ (spaCy/regex/│ (fine-tuned         │                     │
 │  CSV lookup) │  DistilBERT)        │                     │
 └──────┬───────┴─────────┬──────────┘                     │
        ▼                 ▼                                 │
   structured JSON    intent + sentiment labels              │
        └────────────────┬─────────────────────────────────┘
                          ▼
                 Lead Scoring (XGBoost)
                          ▼
                 SHAP Explainability
                          ▼
                    Lead record in DB
                          ▼
                 Admin Dashboard (Streamlit)
```

---

## 5. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ | Type hints required |
| Backend / API | Flask | REST API, webhook receiver |
| Voice AI | Vapi AI | Voice calls, STT, webhook integration |
| NLP extraction | spaCy, regex, pandas, CSV lookup tables | Rule-based, no deep learning here |
| Sentiment & Intent | Fine-tuned DistilBERT (HuggingFace `transformers`) | Two-stage: pretrain on general sentiment corpus → domain-adapt on labeled real-estate transcripts. Tokenization uses DistilBERT's own tokenizer, not TF-IDF |
| Lead Scoring | XGBoost | Explainable via SHAP; trained on NLP + intent + sentiment features |
| Explainability | SHAP | Feature-level explanation per lead score |
| Database | PostgreSQL (SQLAlchemy ORM) | Swap to SQLite only for local dev if needed |
| Admin Dashboard | Streamlit + Plotly | Auth via `streamlit-authenticator` or a lightweight Flask-session-backed gate |
| Auth | Role-based (Admin / Sales Agent) | Simple JWT or session-based; no need for enterprise SSO |
| Deployment | Docker Compose (backend + dashboard + Postgres) | Cloud target: Render/Railway/AWS EC2 — decide in Phase 9 |

---

## 6. Folder Structure

```
EstateIQ/
├── backend/
│   ├── app.py                     # Flask app entrypoint
│   ├── config.py
│   ├── models/                    # SQLAlchemy models
│   │   ├── lead.py
│   │   ├── transcript.py
│   │   └── user.py
│   ├── routes/
│   │   ├── webhook.py             # Vapi webhook receiver
│   │   ├── leads.py               # CRUD for leads (admin)
│   │   └── auth.py
│   ├── dataset/
│   │   ├── locations.csv
│   │   ├── property_type.csv
│   │   └── amenities.csv
│   ├── nlp/
│   │   ├── extractor.py
│   │   └── preprocess.py
│   ├── sentiment_intent/
│   │   ├── train_sentiment.py
│   │   ├── train_intent.py
│   │   ├── predict.py
│   │   └── domain_adapt.py
│   ├── lead_scoring/
│   │   ├── feature_builder.py
│   │   ├── train.py
│   │   ├── score.py
│   │   └── explainability.py      # SHAP
│   ├── models_store/              # saved model artifacts (.pt / .joblib)
│   └── requirements.txt
├── dashboard/
│   ├── app.py                     # Streamlit entrypoint
│   ├── pages/
│   │   ├── 1_Leads.py
│   │   ├── 2_Analytics.py
│   │   └── 3_Lead_Detail.py
│   └── auth.py
├── data_generation/
│   └── synthetic_transcripts.py   # 60% Nashik / 40% Pune synthetic dialogue dataset
├── website_widget/
│   └── talk_with_agent.html       # Vapi web widget embed
├── tests/
├── docker-compose.yml
└── CLAUDE.md
```

---

## 7. Development Principles (non-negotiable)

- Each module (NLP / sentiment-intent / lead scoring) works independently and can be tested in isolation.
- The NLP module contains **no** intent or sentiment logic. Intent/sentiment code contains **no** NLP feature-extraction logic.
- Separate `train.py` and `predict.py`/`score.py` per ML module.
- Prefer classes with clear single responsibility; type hints everywhere; PEP8.
- No hardcoded location/property-type/amenity values — always read from the CSV lookup tables.
- Return structured dictionaries from every extractor/predictor (never free text between modules).
- Use `logging`, not `print`.
- Every model training run is reproducible: fixed seeds, versioned model artifacts, a metrics log (accuracy/F1 at minimum) saved alongside the artifact.

---

## 8. Phase-by-Phase Plan

This maps onto the previously agreed 8-week timeline (Jul 18 – Sep 14), adjusted for the confirmed scope (DistilBERT, no property recommendations).

### Phase 1 — Foundation & Data (Week 1, Jul 18–24)
1. Scaffold the folder structure above; set up Flask app skeleton, Postgres, SQLAlchemy models (`Lead`, `Transcript`, `User`).
2. Build `locations.csv`, `property_type.csv`, `amenities.csv` for the Nashik/Pune company (multiple named developments across both cities).
3. Build the synthetic call-transcript dataset generator: 60% Nashik / 40% Pune, varied intents (buy/rent/invest/inquiry/schedule visit/callback), varied sentiment tone, varied completeness (some customers give full requirements, some vague/incomplete — this matters for scoring realism later).
4. Source a general-purpose public sentiment dataset (e.g. SST-2 / IMDB / Twitter sentiment) for DistilBERT stage-1 pretraining.
5. Define the DB schema for a "lead record" end to end (raw transcript → extracted features → intent → sentiment → score → category → status).

**Exit criteria:** DB migrations run cleanly; synthetic dataset (≥ a few hundred transcripts) generated and stored; general sentiment dataset downloaded and inspected.

### Phase 2 — Sentiment & Intent Model (Week 2, Jul 25–31)
1. Stage 1: fine-tune DistilBERT on the general sentiment dataset (Positive/Neutral/Negative).
2. Stage 2: domain-adapt on the labeled synthetic real-estate transcripts.
3. Train a second DistilBERT head (or a separate lightweight classifier on top of DistilBERT embeddings) for intent classification (Buy / Rent / Inquiry / Schedule Visit / Investment / Commercial / Request Callback).
4. Evaluate both (accuracy, F1, confusion matrix); save artifacts + metrics to `models_store/`.
5. Wrap both in `sentiment_intent/predict.py` returning a structured dict: `{"sentiment": "...", "sentiment_confidence": ..., "intent": "...", "intent_confidence": ...}`.

**Exit criteria:** both models beat a trivial baseline (majority class) by a clear margin on a held-out split; `predict.py` callable end-to-end on a raw transcript string.

### Phase 3 — NLP Feature Extraction (Week 3, Jul 25–31, parallel-able with Phase 2)
1. Build `nlp/preprocess.py`: cleaning, sentence segmentation via spaCy.
2. Build `nlp/extractor.py`: rule-based + CSV-lookup extraction for city, area, property type, category, BHK, budget, amenities, furnishing.
3. Handle synonyms/fuzzy matching (e.g. "2BHK", "2 bedroom", "two bhk" → `bhk: 2`).
4. Handle missing values gracefully (return `null`/`None`, never fabricate).
5. Unit test the extractor against the synthetic transcript set; measure extraction accuracy per field.

**Exit criteria:** extractor produces the exact structured JSON shape shown in the reference doc, with measured per-field accuracy on synthetic data.

### Phase 4 — Lead Scoring Model (Week 4, Aug 8–14)
1. Build `lead_scoring/feature_builder.py`: combine NLP output + intent + sentiment (+ derived features like "has budget", "has BHK", "requested premium amenities", "transcript completeness") into a single feature vector.
2. Label training data: since real conversion outcomes don't exist yet, define a rule-based/heuristic ground-truth label for the synthetic dataset first (documented clearly as a bootstrap heuristic, not real business truth), train XGBoost on it, and design the schema so **real agent-marked outcomes (Converted/Lost) captured in Phase 8+ become the real training signal later**.
3. Output: lead score (0–100) and category (Hot ≥ threshold, Warm, Cold).
4. Add SHAP explainability: for each scored lead, top contributing features (e.g. "Positive sentiment," "Buy intent," "Budget stated," "Requested parking").
5. Wrap in `lead_scoring/score.py` returning `{"score": 92, "category": "Hot", "explanation": [...]}`.

**Exit criteria:** model trains and scores end-to-end on synthetic data; SHAP explanations are human-readable, not raw feature indices.

### Phase 5 — Voice AI Integration (Week 5, Aug 15–21)
1. Configure Vapi AI agent: system prompt for the sales-agent persona, the question flow to elicit the structured fields, call termination logic.
2. Build `routes/webhook.py`: receives Vapi's call-ended webhook, extracts transcript + call metadata, persists raw transcript to DB.
3. On transcript receipt, trigger the pipeline: NLP extraction → sentiment/intent → lead scoring → save full lead record.
4. Build `website_widget/talk_with_agent.html`: minimal page embedding Vapi's web widget, calling the agent on click. No listing/property content shown here (out of scope).

**Exit criteria:** a real (or test) voice call through Vapi results in a fully-populated lead record in the DB with no manual intervention.

### Phase 6 — Backend API for Admin (Week 6, Aug 22–24)
1. `routes/leads.py`: list leads (paginated, filterable by category/city/status/date), get single lead detail, update lead status.
2. `routes/auth.py`: login/logout, role-based access (Admin sees everything + analytics config; Sales Agent sees assigned/all leads and can update status).
3. Aggregate analytics endpoints: totals by category, missed leads (no status update within SLA window, e.g. 48h), completed buyers, trend over time, breakdown by city/area/property type.

**Exit criteria:** Postman/HTTPie-verified API covering all dashboard needs; auth blocks unauthenticated access.

### Phase 7 — Admin Dashboard (Week 6–7, Aug 25–Sep 1)
1. Streamlit multipage app: Leads table (sortable/filterable), Lead Detail page (transcript + extracted features + intent + sentiment + score + SHAP chart), Analytics page (Plotly charts: inquiries over time, category breakdown, city/area breakdown, missed-lead rate, conversion rate).
2. Wire in role-based auth gate.
3. Status-update action on the Lead Detail page (feeds back into future model retraining).

**Exit criteria:** a sales agent can log in, see the lead list sorted by score, open a lead, understand why it scored the way it did, and mark it Contacted/Converted/Lost.

### Phase 8 — Testing & Evaluation (Week 8, Sep 2–7)
1. Unit tests per module (NLP extractor, sentiment/intent predict, lead scoring).
2. Integration test: simulated transcript → full pipeline → DB record → dashboard reflects it.
3. Model evaluation report: sentiment/intent metrics, lead-scoring feature importances (global SHAP summary), extraction accuracy by field.
4. Load-test the webhook receiver lightly (simulate multiple concurrent calls ending).

**Exit criteria:** test suite passes; a short evaluation report (metrics + limitations) is written — this becomes part of the final report.

### Phase 9 — Deployment (Sep 8–11)
1. Dockerize backend, dashboard, and Postgres via `docker-compose.yml`.
2. Deploy to chosen host (Render/Railway/EC2 — pick based on budget/familiarity).
3. Configure Vapi webhook URL to point at the deployed backend.
4. Smoke-test the full flow in the deployed environment.

**Exit criteria:** a live URL where a real call can be placed and the result shows up in the deployed dashboard.

### Phase 10 — Documentation, Demo & Buffer (Sep 12–14)
1. Architecture diagram, README, module docs.
2. Demo script: place a live call, walk mentor through NLP output → sentiment/intent → lead score + SHAP explanation → dashboard analytics.
3. Buffer for bug fixes found during rehearsal.

---

## 9. Open Items / Decisions Log

- **Confirmed:** sentiment & intent use fine-tuned DistilBERT, not TF-IDF + Logistic Regression.
- **Confirmed:** no property recommendation or follow-up-suggestion feature — lead scoring/prioritization only.
- **Bootstrap labeling risk (Phase 4):** since there's no real conversion history at project start, the initial lead-scoring ground truth is a documented heuristic on synthetic data. Call this out explicitly in the final report as a limitation, with real agent-marked outcomes (Phase 7 status updates) as the path to a truer model post-launch.
- **Not yet decided:** exact deployment host for Phase 9 (Render vs Railway vs EC2) — revisit once Phase 8 is complete and budget/time is clearer.
