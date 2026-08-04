# EstateIQ NLP Demo - Run Sheet

Live demo of the NLP feature-extraction module: a voice call transcript goes in,
structured lead features come out.

**Total time: about 4 minutes.** Two terminals. No internet needed - the model
runs from disk, and nothing calls an external service.

---

## Before you start (do this 5 minutes early)

| Check | Command | Expect |
|---|---|---|
| You are in the project root | `cd "C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP"` | - |
| Model artifact exists | `dir backend\models_store\ner` | a `muril_ner_v1_..._all` folder |
| Port 5000 is free | `netstat -ano \| findstr :5000` | no output |

If port 5000 is busy, use another port everywhere:
`python app.py --port 5050` and `python demo/run_demo.py --url http://localhost:5050`

---

## STEP 1 - Start the server (Terminal 1)

```
cd backend
python app.py
```

**Wait for this banner. Do not move on until you see it:**

```
============================================================
  ESTATEIQ IS READY  (model loaded in 3.4s)
  Webhook: POST /vapi/webhook
============================================================
```

The model is now in memory. Every later request is ~2 seconds, with no cold-load
pause in front of the audience.

> **Say:** "The server has loaded our fine-tuned MuRIL model. This is the same
> endpoint the live Vapi voice agent posts to when a real call ends."

**Leave this terminal running.** Don't type in it again.

---

## STEP 2 - Confirm it's ready (Terminal 2)

```
cd "C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP"
python demo/run_demo.py --check
```

Expect:

```
  [OK] Server is up at http://localhost:5000
  [OK] Extraction model loaded: True
```

> **Say:** "Model's loaded and the webhook is live. Let's send it some calls."

---

## STEP 3 - Run the demo

```
python demo/run_demo.py
```

It pauses before each case and waits for **Enter**, so you control the pace.
Press Enter, talk over the output, then Enter again.

---

### Demo 1 - Clean full requirement

> **Say first:** "This is a normal enquiry call. Watch what we pull out of it."

Press Enter. Point at the **left/top block** (the conversation), then the
**bottom block** (the extracted fields).

Expected:

| Field | Value |
|---|---|
| city | Pune |
| area | Baner |
| property_type | Apartment |
| category | Residential |
| bhk | 2 |
| budget | 60 lakhs -> **6,000,000 rupees** |
| furnishing | Fully Furnished |
| amenities | Swimming Pool, Gym |

> **Point out:** "Nine fields, all from free-flowing speech. Nobody filled in a
> form. And notice `budget_value` - we convert '60 lakhs' into a number, because
> the lead-scoring model needs to compare budgets arithmetically."

---

### Demo 2 - Code-mixed Hindi/English

> **Say first:** "Real callers in Nashik and Pune don't speak clean English.
> Here's a Hinglish call."

Expected: **area = Wakad, city = Pune, bhk = 3, budget = 8,000,000, amenities = Gym, Parking**

> **Point out:** "'Mujhe Wakad mein 3 BHK flat chahiye' - Hindi and English in
> one sentence. This is why we chose MuRIL, a multilingual model trained on
> Indian languages, instead of an English-only one."
>
> **Bonus if asked:** "The city says Pune even though the caller never said
> 'Pune' - we infer it from the area via our locations table."

---

### Demo 3 - Negation

> **Say first:** "Now the interesting one. Listen for what this caller does
> **not** want."

Expected: **property_type = Apartment**, and a flagged line
`! negated  PROPERTY_TYPE:villa`

> **Point out:** "The caller says the word 'villa' out loud. A keyword-matching
> system would record Villa and send this lead the wrong properties. We
> recognise it's a rejection - we record the apartment, and we keep the villa
> separately as a rejection, which is useful for the sales agent to see."

---

### Demo 4 - Multi-city

> **Say first:** "Last one. This caller mentions two cities."

Expected: **city = Nashik** (not Mumbai), area = Gangapur Road, bhk = 3,
budget = **12,000,000**

> **Point out:** "Mumbai is where they live. Nashik is where they're buying. The
> system picks the city they're shopping in, not the one they're calling from.
> Also, 'one crore twenty lakhs' - spelled out in words - becomes 12,000,000."

---

## STEP 4 - Close

> **Say:** "That structured JSON is the input to the next stage: sentiment and
> intent, then the XGBoost lead score that tells the sales team who to call
> back first."

**If they ask "how good is it?":**

> "On a held-out test set of 60 hand-annotated transcripts, the hybrid scores
> **0.97 F1** at entity level. The rule-based baseline gets 0.92 and the neural
> model alone 0.94 - we route each field to whichever performs better, which is
> why the combination beats both."

**If they ask "why hybrid, not just the model?":**

> "Rules are perfect on fixed lists - area names, amenities - because those are
> exact lookups against our database. The model is far better on budget, where
> rules only catch about half the mentions, because people say budgets in
> endless different ways. So we let each one do what it's good at."

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `[X] SERVER NOT RUNNING` | Terminal 1 isn't up, or is on another port. Restart it; wait for the READY banner. |
| Port 5000 already in use | `python app.py --port 5050`, then `python demo/run_demo.py --url http://localhost:5050` |
| First case takes 10+ seconds | Model wasn't preloaded. Stop Terminal 1, run `python app.py` again (preload is on by default). |
| A case errors mid-demo | Skip it: `python demo/run_demo.py --only 1` runs a single case. |
| Everything is broken | Fall back to the offline path - no server needed: `cd backend` then `python -m nlp.predict --demo` |

**Rehearse the fallback once** so it's muscle memory.

---

## Command cheat-sheet

```
# Terminal 1
cd backend
python app.py

# Terminal 2
python demo/run_demo.py --check      # readiness only
python demo/run_demo.py              # full demo, pauses between cases
python demo/run_demo.py --no-pause   # runs straight through
python demo/run_demo.py --only 3     # just the negation case

# Offline fallback (no server at all)
cd backend
python -m nlp.predict --demo
```
