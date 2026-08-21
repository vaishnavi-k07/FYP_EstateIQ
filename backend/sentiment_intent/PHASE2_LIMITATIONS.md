# Phase 2 — closing state, known limitations and queued work

**Phase 2 is complete. Both heads are promoted.**

| head | artifact | promoted on |
|---|---|---|
| Sentiment (4-class) | `distilbert_sentiment_v1_20260807_141911` | held-out-phrase generalisation + Enthusiastic/Frustrated separation |
| Intent (6-class) | `distilbert_intent_v1_20260805_183707` | held-out-phrase macro-F1 0.933 (gap 0.067) |

Neither was promoted on its test-split score. A substring matcher over the
generator's own phrase pools scores **1.0** on any test split of a
template-generated corpus, so that number measures memorisation, not language
understanding. The phrase-holdout diagnostic — which withholds whole phrasings
(intent) or whole emotional ideas (sentiment) from training and evaluates only
on those — is the evidence both promotions rest on.

## The promoted sentiment model

`j-hartmann/emotion-english-distilroberta-base` encoder + a fresh 4-class head
fine-tuned on `sentiment_v5.jsonl` (1312 transcripts; 919 train / 197 val / 196
test). Test macro-F1 0.880.

Three rounds got it here, each fixing something the previous one measured:

1. **v3 → v4.** v3 *named* the tone ("we're really excited"), so the model
   learned affect adjectives and inverted when they were withheld — 42% of
   held-out Enthusiastic transcripts became Frustrated. v4 replaced named
   emotion with situational emotion across 38 ideas per tone.
2. **synthetic-only → hybrid.** A pretrained emotion encoder beat the
   synthetic-only DistilBERT on unseen emotional ideas, 0.493 vs 0.407, with
   Enthusiastic recall 0.212 → 0.606 *and* precision 0.306 → 0.538.
3. **v4 → v5.** Every v4 transcript carried an emotional idea, so bare
   requirement text was unrepresented and read as Enthusiastic at 0.886. v5
   adds 96 Neutral transcripts with no emotional expression at all. That line
   now reads Neutral at 0.996.

### Measured on withheld emotional ideas

Re-run on v5 (627 train / 135 seen-idea / 416 unseen-idea, 0 dropped):

| | seen-idea | **unseen-idea** | gap |
|---|---|---|---|
| **promoted (hybrid on v5)** | 0.925 | **0.525** | 0.401 |
| hybrid on v4 | 0.859 | 0.493 | 0.365 |
| synthetic-only v4 (previous) | 0.840 | 0.407 | 0.433 |

Unseen-idea macro-F1 improved again with v5, 0.493 → **0.525**, against a 0.25
chance floor and 0.100 for a phrase matcher restricted to what the model saw.
Per-class on unseen ideas: Enthusiastic 0.529, Neutral 0.563, Hesitant 0.482,
Frustrated 0.524 — the most even spread of any candidate so far, and
Enthusiastic is no longer the outlier it was at 0.250.

Enthusiastic/Frustrated confusion on unseen ideas: **E→F 20.2%, F→E 17.3%**
(combined 18.8%), against the v3 model's one-way 42.2%. Symmetric and roughly
halved.

> **Caveat on the gap number.** v5's 0.401 is not directly comparable to v4's
> 0.365. Tone-free rows carry no phrasing to withhold, so they are always
> training-eligible and land only on the seen-idea side — they lift the
> seen-idea score (0.859 → 0.925) and therefore widen the gap arithmetically.
> The unseen-idea score, which is what the promotion rests on, is unaffected by
> them and went **up**. This is asserted by
> `test_tone_free_rows_never_reach_the_held_out_set`.

---

## Known limitations

### 1. Hinglish Frustrated is unreliable — accepted

The encoder is English-pretrained. On emotional ideas withheld from training:

| | English (n=288) | code-mixed (n=128) |
|---|---|---|
| **promoted hybrid (v5)** | **0.584** | 0.376 |
| hybrid on v4 | 0.535 | 0.373 |
| synthetic-only v4 | 0.406 | 0.396 |
| xlm-emo hybrid (multilingual) | 0.418 | **0.457** |

Every gain from the English encoder has landed on English. Against the previous
synthetic-only head it is **+0.18 on English and −0.02 on code-mixed**, and the
v5 retrain widened the split further rather than closing it — English 0.535 →
0.584 while code-mixed stayed flat at 0.373 → 0.376, taking the register gap
from 0.16 to **0.21**. The 96 tone-free Neutral rows added in v5 are
English-only, which is the likely cause; making a share of them code-mixed is
the cheapest thing to try if this gap starts to matter.

Since calls are mostly English with a Hinglish minority, the trade is
favourable in aggregate. But it is a real gap, and it is concentrated on
**Frustrated**: the hand-written Hinglish frustration line
(*"yaar kitni baar bataya hai, phir se same baat"*) is misread as Hesitant,
while the Hinglish enthusiasm line is read correctly at 0.99.

*Business impact:* a fed-up Hindi-speaking caller may be filed Hesitant rather
than Frustrated. Both are follow-up categories, so the lead is still actioned —
it is a missed escalation signal, not a missed lead.

*Mitigation available now:* `language_hint` is a stored field in the Phase 4
lead-record schema, so this can be measured per segment in production rather
than assumed.

### 2. Held-out gap ~0.40

Strong on emotional ideas it has seen (0.925), materially weaker on ideas it
has not (0.525). The model still learns situations more than it learns tone in
the abstract. It does generalise — 0.525 against a 0.25 chance floor and 0.100
for a phrase matcher restricted to what the model saw — but nowhere near as
well as the 0.880 test-split headline suggests.

*Do not quote the test-split number as evidence of generalisation.* Quote the
held-out one.

### 3. Hesitant is the weakest class

Per-class F1 on withheld ideas: Hesitant 0.482 against 0.529 / 0.563 / 0.524
for the others, and its recall is the lowest at 0.394 — it is under-predicted
rather than confused in both directions. Hesitant↔Frustrated is the main
residual confusion, and all three tone errors in the spot check are that pair.
The v4 synthetic-only head scored Hesitant marginally higher (0.488), so this
is the one class where the hybrid did not clearly win.

Judged an acceptable trade because the errors it replaced crossed the
lead-scoring decision boundary (Enthusiastic↔Frustrated) and these do not.

### 4. Everything is measured on synthetic data

No real call has been scored. The generator's structural guards are green
(no affect-word shortcuts, no strandable tokens, no substring collisions,
messiness balanced to exactly 50% within every tone, family and register) and
the holdout is honest, but synthetic is synthetic. The first real Vapi
transcripts in Phase 5 are the actual test.

---

## Queued / future work

**Deferred by decision — nice-to-have, not blockers:**

1. **MuRIL as the sentiment encoder.** The queued experiment to close the
   Hinglish gap. MuRIL is already arriving for Phase 3 NER, so the cost is
   mostly a training run. Worth doing if production shows the code-mixed share
   is higher than expected, or if missed Frustrated escalations turn out to
   matter commercially.
2. **Register routing.** Alternative to (1): run English calls through the
   promoted model and code-mixed calls through the multilingual hybrid, which
   is the best Hinglish model measured (0.457). Cheaper than retraining but
   doubles the serving footprint.

**Independent of encoder choice:**

3. **Widen the idea pool again.** 38 ideas per tone still under-represents the
   space; breadth was what moved the pilot off the chance floor, and there is
   no reason to think the curve has flattened.
4. **Recover Hesitant.** Its ideas are the least distinct from Frustrated ones.
   Worth writing hesitancy ideas that are clearly *not* aggrieved.
5. **Confidence thresholding in lead scoring.** The model's uncertainty tracks
   genuine ambiguity, so `sentiment_confidence` should gate action rather than
   being ignored. Already a stored field in the Phase 4 schema.

---

## Reproduce

```
python -m data_generation.sentiment_pilot                       # v5, 1312 rows
python -m sentiment_intent.train_sentiment --corpus v5 \
    --base-model j-hartmann/emotion-english-distilroberta-base
python -m sentiment_intent.phrase_holdout --task sentiment --corpus v5 \
    --base-model j-hartmann/emotion-english-distilroberta-base --label hybrid-emodistilroberta
python -m sentiment_intent.spot_check
python -m sentiment_intent.compare_candidates
```

Environment note: `sentencepiece` and `tiktoken` were installed for the XLM-R
tokenizers used by the multilingual candidates.

## Related documents

- `PRETRAINED_COMPARISON.md` — the synthetic vs pretrained vs hybrid head-to-head.
- `SPOT_CHECK.md` — 10 hand-written utterances across four candidate models.
- `PILOT_FINDINGS.md` — why the corpus was widened before it was scaled.
- `../../docs/lead_record_schema.md` — Phase 4 schema draft, awaiting review.
