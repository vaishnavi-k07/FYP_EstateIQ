# Phase 2b — sentiment corpus findings

Two rounds. The 288-transcript pilot (2026-08-06) removed the affect-word
shortcut but could not answer whether that helped, because it never left the
chance floor. The widened corpus (v4, 1216 transcripts over 38 emotional ideas
per tone) clears the interpretability gate and gives a real answer.

**Current verdict: the gate is cleared, and the answer is mixed.** The model
learns the task well in-distribution (0.840 macro-F1) but transfers only weakly
to emotional ideas it has never seen (0.407 against a 0.25 chance floor). The
gap did not close — it widened, for a reason worth understanding before the
next change.

---

## Artifacts

| Artifact | Path |
|---|---|
| Emotion pools (38 ideas/tone) | `data_generation/pilot_emotion_pools.py` |
| Generator | `data_generation/sentiment_pilot.py` |
| v4 corpus (1216) | `data_generation/output/sentiment_v4.jsonl` |
| v4 summary | `data_generation/output/sentiment_v4_summary.json` |
| Pilot corpus (288, kept) | `data_generation/output/sentiment_pilot.jsonl` |
| v4 head (**not promoted**) | `models_store/sentiment_intent/distilbert_sentiment_v1_20260806_150604` |
| v4 holdout head (diagnostic) | `models_store/sentiment_intent/holdout/sentiment_v4` |
| v4 holdout report | `phrase_holdout_report_v4.json` |
| Pilot holdout report | `phrase_holdout_report_pilot.json` |
| Size-matched v3 control | `phrase_holdout_report_cap136.json` |

```
python -m data_generation.sentiment_pilot                                  # 1216, v4
python -m data_generation.sentiment_pilot --transcripts-per-idea 4 --name sentiment_pilot
python -m sentiment_intent.train_sentiment --corpus v4 --no-pointer
python -m sentiment_intent.phrase_holdout  --task sentiment --corpus v4
```

---

## 1. In-distribution first — the interpretability gate

**0.8398 macro-F1** (accuracy 0.8417, n=120), against a 0.25 chance floor and a
0.0987 majority baseline. The gate was ~0.60. **Cleared**, so everything below
is worth reading.

For contrast, the same design at 18 ideas per tone reached 0.245 — chance. The
standard 70/15/15 head on the full v4 corpus reaches **0.8755** test macro-F1
(850 train). Training loss fell 1.40 → 0.12 with validation loss following it
down (1.37 → 0.50), which is learning, not the memorise-and-plateau curve the
pilot showed.

## 2. Held-out gap

| Run | train n | in-distribution | held-out | gap |
|---|---|---|---|---|
| v3 (affect words, reference) | 322 | 0.713 | 0.478 | 0.235 |
| pilot (18 ideas/tone) | 136 | 0.245 | 0.196 | 0.049 |
| **v4 (38 ideas/tone)** | 560 | **0.840** | **0.407** | **0.433** |

Held-out n=416, exactly 104 per tone, 0 transcripts dropped as phrase-mixed.
The restricted template matcher scores 0.100 on the same set, so the model is
doing something a substring matcher cannot.

**The gap widened, and the raw held-out numbers are not comparable across
rows.** v4's held-out test is strictly harder than v3's in two ways: it
withholds 13 whole *ideas* per tone rather than paraphrases of surviving
clauses, and there is no affect vocabulary left to bridge a seen idea to an
unseen one. In v3, a withheld "we're thrilled to be starting" still shared
"excited / delighted / keen" with clauses the model kept, so part of that 0.478
is the shortcut still working across the holdout boundary. The comparisons that
do hold are v4 against chance (0.407 vs 0.250) and v4 in-distribution against
v4 held-out (0.840 vs 0.407).

Read plainly: the model transfers, but retains under half its in-distribution
performance on unseen ideas. It is still learning ideas more than it is
learning tone.

## 3. Enthusiastic ↔ Frustrated, both directions (held-out)

| Run | Enth→Frus | Frus→Enth | combined |
|---|---|---|---|
| v3 | 19/45 = **42.2%** | 1/40 = 2.5% | 23.5% |
| pilot | 2/24 = 8.3% | 8/24 = 33.3% | 20.8% |
| **v4** | 32/104 = **30.8%** | 18/104 = **17.3%** | **24.0%** |

The one-way 42% collapse is gone, but it has become a two-way confusion of
similar total size. Enthusiastic is the weakest class by a distance — recall
0.212, and its 104 held-out transcripts scatter almost evenly across all four
labels (22 / 21 / 29 / 32). Hesitant and Frustrated hold up best (recall 0.567
and 0.519). So the specific v3 pathology — a fixed polarity inversion driven by
a stranded token — is fixed; what remains is that unseen *Enthusiastic
situations* are simply not recognised as enthusiasm.

In-distribution the same pair is nearly clean: 4 of 30 Enthusiastic go to
Frustrated, 0 of 30 the other way.

## 4. Clean vs messy

| set | clean | messy | clean − messy |
|---|---|---|---|
| in-distribution | 0.8131 (n=71) | 0.8707 (n=49) | **−0.058** |
| held-out | 0.4145 (n=208) | 0.3980 (n=208) | **+0.017** |

Now that the model is above chance this is a real measurement, and it says
messiness costs essentially nothing — the two deltas are small and point in
opposite directions, which is what noise looks like rather than an effect. That
is the intended outcome: messiness is dealt at exactly 50% within every tone,
family and register, so it carries no label information, and the disfluencies
evidently do not obscure the tone either.

---

## What to do next

The bottleneck is no longer the shortcut or the corpus size. It is that 25
training ideas per tone still under-represent the space of ways an emotion
appears, and Enthusiastic worst of all.

1. **Widen again, asymmetrically.** Enthusiastic needs the most new ideas; its
   held-out recall is half of Hesitant's. Target 60+ ideas per tone.
2. **Consider reducing the holdout fraction** from 1/3 to 1/4 once the pool is
   wider, so the model trains on more ideas while the test stays honest.
3. **A stronger encoder is now worth trying.** At 0.84 in-distribution the head
   is no longer data-starved, so the ceiling may be DistilBERT's; the Phase 3
   NER work already pulls in MuRIL, which handles the code-mixed half natively.
4. **Do not read the test-split number as progress.** The template matcher
   scores 1.0 on any split of a template-generated corpus. The held-out-phrase
   score is the only one that means anything here.

## Not done, deliberately

- Sentiment **not promoted**. `sentiment_latest.json` still points at
  `distilbert_sentiment_v1_20260804_183551`, awaiting confirmation.
- NER corpus, `phrase_pools.py` and the Phase 3 dataset untouched.
- Intent remains promoted at `distilbert_intent_v1_20260805_183707`
  (held-out-phrase macro-F1 0.933, gap 0.067).
