# Phase 2c — synthetic vs pretrained vs hybrid (2026-08-07)

**Recommendation: replace the v4 baseline encoder with a pretrained emotion
encoder. Use `j-hartmann/emotion-english-distilroberta-base` as the hybrid
encoder.** It gains +0.086 macro-F1 on unseen emotional ideas and nearly
triples Enthusiastic recall (0.212 → 0.606) with precision rising too, at the
same parameter count as the current baseline.

Zero-shot loses badly and is not a candidate. Multilingual loses to English on
everything except the code-mixed subset.

---

## Method

All candidates scored on identical rows — the v4 phrase-holdout partition:
560 train / 120 seen-idea test / 416 unseen-idea test, 0 dropped.

- **seen-idea** = emotional ideas present in training. Context only: a
  substring matcher scores 1.0 on any seen-idea split of a template corpus.
- **unseen-idea** = the 13 ideas per tone withheld from training entirely.
  **This is the number that decides the comparison**, because the baseline's
  known weakness is exactly here.

Zero-shot models get two folds onto our four tones: an *a-priori* hand-written
mapping, and a *fitted* one where each source label is assigned to the tone it
co-occurs with most **on the training split only** (no gradients). The fitted
fold exists so a bad guess of mine is not scored as a weakness of the model.

## Results

| candidate | seen | **unseen** | gap | EnthR | EnthP | E→F | F→E | eng | code |
|---|---|---|---|---|---|---|---|---|---|
| hybrid: goemotions-roberta | 0.804 | **0.498** | **0.306** | 0.510 | 0.453 | 34% | 19% | 0.540 | 0.403 |
| **hybrid: emotion-distilroberta** | 0.859 | **0.493** | 0.365 | **0.606** | **0.538** | **18%** | 20% | 0.535 | 0.373 |
| hybrid: xlm-emo (multilingual) | 0.883 | 0.431 | 0.452 | 0.337 | 0.340 | 17% | 20% | 0.418 | **0.457** |
| **v4 baseline** (DistilBERT+SST-2) | 0.840 | 0.407 | 0.433 | 0.212 | 0.306 | 31% | 17% | 0.406 | 0.396 |
| zero-shot: xlm-emo | 0.243 | 0.291 | −0.048 | 0.721 | 0.307 | 19% | 47% | 0.313 | 0.245 |
| zero-shot: xlmr-sentiment | 0.186 | 0.262 | −0.075 | 0.202 | 0.553 | 25% | 4% | 0.236 | 0.306 |
| zero-shot: goemotions-roberta | 0.197 | 0.244 | −0.046 | 0.202 | 0.350 | 18% | 18% | 0.288 | 0.138 |
| zero-shot: emotion-distilroberta | 0.154 | 0.174 | −0.052 | 0.048 | 0.833 | 0% | 1% | 0.197 | 0.123 |

Chance for four balanced classes is ~0.25. Zero-shot rows show the better of
the two folds. Their gaps are negative because neither set was trained on.

### Per-class F1, unseen ideas

| tone | v4 | emo-distilroberta | goemotions | xlm-emo |
|---|---|---|---|---|
| Enthusiastic | 0.250 | **0.570** | 0.480 | 0.338 |
| Neutral | 0.447 | **0.520** | 0.508 | 0.452 |
| Hesitant | **0.488** | 0.429 | 0.451 | 0.383 |
| Frustrated | 0.443 | 0.454 | 0.553 | **0.551** |

## Reading

**1. Zero-shot is not viable — best 0.291 against the baseline's 0.407.** Every
off-the-shelf model lands at or barely above chance. The reason is structural,
not a bad fold: the fitted mapping barely moves the numbers. Our four tones are
not a valence axis. Hesitant and Frustrated are both "negative" and mean
opposite things to a sales team, and no public label space carves that joint.
Both multilingual models were left unable to reach one tone at all — xlm-emo
scores Neutral F1 **exactly 0.000**, xlmr-sentiment the same for Hesitant.

That also explains the one eye-catching zero-shot number: xlm-emo's 0.721
Enthusiastic recall is not understanding, it is spraying. Precision is 0.307
and it cannot predict Neutral, so everything positive-ish becomes Enthusiastic.
Recall without precision is exactly the trap here.

**2. The hybrid is a real gain, and it lands where the baseline was weakest.**
Both English emotion encoders beat v4 on unseen ideas by ~0.09. More
importantly the gain is concentrated on Enthusiastic — the class the baseline
was failing — and **precision rises alongside recall** (0.306 → 0.538), so it is
genuinely recognising enthusiasm rather than over-predicting it. The
generalisation gap narrows from 0.433 to 0.365 (emo-distilroberta) or 0.306
(goemotions).

**3. Enthusiastic and Frustrated do not re-invert.** emo-distilroberta is the
most symmetric of any candidate: E→F 18%, F→E 20%, combined 19.2% against the
baseline's 24.0%. goemotions is worse on this axis specifically (E→F 34%),
which is why it is not the recommendation despite a fractionally higher
headline.

**4. English vs code-mixed: the English encoders buy English, not Hinglish.**

| candidate | english (n=288) | code-mixed (n=128) | delta |
|---|---|---|---|
| v4 baseline | 0.406 | 0.396 | +0.01 |
| emo-distilroberta | 0.535 | 0.373 | +0.16 |
| goemotions | 0.540 | 0.403 | +0.14 |
| xlm-emo | 0.418 | **0.457** | −0.04 |

The baseline is register-neutral because it only ever learned our synthetic
data. The English hybrids gain ~+0.13 on English and are flat-to-slightly-worse
on code-mixed. The multilingual hybrid is the only candidate that is *better*
on Hinglish than English — and it is the only one that beats every other
candidate on the code-mixed subset (0.457).

Since our calls are mostly English with a Hinglish minority, the weighted
outcome favours the English encoder. But the multilingual one is not useless:
it is the best Hinglish model measured, and the best Frustrated model measured.

**5. Clean vs messy stays a non-effect** for every candidate (largest delta
0.02, and it changes sign across candidates). Messiness is dealt at exactly 50%
within every tone, so this remains the expected result.

## Recommendation

Adopt **hybrid: `j-hartmann/emotion-english-distilroberta-base` encoder + fresh
4-class head fine-tuned on the v4 corpus**.

- Unseen-idea macro-F1 **0.493 vs 0.407**, +0.086 (+21% relative).
- Enthusiastic recall **0.606 vs 0.212**, precision **0.538 vs 0.306** — the
  specific weakness this exercise set out to fix.
- Lowest E↔F confusion of any candidate (19.2%), so no inversion returns.
- Same size class as the current baseline (DistilRoBERTa vs DistilBERT), so
  inference cost is unchanged.

goemotions-roberta is a defensible alternative — marginally higher unseen-idea
score (0.498) and the narrowest gap (0.306) — but it is twice the size, worse
on Enthusiastic, and notably worse on the E→F cell (34%).

### Worth doing next

1. **Route by register, or fine-tune the multilingual encoder harder.** xlm-emo
   is the best Hinglish model measured (0.457 code-mixed vs 0.373). A single
   multilingual encoder that matches the English one on English would dominate;
   MuRIL is already coming in for Phase 3 and has not been tried here.
2. **Hesitant regressed** (0.488 → 0.429) with the English hybrid. It is the
   one class the baseline did better on, worth a look before promotion.
3. The gap is still 0.365. Widening the idea pool further remains the other
   lever, independent of encoder choice.

## Not done, deliberately

- Sentiment pointer **unchanged** — still `distilbert_sentiment_v1_20260804_183551`.
  Nothing here is promoted, awaiting confirmation.
- Intent remains promoted at `distilbert_intent_v1_20260805_183707`.
- NER corpus, `phrase_pools.py` and the Phase 3 dataset untouched.

## Reproduce

```
python -m sentiment_intent.pretrained_baseline
python -m sentiment_intent.phrase_holdout --task sentiment --corpus v4 \
    --base-model j-hartmann/emotion-english-distilroberta-base --label hybrid-emodistilroberta
python -m sentiment_intent.phrase_holdout --task sentiment --corpus v4 \
    --base-model SamLowe/roberta-base-go_emotions --label hybrid-goemotions
python -m sentiment_intent.phrase_holdout --task sentiment --corpus v4 \
    --base-model MilaNLProc/xlm-emo-t --label hybrid-xlmemo
python -m sentiment_intent.compare_candidates
```

Note: `sentencepiece` and `tiktoken` were installed for the XLM-R tokenizers.
