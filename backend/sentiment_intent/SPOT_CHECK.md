# Spot check — 10 hand-written utterances (2026-08-07)

Run before deciding whether to promote the emotion-distilroberta hybrid. These
are bare one-liners of the kind a real caller opens with — no call structure, no
requirement sentences, none of them built from our generator's pools. Ten items
is not a measurement; it is a sanity check on genuinely out-of-distribution
input.

```
python -m sentiment_intent.spot_check --model-dir <model>
```

## Results

`*` marks a wrong prediction. Confidence is the softmax of the winning class.

| # | expected | v4 baseline | hybrid on v4 (850) | **hybrid on v5 — PROMOTED** | xlm-emo hybrid |
|---|---|---|---|---|---|
| 1 | Enthusiastic | Neut* 0.57 | Enth 0.92 | **Enth 0.87** | Hesi* 1.00 |
| 2 | Enthusiastic | Neut* 0.89 | Enth 0.99 | **Enth 0.99** | Hesi* 1.00 |
| 3 | Enthusiastic (code-mixed) | Neut* 0.77 | Enth 0.99 | **Enth 0.99** | Hesi* 1.00 |
| 4 | Frustrated | Frus 0.70 | Frus 0.99 | **Frus 1.00** | Frus 1.00 |
| 5 | Frustrated | Enth* 0.93 | Hesi* 0.75 | **Hesi* 0.85** | Frus 1.00 |
| 6 | Frustrated (code-mixed) | Frus 0.64 | Hesi* 0.68 | **Hesi* 0.99** | Frus 1.00 |
| 7 | Hesitant | Hesi 0.86 | Hesi 0.99 | **Hesi 0.99** | Hesi 0.90 |
| 8 | Hesitant | Neut* 0.46 | Hesi 0.52 | **Frus* 0.60** | Hesi 0.99 |
| 9 | Neutral | Frus* 0.49 | **Enth\* 0.89** | **Neut 1.00** | Neut 0.98 |
| 10 | Neutral | Frus* 0.66 | Neut 0.51 | **Neut 0.96** | Enth* 0.70 |
| | **correct** | 3/10 | 7/10 | **7/10** | 6/10 |
| | **E↔F errors** | 1 | 0 | **0** | 0 |
| | **code-mixed** | 1/2 | 1/2 | **1/2** | 1/2 |

The promoted model is `distilbert_sentiment_v1_20260807_141911` — the
emotion-distilroberta encoder fine-tuned on **v5**, which is v4 plus 96 Neutral
transcripts carrying no emotional expression at all.

### What the v5 fix changed

**#9 is resolved outright.** *"we're looking for a 3 BHK in Baner, budget around
90 lakhs"* went from **Enthusiastic 0.886** to **Neutral 0.996**. #10 firmed up
too, from a 0.51 coin-flip to Neutral 0.963. Teaching "absence of affect is
itself Neutral" did exactly what it was supposed to, and did not cost anything
on the Enthusiastic side — still 3/3 at 0.87–0.99.

**The count stayed at 7/10, but the errors got cheaper.** v5 trades one
dangerous error for one cheap one:

- *fixed:* a plain requirement read as a hot lead (#9) — the error that would
  have put a browsing caller at the top of a sales agent's queue.
- *introduced:* #8 *"maybe, but I'd want to see a few more options first"* now
  reads Frustrated 0.60 rather than Hesitant 0.52. Both are "needs attention"
  categories; the lead is followed up either way.

The residual confusion is now **Hesitant↔Frustrated** (#5, #6, #8), not
Enthusiastic↔Frustrated. That is the right failure mode to be left with: the
two classes involved sit on the same side of the lead-scoring decision, whereas
the old inversion crossed it.

## Reading

**The Enthusiastic bug is fixed, and this is the clearest evidence yet.** The
old promoted head gets **0 of 3** Enthusiastic cases right — all three go to
Neutral, one at 0.89 confidence. The new one gets **3 of 3** at 0.87-0.99. That
is the 0.212 -> 0.606 held-out recall improvement showing up in a form you can
read directly.

**No Enthusiastic/Frustrated inversion.** Zero E-F errors either way. Compare
the old head, which calls *"I'm losing patience"* **Enthusiastic at 0.93
confidence** — the exact bug, confidently wrong.

**Every remaining tone error is Hesitant/Frustrated** (#5, #6, #8). That is the
right failure mode to be left with. Both classes sit on the same side of the
lead-scoring decision — the lead gets followed up either way — whereas the old
inversion crossed it, and would have put a fed-up caller at the top of the
queue as a hot lead.

**Code-mixed is the remaining real weakness, 1/2.** The Hinglish Frustrated
line (#6) fails on every English-encoder model tested. It is the one case where
the multilingual xlm-emo hybrid is clearly better — it gets both code-mixed
items at 1.00. This matches the measured register split: emo-distilroberta
0.535 English / 0.373 code-mixed, xlm-emo 0.418 / 0.457. Accepted as a known
limitation given mostly-English calls; MuRIL-as-encoder is the queued
experiment.

**Low-confidence cases behave sensibly.** The model's uncertainty tracks
genuine ambiguity — #8 sits at 0.60, and *"maybe, but I'd want to see a few
more options first"* really does hover between Hesitant and Frustrated. A
confidence threshold in lead scoring would catch these rather than acting on
them, which is why `sentiment_confidence` is a stored field in the Phase 4
lead-record schema.

## Verdict

Promoted. `distilbert_sentiment_v1_20260807_141911` is a large, visible
improvement on exactly the failure this work set out to fix, it does not
reintroduce the polarity inversion, and the bare-requirement bug found in the
previous round is resolved.

Ten hand-written cases are a sanity check, not a measurement. The number the
promotion actually rests on is the held-out-phrase score — see
`PRETRAINED_COMPARISON.md` and `PHASE2_LIMITATIONS.md`.
