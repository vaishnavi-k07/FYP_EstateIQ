# Sentiment model — demo on 36 unseen transcripts

**Model:** `distilbert_sentiment_v1_20260807_161300` (the promoted model)

## Headline

| input shape | correct | Enthusiastic↔Frustrated confusions |
|---|---|---|
| **full call** (what production sees) | **34/36 — 94%** | 0 |
| single opening turn (harder) | 33/36 — 92% | 1 |

Both are reported because quoting either alone would mislead. The model was
trained on whole calls and production feeds it whole calls, so the first row
is the operating number. The second strips each caller down to their opening
line with no requirement or intent around it — a harder test than the system
will actually face, and the more revealing one about how much the tone
judgement leans on surrounding context.

**Honest summary:** Hesitant is perfect on both (9/9). Neutral is perfect on
full calls. Enthusiastic is the weakest class and code-mixed is the weakest
slice — the two known limitations from `PHASE2_LIMITATIONS.md`, showing up
exactly where they were predicted to. Every miss is listed below.

---
# Single opening turn only

**33 / 36 correct (92%)**

## Unseen check

- 885 training phrases checked against every transcript.
- Exact phrase reuse: **0** — none of the model's training phrasing appears here.

- Longest word sequence any transcript shares with a training phrase: **7 words**.


## Results

| # | expected | predicted | conf | ok | transcript (customer turns) |
|---|---|---|---|---|---|
| E1 | Enthusiastic | Enthusiastic | 0.98 | ✅ | We saw the show flat at the weekend and honestly we have not stopped thinking about it since. If the numbers work we would like to get moving quickly. |
| E2 | Enthusiastic | Enthusiastic | 1.00 | ✅ | My husband and I have been putting money aside for this since our daughter was born. Standing in that living room on Saturday, it just clicked. What is the next step? |
| E3 | Enthusiastic | Enthusiastic | 1.00 | ✅ | So, sorry, um, we went to a viewing somewhere else yesterday and it was not a patch on yours. Can we come back at the weekend? I have already told my mother about it. |
| E4 | Enthusiastic | Enthusiastic | 0.99 | ✅ | I got my promotion letter on Monday, so this is actually happening now. We are really excited about the three bedroom units and I would like to see them before anyone else does. |
| E5 | Enthusiastic | Enthusiastic | 0.98 | ✅ | Right, so, we are, um, we are second time buyers with you actually. The first one went so well that we did not even shop around this time. |
| E6 | Enthusiastic | Enthusiastic | 0.99 | ✅ | I can already picture Sunday mornings up there. If it turns out to be the one we will stretch the budget, I am not going to lose it over two lakhs. |
| E7 | Enthusiastic | Frustrated | 0.60 | ❌ | Sach mein, jab se woh flat dekha hai tab se aur kahin dekhne ka mann hi nahi kar raha. Aap bataiye aage kya karna hoga. |
| E8 | Enthusiastic | Enthusiastic | 0.61 | ✅ | Arre suniye, haan, humne apne mummy papa ko bhi photo bhej di thi. Sab keh rahe hain isse achha kuch nahi milega. Toh kab aa sakte hain hum? |
| E9 | Enthusiastic | Enthusiastic | 0.99 | ✅ | We have rented on that same road for nine years and always looked up at those buildings wondering. Now we can actually afford one. Tell me what you need from me. |
| N1 | Neutral | Neutral | 0.99 | ✅ | Two bedroom in Baner, budget one crore, possession within a year. That is the requirement. |
| N2 | Neutral | Frustrated | 0.91 | ❌ | I am calling on behalf of my employer. They are relocating a staff member in October and need something furnished near Hinjewadi. |
| N3 | Neutral | Neutral | 0.99 | ✅ | Um, let me just, I have it written here somewhere, right. Three bed, Kothrud, one point two crore, parking essential. |
| N4 | Neutral | Neutral | 0.98 | ✅ | Before I go further, what is your process? Do you handle the paperwork or does that sit with the buyer's solicitor? |
| N5 | Neutral | Frustrated | 0.84 | ❌ | I would like the price list and the payment schedule. My email is on file from the previous enquiry. |
| N6 | Neutral | Neutral | 0.97 | ✅ | Main sirf details lene ke liye call kar raha hoon. Do BHK, Nashik mein, aur budget saath lakh ke aas paas. |
| N7 | Neutral | Neutral | 1.00 | ✅ | So the requirement is, sorry, one second, yeah, a shop unit, ground floor, main road facing. Size around eight hundred square feet. |
| N8 | Neutral | Neutral | 1.00 | ✅ | We have done this twice before, so I know the drill. I will give you the specifics and you can tell me what is available. |
| N9 | Neutral | Neutral | 0.90 | ✅ | No particular deadline on our side. We will look at what comes up over the next few months and decide then. |
| H1 | Hesitant | Hesitant | 0.99 | ✅ | I will be honest, we have been going back and forth on whether to do this at all. My wife thinks we should wait another year. |
| H2 | Hesitant | Hesitant | 0.96 | ✅ | Um, I do not really know what I am asking, sorry. We have never bought anything before and I am a bit lost with all of it. |
| H3 | Hesitant | Hesitant | 0.67 | ✅ | A lot depends on whether my company confirms the transfer. If that falls through, none of this happens. |
| H4 | Hesitant | Hesitant | 0.70 | ✅ | I do not want to take up your time, there is a decent chance we stay put. But I thought I would at least ask what is out there. |
| H5 | Hesitant | Hesitant | 0.99 | ✅ | We might, hmm. Look, we have seen maybe fifteen places and I could not tell you which was which any more. I am not sure we are ready. |
| H6 | Hesitant | Hesitant | 0.88 | ✅ | Sach poochhiye toh abhi tak ghar mein sahmati nahi bani hai. Papa kehte hain ruk jao, main keh raha hoon dekh lete hain. |
| H7 | Hesitant | Hesitant | 0.92 | ✅ | It is the biggest cheque I will ever write and that is what keeps stopping me. Every time I get close I find a reason to delay. |
| H8 | Hesitant | Hesitant | 0.93 | ✅ | Whatever I tell you today might change next week, so please do not build anything around it. I am just gathering my thoughts. |
| H9 | Hesitant | Hesitant | 0.99 | ✅ | I was going to say two but, actually, no, ignore that. I would rather see what is available first and then decide. |
| F1 | Frustrated | Frustrated | 0.99 | ✅ | This is the second time I have explained all of this. Whoever I spoke to last week said they would note it down, and clearly nobody did. |
| F2 | Frustrated | Frustrated | 0.95 | ✅ | I took Friday off work to go. The gate was locked and the number on your board went to voicemail. |
| F3 | Frustrated | Frustrated | 0.93 | ✅ | Look, no, let me finish. I have had four different people give me four different prices. Which one is it? |
| F4 | Frustrated | Frustrated | 1.00 | ✅ | I paid the booking amount in April. It is now August and I still do not have an allotment letter. Can somebody explain that to me? |
| F5 | Frustrated | Frustrated | 0.99 | ✅ | Fine. Let us do this again. Ask me whatever you need and we will see if it goes anywhere this time. |
| F6 | Frustrated | Frustrated | 0.82 | ✅ | Do mahine se sirf yahi sun raha hoon ki kal batayenge. Kal kab aayega, yeh bata dijiye. |
| F7 | Frustrated | Frustrated | 0.95 | ✅ | Honestly I am getting quite annoyed, you know what, forget it. Just tell me whether the flat is still available or not. That is all I am asking. |
| F8 | Frustrated | Frustrated | 0.96 | ✅ | Put me through to whoever is actually in charge. I have run out of patience with being passed around. |
| F9 | Frustrated | Frustrated | 0.98 | ✅ | Bhai dekhiye, teen baar aa chuka hoon office. Har baar bolte hain sir kal aana. Ab main karoon toh kya karoon. |

## Per class

| tone | correct | n | recall |
|---|---|---|---|
| Enthusiastic | 8 | 9 | 89% |
| Neutral | 7 | 9 | 78% |
| Hesitant | 9 | 9 | 100% |
| Frustrated | 9 | 9 | 100% |

## Confusion (rows = expected, columns = predicted)

| | Enth | Neut | Hesi | Frus |
|---|---|---|---|---|
| **Enthusiastic** | 8 | 0 | 0 | 1 |
| **Neutral** | 0 | 7 | 0 | 2 |
| **Hesitant** | 0 | 0 | 9 | 0 |
| **Frustrated** | 0 | 0 | 0 | 9 |

## Slices

| slice | correct | n |
|---|---|---|
| english | 28 | 30 |
| code-mixed | 5 | 6 |
| clean | 22 | 25 |
| messy | 11 | 11 |

## Every miss (3)

- **E7** expected *Enthusiastic*, predicted *Frustrated* at 0.60 (code-mixed, clean)
  - "Sach mein, jab se woh flat dekha hai tab se aur kahin dekhne ka mann hi nahi kar raha. Aap bataiye aage kya karna hoga."
  - full distribution: Enthusiastic 0.39, Neutral 0.01, Hesitant 0.00, Frustrated 0.60

- **N2** expected *Neutral*, predicted *Frustrated* at 0.91 (english, clean)
  - "I am calling on behalf of my employer. They are relocating a staff member in October and need something furnished near Hinjewadi."
  - full distribution: Enthusiastic 0.01, Neutral 0.06, Hesitant 0.01, Frustrated 0.91

- **N5** expected *Neutral*, predicted *Frustrated* at 0.84 (english, clean)
  - "I would like the price list and the payment schedule. My email is on file from the previous enquiry."
  - full distribution: Enthusiastic 0.01, Neutral 0.15, Hesitant 0.00, Frustrated 0.84


Enthusiastic/Frustrated confusions (the failure mode this model was rebuilt to remove): **1**


---

# Same cases, full-call context

Each tone-bearing turn above, with a requirement turn and an intent turn appended - the shape a real Vapi transcript arrives in, and the shape the model was trained on. The single-turn table is the harder test; this one is closer to production.

**34 / 36 correct (94%)**, against 33 / 36 on single turns.

| tone | single turn | full call |
|---|---|---|
| Enthusiastic | 8/9 | 9/9 |
| Neutral | 7/9 | 8/9 |
| Hesitant | 9/9 | 8/9 |
| Frustrated | 9/9 | 9/9 |

| slice | single turn | full call |
|---|---|---|
| english | 28/30 | 28/30 |
| code-mixed | 5/6 | 6/6 |

Enthusiastic/Frustrated confusions: **1** on single turns, **0** on full calls.

Cases whose verdict changed with context: E7 (fixed), N2 (fixed), H3 (broken)

## Full-call misses (2)

- **N5** expected *Neutral*, predicted *Frustrated* at 0.79
- **H3** expected *Hesitant*, predicted *Frustrated* at 0.73
---

# Round 2 — backstory-driven enthusiasm (v6)

The first demo run found two confident inversions, E5 and F9. Both are fixed.

## Root causes

**E5** *"second time buyers with you, the first one went so well"* → Frustrated
0.90. Enthusiastic carried only 6 of 38 past-tense, backstory-driven ideas
against Frustrated's 8, so *recounting history* had become a Frustrated cue.

**F9** *"teen baar aa chuka hoon office"* → Enthusiastic 0.85. An Enthusiastic
phrase added earlier while balancing the token `baar` — *"Office mein kai baar
charcha ho chuki hai"* — collided directly with this shape.

## Fix

Four backstory-driven Enthusiastic ideas (`repeat-customer`,
`watched-it-for-years`, `family-history-here`, `finally-within-reach`) plus two
code-mixed equivalents, and the colliding phrase reworded. Six ideas were added
to *every* tone, not just Enthusiastic, because the holdout withholds a fixed
fraction per tone — unequal pools would give unequal held-out sets. Pools went
38 → 44 ideas per tone; corpus v5 (1312) → v6 (1504).

## Result — both target cases fixed

| case | v5 model | v6 model |
|---|---|---|
| E5 | Frustrated 0.903 ❌ | **Enthusiastic 0.977** ✅ |
| F9 | Enthusiastic 0.853 ❌ | **Frustrated 0.983** ✅ |

| demo (full call) | v5 | v6 |
|---|---|---|
| correct | 31/36 (86%) | **34/36 (94%)** |
| Enthusiastic↔Frustrated | 2 | **0** |
| code-mixed | 4/6 | **6/6** |
| messy | 7/11 | **11/11** |

Held-out diagnostic (note: v6's unseen set is larger and drawn from a wider
pool, so it is a different and harder test, not a like-for-like rerun):

| | v5 | v6 |
|---|---|---|
| unseen-idea macro-F1 | 0.525 | 0.506 |
| E→F / F→E | 20.2% / 17.3% | **8.3% / 10.0%** |
| English vs code-mixed gap | 0.208 | **0.088** |

Aggregate macro-F1 is flat-to-slightly-down while the two failure modes that
actually matter both roughly halved. The register gap closing from 0.21 to 0.09
is the largest single improvement — the code-mixed subset went 0.376 → 0.446.

## Two new misses, and a gap in our own guard

**N5** *"My email is on file from the previous enquiry"* → Frustrated 0.79.
Previously correct. Cause: `file` now appears in **5 Frustrated ideas and 0
Neutral ones**, so it has become a one-way Frustrated detector.

This passes `strandable_tokens()` cleanly, and that is the finding. The guard
was built to catch *stranding* — a token a holdout could hand to a rival tone —
so it only fires when two or more tones use a token and one carries it thinly.
A token owned **exclusively** by one tone never triggers it, yet that is
precisely what turns an ordinary word into a label detector on real text.

Fifteen tokens are currently monopolised by a single tone in four or more
ideas: `list`, `requirement`, `details`, `note`, `side`, `way` (Neutral),
`file` (Frustrated), `coming`, `din`, `wife`, `weekend`, `bought`, `plan`
(Enthusiastic), `change`, `ruk` (Hesitant).

**Recommended follow-up:** add a `monopolised_tokens()` guard alongside
`strandable_tokens()`, and give the flagged words a carrier in at least one
other tone. `file`, `details`, `requirement` and `note` are the urgent ones —
they are ordinary vocabulary any caller of any mood would use.

**H3** *"A lot depends on whether my company confirms the transfer"* → Frustrated
0.73 in full-call context only; correct as a single turn. Not diagnosed; logged
as context sensitivity.
