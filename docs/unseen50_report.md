# EstateIQ — held-out evaluation on 50 unseen transcripts

## 0. What actually ran

| Module | Loaded | Artifact |
|---|---|---|
| Extraction | MuRIL hybrid (nlp.predict.HybridExtractor) | `C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP\backend\models_store\ner\muril_ner_v1_20260804_074730_all` |
| Sentiment | DistilBERT 4-class sentiment head (sentiment_intent.predict) | `C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP\backend\models_store\sentiment_intent\distilbert_sentiment_v1_20260925_195703` |
| Intent | DistilBERT 6-class intent head (sentiment_intent.predict) | `C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP\backend\models_store\sentiment_intent\distilbert_intent_v1_20260928_163707` |

## 1. The test set

50 transcripts, written for this evaluation. Sentiment balance: Enthusiastic 13, Neutral 12, Hesitant 13, Frustrated 12.
Register: 37 English, 13 code-mixed (Hinglish). Style: 37 clean, 13 disfluent.

**Unseen check** — every case measured against the training corpora:

- Sources checked: 0 NER training transcripts, 885 sentiment generator phrases.
- Verbatim training phrases reused: **0**.
- Longest contiguous word run any case shares with training text: **4 words**.

## 2. Extraction — per-case results

`✅` = all seven fields exact. Misses list `field: expected → predicted`.

| # | Reg | Tags | Result | Misses |
|---|---|---|---|---|
| E01 | engl | — | ✅ | — |
| E02 | engl | spelled-budget | ❌ | **BUDGET**: 25000000 → — |
| E03 | code | — | ✅ | — |
| E04 | engl | range-budget, spelled-budget | ❌ | **FURNISHING**: — → Liveable |
| E05 | engl | — | ✅ | — |
| E06 | code | — | ✅ | — |
| E07 | engl | — | ✅ | — |
| E08 | engl | multi-city, distractor | ✅ | — |
| E09 | code | negation | ❌ | **PROPERTY_TYPE**: Apartment → Villa |
| E10 | engl | — | ✅ | — |
| E11 | engl | — | ✅ | — |
| E12 | engl | spelled-budget | ✅ | — |
| E13 | code | bare-rupees, monthly-rent | ✅ | — |
| N01 | engl | — | ✅ | — |
| N02 | engl | — | ✅ | — |
| N03 | code | — | ✅ | — |
| N04 | engl | range-budget | ✅ | — |
| N05 | engl | bare-rupees, monthly-rent | ✅ | — |
| N06 | code | — | ✅ | — |
| N07 | engl | multi-city | ✅ | — |
| N08 | engl | — | ✅ | — |
| N09 | engl | negation | ✅ | — |
| N10 | code | bare-rupees, monthly-rent | ❌ | **BUDGET**: 22000 → — |
| N11 | engl | bare-rupees, monthly-rent | ❌ | **BUDGET**: 12000 → — |
| N12 | engl | spelled-budget, negation | ✅ | — |
| H01 | engl | — | ✅ | — |
| H02 | engl | — | ❌ | **PROPERTY_TYPE**: — → Independent House |
| H03 | code | — | ✅ | — |
| H04 | engl | negation | ✅ | — |
| H05 | engl | multi-city, distractor | ✅ | — |
| H06 | code | — | ✅ | — |
| H07 | engl | — | ✅ | — |
| H08 | engl | — | ✅ | — |
| H09 | engl | spelled-budget | ✅ | — |
| H10 | code | — | ✅ | — |
| H11 | engl | multi-city, negation | ✅ | — |
| H12 | engl | — | ✅ | — |
| H13 | engl | bare-rupees | ✅ | — |
| F01 | engl | — | ✅ | — |
| F02 | engl | — | ✅ | — |
| F03 | code | — | ✅ | — |
| F04 | engl | negation, multi-city, distractor | ✅ | — |
| F05 | engl | — | ❌ | **CITY**: Pune → January |
| F06 | code | — | ✅ | — |
| F07 | engl | multi-city, distractor | ✅ | — |
| F08 | engl | — | ❌ | **FURNISHING**: — → Furnishing |
| F09 | engl | range-budget | ✅ | — |
| F10 | code | negation | ✅ | — |
| F11 | engl | — | ✅ | — |
| F12 | engl | — | ✅ | — |

### 2.1 Per-entity precision / recall / F1

| Entity | Support | TP | FP | FN | Precision | Recall | F1 | Field acc. |
|---|---|---|---|---|---|---|---|---|
| CITY | 50 | 49 | 1 | 1 | 0.980 | 0.980 | 0.980 | 98.0% |
| AREA | 50 | 50 | 0 | 0 | 1.000 | 1.000 | 1.000 | 100.0% |
| PROPERTY_TYPE | 35 | 34 | 2 | 1 | 0.944 | 0.971 | 0.958 | 96.0% |
| BHK | 37 | 37 | 0 | 0 | 1.000 | 1.000 | 1.000 | 100.0% |
| BUDGET | 50 | 47 | 0 | 3 | 1.000 | 0.940 | 0.969 | 94.0% |
| AMENITY | 91 | 91 | 0 | 0 | 1.000 | 1.000 | 1.000 | 100.0% |
| FURNISHING | 42 | 42 | 2 | 0 | 0.955 | 1.000 | 0.977 | 96.0% |
| **micro (all)** | 355 | 350 | 5 | 5 | 0.986 | 0.986 | **0.986** | 97.7% |

- Overall extraction accuracy (exact field matches): **97.7%** (342/350 field slots).
- Cases with all seven fields correct: **42/50**.
- Macro-F1 across the seven entities: **0.983**.

### 2.2 Extraction by subset

| Subset | Cases | Micro-F1 | Field acc. | Fully correct |
|---|---|---|---|---|
| english | 37 | 0.986 | 97.7% | 31/37 |
| code-mixed | 13 | 0.984 | 97.8% | 11/13 |
| clean | 37 | 0.983 | 97.3% | 30/37 |
| messy | 13 | 0.994 | 98.9% | 12/13 |
| bare-rupees | 5 | 0.972 | 94.3% | 3/5 |
| distractor | 4 | 1.000 | 100.0% | 4/4 |
| monthly-rent | 4 | 0.966 | 92.9% | 2/4 |
| multi-city | 6 | 1.000 | 100.0% | 6/6 |
| negation | 7 | 0.980 | 98.0% | 6/7 |
| range-budget | 3 | 0.971 | 95.2% | 2/3 |
| spelled-budget | 5 | 0.969 | 94.3% | 3/5 |

### 2.3 Every extraction miss, in full

**E02** (english, clean, spelled-budget) — `BUDGET`
> Happy to. We have been saving towards this for six years and the moment the sanction letter came through I told my husband, right, let us do it prope…
- expected: `25000000`
- predicted: `—`
- why the case is hard: 'two and a half crore' — spelled-out fractional amount.
- root cause: MuRIL *did* find the span `two and a half crore`. `BudgetParser` cannot read it: its number-word run breaks at 'half', which is in neither `_NUMBER_WORDS` nor `_NUMBER_SCALES`, so the run never reaches the unit. `_resolve_budget` then rejects the unparseable span and the `_BUDGET_SCAN` fallback fails for the same reason. Fractional spelled amounts are a gap in the parser, not in the NER model.

**E04** (english, messy, range-budget, spelled-budget) — `FURNISHING`
> So — okay, um, sorry, I am slightly over-caffeinated. Right. College Road, Nashik. Three BHK. We saw four other builders and yours is the only one wh…
- expected: `—`
- predicted: `Liveable`
- why the case is hard: Spelled-out range; 'eighty one' at the end is an age, not money.
- root cause: MuRIL tagged the adjective `liveable` as FURNISHING. There is no furnishing catalog to validate against, so `_canonical` title-cases whatever span it is handed (nlp/predict.py:540) and any model false positive becomes a furnishing value verbatim.

**E09** (code-mixed, clean, negation) — `PROPERTY_TYPE`
> Bahut acchi feeling aa rahi hai is baar! Pimple Saudagar mein 2 BHK flat dekhna hai. Villa nahi chahiye, maintenance bahut ho jaata hai. 65 lakh budg…
- expected: `Apartment`
- predicted: `Villa`
- why the case is hard: Negation is in Hindi ('villa nahi chahiye'), not English.
- root cause: `_NEGATION_CUES` in nlp/predict.py is English-only, so 'Villa nahi chahiye' is not recognised as negation. The villa span stays active and wins as the last PROPERTY_TYPE mention, overriding the flat the caller actually asked for. English negation works — see H11 and F04.

**N10** (code-mixed, clean, bare-rupees, monthly-rent) — `BUDGET`
> Ravet mein 2 BHK rent pe. 22000 per month. Visitor parking aur lift chahiye. Semi furnished.
- expected: `22000`
- predicted: `—`
- why the case is hard: Monthly rent with no unit word at all ('22000 per month').
- root cause: `_BUDGET_SCAN` is anchored on a currency unit (crore/lakh/rupees/rs) so that bare numbers cannot be mistaken for money. '22000 per month' has no unit token, and no model BUDGET span survived, so nothing is reported.

**N11** (english, clean, bare-rupees, monthly-rent) — `BUDGET`
> I need PG accommodation in Untwadi for my daughter. Around 12000 a month. Housekeeping, Wi-Fi and 24x7 security are the priorities. Fully furnished.
- expected: `12000`
- predicted: `—`
- root cause: Same cause as N10: '12000 a month' carries no currency unit for `_BUDGET_SCAN` to anchor on. Monthly rent stated as a bare figure is currently unreachable.

**H02** (english, clean) — `PROPERTY_TYPE`
> We have been going back and forth on this for months. Three BHK on Sharanpur Road, Nashik, in theory. 90 lakhs is the most we could manage and even t…
- expected: `—`
- predicted: `Independent House`
- root cause: No PROPERTY_TYPE span survives — correctly, the caller never names one. `extract()` then falls back to the rule extractor over the whole transcript (nlp/predict.py:632), which regex-matches 'house' inside the amenity 'club house' and maps it through property_type_synonyms.csv. The fallback exists to recover catalog-collision terms; here it invents a property type out of an amenity.

**F05** (english, clean) — `CITY`
> What I need is for someone to action this. Office space in Yerawada, 2.6 crore, and I have been chasing since January. A conference room and power ba…
- expected: `Pune`
- predicted: `January`
- root cause: MuRIL tagged `January` as CITY. It is the only CITY candidate, so `_choose_city` returns it through the single-candidate branch (nlp/predict.py:492) without ever applying the known-city or distractor filter those lines exist for. Because city then reads non-null, the area-to-city lookup that would have given Yerawada → Pune never runs.

**F08** (english, clean) — `FURNISHING`
> Residential plot in Adgaon. 55 lakhs. A boundary wall and water supply, that is it. Furnishing does not apply, it is a plot. I have now repeated this…
- expected: `—`
- predicted: `Furnishing`
- why the case is hard: 'boundary wall' is not in amenities.csv, so it cannot be represented and is not counted against either system.
- root cause: Same passthrough as E04: MuRIL tagged the literal word `Furnishing` and `_canonical` title-cased it into a value. The caller said furnishing does not apply.

## 3. Sentiment

### 3.1 Per-case results

| # | Reg | Transcript (customer turns) | Expected | Predicted | Conf. | |
|---|---|---|---|---|---|---|
| E01 | engl | Afternoon! My wife and I drove past your Baner site on Sunday and we talked about nothing else the whole way … | Enthusiastic | Enthusiastic | 1.00 | ✅ |
| E02 | engl | Happy to. We have been saving towards this for six years and the moment the sanction letter came through I to… | Enthusiastic | Enthusiastic | 0.99 | ✅ |
| E03 | code | Arre bahut din se soch rahe the, finally decide kiya! Wakad mein 2 BHK flat chahiye, budget 85 lakh tak comfo… | Enthusiastic | Enthusiastic | 0.99 | ✅ |
| E04 | engl | So — okay, um, sorry, I am slightly over-caffeinated. Right. College Road, Nashik. Three BHK. We saw four oth… | Enthusiastic | Enthusiastic | 0.99 | ✅ |
| E05 | engl | This is a great call to be making. My practice has outgrown its current floor and we are finally in a positio… | Enthusiastic | Enthusiastic | 0.99 | ✅ |
| E06 | code | Namaste! Dekhiye, main pichle do saal se dekh raha tha but kabhi feel nahi aaya. Aapke Panchavati wale projec… | Enthusiastic | Enthusiastic | 0.68 | ✅ |
| E07 | engl | Morning. I will be honest, I have been waiting three years for something in Koregaon Park to come up, and I s… | Enthusiastic | Enthusiastic | 1.00 | ✅ |
| E08 | engl | I am in Bengaluru right now, but the transfer came through last week and I could not be happier about it. We … | Enthusiastic | Enthusiastic | 1.00 | ✅ |
| E09 | code | Bahut acchi feeling aa rahi hai is baar! Pimple Saudagar mein 2 BHK flat dekhna hai. Villa nahi chahiye, main… | Enthusiastic | Enthusiastic | 0.99 | ✅ |
| E10 | engl | Good news, hopefully. My first proper salary hike came through and I want to put it into property instead of … | Enthusiastic | Enthusiastic | 0.99 | ✅ |
| E11 | engl | Right so — my brother and I, we have — sorry, this is the third time I have tried to call today and I keep ge… | Enthusiastic | Enthusiastic | 1.00 | ✅ |
| E12 | engl | We hesitated on a place a decade ago and regretted it ever since, so I am thrilled to be doing this properly … | Enthusiastic | Enthusiastic | 0.98 | ✅ |
| E13 | code | Ekdum sahi timing pe call kiya maine! Kothrud mein 1 BHK flat rent pe chahiye. Monthly 28000 rupees tak thik … | Enthusiastic | Frustrated | 0.94 | ❌ |
| N01 | engl | I am collecting information at this stage. Two BHK in Aundh, Pune. Budget somewhere near 80 lakhs. Lift and p… | Neutral | Neutral | 1.00 | ✅ |
| N02 | engl | We need a warehouse in Satpur, Nashik. Roughly 1.8 crore. Power backup and 24x7 security are required for ins… | Neutral | Neutral | 1.00 | ✅ |
| N03 | code | Hadapsar mein 3 BHK flat dekhna hai. Budget 95 lakh. Covered parking aur gym hona chahiye. Semi furnished. De… | Neutral | Neutral | 0.99 | ✅ |
| N04 | engl | Residential plot in Nashik Road. Between 35 and 50 lakhs. No particular amenity requirements. Furnishing is n… | Neutral | Neutral | 1.00 | ✅ |
| N05 | engl | A serviced apartment in Viman Nagar for a twelve month corporate lease. The company ceiling is 75000 rupees a… | Neutral | Neutral | 0.79 | ✅ |
| N06 | code | Haan toh... Ambad mein godown chahiye. Size thik thaak, 60 lakh ke aas paas. Security aur power backup. Unfur… | Neutral | Enthusiastic | 0.99 | ❌ |
| N07 | engl | We looked at Nashik earlier in the year, but we are searching in Pune now. Bavdhan specifically. 3 BHK, 1.15 … | Neutral | Neutral | 1.00 | ✅ |
| N08 | engl | Duplex in Panchavati. Four bedrooms. 1.75 crore is the figure I have been given. Rainwater harvesting and sol… | Neutral | Frustrated | 0.94 | ❌ |
| N09 | engl | An apartment in Magarpatta. Not a studio apartment, a proper 2 BHK. Budget is 88 lakhs. Gym and swimming pool… | Neutral | Neutral | 0.99 | ✅ |
| N10 | code | Ravet mein 2 BHK rent pe. 22000 per month. Visitor parking aur lift chahiye. Semi furnished. | Neutral | Neutral | 0.87 | ✅ |
| N11 | engl | I need PG accommodation in Untwadi for my daughter. Around 12000 a month. Housekeeping, Wi-Fi and 24x7 securi… | Neutral | Neutral | 0.99 | ✅ |
| N12 | engl | Um. Farm house. Talegaon. Or, no, sorry — farm land, actually. My father wants farm land. Two crore. Water su… | Neutral | Hesitant | 0.86 | ❌ |
| H01 | engl | I am not sure yet, to be honest. Maybe a 2 BHK in Kondhwa. My husband thinks we should wait another year. Bud… | Hesitant | Hesitant | 1.00 | ✅ |
| H02 | engl | We have been going back and forth on this for months. Three BHK on Sharanpur Road, Nashik, in theory. 90 lakh… | Hesitant | Hesitant | 0.96 | ✅ |
| H03 | code | Dekhiye, abhi decide nahi kiya hai properly. Chinchwad mein 2 BHK flat ka idea hai. 70 lakh ke aas paas. Gym … | Hesitant | Hesitant | 0.99 | ✅ |
| H04 | engl | I — hmm. Okay. Not a villa, we cannot maintain a villa. Something smaller. An apartment in Canada Corner mayb… | Hesitant | Hesitant | 0.99 | ✅ |
| H05 | engl | That is part of the problem, I keep changing my mind. I was considering Mumbai, then Pune. Right now I am lea… | Hesitant | Hesitant | 0.99 | ✅ |
| H06 | code | Umm... pata nahi. Jail Road pe 2 BHK dekha tha, par... 55 lakh se zyada nahi ja sakte. Lift hona chahiye bas.… | Hesitant | Enthusiastic | 0.97 | ❌ |
| H07 | engl | A row house in Balewadi, possibly. My loan pre-approval is only 1.2 crore and the ones I have seen start high… | Hesitant | Neutral | 0.51 | ❌ |
| H08 | engl | A commercial shop in Deolali. I am honestly unsure whether retail is the right bet this year. 45 lakhs, give … | Hesitant | Hesitant | 0.94 | ✅ |
| H09 | engl | I keep starting this conversation and then stopping. Warje. Two BHK. Around seventy five lakhs, I think. Powe… | Hesitant | Frustrated | 0.53 | ❌ |
| H10 | code | Confusion hai thoda. Tathawade mein 3 BHK apartment, 85 lakh tak. Swimming pool aur children's play area dekh… | Hesitant | Frustrated | 0.98 | ❌ |
| H11 | engl | We are currently in Pune, but the job may move us to Nashik. Makhmalabad, if it happens. Not a flat — an inde… | Hesitant | Hesitant | 0.75 | ✅ |
| H12 | engl | Possibly a studio apartment in Fatima Nagar. I say possibly because I have said yes to two places already and… | Hesitant | Hesitant | 0.75 | ✅ |
| H13 | engl | So — I think — look, the honest answer is I am not committed. Ozar. 2 BHK. I have 4200000 rupees approved and… | Hesitant | Neutral | 0.88 | ❌ |
| F01 | engl | You can start by explaining why nobody has called me back. This is my fourth attempt. 3 BHK in Kalyani Nagar,… | Frustrated | Frustrated | 0.99 | ✅ |
| F02 | engl | The requirement has not changed since March. Pathardi Phata, Nashik. Four BHK villa. 2.2 crore. Swimming pool… | Frustrated | Frustrated | 0.99 | ✅ |
| F03 | code | Kitni baar bataana padega? Wanowrie mein 2 BHK flat, 72 lakh. Lift aur power backup. Semi furnished. Teen baa… | Frustrated | Frustrated | 0.97 | ✅ |
| F04 | engl | Right, so — no. No, listen. I do not want a villa, I have said that every single time. An apartment. Mumbai N… | Frustrated | Frustrated | 0.99 | ✅ |
| F05 | engl | What I need is for someone to action this. Office space in Yerawada, 2.6 crore, and I have been chasing since… | Frustrated | Hesitant | 0.59 | ❌ |
| F06 | code | Dekhiye, mera time waste ho raha hai. Rane Nagar mein 3 BHK chahiye, 78 lakh. Gym, security, parking. Unfurni… | Frustrated | Frustrated | 0.99 | ✅ |
| F07 | engl | Pune. I already told your Hyderabad office that, and they transferred me twice. Pashan. 3 BHK apartment. 1.45… | Frustrated | Frustrated | 1.00 | ✅ |
| F08 | engl | Residential plot in Adgaon. 55 lakhs. A boundary wall and water supply, that is it. Furnishing does not apply… | Frustrated | Frustrated | 0.99 | ✅ |
| F09 | engl | Moshi. 2 BHK. Between 55 and 68 lakhs. Lift and visitor parking. Semi furnished. And before you ask, yes I fi… | Frustrated | Frustrated | 1.00 | ✅ |
| F10 | code | Bahut ho gaya ab. Punawale mein 3 BHK flat, 92 lakh. Ground floor nahi chahiye. Covered parking aur gym. Semi… | Frustrated | Frustrated | 1.00 | ✅ |
| F11 | engl | Finally. Right — Govind Nagar, Nashik. A — sorry, hold on — a 3 BHK. 68 lakhs. Lift, power backup, children's… | Frustrated | Frustrated | 1.00 | ✅ |
| F12 | engl | A showroom in Lohegaon. 3.4 crore. Visitor parking and CCTV surveillance. Unfurnished. I want a callback from… | Frustrated | Frustrated | 0.98 | ✅ |

### 3.2 Headline

- Accuracy: **80.0%** (40/50)
- Macro-F1: **0.795**
- Majority-class baseline on this set: 26.0%

### 3.3 Per-class precision / recall / F1

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Enthusiastic | 13 | 12 | 2 | 1 | 0.857 | 0.923 | 0.889 |
| Neutral | 12 | 9 | 2 | 3 | 0.818 | 0.750 | 0.783 |
| Hesitant | 13 | 8 | 2 | 5 | 0.800 | 0.615 | 0.696 |
| Frustrated | 12 | 11 | 4 | 1 | 0.733 | 0.917 | 0.815 |

### 3.4 Confusion matrix (rows = truth, columns = predicted)

| truth \ pred | Enthusiastic | Neutral | Hesitant | Frustrated |
|---|---|---|---|---|
| **Enthusiastic** | **12** | 0 | 0 | 1 |
| **Neutral** | 1 | **9** | 1 | 1 |
| **Hesitant** | 1 | 2 | **8** | 2 |
| **Frustrated** | 0 | 0 | 1 | **11** |

### 3.5 Dangerous vs safe confusions

- **Enthusiastic ↔ Frustrated (dangerous — inverts callback priority): 1**
  - `E13` Enthusiastic → Frustrated (conf 0.94)
- **Hesitant ↔ Frustrated (safe — lead stays in roughly the right queue): 3**
  - `H09` Hesitant → Frustrated (conf 0.53)
  - `H10` Hesitant → Frustrated (conf 0.98)
  - `F05` Frustrated → Hesitant (conf 0.59)

### 3.6 English vs code-mixed

| Subset | Cases | Accuracy | Macro-F1 |
|---|---|---|---|
| english | 37 | 83.8% | 0.839 |
| code-mixed | 13 | 69.2% | 0.679 |
| clean | 37 | 89.2% | 0.891 |
| messy | 13 | 53.8% | 0.437 |

### 3.7 Every sentiment miss, in full

**E13** (code-mixed, messy) — expected **Enthusiastic**, predicted **Frustrated** at 0.94
> Ekdum sahi timing pe call kiya maine! Kothrud mein 1 BHK flat rent pe chahiye. Monthly 28000 rupees tak thik hai. Lift aur security zaroori. Fully fu…
- full distribution: Enthusiastic 0.06, Neutral 0.00, Hesitant 0.00, Frustrated 0.94

**N06** (code-mixed, messy) — expected **Neutral**, predicted **Enthusiastic** at 0.99
> Haan toh... Ambad mein godown chahiye. Size thik thaak, 60 lakh ke aas paas. Security aur power backup. Unfurnished.
- full distribution: Enthusiastic 0.99, Neutral 0.01, Hesitant 0.00, Frustrated 0.00

**N08** (english, clean) — expected **Neutral**, predicted **Frustrated** at 0.94
> Duplex in Panchavati. Four bedrooms. 1.75 crore is the figure I have been given. Rainwater harvesting and solar panels if available. Unfurnished.
- full distribution: Enthusiastic 0.02, Neutral 0.04, Hesitant 0.00, Frustrated 0.94

**N12** (english, messy) — expected **Neutral**, predicted **Hesitant** at 0.86
> Um. Farm house. Talegaon. Or, no, sorry — farm land, actually. My father wants farm land. Two crore. Water supply is the only thing that matters out …
- full distribution: Enthusiastic 0.00, Neutral 0.11, Hesitant 0.86, Frustrated 0.02

**H06** (code-mixed, messy) — expected **Hesitant**, predicted **Enthusiastic** at 0.97
> Umm... pata nahi. Jail Road pe 2 BHK dekha tha, par... 55 lakh se zyada nahi ja sakte. Lift hona chahiye bas. Furnishing ka abhi soch nahi hai. Ghar …
- full distribution: Enthusiastic 0.97, Neutral 0.00, Hesitant 0.01, Frustrated 0.01

**H07** (english, clean) — expected **Hesitant**, predicted **Neutral** at 0.51
> A row house in Balewadi, possibly. My loan pre-approval is only 1.2 crore and the ones I have seen start higher, so I am hesitant. Three bedrooms. A …
- full distribution: Enthusiastic 0.01, Neutral 0.51, Hesitant 0.47, Frustrated 0.01

**H09** (english, messy) — expected **Hesitant**, predicted **Frustrated** at 0.53
> I keep starting this conversation and then stopping. Warje. Two BHK. Around seventy five lakhs, I think. Power backup, and a lift because of my knees…
- full distribution: Enthusiastic 0.00, Neutral 0.00, Hesitant 0.47, Frustrated 0.53

**H10** (code-mixed, clean) — expected **Hesitant**, predicted **Frustrated** at 0.98
> Confusion hai thoda. Tathawade mein 3 BHK apartment, 85 lakh tak. Swimming pool aur children's play area dekh rahe the but zaroori nahi. Unfurnished.…
- full distribution: Enthusiastic 0.00, Neutral 0.01, Hesitant 0.01, Frustrated 0.98

**H13** (english, messy) — expected **Hesitant**, predicted **Neutral** at 0.88
> So — I think — look, the honest answer is I am not committed. Ozar. 2 BHK. I have 4200000 rupees approved and nothing beyond that. Water supply and p…
- full distribution: Enthusiastic 0.00, Neutral 0.88, Hesitant 0.10, Frustrated 0.01

**F05** (english, clean) — expected **Frustrated**, predicted **Hesitant** at 0.59
> What I need is for someone to action this. Office space in Yerawada, 2.6 crore, and I have been chasing since January. A conference room and power ba…
- full distribution: Enthusiastic 0.00, Neutral 0.00, Hesitant 0.59, Frustrated 0.41

## 4. Intent

> **Read this first.** These 50 calls were written for extraction and sentiment, not intent, so most callers never say what they want to do: 7 labels are *stated*, 38 *inferred* and 5 *ambiguous* (split in §4.5). The set is Buy 41, Rent 4, Inquiry 2, Schedule Visit 0, Investment 2, Request Callback 1 — a model that always answered **Buy** would score 82.0%. A class with 1–2 cases moves its per-class numbers by 50 points or more on a single call.

### 4.1 Per-case results

| # | Basis | Transcript (customer turns) | Expected | Predicted | Conf. | |
|---|---|---|---|---|---|---|
| E01 | inferred | Afternoon! My wife and I drove past your Baner site on Sunday and we talked about nothing else the … | Buy | Investment | 0.49 | ❌ |
| E02 | inferred | Happy to. We have been saving towards this for six years and the moment the sanction letter came th… | Buy | Investment | 0.73 | ❌ |
| E03 | inferred | Arre bahut din se soch rahe the, finally decide kiya! Wakad mein 2 BHK flat chahiye, budget 85 lakh… | Buy | Buy | 0.51 | ✅ |
| E04 | inferred | So — okay, um, sorry, I am slightly over-caffeinated. Right. College Road, Nashik. Three BHK. We sa… | Buy | Buy | 0.80 | ✅ |
| E05 | stated | This is a great call to be making. My practice has outgrown its current floor and we are finally in… | Buy | Investment | 0.77 | ❌ |
| E06 | inferred | Namaste! Dekhiye, main pichle do saal se dekh raha tha but kabhi feel nahi aaya. Aapke Panchavati w… | Buy | Inquiry | 0.60 | ❌ |
| E07 | inferred | Morning. I will be honest, I have been waiting three years for something in Koregaon Park to come u… | Buy | Rent | 0.48 | ❌ |
| E08 | inferred | I am in Bengaluru right now, but the transfer came through last week and I could not be happier abo… | Buy | Investment | 0.62 | ❌ |
| E09 | ambiguous | Bahut acchi feeling aa rahi hai is baar! Pimple Saudagar mein 2 BHK flat dekhna hai. Villa nahi cha… | Buy | Buy | 0.96 | ✅ |
| E10 | ambiguous | Good news, hopefully. My first proper salary hike came through and I want to put it into property i… | Investment | Investment | 0.64 | ✅ |
| E11 | inferred | Right so — my brother and I, we have — sorry, this is the third time I have tried to call today and… | Buy | Rent | 0.39 | ❌ |
| E12 | inferred | We hesitated on a place a decade ago and regretted it ever since, so I am thrilled to be doing this… | Buy | Rent | 0.92 | ❌ |
| E13 | stated | Ekdum sahi timing pe call kiya maine! Kothrud mein 1 BHK flat rent pe chahiye. Monthly 28000 rupees… | Rent | Request Callback | 0.66 | ❌ |
| N01 | stated | I am collecting information at this stage. Two BHK in Aundh, Pune. Budget somewhere near 80 lakhs. … | Inquiry | Inquiry | 0.69 | ✅ |
| N02 | inferred | We need a warehouse in Satpur, Nashik. Roughly 1.8 crore. Power backup and 24x7 security are requir… | Buy | Buy | 0.94 | ✅ |
| N03 | ambiguous | Hadapsar mein 3 BHK flat dekhna hai. Budget 95 lakh. Covered parking aur gym hona chahiye. Semi fur… | Inquiry | Buy | 0.68 | ❌ |
| N04 | inferred | Residential plot in Nashik Road. Between 35 and 50 lakhs. No particular amenity requirements. Furni… | Buy | Buy | 0.78 | ✅ |
| N05 | stated | A serviced apartment in Viman Nagar for a twelve month corporate lease. The company ceiling is 7500… | Rent | Rent | 0.99 | ✅ |
| N06 | inferred | Haan toh... Ambad mein godown chahiye. Size thik thaak, 60 lakh ke aas paas. Security aur power bac… | Buy | Buy | 0.95 | ✅ |
| N07 | inferred | We looked at Nashik earlier in the year, but we are searching in Pune now. Bavdhan specifically. 3 … | Buy | Investment | 0.71 | ❌ |
| N08 | inferred | Duplex in Panchavati. Four bedrooms. 1.75 crore is the figure I have been given. Rainwater harvesti… | Buy | Investment | 0.63 | ❌ |
| N09 | inferred | An apartment in Magarpatta. Not a studio apartment, a proper 2 BHK. Budget is 88 lakhs. Gym and swi… | Buy | Buy | 0.93 | ✅ |
| N10 | stated | Ravet mein 2 BHK rent pe. 22000 per month. Visitor parking aur lift chahiye. Semi furnished. | Rent | Rent | 0.93 | ✅ |
| N11 | stated | I need PG accommodation in Untwadi for my daughter. Around 12000 a month. Housekeeping, Wi-Fi and 2… | Rent | Rent | 0.99 | ✅ |
| N12 | inferred | Um. Farm house. Talegaon. Or, no, sorry — farm land, actually. My father wants farm land. Two crore… | Buy | Investment | 0.67 | ❌ |
| H01 | inferred | I am not sure yet, to be honest. Maybe a 2 BHK in Kondhwa. My husband thinks we should wait another… | Buy | Investment | 0.56 | ❌ |
| H02 | inferred | We have been going back and forth on this for months. Three BHK on Sharanpur Road, Nashik, in theor… | Buy | Buy | 0.77 | ✅ |
| H03 | inferred | Dekhiye, abhi decide nahi kiya hai properly. Chinchwad mein 2 BHK flat ka idea hai. 70 lakh ke aas … | Buy | Buy | 0.74 | ✅ |
| H04 | inferred | I — hmm. Okay. Not a villa, we cannot maintain a villa. Something smaller. An apartment in Canada C… | Buy | Investment | 0.98 | ❌ |
| H05 | inferred | That is part of the problem, I keep changing my mind. I was considering Mumbai, then Pune. Right no… | Buy | Buy | 0.58 | ✅ |
| H06 | inferred | Umm... pata nahi. Jail Road pe 2 BHK dekha tha, par... 55 lakh se zyada nahi ja sakte. Lift hona ch… | Buy | Rent | 0.98 | ❌ |
| H07 | inferred | A row house in Balewadi, possibly. My loan pre-approval is only 1.2 crore and the ones I have seen … | Buy | Investment | 0.79 | ❌ |
| H08 | ambiguous | A commercial shop in Deolali. I am honestly unsure whether retail is the right bet this year. 45 la… | Investment | Inquiry | 0.50 | ❌ |
| H09 | inferred | I keep starting this conversation and then stopping. Warje. Two BHK. Around seventy five lakhs, I t… | Buy | Investment | 0.75 | ❌ |
| H10 | inferred | Confusion hai thoda. Tathawade mein 3 BHK apartment, 85 lakh tak. Swimming pool aur children's play… | Buy | Buy | 0.98 | ✅ |
| H11 | inferred | We are currently in Pune, but the job may move us to Nashik. Makhmalabad, if it happens. Not a flat… | Buy | Investment | 0.98 | ❌ |
| H12 | inferred | Possibly a studio apartment in Fatima Nagar. I say possibly because I have said yes to two places a… | Buy | Buy | 0.97 | ✅ |
| H13 | inferred | So — I think — look, the honest answer is I am not committed. Ozar. 2 BHK. I have 4200000 rupees ap… | Buy | Buy | 0.95 | ✅ |
| F01 | ambiguous | You can start by explaining why nobody has called me back. This is my fourth attempt. 3 BHK in Kaly… | Buy | Buy | 0.69 | ✅ |
| F02 | inferred | The requirement has not changed since March. Pathardi Phata, Nashik. Four BHK villa. 2.2 crore. Swi… | Buy | Buy | 0.80 | ✅ |
| F03 | inferred | Kitni baar bataana padega? Wanowrie mein 2 BHK flat, 72 lakh. Lift aur power backup. Semi furnished… | Buy | Inquiry | 0.48 | ❌ |
| F04 | inferred | Right, so — no. No, listen. I do not want a villa, I have said that every single time. An apartment… | Buy | Investment | 0.97 | ❌ |
| F05 | inferred | What I need is for someone to action this. Office space in Yerawada, 2.6 crore, and I have been cha… | Buy | Buy | 0.70 | ✅ |
| F06 | inferred | Dekhiye, mera time waste ho raha hai. Rane Nagar mein 3 BHK chahiye, 78 lakh. Gym, security, parkin… | Buy | Schedule Visit | 0.97 | ❌ |
| F07 | inferred | Pune. I already told your Hyderabad office that, and they transferred me twice. Pashan. 3 BHK apart… | Buy | Buy | 0.86 | ✅ |
| F08 | inferred | Residential plot in Adgaon. 55 lakhs. A boundary wall and water supply, that is it. Furnishing does… | Buy | Buy | 0.70 | ✅ |
| F09 | inferred | Moshi. 2 BHK. Between 55 and 68 lakhs. Lift and visitor parking. Semi furnished. And before you ask… | Buy | Request Callback | 0.96 | ❌ |
| F10 | inferred | Bahut ho gaya ab. Punawale mein 3 BHK flat, 92 lakh. Ground floor nahi chahiye. Covered parking aur… | Buy | Schedule Visit | 0.47 | ❌ |
| F11 | inferred | Finally. Right — Govind Nagar, Nashik. A — sorry, hold on — a 3 BHK. 68 lakhs. Lift, power backup, … | Buy | Buy | 0.97 | ✅ |
| F12 | stated | A showroom in Lohegaon. 3.4 crore. Visitor parking and CCTV surveillance. Unfurnished. I want a cal… | Request Callback | Request Callback | 0.91 | ✅ |

### 4.2 Headline

- Accuracy: **50.0%** (25/50)
- Majority-class baseline (always Buy): 82.0%
- Macro precision / recall / F1: 0.339 / 0.536 / **0.354** (averaged over: Buy, Rent, Inquiry, Schedule Visit, Investment, Request Callback)

### 4.3 Per-class precision / recall / F1

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Buy | 41 | 19 | 1 | 22 | 0.950 | 0.463 | 0.623 |
| Rent | 4 | 3 | 4 | 1 | 0.429 | 0.750 | 0.545 |
| Inquiry | 2 | 1 | 3 | 1 | 0.250 | 0.500 | 0.333 |
| Schedule Visit | 0 | 0 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| Investment | 2 | 1 | 13 | 1 | 0.071 | 0.500 | 0.125 |
| Request Callback | 1 | 1 | 2 | 0 | 0.333 | 1.000 | 0.500 |

### 4.4 Confusion matrix (rows = truth, columns = predicted)

| truth \ pred | Buy | Rent | Inquiry | Schedule Visit | Investment | Request Callback |
|---|---|---|---|---|---|---|
| **Buy** | **19** | 4 | 2 | 2 | 13 | 1 |
| **Rent** | 0 | **3** | 0 | 0 | 0 | 1 |
| **Inquiry** | 1 | 0 | **1** | 0 | 0 | 0 |
| **Schedule Visit** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Investment** | 0 | 0 | 1 | 0 | **1** | 0 |
| **Request Callback** | 0 | 0 | 0 | 0 | 0 | **1** |

### 4.5 By how the label was reached, and by subset

| Subset | Cases | Accuracy | Macro-F1 |
|---|---|---|---|
| stated | 7 | 71.4% | 0.631 |
| inferred | 38 | 44.7% | 0.618 |
| ambiguous | 5 | 60.0% | 0.489 |
| english | 37 | 51.4% | 0.530 |
| code-mixed | 13 | 46.2% | 0.375 |
| clean | 37 | 56.8% | 0.530 |
| messy | 13 | 30.8% | 0.250 |

### 4.6 Every intent miss, in full

**E01** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.49
> Afternoon! My wife and I drove past your Baner site on Sunday and we talked about nothing else the whole way back. Three bedrooms is what we are afte…
- full distribution: Buy 0.30, Rent 0.01, Inquiry 0.15, Schedule Visit 0.03, Investment 0.49, Request Callback 0.01

**E02** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.73
> Happy to. We have been saving towards this for six years and the moment the sanction letter came through I told my husband, right, let us do it prope…
- full distribution: Buy 0.06, Rent 0.16, Inquiry 0.01, Schedule Visit 0.04, Investment 0.73, Request Callback 0.01

**E05** (stated label, english) — expected **Buy**, predicted **Investment** at 0.77
> This is a great call to be making. My practice has outgrown its current floor and we are finally in a position to own rather than lease. Office space…
- full distribution: Buy 0.03, Rent 0.18, Inquiry 0.01, Schedule Visit 0.01, Investment 0.77, Request Callback 0.01

**E06** (inferred label, code-mixed) — expected **Buy**, predicted **Inquiry** at 0.60
> Namaste! Dekhiye, main pichle do saal se dekh raha tha but kabhi feel nahi aaya. Aapke Panchavati wale project ka layout dekha na, ekdum dil khush ho…
- full distribution: Buy 0.01, Rent 0.13, Inquiry 0.60, Schedule Visit 0.07, Investment 0.17, Request Callback 0.01

**E07** (inferred label, english) — expected **Buy**, predicted **Rent** at 0.48
> Morning. I will be honest, I have been waiting three years for something in Koregaon Park to come up, and I saw your listing at six this morning and …
- full distribution: Buy 0.01, Rent 0.48, Inquiry 0.09, Schedule Visit 0.31, Investment 0.09, Request Callback 0.03

**E08** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.62
> I am in Bengaluru right now, but the transfer came through last week and I could not be happier about it. We are relocating to Hinjewadi, Pune. 3 BHK…
- full distribution: Buy 0.31, Rent 0.01, Inquiry 0.03, Schedule Visit 0.01, Investment 0.62, Request Callback 0.01

**E11** (inferred label, english) — expected **Buy**, predicted **Rent** at 0.39
> Right so — my brother and I, we have — sorry, this is the third time I have tried to call today and I keep getting cut off, but no no, I am delighted…
- full distribution: Buy 0.03, Rent 0.39, Inquiry 0.04, Schedule Visit 0.20, Investment 0.09, Request Callback 0.24

**E12** (inferred label, english) — expected **Buy**, predicted **Rent** at 0.92
> We hesitated on a place a decade ago and regretted it ever since, so I am thrilled to be doing this properly this time. Trimbak Road, Nashik. 3 BHK a…
- full distribution: Buy 0.00, Rent 0.92, Inquiry 0.01, Schedule Visit 0.02, Investment 0.05, Request Callback 0.00

**E13** (stated label, code-mixed) — expected **Rent**, predicted **Request Callback** at 0.66
> Ekdum sahi timing pe call kiya maine! Kothrud mein 1 BHK flat rent pe chahiye. Monthly 28000 rupees tak thik hai. Lift aur security zaroori. Fully fu…
- full distribution: Buy 0.07, Rent 0.04, Inquiry 0.02, Schedule Visit 0.07, Investment 0.14, Request Callback 0.66

**N03** (ambiguous label, code-mixed) — expected **Inquiry**, predicted **Buy** at 0.68
> Hadapsar mein 3 BHK flat dekhna hai. Budget 95 lakh. Covered parking aur gym hona chahiye. Semi furnished. Details bhej dijiye.
- full distribution: Buy 0.68, Rent 0.00, Inquiry 0.13, Schedule Visit 0.02, Investment 0.16, Request Callback 0.01

**N07** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.71
> We looked at Nashik earlier in the year, but we are searching in Pune now. Bavdhan specifically. 3 BHK, 1.15 crore. A children's play area matters to…
- full distribution: Buy 0.01, Rent 0.02, Inquiry 0.17, Schedule Visit 0.08, Investment 0.71, Request Callback 0.01

**N08** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.63
> Duplex in Panchavati. Four bedrooms. 1.75 crore is the figure I have been given. Rainwater harvesting and solar panels if available. Unfurnished.
- full distribution: Buy 0.13, Rent 0.00, Inquiry 0.22, Schedule Visit 0.01, Investment 0.63, Request Callback 0.01

**N12** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.67
> Um. Farm house. Talegaon. Or, no, sorry — farm land, actually. My father wants farm land. Two crore. Water supply is the only thing that matters out …
- full distribution: Buy 0.26, Rent 0.00, Inquiry 0.05, Schedule Visit 0.01, Investment 0.67, Request Callback 0.01

**H01** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.56
> I am not sure yet, to be honest. Maybe a 2 BHK in Kondhwa. My husband thinks we should wait another year. Budget would be around 62 lakhs if we did g…
- full distribution: Buy 0.35, Rent 0.02, Inquiry 0.03, Schedule Visit 0.03, Investment 0.56, Request Callback 0.01

**H04** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.98
> I — hmm. Okay. Not a villa, we cannot maintain a villa. Something smaller. An apartment in Canada Corner maybe. Three bedrooms. 1.1 crore, though I k…
- full distribution: Buy 0.00, Rent 0.01, Inquiry 0.00, Schedule Visit 0.00, Investment 0.98, Request Callback 0.00

**H06** (inferred label, code-mixed) — expected **Buy**, predicted **Rent** at 0.98
> Umm... pata nahi. Jail Road pe 2 BHK dekha tha, par... 55 lakh se zyada nahi ja sakte. Lift hona chahiye bas. Furnishing ka abhi soch nahi hai. Ghar …
- full distribution: Buy 0.00, Rent 0.98, Inquiry 0.00, Schedule Visit 0.01, Investment 0.01, Request Callback 0.00

**H07** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.79
> A row house in Balewadi, possibly. My loan pre-approval is only 1.2 crore and the ones I have seen start higher, so I am hesitant. Three bedrooms. A …
- full distribution: Buy 0.05, Rent 0.00, Inquiry 0.14, Schedule Visit 0.01, Investment 0.79, Request Callback 0.01

**H08** (ambiguous label, english) — expected **Investment**, predicted **Inquiry** at 0.50
> A commercial shop in Deolali. I am honestly unsure whether retail is the right bet this year. 45 lakhs, give or take. CCTV surveillance would help. U…
- full distribution: Buy 0.09, Rent 0.00, Inquiry 0.50, Schedule Visit 0.01, Investment 0.38, Request Callback 0.01

**H09** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.75
> I keep starting this conversation and then stopping. Warje. Two BHK. Around seventy five lakhs, I think. Power backup, and a lift because of my knees…
- full distribution: Buy 0.07, Rent 0.02, Inquiry 0.02, Schedule Visit 0.13, Investment 0.75, Request Callback 0.02

**H11** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.98
> We are currently in Pune, but the job may move us to Nashik. Makhmalabad, if it happens. Not a flat — an independent house. Four bedrooms. 1.3 crore …
- full distribution: Buy 0.01, Rent 0.01, Inquiry 0.01, Schedule Visit 0.00, Investment 0.98, Request Callback 0.00

**F03** (inferred label, code-mixed) — expected **Buy**, predicted **Inquiry** at 0.48
> Kitni baar bataana padega? Wanowrie mein 2 BHK flat, 72 lakh. Lift aur power backup. Semi furnished. Teen baar form bhara hai maine aapke website pe …
- full distribution: Buy 0.02, Rent 0.01, Inquiry 0.48, Schedule Visit 0.32, Investment 0.03, Request Callback 0.15

**F04** (inferred label, english) — expected **Buy**, predicted **Investment** at 0.97
> Right, so — no. No, listen. I do not want a villa, I have said that every single time. An apartment. Mumbai Naka. Three bedrooms. 85 lakhs. A lift th…
- full distribution: Buy 0.00, Rent 0.01, Inquiry 0.01, Schedule Visit 0.00, Investment 0.97, Request Callback 0.00

**F06** (inferred label, code-mixed) — expected **Buy**, predicted **Schedule Visit** at 0.97
> Dekhiye, mera time waste ho raha hai. Rane Nagar mein 3 BHK chahiye, 78 lakh. Gym, security, parking. Unfurnished. Pichle mahine aapke agent ne site …
- full distribution: Buy 0.00, Rent 0.00, Inquiry 0.01, Schedule Visit 0.97, Investment 0.00, Request Callback 0.01

**F09** (inferred label, english) — expected **Buy**, predicted **Request Callback** at 0.96
> Moshi. 2 BHK. Between 55 and 68 lakhs. Lift and visitor parking. Semi furnished. And before you ask, yes I filled the form, twice, and no, nobody ran…
- full distribution: Buy 0.02, Rent 0.00, Inquiry 0.01, Schedule Visit 0.01, Investment 0.00, Request Callback 0.96

**F10** (inferred label, code-mixed) — expected **Buy**, predicted **Schedule Visit** at 0.47
> Bahut ho gaya ab. Punawale mein 3 BHK flat, 92 lakh. Ground floor nahi chahiye. Covered parking aur gym. Semi furnished. Do hafte se follow up kar ra…
- full distribution: Buy 0.03, Rent 0.01, Inquiry 0.18, Schedule Visit 0.47, Investment 0.29, Request Callback 0.01


## 5. Final performance matrix — all modules

One run, same 50 transcripts. Extraction is scored per field slot (7 fields × 50 cases); sentiment and intent per transcript. Precision, recall and F1 are macro averages unless marked micro.

| Module | Unit | Accuracy | Precision | Recall | F1 | Baseline |
|---|---|---|---|---|---|---|
| NLP extraction (micro) | 350 field slots | 97.7% | 0.986 | 0.986 | **0.986** | — |
| NLP extraction (macro, 7 fields) | 350 field slots | 97.7% | 0.983 | 0.984 | 0.983 | — |
| Sentiment (4-class) | 50 transcripts | **80.0%** | 0.802 | 0.801 | **0.795** | majority 26.0% |
| Intent (6-class) | 50 transcripts | **50.0%** | 0.339 | 0.536 | **0.354** | majority 82.0% |

Per-field extraction scores are in §2.1, per-class sentiment in §3.3, per-class intent in §4.3.


## 6. Metrics summary

| | Extraction | Sentiment |
|---|---|---|
| Ran | MuRIL hybrid (nlp.predict.HybridExtractor) | DistilBERT 4-class sentiment head (sentiment_intent.predict) |
| Headline | micro-F1 **0.986** | accuracy **80.0%** |
| Secondary | macro-F1 0.983 | macro-F1 0.795 |
| Accuracy | 97.7% of 350 field slots | 40/50 transcripts |
| Clean sweeps | 42/50 cases, all 7 fields | — |
| English | micro-F1 0.986 | accuracy 83.8% |
| Code-mixed | micro-F1 0.984 | accuracy 69.2% |
| Dangerous confusions | — | 1 Enthusiastic↔Frustrated |
| Safe confusions | — | 3 Hesitant↔Frustrated |

### 6.1 What is wrong with extraction, grouped

1. **Model spans become furnishing values unchecked** — 2 misses (`E04`, `F08`). FURNISHING is the one model-owned field with no catalog behind it, so `_canonical` title-cases any span MuRIL emits. Constraining it to the four canonical values would turn both of these into correct nulls.
2. **Budget forms the parser cannot reach** — 3 misses (`E02`, `N10`, `N11`). Fractional spelled amounts ('two and a half crore') and bare monthly rent with no unit word ('22000 per month'). Both are parser gaps, not NER gaps — in E02 the model found the right span and the parser dropped it. Monthly rent matters for the Rent intent specifically.
3. **Negation is English-only** — 1 miss (`E09`). English negation is handled correctly (H11, F04, N09, N12 all pass). Hindi/Marathi cues — nahi, nako, mat — are absent from `_NEGATION_CUES`, so a code-mixed caller's ruled-out requirement is read as a wanted one.
4. **Unvalidated single-candidate city** — 1 miss (`F05`). `_choose_city` applies its known-city and distractor filters only when there are two or more candidates, so a lone model false positive is accepted and suppresses the area-to-city lookup that would have corrected it.
5. **Rule fallback invents a property type** — 1 miss (`H02`). The whole-transcript rule fallback matches substrings, so 'house' inside the amenity 'club house' produces a property type the caller never asked for.
