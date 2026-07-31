# NER Label Schema — EstateIQ NLP Extraction (Stage A)

Defines the BIO tag set used to annotate real-estate call transcripts for
fine-tuning MuRIL (`AutoModelForTokenClassification`), per CLAUDE.md Phase 3.

## 1. Entity types

| Entity | Meaning | Example surface forms |
|---|---|---|
| `CITY` | City the lead is asking about | "Pune", "Nashik" |
| `AREA` | Locality/neighbourhood within a city | "Baner", "Wakad", "Gangapur Road" |
| `PROPERTY_TYPE` | Kind of property | "apartment", "villa", "row house", "flat" |
| `BHK` | Bedroom count | "2 BHK", "3 bedroom", "two bhk" |
| `BUDGET` | Price/budget figure | "60 lakhs", "1.2 crore", "80L" |
| `AMENITY` | Requested amenity/feature | "swimming pool", "gym", "covered parking" |
| `FURNISHING` | Furnishing status | "fully furnished", "unfurnished" |

## 2. Tag set (BIO)

Each entity type gets a `B-` (beginning) and `I-` (inside) tag, plus the
shared `O` (outside) tag. 15 labels total:

```
O
B-CITY           I-CITY
B-AREA           I-AREA
B-PROPERTY_TYPE  I-PROPERTY_TYPE
B-BHK            I-BHK
B-BUDGET         I-BUDGET
B-AMENITY        I-AMENITY
B-FURNISHING     I-FURNISHING
```

Standard BIO rules apply:
- `B-X` marks the first token of an entity span of type `X`.
- `I-X` marks every subsequent token of the same span.
- `O` marks tokens outside any entity.
- Adjacent entities of the same type with no separator (rare) still get a
  fresh `B-X`, never two consecutive spans merged into one.

## 3. Negation is NOT a model label

Negated mentions (e.g. "not a villa", "I don't want a 3 BHK") are tagged
**identically** to affirmative mentions — the entity span still gets its
normal `B-X`/`I-X` tags. There is no `B-NEG-*` or polarity label in this
tag set.

**Why:** negation detection depends on cue words and their distance from
the entity span (see `_NEGATION_CUES` / `_is_negated` in the existing
`nlp/extractor.py`), which is a lexical/contextual pattern-matching problem,
not a span-identification problem. Folding it into the NER label set would
double the label space (14 entity tags → 28) for a signal that a simple
post-processing pass handles more reliably and with far less labeled data.

**Where it's actually resolved:** Stage E (post-processing & integration,
per CLAUDE.md Phase 3). The trained model emits entity spans exactly as
tagged here — including negated ones. A rule-based post-processing pass
then inspects the text window preceding each predicted span for negation
cues and either drops the span or flags it as `negated: true` before the
result is normalized against the CSV lookup tables.

Annotators: **do not** invent a negation label in Label Studio. Tag the
entity span as usual regardless of polarity.

## 4. Worked examples

### Example 1 — straightforward English

> "I am looking for a 2 BHK apartment in Baner, Pune within 60 lakhs."

| Token | Tag |
|---|---|
| I | O |
| am | O |
| looking | O |
| for | O |
| a | O |
| 2 | B-BHK |
| BHK | I-BHK |
| apartment | B-PROPERTY_TYPE |
| in | O |
| Baner | B-AREA |
| , | O |
| Pune | B-CITY |
| within | O |
| 60 | B-BUDGET |
| lakhs | I-BUDGET |
| . | O |

### Example 2 — negation (tagged normally, resolved later in Stage E)

> "Not a villa, I want an independent house with a swimming pool, fully furnished."

| Token | Tag |
|---|---|
| Not | O |
| a | O |
| villa | B-PROPERTY_TYPE |
| , | O |
| I | O |
| want | O |
| an | O |
| independent | B-PROPERTY_TYPE |
| house | I-PROPERTY_TYPE |
| with | O |
| a | O |
| swimming | B-AMENITY |
| pool | I-AMENITY |
| , | O |
| fully | B-FURNISHING |
| furnished | I-FURNISHING |
| . | O |

Note "villa" is still tagged `B-PROPERTY_TYPE` even though the customer is
ruling it out — the negation cue "Not" immediately before it is resolved by
the Stage E post-processing layer, not by the model.

### Example 3 — code-mixed Hindi/Marathi/English

> "Mujhe Wakad mein 3 BHK flat chahiye, budget around 80 lakh, gym aur parking zaroor chahiye."
>
> ("I need a 3 BHK flat in Wakad, budget around 80 lakh, definitely want gym and parking.")

| Token | Tag |
|---|---|
| Mujhe | O |
| Wakad | B-AREA |
| mein | O |
| 3 | B-BHK |
| BHK | I-BHK |
| flat | B-PROPERTY_TYPE |
| chahiye | O |
| , | O |
| budget | O |
| around | O |
| 80 | B-BUDGET |
| lakh | I-BUDGET |
| , | O |
| gym | B-AMENITY |
| aur | O |
| parking | B-AMENITY |
| zaroor | O |
| chahiye | O |
| . | O |

MuRIL's multilingual subword tokenizer will split some of these tokens
further (e.g. `chahiye` → multiple wordpieces); the BIO label is assigned
at the whole-word level as shown here and propagated to sub-tokens during
training (first-subtoken labeling, remaining subtokens `I-X` or masked —
finalized in Stage D).

## 5. Span boundary conventions

- Multi-word entities (`"independent house"`, `"swimming pool"`, `"fully
  furnished"`) are single spans — do not split them into separate
  single-token entities.
- Numeric unit pairs (`"2 BHK"`, `"60 lakhs"`) are single spans covering
  both the number and its unit word, not just the number.
- City/area names that are themselves multi-word (`"Viman Nagar"`, "Pimple
  Saudagar") are tagged as one `B-AREA I-AREA` span.
- Punctuation and connector words (`"in"`, `"within"`, `"mein"`, `"aur"`)
  are always `O`.
