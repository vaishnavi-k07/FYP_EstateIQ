from nlp.extractor import NLPExtractor

extractor = NLPExtractor()

test_cases = [
    # Hinglish / mixed language — very common on real calls
    "haan so basically mujhe ek 2bhk chahiye Pune mein, budget around 55 lakh hai",

    # Exact rupee amount instead of lakh/crore
    "budget is exactly 8500000 rupees, looking for a 3 bhk in Nashik",

    # "under X" / "below X" framing instead of "around X"
    "I want something under 50 lakhs, 2 bhk apartment, doesn't matter which area",

    # "no budget constraint" — should NOT extract a false number
    "budget is not really a constraint, we just want the right 4 bhk villa in Pune",

    # Comparing two options in one sentence — ambiguous which is the actual requirement
    "I was considering either a 2bhk or a 3bhk, but I think we'll go with 3bhk finally, in Pune",

    # City name embedded inside an unrelated word (collision trap)
    "I run a punery... I mean a bakery business, need commercial space, not related to Pune though",

    # Area name that's also a common English word (collision trap — check your locations.csv)
    "looking for something in Wagholi, budget 40 lakhs, 2 bhk",

    # Amenity negation
    "I don't need a swimming pool or clubhouse, just basic parking and security is enough",

    # Furnishing typo / casual spelling
    "want it semifurnished, 2bhk, somewhere in Pune",

    # BHK expressed as "1RK" (common Indian real estate term, likely unsupported)
    "just need a 1RK for now, budget around 15 lakh, in Nashik",

    # Long rambling realistic call transcript
    "hello yes hi, so my name is Priya, um actually my husband and I have been "
    "looking for a while now, we currently live on rent in Pune, Kothrud side, "
    "and we want to finally buy something, probably 2 bhk, maybe 3 if budget allows, "
    "budget wise we are thinking 65 to 75 lakhs, we'd like a lift and parking at least, "
    "gym would be nice but not necessary, and ideally semi furnished so we don't have "
    "to spend more on interiors",

    # Explicit "no amenities needed"
    "no specific amenities needed, just a basic 1 bhk flat, budget 25 lakh, Nashik",

    # Two cities mentioned — one is current location, one is requirement
    "I'm currently based in Mumbai but I want to buy a property in Pune, 3bhk, budget 90 lakhs",

    # Office space, category check with synonym
    "we're a startup looking for an office space, around Nashik, nothing fancy",

    # Very short, single-word style answers concatenated (voice agent Q&A style)
    "Pune. 2bhk. 60 lakhs. Semi furnished. Parking and gym.",

    # Budget stated in words not digits at all
    "budget is around fifty lakhs, looking for a 2 bhk apartment",

    # Plot vs apartment vs villa — synonym stress test
    "not looking for a flat or a villa, just a plain residential plot, in Nashik",

    # Filler-heavy uncertain call, might just be a tire-kicker
    "umm I don't know, just checking what's available, maybe something small, not sure about budget honestly",

    # Amenity list with commas and "and" mixed, plus an amenity that's also part of a property term
    "want a gated society, garden, and power backup, looking for an apartment not a bungalow",
]

for i, text in enumerate(test_cases, 1):
    result = extractor.extract(text)
    print(f"\n--- Test {i} ---")
    print("Input:", text)
    print("Output:", result)