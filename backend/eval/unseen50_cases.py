"""50 unseen transcripts with hand-written ground truth, for NLP, sentiment and intent.

Every transcript here was written for this evaluation. None of them come from
``data_generation`` — not from ``synthetic_transcripts.jsonl`` (the NER training
corpus) and not from any sentiment corpus (v3/pilot/v4/v5/v6). The point of the
set is generalisation, so ``run_unseen50.leakage_report`` proves that rather
than asserting it: it checks every generator phrase as a substring and measures
the longest contiguous word run each transcript shares with training text.

Ground truth is one careful human reading of each call, written *before* any
model was run on it. Where the correct answer needed a catalog convention
rather than a judgement — "flat" is an ``Apartment``, bare "parking" is
``Parking``, "security" is ``24x7 Security`` — the convention comes from
``dataset/property_type_synonyms.csv`` and ``dataset/amenity_synonyms.csv``, not
from what the extractor happens to do. Several cases are labelled with answers
the pipeline is expected to get wrong (spelled-out decimals, Hindi negation,
monthly rent with no unit word). Those stay labelled the way a human reads them,
because the alternative is grading the model against itself.

``budget`` ground truth is the numeric rupee value (``budget_value``), since
that is what Phase 4 lead scoring consumes. Ranges resolve to the upper bound —
a stated ceiling is what the caller means.

Fields per case:
    id          E/N/H/F + index, grouped by the sentiment label
    sentiment   expected 4-class label
    register    "english" | "code-mixed"  (drives the subset breakdown)
    style       "clean" | "messy"         (messy = disfluent, self-correcting)
    tags        hard-case markers: negation / multi-city / spelled-budget /
                range-budget / bare-rupees / monthly-rent / distractor
    transcript  Agent + Customer turns, as Vapi would deliver them
    expected    the seven extraction fields
    note        why a case is hard, where that is not obvious
    intent      expected 6-class intent (attached from ``INTENT_LABELS``)
    intent_basis  "stated" | "inferred" | "ambiguous" — how the label was reached

Intent was labelled after the rest of this file, and *not* designed in: these
calls were written to exercise extraction and tone, so most callers never say
what they want to do. Every intent label was still fixed from the transcript
text alone before the intent model was run on any case, and each one records
how it was reached, so the metrics can be split by how much the label itself
can be trusted. The set is heavily ``Buy`` and has no ``Schedule Visit`` case at
all; intent numbers from it are a sanity check, not a balanced benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# The seven entity types the extraction contract exposes. ``category`` is
# excluded deliberately: it is derived from ``property_type`` via
# property_type.csv, so scoring it would double-count the property-type
# decision rather than measure anything new.
ENTITY_FIELDS: Tuple[str, ...] = (
    "city",
    "area",
    "property_type",
    "bhk",
    "budget_value",
    "amenities",
    "furnishing",
)

SENTIMENT_CLASSES: Tuple[str, ...] = (
    "Enthusiastic",
    "Neutral",
    "Hesitant",
    "Frustrated",
)

# Same order as ``sentiment_intent.data.INTENT_CLASSES``.
INTENT_CLASSES: Tuple[str, ...] = (
    "Buy",
    "Rent",
    "Inquiry",
    "Schedule Visit",
    "Investment",
    "Request Callback",
)


def _case(
    case_id: str,
    sentiment: str,
    register: str,
    style: str,
    transcript: str,
    *,
    city: Optional[str] = None,
    area: Optional[str] = None,
    property_type: Optional[str] = None,
    bhk: Optional[int] = None,
    budget_value: Optional[float] = None,
    amenities: Optional[List[str]] = None,
    furnishing: Optional[str] = None,
    tags: Tuple[str, ...] = (),
    note: str = "",
) -> Dict[str, Any]:
    return {
        "id": case_id,
        "sentiment": sentiment,
        "register": register,
        "style": style,
        "tags": tags,
        "note": note,
        "transcript": transcript,
        "expected": {
            "city": city,
            "area": area,
            "property_type": property_type,
            "bhk": bhk,
            "budget_value": budget_value,
            "amenities": list(amenities or []),
            "furnishing": furnishing,
        },
    }


CASES: Tuple[Dict[str, Any], ...] = (
    # ------------------------------------------------------------------ #
    # Enthusiastic (13)
    # ------------------------------------------------------------------ #
    _case(
        "E01", "Enthusiastic", "english", "clean",
        "Agent: Good afternoon, thanks for calling. How can I help?\n"
        "Customer: Afternoon! My wife and I drove past your Baner site on Sunday "
        "and we talked about nothing else the whole way back. Three bedrooms is "
        "what we are after, and we can stretch to 1.4 crore if the finishing is "
        "right. A gym and a swimming pool in the complex would honestly seal it "
        "for us. Semi furnished is perfect.",
        city="Pune", area="Baner", property_type=None, bhk=3,
        budget_value=14_000_000,
        amenities=["Gym", "Swimming Pool"], furnishing="Semi Furnished",
        note="No property type is ever stated — tests that a null stays null.",
    ),
    _case(
        "E02", "Enthusiastic", "english", "clean",
        "Agent: Tell me a bit about what you are looking for.\n"
        "Customer: Happy to. We have been saving towards this for six years and "
        "the moment the sanction letter came through I told my husband, right, "
        "let us do it properly. A villa on Gangapur Road in Nashik, four "
        "bedrooms, and we have two and a half crore cleared. A private garden is "
        "the one thing I will not compromise on. Fully furnished if that is on "
        "the table.",
        city="Nashik", area="Gangapur Road", property_type="Villa", bhk=4,
        budget_value=25_000_000,
        amenities=["Private Garden"], furnishing="Fully Furnished",
        tags=("spelled-budget",),
        note="'two and a half crore' — spelled-out fractional amount.",
    ),
    _case(
        "E03", "Enthusiastic", "code-mixed", "clean",
        "Agent: Boliye, kaise help kar sakta hoon?\n"
        "Customer: Arre bahut din se soch rahe the, finally decide kiya! Wakad "
        "mein 2 BHK flat chahiye, budget 85 lakh tak comfortable hai. Gym aur "
        "covered parking mil jaaye toh perfect. Unfurnished chalega, hum khud "
        "karwa lenge.",
        city="Pune", area="Wakad", property_type="Apartment", bhk=2,
        budget_value=8_500_000,
        amenities=["Gym", "Covered Parking"], furnishing="Unfurnished",
    ),
    _case(
        "E04", "Enthusiastic", "english", "messy",
        "Agent: What are you hoping to find?\n"
        "Customer: So — okay, um, sorry, I am slightly over-caffeinated. Right. "
        "College Road, Nashik. Three BHK. We saw four other builders and yours "
        "is the only one where the sample unit actually felt liveable. Seventy "
        "to ninety lakh, and we can go to the top of that. Lift and power backup "
        "are must haves, my mother is eighty one.",
        city="Nashik", area="College Road", property_type=None, bhk=3,
        budget_value=9_000_000,
        amenities=["Lift", "Power Backup"], furnishing=None,
        tags=("range-budget", "spelled-budget"),
        note="Spelled-out range; 'eighty one' at the end is an age, not money.",
    ),
    _case(
        "E05", "Enthusiastic", "english", "clean",
        "Agent: How can I help?\n"
        "Customer: This is a great call to be making. My practice has outgrown "
        "its current floor and we are finally in a position to own rather than "
        "lease. Office space in Kharadi, Pune. Our budget is 3.2 crore. A "
        "conference room and visitor parking in the building would be ideal. "
        "Fully furnished preferred so we can move in fast.",
        city="Pune", area="Kharadi", property_type="Office Space", bhk=None,
        budget_value=32_000_000,
        amenities=["Conference Room", "Visitor Parking"],
        furnishing="Fully Furnished",
    ),
    _case(
        "E06", "Enthusiastic", "code-mixed", "messy",
        "Agent: Haan boliye.\n"
        "Customer: Namaste! Dekhiye, main pichle do saal se dekh raha tha but "
        "kabhi feel nahi aaya. Aapke Panchavati wale project ka layout dekha na, "
        "ekdum dil khush ho gaya. Row house chahiye, 3 BHK, 1.1 crore tak jaa "
        "sakte hain. Club house aur children's play area zaroori hai, do chhote "
        "bachche hain.",
        city="Nashik", area="Panchavati", property_type="Row House", bhk=3,
        budget_value=11_000_000,
        amenities=["Club House", "Children's Play Area"], furnishing=None,
        note="City is never said — it has to come from the area lookup.",
    ),
    _case(
        "E07", "Enthusiastic", "english", "clean",
        "Agent: Good morning, what can I do for you?\n"
        "Customer: Morning. I will be honest, I have been waiting three years "
        "for something in Koregaon Park to come up, and I saw your listing at "
        "six this morning and have been refreshing it since. Penthouse, four "
        "bedrooms, and money is not the obstacle — call it 4.5 crore. Rooftop "
        "lounge and concierge service, that is the standard I am after. Fully "
        "furnished.",
        city="Pune", area="Koregaon Park", property_type="Penthouse", bhk=4,
        budget_value=45_000_000,
        amenities=["Rooftop Lounge", "Concierge Service"],
        furnishing="Fully Furnished",
        note="'money is not the obstacle' puts a negation cue near the budget.",
    ),
    _case(
        "E08", "Enthusiastic", "english", "clean",
        "Agent: Where are you based at the moment?\n"
        "Customer: I am in Bengaluru right now, but the transfer came through "
        "last week and I could not be happier about it. We are relocating to "
        "Hinjewadi, Pune. 3 BHK apartment, up to 95 lakhs. A Wi-Fi ready "
        "building and a jogging track would be lovely. Semi furnished.",
        city="Pune", area="Hinjewadi", property_type="Apartment", bhk=3,
        budget_value=9_500_000,
        amenities=["Wi-Fi", "Jogging Track"], furnishing="Semi Furnished",
        tags=("multi-city", "distractor"),
        note="Calling from Bengaluru, buying in Pune.",
    ),
    _case(
        "E09", "Enthusiastic", "code-mixed", "clean",
        "Agent: Kya dhoondh rahe hain aap?\n"
        "Customer: Bahut acchi feeling aa rahi hai is baar! Pimple Saudagar mein "
        "2 BHK flat dekhna hai. Villa nahi chahiye, maintenance bahut ho jaata "
        "hai. 65 lakh budget hai. Swimming pool aur 24x7 security ho toh bohot "
        "accha. Semi furnished.",
        city="Pune", area="Pimple Saudagar", property_type="Apartment", bhk=2,
        budget_value=6_500_000,
        amenities=["Swimming Pool", "24x7 Security"], furnishing="Semi Furnished",
        tags=("negation",),
        note="Negation is in Hindi ('villa nahi chahiye'), not English.",
    ),
    _case(
        "E10", "Enthusiastic", "english", "clean",
        "Agent: What brings you in today?\n"
        "Customer: Good news, hopefully. My first proper salary hike came "
        "through and I want to put it into property instead of letting it sit. "
        "Studio apartment in Cidco, Nashik. Around 42 lakhs. If it has a lift "
        "and covered parking I am sold. Unfurnished is fine.",
        city="Nashik", area="Cidco", property_type="Studio Apartment", bhk=None,
        budget_value=4_200_000,
        amenities=["Lift", "Covered Parking"], furnishing="Unfurnished",
    ),
    _case(
        "E11", "Enthusiastic", "english", "messy",
        "Agent: Go ahead.\n"
        "Customer: Right so — my brother and I, we have — sorry, this is the "
        "third time I have tried to call today and I keep getting cut off, but "
        "no no, I am delighted we finally connected. Wagholi. A bungalow, four "
        "bedrooms, up to 1.6 crore between us. A private garden and a bit of "
        "open parking. Unfurnished.",
        city="Pune", area="Wagholi", property_type="Villa", bhk=4,
        budget_value=16_000_000,
        amenities=["Private Garden", "Open Parking"], furnishing="Unfurnished",
        note="Enthusiastic caller who opens with a complaint — the dangerous "
             "Enthusiastic/Frustrated confusion, deliberately baited. "
             "'bungalow' maps to Villa per property_type_synonyms.csv.",
    ),
    _case(
        "E12", "Enthusiastic", "english", "clean",
        "Agent: How can I help?\n"
        "Customer: We hesitated on a place a decade ago and regretted it ever "
        "since, so I am thrilled to be doing this properly this time. Trimbak "
        "Road, Nashik. 3 BHK apartment. We have one crore twenty lakhs "
        "arranged. A modular kitchen and a club house. Fully furnished.",
        city="Nashik", area="Trimbak Road", property_type="Apartment", bhk=3,
        budget_value=12_000_000,
        amenities=["Modular Kitchen", "Club House"], furnishing="Fully Furnished",
        tags=("spelled-budget",),
        note="'one crore twenty lakhs' is one compound amount, not a range.",
    ),
    _case(
        "E13", "Enthusiastic", "code-mixed", "messy",
        "Agent: Haan bataiye.\n"
        "Customer: Ekdum sahi timing pe call kiya maine! Kothrud mein 1 BHK "
        "flat rent pe chahiye. Monthly 28000 rupees tak thik hai. Lift aur "
        "security zaroori. Fully furnished ho toh best, kal hi shift ho sakta "
        "hoon.",
        city="Pune", area="Kothrud", property_type="Apartment", bhk=1,
        budget_value=28_000,
        amenities=["Lift", "24x7 Security"], furnishing="Fully Furnished",
        tags=("bare-rupees", "monthly-rent"),
    ),

    # ------------------------------------------------------------------ #
    # Neutral (12)
    # ------------------------------------------------------------------ #
    _case(
        "N01", "Neutral", "english", "clean",
        "Agent: What can I help you with?\n"
        "Customer: I am collecting information at this stage. Two BHK in Aundh, "
        "Pune. Budget somewhere near 80 lakhs. Lift and power backup are the "
        "usual requirements. Semi furnished. That is all for now.",
        city="Pune", area="Aundh", property_type=None, bhk=2,
        budget_value=8_000_000,
        amenities=["Lift", "Power Backup"], furnishing="Semi Furnished",
    ),
    _case(
        "N02", "Neutral", "english", "clean",
        "Agent: Tell me the requirement.\n"
        "Customer: We need a warehouse in Satpur, Nashik. Roughly 1.8 crore. "
        "Power backup and 24x7 security are required for insurance reasons. "
        "Unfurnished, obviously.",
        city="Nashik", area="Satpur", property_type="Warehouse", bhk=None,
        budget_value=18_000_000,
        amenities=["Power Backup", "24x7 Security"], furnishing="Unfurnished",
    ),
    _case(
        "N03", "Neutral", "code-mixed", "clean",
        "Agent: Kya requirement hai?\n"
        "Customer: Hadapsar mein 3 BHK flat dekhna hai. Budget 95 lakh. Covered "
        "parking aur gym hona chahiye. Semi furnished. Details bhej dijiye.",
        city="Pune", area="Hadapsar", property_type="Apartment", bhk=3,
        budget_value=9_500_000,
        amenities=["Covered Parking", "Gym"], furnishing="Semi Furnished",
    ),
    _case(
        "N04", "Neutral", "english", "clean",
        "Agent: Go ahead.\n"
        "Customer: Residential plot in Nashik Road. Between 35 and 50 lakhs. No "
        "particular amenity requirements. Furnishing is not a relevant question "
        "for a plot.",
        city="Nashik", area="Nashik Road", property_type="Residential Plot",
        bhk=None, budget_value=5_000_000,
        amenities=[], furnishing=None,
        tags=("range-budget",),
        note="Empty amenity list and a null furnishing — both should stay empty.",
    ),
    _case(
        "N05", "Neutral", "english", "clean",
        "Agent: What are you after?\n"
        "Customer: A serviced apartment in Viman Nagar for a twelve month "
        "corporate lease. The company ceiling is 75000 rupees a month. "
        "Housekeeping and Wi-Fi need to be included. Fully furnished.",
        city="Pune", area="Viman Nagar", property_type="Serviced Apartment",
        bhk=None, budget_value=75_000,
        amenities=["Housekeeping", "Wi-Fi"], furnishing="Fully Furnished",
        tags=("bare-rupees", "monthly-rent"),
        note="'twelve month' must not be read as a BHK count.",
    ),
    _case(
        "N06", "Neutral", "code-mixed", "messy",
        "Agent: Boliye.\n"
        "Customer: Haan toh... Ambad mein godown chahiye. Size thik thaak, 60 "
        "lakh ke aas paas. Security aur power backup. Unfurnished.",
        city="Nashik", area="Ambad", property_type="Godown", bhk=None,
        budget_value=6_000_000,
        amenities=["24x7 Security", "Power Backup"], furnishing="Unfurnished",
    ),
    _case(
        "N07", "Neutral", "english", "clean",
        "Agent: Which city?\n"
        "Customer: We looked at Nashik earlier in the year, but we are "
        "searching in Pune now. Bavdhan specifically. 3 BHK, 1.15 crore. A "
        "children's play area matters to us. Semi furnished.",
        city="Pune", area="Bavdhan", property_type=None, bhk=3,
        budget_value=11_500_000,
        amenities=["Children's Play Area"], furnishing="Semi Furnished",
        tags=("multi-city",),
        note="Both cities are in-catalog; only the second is being shopped in.",
    ),
    _case(
        "N08", "Neutral", "english", "clean",
        "Agent: Go ahead please.\n"
        "Customer: Duplex in Panchavati. Four bedrooms. 1.75 crore is the "
        "figure I have been given. Rainwater harvesting and solar panels if "
        "available. Unfurnished.",
        city="Nashik", area="Panchavati", property_type="Duplex", bhk=4,
        budget_value=17_500_000,
        amenities=["Rainwater Harvesting", "Solar Panels"],
        furnishing="Unfurnished",
    ),
    _case(
        "N09", "Neutral", "english", "clean",
        "Agent: What sort of place?\n"
        "Customer: An apartment in Magarpatta. Not a studio apartment, a proper "
        "2 BHK. Budget is 88 lakhs. Gym and swimming pool. Semi furnished.",
        city="Pune", area="Magarpatta", property_type="Apartment", bhk=2,
        budget_value=8_800_000,
        amenities=["Gym", "Swimming Pool"], furnishing="Semi Furnished",
        tags=("negation",),
        note="The negated type is more specific than the wanted one.",
    ),
    _case(
        "N10", "Neutral", "code-mixed", "clean",
        "Agent: Kya chahiye aapko?\n"
        "Customer: Ravet mein 2 BHK rent pe. 22000 per month. Visitor parking "
        "aur lift chahiye. Semi furnished.",
        city="Pune", area="Ravet", property_type=None, bhk=2,
        budget_value=22_000,
        amenities=["Visitor Parking", "Lift"], furnishing="Semi Furnished",
        tags=("bare-rupees", "monthly-rent"),
        note="Monthly rent with no unit word at all ('22000 per month').",
    ),
    _case(
        "N11", "Neutral", "english", "clean",
        "Agent: How can I help?\n"
        "Customer: I need PG accommodation in Untwadi for my daughter. Around "
        "12000 a month. Housekeeping, Wi-Fi and 24x7 security are the "
        "priorities. Fully furnished.",
        city="Nashik", area="Untwadi", property_type="PG Accommodation",
        bhk=None, budget_value=12_000,
        amenities=["Housekeeping", "Wi-Fi", "24x7 Security"],
        furnishing="Fully Furnished",
        tags=("bare-rupees", "monthly-rent"),
    ),
    _case(
        "N12", "Neutral", "english", "messy",
        "Agent: Yes?\n"
        "Customer: Um. Farm house. Talegaon. Or, no, sorry — farm land, "
        "actually. My father wants farm land. Two crore. Water supply is the "
        "only thing that matters out there. Furnishing does not apply.",
        city="Pune", area="Talegaon", property_type="Farm Land", bhk=None,
        budget_value=20_000_000,
        amenities=["Water Supply"], furnishing=None,
        tags=("spelled-budget", "negation"),
        note="Mid-sentence self-correction: the second property type is the "
             "real one.",
    ),

    # ------------------------------------------------------------------ #
    # Hesitant (13)
    # ------------------------------------------------------------------ #
    _case(
        "H01", "Hesitant", "english", "clean",
        "Agent: What are you looking for?\n"
        "Customer: I am not sure yet, to be honest. Maybe a 2 BHK in Kondhwa. "
        "My husband thinks we should wait another year. Budget would be around "
        "62 lakhs if we did go ahead. Parking, I suppose. Semi furnished, maybe.",
        city="Pune", area="Kondhwa", property_type=None, bhk=2,
        budget_value=6_200_000,
        amenities=["Parking"], furnishing="Semi Furnished",
        note="'not sure' sits two sentences before the BHK — it must not scope "
             "onto it.",
    ),
    _case(
        "H02", "Hesitant", "english", "clean",
        "Agent: Tell me more.\n"
        "Customer: We have been going back and forth on this for months. Three "
        "BHK on Sharanpur Road, Nashik, in theory. 90 lakhs is the most we "
        "could manage and even that feels like a stretch. Lift, club house. "
        "Unfurnished, probably. I might call back after speaking to my brother.",
        city="Nashik", area="Sharanpur Road", property_type=None, bhk=3,
        budget_value=9_000_000,
        amenities=["Lift", "Club House"], furnishing="Unfurnished",
    ),
    _case(
        "H03", "Hesitant", "code-mixed", "clean",
        "Agent: Boliye.\n"
        "Customer: Dekhiye, abhi decide nahi kiya hai properly. Chinchwad mein "
        "2 BHK flat ka idea hai. 70 lakh ke aas paas. Gym waghera hona chahiye "
        "shayad. Semi furnished. Thoda time chahiye sochne ke liye.",
        city="Pune", area="Chinchwad", property_type="Apartment", bhk=2,
        budget_value=7_000_000,
        amenities=["Gym"], furnishing="Semi Furnished",
    ),
    _case(
        "H04", "Hesitant", "english", "messy",
        "Agent: Go on.\n"
        "Customer: I — hmm. Okay. Not a villa, we cannot maintain a villa. "
        "Something smaller. An apartment in Canada Corner maybe. Three "
        "bedrooms. 1.1 crore, though I keep moving that number. A lift at "
        "minimum. I do not know about furnishing yet.",
        city="Nashik", area="Canada Corner", property_type="Apartment", bhk=3,
        budget_value=11_000_000,
        amenities=["Lift"], furnishing=None,
        tags=("negation",),
        note="Two mentions of the ruled-out type, only the first with an "
             "explicit cue.",
    ),
    _case(
        "H05", "Hesitant", "english", "clean",
        "Agent: Where are you looking?\n"
        "Customer: That is part of the problem, I keep changing my mind. I was "
        "considering Mumbai, then Pune. Right now I am leaning towards Undri. 3 "
        "BHK, maybe 1.05 crore. Covered parking and a garden. Semi furnished, I "
        "think.",
        city="Pune", area="Undri", property_type=None, bhk=3,
        budget_value=10_500_000,
        amenities=["Covered Parking", "Garden"], furnishing="Semi Furnished",
        tags=("multi-city", "distractor"),
    ),
    _case(
        "H06", "Hesitant", "code-mixed", "messy",
        "Agent: Haan?\n"
        "Customer: Umm... pata nahi. Jail Road pe 2 BHK dekha tha, par... 55 "
        "lakh se zyada nahi ja sakte. Lift hona chahiye bas. Furnishing ka abhi "
        "soch nahi hai. Ghar mein baat karni padegi.",
        city="Nashik", area="Jail Road", property_type=None, bhk=2,
        budget_value=5_500_000,
        amenities=["Lift"], furnishing=None,
    ),
    _case(
        "H07", "Hesitant", "english", "clean",
        "Agent: What did you have in mind?\n"
        "Customer: A row house in Balewadi, possibly. My loan pre-approval is "
        "only 1.2 crore and the ones I have seen start higher, so I am "
        "hesitant. Three bedrooms. A jogging track and a club house would be "
        "nice but I would not pay extra for them. Semi furnished.",
        city="Pune", area="Balewadi", property_type="Row House", bhk=3,
        budget_value=12_000_000,
        amenities=["Jogging Track", "Club House"], furnishing="Semi Furnished",
        note="A negation cue follows the amenities — it must not retroactively "
             "cancel them.",
    ),
    _case(
        "H08", "Hesitant", "english", "clean",
        "Agent: Go ahead.\n"
        "Customer: A commercial shop in Deolali. I am honestly unsure whether "
        "retail is the right bet this year. 45 lakhs, give or take. CCTV "
        "surveillance would help. Unfurnished.",
        city="Nashik", area="Deolali", property_type="Commercial Shop",
        bhk=None, budget_value=4_500_000,
        amenities=["CCTV Surveillance"], furnishing="Unfurnished",
    ),
    _case(
        "H09", "Hesitant", "english", "messy",
        "Agent: Yes, tell me.\n"
        "Customer: I keep starting this conversation and then stopping. Warje. "
        "Two BHK. Around seventy five lakhs, I think. Power backup, and a lift "
        "because of my knees. Furnishing, no strong view.",
        city="Pune", area="Warje", property_type=None, bhk=2,
        budget_value=7_500_000,
        amenities=["Power Backup", "Lift"], furnishing=None,
        tags=("spelled-budget",),
    ),
    _case(
        "H10", "Hesitant", "code-mixed", "clean",
        "Agent: Kya soch rahe hain?\n"
        "Customer: Confusion hai thoda. Tathawade mein 3 BHK apartment, 85 lakh "
        "tak. Swimming pool aur children's play area dekh rahe the but zaroori "
        "nahi. Unfurnished. Abhi final nahi hai kuch bhi.",
        city="Pune", area="Tathawade", property_type="Apartment", bhk=3,
        budget_value=8_500_000,
        amenities=["Swimming Pool", "Children's Play Area"],
        furnishing="Unfurnished",
    ),
    _case(
        "H11", "Hesitant", "english", "clean",
        "Agent: Which area?\n"
        "Customer: We are currently in Pune, but the job may move us to Nashik. "
        "Makhmalabad, if it happens. Not a flat — an independent house. Four "
        "bedrooms. 1.3 crore at the outside. A private garden and open parking. "
        "Unfurnished. It all depends on whether the transfer is confirmed.",
        city="Nashik", area="Makhmalabad", property_type="Independent House",
        bhk=4, budget_value=13_000_000,
        amenities=["Private Garden", "Open Parking"], furnishing="Unfurnished",
        tags=("multi-city", "negation"),
        note="Both cities are in-catalog and the current one is named first.",
    ),
    _case(
        "H12", "Hesitant", "english", "clean",
        "Agent: How can I help?\n"
        "Customer: Possibly a studio apartment in Fatima Nagar. I say possibly "
        "because I have said yes to two places already and backed out both "
        "times. 38 lakhs. Lift, security. Fully furnished would remove one "
        "decision from my life.",
        city="Pune", area="Fatima Nagar", property_type="Studio Apartment",
        bhk=None, budget_value=3_800_000,
        amenities=["Lift", "24x7 Security"], furnishing="Fully Furnished",
    ),
    _case(
        "H13", "Hesitant", "english", "messy",
        "Agent: Right.\n"
        "Customer: So — I think — look, the honest answer is I am not "
        "committed. Ozar. 2 BHK. I have 4200000 rupees approved and nothing "
        "beyond that. Water supply and power backup, because out there it "
        "matters. Unfurnished.",
        city="Nashik", area="Ozar", property_type=None, bhk=2,
        budget_value=4_200_000,
        amenities=["Water Supply", "Power Backup"], furnishing="Unfurnished",
        tags=("bare-rupees",),
    ),

    # ------------------------------------------------------------------ #
    # Frustrated (12)
    # ------------------------------------------------------------------ #
    _case(
        "F01", "Frustrated", "english", "clean",
        "Agent: How can I help today?\n"
        "Customer: You can start by explaining why nobody has called me back. "
        "This is my fourth attempt. 3 BHK in Kalyani Nagar, 1.9 crore, I have "
        "said this to three different people now. Covered parking and a gym. "
        "Semi furnished. Please write it down this time.",
        city="Pune", area="Kalyani Nagar", property_type=None, bhk=3,
        budget_value=19_000_000,
        amenities=["Covered Parking", "Gym"], furnishing="Semi Furnished",
    ),
    _case(
        "F02", "Frustrated", "english", "clean",
        "Agent: Tell me the requirement.\n"
        "Customer: The requirement has not changed since March. Pathardi "
        "Phata, Nashik. Four BHK villa. 2.2 crore. Swimming pool, club house. "
        "Fully furnished. Your site listed a unit that was sold eight months "
        "ago and nobody bothered to take it down.",
        city="Nashik", area="Pathardi Phata", property_type="Villa", bhk=4,
        budget_value=22_000_000,
        amenities=["Swimming Pool", "Club House"], furnishing="Fully Furnished",
        note="'has not changed' is a negation cue sitting near the "
             "requirement, but negates nothing.",
    ),
    _case(
        "F03", "Frustrated", "code-mixed", "clean",
        "Agent: Boliye sir.\n"
        "Customer: Kitni baar bataana padega? Wanowrie mein 2 BHK flat, 72 "
        "lakh. Lift aur power backup. Semi furnished. Teen baar form bhara hai "
        "maine aapke website pe aur ek bhi reply nahi aaya.",
        city="Pune", area="Wanowrie", property_type="Apartment", bhk=2,
        budget_value=7_200_000,
        amenities=["Lift", "Power Backup"], furnishing="Semi Furnished",
    ),
    _case(
        "F04", "Frustrated", "english", "messy",
        "Agent: Go ahead.\n"
        "Customer: Right, so — no. No, listen. I do not want a villa, I have "
        "said that every single time. An apartment. Mumbai Naka. Three "
        "bedrooms. 85 lakhs. A lift that actually works, unlike the one in the "
        "building your colleague showed me. Semi furnished.",
        city="Nashik", area="Mumbai Naka", property_type="Apartment", bhk=3,
        budget_value=8_500_000,
        amenities=["Lift"], furnishing="Semi Furnished",
        tags=("negation", "multi-city", "distractor"),
        note="'Mumbai Naka' is a Nashik area whose name contains a distractor "
             "city.",
    ),
    _case(
        "F05", "Frustrated", "english", "clean",
        "Agent: What do you need?\n"
        "Customer: What I need is for someone to action this. Office space in "
        "Yerawada, 2.6 crore, and I have been chasing since January. A "
        "conference room and power backup. Unfurnished. If this call goes the "
        "same way as the last three I am going elsewhere.",
        city="Pune", area="Yerawada", property_type="Office Space", bhk=None,
        budget_value=26_000_000,
        amenities=["Conference Room", "Power Backup"], furnishing="Unfurnished",
    ),
    _case(
        "F06", "Frustrated", "code-mixed", "messy",
        "Agent: Haan?\n"
        "Customer: Dekhiye, mera time waste ho raha hai. Rane Nagar mein 3 BHK "
        "chahiye, 78 lakh. Gym, security, parking. Unfurnished. Pichle mahine "
        "aapke agent ne site visit ka time diya aur khud aaya hi nahi.",
        city="Nashik", area="Rane Nagar", property_type=None, bhk=3,
        budget_value=7_800_000,
        amenities=["Gym", "24x7 Security", "Parking"], furnishing="Unfurnished",
    ),
    _case(
        "F07", "Frustrated", "english", "clean",
        "Agent: Which city are you looking in?\n"
        "Customer: Pune. I already told your Hyderabad office that, and they "
        "transferred me twice. Pashan. 3 BHK apartment. 1.45 crore. Club house "
        "and covered parking. Fully furnished. This is the last time I explain "
        "it.",
        city="Pune", area="Pashan", property_type="Apartment", bhk=3,
        budget_value=14_500_000,
        amenities=["Club House", "Covered Parking"], furnishing="Fully Furnished",
        tags=("multi-city", "distractor"),
    ),
    _case(
        "F08", "Frustrated", "english", "clean",
        "Agent: Yes?\n"
        "Customer: Residential plot in Adgaon. 55 lakhs. A boundary wall and "
        "water supply, that is it. Furnishing does not apply, it is a plot. I "
        "have now repeated this to your chatbot, your form and two humans.",
        city="Nashik", area="Adgaon", property_type="Residential Plot",
        bhk=None, budget_value=5_500_000,
        amenities=["Water Supply"], furnishing=None,
        note="'boundary wall' is not in amenities.csv, so it cannot be "
             "represented and is not counted against either system.",
    ),
    _case(
        "F09", "Frustrated", "english", "clean",
        "Agent: Go ahead.\n"
        "Customer: Moshi. 2 BHK. Between 55 and 68 lakhs. Lift and visitor "
        "parking. Semi furnished. And before you ask, yes I filled the form, "
        "twice, and no, nobody rang.",
        city="Pune", area="Moshi", property_type=None, bhk=2,
        budget_value=6_800_000,
        amenities=["Lift", "Visitor Parking"], furnishing="Semi Furnished",
        tags=("range-budget",),
    ),
    _case(
        "F10", "Frustrated", "code-mixed", "clean",
        "Agent: Kya chahiye?\n"
        "Customer: Bahut ho gaya ab. Punawale mein 3 BHK flat, 92 lakh. Ground "
        "floor nahi chahiye. Covered parking aur gym. Semi furnished. Do hafte "
        "se follow up kar raha hoon, koi jawab nahi.",
        city="Pune", area="Punawale", property_type="Apartment", bhk=3,
        budget_value=9_200_000,
        amenities=["Covered Parking", "Gym"], furnishing="Semi Furnished",
        tags=("negation",),
    ),
    _case(
        "F11", "Frustrated", "english", "messy",
        "Agent: Right, I have you now.\n"
        "Customer: Finally. Right — Govind Nagar, Nashik. A — sorry, hold on — "
        "a 3 BHK. 68 lakhs. Lift, power backup, children's play area. "
        "Unfurnished. Your system has lost my details twice and I am not doing "
        "this a fourth time.",
        city="Nashik", area="Govind Nagar", property_type=None, bhk=3,
        budget_value=6_800_000,
        amenities=["Lift", "Power Backup", "Children's Play Area"],
        furnishing="Unfurnished",
    ),
    _case(
        "F12", "Frustrated", "english", "clean",
        "Agent: What property type?\n"
        "Customer: A showroom in Lohegaon. 3.4 crore. Visitor parking and CCTV "
        "surveillance. Unfurnished. I want a callback from a manager, not "
        "another junior reading from a script.",
        city="Pune", area="Lohegaon", property_type="Showroom", bhk=None,
        budget_value=34_000_000,
        amenities=["Visitor Parking", "CCTV Surveillance"],
        furnishing="Unfurnished",
    ),
)


# (intent, basis). "stated": the caller says it in so many words. "inferred": no
# intent phrase, but a purchase budget / loan / ownership framing makes Buy the
# only sensible reading. "ambiguous": two readings are defensible; the label is
# the one a sales agent would act on, and the alternative is noted.
INTENT_LABELS: Dict[str, Tuple[str, str]] = {
    "E01": ("Buy", "inferred"),
    "E02": ("Buy", "inferred"),            # sanction letter = home loan
    "E03": ("Buy", "inferred"),
    "E04": ("Buy", "inferred"),
    "E05": ("Buy", "stated"),              # "own rather than lease"
    "E06": ("Buy", "inferred"),
    "E07": ("Buy", "inferred"),
    "E08": ("Buy", "inferred"),
    "E09": ("Buy", "ambiguous"),           # "flat dekhna hai": looking for vs. visit
    "E10": ("Investment", "ambiguous"),    # "put it into property instead of letting it sit"; alt Buy
    "E11": ("Buy", "inferred"),
    "E12": ("Buy", "inferred"),
    "E13": ("Rent", "stated"),
    "N01": ("Inquiry", "stated"),          # "collecting information at this stage"
    "N02": ("Buy", "inferred"),
    "N03": ("Inquiry", "ambiguous"),       # "details bhej dijiye"; alt Buy
    "N04": ("Buy", "inferred"),
    "N05": ("Rent", "stated"),             # corporate lease, per month
    "N06": ("Buy", "inferred"),
    "N07": ("Buy", "inferred"),
    "N08": ("Buy", "inferred"),
    "N09": ("Buy", "inferred"),
    "N10": ("Rent", "stated"),
    "N11": ("Rent", "stated"),             # PG, per month
    "N12": ("Buy", "inferred"),
    "H01": ("Buy", "inferred"),
    "H02": ("Buy", "inferred"),            # caller will call back; not a callback request
    "H03": ("Buy", "inferred"),
    "H04": ("Buy", "inferred"),
    "H05": ("Buy", "inferred"),
    "H06": ("Buy", "inferred"),
    "H07": ("Buy", "inferred"),            # loan pre-approval
    "H08": ("Investment", "ambiguous"),    # "whether retail is the right bet"; alt Buy
    "H09": ("Buy", "inferred"),
    "H10": ("Buy", "inferred"),
    "H11": ("Buy", "inferred"),
    "H12": ("Buy", "inferred"),
    "H13": ("Buy", "inferred"),
    "F01": ("Buy", "ambiguous"),           # complains nobody called back; alt Request Callback
    "F02": ("Buy", "inferred"),
    "F03": ("Buy", "inferred"),
    "F04": ("Buy", "inferred"),
    "F05": ("Buy", "inferred"),
    "F06": ("Buy", "inferred"),
    "F07": ("Buy", "inferred"),
    "F08": ("Buy", "inferred"),
    "F09": ("Buy", "inferred"),
    "F10": ("Buy", "inferred"),
    "F11": ("Buy", "inferred"),
    "F12": ("Request Callback", "stated"), # "I want a callback from a manager"
}

assert set(INTENT_LABELS) == {case["id"] for case in CASES}, "intent label missing"
assert all(label in INTENT_CLASSES for label, _ in INTENT_LABELS.values())
for _case_dict in CASES:
    _case_dict["intent"], _case_dict["intent_basis"] = INTENT_LABELS[_case_dict["id"]]
del _case_dict


def by_sentiment() -> Dict[str, int]:
    counts: Dict[str, int] = {label: 0 for label in SENTIMENT_CLASSES}
    for case in CASES:
        counts[case["sentiment"]] += 1
    return counts


def by_intent() -> Dict[str, int]:
    counts: Dict[str, int] = {label: 0 for label in INTENT_CLASSES}
    for case in CASES:
        counts[case["intent"]] += 1
    return counts
