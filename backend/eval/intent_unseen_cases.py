"""60 unseen transcripts for intent: balanced, and intent never stated outright.

Why this set exists
-------------------
``unseen50_cases`` was written for extraction and sentiment. Its intent labels
were added afterwards, 41 of them are Buy, none is Schedule Visit, and most were
inferred by the labeller rather than said by the caller. That set cannot tell
a model that understands intent from one that says "Buy" every time.

This one is built for intent alone:

* **Balanced** — 10 calls per intent, 7 English and 3 code-mixed in each.
* **Inferred, never stated** — no caller uses the vocabulary the generators
  use to *name* an intent (buy / purchase / rent / lease / invest / visit /
  view / callback / call back / enquiry / information / explore, or the
  romanised equivalents). The intent has to be read from the situation:
  a stamp-duty question means Buy, a deposit question means Rent, "I won't
  live in it" means Investment. :func:`stated_intent_leaks` enforces this and
  the runner refuses to score the set if anything leaks.
* **Written before the training data.** Every case here was fixed before the
  implied-intent generator's idea pools existed, so the pools could not be
  shaped around the test. The pools were then written to avoid these specific
  situations, and ``run_intent_unseen`` measures the longest word run each
  case shares with the training corpus rather than asserting there is none.

Honest limit: one person wrote both this set and the generator. The
phrase-holdout diagnostic (``sentiment_intent.phrase_holdout --task intent``)
is the stricter test of generalisation; this set is the human-readable one.

Each label is one careful human reading. Where a call could be read two ways,
it was rewritten until it could not — a test case whose label is arguable
measures the labeller, not the model.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

INTENT_CLASSES: Tuple[str, ...] = (
    "Buy",
    "Rent",
    "Inquiry",
    "Schedule Visit",
    "Investment",
    "Request Callback",
)

# Words that *name* an intent. Any of these in a customer turn would let a
# phrase-matcher pass the case, which is exactly what this set must not allow.
STATED_INTENT_PATTERNS: Tuple[str, ...] = (
    r"\bbuy", r"\bbought\b", r"\bpurchas", r"\brent", r"\blease", r"\bleasing\b",
    r"\binvest", r"\bvisit", r"\bview", r"call ?back", r"\bcall me\b",
    r"\benquir", r"\binquir", r"\binformation\b", r"\bexplore\b",
    r"\bkhareed", r"\bkiray", r"\bdekhna\b", r"\bjaankari\b",
)

_AGENT_OPEN = "Agent: Good afternoon, thank you for calling. How can I help you today?"
_AGENT_MORE = "Agent: Sure. Anything else I should note down?"


def _case(case_id: str, intent: str, register: str, style: str,
          first: str, second: str = "", note: str = "") -> Dict[str, Any]:
    lines = [_AGENT_OPEN, f"Customer: {first}"]
    if second:
        lines += [_AGENT_MORE, f"Customer: {second}"]
    return {
        "id": case_id,
        "intent": intent,
        "register": register,
        "style": style,
        "transcript": "\n".join(lines),
        "note": note,
    }


CASES: Tuple[Dict[str, Any], ...] = (
    # ------------------------------ Buy ------------------------------ #
    _case("B01", "Buy", "english", "clean",
          "My father retires in March and we want the registry done in his name "
          "before then. A 3 BHK in Baner, up to 1.3 crore.",
          "It has to be something that stays in the family for good."),
    _case("B02", "Buy", "english", "clean",
          "Our PF withdrawal came through and my wife's gold is going towards the "
          "down payment. 2 BHK in Wakad, around 80 lakh all in."),
    _case("B03", "Buy", "english", "messy",
          "Right so, um, I've budgeted stamp duty and registration separately, on "
          "top of the ninety lakh. College Road, Nashik, 3 BHK.",
          "Oh and covered parking."),
    _case("B04", "Buy", "english", "clean",
          "I want a home nobody can ever ask us to vacate again. We have moved "
          "four times in six years. 2 BHK in Kothrud, 95 lakh."),
    _case("B05", "Buy", "english", "clean",
          "Is the title clear, and does the builder handle the conveyance deed? "
          "My lawyer will ask before we sign the agreement for sale.",
          "3 BHK, Hadapsar, 1.1 crore."),
    _case("B06", "Buy", "english", "messy",
          "What's the payment plan — is it construction-linked? We'd want to "
          "block a unit with the token amount, sorry, the, uh, booking amount. "
          "Undri, 2 BHK."),
    _case("B07", "Buy", "english", "clean",
          "Once the possession letter comes, my mother moves in and the flat is "
          "hers. Gangapur Road, 2 BHK, up to 70 lakh."),
    _case("B08", "Buy", "code-mixed", "clean",
          "Apna ghar chahiye jisme koi makaan-malik na ho. Registry mere naam pe "
          "honi chahiye. Chinchwad mein 2 BHK, 75 lakh."),
    _case("B09", "Buy", "code-mixed", "messy",
          "Down payment ke liye FD tod di hai, pachees lakh ready hai, baaki bank "
          "se. Umm, Pimple Saudagar, 2 BHK."),
    _case("B10", "Buy", "code-mixed", "clean",
          "Stamp duty ka hisaab bhi laga liya hai. Possession 2027 tak chahiye, "
          "Nashik Road pe 3 BHK, ek crore tak."),

    # ------------------------------ Rent ----------------------------- #
    _case("R01", "Rent", "english", "clean",
          "My company has put me on a two-year project here and HR reimburses up "
          "to 35,000 a month for housing. 2 BHK near Hinjewadi."),
    _case("R02", "Rent", "english", "clean",
          "What is the deposit — two months or three? And is maintenance "
          "included in the monthly amount? 1 BHK in Viman Nagar, fully furnished."),
    _case("R03", "Rent", "english", "messy",
          "We're, uh, three students sharing, and each of us can do about nine "
          "thousand a month. Near the university, Aundh side. Furnished please."),
    _case("R04", "Rent", "english", "clean",
          "I'm only here until my posting ends next June, then we go back to "
          "Delhi. 2 BHK in Kharadi, under 30,000 a month."),
    _case("R05", "Rent", "english", "clean",
          "Does the owner allow a small dog? I can pay 18,000 every month and "
          "move in on the first. 1 BHK, Baner."),
    _case("R06", "Rent", "english", "messy",
          "Our building is being redeveloped, and the builder pays us a monthly "
          "allowance for thirty months while it's — while it goes up again. So we "
          "need somewhere temporary in Erandwane, 3 BHK."),
    _case("R07", "Rent", "english", "clean",
          "My wife's hospital posting here is for eighteen months and her "
          "allowance covers 25,000 a month. Furnished 2 BHK near Deccan."),
    _case("R08", "Rent", "code-mixed", "clean",
          "Deposit kitna lagega? Mahine ka 20 hazaar se zyada nahi de paunga. "
          "Owner family ko dega na? Wanowrie mein 1 BHK."),
    _case("R09", "Rent", "code-mixed", "clean",
          "Office ne do saal ke liye yahan bheja hai, company har mahine 30 hazaar "
          "tak deti hai. Magarpatta ke paas 2 BHK."),
    _case("R10", "Rent", "code-mixed", "messy",
          "Hostel chhodna hai, matlab, do dost milke rahenge, har mahine aath-aath "
          "hazaar. Katraj side, furnished."),

    # --------------------------- Inquiry ----------------------------- #
    _case("Q01", "Inquiry", "english", "clean",
          "We're two years away from doing anything. I'd just like a sense of "
          "where prices are in Baner right now for a 3 BHK."),
    _case("Q02", "Inquiry", "english", "clean",
          "Is your Bavdhan project RERA-registered, and what is the ratio of "
          "carpet area to super built-up?",
          "No requirement as such, I'm noting it down."),
    _case("Q03", "Inquiry", "english", "clean",
          "My cousin asked me to find out roughly what a 2 BHK goes for in Wakad "
          "these days. There's no plan on my side."),
    _case("Q04", "Inquiry", "english", "messy",
          "Could you, um, email me the price list and the floor plans? I'll go "
          "through them later, no rush on anything."),
    _case("Q05", "Inquiry", "english", "clean",
          "Is Pimple Saudagar better value than Wakad at the moment? I'm only "
          "trying to get a feel for the market."),
    _case("Q06", "Inquiry", "english", "clean",
          "I'm writing a piece on housing in Nashik. What's the going rate per "
          "square foot on College Road?"),
    _case("Q07", "Inquiry", "english", "messy",
          "What amenities come as standard in your projects, and what are the "
          "maintenance charges, roughly? No decision on my end, I'm, yeah, just "
          "noting it."),
    _case("Q08", "Inquiry", "code-mixed", "clean",
          "Sirf rate jaanna tha. Kharadi mein 3 BHK kitne ka chal raha hai "
          "aajkal?"),
    _case("Q09", "Inquiry", "code-mixed", "clean",
          "Brochure aur price list WhatsApp kar dijiye, aaram se padh lunga. Abhi "
          "koi jaldi nahi hai."),
    _case("Q10", "Inquiry", "code-mixed", "messy",
          "Abhi koi plan nahi hai, bas, woh, pata karna tha ki Nashik Road pe "
          "square foot ka kya rate hai."),

    # ------------------------ Schedule Visit ------------------------- #
    _case("V01", "Schedule Visit", "english", "clean",
          "Is the sample flat open on Sunday mornings? My parents can only come "
          "then. It's the 3 BHK in Balewadi."),
    _case("V02", "Schedule Visit", "english", "clean",
          "Can your site person meet us at the gate around four tomorrow? I'd "
          "like to walk through the actual unit, not the brochure."),
    _case("V03", "Schedule Visit", "english", "messy",
          "Before anything else I want to stand on that balcony and, um, check "
          "the morning sun myself. When can that happen? Tower B, 2 BHK."),
    _case("V04", "Schedule Visit", "english", "clean",
          "Send me the site address and a slot this week — I'll take half a day "
          "off work for it. Kharadi, 2 BHK."),
    _case("V05", "Schedule Visit", "english", "clean",
          "We're in Nashik only on the 14th and 15th. Can someone show us around "
          "the Gangapur Road tower on one of those days?"),
    _case("V06", "Schedule Visit", "english", "messy",
          "Photos can, you know, lie. I need to physically go inside the "
          "seventh-floor unit. Is Saturday at eleven possible?"),
    _case("V07", "Schedule Visit", "english", "clean",
          "Can I bring my contractor along to check the plumbing and wall "
          "finishes in person before the weekend? 3 BHK, Wakad."),
    _case("V08", "Schedule Visit", "code-mixed", "clean",
          "Sunday ko sample flat khula rehta hai kya? Hum dopahar mein aa "
          "jayenge. Baner wala project."),
    _case("V09", "Schedule Visit", "code-mixed", "clean",
          "Kal shaam paanch baje site pe mil sakte hain? Andar se flat check "
          "karna hai."),
    _case("V10", "Schedule Visit", "code-mixed", "messy",
          "Mummy papa ko bhi saath laana hai, toh, haan, weekend pe site pe aapka "
          "kab time rahega?"),

    # -------------------------- Investment --------------------------- #
    _case("I01", "Investment", "english", "clean",
          "It isn't for living in. I want something that appreciates once the "
          "metro line opens, and I'll sell in five or six years."),
    _case("I02", "Investment", "english", "clean",
          "What returns are owners getting on units near the IT park? I'd hold "
          "it and let a tenant pay for it."),
    _case("I03", "Investment", "english", "clean",
          "I've money sitting in fixed deposits at six percent. I'd rather it "
          "grew in bricks. A 1 BHK anywhere in Hinjewadi Phase 2."),
    _case("I04", "Investment", "english", "messy",
          "A shop that already has a tenant paying would be ideal — I live in "
          "Dubai, so, uh, I won't ever use it myself. Budget 60 lakh."),
    _case("I05", "Investment", "english", "clean",
          "Pre-launch pricing, then exit at possession — how much did the last "
          "phase go up between launch and handover?"),
    _case("I06", "Investment", "english", "clean",
          "My CA says I should move some money out of equities. Land near the "
          "new ring road seems the obvious place for it."),
    _case("I07", "Investment", "english", "messy",
          "I already own the flat I live in. This one would be purely for capital "
          "growth — which, um, which micro-market has appreciated most?"),
    _case("I08", "Investment", "code-mixed", "clean",
          "Rehne ke liye nahi hai. Paisa aisi jagah lagana hai jahan value badhe, "
          "metro aane ke baad bech denge."),
    _case("I09", "Investment", "code-mixed", "clean",
          "Main Singapore mein rehta hoon, wahan se use nahi karunga. Bas aage "
          "chalke achha return mile, Kharadi side."),
    _case("I10", "Investment", "code-mixed", "messy",
          "Plot le ke rakhna hai, das saal baad, matlab, jab rate double ho jaye "
          "tab bachchon ke kaam aayega."),

    # ----------------------- Request Callback ------------------------ #
    _case("C01", "Request Callback", "english", "clean",
          "I'm driving right now. Can one of your people ring me this evening "
          "after seven?"),
    _case("C02", "Request Callback", "english", "clean",
          "I'm about to go into a meeting. Please have a senior executive phone "
          "me tomorrow morning on this same number."),
    _case("C03", "Request Callback", "english", "clean",
          "My husband handles all this. He's free around lunch — get someone to "
          "phone him on the number I gave on the form."),
    _case("C04", "Request Callback", "english", "messy",
          "The network here is, sorry, I can barely hear you. Better someone "
          "dials me again when I'm home, say after eight."),
    _case("C05", "Request Callback", "english", "clean",
          "Could your manager get in touch with me before Friday? I'd rather go "
          "through the details with someone senior on the phone."),
    _case("C06", "Request Callback", "english", "clean",
          "I'm at the hospital with my father. Please try me again tomorrow, any "
          "time after ten."),
    _case("C07", "Request Callback", "english", "messy",
          "It's a bad moment — I'm, uh, about to board. Could someone from your "
          "team get back to me on this number on Monday?"),
    _case("C08", "Request Callback", "code-mixed", "clean",
          "Abhi office mein hoon, shaam ko saat baje ke baad phone kar dena "
          "please."),
    _case("C09", "Request Callback", "code-mixed", "clean",
          "Mere pati se baat kar lijiye, unka number note kar lo. Raat aath baje "
          "ke baad free rehte hain."),
    _case("C10", "Request Callback", "code-mixed", "messy",
          "Abhi gaadi chala raha hoon, toh, kal subah kisi ko bolo mujhe phone "
          "karne ko."),
)


def stated_intent_leaks() -> List[Tuple[str, str]]:
    """(case id, matched text) for every customer turn that names an intent."""
    leaks: List[Tuple[str, str]] = []
    for case in CASES:
        spoken = " ".join(
            line.split(":", 1)[1] for line in case["transcript"].splitlines()
            if line.startswith("Customer:")
        ).lower()
        for pattern in STATED_INTENT_PATTERNS:
            match = re.search(pattern, spoken)
            if match:
                leaks.append((case["id"], match.group(0)))
    return leaks


def by_intent() -> Dict[str, int]:
    counts = {label: 0 for label in INTENT_CLASSES}
    for case in CASES:
        counts[case["intent"]] += 1
    return counts
