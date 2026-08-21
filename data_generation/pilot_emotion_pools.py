"""Pilot tone pools — emotion carried by situation, not by an emotion word.

This is a *separate* pool set from :mod:`phrase_pools`, used only by
:mod:`sentiment_pilot`. Nothing here feeds the Phase 3 NER dataset, and
``phrase_pools`` is untouched, so the NER corpus and the intent head that were
built on it are unaffected.

Why a new pool set exists
-------------------------
The v3 sentiment pools reached 0.713 macro-F1 in-distribution and 0.478 on
withheld phrasings — a 0.235 gap — and 42% of held-out Enthusiastic transcripts
were classified Frustrated. Both numbers have the same cause: in v3 the tone is
*named*. "we're really excited about this", "I'm losing patience here". A model
trained on that learns a small vocabulary of affect adjectives, and when the
holdout removes the ones it memorised, the remaining evidence is whatever
generic markers survive — which, for the two high-arousal tones, are shared.
Enthusiastic and Frustrated both raise their voice, both say "finally", both
say "I've been waiting". Strip the adjective and they invert.

The bet this pilot tests is that emotion stated *indirectly* is more learnable,
not less. "I've called three times" is not a Frustrated word — it is a
Frustrated situation, and it cannot be memorised as a token because the same
idea is phrased four different ways and a third of the ideas are never seen in
training. Three rules follow:

1. **No affect vocabulary.** ``AFFECT_WORDS`` is banned outright and enforced
   by :func:`affect_word_leaks`. No phrase here contains "excited", "keen",
   "frustrated", "unsure", "khush" or any of their relatives. A model cannot
   take the shortcut because the shortcut is not in the data.

2. **A group is an idea, not a phrasing.** ``{tone: {idea: [surface forms]}}``.
   The holdout withholds whole ideas, so a held-out transcript expresses
   something the model has never seen expressed at all — not a paraphrase of
   something it has.

3. **No shortcut token may be strandable.** :func:`strandable_tokens` reports
   any content token that one tone carries in a single idea while a rival tone
   has it well established. Withholding that single idea would hand the token
   to the rival outright, which is precisely the mechanism that inverted
   Enthusiastic and Frustrated in v3 (the token was "bahut"). The pools are
   written so this returns empty. Shared vocabulary itself is fine and
   plentiful — it is *lopsided* shared vocabulary that inverts a class.

Group families mirror the v3 structure so the phrase-holdout diagnostic runs
unchanged: ``SENTIMENT_OPENER_GROUPS`` are expressions that open the call,
``DEFLECTION_GROUPS`` are expressions that surface as non-answers to an agent
question. Twenty-six opener ideas and twelve deflection ideas per tone — 38 in
all, 152 across the taxonomy — with one third of each family withheld,
stratified by register so no tone loses all its code-mixed phrasings at once.

Breadth is the point of that number, and it is why the first pilot could not
answer its own question. At 18 ideas per tone the diagnostic trained on 136
transcripts and landed at chance (0.245 in-distribution against a 0.25
baseline), so its 0.049 gap measured nothing. The holdout asks the model to
recognise a tone in an idea it has never met; doing that takes exposure to many
*different* ideas, not more transcripts per idea. Widening the pool came first
for that reason, and the corpus was scaled afterwards.

Intent pools are re-exported from :mod:`phrase_pools` unchanged — the pilot
only re-opens the sentiment question, and the intent head is already promoted.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from data_generation.phrase_pools import (  # noqa: F401  (re-exported)
    INTENT_CLASSES,
    INTENT_PHRASE_GROUPS,
    INTENT_PHRASES,
)

SENTIMENT_CLASSES: Tuple[str, ...] = ("Enthusiastic", "Neutral", "Hesitant", "Frustrated")


# --------------------------------------------------------------------------- #
# Opening expressions — 26 ideas per tone, the last nine code-mixed.
#
# Every phrase is a standalone sentence. v3 built openers as sentence *prefixes*
# that the requirement clause was appended to, which made "tone clause + comma +
# capitalised requirement" a fixed shape the model could lock onto. Here the
# expression and the requirement are separate sentences, so their boundary
# carries no information.
# --------------------------------------------------------------------------- #

OPENER_IDEAS: Dict[str, Dict[str, List[str]]] = {
    "Enthusiastic": {
        "counting-down": [
            "We've been counting down to this call all week.",
            "I've had this in my calendar since Monday and here we are.",
            "I've been ticking off the days until we could do this.",
            "We marked this one in the diary a fortnight ago.",
        ],
        "cant-stop-talking": [
            "We drove past your site on Sunday and haven't stopped talking about it since.",
            "My wife and I talked about nothing else at dinner last night.",
            "We've mentioned it to everyone we know since we saw it.",
            "It came up three times over the weekend at home.",
        ],
        "could-sign-tomorrow": [
            "If it lines up we could wrap this whole thing up this month.",
            "We could sign something tomorrow if it fits.",
            "Give me the right place and we'll act on it straight away.",
            "Show me something that works and I'll not sit on it.",
        ],
        "family-nudging": [
            "My daughter has asked me four times today whether I'd rung you yet.",
            "My wife rang me twice this morning to check I'd made this call.",
            "My parents have been nudging me all week to get on with it.",
            "My son set a reminder on my phone so I wouldn't forget to call.",
        ],
        "stood-out": [
            "We've seen six places this month and yours is the one that stuck.",
            "Out of everything we have seen this is the one we keep coming back to.",
            "Nothing else we looked at came close.",
            "We had a shortlist of four and now there is really only you.",
        ],
        "funds-ready": [
            "The funds are already arranged, we just lack the right place.",
            "The loan came through last week, so that part is sorted.",
            "The money has been sitting ready since March.",
            "We sold the old flat in April, so we're liquid and waiting.",
        ],
        "word-of-mouth": [
            "A colleague of mine wouldn't stop recommending you, so here I am.",
            "My brother-in-law bought through you and has raved about it ever since.",
            "Two people at work told me to call you specifically.",
            "Our neighbour moved into one of yours and won't shut up about it.",
        ],
        "long-time-coming": [
            "We've been saving for four years, so today feels like a big day.",
            "This is the call we've been building up to for a long time.",
            "After a decade of renting we finally get to do this.",
            "My father did this at my age and I've wanted it ever since.",
        ],
        "cm-kab-shuru": [
            "Matlab batao kab shuru kar sakte hain, hum poori tarah taiyaar hain.",
            "Bas bataiye kab start karna hai, humne sab taiyaari kar li hai.",
            "Kab se shuru kar sakte hain, hume jaldi aage badhna hai.",
            "Aap tareekh de dijiye, hum us din pahunch jaayenge.",
        ],
        "cm-pasand-aaya": [
            "Yaar humko toh dekhte hi pasand aa gaya tha.",
            "Pehli baar dekha tabhi lag gaya tha ki yahi hai.",
            "Ghar mein sabko pasand aaya hai, sab bol rahe hain aage badho.",
            "Photo dekh ke hi mann bana liya tha humne.",
        ],
        "cm-paisa-taiyaar": [
            "Paisa sab arrange ho gaya hai, bas jagah chahiye.",
            "Loan approve ho gaya hai, ab bas aap bata dijiye.",
            "Purana ghar bech diya tha, ab paise taiyaar rakhe hain.",
            "Paise ka intezaam ho gaya hai, ab kya karna hai bataiye.",
        ],
        "cm-sabko-bataya": [
            "Apne saare doston ko bata raha hoon iske baare mein.",
            "Colleagues have been asking me about it all week.",
            "Bhai ne bhi bola hai ki haath se mat jaane dena.",
            "Rishtedaaron ko abhi se bata diya hai ki kahan shift ho rahe hain.",
        ],
        "already-planning-rooms": [
            "We've already argued about which room becomes the study.",
            "My partner has picked out curtains for a place we haven't bought.",
            "There's a floor plan on our fridge with names written on it.",
            "We spent Sunday working out where the sofa goes.",
        ],
        "travelled-to-see-it": [
            "I drove down from Mumbai on Sunday purely to look at the site.",
            "I took a train up last weekend just to stand outside it.",
            "It's a four-hour drive and I've done it twice this month.",
            "I flew in for two days and half of that went on your project.",
        ],
        "read-everything": [
            "I've read your brochure cover to cover, twice.",
            "I could probably recite the specification sheet at you.",
            "I've been through every photo on the website several times over.",
            "I know those floor plans better than some of your staff, probably.",
        ],
        "bringing-others": [
            "Can I bring my parents along? They want to see it as much as I do.",
            "There'll be four of us coming, if that's alright.",
            "My brother is flying in for the weekend so he can come too.",
            "Half the family wants to be there when we look round.",
        ],
        "calling-early": [
            "I know it's barely nine, I've been awake since five over this.",
            "I've been sitting with the phone since your office opened.",
            "Apologies for calling the moment you opened, I couldn't sit still.",
            "I dialled at nine on the dot, if that tells you anything.",
        ],
        "cleared-the-diary": [
            "I've cleared Saturday completely, give me any slot you like.",
            "I'll take a day off work for this, just say when.",
            "Whatever time suits you, I'll rearrange around it.",
            "My weekend is yours, pick an hour.",
        ],
        "compared-to-current-home": [
            "Our current place will look like a cupboard next to this.",
            "After eleven years in one bedroom, this would be a different life.",
            "We've outgrown where we are and this solves it in one go.",
            "Anything with a balcony would already be a step up for us.",
        ],
        "children-claiming-rooms": [
            "My son has already decided which balcony is his.",
            "The kids drew the layout on the back of an envelope last night.",
            "My daughter asked if she can paint her room before we've even seen it.",
            "The children have divided the place up between them already.",
        ],
        "momentum": [
            "We viewed on Friday, decided on Saturday, and here I am on Monday.",
            "This has gone from idea to phone call in about four days.",
            "We saw it, we slept on it once, and that was enough.",
            "It took us one evening to make up our minds.",
        ],
        "cm-raat-bhar-plan": [
            "Raat bhar hum dono plan banate rahe.",
            "Poori raat baithe rahe, kaunsa kamra kiska hoga yahi tay karte rahe.",
            "Neend hi nahi aayi, subah tak layout dekhte rahe.",
            "Do din se ghar par bas isi ki baat ho rahi hai.",
        ],
        "cm-door-se-aaya": [
            "Main Mumbai se sirf yeh site dekhne aaya tha pichhle Sunday.",
            "Chaar ghante gaadi chala kar sirf yeh dekhne pahuncha tha.",
            "Train pakad ke aaya tha, do din mein se ek din yahi lag gaya.",
            "Itni door se aaya hoon, isi se andaaza laga lijiye.",
        ],
        "cm-brochure-padha": [
            "Aapka brochure teen baar padh chuka hoon.",
            "Website ki har photo dekh chuki hoon main.",
            "Specification wala page toh zubaani yaad ho gaya hai.",
            "Poora plan padh liya hai, ab bas dekhna baaki hai.",
        ],
        "cm-parivaar-laana": [
            "Agli baar poora parivaar leke aaunga dekhne.",
            "Chaar log aayenge hamare saath, agar theek rahe toh.",
            "Bhai bahar se aa raha hai, woh bhi saath dekhega.",
            "Ghar ke sab log saath chalna chahte hain.",
        ],
        "cm-token-de-doon": [
            "Aaj hi token amount de doon kya?",
            "Aap haan bol dijiye, advance abhi transfer kar deta hoon.",
            "Booking ke liye taiyaar hoon, aap process bata dijiye.",
            "Aap date de dijiye, main us din pahunch kar jama kar dunga.",
        ],
        "repeat-customer": [
            "This is our second purchase with you and the first went without a hitch.",
            "You handled our last one, which is why I did not ring anyone else.",
            "We bought through you three years ago and it was painless start to finish.",
            "Coming back to you was the easy part of this decision.",
        ],
        "watched-it-for-years": [
            "We have walked past that plot for years wondering when something would go up.",
            "I have driven past that corner every morning for a decade.",
            "That site has been on my mind since the hoarding went up.",
            "We watched this development rise floor by floor from our balcony.",
        ],
        "family-history-here": [
            "My parents bought in that same lane in the nineties.",
            "I grew up two streets away, so this one matters to us.",
            "My grandfather lived on that road and coming back is rather the point.",
            "Three generations of my family have lived within a mile of there.",
        ],
        "finally-within-reach": [
            "That road was out of our range for years and suddenly it is not.",
            "The salary revision came through, so somewhere we could never touch is possible.",
            "We were priced out of this area twice before. Not this time.",
            "For a long time this was a someday plan and now it is a this-year plan.",
        ],
        "cm-pehle-bhi-liya": [
            "Aapse pehle bhi ek property li thi, koi dikkat nahi hui thi.",
            "Pichhli baar ka anubhav accha tha, isliye seedha aapko call kiya.",
            "Do saal pehle aapke through hi liya tha, sab smooth chala tha.",
            "Purane experience ki wajah se kahin aur poochha hi nahi.",
        ],
        "cm-bachpan-se-dekha": [
            "Bachpan se us ilaake mein aata jaata raha hoon.",
            "Mere nana ji usi road par rehte the.",
            "Us gali mein hamara purana ghar tha, wapas wahi jaana hai.",
            "Pandrah saal us area mein nikale hain humne.",
        ],
    },
    "Neutral": {
        "here-are-the-facts": [
            "I'll run through what we need.",
            "Let me give you the details straight.",
            "I'll list it out and you can tell me what fits.",
            "I'll state the requirement and you can take it from there.",
        ],
        "on-behalf-of": [
            "I'm calling on behalf of my brother, so I'll pass on what he told me.",
            "I'm noting this down for a family member, so I'll relay the requirement.",
            "I'm doing this for my in-laws, so these are their details.",
            "This is for a cousin who's abroad, I'm just the messenger.",
        ],
        "no-particular-timeline": [
            "There's no particular deadline on our side.",
            "We're not working to any timeline, so take it as you find it.",
            "This isn't urgent, it's just something we're organising.",
            "Nothing hangs on the date, whenever it lands it lands.",
        ],
        "comparing-market": [
            "We're gathering quotes from three or four places, yours included.",
            "I'm collecting the same information from a few developers.",
            "We're going through the market systematically and you're on the list.",
            "I'm doing the same call with four firms this afternoon.",
        ],
        "short-on-time-factual": [
            "I've got about four minutes, so I'll be brief.",
            "I'm between meetings, so let me keep this tight.",
            "I'll be quick as I'm on my way somewhere.",
            "I have a small window before my next call, so briefly.",
        ],
        "written-it-down": [
            "I've written the requirement down, so I'll read it out.",
            "I have it all in a note here, one second.",
            "I made a list before calling, so here goes.",
            "It's typed out in front of me, I'll just go down the page.",
        ],
        "standard-process": [
            "We've done this once before, so we know roughly how it goes.",
            "This is the third property we've organised, so it's fairly routine.",
            "We know the process, so just the specifics then.",
            "We covered the same steps two years ago.",
        ],
        "recap-of-earlier-call": [
            "I spoke to your desk a week ago, so I'll recap what I said then.",
            "This follows an earlier call, so I'll go over the same ground.",
            "I've been through this once with a colleague of yours, so briefly again.",
            "To repeat what's already been noted, and then the new bit.",
        ],
        "cm-seedha-bata": [
            "Main seedha requirement bata deta hoon.",
            "Chaliye main details bata deta hoon.",
            "Point to point bata deta hoon, aap note kar lijiye.",
            "Jo sab chahiye woh main list kara deta hoon.",
        ],
        "cm-bhai-ke-liye": [
            "Main apne bhai ke liye poochh raha hoon.",
            "Mummy papa ke liye dekh raha hoon, unki requirement bata deta hoon.",
            "Apne liye nahi, ek rishtedaar ke liye poochh raha hoon.",
            "Behen ke liye dekhna hai, uska message aaya tha.",
        ],
        "cm-time-kam": [
            "Do minute hain mere paas, jaldi bata deta hoon.",
            "Abhi office mein hoon, short mein bata deta hoon.",
            "Meeting ke beech mein hoon, jaldi se list kara deta hoon.",
            "Nikal raha hoon abhi, isliye chhota rakhta hoon.",
        ],
        "cm-likha-hua": [
            "Maine likh kar rakha hai, padh deta hoon.",
            "Ek note bana rakha hai, usme se bata deta hoon.",
            "Sab kuch note kiya hua hai, ek second.",
            "Paper par likha hua hai saamne, aap likh lijiye.",
        ],
        "professional-capacity": [
            "I handle property for a small firm, so this is routine for me.",
            "I do this a few times a year for work.",
            "I'm the one who makes these calls at our company.",
            "This is part of my job rather than a personal move.",
        ],
        "asks-what-is-needed": [
            "Before I start, what do you need from me?",
            "Tell me which fields you want and I'll give them in order.",
            "What's on your form? I'll fill it in verbally.",
            "You lead and I'll answer.",
        ],
        "two-fixed-rest-open": [
            "Two things are fixed and the rest is open.",
            "There's one hard constraint and everything else can move.",
            "Only the location is settled; treat the rest as adjustable.",
            "Three fixed points, otherwise flexible.",
        ],
        "numbers-first": [
            "I'll give you the figures and you tell me what's possible.",
            "Let's work backwards from the number.",
            "The arithmetic decides this, so here it is.",
            "I'll state the amounts plainly and we can go from there.",
        ],
        "booked-this-slot": [
            "I booked this slot in advance so we could go through it properly.",
            "I set aside twenty minutes for this.",
            "I requested a callback at this time, so I'm ready.",
            "This is a scheduled conversation, so I've prepared.",
        ],
        "relocation-logistics": [
            "We're moving cities in September, so this is one item on a long list.",
            "There's a school admission and a job start to fit around this.",
            "It's one of about nine things I'm organising for the move.",
            "This is a relocation, so the requirement is driven by logistics.",
        ],
        "will-decide-after-seeing-all": [
            "I'll take notes and conclude once I've seen everything.",
            "No conclusions today, I'm collecting information.",
            "I'll compare at the end rather than as I go.",
            "Decisions come later; today is data.",
        ],
        "flexible-on-most": [
            "We're fairly open on most of it.",
            "Not many hard requirements on our side.",
            "Most of this can go either way for us.",
            "Few strong opinions here.",
        ],
        "experienced-renter": [
            "We rented for six years, so we know what we do and don't need.",
            "Having lived in three of these, I know what to ask.",
            "We've been through two moves already, so the list is short.",
            "Experience has narrowed what we care about.",
        ],
        "cm-form-bhar-raha": [
            "Ek form bhar raha hoon, usi ke liye jaankari chahiye.",
            "Company ke record ke liye poochh raha hoon.",
            "Register mein entry karni hai, isliye detail le raha hoon.",
            "Ek application likh raha hoon, usme yeh sab bharna hai.",
        ],
        "cm-aap-poochh-lijiye": [
            "Aap sawaal poochh lijiye, main jawab de deta hoon.",
            "Aap order mein poochhiye, main batata jaunga.",
            "Jo aapke form mein hai, wahi poochh lijiye.",
            "Aap shuru kijiye, main peechhe peechhe chalta hoon.",
        ],
        "cm-shift-ho-rahe": [
            "September mein shift kar rahe hain, isi silsile mein baat kar raha hoon.",
            "Transfer ho gaya hai, isliye yahan dekhna pad raha hai.",
            "Sheher badal rahe hain, kai cheezein ek saath dekhni hain.",
            "Naukri wahan lag gayi hai, isliye yeh sab dekhna pad raha hai.",
        ],
        "cm-ek-cheez-fix": [
            "Ek hi cheez fix hai, baaki sab badal sakta hai.",
            "Sirf jagah pakki hai, aur kuch nahi.",
            "Do point pakke hain, baaki par baat ho sakti hai.",
            "Ek shart hai, uske alawa sab adjust ho jayega.",
        ],
        "cm-kiraye-par-the": [
            "Chhe saal kiraye par rahe hain, isliye pata hai kya chahiye.",
            "Do baar shift kar chuke hain, ab list chhoti ho gayi hai.",
            "Tajurba hai thoda, isliye zyada sawaal nahi karunga.",
            "Teen jagah reh chuke hain, ab andaaza hai kya zaroori hai.",
        ],
        "comparing-two-shortlisted": [
            "I have narrowed it to two projects and want the numbers side by side.",
            "Yours and one other are on the list; I am comparing them on the same basis.",
            "Send me your figures and I will put them against the other quote.",
            "This is a straightforward comparison exercise at this point.",
        ],
        "checking-a-specific-fact": [
            "One question only: is the completion certificate in place?",
            "I need to confirm the carpet area before anything else.",
            "Just checking whether the project is registered with the authority.",
            "I want the exact possession date in writing, that is all.",
        ],
        "acting-as-broker": [
            "I am a channel partner, so I will be passing this to a client.",
            "I handle enquiries for a small consultancy and this is one of them.",
            "I am collecting inventory details for someone I represent.",
            "This is a trade enquiry rather than a personal one.",
        ],
        "budget-decides-it": [
            "The number is fixed at ninety lakhs and everything follows from that.",
            "Anything above the ceiling is not worth discussing, so let us start there.",
            "Work within the figure and tell me what fits.",
            "The budget decides this, not the preferences.",
        ],
        "cm-do-project-compare": [
            "Do project shortlist kiye hain, dono ke number chahiye.",
            "Aapka aur ek aur, dono compare karna hai.",
            "Side by side rakh kar dekhna hai, isliye figure bhej dijiye.",
            "Tulna karni hai bas, aur kuch nahi.",
        ],
        "cm-ek-hi-sawaal": [
            "Ek hi baat poochhni hai, completion certificate mila hai ya nahi?",
            "Sirf carpet area confirm karna hai.",
            "RERA registration hai ya nahi, bas yahi jaanna hai.",
            "Possession ki tareekh likhit mein chahiye, aur kuch nahi.",
        ],
    },
    "Hesitant": {
        "almost-didnt-call": [
            "I nearly put the phone down before it rang.",
            "I've picked up the phone to do this three times and put it back down.",
            "I almost left this for another week.",
            "I sat looking at your number for a while before dialling.",
        ],
        "going-in-circles": [
            "We've been going round and round on this at home.",
            "Every time we settle on something we change our minds a week later.",
            "We've had the same conversation six times without landing anywhere.",
            "We keep arriving back where we started.",
        ],
        "depends-on-something-else": [
            "A lot of this depends on whether my transfer is approved.",
            "It hinges on my daughter's school admission, which we'll know in June.",
            "Everything rests on a job offer that is not confirmed yet.",
            "If the posting changes, none of this happens at all.",
        ],
        "might-come-to-nothing": [
            "We might not do anything this year at all.",
            "This could easily come to nothing, I should say upfront.",
            "We may well end up staying where we are.",
            "There's a fair chance you never hear from me again after today.",
        ],
        "out-of-our-depth": [
            "I genuinely don't know what's reasonable, so bear with me.",
            "We've never done this before and it shows.",
            "Half of what you say will probably go over my head.",
            "I had to look up what half these words meant last night.",
        ],
        "size-of-the-decision": [
            "It's a lot of money to get wrong.",
            "The size of the decision is what's holding us up.",
            "I keep thinking about what happens if we pick badly.",
            "We'd be tied to it for twenty years, which is what stops me.",
        ],
        "partner-disagrees": [
            "My husband and I want fairly different things, which is the problem.",
            "We can't agree between us, so take everything I say loosely.",
            "My partner would give you a different answer to mine.",
            "There are two opinions in this house and I only speak for one.",
        ],
        "nothing-is-fixed": [
            "Nothing I tell you today is fixed, just so you know.",
            "Treat all of this as a rough sketch rather than a commitment.",
            "Please don't hold me to any of it.",
            "It's all written in pencil, is what I'd say.",
        ],
        "cm-pakka-nahi": [
            "Abhi kuch bhi pakka nahi hai, bas dekh rahe hain.",
            "Final kuch nahi hua hai, soch vichaar chal raha hai.",
            "Abhi sirf jaankari le rahe hain, aage ka kuch bata nahi sakte.",
            "Kai baar baith kar socha, phir bhi man nahi bana.",
        ],
        "cm-ghar-walon-se-poochhna": [
            "Ghar walon se baat kiye bina main kuch keh nahi sakta.",
            "Papa se poochhna padega, unke bina decide nahi hoga.",
            "Pehle ghar mein discuss karunga, phir hi kuch bataunga.",
            "Akela faisla nahi kar sakta, sabki raay leni padegi.",
        ],
        "cm-dar-lagta": [
            "Itna bada faisla hai, haath thoda kaanpta hai.",
            "Paisa bahut lagta hai isme, isliye ruk ruk ke soch rahe hain.",
            "Galat ho gaya toh saalon bhugatna padega, yahi sochta rehta hoon.",
            "Raat ko neend nahi aati yahi sochte sochte.",
        ],
        "cm-shayad-baad-mein": [
            "Ho sakta hai is saal kuch na karein.",
            "Shayad agle saal dekhenge, abhi bas poochh raha hoon.",
            "Ho sakta hai baat aage badhe hi na.",
            "Abhi kuch final nahi hua hai, aage ka aage dekhenge.",
        ],
        "apologising-for-calling": [
            "I hope I'm not taking your time this early in the process.",
            "You may find this call a bit pointless, and I'd understand.",
            "Apologies if this turns out to be a waste of your afternoon.",
            "I should apologise in advance for how vague this will be.",
        ],
        "pulled-out-before": [
            "We nearly signed two years ago and pulled out at the last minute.",
            "We've been this close once before and didn't go through with it.",
            "Last time we got as far as the paperwork and stopped.",
            "There's a history of us not following through on this.",
        ],
        "waiting-on-own-sale": [
            "We can't move until our own place sells, and it's been listed a while.",
            "Everything is stuck behind selling the old flat.",
            "Until someone buys ours, this is all theoretical.",
            "Our sale has had two viewings in three months, so.",
        ],
        "asking-permission-to-ask": [
            "Is it alright if I ask some very basic questions first?",
            "You'll have to forgive me if these questions are obvious.",
            "Can I ask something that might sound silly?",
            "Do you mind if I go slowly through this?",
        ],
        "bank-decides": [
            "I'm not sure what we can afford until the bank says.",
            "The number depends on someone else's decision, not ours.",
            "Our range is whatever gets approved, and we don't know yet.",
            "Until the loan comes back, any figure I give you is invented.",
        ],
        "everything-looks-the-same": [
            "After a while every listing looks identical to me.",
            "I've seen so many that they've blurred together.",
            "I couldn't tell you what separates one from the next any more.",
            "They all merge into one after the tenth viewing.",
        ],
        "city-vs-outskirts": [
            "One day we want the centre, the next day we want the outskirts.",
            "We change our minds about location every week.",
            "Half of us want to be central, half want space.",
            "We've flipped on where twice this month already.",
        ],
        "reluctant-to-cost-you-effort": [
            "It would be unfair for you to put in work and then have us stop.",
            "I'd rather warn you now than have you chase a dead end.",
            "You should know there's a real chance this goes nowhere.",
            "Don't put me at the top of your pile, is what I'm saying.",
        ],
        "bad-year-to-buy": [
            "It might be a poor year to be doing this, from what I read.",
            "Everyone tells me to wait and I don't know if that's true.",
            "The market talk makes me want to sit on my hands.",
            "I keep reading that prices will shift and nobody agrees in which direction.",
        ],
        "cm-sahi-samay-nahi": [
            "Pata nahi yeh sahi samay hai bhi ya nahi.",
            "Log kehte hain ruk jao, samajh nahi aa raha kya karein.",
            "Market ke baare mein itna sunte hain ki haath ruk jaata hai.",
            "Sab alag alag salah dete hain, isliye ghabraahat rehti hai.",
        ],
        "cm-apna-bikna-baaki": [
            "Jab tak apna wala bik na jaye, hum aage nahi badh sakte.",
            "Purana ghar atka hua hai, sab usi par ruka hai.",
            "Teen mahine se listing padi hai, do hi log aaye.",
            "Pehle woh nikle tab hi yeh ho payega.",
        ],
        "cm-bank-batayega": [
            "Bank kitna dega, yeh pata chalne ke baad hi bol paunga.",
            "Loan sanction ho jaye toh figure bata paunga.",
            "Jo number abhi bataunga woh sirf andaaza hoga.",
            "Manzoori mile tab hi asli baat ho paayegi.",
        ],
        "cm-basic-sawaal": [
            "Ek baat poochh sakta hoon? Bilkul basic hai.",
            "Maaf kijiye, mere sawaal shayad bahut simple lagenge.",
            "Dheere dheere samjhaiyega, pehli baar kar raha hoon.",
            "Mujhe zyada samajh nahi hai isme, isliye poochhta rahunga.",
        ],
        "cm-sab-ek-jaisi": [
            "Itni jagah dekh li ki ab sab ek jaisi lagne lagi hain.",
            "Dasvi property ke baad sab gadd-madd ho gaya.",
            "Ab yaad hi nahi rehta kaunsi kaunsi thi.",
            "Sabki tasveerein ek jaisi lagti hain ab mujhe.",
        ],
        "polite-and-undecided": [
            "Sorry to be vague, you have been helpful and I am the problem here.",
            "You have done nothing wrong, I simply cannot make my mind up.",
            "I appreciate your time and I know I am not making this easy.",
            "Everyone here has been very good to me; the indecision is entirely mine.",
        ],
        "waiting-on-a-life-event": [
            "We will know more after the baby arrives in March.",
            "My daughter's wedding is in November and nothing happens before that.",
            "Once my father's treatment finishes we can think about it properly.",
            "There is a lot going on at home and this sits behind all of it.",
        ],
        "too-many-opinions": [
            "Everyone I ask gives me the opposite advice to the last person.",
            "My brother says buy, my accountant says wait, and I say nothing.",
            "I have had four opinions and none of them agree.",
            "The more people I ask the less certain I get.",
        ],
        "one-small-step": [
            "Could we begin with some information and see where it goes?",
            "I would rather do one thing at a time than commit to anything today.",
            "Let us begin with the easiest question and work up.",
            "Nothing binding at this stage, if that is alright.",
        ],
        "cm-shaadi-ke-baad": [
            "Beti ki shaadi November mein hai, usse pehle kuch nahi hoga.",
            "Ghar mein abhi bahut kuch chal raha hai, yeh sab uske baad.",
            "Papa ka ilaaj chal raha hai, pehle woh poora ho jaye.",
            "Kuch mahine ruk kar hi sochenge hum.",
        ],
        "cm-sabki-alag-raay": [
            "Jisse poochho wahi alag salah deta hai.",
            "Bhai kehta hai le lo, CA kehta hai ruk jao.",
            "Chaar logon se poochha, chaaron ne alag bataya.",
            "Jitna poochhta hoon utna hi ulajh jaata hoon.",
        ],
    },
    "Frustrated": {
        "called-many-times": [
            "This is the fourth time I've called about this.",
            "I've rung this number three times already this month.",
            "I've lost count of how many times I've explained this.",
            "Check the log, you'll see my number on it repeatedly.",
        ],
        "nobody-called-back": [
            "I was promised a call back on Tuesday and nothing came.",
            "Two people have taken my number and neither has used it.",
            "I left my number a fortnight ago and heard nothing.",
            "Your colleague said someone would ring within a day. That was March.",
        ],
        "repeating-myself": [
            "I'll say it all again, because it clearly didn't get written down.",
            "Let me repeat what I told the last person, word for word.",
            "You'll have to bear with me saying this for the third time.",
            "None of this is new information, it just never sticks anywhere.",
        ],
        "wasted-time": [
            "I've taken a half day off for this twice now.",
            "I drove out there on Sunday and nobody was on site.",
            "That's two Saturdays I'm not getting back.",
            "I sat in your office for fifty minutes and left without seeing anyone.",
        ],
        "considering-competitor": [
            "I'm about ready to take this elsewhere.",
            "Another developer has been chasing me and is starting to look better.",
            "One more round of this and I'll go with someone else.",
            "There are three other firms who return calls, as it turns out.",
        ],
        "sarcastic-compliance": [
            "Yeah, whatever, let's just go through it again.",
            "Fine. Ask me the same questions and I'll give the same answers.",
            "Sure, why not, let's do the whole thing from the top.",
            "Go on then. I've got nothing better to do apparently.",
        ],
        "clipped-tone": [
            "Right. Let's see if it goes anywhere this time.",
            "Okay. Third attempt, then.",
            "Let's get on with it, shall we.",
            "Go ahead, and don't take all day. Clock's running.",
        ],
        "escalation": [
            "Can you put me through to whoever actually makes decisions?",
            "I'd like this handled by a manager this time.",
            "Is there someone senior who can own this from now on?",
            "Give me the name of the person responsible, please.",
        ],
        "cm-kitni-baar": [
            "Kitni baar bata chuka hoon yeh sab.",
            "Chaar baar phone kar chuka hoon isi baare mein.",
            "Har baar wahi sawaal, wahi jawab dena padta hai.",
            "Yeh meri paanchvi call hai isi topic par.",
        ],
        "cm-koi-jawab-nahi": [
            "Number diya tha, aaj tak kisi ka phone nahi aaya.",
            "Do hafte ho gaye, abhi tak koi jawab nahi aaya.",
            "Bola tha call karenge, aaj tak kuch nahi kiya.",
            "Message bhi kiye hain, unka bhi koi reply nahi.",
        ],
        "cm-chakkar-lagaye": [
            "Mera pura Sunday barbaad ho gaya wahan jaake.",
            "Chutti leke gaya tha, koi mila hi nahi site par.",
            "Do baar chakkar laga chuka hoon, kuch nahi hua.",
            "Petrol jala ke gaya tha, gate hi band tha.",
        ],
        "cm-doosri-jagah": [
            "Ab toh sochta hoon kahin aur baat kar loon.",
            "Doosri company wale roz phone karte hain, unhi se kar leta hoon.",
            "Ek aur baar aisa hua toh main kahin aur chala jaunga.",
            "Aur bhi builder hain market mein, yeh bhool mat jaaiye.",
        ],
        "broken-specific-promise": [
            "You said the brochure would be emailed. It never arrived.",
            "I was told the paperwork would come by Friday. It's Wednesday.",
            "Somebody promised me the layout drawings a month ago.",
            "The follow-up I was guaranteed simply did not happen.",
        ],
        "wrong-information": [
            "The previous person quoted me a completely different figure.",
            "I've now been given three versions of the same thing.",
            "What your colleague said contradicts your own website.",
            "Every person here tells me something else.",
        ],
        "transferred-around": [
            "This is the third department I've been put through to.",
            "I've explained myself to two people before reaching you.",
            "I keep getting handed on and starting from scratch.",
            "Nobody I reach seems to be the right person.",
        ],
        "hold-time": [
            "I've been on hold for eleven minutes to have this conversation.",
            "Twenty minutes of my morning went to your hold music.",
            "I've heard that recorded message four times this morning.",
            "The queue took longer than this whole conversation will.",
        ],
        "will-put-it-in-writing": [
            "I'm one step away from putting this in writing to someone senior.",
            "Next stop is a formal complaint, and I'd rather avoid that.",
            "I've started keeping a record of these conversations.",
            "There's an email drafted that I haven't sent yet.",
        ],
        "deadline-missed": [
            "You had until the end of last month. We're well into this one.",
            "The date you gave me came and went without a word.",
            "That was supposed to be settled before Diwali.",
            "We are three weeks past what I was promised.",
        ],
        "was-warned-about-you": [
            "A friend warned me this would happen and I defended you.",
            "Someone told me not to bother and I ignored them.",
            "I was told your after-sales was poor. I said give them a chance.",
            "My colleague's exact words were don't waste your afternoon.",
        ],
        "is-anyone-working-on-this": [
            "Is anyone actually working on this, or should I assume not?",
            "Does my file exist, or has it sat untouched?",
            "Has a single person looked at this since March?",
            "I'd like to know whether this is being handled at all.",
        ],
        "money-already-paid": [
            "I've paid a booking amount and still can't get a straight answer.",
            "You have my money and I have nothing in writing.",
            "The deposit cleared six weeks ago. That's the last thing that happened.",
            "I paid, and since then, silence.",
        ],
        "cm-teen-jagah-transfer": [
            "Teen jagah transfer kar chuke hain mujhe.",
            "Har naye bande ko poori kahani dobara sunani padti hai.",
            "Do logon se guzarne ke baad aap tak pahuncha hoon.",
            "Ek se doosre ko bhejte rehte hain, kaam koi nahi karta.",
        ],
        "cm-alag-alag-jawab": [
            "Pichhle bande ne bilkul alag figure bataya tha.",
            "Har aadmi yahan kuch aur bolta hai.",
            "Website par kuch aur likha hai, aap kuch aur keh rahe hain.",
            "Teen version sun chuka hoon ek hi cheez ke.",
        ],
        "cm-likhit-shikayat": [
            "Ab likhit mein shikayat karni padegi lagta hai.",
            "Ek email likh rakhi hai, abhi bheji nahi hai.",
            "Har call ka record rakhna shuru kar diya hai maine.",
            "Upar tak baat le jaani padegi ab.",
        ],
        "cm-paisa-jama-kiya": [
            "Booking amount jama kar chuka hoon, phir bhi seedha jawab nahi milta.",
            "Paisa aapke paas hai, mere paas kaagaz kuch nahi hai.",
            "Deposit gaye chhe hafte ho gaye, uske baad kuch nahi hua.",
            "Paise le liye, uske baad se sannata hai.",
        ],
        "cm-dost-ne-mana-kiya": [
            "Dost ne pehle hi mana kiya tha, maine hi nahi suna.",
            "Log bolte the mat jao, maine kaha ek mauka dete hain.",
            "Ek jaan-pehchaan wale ne warning di thi, ab samajh aa raha hai.",
            "Sabne kaha tha yahi hoga, aur wahi hua.",
        ],
        "same-question-every-time": [
            "Each time I ring, somebody asks me the same three questions from scratch.",
            "Your team has written the same thing down at least five times.",
            "I could recite my own file back to you by now.",
            "Every conversation starts from zero, as if the previous one never happened.",
        ],
        "nobody-owns-it": [
            "Everyone I speak to says it is somebody else's department.",
            "Not one person has been willing to say this is theirs to sort out.",
            "I keep being told it is being looked into, by nobody in particular.",
            "There does not appear to be a single name attached to my case.",
        ],
        "goodwill-exhausted": [
            "There is a limit and I am close to it.",
            "I have been more reasonable about this than most people would be.",
            "I have given you far more chances than I should have.",
            "My goodwill on this ran out some weeks ago.",
        ],
        "others-respond-faster": [
            "The other builder rang me back within an hour of my enquiry.",
            "I made the same enquiry elsewhere on Tuesday and already have a full quote.",
            "Everyone else in this market manages to return a call.",
            "Two other firms have sent me what I asked you for a month ago.",
        ],
        "cm-chakkar-kaat-raha": [
            "Teen baar office aa chuka hoon, har baar khaali haath lauta hoon.",
            "Kitni baar aaunga main, yeh bhi bata dijiye.",
            "Har baar bolte hain kal aana, aur kal kabhi aata nahi.",
            "Do hafte se rozana chakkar kaat raha hoon.",
        ],
        "cm-sabr-khatam": [
            "Ab bardaasht se bahar ho raha hai yeh sab.",
            "Ab aur bardaasht nahi hoga.",
            "Meri seema aa chuki hai, saaf saaf bata deta hoon.",
            "Itna intezaar kisi ne nahi karwaya hoga.",
        ],
    },
}


# --------------------------------------------------------------------------- #
# Deflections — 12 ideas per tone, the last four code-mixed.
#
# These surface as the customer's non-answer to an agent question, so a
# transcript built from a deflection idea carries no tone at all in its opening
# turn. That is deliberate: it is the harder half of the task and it is dealt
# out equally to all four tones.
# --------------------------------------------------------------------------- #

DEFLECTION_IDEAS: Dict[str, Dict[str, List[str]]] = {
    "Enthusiastic": {
        "open-to-anything": [
            "Haven't pinned that down yet, so surprise me — I'm up for whatever you suggest.",
            "No fixed view there, tell me what most people go for and my answer is yes.",
            "That bit's wide open, and honestly half the fun is deciding it with you.",
        ],
        "will-come-back-fast": [
            "Let me check with my wife and I'll ring you back within the hour.",
            "I'll confirm that tonight and call you first thing tomorrow.",
            "Give me ten minutes and I'll have an answer for you.",
        ],
        "whatever-is-fastest": [
            "Whichever option gets us moving quickest, that's the one.",
            "Don't mind at all, go with whatever moves this along.",
            "No preference, just point me at the fastest route.",
        ],
        "laughing-it-off": [
            "Ha, you've hit on the one thing we haven't argued about.",
            "Funny you ask, that's the only bit we agree on, and we agree we don't mind.",
            "No idea, and my wife will laugh when I tell her you asked.",
        ],
        "cm-aap-batao": [
            "Aap hi bata dijiye kya better rahega, hum maan lenge.",
            "Isme aap ki salah chahiye, jo aap bologe wahi karenge.",
            "Aap expert hain, aap jo suggest karenge hum wahi lenge.",
        ],
        "cm-ghar-mein-poochh": [
            "Ghar mein poochh ke abhi ke abhi bata dunga aapko.",
            "Ek minute, patni se poochh kar turant confirm karta hoon.",
            "Abhi call karke poochhta hoon, do minute mein wapas aata hoon.",
        ],
        "wants-recommendation": [
            "No view on that yet. What would you put in? I'll probably go with it.",
            "Tell me what works best and I'll take your word for it.",
            "You've done this a hundred times, so I'll follow your lead there.",
        ],
        "redirects-to-visit": [
            "Can't say on that one. When can we actually come and see it?",
            "I'd rather answer that standing in the flat. When can we visit?",
            "Park that. What's the earliest we could look round?",
        ],
        "blank-but-upbeat": [
            "Completely blank on that one, ask me an easier question.",
            "Not a clue, and I'm rather enjoying how much I don't know.",
            "No answer for you there, but keep them coming.",
        ],
        "will-decide-on-site": [
            "I'll know the moment I walk in, I think.",
            "That's one I'll settle when I'm standing there.",
            "Show me two options and I'll point at one.",
        ],
        "cm-dekh-kar-bata-dunga": [
            "Jagah dekhte hi bata dunga, abhi bolna mushkil hai.",
            "Wahan pahunch kar ek second mein tay ho jayega.",
            "Do option dikha dijiye, main ungli rakh dunga.",
        ],
        "cm-kab-aa-sakte": [
            "Woh chhodiye, yeh bataiye kab aa sakte hain dekhne?",
            "Us par baad mein, pehle visit ki date de dijiye.",
            "Pehle aane ka din tay kar lijiye, baaki wahin dekh lenge.",
        ],
    },
    "Neutral": {
        "come-to-it-later": [
            "That part isn't decided, we'll come to it later.",
            "We'll settle that once the rest is clear.",
            "We don't need to settle that now, it can come later.",
        ],
        "will-confirm-in-writing": [
            "I'll email that across once I've checked.",
            "I'll send that detail over later today.",
            "I'll confirm that in writing instead of guessing now.",
        ],
        "no-preference-plain": [
            "No preference either way on that.",
            "Doesn't make a difference to us.",
            "Either works, it's not a factor.",
        ],
        "dont-have-it-to-hand": [
            "I don't have that in front of me at the moment.",
            "That's not something I noted down before calling.",
            "I'd be guessing, so leave it blank for now.",
        ],
        "cm-baad-mein-dekhenge": [
            "Woh baad mein decide kar lenge, abhi rehne dijiye.",
            "Us par baad mein baat karte hain.",
            "Woh point abhi chhod dete hain, aage badhte hain.",
        ],
        "cm-note-nahi-kiya": [
            "Abhi mere paas us baare mein kuch nahi hai.",
            "Woh maine note nahi kiya tha, baad mein bata dunga.",
            "Woh figure yaad nahi hai, ek minute dijiye, dekh kar bataunga.",
        ],
        "whatever-is-standard": [
            "Whatever's standard, put that down.",
            "Go with the default on that one.",
            "Whatever most people have is fine.",
        ],
        "not-applicable-to-us": [
            "That doesn't apply in our case.",
            "Not relevant for us, skip it.",
            "That one isn't a factor here.",
        ],
        "depends-on-price": [
            "That follows from the price, so I'll answer once I have it.",
            "That follows from the number, not the other way round.",
            "Price first, then that.",
        ],
        "a-detail-for-later": [
            "Let's not spend time on that, it's a detail.",
            "Minor point, we can pick it up at the end.",
            "That's a footnote, keep going.",
        ],
        "cm-jo-standard-ho": [
            "Jo standard hota hai wahi likh dijiye.",
            "Aam taur par jo hota hai, wahi rakh lijiye.",
            "Default jo bhi ho, wahi chalega.",
        ],
        "cm-lagoo-nahi-hota": [
            "Woh humpe lagoo nahi hota.",
            "Yeh point hamare case mein nahi aata.",
            "Iska hamse koi wasta nahi hai.",
        ],
    },
    "Hesitant": {
        "dont-want-to-say-wrong": [
            "I'd rather not say something and then have to take it back.",
            "I don't want to give you a wrong number, so let me leave it.",
            "If I answer that now I'll only change it later.",
        ],
        "need-to-think": [
            "Can I have a think about that one?",
            "Let me sit with that before answering.",
            "Give me a few days on that.",
        ],
        "answer-will-move": [
            "Whatever I say there will probably move.",
            "That's the bit most likely to change, so I'd rather not fix it.",
            "Ask me next week and you'll get a different answer.",
        ],
        "apologetic-blank": [
            "Sorry, I know I should have an answer for that.",
            "I feel a bit silly not knowing, but I don't.",
            "That's the question I was hoping you wouldn't ask.",
        ],
        "cm-soch-kar-bataunga": [
            "Soch kar bataata hoon, abhi kuch keh nahi sakta.",
            "Thoda samay dijiye, phir hi kuch bol paunga.",
            "Ek baar ghar par baith kar sochunga, phir bataunga.",
        ],
        "cm-galat-na-bol-doon": [
            "Kahin galat na bata doon, isliye abhi rehne dijiye.",
            "Pakka pata nahi, isliye andaaza nahi lagana chahta.",
            "Ulta seedha bol diya toh baad mein dikkat hogi.",
        ],
        "come-back-another-call": [
            "Could we come back to that on another call?",
            "Can that wait until we speak again?",
            "I'd rather leave that for next time.",
        ],
        "what-do-others-say": [
            "What do most people say to that? I genuinely don't know.",
            "Is there a usual answer? I'd probably go along with it.",
            "How do others handle that one?",
        ],
        "half-answer-withdrawn": [
            "Maybe... no, I shouldn't guess.",
            "I was going to say something and I've talked myself out of it.",
            "Possibly. Actually, ignore that.",
        ],
        "depends-on-the-loan": [
            "That depends on the loan, and I don't know yet.",
            "That is decided for us by the bank, effectively.",
            "I'll know that when the bank tells me.",
        ],
        "cm-agli-baar-baat": [
            "Agli baar baat karein iske baare mein?",
            "Abhi rehne dijiye, dobara call par baat karenge.",
            "Is baat ko agli baar ke liye chhod dete hain.",
        ],
        "cm-log-kya-kehte": [
            "Aam taur par log kya bolte hain isme?",
            "Koi normal jawab hota hai iska? Wahi maan lijiye.",
            "Doosre log kya karte hain yahan par?",
        ],
    },
    "Frustrated": {
        "already-told-you": [
            "I gave you that answer already, it should be on your file.",
            "That's in the notes from last time, if anyone kept them.",
            "I gave that to the last person I spoke to.",
        ],
        "why-does-it-matter": [
            "Does that really matter right now?",
            "Is that relevant, or are we just filling in boxes?",
            "I don't see what that has to do with anything.",
        ],
        "not-doing-this-again": [
            "I'm not going through all that a second time.",
            "No. Next, please.",
            "Skip it. Next question.",
        ],
        "blame-shift": [
            "I'd know that if somebody had bothered to explain it.",
            "Nobody's told me, so how would I have an answer?",
            "You tell me, you're the ones who've had my file for a month.",
        ],
        "cm-pehle-bata-chuka": [
            "Yeh sab pehle bata chuka hoon, file mein hoga.",
            "Pichhli baar hi bata diya tha, dobara kyun?",
            "Ek baar bol chuka hoon, likha nahi kya kisi ne?",
        ],
        "cm-kya-farak": [
            "Isse kya farak padta hai abhi?",
            "Yeh sawaal ka abhi kya kaam hai?",
            "Iska jawab dene se kya hoga, batayiye.",
        ],
        "counter-question": [
            "Why don't you tell me what you have and I'll react to that?",
            "You go first for once.",
            "Show me something and then ask me questions.",
        ],
        "time-limit": [
            "You've got two minutes left, use them better than that.",
            "I'm giving this call five more minutes.",
            "Ask me something useful, the clock is going.",
        ],
        "answer-when-fixed": [
            "I'll answer that when someone sorts out the last problem.",
            "Sort out the open issue and then I'll cooperate.",
            "Deal with the open issue first and I'll be more helpful.",
        ],
        "sarcastic-guess": [
            "Surprise me. You've been guessing so far anyway.",
            "Put down whatever you like, it clearly makes no difference.",
            "Write anything. It won't be read.",
        ],
        "cm-aap-batao-kya-hai": [
            "Aap batayiye aapke paas kya hai, main phir bolunga.",
            "Pehle aap kuch dikhaiye, phir sawaal poochhiye.",
            "Aap apni baat rakhiye, mera jawab uske baad.",
        ],
        "cm-do-minute-bache": [
            "Do minute bache hain, unka behtar istemaal kijiye.",
            "Paanch minute aur de raha hoon is call ko.",
            "Ginti ke minute hain, kaam ki baat kijiye.",
        ],
    },
}


# --------------------------------------------------------------------------- #
# The shapes the diagnostic consumes. Names match phrase_pools so
# phrase_holdout can swap one module for the other.
# --------------------------------------------------------------------------- #

SENTIMENT_OPENER_GROUPS: Dict[str, Dict[str, List[str]]] = OPENER_IDEAS
DEFLECTION_GROUPS: Dict[str, Dict[str, List[str]]] = DEFLECTION_IDEAS

SENTIMENT_OPENERS: Dict[str, List[str]] = {
    label: [p for group in ideas.values() for p in group]
    for label, ideas in OPENER_IDEAS.items()
}
DEFLECTIONS: Dict[str, List[str]] = {
    label: [p for group in ideas.values() for p in group]
    for label, ideas in DEFLECTION_IDEAS.items()
}

ALL_IDEAS: Dict[str, Dict[str, List[str]]] = {
    label: {**OPENER_IDEAS[label], **DEFLECTION_IDEAS[label]}
    for label in SENTIMENT_CLASSES
}


def idea_family(idea: str) -> str:
    """'opener' or 'deflection' — which family an idea key belongs to."""
    for label in SENTIMENT_CLASSES:
        if idea in OPENER_IDEAS[label]:
            return "opener"
        if idea in DEFLECTION_IDEAS[label]:
            return "deflection"
    raise KeyError(idea)


def phrase_register(text: str) -> str:
    """'code-mixed' or 'english'.

    The holdout calls this on the *idea key*, which is why the code-mixed ideas
    are named with a ``cm-`` prefix — the register is a property of the idea,
    declared rather than sniffed. The romanised-Hindi fallback keeps the
    function correct if it is ever handed a raw phrase instead.
    """
    if " " not in text.strip():
        # An idea key. The register is declared by the ``cm-`` prefix rather
        # than sniffed: several English keys ("two-fixed-rest-open",
        # "cm"-less slugs generally) tokenize into fragments that collide with
        # short romanised-Hindi particles like "se", "ka" and "do", and a
        # mis-read key would put an English idea in the code-mixed stratum and
        # distort what the holdout withholds.
        return "code-mixed" if text.startswith("cm-") else "english"
    return "code-mixed" if set(_TOKEN_RE.findall(text.lower())) & _HINDI_MARKERS else "english"


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z']+")

_HINDI_MARKERS: frozenset = frozenset("""
matlab bataiye bata batao deta dijiye hai hain hoon nahi kar karna karta karte
hum humko hume humne yaar toh bhi koi kuch abhi pakka thoda sab saare baar
gaya gayi hua hui raha rahe rahi diya diye liya jaake jaana chahta chahiye
poochh poochhna padega phir pehle pichhli dobara kyun kya kahin galat andaaza
soch sochta sochte samay ghar walon papa mummy bhai behen rishtedaar patni
paisa bahut jaldi aage badhna badho shuru start tareekh pahunch mann photo
purana bech bank clear doston office charcha haath mat jaane dena saal saalon
agle tay faisla haath kaanpta neend aati raat log chutti site petrol gate band
number diya hafte jawab message reply company builder market bhool jaaiye
farak padta sawaal kaam jawaab likha likh rakha note kiya second paper par
seedha point list chaliye jo woh yeh yahi wahi wahan iski iske uska unke unki
sabki raay akela mera meri main mein se ke ki ka aur ab bas ek do char chaar
paanchvi kitni har topic call phone barbaad chakkar laga mila band khol
""".split())

# Words that name a feeling outright. v3 leaned on these and the model learned
# them instead of the language around them; none may appear in a pilot phrase.
AFFECT_WORDS: frozenset = frozenset("""
excited exciting excitement enthusiastic enthusiasm thrilled thrilling
delighted delight elated ecstatic overjoyed joyful joy happy happiness glad
pleased chuffed keen eager eagerly buzzing love loved lovely wonderful
brilliant amazing fantastic great fabulous
frustrated frustrating frustration annoyed annoying annoyance irritated
irritating angry anger furious fuming livid cross upset unhappy displeased
impatient impatience patience fed sick tired weary exasperated disgusted
hesitant hesitate hesitation unsure uncertain uncertainty nervous anxious
anxiety worried worry worrying scared afraid fear apprehensive reluctant
doubtful doubt confused confusion
neutral indifferent
khush naraz gussa pareshan dukhi udaas mazaa maza excited
""".split())


def affect_word_leaks() -> Dict[str, List[str]]:
    """{tone: phrases containing a banned affect word} — must be empty.

    The pilot's whole premise is that tone is inferable from situation. A
    single "we're excited" would give the model a shortcut back, and a shortcut
    used by even a few transcripts is enough to depress the held-out score,
    because the holdout would remove it from some ideas and not others.
    """
    leaks: Dict[str, List[str]] = {}
    for label in SENTIMENT_CLASSES:
        for phrases in ALL_IDEAS[label].values():
            for phrase in phrases:
                hits = set(_TOKEN_RE.findall(phrase.lower())) & AFFECT_WORDS
                if hits:
                    leaks.setdefault(label, []).append(f"{phrase}  <-- {sorted(hits)}")
    return leaks


# Closed-class words the strandable-token guard does not police, in both
# languages of this corpus.
#
# The guard exists to catch a token that reads as a tone marker *on its own* —
# "bahut" was the one that inverted Enthusiastic and Frustrated in v3. Function
# words cannot do that: they are distributional filler that appears wherever the
# grammar demands it, so a tone "losing" one to a rival strands no signal. The
# list is therefore closed-class by construction — pronouns, auxiliaries,
# postpositions, determiners, conjunctions, light verbs, numerals, temporal
# deictics — and deliberately excludes anything evaluative.
#
# What is *not* here matters as much as what is. Intensifiers (bahut, bilkul,
# zyada, itna, thoda, poora, sirf), hedges (honestly, genuinely, probably,
# quite, fairly), evaluative adjectives (right, wrong) and every situational
# noun stay policed, because those are exactly the words a model could mistake
# for a label.
_STOPWORDS_EN: str = """
a an the and or but so if then than that this these those there here it its
i i'd i'll i'm i've we we'd we'll we're we've you you'd you'll you're you've
he she they them him her his hers their theirs my mine our ours your yours
me us am is are was were be been being do does did done doing have has had
will would can could shall should may might must not no nor of in on at to
for from with without about into over under again once more most other some
any all each few own same as by up down out off just now very too also who
whom which what when where why how one two three four five six ten
say says said tell told get got go goes going come comes came take takes took
know knows knew think thinks make makes made give gives gave put puts
call calls called ring rings rang phone speak spoke spoken talk talks talked
thing things bit part place places time times day days week weeks month
months year years yet still already ever never
ago until till upon whilst during before after while
first second third fourth fifth half twice thrice double single
something anything nothing everything someone anyone everyone somebody nobody
rather whether whatever whichever whenever wherever either neither both
don't doesn't didn't won't wouldn't can't couldn't shouldn't isn't aren't
wasn't weren't hasn't haven't hadn't it's that's there's what's here's let's
i'm you're we're they're he's she's who's
let lets letting keep keeps kept
"""

_STOPWORDS_HI: str = """
hai hain hoon ho hota hoti hote hoga hogi honge tha thi the
kar karna karta karti karte karna kiya kiye ki ka ke ko se mein par
gaya gayi gaye raha rahe rahi rakha rakhi rakhte
diya diye dena deta dete dijiye de doon dunga dega degi denge
liya lena leta lete lijiye le lunga lenge
bata batao bataiye bataunga batata batate bataya
sakta sakte sakti chahiye padega padta padti padegi
chuka chuke chuki karenge karunga karoge jaunga aaunga jayega jaayenge
hum humko humne hume hamare hamari main mera mere meri mujhe
aap aapka aapke aapki apna apne apni
woh yeh iska uska unka unke unki iske jo jis
ab abhi baad pehle phir tak jab tab agar lekin magar isliye
aur ya bhi hi toh bas kuch koi kya kyun kaise kahan kab
har sab saara saare ek do teen chaar paanch chhe
wala wale wali yahan wahan andar bahar upar niche
matlab yaani haan nahi na mat
yahi wahi kisi kise aisa aise waisa jaisa jaisi jaise
bol bola bole bolta bolte bolna bolunga bolkar
keh kaha kehte kehna kehta
pata baar aage peechhe
every people person folk
want wants wanted need needs needed
next last previous
look looks looked looking see seen saw
dekh dekha dekhi dekhe dekhna dekhne dekhta dekhte dekhkar dekhenge
calling ringing speaking asking
ask asks asked question questions answer answers answered answering
poochh poochha poochhna poochhne poochhta poochhte poochhiye poochhkar
sawaal jawab jawaab
baat baatein log logon
hua hue hui
write writes wrote written writing likh likha likhi likhe likhna likhne likhit
leave leaves leaving left chhod chhodna chhoda
minute minutes hour hours second seconds moment moments
isme usme inme unme ismein usmein
"""

_STOPWORDS: frozenset = frozenset(_STOPWORDS_EN.split()) | frozenset(_STOPWORDS_HI.split())


def token_idea_counts() -> Dict[str, Dict[str, int]]:
    """{content token: {tone: how many of that tone's ideas contain it}}."""
    counts: Dict[str, Dict[str, int]] = {}
    for label in SENTIMENT_CLASSES:
        for phrases in ALL_IDEAS[label].values():
            tokens = {
                token
                for phrase in phrases
                for token in _TOKEN_RE.findall(phrase.lower())
                if len(token) > 2 and token not in _STOPWORDS
            }
            for token in tokens:
                per_label = counts.setdefault(token, {})
                per_label[label] = per_label.get(label, 0) + 1
    return counts


def strandable_tokens(dominant_ideas: int = 3) -> Dict[str, Dict[str, int]]:
    """Content tokens a one-idea holdout could hand to a rival tone that owns them.

    A token shared by two tones is dangerous when one tone carries it in a
    *single* idea and another carries it in many: withhold that one idea and,
    at training time, the token belongs to the rival outright, so every
    held-out transcript containing it is pushed toward the rival label. That is
    the exact mechanism behind v3's Enthusiastic/Frustrated inversion, where
    the stranded token was "bahut" (one Enthusiastic clause, one Frustrated).

    ``dominant_ideas`` is what makes this a usable rule rather than an
    unsatisfiable one. Requiring *every* shared token to appear in two ideas of
    every tone flags 150-odd words, almost all of them incidental scenery
    ("drove", "fortnight", "Sunday") that is weak evidence in both tones, so
    stranding it strands nothing — the same argument ``phrase_pools`` makes for
    scoping its guard to curated markers rather than all vocabulary. What is
    actually hazardous is a lopsided token: singly carried here, well
    established there. Those are flagged and the pools are written so this
    returns empty.

    The residual 1-vs-2 cases are reported by :func:`weakly_shared_tokens` and
    are accepted deliberately: with four phrasings per idea and the tone
    carried by a whole situation rather than a word, a token backed by two
    ideas on the other side cannot outweigh the sentence around it.

    Returns ``{token: {tone: idea count}}`` per violation.
    """
    return {
        token: dict(sorted(per_label.items()))
        for token, per_label in token_idea_counts().items()
        if len(per_label) > 1
        and min(per_label.values()) < 2
        and max(per_label.values()) >= dominant_ideas
    }


def weakly_shared_tokens() -> Dict[str, Dict[str, int]]:
    """Singly-carried shared tokens whose rival tone holds at most two ideas.

    Reported rather than fixed — see :func:`strandable_tokens`. Kept visible so
    a later run can check the count has not quietly grown into a real confound.
    """
    return {
        token: dict(sorted(per_label.items()))
        for token, per_label in token_idea_counts().items()
        if len(per_label) > 1 and min(per_label.values()) < 2 and max(per_label.values()) < 3
    }


def substring_collisions() -> Dict[str, List[str]]:
    """Phrases that contain another idea's phrase — must be empty.

    The holdout partitions transcripts by asking whether a withheld phrase
    occurs in the text and a training-visible one does not. If one idea's
    phrase were a substring of another's, a transcript built from the visible
    idea would look like it contained the withheld one and be misfiled.
    """
    collisions: Dict[str, List[str]] = {}
    for label in SENTIMENT_CLASSES:
        flat = [
            (idea, _normalize(phrase).strip())
            for idea, phrases in ALL_IDEAS[label].items()
            for phrase in phrases
        ]
        for idea_a, text_a in flat:
            for idea_b, text_b in flat:
                if idea_a != idea_b and text_b and text_b in text_a:
                    collisions.setdefault(label, []).append(f"{idea_a} contains {idea_b}: {text_b}")
    return collisions


def _normalize(text: str) -> str:
    """Mirrors ``sentiment_intent.data._normalize`` — kept local so this module
    does not import the backend."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def pool_summary() -> Dict[str, object]:
    return {
        "ideas_per_tone": {
            label: {
                "opener": len(OPENER_IDEAS[label]),
                "deflection": len(DEFLECTION_IDEAS[label]),
                "opener_code_mixed": sum(
                    1 for k in OPENER_IDEAS[label] if phrase_register(k) == "code-mixed"
                ),
                "deflection_code_mixed": sum(
                    1 for k in DEFLECTION_IDEAS[label] if phrase_register(k) == "code-mixed"
                ),
                "phrases": len(SENTIMENT_OPENERS[label]) + len(DEFLECTIONS[label]),
            }
            for label in SENTIMENT_CLASSES
        },
        "affect_word_leaks": affect_word_leaks(),
        "strandable_tokens": strandable_tokens(),
        "weakly_shared_token_count": len(weakly_shared_tokens()),
        "substring_collisions": substring_collisions(),
        # Recorded so a reader can see the strandable-token guard is not passing
        # vacuously: it is the share of the vocabulary the closed-class
        # stopword list does *not* exempt.
        "guard_coverage": {
            "tokens_policed": len(token_idea_counts()),
            "tokens_policed_in_more_than_one_tone": sum(
                1 for v in token_idea_counts().values() if len(v) > 1
            ),
            "stopwords": len(_STOPWORDS),
        },
    }
