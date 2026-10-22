# EstateIQ — intent on 60 unseen, inferred-intent transcripts

Model: `C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP\backend\models_store\sentiment_intent\distilbert_intent_v1_20260928_163707`  
Training corpus checked for overlap: `C:\Users\VAISHNAVI KANKAREJ\EstateIQ_FYP\data_generation\output\intent_v1.jsonl`

## 1. The test set

60 transcripts written for this evaluation before the training data, Buy 10, Rent 10, Inquiry 10, Schedule Visit 10, Investment 10, Request Callback 10. 7 English and 3 code-mixed per intent. No caller names an intent (buy / rent / invest / visit / call back / enquiry and their Hindi equivalents are banned and checked: **0 leaks**).

Longest word run any case shares with the training corpus: **4 words** (median 3).

## 2. Headline

- Accuracy: **71.7%** (43/60) — chance is 16.7%
- Macro precision / recall / F1: 0.719 / 0.717 / **0.715**

## 3. Per-class precision / recall / F1

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Buy | 10 | 7 | 2 | 3 | 0.778 | 0.700 | 0.737 |
| Rent | 10 | 6 | 2 | 4 | 0.750 | 0.600 | 0.667 |
| Inquiry | 10 | 6 | 4 | 4 | 0.600 | 0.600 | 0.600 |
| Schedule Visit | 10 | 10 | 2 | 0 | 0.833 | 1.000 | 0.909 |
| Investment | 10 | 5 | 6 | 5 | 0.455 | 0.500 | 0.476 |
| Request Callback | 10 | 9 | 1 | 1 | 0.900 | 0.900 | 0.900 |

## 4. Confusion matrix (rows = truth, columns = predicted)

| truth \ pred | Buy | Rent | Inquiry | Schedule Visit | Investment | Request Callback |
|---|---|---|---|---|---|---|
| **Buy** | **7** | 0 | 1 | 0 | 2 | 0 |
| **Rent** | 0 | **6** | 2 | 0 | 2 | 0 |
| **Inquiry** | 0 | 0 | **6** | 1 | 2 | 1 |
| **Schedule Visit** | 0 | 0 | 0 | **10** | 0 | 0 |
| **Investment** | 2 | 2 | 1 | 0 | **5** | 0 |
| **Request Callback** | 0 | 0 | 0 | 1 | 0 | **9** |

## 5. By subset

| Subset | Cases | Accuracy | Macro-F1 |
|---|---|---|---|
| english | 42 | 78.6% | 0.778 |
| code-mixed | 18 | 55.6% | 0.549 |
| clean | 42 | 69.0% | 0.689 |
| messy | 18 | 77.8% | 0.764 |

## 6. Every miss, in full

**B05** (english, clean) — expected **Buy**, predicted **Inquiry** at 0.95
> Is the title clear, and does the builder handle the conveyance deed? My lawyer will ask before we sign the agreement for sale. 3 BHK, Hadapsar, 1.1 c…
- full distribution: Buy 0.01, Rent 0.00, Inquiry 0.95, Schedule Visit 0.01, Investment 0.03, Request Callback 0.00

**B08** (code-mixed, clean) — expected **Buy**, predicted **Investment** at 0.88
> Apna ghar chahiye jisme koi makaan-malik na ho. Registry mere naam pe honi chahiye. Chinchwad mein 2 BHK, 75 lakh.
- full distribution: Buy 0.02, Rent 0.06, Inquiry 0.02, Schedule Visit 0.02, Investment 0.88, Request Callback 0.01

**B10** (code-mixed, clean) — expected **Buy**, predicted **Investment** at 0.92
> Stamp duty ka hisaab bhi laga liya hai. Possession 2027 tak chahiye, Nashik Road pe 3 BHK, ek crore tak.
- full distribution: Buy 0.03, Rent 0.01, Inquiry 0.02, Schedule Visit 0.03, Investment 0.92, Request Callback 0.00

**R02** (english, clean) — expected **Rent**, predicted **Inquiry** at 0.84
> What is the deposit — two months or three? And is maintenance included in the monthly amount? 1 BHK in Viman Nagar, fully furnished.
- full distribution: Buy 0.01, Rent 0.01, Inquiry 0.84, Schedule Visit 0.01, Investment 0.12, Request Callback 0.00

**R06** (english, messy) — expected **Rent**, predicted **Investment** at 0.89
> Our building is being redeveloped, and the builder pays us a monthly allowance for thirty months while it's — while it goes up again. So we need some…
- full distribution: Buy 0.04, Rent 0.05, Inquiry 0.00, Schedule Visit 0.01, Investment 0.89, Request Callback 0.01

**R08** (code-mixed, clean) — expected **Rent**, predicted **Inquiry** at 0.87
> Deposit kitna lagega? Mahine ka 20 hazaar se zyada nahi de paunga. Owner family ko dega na? Wanowrie mein 1 BHK.
- full distribution: Buy 0.01, Rent 0.01, Inquiry 0.87, Schedule Visit 0.07, Investment 0.04, Request Callback 0.00

**R09** (code-mixed, clean) — expected **Rent**, predicted **Investment** at 0.52
> Office ne do saal ke liye yahan bheja hai, company har mahine 30 hazaar tak deti hai. Magarpatta ke paas 2 BHK.
- full distribution: Buy 0.04, Rent 0.02, Inquiry 0.10, Schedule Visit 0.30, Investment 0.52, Request Callback 0.02

**Q01** (english, clean) — expected **Inquiry**, predicted **Investment** at 0.90
> We're two years away from doing anything. I'd just like a sense of where prices are in Baner right now for a 3 BHK.
- full distribution: Buy 0.01, Rent 0.01, Inquiry 0.07, Schedule Visit 0.01, Investment 0.90, Request Callback 0.00

**Q04** (english, messy) — expected **Inquiry**, predicted **Request Callback** at 0.51
> Could you, um, email me the price list and the floor plans? I'll go through them later, no rush on anything.
- full distribution: Buy 0.02, Rent 0.01, Inquiry 0.22, Schedule Visit 0.22, Investment 0.01, Request Callback 0.51

**Q08** (code-mixed, clean) — expected **Inquiry**, predicted **Schedule Visit** at 0.79
> Sirf rate jaanna tha. Kharadi mein 3 BHK kitne ka chal raha hai aajkal?
- full distribution: Buy 0.01, Rent 0.01, Inquiry 0.18, Schedule Visit 0.79, Investment 0.01, Request Callback 0.01

**Q10** (code-mixed, messy) — expected **Inquiry**, predicted **Investment** at 0.68
> Abhi koi plan nahi hai, bas, woh, pata karna tha ki Nashik Road pe square foot ka kya rate hai.
- full distribution: Buy 0.01, Rent 0.23, Inquiry 0.03, Schedule Visit 0.04, Investment 0.68, Request Callback 0.01

**I01** (english, clean) — expected **Investment**, predicted **Rent** at 0.66
> It isn't for living in. I want something that appreciates once the metro line opens, and I'll sell in five or six years.
- full distribution: Buy 0.00, Rent 0.66, Inquiry 0.01, Schedule Visit 0.01, Investment 0.31, Request Callback 0.01

**I04** (english, messy) — expected **Investment**, predicted **Buy** at 0.85
> A shop that already has a tenant paying would be ideal — I live in Dubai, so, uh, I won't ever use it myself. Budget 60 lakh.
- full distribution: Buy 0.85, Rent 0.00, Inquiry 0.03, Schedule Visit 0.00, Investment 0.10, Request Callback 0.01

**I05** (english, clean) — expected **Investment**, predicted **Inquiry** at 0.72
> Pre-launch pricing, then exit at possession — how much did the last phase go up between launch and handover?
- full distribution: Buy 0.23, Rent 0.00, Inquiry 0.72, Schedule Visit 0.01, Investment 0.02, Request Callback 0.01

**I06** (english, clean) — expected **Investment**, predicted **Buy** at 0.74
> My CA says I should move some money out of equities. Land near the new ring road seems the obvious place for it.
- full distribution: Buy 0.74, Rent 0.00, Inquiry 0.01, Schedule Visit 0.01, Investment 0.23, Request Callback 0.01

**I08** (code-mixed, clean) — expected **Investment**, predicted **Rent** at 0.78
> Rehne ke liye nahi hai. Paisa aisi jagah lagana hai jahan value badhe, metro aane ke baad bech denge.
- full distribution: Buy 0.00, Rent 0.78, Inquiry 0.01, Schedule Visit 0.01, Investment 0.19, Request Callback 0.01

**C09** (code-mixed, clean) — expected **Request Callback**, predicted **Schedule Visit** at 0.80
> Mere pati se baat kar lijiye, unka number note kar lo. Raat aath baje ke baad free rehte hain.
- full distribution: Buy 0.02, Rent 0.09, Inquiry 0.01, Schedule Visit 0.80, Investment 0.02, Request Callback 0.06
