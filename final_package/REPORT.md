# Method report: team Model Phi, Decode & Detect (24 October 2026)

**Task.** Give each of the 40,101 test payments a suspicion score from 0 to 1; the score is 100 x average precision (AP). Scores rank payments **for review**. A high score is "flagged for review", never a finding about any account.

## 1. Data and sources

Only organiser data: `train/` (40,131 payments, 1,250 labelled suspicious, 3.1%), the unlabelled `validation/` (33,862 payments, used only to check for distribution shift) and the test download (40,101 payments, no labels). The data is a synthetic seven-day ledger of 700 wallets in the XRPL schema. No external data, API, service or pre-trained model is used (`THIRD_PARTY.md`). Amounts are valued in US dollars from the organiser's hourly reference prices; the real cost of a conversion is the first executed transfer leg (`SendMax` is only a ceiling); there is one score per transaction. Wallet addresses are never features.

## 2. Method

The entry blends two models built separately by the two team members, both given the same relay-motif features, and both adapted to the test week by self-training.

**Model A (`ours/`).** 205 behavioural features per payment, computed once over a split's own graph: bursts of sends and receipts around each wallet at windows from 5 minutes to 12 hours; whether the receiver passes money on, and how closely the amount matches; first-time sender-receiver pairs and clusters of them (planted shapes are built almost entirely from first-time pairs: 95% of planted payments against 28% of background); chains and short cycles; gap regularity; clusters of payments linked by shared wallets (statistics replaced by a flag above 60 payments, because a giant cluster describes the whole split). Asset and conversion features are left out because the organiser states that asset and conversion choices do not inspect labels. Learner: LightGBM (600 trees, three seeds) plus ExtraTrees (500 trees), rank-blended 0.65 / 0.35.

**Relay-motif features (`ours/src/dd/motifs.py`, 26 features, used by both models).** Two label-free temporal-motif detectors over the whole week. *Aggregate relay:* for each wallet, a compact inbound burst (within 90 minutes), a 3.5 to 5.5 hour delay, then a compact outbound burst whose total value is roughly conserved (out/in between 0.5 and 1.5), with at least three counterparties, repeated on separate occasions with consistent delay and value ratio. *Distributed relay:* for each directed pair A to B that repeats, a feeder burst into A ending 1.5 to 3.5 hours before a bridge payment and a payout burst from B starting 1.5 to 3.0 hours after it, with both values conserved, repeated at several bridges. Burst members are restricted to rare pairs (relay payments sit on pairs that pay at most four times a week; recurring commerce does not), and no single payout may exceed what the burst took in. Every payment in a detected motif carries the motif's repeat count, delays, timing consistency, value ratios, counts and a confidence score. A hard "lift detected motifs to the top" rule was tested and rejected: in training, payments in detected motifs are only 3 to 4% suspicious, because legitimate relay look-alikes follow the same timing template.

**Model B (`shawn/`).** 218 features: the payment, how often the pair recurs, how unusual the amount is for each wallet, local timing, clusters of one-off payments, two-hop neighbourhoods, money flowing in and out, relay hand-off matching; plus a learned scenario reconstruction (a link model trained on the `scenario_id` column of `train/labels.csv` scores whether two payments sharing a wallet belong to the same scenario), helper models for "part of a planted shape?" and "part of a relay?" (trained on the `family` column), a two-stage LightGBM that adds neighbours' out-of-fold scores, and a look-alike expert. Nothing from `labels.csv` is read when scoring the test data. The relay-motif features are added to its feature table by a wrapper (`ours/src/run_shawn_with_motifs.py`); model B's own files are unchanged.

**Self-training on the test week.** The test week contains "batch variations absent from training" (organiser README). Each model is re-trained once with pseudo-labels: the most confident test payments of a teacher ranking are labelled suspicious and the bottom 40% normal, added at half weight; the test week is re-scored with 2-fold cross-fitting, so no payment is scored by a model that saw its own pseudo-label; the result is averaged with the un-adapted score. No test label is used: pseudo-labels come only from our own rankings. [FINAL: describe the teacher(s) and pseudo-label counts of the submitted variant.]

**Blend.** 50/50 average of the two models' rank scores, ranked again so that scores are unique in [0, 1].

## 3. Evaluation

Five-fold cross-validation on `train/`, grouped so that no scenario sits on both sides of a split. For model A the groups are relay cohorts, other planted scenarios and background senders (train is a single connected wallet component, so wallet-disjoint folds are impossible); for model B they are scenarios.

| Model (out-of-fold, train) | Pooled AP |
|---|---|
| Organiser starter notebook (12 rules, logistic regression), under model A's folds | 29.4 |
| Model A without / with relay-motif features | 62.3 / 63.0 |
| Model B without / with relay-motif features (its own folds) | 61.6 / 61.9 |
| 50/50 blend of A and B without motifs / with motifs on A | 64.2 / 64.9 |

Fold-to-fold spread is about 9 AP points, so differences under one point are noise. Self-training cannot be cross-validated on train (it adapts to the test week). Robustness checks on model A: leaving a whole motif family out of training still ranks its payments above background (AUC 0.95 to 1.00); training and scoring on disjoint halves of the wallets gives the same AP with and without the per-wallet weekly counts (64.3 vs 64.4), so the model is not memorising wallets; features that fingerprinted a split (1-hour clusters, statistics of giant clusters) were removed after a train-versus-validation check.

**Leaderboard (test week), in the order tried.** Blend without motifs 71.3; with motif features in model A 71.6; in both models 71.7; plus self-training of model A 71.8; plus self-training of model B 72.0. [FINAL: add the submitted variant.] The test week scores about 7 to 10 points above cross-validation for both team members' models. Because changes worth less than a point in cross-validation did not reliably transfer, the last choices (motif features in model B, self-training settings, which file to submit) were made on the leaderboard, among variants produced by the same code. Two leaderboard probes of the motif "lift" rule (raising the most regular detected motifs into the review band) gave no gain and were dropped.

## 4. Findings

- Planted easy motifs run at about one step per minute between first-time counterparties (median gap 42 to 80 seconds by family). Legitimate look-alikes of the same shape run at about 7 to 12 minutes with irregular gaps. Planted medium motifs are slowed to about 10 to 28 minutes and are more evenly spaced than their look-alikes, which is the main thing that still separates them.
- Every planted shape has a legitimate look-alike with the same structure, so most errors are look-alikes: in model A's top 1% in training, 68 of 401 payments are false positives, mostly bipartite (26), stack (12), chain (8) and fan-out-in (8) look-alikes.
- Relay timing is a template shared by planted relays and their look-alikes (both run the 3.5 to 5.5 hour and roughly 2 hour delays). What differs is value conservation in every batch, and that is hard to measure because about twenty relay scenarios overlap on each wallet at the same time.
- Transaction metadata carried no label signal in our checks (fee, flags, transaction index, last-ledger offset) or in Shawn's (sequence numbers, `SendMax` slack, paths, ledger position).

## 5. Limits, uncertainty and bias

- **The two relay families are largely unsolved.** They hold 200 of the 1,250 suspicious payments (16%): each relay cohort is about 100 near-identical relay scenarios with exactly one planted. The motif features roughly double the distributed-relay AP in cross-validation (1.3 to 2.6), which is still far from useful precision; the leaderboard reports none of the 200 in the top 1%.
- Medium cases and the 25 deliberately random "ambiguous" payments are weak or unpredictable by design.
- Self-training and leaderboard-informed selection adapt the entry to this test week; the gains may not carry over to another week.
- The data is synthetic: a high score does not show that the method works on real ledgers. The first-time-pair signal in particular reflects how scenarios were injected. This tool is not a certified compliance solution and gives no legal or financial advice.

Worked cases with real transaction hashes (flagged, a look-alike we do not flag, a missed case): `CASES.md`.
