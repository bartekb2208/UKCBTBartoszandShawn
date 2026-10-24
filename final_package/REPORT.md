# Method report: team Model Phi, Decode & Detect (24 October 2026)

**Task.** Give each of the 40,101 test payments a suspicion score from 0 to 1; the score is 100 x average precision (AP). Scores rank payments **for review**. A high score is "flagged for review", never a finding about any account.

## 1. Data and sources

Only organiser data: `train/` (40,131 payments, 1,250 labelled suspicious, 3.1%), the unlabelled `validation/` (33,862 payments, used only to check for distribution shift) and the test download (40,101 payments). The data is a synthetic seven-day ledger of 700 wallets in the XRPL schema. No external data, API, service or pre-trained model is used (`THIRD_PARTY.md`). Amounts are valued in US dollars from the organiser's hourly reference prices; the real cost of a conversion is the first executed transfer leg (`SendMax` is only a ceiling); there is one score per transaction. Wallet addresses are never features.

## 2. Method

The entry is a 50/50 average of the rank scores of two models built separately by the two team members. The weight was fixed before any test score existed.

**Model A (`ours/`).** 205 behavioural features per payment, computed once over a split's own graph: bursts of sends and receipts around each wallet at windows from 5 minutes to 12 hours; whether the receiver passes money on, and how closely the amount matches; first-time sender-receiver pairs and clusters of them (planted shapes are built almost entirely from first-time pairs: 95% of planted payments against 28% of background); chains and short cycles; gap regularity; clusters of payments linked by shared wallets. Cluster statistics are replaced by a flag when a cluster is larger than 60 payments, because a giant cluster describes the whole split rather than a motif (this removed a train-versus-validation fingerprint). Asset and conversion features are left out because the organiser states that asset and conversion choices do not inspect labels. Learner: LightGBM (600 trees, three seeds) plus ExtraTrees (500 trees), rank-blended 0.65 / 0.35.

**Model B (`shawn/`).** 218 features: the payment, how often the pair recurs, how unusual the amount is for each wallet, local timing, clusters of one-off payments, two-hop neighbourhoods, money flowing in and out, relay hand-off matching. It adds a learned scenario reconstruction (a link model scores whether two payments sharing a wallet belong to the same scenario; confident links are joined and each reconstructed scenario is described), two helper models ("part of a planted shape?" and "part of a relay?"), a two-stage LightGBM that adds neighbours' out-of-fold scores, and a "look-alike expert" trained only on planted-shape payments. The final score is a rank blend of the main model and the expert. The link model and helper models are trained on the `scenario_id` and `family` columns of `train/labels.csv`; nothing from those columns is read when scoring the test data.

## 3. Evaluation

Five-fold cross-validation on `train/`, grouped so that no scenario sits on both sides of a split. For model A the groups are relay cohorts, other planted scenarios, and background senders (train is a single connected wallet component, so wallet-disjoint folds are impossible); for model B they are scenarios.

| Model | Pooled out-of-fold AP | Mean of fold AP |
|---|---|---|
| Organiser starter notebook (12 rules, logistic regression), under model A's folds | 29.4 | 32.4 +/- 10.2 |
| Model A (`ours/src/cv.py`) | 62.3 | 63.5 +/- 9.1 |
| Model B (his `run.py`; his own folds) | 61.6 | n/a |
| 50/50 blend of the two out-of-fold scores (`analysis/blend_cv.py`) | 64.2 | 65.0 |

The blend figure mixes two fold designs, so it is an estimate; the best weight on out-of-fold scores was about 0.6 for model A (64.3) and the curve is flat between 0.4 and 0.8. Fold-to-fold spread is about 9 AP points, so differences under one point are noise. AP by family for model A (that family's planted payments against its own look-alikes): fan_out_in 91.0, fan_out 86.8, cycle 86.0, multi_stage 83.3, fan_in 79.2, stack 78.6, chain 77.2, random 77.0, bipartite 60.0, aggregate_relay 1.2, distributed_relay 1.3. Precision in the top 1% (401 payments) is 83%, recall there 27%; in the top 3% precision is 67% and recall 64%.

Robustness checks, all in `ours/src/`: leaving a whole motif family out of training still ranks its payments above background (AUC 0.95 to 1.00, `lofo.py`); training and scoring on disjoint halves of the wallets gives 64.3 AP (standard error 1.6), and 64.4 without the per-wallet weekly counts, so the model is not memorising wallets (`wallet_split.py`). A train-versus-validation classifier (`shift_check.py`, labels unused) separates the two splits easily (AUC 0.87 even without the per-wallet weekly counts, which on their own identify wallets), so the splits are not identically distributed. We removed the features that fingerprinted a split most strongly (1-hour clusters, statistics of giant clusters); the model never receives a split identifier. Repeated runs of the whole pipeline give byte-identical files.

## 4. Findings

- Planted easy motifs run at about one step per minute between first-time counterparties (median gap 42 to 80 seconds by family). Legitimate look-alikes of the same shape run at about 7 to 12 minutes with irregular gaps. Planted medium motifs are slowed to about 10 to 28 minutes and then look like the look-alikes. Tempo and first-time pairs are the strongest signals.
- Every planted shape has a legitimate look-alike with the same structure, so most errors are look-alikes: in the top 1%, 68 of the 401 payments are false positives, mostly bipartite (26), stack (12), chain (8) and fan-out-in (8) look-alikes.
- Transaction metadata carried no label signal in our checks (fee, flags, transaction index, last-ledger offset) or in Shawn's (sequence numbers, `SendMax` slack, paths, ledger position; see `shawn/report.md`).

## 5. Limits, uncertainty and bias

- **The two relay families are not solved.** They hold 200 of the 1,250 suspicious payments (16%): each relay cohort is about 100 near-identical relay scenarios with exactly one planted. Only 33 of the 200 planted relay payments reach the top 5,000 (median rank 12,128 of 40,131), no better than chance. Inside its own scenario a planted relay hub never pays out more than it received, but that is only visible when the scenario's payments are already grouped, and we could not recover the grouping reliably.
- Medium cases (median rank 2,408) and the 25 deliberately random "ambiguous" payments are weak or unpredictable by design.
- The test set contains batch variations absent from training, so cross-validation may not transfer; for model A and the blend we used cross-validation, not leaderboard feedback, to choose features, models and the weight; the only leaderboard-informed choice is which of five candidate files to submit. [FILL: Shawn to confirm whether any model B design choice used leaderboard feedback, and say so here if it did.]
- The data is synthetic: a high score does not show that the method works on real ledgers. The first-time-pair signal in particular reflects how scenarios were injected. This tool is not a certified compliance solution and gives no legal or financial advice.

Worked cases with real transaction hashes (two flagged, one look-alike we do not flag, one we miss): `CASES.md`.
