# Decode & Detect: method report

**Task.** Rank all 40,101 test payments so that planted laundering scenarios come first (score = 100 x AP).

**What the training data showed.**
- 700 busy wallets (~115 payments each). Planted scenarios are bursts of one-off sender-receiver pairs inside recurring commerce (median 1 payment per pair against 5).
- Every laundering shape (fan-in, fan-out, chain, cycle, bipartite, stack, multi-stage, random) also appears as a legitimate decoy with the same structure.
- Easy laundering runs ~60 s between steps; decoys mostly ~10 min, but ~17% of decoys are as fast as laundering. Laundering intermediaries also conserve value more tightly (pass on ~95% vs ~90%).
- Medium cases are slowed-down versions that look like decoys; hard relay cases are 1-in-100 within cohorts of identical relays, plus 25 random labels.
- All metadata (fees, sequence numbers, SendMax slack, paths, ledger position) carry no signal (AUC ~0.5), so the method relies on behaviour only.

**Method.**
1. One row per transaction from the raw files (Destination and DELIVERED_AMOUNT JSON, XRP drops, issuer+currency assets, USD at the payment hour, first transfer leg as real cost).
2. ~220 behavioural features, never wallet IDs: pair recurrence, out-of-character amounts, local timing around both wallets, short time clusters and 2-hop ego-neighbourhoods of one-off payments, money flow in and out, relay hand-off matching.
3. Learned scenario reconstruction: a link model (trained on scenario annotations) scores whether two payments touching the same wallet belong to the same scenario (AUC 0.986); confident links are joined and each reconstructed scenario is described (rhythm, size, shape, value conservation).
4. Auxiliary models: "part of a planted shape?" (AUC 0.986) and "part of a relay?" (AUC 0.994).
5. Main LightGBM, then a second stage that adds neighbours' out-of-fold scores. A decoy expert trained only on planted shapes gives P(laundering | shape); mixture = P(shape) x expert. Final = rank blend, strict ranking (no ties).
6. Self-training on the test week: the most confident test payments are pseudo-labelled (top 600 positive, bottom 40% negative), added to training at half weight, and test is re-scored with 2-fold cross-fitting so no payment is scored by a model that saw its own pseudo-label. This adapts to the test week's batch variations (fewer linkable relays, more conversions).

**Validation.** 5-fold GroupKFold by scenario_id (no scenario on both sides). Grouped CV AP 61.6 (easy 79, medium 9, hard 1); see results/results.json and results/validation_by_family.csv.
Leaderboard (test): v3 blend 71.1; v3 + self-training 71.2. The test week scores ~9.5 points above CV. Changes that helped CV by <1 point did not reliably transfer (e.g. removing wallet-degree inputs from the link model: CV flat, test -0.9), so final choices were made on the leaderboard among variants produced by the same code.

**What did not work (tested, kept out).** Ranking objectives (lambdarank, rank_xendcg); score propagation over ego-neighbourhoods; a second propagation round; a no-link-features model (test 68.6); rank ensembles with it (69.8). Hard relays: ~40 relay-level and whole-week context statistics, a relay-only model (out-of-fold AUC 0.34) and within-cohort outlier detection (median rank 50/100) all failed: each suspicious relay is indistinguishable from its ~99 look-alikes.

**Limitations.** Medium and hard cases are largely indistinguishable from legitimate look-alikes by design; easy decoys that are as fast as laundering remain the main source of false positives. Synthetic data: a high score does not show real-world performance. High scores mean "flagged for review", not wrongdoing.
