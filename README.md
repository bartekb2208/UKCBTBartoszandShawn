# Decode & Detect: ranking suspicious XRPL payments

## Run (exact command)
```
pip install -r requirements.txt          # Python 3.10+
python run.py --train ../train --test ../test
python package.py --test ../test         # checks every submission rule, builds submission_code.zip
```
Paths are the organisers' folders as downloaded. Dry run without the test download: `--test ../validation`.
Output: `out/submission.csv` (TRANSACTION_HASH,score), every test transaction once, all scores unique.
Hardware: CPU only, ~13 minutes on a 2-core machine (faster with more cores), ~3 GB RAM. Seed 42. No network, APIs or LLM calls at run time.

## Submission variants (all written by the same code)
`python run.py --train ../train --test ../test` writes four files to out/:
- `submission.csv`: rank blend of main model and mixture (weight chosen on grouped CV)
- `submission_main_only.csv`, `submission_mixture_only.csv`: the two components alone
- `submission_selftrain.csv`: blend re-scored after self-training on confident test pseudo-labels (2-fold cross-fitted on test)
- `submission_selftrain_2rounds.csv`, `submission_selftrain_top900.csv`: two self-training rounds; 900 pseudo-positives
- `submission_floor01_blend.csv`, `submission_floor03_blend.csv`: blends with a mixture that ranks planted shapes above relay/background
`python run.py --no-links --out out_nolinks ...` writes the same four files without the learned scenario-link features.
Ensemble (rank average, equal weights, strict ranking):
`python ensemble.py out/ensemble_AB.csv out/submission.csv out_nolinks/submission.csv out/submission_mixture_only.csv out_nolinks/submission_mixture_only.csv`

## Findings from the training data (drive the design)
- 700 busy wallets, ~115 payments each. Planted scenarios are bursts of one-off pairs (median 1 payment per pair)
  inside recurring commerce (median 5 per pair).
- Every laundering shape also appears as a legitimate decoy with the same structure.
  Easy suspicious scenarios run ~60 s between steps; decoys ~10 min; medium suspicious ~18 min and more regular.
- Hard cases: relay cohorts of ~100 identical-size relay scenarios, exactly one suspicious per cohort,
  plus 25 deliberately random labels. Only weak signal (faster hand-off) found so far.

## Method
1. `load.py` one row per transaction: Destination and DELIVERED_AMOUNT from JSON, XRP drops, issuer:currency assets,
   USD at the hour of the payment, first transfer leg as the real cost (SendMax is only a ceiling).
2. `features.py` (~160 features, never wallet IDs): payment, pair recurrence, out-of-character amounts,
   local timing around both wallets, **time clusters of one-off payments** at 2/5-minute windows and **ego-neighbourhoods**
   (2 wallet-hops, 3/10/30/60 min, no transitive chaining): size, duration, gap rhythm, shape, value, money flow in/out over 1-12 h, relay hand-off matching.
3. `model.py` LightGBM, 5-fold GroupKFold by `scenario_id` (no scenario on both sides), 3 seeds.
   Stage 2 adds neighbours' out-of-fold scores (same cluster, same wallet within 10 min / 1 h).
   Final scores are a strict ranking (no ties).

## Validation (grouped 5-fold, training split)
See `out/results.json` and `out/validation_by_family.csv`. Current: AP 60.4 (easy 79, medium 7, hard 1), see results.json for AUC and top-1% precision.
`labels.csv` metadata (family, difficulty, scenario_id) is used only for grouping and reporting, never as features.

## Not written by us
pandas, numpy, scipy, scikit-learn, LightGBM, pyarrow (see requirements.txt). AI assistance: see PROMPTS.md.

## Language
High scores mean "flagged for review". Synthetic data; nothing here concerns real people or accounts.
