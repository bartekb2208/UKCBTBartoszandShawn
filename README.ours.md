# Decode & Detect: motif-shape ranker

UKCBT × XRPL hackathon entry, 24 October 2026. Ranks every test payment by suspicion for review, using the shape of the activity around it: bursts, first-time counterparties, pass-through and reconstructed motifs. It never uses wallet identity.

**Team:** _(team name)_ · **Captain:** _(name)_ · **Members:** Bartosz Bielecki, Shawn _(surname)_ · **Commit:** _(git commit of the upload)_

## Reproduce the predictions

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14.6
# put the organiser data next to this README:
#   decode-and-detect-training-data/train/   (training download)
#   test-data/                               (test download, unzipped)
PY=.venv/bin/python ./run.sh                                          # writes predictions.csv
```

This is the same as `python src/predict.py --train decode-and-detect-training-data/train --test test-data --out predictions.csv`. It takes about 95 to 150 seconds on an Apple M4 (10 cores, 16 GB) and peaks at about 0.8 GB of RAM. All seeds are fixed, and two runs give byte-identical CSVs (checked with MD5 on `validation/`). The script checks the CSV before writing it: every test hash exactly once, scores in [0, 1], no ties.

## Method

1. **Load** (`src/dd/load.py`): one row per transaction. The delivered amount is valued in USD at the hourly reference price for its asset (issuer + currency). Actual cost is the first executed transfer leg, not `SendMax`. We also record leg count, conversion flag and spread.
2. **Features** (`src/dd/features.py`, 243 per payment, computed once over the whole split):
   - *Bursts:* the sender's sends and the receiver's receipts within ±5 min, 30 min, 2 h and 12 h: counts, distinct counterparties, value spread, same-currency share, and the ratio to that wallet's own weekly rate.
   - *Pass-through:* whether the receiver sends on within each window, how much, and how close the best amount match is; the mirror image for the sender's funding.
   - *First-time pairs:* planted motifs are built from first-time sender–receiver pairs. For each endpoint we describe its first-time-pair activity within ±30 min, 1 h and 3 h (count, gap median, minimum and CV, span, value spread), and we cluster first-time-pair payments that share a wallet within 20 and 40 minutes.
   - *Motif clusters:* payments that touch the same wallet within 5 and 20 minutes. We use cluster size, span, wallets per payment, sender and receiver ratios, currencies and conversion share. Clusters above 60 payments (40 for first-time pairs) are flagged `giant` instead, because their statistics describe the whole split.
   - *Chains and cycles:* forward and backward similar-amount hops (30 min, 25% tolerance; and 2 h, 40%), value kept along the chain, money returning to the sender in 2 or 3 hops.
   - *Shape:* gap regularity, the share of new counterparties, same-currency runs, the amount against each wallet's usual amount, and time of day.
   - Weekly wallet counts are within-split percentiles, so a busier or quieter week does not shift them. Issuer addresses are not used (they differ between splits).
3. **Model** (`src/predict.py`): LightGBM (binary objective, 600 trees, 3 seeds averaged) and ExtraTrees (500 trees), blended by rank as 0.65 / 0.35. Final scores are rank-normalised into [0, 1] with no ties.

## Validation

5-fold grouped, stratified cross-validation on `train/` (`src/dd/folds.py`). Train has only 700 wallets in one connected component, so wallet-cluster folds are impossible. Relay rows are grouped by cohort, so whole cohorts are unseen in each fold, as in the test set. Other planted rows are grouped by scenario and background rows by sender. Stratification is on family × label.

| Model | Pooled OOF AP | Fold mean ± sd |
|---|---|---|
| Starter notebook (12 rules, logistic regression) under our folds | 29.4 | 32.4 ± 10.2 |
| This entry (`src/cv.py`) | **62.0** | 63.5 ± 10.1 (folds 55.5, 51.6, 62.4, 80.6, 67.5) |

Per-family AP (family positives against that family's own look-alikes): fan_out_in 90.7, cycle 88.4, fan_out 86.5, multi_stage 83.3, stack 80.1, chain 80.0, fan_in 78.0, random 76.9, bipartite 57.9, distributed_relay 1.3, aggregate_relay 1.1. Precision in the top 1% is 84.3%. Fold spread is wide because the folds hold different numbers of unpredictable relay positives.

Robustness checks:
- **Leave-one-family-out** (`src/lofo.py`): with a whole motif family removed from training, its positives still rank above background with AUC 0.95 to 1.00.
- **Wallet-disjoint split** (`src/wallet_split.py`): trained and scored on disjoint halves of the wallets. Removing the weekly wallet features changes AP by 0.1 (within noise), so the model is not memorising wallets.
- **Shift check** (`src/shift_check.py`): train against `validation/`. We removed features that fingerprinted the split (1-hour clusters; statistics of giant clusters).

`NOTES.md` has the full log of experiments, including the ones that did not help.

## Findings and limitations

- Easy and medium motifs differ mainly in tempo. Planted easy motifs run at about 1-minute gaps, look-alikes at about 10 minutes, and planted medium motifs at 15 to 25 minutes. A detector tuned only for "fast" would miss the slow ones and flag the fast legitimate batches.
- **Relay families are not solved.** Each relay cohort is 100 near-identical scenarios with one planted. Our features rank planted relay payments no better than chance (within-relay AP ≈ 1%, the base rate). The planted hub passes on value more evenly in each episode, but detecting that needs scenario membership we cannot recover reliably. These 200 payments cap the score.
- The 25 `ambiguous` payments are unpredictable by design.
- The data is synthetic. The first-time-pair signal reflects how scenarios were injected into the simulated ledger, and a high score does not show that the method would work on real crime.
- Scores rank payments **for review**. A high score is not a finding that any address is criminal.

## Files

- `src/predict.py`: the submission pipeline. `src/dd/`: loader, features, folds, model, CSV checks, starter baseline.
- `src/run_baseline.py` (builds the cache and fold map, scores the starter baseline), `src/exp.py`, `src/cv.py`, `src/lofo.py`, `src/wallet_split.py`, `src/shift_check.py`, `src/cases.py`: validation and analysis scripts. Run `src/run_baseline.py` first; it writes `cache/`.
- `AI_USE.md`: AI assistance disclosure. `THIRD_PARTY.md`: what we did not write. `LICENSE`: MIT.
