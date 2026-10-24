# Build notes (ledger)

Pooled out-of-fold AP on train/ under the fixed 5-fold map (`cache/fold_map.csv`). Fold spread is about ±11 because relay positives are unevenly spread over folds, so treat changes under ~1 point as noise.

## Data facts checked

- 700 wallets in train, all in ONE connected component; each wallet appears in about 24 scenarios. No wallet belongs to only one scenario.
- `cohort_id` is set only for `aggregate_relay` and `distributed_relay` (12 cohorts × 100 scenarios; exactly 1 planted scenario per cohort; all scenarios in a cohort have the same size).
- Every scenario is all-suspicious or all-normal (decoy). 1,497 scenarios.
- Conversions: 2 legs (direct) or 3 legs (XRP bridge). Leg 0 is the sender's actual cost. Transfer `VALUE` is in asset units (XRP, not drops).
- `SEQUENCE` is a clean per-account counter; FLAGS, MEMOS, SOURCE_TAG and LastLedgerSequence-minus-ledger are constant. `SOURCE_FEE` and `TRANSACTION_INDEX` have no label signal. No metadata artefacts found.
- Issuer addresses differ between splits, so assets enter features by currency code only.

## Rulings

- Folds: wallet-cluster grouping is impossible (one component). Group relay rows by cohort, other planted rows by scenario, background by sender; stratify on family × label. Cost if wrong: CV over-optimistic through wallets shared across folds.
- Relay families look unpredictable from transaction features (relay-only model AP ≈ base rate 1%). Planted hubs pass on value more evenly per episode (planted scenario ranks ~22/100 in its cohort on worst-episode pass-through error), but that needs true scenario membership. Effort goes to the other families instead. Cost if wrong: up to ~19 AP left on the table (perfect relay ranking → 75).
- Dropped 1-hour motif clusters: they form a giant component whose statistics fingerprint the split (shift AUC 1.0). CV rose.
- Cluster statistics are kept only for clusters of up to 60 payments; larger ones get a `giant` flag. CV +1.4, shift gone.
- Weekly wallet counts are within-split percentiles (validation/ has 15% fewer payments). Remaining multivariate shift AUC 0.88 comes from short-window counts in a quieter week; accepted.
- Stage-2 neighbour-score smoothing gave no gain; not used.
- Relay, second look (test day): within its own scenario a planted relay hub always receives first and never pays out more than it has received (min running balance ≥ 0; only 20% of decoys satisfy this). But even with oracle relay membership, per-wallet episode balance gives within-relay AUC 0.58: each wallet mixes several overlapping relay scenarios in the same hours (a cohort's 100 scenarios fire together). Abandoned. Cohort timing does not single out the planted scenario (checked all 12).
- Asset and conversion features dropped (v2): the README says asset and conversion choices do not inspect labels, and test/ converts more (legs 1/2/3 = 50/38/12% vs train 59/31/10%). LGBM CV 61.2 → 60.7 (noise). Cost if wrong: about 0.5 AP.
- Final model: binary LightGBM + ExtraTrees rank blend, trained on all rows. The relay gate and ranking objectives were within noise, so the simplest option was kept. Cost if wrong: under 1 AP.

## Results log

| Run | Pooled AP | Fold mean ± sd | Notes |
|---|---|---|---|
| Starter LR (12 rules) | 29.4 | 32.4 ± 10.2 | baseline |
| LGBM, 151 features | 56.0 | 59.3 ± 11.4 | relay ≈ 1 |
| + relay gate B×(1−A) | 57.8 | 60.0 ± 11.3 | |
| + shape/novelty features | 57.9 | 61.8 ± 11.1 | single model |
| − 1h clusters | 58.3 | 62.0 ± 11.5 | |
| + cluster cap 60 | 59.6 | 62.4 ± 10.5 | |
| + first-time-pair features and clusters | 61.4 | 62.9 ± 9.9 | biggest single gain after the base features |
| novel clusters at 10/80 min, cap 30 | 60.7 | 62.5 ± 9.8 | reverted |
| + slow chain (2 h, 40%) | 61.45 | 63.1 ± 9.7 | kept (neutral) |
| learned scenario linker (pair model → clusters) | 61.1 | 63.1 ± 11.3 | dropped; oracle true-scenario stats give only 64.8 |
| relay gate B×(1−A) on current features | 61.2 | 62.3 ± 10.5 | dropped |
| value-conservation features | 61.2 | 62.5 ± 10.1 | dropped (relay-only AP 1.65 vs base 1.0) |
| rank_xendcg / lambdarank (random 2k-row groups) | 60.3 / 60.9 | | not better than binary |
| ExtraTrees alone | 59.7 | 61.8 ± 10.5 | |
| **Final: LGBM ×3 seeds + ExtraTrees, rank blend 0.65/0.35** | **62.0** | **63.5 ± 10.1** | `src/cv.py`, deterministic |

## Checks

- Leave-one-family-out (`src/lofo.py`): held-out family positives vs background AUC 0.95–1.00; within-family AP 73–93.
- Wallet-disjoint halves (`src/wallet_split.py`): all features 64.8 ± 1.6, without weekly wallet features 64.7 ± 1.6 → no wallet memorisation.
- Reproducibility: two runs on validation/ give byte-identical CSVs after (a) sorting feature columns (train/test column order differed: real bug, fixed and asserted), (b) LightGBM `deterministic=True`, (c) single-threaded ExtraTrees prediction.
