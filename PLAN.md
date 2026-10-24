# Decode & Detect: Plan

UKCBT x XRPL hackathon, 24 October 2026, IDEALondon. Submission closes **16:30 London**; aim to upload by 16:00.

## Objective

Maximise **100 x Average Precision (AP)** on the 40,101-row test set. Only the ranking order counts.

Approach: a LightGBM transaction ranker on wallet and graph features, a wallet-level score blended in by rank, and a few readable rules for the report. Validate with 5-fold grouped cross-validation.

## What the data is

Folder `decode-and-detect-training-data/`:

- `README.md`, `payments.py`, `starter_notebook.ipynb`, `checksums.json`
- `train/`: `transactions.parquet`, `transfers.parquet`, `blocks.parquet`, `labels.csv`, `market_rates.csv`
- `validation/`: same files, **no labels** (practice data, not scored; it came with the download)
- Test data is a separate download, to be unzipped into `test-data/` (not present yet). 40,101 transactions.

Facts (from the README and a profile of `labels.csv`; nothing has been modelled):

- Train: 40,131 transactions, 60,557 transfer legs, 1,250 suspicious (3.1%).
- Wallets and hashes are disjoint between splits. Test has batch variations absent from training.
- Suspicious mix: easy 800, medium 225, hard 225. Of the hard ones, 25 are `ambiguous` with labels independent of any feature (do not chase them).
- `labels.csv` columns: `TRANSACTION_HASH, is_suspicious, family, difficulty, scenario_id, cohort_id`. The last four are not available at test time. Use them for analysis and fold building only.

| Family | Suspicious | Rows | Note |
|---|---|---|---|
| aggregate_relay | 102 | 10,200 | 1% positive, main false-positive trap |
| distributed_relay | 98 | 9,800 | same |
| fan_out_in | 178 | 330 | |
| multi_stage | 173 | 338 | longer flows |
| bipartite | 134 | 286 | |
| stack | 116 | 220 | |
| fan_in / fan_out | 101 / 92 | 177 / 167 | |
| cycle / chain / random | 78 / 73 / 80 | 191 / 139 / 258 | |
| ambiguous | 25 | 25 | unpredictable by design |
| background | 0 | 18,000 | |

About 20,000 rows sit in the two relay families, with about 200 positives. Ranking those above their look-alikes decides most of the AP.

## Data rules and traps (README + slides)

1. One score per **transaction**, never per transfer. Join transfers on `TRANSACTION_HASH` (many to one).
2. XRP amounts are in drops (1 XRP = 1,000,000). For XRP rows `SOURCE_VALUE` is in drops.
3. An asset is `TOKEN_ADDRESS` plus `SYMBOL`. USD from two issuers is two assets.
4. `SendMax` is a ceiling, not the cost. Read the first executed transfer leg for the actual cost. `DELIVERED_AMOUNT` is what arrived.
5. Value in USD using `market_rates.csv`: `PRICE_MICRO_USD / 1e6`, hour containing the timestamp (`ts // 3600 * 3600`). `ASSET` is `XRP` or `issuer:currency`.
6. `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT` are JSON text.
7. Every fee is burned in XRP (10 to 15 drops), including token payments.
8. `payments.py` gives one effective sender-to-recipient edge per transaction, valued in USD. The raw transfer legs remain available for detail.
9. Which split a row came from is **not** a feature. No external data, labels, generator seeds or organiser files.

## Validation design (5-fold)

- **Main score:** 5-fold stratified, grouped. Rows that share wallets must stay in the same fold. Planted rows group by `scenario_id`; normal rows group by wallet cluster. Stratify on the label (about 250 positives per fold). Fixed seed; save the fold map to disk and reuse it for every experiment.
- **Open question, first task:** in a one-week network with exchange and payroll hubs, connected components may merge into one giant group. Check the component-size distribution. If so, remove very high-degree hubs before computing components, and fall back to `scenario_id` for planted rows. Fix the fold rule before any modelling.
- **Stress test:** leave-one-family-out (hold out a whole family, fit on the rest). Used to choose features, not for the headline score.
- **Shift check (labels unused):** train a classifier to separate `train/` from `validation/` on features only. Features that separate them easily are shift risks.
- Build graph features **once over a whole split**, as will be done for test. Folds split only the model fit.
- Choose features, model and hyperparameters from fold scores only. Report mean AP and per-fold spread; a change smaller than the spread is noise. Report per-family AP on every run.
- Seal one or two families for a single final check before freezing.
- Any rule threshold is chosen inside the training folds.
- A feature with outsized importance needs an explanation (it may be a generator artefact such as amount precision or ledger-index patterns). Test with and without it.

## Feature families

All behavioural; wallet addresses never enter the model. Windows of 10 minutes, 1 hour, 24 hours and the full week.

| Family | Features | Targets |
|---|---|---|
| Value and asset | USD value, asset pair, conversion spread (first-leg cost vs delivered), leg count, fee | all |
| Wallet activity | sender/receiver counts, totals, distinct counterparties, first-seen gap | fan-in, fan-out |
| Pass-through | forwarded share of received value, time to forward, similar-amount forwarding at several tolerances | relay, chain |
| Path and depth | longest upstream/downstream chain, one-use middle wallets, value kept per hop, delay per hop | stack, multi-stage, chain |
| Group shape | dense two-set blocks, repeated two-hop paths between a pair, source-out vs destination-in value | bipartite, scatter-gather |
| Cycles | return to start within a window, loop length, value returned vs sent, asset change | cycle |
| Amount structure | payout similarity within a sender, just-under-a-level clustering, round amounts | fan-out |
| Relay look-alikes | timing regularity, counterparty churn, amount spread, recurrence across the week | aggregate/distributed relay |
| Cluster context | cluster size, density, in vs out value, out-of-fold neighbour suspicion | hard, multi-stage |

## Model stack

1. **LightGBM** on all features. Try the binary objective, then a ranking objective; keep the higher fold AP.
2. **Wallet-level view:** score wallets/clusters on aggregated features, project onto transactions, rank-blend with layer 1. Out-of-fold only.
3. **Readable rules:** 5 to 8 one-line flags (for example forwarded 90% within 10 minutes). For the report, and a blend term only if they raise fold AP.

Baseline to beat: the starter notebook (logistic regression, 12 rules), run under our own folds. Final scores: rank-normalised into [0, 1], unique values, no ties.

## Judges' six ideas, mapped

| Idea | Role |
|---|---|
| Mule network finder | Core: cluster wallets, score the cluster, project back |
| Money trail tracker | Core: depth, chains, one-use middles, value per layer |
| Alert ranker | Core: this is the task; ranking objective, rank blend |
| Limit-dodge detector | Feature: amount similarity inside fan-out/bipartite |
| Fake trading detector | Feature: cycles, round trips, asset hopping |
| Case summary writer (LLM) | Pitch only. No effect on AP. Must not feed predictions; disclose if used |

## Timeline (offsets from start; assumes 08:30 start, adjust)

1. +0:00 to +0:30: plan locked, test data in `test-data/`.
2. +0:30 to +1:30: loader with USD valuation, fold map (hub-aware), starter baseline under 5 folds. **Gate:** upload a valid CSV early to confirm format.
3. +1:30 to +4:30: add feature families one at a time; keep those beating fold spread; leave-one-family-out after each batch. **Gate at +3:00:** if fold AP is flat for two rounds, switch effort to relay look-alikes.
4. +4:30 to +6:00: wallet-level blend, rank blend, shift check, sealed-family check. **Gate:** freeze features.
5. +6:00 to 15:00: reproduce from a clean checkout in one command; output must match the chosen CSV.
6. 15:00 to 16:00: package and upload.

## Risks

| Risk | Response |
|---|---|
| Relay look-alikes flood the top | Relay features, per-family AP each run, relay-only model as a blend term |
| Folds leak via shared wallets | Hub-aware grouping, leave-one-family-out, compare fold AP to first test score |
| Test has unseen variations | Shape-based features, shift check, stability across folds |
| Generator artefacts used as signal | Question outsized features, ablate |
| Over-tuning to folds | Cap experiments per fold set, sealed family |
| Reproduction fails late | Reproduce at +6:00, pin versions and seeds from run one |

## Deliverables

- **Predictions CSV:** `TRANSACTION_HASH,score`; all 40,101 rows once; scores in [0, 1]; at most 12 MiB; optional `reason` column is unscored. Validate locally before every upload.
- **Code ZIP** (8 MiB compressed, 64 MiB uncompressed, 2,000 files; no data, no virtualenv, no secrets; no encrypted/ZIP64/multi-part): all source; README with the exact command and run time; pinned dependency list and Python version; `LICENSE` (MIT or BSD-3); list of what we did not write (libraries, models, services, AI assistance); `AI_USE.md` with model, provider, prompts and settings; team name, captain, final member list, commit or version, validation results.
- **Method write-up:** under 10 slides or up to 4 pages covering methodology, features, model, validation, results, findings and limitations.
- Responsible language: say "flagged for review" and show evidence; never call an address criminal.

## Status (12:10, 24 Oct)

Built and verified on train/ and validation/. Out-of-fold AP **62.0** (starter baseline under the same folds: 29.4). Waiting for `test-data/`; then `PY=.venv/bin/python ./run.sh` and upload `predictions.csv` with `cache/code.zip` (rebuild with `./make_zip.sh`). Changes from this plan, all logged in `NOTES.md`: folds group relay rows by cohort (train is one connected component of 700 wallets, so wallet clusters can't be used); the wallet-level blend and stage-2 smoothing were tried and dropped (no fold gain); the final model is LightGBM plus ExtraTrees with a rank blend.

## Criteria coverage

| Criterion | How met |
|---|---|
| Highest AP wins; top three verified entries win | OOF AP 62.0 vs baseline 29.4; rank-normalised unique scores; tuned on folds only. Relay families (200 positives) remain unsolved |
| Reproducible or may be rejected | Done: `run.sh`, pinned deps, fixed seeds; clean unzip run gives a byte-identical CSV (MD5) |
| Team must understand the solution | Features tied to named shapes; rules and importance table |
| Write-up under 10 slides / 4 pages | Material ready in README (method, validation, limits) and CASES.md (flagged, not flagged, missed); slides not made |
| AI use disclosed | `AI_USE.md` written; planning-chat model still to fill in |
| Open source, built on the day | MIT licence; organiser data only |
| Limit false positives | Relay features; per-family false-positive table |
