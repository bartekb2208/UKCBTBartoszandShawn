# Primer for the build session

Read this first, then `PLAN.md`. This is context carried over from a planning chat; it is not a spec of its own.

## Task in one line

Rank every test transaction by suspicion. Score = 100 x Average Precision on private organiser labels. Submit a predictions CSV and a code ZIP at https://hackathon.ukcbt.org by **16:30 London, 24 October 2026**. 50 uploads per team per UTC day; the leaderboard shows the best completed run and stays provisional until verified.

## Working agreement with the user

- **Nothing is built yet.** The plan was finalised first. Before writing code, use the superpowers skills (brainstorming, then writing-plans) and confirm the plan with the user.
- **Do not show the data to a model, and do not train, until the plan is agreed.** Profiling counts and schemas is fine.
- Validation must be **5-fold**, grouped so wallets and scenarios do not straddle folds, imitating test conditions (see `PLAN.md`).
- The user's teammate is ideating a competing approach. This plan is being judged against it on how well it meets the criteria, so keep the criteria-coverage table in `PLAN.md` honest and current.
- The user did not read earlier long answers in full while supplying screenshots. Keep status updates short and plain.

## Environment

- Repo root: `/Users/bartek/Documents/GitHub/Hackathons/UKCBTBartoszandShawn`
- Python 3.14.6; a venv exists at `.venv/` with pandas, pyarrow, scikit-learn, lightgbm, networkx. Use `.venv/bin/python`.
- `.gitignore` covers `*.pptx`, `*.ppt`, `*.pdf`, `decode-and-detect-training-data/`, `.venv/`, `plan.html`, `.DS_Store`. Add `test-data/` when it arrives. Never commit data.
- Remote: https://github.com/bartekb2208/UKCBTBartoszandShawn.git, branch `main`. The two training slide files were committed once (commit `2ce1df3`) and later untracked; they remain in history.
- `.gitignore` edits since the last push are uncommitted. The user asked **not** to commit `plan.html`. `PLAN.md` and `PRIMER.md` are untracked; do not commit them unless asked.
- Training slides and PDF in the repo root (read earlier): `Decode & Detect AML Training - final.pptx` and `XRPL AI Hack Training - UKCBT - October 2026.pdf`.

## Where things are

- Data: `decode-and-detect-training-data/` (see `PLAN.md`). `train/` is labelled; `validation/` is unlabelled practice data that came with the download.
- Test data: separate download, to be placed in `test-data/` next to `train/`. Not present at the time of writing.
- `starter_notebook.ipynb`: logistic regression on 12 hand-made rules (windows of 10 min and 1 hour); holds back 20%; the baseline to beat under our own folds. `payments.py` builds one USD-valued edge per transaction from `transactions.parquet` and `market_rates.csv`. In its output, `SOURCE_VALUE` is in micro-USD and `VALUE` is in USD (reading the code; not run). Check units before reusing it.
- Published plan page (private): https://claude.ai/artifact/MkcGPkBMm79MxFcAzkdLcW (source `plan.html`).

## Facts worth remembering from the briefing

- The test data is synthetic, in the AWS/SonarX XRPL schema. A high score does not prove a method works on real crime.
- Reference prices are valuations, not executable quotes; actual conversion costs include variable spreads and integer rounding.
- Hard cases span several payments and changing counterparties over longer histories: use whole-week and graph context, not single amounts.
- Judges' six suggested ideas: mule network finder, money trail tracker, limit-dodge detector, fake trading detector, alert ranker, LLM case summary writer. Only the LLM one cannot raise the score.
- Rules: organiser data only; any language or model; AI use allowed but must be disclosed (model, provider, prompts, settings); work must be built on the day; reproducible or the entry may be rejected; released under an approved open-source licence by the deadline; every team member accepts the terms (https://hackathon.ukcbt.org/terms).
- The XRPL deck's pitch advice: show two or three flagged cases with real transaction hashes, show what you deliberately do not flag, be honest about limits.
- The PDF deck mentions pulling real Mainnet data, but the competition rules say organiser data only. Follow the rules.

## Unverified and to check early

- Whether wallet connected components merge into one giant group (hub-aware grouping).
- Whether `cohort_id` is populated only for the two relay families (inferred from group counts, not checked directly).
- Exact column semantics in `transfers.parquet` for conversions (2 or 3 legs; first leg = actual cost).
- That folds track the test score: compare fold AP with the first uploaded test score.

## First actions

1. Confirm the plan with the user; check `test-data/` is present.
2. Write the loader (USD valuation, JSON parsing) and the fold map; check component sizes first.
3. Run the starter baseline under the 5 folds; record mean AP, spread and per-family AP.
4. Upload one valid CSV early to confirm the format.
5. Start `AI_USE.md` now and keep it updated.
