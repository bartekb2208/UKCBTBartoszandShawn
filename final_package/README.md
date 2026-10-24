# Model Phi: Decode & Detect entry (a blend of two models)

UKCBT x XRPL "Decode & Detect" hackathon, 24 October 2026. The entry ranks every test payment for review: scores run from 0 to 1, higher means more suspicious. A high score means "flagged for review", never a finding about any account.

- **Team and commit:** `TEAM.md` · **Licence:** MIT, `LICENSE` · **Method report:** `REPORT.md` · **Examples with hashes:** `CASES.md`
- **What we did not write:** `THIRD_PARTY.md` · **AI use (model, provider, prompts, settings):** `AI_USE.md` and `shawn/PROMPTS.md`

## Reproduce `predictions.csv`

Python 3.14.6. From this folder:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Put the organiser data next to this README:
#   decode-and-detect-training-data/train/   transactions.parquet, transfers.parquet, market_rates.csv, labels.csv
#   decode-and-detect-test-data/             transactions.parquet, transfers.parquet, market_rates.csv
PY=.venv/bin/python ./run_v5.sh               # writes predictions.csv
```

Other data locations: `TRAIN=<dir> TEST=<dir> PY=<python> ./run_v5.sh`.

**Time and hardware:** about 6 minutes on an Apple M4 laptop (10 cores, 16 GB): model A about 100 seconds, model B about 230 seconds, the blend a few seconds. CPU only, under 1 GB of memory, no network access. All seeds are fixed; repeated runs give byte-identical files.

`run_v5.sh` does three things:
1. `ours/src/predict.py` (model A) trains on `train/` and scores the test data.
2. `shawn/run.py` (model B) does the same with a different method.
3. `blend_v5.py` averages the two rank scores 50/50, ranks the result again so that all scores are unique and lie in [0, 1], and checks the file: every test transaction exactly once, no ties, scores in range.

The output `predictions.csv` has the columns `TRANSACTION_HASH,score` and an optional unscored `reason` column holding a short evidence note for the top 2%.

## What is in this ZIP

| Path | What it is |
|---|---|
| `ours/src/` | Model A: loader, features, LightGBM + ExtraTrees, folds, evaluation, and the validation scripts below |
| `shawn/` | Model B: loader, features, scenario reconstruction, two-stage LightGBM (run with `run.py`), its README, report and prompt record |
| `blend_v5.py`, `run_v5.sh` | The blend and the one-command runner |
| `analysis/` | Scripts for the optional blend-weight check |

No data, caches, virtual environments or credentials are included. There are no model weights to include: both models are trained from scratch on `train/` every time the code runs.

## Validation (optional; not needed for `predictions.csv`)

Run from this folder, in this order. The numbers are in `REPORT.md`.

```sh
python ours/src/run_baseline.py            # fold map, and the organiser's starter rules as a baseline (29.4); writes cache/
python ours/src/cv.py                      # model A, 5-fold grouped cross-validation (62.3); writes cache/oof_submission.csv
python ours/src/exp.py --rebuild           # caches features for the three checks below
python ours/src/lofo.py                    # leave one motif family out
python ours/src/wallet_split.py            # train and score on disjoint halves of the wallets
python ours/src/shift_check.py             # which features separate train/ from validation/ (labels not used)
python analysis/capture_shawn_oof.py <train dir> <test dir> <out dir>   # model B's out-of-fold scores
python analysis/blend_cv.py <out dir>/oof_arrays.npz <train dir>        # AP of the blend at each weight
```

## Notes

- The blend weight (0.5) was fixed before any test score existed. A later out-of-fold check gave 64.2 at 0.5 and 64.3 at the best weight (0.6), so the weight is not tuned.
- `ours/NOTES.md` is a chronological log of experiments, including ones that did not help; later rows supersede earlier ones. The final model A is the code in `ours/src/predict.py` (205 features, asset and conversion features left out, cross-validated AP 62.3).
- `shawn/README.md` was written for an earlier version of model B: its feature count (about 160), cross-validated AP (60.4) and run paths are out of date. Current figures are in `REPORT.md` and `shawn/report.md` (218 features, AP 61.6), and the way to run it is `run_v5.sh`.
- The data is synthetic. Nothing here is evidence about any real person or account.
