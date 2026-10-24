# Model Phi: Decode & Detect entry

UKCBT x XRPL "Decode & Detect" hackathon, 24 October 2026. The entry ranks every test payment for review: scores run from 0 to 1, higher means more suspicious. A high score means "flagged for review", never a finding about any account.

- **Team and commit:** `TEAM.md` · **Licence:** MIT, `LICENSE` · **Method report:** `REPORT.md` · **Examples with hashes:** `CASES.md`
- **What we did not write:** `THIRD_PARTY.md` · **AI use (models, providers, prompts, settings):** `AI_USE.md` and `shawn/PROMPTS.md`

## This version

__NOTE__

## Reproduce `predictions.csv`

Python 3.14.6. From this folder:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Put the organiser data next to this README:
#   decode-and-detect-training-data/train/   transactions.parquet, transfers.parquet, market_rates.csv, labels.csv
#   decode-and-detect-test-data/             transactions.parquet, transfers.parquet, market_rates.csv
PY=.venv/bin/python ./run.sh                   # writes predictions.csv
```

Other data locations: `TRAIN=<dir> TEST=<dir> PY=<python> ./run.sh`. `run.sh` lists every step; intermediate files go to `work/`.

**Time and hardware:** __TIME__ on an Apple M4 laptop (10 cores, 16 GB), CPU only, under 1 GB of memory per step, no network access. All seeds are fixed; a clean unzip of this ZIP rebuilt the submitted `predictions.csv` byte-for-byte (checked by MD5).

## What is in this ZIP

| Path | What it is |
|---|---|
| `ours/src/` | Model A: loader, features, relay-motif detectors (`dd/motifs.py`), LightGBM + ExtraTrees (`predict_v6.py`), self-training (`self_train_v12.py`), wrappers that add the motif features and an external teacher to model B (`run_shawn_with_motifs.py`, `run_shawn_cotrain.py`), and the validation scripts |
| `shawn/` | Model B: loader, features, scenario reconstruction, two-stage LightGBM with self-training (`run.py`), its README, report and prompt record. Its files are unchanged; the wrappers above add to it at run time |
| `blend_v5.py`, `blend_n.py` | Rank blends; each validates the output file (every test transaction once, unique scores in [0, 1]) |

No data, caches, virtual environments or credentials are included. There are no model weights to include: every model is trained from scratch on `train/` each time the code runs.

## Validation (optional; not needed for `predictions.csv`)

Run from this folder. Numbers are in `REPORT.md`.

```sh
python ours/src/run_baseline.py            # fold map, and the organiser's starter rules as a baseline (29.4); writes cache/
python ours/src/cv.py                      # model A, 5-fold grouped cross-validation (62.3 without motif features)
(cd shawn && python run.py --train ../decode-and-detect-training-data/train --test ../decode-and-detect-test-data --out ../out_b)   # model B; out_b/results.json
```

## Notes

- `ours/NOTES.md` is a chronological log of experiments, including ones that did not help; later rows supersede earlier ones.
- `shawn/README.md` describes model B in its own words: its run paths assume its own folder layout (use `run.sh` here), and its cross-validated AP of 60.4 predates the final version of model B (61.6; with the motif features 61.9).
- The data is synthetic. Nothing here is evidence about any real person or account.
