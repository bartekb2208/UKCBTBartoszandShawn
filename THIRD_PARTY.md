# What we did not write ourselves

## Libraries (installed from PyPI; versions pinned in `requirements.txt`)

| Library | Licence | Used for |
|---|---|---|
| pandas | BSD-3-Clause | tables |
| pyarrow | Apache-2.0 | Parquet reading |
| numpy | BSD-3-Clause | arrays |
| scipy | BSD-3-Clause | connected components, rank blending |
| scikit-learn | BSD-3-Clause | ExtraTrees, fold splitter, average precision |
| lightgbm | MIT | gradient-boosted trees |

## Organiser material

- Data in `decode-and-detect-training-data/` and `test-data/` (not included in the ZIP).
- `src/dd/baseline.py` is a port of the organiser starter notebook's 12 rules, used only as the baseline to beat. `src/dd/load.py` follows the valuation rules in the organiser README and `payments.py`.

## Pre-trained models and outside services

None. No pre-trained model, external API or outside data is used to make predictions.

## AI assistance

The code was written with an AI coding assistant (Claude Opus 5.5 by Anthropic, via Claude Code). See `AI_USE.md`.
