# What we did not write ourselves (terms cl. 3.4 and 6.1)

## Libraries (installed from PyPI, versions pinned in `requirements.txt`; none is copied into this ZIP)

| Library | Licence (read from the installed package metadata) | Used for |
|---|---|---|
| pandas 3.1.0 | BSD-3-Clause | tables |
| pyarrow 26.0.0 | Apache-2.0 | reading Parquet |
| numpy 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | arrays |
| scipy 1.18.1 | BSD (SciPy licence) | connected components, rank statistics |
| scikit-learn 1.9.1 | BSD-3-Clause | ExtraTrees, cross-validation splitters, average precision |
| lightgbm 4.7.0 | MIT | gradient-boosted trees |

All are open-source licences that permit commercial reuse. The libraries are installed, not redistributed, so their licence texts stay with the packages.

## Organiser material

- The datasets (`train/`, `validation/`, test) are the organisers' and are **not** included in this ZIP or the repository.
- `ours/src/dd/baseline.py` is a port of the 12 rules and logistic regression in the organiser's starter notebook. It is used only to report the baseline score (29.4 under our folds) and is **not** part of the prediction path.
- The USD valuation in `ours/src/dd/load.py` and `shawn/load.py` follows the rules in the organiser's data README and `payments.py` (hourly reference price, first executed transfer leg as the real cost). The code is our own.

## Models, weights, external data, APIs and services

None. No pre-trained model or weights, no external dataset, no API or network call, and no LLM call is used to make predictions. Every model is trained from scratch on `train/` each time the code runs, so there are no weights to include.

## Public research consulted

Shawn's design notes (`shawn/PROMPTS.md`, prompt 2) record two papers read as background: arXiv 2402.08593 (IBM Graph Feature Preprocessor) and arXiv 2306.11586 (Multi-GNN). They informed the choice of graph-pattern features plus gradient boosting. Neither project's library is imported anywhere in this code (the only third-party imports are the six libraries above).

## AI assistance

See `AI_USE.md` and `shawn/PROMPTS.md`.
