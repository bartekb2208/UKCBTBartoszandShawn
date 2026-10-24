# v3: our model v2 + one round of self-training on the test split
Top 500 test payments pseudo-labelled suspicious, bottom 60% normal, refit, rescore. No train-CV equivalent.
Reproduce: `python src/self_train.py --out predictions.csv`.
