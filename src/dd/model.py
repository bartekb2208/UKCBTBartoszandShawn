"""LightGBM ranker trained out-of-fold."""
import lightgbm as lgb
import numpy as np

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=600, verbose=-1, seed=7, num_threads=8, deterministic=True, force_row_wise=True)


def fit(X, y, params=None):
    p = {**PARAMS, **(params or {})}
    return lgb.LGBMClassifier(**p).fit(X, y)


def oof_predict(X, y, folds, params=None):
    oof = np.zeros(len(X))
    for k in np.unique(folds):
        m = fit(X[folds != k], y[folds != k], params)
        oof[folds == k] = m.predict_proba(X[folds == k])[:, 1]
    return oof
