"""Train on train/ and write a predictions CSV for a test split.

Usage: python src/predict.py --train <train dir> --test <test dir> --out predictions.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).parent))
from dd.features import build_features  # noqa: E402
from dd.load import load_split  # noqa: E402
from dd.model import fit  # noqa: E402
from dd.submit import rank_scores, reasons, validate  # noqa: E402

SEEDS = (7, 11, 23)
ET_WEIGHT = 0.35


def score(X_train, labels, X_test):
    """Rank blend of seed-bagged LightGBM and ExtraTrees, both trained on all rows."""
    y = labels.is_suspicious.to_numpy()
    gbm = np.zeros(len(X_test))
    for seed in SEEDS:
        gbm += fit(X_train, y, dict(seed=seed)).predict_proba(X_test)[:, 1]
    et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.3, n_jobs=8, random_state=0)
    et = et.fit(X_train.fillna(-999), y)
    et.n_jobs = 1  # threaded prediction sums trees in a varying order; keep output byte-stable
    et = et.predict_proba(X_test.fillna(-999))[:, 1]
    return (1 - ET_WEIGHT) * rankdata(gbm) / len(gbm) + ET_WEIGHT * rankdata(et) / len(et)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="decode-and-detect-training-data/train")
    ap.add_argument("--test", default="test-data")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    t0 = time.time()
    train = load_split(a.train)
    labels = pd.read_csv(Path(a.train) / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
    labels = train[["hash"]].merge(labels, on="hash", how="left", validate="1:1")
    test = load_split(a.test)
    X_train, X_test = build_features(train), build_features(test)
    assert list(X_train.columns) == list(X_test.columns), "train/test feature columns differ"
    print(f"features built: train {X_train.shape}, test {X_test.shape} ({time.time() - t0:.0f}s)")
    s = score(X_train, labels, X_test)
    ranked = rank_scores(s, test.hash)
    pred = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": ranked, "reason": reasons(X_test, ranked)})
    validate(pred, a.test)
    pred.to_csv(a.out, index=False, float_format="%.8f")
    print(f"wrote {len(pred)} rows to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
