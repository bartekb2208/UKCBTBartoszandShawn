"""Self-training of model A (+ relay motifs) on the test week, with pseudo-labels from a teacher ranking.

The teacher's top payments are pseudo-labelled suspicious and its bottom share normal, added to training at
half weight, and the test week is re-scored with 2-fold cross-fitting: each half is scored by models that never
saw that half's pseudo-labels. Output = 0.5 x rank(self-trained) + 0.5 x rank(model A without self-training).

Usage: python src/self_train_v12.py --teacher <csv> --out <csv> [--top 600] [--neg 0.4] [--union]
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dd.load import load_split  # noqa: E402
from dd.model import PARAMS  # noqa: E402
from dd.submit import rank_scores, reasons, validate  # noqa: E402
from predict import ET_WEIGHT, SEEDS  # noqa: E402
from predict_v6 import assemble  # noqa: E402

SEED = 20261024


def score_weighted(X, y, w, Xt):
    """The model A recipe (LightGBM x3 seeds + ExtraTrees, rank blend) with sample weights."""
    gbm = np.zeros(len(Xt))
    for seed in SEEDS:
        m = lgb.LGBMClassifier(**{**PARAMS, "seed": seed}).fit(X, y, sample_weight=w)
        gbm += m.predict_proba(Xt)[:, 1]
    et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.3, n_jobs=8, random_state=0)
    et = et.fit(X.fillna(-999), y, sample_weight=w)
    et.n_jobs = 1
    p_et = et.predict_proba(Xt.fillna(-999))[:, 1]
    return (1 - ET_WEIGHT) * rankdata(gbm) / len(gbm) + ET_WEIGHT * rankdata(p_et) / len(p_et)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="decode-and-detect-training-data/train")
    ap.add_argument("--test", default="decode-and-detect-test-data")
    ap.add_argument("--teacher", required=True, help="predictions CSV whose ranking supplies the pseudo-labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--neg", type=float, default=0.4, help="bottom share of the teacher ranking used as normal")
    ap.add_argument("--union", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    train, test = load_split(a.train), load_split(a.test)
    labels = pd.read_csv(Path(a.train) / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
    y = train[["hash"]].merge(labels, on="hash", how="left", validate="1:1").is_suspicious.to_numpy()
    _, U_train = assemble(train, a.train, a.union)
    X_test, U_test = assemble(test, a.test, a.union)
    assert list(U_train.columns) == list(U_test.columns)

    base = score_weighted(U_train, y, np.ones(len(y)), U_test)        # model A without self-training
    teacher = pd.read_csv(a.teacher).set_index("TRANSACTION_HASH").score.reindex(test.hash).to_numpy()
    r = rankdata(-teacher)
    pos, neg = r <= a.top, r >= (1 - a.neg) * len(r)
    halves = np.random.RandomState(SEED).randint(0, 2, len(test))
    st = np.zeros(len(test))
    for h in (0, 1):
        use = (pos | neg) & (halves != h)
        X = pd.concat([U_train, U_test[use]], ignore_index=True)
        yy = np.r_[y, pos[use].astype(int)]
        w = np.r_[np.ones(len(y)), np.full(int(use.sum()), 0.5)]
        st[halves == h] = score_weighted(X, yy, w, U_test[halves == h])
    final = 0.5 * rankdata(st) / len(st) + 0.5 * rankdata(base) / len(base)
    ranked = rank_scores(final, test.hash)
    pred = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": ranked, "reason": reasons(X_test, ranked)})
    validate(pred, a.test)
    pred.to_csv(a.out, index=False, float_format="%.8f")
    print(f"pseudo-labels: {pos.sum()} suspicious, {neg.sum()} normal; wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
