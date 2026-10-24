"""Variant: one round of self-training on the test split.

Fit on train/, score test/, add the most confident test payments as pseudo-labels
(top as suspicious, bottom as normal), refit, score test/ again.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from dd.features import build_features
from dd.load import load_split
from dd.submit import rank_scores, reasons, validate
from predict import score

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="decode-and-detect-training-data/train")
ap.add_argument("--test", default="decode-and-detect-test-data")
ap.add_argument("--out", default="predictions_selftrain.csv")
ap.add_argument("--top", type=int, default=500)
ap.add_argument("--bottom", type=float, default=0.6)
a = ap.parse_args()
t0 = time.time()
train = load_split(a.train)
labels = pd.read_csv(Path(a.train) / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
labels = train[["hash"]].merge(labels, on="hash", how="left", validate="1:1")
test = load_split(a.test)
X_train, X_test = build_features(train), build_features(test)
s0 = score(X_train, labels, X_test)
order = np.argsort(-s0)
pos, neg = order[: a.top], order[int(len(order) * (1 - a.bottom)):]
X_aug = pd.concat([X_train, X_test.iloc[pos], X_test.iloc[neg]], ignore_index=True)
lab_aug = pd.DataFrame({"is_suspicious": np.r_[labels.is_suspicious.to_numpy(), np.ones(len(pos)), np.zeros(len(neg))]})
s1 = score(X_aug, lab_aug, X_test)
ranked = rank_scores(s1, test.hash)
pred = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": ranked, "reason": reasons(X_test, ranked)})
validate(pred, a.test)
pred.to_csv(a.out, index=False, float_format="%.8f")
print(f"wrote {a.out} ({time.time() - t0:.0f}s); rank corr with round 0: {pd.Series(s0).corr(pd.Series(s1), method='spearman'):.4f}")
