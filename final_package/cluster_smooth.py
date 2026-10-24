"""Scenario-consensus smoothing of a predictions file.

Payments on rare pairs (a pair paying at most twice in the week) that touch the same wallet within 20 minutes are
linked into clusters; in every cluster of 2 to 25 payments each score moves a fraction `lambda` towards the mean
score of the cluster's other payments. Planted scenarios share one label, so the consensus repairs partial misses.
On out-of-fold scores in training this raised pooled AP from 64.9 to 66.3 at lambda = 0.10 (every fold improved).

Usage: python cluster_smooth.py <predictions.csv> <test data dir> <out.csv> [lambda=0.10] [cap=25] [gap_s=1200]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for p in (Path(__file__).resolve().parent / "ours" / "src", Path(__file__).resolve().parent.parent / "src"):
    sys.path.insert(0, str(p))
from dd.features import motif_clusters, prepare  # noqa: E402
from dd.load import load_split  # noqa: E402
from dd.submit import rank_scores, validate  # noqa: E402

src, test_dir, out = sys.argv[1:4]
lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
cap = int(sys.argv[5]) if len(sys.argv) > 5 else 25
gap = float(sys.argv[6]) if len(sys.argv) > 6 else 1200
te = prepare(load_split(test_dir))
pred = pd.read_csv(src).set_index("TRANSACTION_HASH").reindex(te.hash)
s = pred.score.to_numpy()
pair_n = np.asarray(pd.MultiIndex.from_arrays([te.src, te.dst]).map(te.groupby(["src", "dst"]).size()))
idx = np.flatnonzero(pair_n <= 2)
cid = -1 - np.arange(len(te))                                   # every other payment is its own singleton
cid[idx] = motif_clusters(te.iloc[idx].reset_index(drop=True), gap)
g = pd.Series(s).groupby(cid)
size, total = g.transform("size").to_numpy(), g.transform("sum").to_numpy()
others = np.where(size > 1, (total - s) / np.maximum(size - 1, 1), s)
ok = (size > 1) & (size <= cap)
smoothed = np.where(ok, (1 - lam) * s + lam * others, s)
ranked = rank_scores(smoothed, te.hash.to_numpy())
reason = pred.reason.fillna("").to_numpy() if "reason" in pred else np.full(len(te), "")
res = pd.DataFrame({"TRANSACTION_HASH": te.hash.to_numpy(), "score": ranked, "reason": reason})
validate(res, test_dir)
res.to_csv(out, index=False, float_format="%.8f")
print(f"smoothed {ok.sum()} payments in {pd.Series(cid[ok]).nunique()} clusters (lambda={lam}, cap={cap}, gap={gap:.0f}s); wrote {out}")
