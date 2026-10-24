"""Rank-blend the two models' test predictions and validate the file against the submission rules.

Usage: python blend_v5.py <model A csv> <model B csv> <weight of A> <out csv> <test data dir>

Each input is rank-normalised, averaged with the given weight, then ranked again so that scores are
unique in [0, 1] (equal blended scores are ordered by transaction hash). The weight (0.5) was fixed
before any test score existed. The optional `reason` column is not scored; it is kept from model A
for the blended top 2%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent / "ours" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # repo layout
from dd.submit import rank_scores, validate  # noqa: E402

path_a, path_b, weight, out, test_dir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
a = pd.read_csv(path_a).set_index("TRANSACTION_HASH")
b = pd.read_csv(path_b).set_index("TRANSACTION_HASH").reindex(a.index)
assert b.score.notna().all(), "the two files do not cover the same transactions"


def rank01(x):
    return rankdata(x) / len(x)


ranked = rank_scores(weight * rank01(a.score) + (1 - weight) * rank01(b.score), a.index.to_numpy())
reason = np.where(ranked >= 0.98, a.reason.fillna("").to_numpy() if "reason" in a else "", "")
pred = pd.DataFrame({"TRANSACTION_HASH": a.index, "score": ranked, "reason": reason})
validate(pred, test_dir)
pred.to_csv(out, index=False, float_format="%.8f")
print(f"wrote {out}: {len(pred)} rows (weight of model A = {weight})")
