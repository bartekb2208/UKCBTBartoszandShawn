"""Rank-blend two prediction CSVs for the test set.

Usage: python src/blend_test.py <ours.csv> <shawn.csv> <w_ours> <out.csv>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).parent))
from dd.submit import rank_scores, validate

ours, shawn, w, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
a = pd.read_csv(ours).set_index("TRANSACTION_HASH")
b = pd.read_csv(shawn).set_index("TRANSACTION_HASH").reindex(a.index)
assert b.score.notna().all(), "hash sets differ"
r = lambda x: rankdata(x) / len(x)
ranked = rank_scores(w * r(a.score) + (1 - w) * r(b.score), a.index.to_numpy())
reason = np.where(ranked >= 0.98, a.reason.fillna("").to_numpy() if "reason" in a else "", "")
pred = pd.DataFrame({"TRANSACTION_HASH": a.index, "score": ranked, "reason": reason})
validate(pred, "decode-and-detect-test-data")
pred.to_csv(out, index=False, float_format="%.8f")
print(f"wrote {out} (w_ours={w})")
