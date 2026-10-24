"""Reciprocal-rank fusion of several prediction CSVs: score = sum over inputs of 1 / (k + rank), rank 1 = most
suspicious. Rewards payments that any input ranks near the top, where average precision is decided.

Usage: python blend_rrf.py <test data dir> <out csv> <k> <csv> [<csv> ...]
The optional `reason` column is taken from the first input for the fused top 2%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent / "ours" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # repo layout
from dd.submit import rank_scores, validate  # noqa: E402

test_dir, out, k, paths = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
first = pd.read_csv(paths[0]).set_index("TRANSACTION_HASH")
fused = np.zeros(len(first))
for p in paths:
    s = pd.read_csv(p).set_index("TRANSACTION_HASH").score.reindex(first.index)
    assert s.notna().all(), f"{p} does not cover the same transactions"
    fused += 1.0 / (k + rankdata(-s.to_numpy(), method="ordinal"))
ranked = rank_scores(fused, first.index.to_numpy())
reason = np.where(ranked >= 0.98, first.reason.fillna("").to_numpy() if "reason" in first else "", "")
pred = pd.DataFrame({"TRANSACTION_HASH": first.index, "score": ranked, "reason": reason})
validate(pred, test_dir)
pred.to_csv(out, index=False, float_format="%.8f")
print(f"wrote {out}: reciprocal-rank fusion (k={k:g}) of {len(paths)} inputs")
