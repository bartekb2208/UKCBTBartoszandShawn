"""Rank-blend several prediction CSVs with fixed weights and validate the result.

Usage: python blend_n.py <test data dir> <out csv> <csv>:<weight> [<csv>:<weight> ...]

Each input is rank-normalised, the weighted average is ranked again so that scores are unique in [0, 1]
(equal blended scores are ordered by transaction hash). The optional `reason` column is not scored; it is
taken from the first input for the blended top 2%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent / "ours" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # repo layout
from dd.submit import rank_scores, validate  # noqa: E402

test_dir, out = sys.argv[1], sys.argv[2]
inputs = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in sys.argv[3:]]
first = pd.read_csv(inputs[0][0]).set_index("TRANSACTION_HASH")
blend = np.zeros(len(first))
for path, w in inputs:
    s = pd.read_csv(path).set_index("TRANSACTION_HASH").score.reindex(first.index)
    assert s.notna().all(), f"{path} does not cover the same transactions"
    blend += w * rankdata(s) / len(s)
ranked = rank_scores(blend / sum(w for _, w in inputs), first.index.to_numpy())
reason = np.where(ranked >= 0.98, first.reason.fillna("").to_numpy() if "reason" in first else "", "")
pred = pd.DataFrame({"TRANSACTION_HASH": first.index, "score": ranked, "reason": reason})
validate(pred, test_dir)
pred.to_csv(out, index=False, float_format="%.8f")
print(f"wrote {out}: {len(pred)} rows from " + ", ".join(f"{Path(p).name} x {w}" for p, w in inputs))
