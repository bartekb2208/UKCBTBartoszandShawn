"""Lift the ultra-regular relay motifs into the review band of an existing predictions file.

Usage: python motif_boost.py <predictions.csv> <test dir> <out.csv> <floor> [dist|dist+agg]
A payment is lifted when it belongs to a detected motif repeated at least three times with timing spread under
0.15 h: distributed bridges must pass the value checks at every repeat; aggregate batches must also conserve
value within 10% on average. Lifted rows get score max(score, floor); then everything is re-ranked.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for p in (Path(__file__).resolve().parent / "ours" / "src", Path(__file__).resolve().parent.parent / "src"):
    sys.path.insert(0, str(p))
from dd.load import load_split  # noqa: E402
from dd.motifs import motif_features  # noqa: E402
from dd.submit import rank_scores, validate  # noqa: E402

src, test_dir, out, floor = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
which = sys.argv[5] if len(sys.argv) > 5 else "dist"
te = load_split(test_dir)
M = motif_features(te)
dist = ((M.distributed_bridge_repeat_count >= 3) & (M.distributed_pass_fraction >= 1) & (M.distributed_delay_consistency < 0.15)).to_numpy()
agg = ((M.aggregate_relay_repeat_count >= 3) & (M.aggregate_delay_consistency < 0.15) & (M.aggregate_value_deviation < 0.10)).to_numpy()
lift = dist | agg if which == "dist+agg" else dist
pred = pd.read_csv(src).set_index("TRANSACTION_HASH").reindex(te.hash)
score = np.where(lift, np.maximum(pred.score.to_numpy(), floor), pred.score.to_numpy())
ranked = rank_scores(score, te.hash.to_numpy())
reason = pred.reason.fillna("").to_numpy() if "reason" in pred else np.full(len(te), "")
reason = np.where(lift, "flagged for review: repeated relay pattern (three or more batches, timing spread under 10 minutes, value conserved)", reason)
res = pd.DataFrame({"TRANSACTION_HASH": te.hash.to_numpy(), "score": ranked, "reason": reason})
validate(res, test_dir)
res.to_csv(out, index=False, float_format="%.8f")
print(f"lifted {lift.sum()} rows ({which}) to at least {floor}; wrote {out}")
