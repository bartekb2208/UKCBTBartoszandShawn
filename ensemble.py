"""Rank-average several prediction files written by run.py into one strict ranking (no ties).
Usage: python ensemble.py out_final.csv out/submission.csv out_nolinks/submission.csv [more.csv ...]
Each input gets equal weight; ties are broken by the first input's score."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from model import strict_ranking


def main():
    dst, srcs = sys.argv[1], sys.argv[2:]
    frames = [pd.read_csv(p).set_index("TRANSACTION_HASH")["score"] for p in srcs]
    idx = frames[0].index
    for f in frames[1:]:
        assert set(f.index) == set(idx), "inputs must cover the same transactions"
    ranks = [rankdata(f.reindex(idx).values) / len(idx) for f in frames]
    final = strict_ranking(np.mean(ranks, axis=0), frames[0].reindex(idx).values)
    out = pd.DataFrame({"TRANSACTION_HASH": idx, "score": final})
    assert out["score"].nunique() == len(out)
    out.to_csv(dst, index=False, float_format="%.10f")
    print(f"[ensemble] {len(srcs)} inputs -> {dst} ({len(out):,} rows, all scores unique)")


if __name__ == "__main__":
    main()
