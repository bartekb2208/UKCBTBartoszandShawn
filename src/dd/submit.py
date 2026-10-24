"""Turn raw scores into an uploadable CSV and check it against the site's rules."""
from pathlib import Path

import numpy as np
import pandas as pd


def rank_scores(raw, hashes):
    """Rank-normalise into [0, 1] with no ties; equal raw scores are ordered by hash."""
    order = np.lexsort((np.asarray(hashes), raw))
    out = np.empty(len(raw))
    out[order] = np.arange(len(raw)) / (len(raw) - 1)
    return out


def validate(pred, test_dir):
    tx = pd.read_parquet(Path(test_dir) / "transactions.parquet", columns=["TRANSACTION_HASH"])
    assert list(pred.columns[:2]) == ["TRANSACTION_HASH", "score"], pred.columns
    assert pred.TRANSACTION_HASH.is_unique, "duplicate hashes"
    assert set(pred.TRANSACTION_HASH) == set(tx.TRANSACTION_HASH), "hash set differs from test transactions"
    assert pred.score.between(0, 1).all() and pred.score.notna().all(), "scores outside [0, 1]"
    assert pred.score.is_unique, "tied scores"
    assert len(pred) <= 100_000


def reasons(X, score, top_share=0.02):
    """Short evidence note for the highest-ranked payments (the reason column is not scored)."""
    out = np.full(len(X), "", dtype=object)
    top = np.flatnonzero(score >= np.quantile(score, 1 - top_share))
    for i in top:
        x = X.iloc[i]
        bits = []
        if x.novel == 1:
            bits.append("first-time sender-receiver pair")
        if x.nc20m_size >= 3:
            bits.append(f"one of {int(x.nc20m_size)} first-time-pair payments linked within 20 min")
        if x.r_in_30m >= 2:
            bits.append(f"receiver got {int(x.r_in_30m) + 1} payments within 30 min")
        if x.s_out_30m >= 2:
            bits.append(f"sender made {int(x.s_out_30m) + 1} payments within 30 min")
        if x.chain_len >= 2:
            bits.append(f"{int(x.chain_len)}-hop chain of similar amounts")
        if x.cyc2 or x.cyc3:
            bits.append("funds return to the sender within 6 h")
        out[i] = "flagged for review: " + ("; ".join(bits) if bits else "pattern score")
    return out
