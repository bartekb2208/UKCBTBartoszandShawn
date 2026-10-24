"""Value conservation inside reconstructed motifs.

Planted shapes with intermediaries (chain, cycle, multi-stage, stack, fan-out-in, random) pass on value more
tightly than their legitimate look-alikes: in fast scenarios the median out/in ratio of an intermediary is
0.90-0.98 for planted shapes and 0.75-0.94 for look-alikes. For each motif cluster (payments linked by a shared
wallet within a short gap) we find the intermediaries (wallets that both receive and send inside the cluster)
and describe how tightly they conserve value; every payment in the cluster gets the cluster's figures, plus the
pass-through of its own sender and receiver when they are intermediaries. Labels are not used.
"""
import numpy as np
import pandas as pd

from .features import motif_clusters

MAX_SIZE = 60
SETS = (("cc5", 300, False), ("cc20", 1200, True))   # (name, gap seconds, rare pairs only)


def conservation_features(df):
    """df: dd.features.prepare(load_split(...)) output (time-sorted, one row per payment)."""
    n = len(df)
    pair_n = np.asarray(pd.MultiIndex.from_arrays([df.src, df.dst]).map(df.groupby(["src", "dst"]).size()))
    out = {}
    for name, gap, rare in SETS:
        idx = np.flatnonzero(pair_n <= 2) if rare else np.arange(n)
        sub = df.iloc[idx].reset_index(drop=True)
        cid = motif_clusters(sub, gap)
        g = pd.DataFrame({"cid": cid, "src": sub.src.to_numpy(), "dst": sub.dst.to_numpy(), "usd": sub.usd.to_numpy(),
                          "cost": sub.cost_usd.to_numpy()})
        size = g.groupby("cid").cid.transform("size").to_numpy()
        inn = g.groupby(["cid", "dst"]).usd.sum()
        outv = g.groupby(["cid", "src"]).cost.sum()                    # what the intermediary actually spent
        inn.index.names = ["cid", "w"]
        outv.index.names = ["cid", "w"]
        J = pd.concat([inn.rename("i"), outv.rename("o")], axis=1).dropna()
        J["lr"] = np.log(J.o / J.i)
        agg = J.groupby(level="cid").lr.agg(n_inter="size", dev=lambda x: float(np.median(np.abs(x))),
                                             med="median", lo="min", hi="max")
        cols = {f"{name}_n_inter": agg.n_inter, f"{name}_dev": agg.dev, f"{name}_med": agg.med,
                f"{name}_min": agg.lo, f"{name}_max": agg.hi}
        keep = size <= MAX_SIZE
        for c, series in cols.items():
            vals = pd.Series(cid).map(series).to_numpy(dtype=float)
            col = np.full(n, np.nan)
            col[idx] = np.where(keep, vals, np.nan)
            out[c] = col
        lr = J.lr
        for role, key in (("s", g.src), ("r", g.dst)):
            vals = np.asarray(pd.MultiIndex.from_arrays([cid, key.to_numpy()]).map(lr), dtype=float)
            col = np.full(n, np.nan)
            col[idx] = np.where(keep, vals, np.nan)
            out[f"{name}_{role}_pass"] = col
    return pd.DataFrame(out, index=df.index)
