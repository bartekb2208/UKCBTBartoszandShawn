"""Wallet-disjoint check: train on payments inside one random half of the wallets,
score payments inside the other half. Blocks wallet memorisation, which grouped
folds cannot (train has one connected component)."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from dd.model import fit


def wallet_disjoint_ap(X, df, y, repeats=6, seed=0, params=None):
    wallets = np.unique(np.r_[df.src, df.dst]); rng = np.random.default_rng(seed); out = []
    for _ in range(repeats):
        half = set(rng.choice(wallets, len(wallets) // 2, replace=False))
        a = df.src.isin(half).to_numpy() & df.dst.isin(half).to_numpy()
        b = ~df.src.isin(half).to_numpy() & ~df.dst.isin(half).to_numpy()
        for tr, te in ((a, b), (b, a)):
            m = fit(X[tr], y[tr], params)
            out.append(100 * average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.mean(out), np.std(out) / np.sqrt(len(out))


if __name__ == "__main__":
    df = pd.read_parquet("cache/train_tx.parquet"); lab = pd.read_parquet("cache/train_labels.parquet")
    X = pd.read_parquet("cache/train_X.parquet"); y = lab.is_suspicious.to_numpy()
    sets = {"all": list(X.columns),
            "no weekly wallet": [c for c in X.columns if not re.match(r"[sr]_wk_|pair_rev_n|v_vs_[sr]_med", c)],
            "no weekly, no burst": [c for c in X.columns if not re.match(r"[sr]_wk_|pair_rev_n|v_vs_[sr]_med|.*burst|.*_frac_", c)]}
    for name, cols in sets.items():
        m, se = wallet_disjoint_ap(X[cols], df, y)
        print(f"{name:22s} {len(cols):3d} features  wallet-disjoint AP {m:.2f} ± {se:.2f}")
