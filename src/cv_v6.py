"""Out-of-fold AP of the v6 variants, from cached train features (run exp.py --rebuild and motif caching first).

Variants:  A_motif = model A features + relay-motif features
           U_motif = model A + Shawn's behavioural features (asset features dropped) + relay-motif features
Usage: python src/cv_v6.py A_motif|U_motif
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from dd.evaluate import print_report, report  # noqa: E402
from dd.features import ASSET_FEATURES  # noqa: E402
from predict import score  # noqa: E402

SH_ASSET = r"sh_(log_usd_paid|spread|out_.*|in_.*|is_conv|n_legs|n_assets|is_path|has_sendmax|has_paths|c\d+_n_assets)"
variant = sys.argv[1]
df = pd.read_parquet("cache/train_tx.parquet")
lab = pd.read_parquet("cache/train_labels.parquet")
X = pd.read_parquet("cache/train_X.parquet")
X = X.drop(columns=[c for c in X.columns if re.fullmatch(ASSET_FEATURES, c)])
parts = [X, pd.read_parquet("cache/train_motifs.parquet")]
if variant == "U_motif":
    S = pd.read_parquet("cache/shawnF_train.parquet").reindex(df.hash.values).reset_index(drop=True)
    parts.append(S.drop(columns=[c for c in S.columns if re.fullmatch(SH_ASSET, c)]))
U = pd.concat(parts, axis=1)
U = U[sorted(U.columns)]
print(variant, U.shape, flush=True)
f = lab.fold.to_numpy()
oof = np.zeros(len(U))
for k in range(5):
    tr, te = f != k, f == k
    oof[te] = score(U[tr].reset_index(drop=True), lab[tr].reset_index(drop=True), U[te])
print_report(report(lab, oof, variant + " recipe"))
np.save(f"cache/oof_{variant}.npy", oof)
