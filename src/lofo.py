"""Leave-one-family-out stress test: fit without a whole planted family, score that family."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from dd.model import fit

lab = pd.read_parquet("cache/train_labels.parquet"); X = pd.read_parquet("cache/train_X.parquet")
y = lab.is_suspicious.to_numpy(); bg = (lab.family == "background").to_numpy()
rng = np.random.default_rng(0); bg_hold = bg & (rng.random(len(y)) < 0.2)
rows = []
for fam in sorted(set(lab.family) - {"background", "ambiguous", "aggregate_relay", "distributed_relay"}):
    held = (lab.family == fam).to_numpy()
    m = fit(X[~held & ~bg_hold], y[~held & ~bg_hold])
    p = m.predict_proba(X[held | bg_hold])[:, 1]
    yy = y[held | bg_hold]; inf = held[held | bg_hold]
    rows.append(dict(family=fam, within_family_AP=100 * average_precision_score(yy[inf], p[inf]),
                     AP_vs_background=100 * average_precision_score(yy, p),
                     AUC_pos_vs_bg=roc_auc_score(yy[(yy == 1) | ~inf], p[(yy == 1) | ~inf])))
print(pd.DataFrame(rows).round(2).to_string(index=False))
