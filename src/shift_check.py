"""Adversarial check: which features separate train/ from validation/ (labels unused)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).parent))
from dd.features import build_features
from dd.load import load_split
from dd.model import fit

D = Path("decode-and-detect-training-data")
Xa = pd.read_parquet("cache/train_X.parquet") if Path("cache/train_X.parquet").exists() else build_features(load_split(D / "train"))
Xb = build_features(load_split(D / "validation")); Xb.to_parquet("cache/validation_X.parquet")
X = pd.concat([Xa, Xb], ignore_index=True); z = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
oof = np.zeros(len(z))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z):
    oof[te] = fit(X.iloc[tr], z[tr], dict(n_estimators=200)).predict_proba(X.iloc[te])[:, 1]
print(f"train-vs-validation AUC {roc_auc_score(z, oof):.3f}")
imp = pd.Series(fit(X, z, dict(n_estimators=200)).booster_.feature_importance("gain"), index=X.columns)
print((imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_string())
for c in (imp / imp.sum()).sort_values(ascending=False).head(6).index:
    print(c, "train med", round(Xa[c].median(), 3), "val med", round(Xb[c].median(), 3), "per-feature AUC", round(roc_auc_score(z, X[c].fillna(-9)), 3))
uni = pd.Series({c: roc_auc_score(z, X[c].fillna(-9)) for c in X.columns})
bad = uni[(uni - 0.5).abs() > 0.1].sort_values()
print(f"features with univariate shift AUC outside [0.4, 0.6]: {len(bad)}")
print(bad.round(3).to_string())
